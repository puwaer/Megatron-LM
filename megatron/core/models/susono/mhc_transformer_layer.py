# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MHCTransformerLayer: TransformerLayer wrapped with mHC multi-stream residuals.

This module extends the standard TransformerLayer so that it operates on a stack
of n parallel residual streams (mHC) rather than a single hidden-state tensor.

Hidden-state convention inside SusonoBlock:
    Standard Megatron: [S, B, D]
    mHC multi-stream:  [n, S, B, D]

At the layer boundary (MHC-Lite):
  1. width_connection: [n,S,B,D] → [S,B,D]       (input-gated aggregation + permutation mix)
  2. (optional Engram)
  3. Standard TransformerLayer forward: [S,B,D] → [S,B,D]
  4. depth_connection: [S,B,D] → [n,S,B,D]        (beta-gated distribution + mixed residuals)

When use_mhc=False the layer behaves identically to a standard TransformerLayer.
"""

from typing import Optional

import torch
from torch import Tensor

from megatron.core.models.engram.engram_module import EngramModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.mhc import ManifoldConstrainedHyperConnection
from megatron.core.transformer.mhc_checkpoint import MHCSelectiveCheckpoint
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)


class MHCTransformerLayer(TransformerLayer):
    """TransformerLayer extended with Manifold-Constrained Hyper-Connection (mHC) residuals
    and optional Engram conditional memory.

    When config.use_mhc is True the module expects hidden_states of shape [n, S, B, D]
    and returns a tensor of the same shape.  When False it behaves like a standard
    TransformerLayer and expects/returns [S, B, D].

    Args:
        config:         TransformerConfig (must have use_mhc / mhc_* fields).
        submodules:     TransformerLayerSubmodules spec.
        layer_number:   1-indexed layer number within the model.
        hidden_dropout: Optional dropout override.
        pg_collection:  Process-group collection for distributed training.
        vp_stage:       Virtual-pipeline stage index.
        is_mtp_layer:   Whether this is a Multi-Token Prediction inner layer.
        engram_module:  Optional pre-constructed EngramModule for this layer.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        engram_module: Optional[EngramModule] = None,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
        )

        # mHC module — created only when the feature is enabled
        self.mhc: Optional[ManifoldConstrainedHyperConnection] = None
        if getattr(config, 'use_mhc', False):
            self.mhc = ManifoldConstrainedHyperConnection(
                hidden_size=config.hidden_size,
                num_streams=config.mhc_num_streams,
                layer_index=layer_number,
                sinkhorn_iterations=config.mhc_sinkhorn_iterations,
                use_fused_kernel=getattr(config, 'mhc_use_fused_kernel', False),
            )

        # Engram module — None for layers where Engram is not applied
        self.engram: Optional[EngramModule] = engram_module

        # ------------------------------------------------------------------
        # Phase 4 — PP async overlap: dedicated CUDA stream + event pairs
        #
        # _mhc_stream (priority=-1, high): executes width_connection and
        #     depth_connection so that their SM kernels can overlap with
        #     concurrent NCCL DMA activity on the default compute stream.
        #
        # Overlap model (PP ≥ 2):
        #   after forward() returns, the PP schedule immediately calls
        #   send_forward (NCCL P2P → DMA engine).  depth_connection was
        #   already dispatched to _mhc_stream; the event barrier
        #   (current_stream.wait_event(_mhc_event_d)) guarantees that
        #   hidden_states is fully computed before send_forward reads it,
        #   but the NCCL DMA and the NEXT layer's width_connection can
        #   start concurrently on independent hardware engines.
        # ------------------------------------------------------------------
        self._mhc_stream: Optional[torch.cuda.Stream] = None
        self._mhc_event_w: Optional[torch.cuda.Event] = None
        self._mhc_event_d: Optional[torch.cuda.Event] = None

        if (
            getattr(config, 'use_mhc', False)
            and getattr(config, 'mhc_async_pp_overlap', False)
            and torch.cuda.is_available()
        ):
            # priority=-1 → high priority; typical NCCL/compute streams use 0.
            # High priority allows the CUDA scheduler to issue mHC SM kernels
            # as soon as the dependency event is satisfied, even while lower-
            # priority streams have pending work.
            self._mhc_stream = torch.cuda.Stream(priority=-1)
            # enable_timing=False keeps event overhead < 1 μs
            self._mhc_event_w = torch.cuda.Event(enable_timing=False)
            self._mhc_event_d = torch.cuda.Event(enable_timing=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        hidden_states: Tensor,
        **kwargs,
    ):
        """Forward pass with mHC multi-stream management and optional Engram memory.

        input_ids (if needed by Engram) is read from the ``_current_input_ids``
        attribute that SusonoBlock sets before calling this layer's forward.

        Args:
            hidden_states: Shape [n, S, B, D] when use_mhc=True, else [S, B, D].
            **kwargs:      All remaining kwargs are forwarded to the parent
                           TransformerLayer (attention_mask, rotary_pos_emb, etc.).

        Returns:
            Tuple (hidden_states, context) where hidden_states has the same leading
            shape as the input ([n,S,B,D] or [S,B,D]).
        """
        # Retrieve input_ids injected by SusonoBlock (None if not set)
        input_ids: Optional[Tensor] = getattr(self, '_current_input_ids', None)

        if self.mhc is not None:
            # --- MHC-Lite path ---
            use_selective = (
                getattr(self.config, 'mhc_selective_recompute', False) and self.training
            )

            if self._mhc_stream is not None:
                # -------------------------------------------------------
                # Phase 4: async PP-overlap path
                # Stream topology:
                #   _mhc_stream  (high priority): width_conn, depth_conn
                #   current_stream (normal prio): Engram, TransformerLayer,
                #                                 PP NCCL send/recv
                #
                # Barrier sequence:
                #   _mhc_stream.wait_stream(cur)  — X must be visible
                #   width_conn on _mhc_stream
                #   _mhc_event_w.record(_mhc_stream)
                #   cur.wait_event(_mhc_event_w)  — x_in must be visible
                #   Engram + TransformerLayer on cur
                #   _mhc_stream.wait_stream(cur)  — x_out must be visible
                #   depth_conn on _mhc_stream
                #   _mhc_event_d.record(_mhc_stream)
                #   cur.wait_event(_mhc_event_d)  — hidden_states ready
                #
                # After return, the PP schedule immediately issues send_forward
                # (NCCL DMA on DMA engine).  The NEXT layer's width_conn starts
                # on _mhc_stream concurrently on the SM — true HW parallelism.
                # -------------------------------------------------------
                cur = torch.cuda.current_stream()

                # ---- Width connection on _mhc_stream ----
                self._mhc_stream.wait_stream(cur)        # X [n,S,B,D] is ready
                with torch.cuda.stream(self._mhc_stream):
                    if use_selective:
                        # Discard new_residuals/beta from autograd tape;
                        # MHCSelectiveCheckpoint recomputes them in backward.
                        x_in, _, _ = self.mhc._width_connection(hidden_states)
                    else:
                        x_in, add_residual = self.mhc(hidden_states)
                self._mhc_event_w.record(self._mhc_stream)
                cur.wait_event(self._mhc_event_w)        # x_in [S,B,D] is ready

                # ---- Optional Engram + standard TransformerLayer (on cur) ----
                if self.engram is not None and input_ids is not None:
                    x_in = x_in + self.engram(input_ids, x_in)   # [S, B, D]

                x_out, context = super().forward(x_in, **kwargs)  # [S, B, D]

                # ---- Depth connection on _mhc_stream ----
                self._mhc_stream.wait_stream(cur)        # x_out [S,B,D] is ready
                with torch.cuda.stream(self._mhc_stream):
                    if use_selective:
                        hidden_states = MHCSelectiveCheckpoint.apply(
                            hidden_states, x_out, self.mhc
                        )                                          # [n, S, B, D]
                    else:
                        hidden_states = add_residual(x_out)        # [n, S, B, D]
                self._mhc_event_d.record(self._mhc_stream)
                cur.wait_event(self._mhc_event_d)        # hidden_states ready

            elif use_selective:
                # Selective recompute path: save only X + x_out; discard intermediates.
                # 1. Width connection with autograd: gets branch_input (x_in);
                #    new_residuals and beta are discarded, freeing H_res/res_coeff
                #    from the autograd tape (no downstream use of those outputs).
                x_in, _, _ = self.mhc._width_connection(hidden_states)  # [S,B,D]

                # 2. Optional Engram memory injection
                if self.engram is not None and input_ids is not None:
                    x_in = x_in + self.engram(input_ids, x_in)   # [S, B, D]

                # 3. Standard TransformerLayer computation
                x_out, context = super().forward(x_in, **kwargs)  # [S, B, D]

                # 4. Depth connection via custom Function (backward recomputes width)
                hidden_states = MHCSelectiveCheckpoint.apply(
                    hidden_states, x_out, self.mhc
                )                                                   # [n, S, B, D]
            else:
                # Standard closure path (retains all intermediates in autograd graph)
                # 1. Width connection: get branch input and depth-connection closure
                #    Captures mixed residuals and beta inside add_residual.
                x_in, add_residual = self.mhc(hidden_states)   # [S,B,D], closure

                # 2. Optional Engram memory injection
                if self.engram is not None and input_ids is not None:
                    x_in = x_in + self.engram(input_ids, x_in)   # [S, B, D]

                # 3. Standard TransformerLayer computation (internal attention + MLP residuals)
                x_out, context = super().forward(x_in, **kwargs)  # [S, B, D]

                # 4. Depth connection: distribute layer output back to n streams
                hidden_states = add_residual(x_out)             # [n, S, B, D]
        else:
            # --- Standard path (mHC disabled) ---
            if self.engram is not None and input_ids is not None:
                hidden_states = hidden_states + self.engram(input_ids, hidden_states)

            hidden_states, context = super().forward(hidden_states, **kwargs)

        return hidden_states, context
