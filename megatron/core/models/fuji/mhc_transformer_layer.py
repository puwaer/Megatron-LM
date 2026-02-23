# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MHCTransformerLayer: TransformerLayer wrapped with mHC multi-stream residuals.

This module extends the standard TransformerLayer so that it operates on a stack
of n parallel residual streams (mHC) rather than a single hidden-state tensor.

Hidden-state convention inside FujiBlock:
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
            )

        # Engram module — None for layers where Engram is not applied
        self.engram: Optional[EngramModule] = engram_module

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
        attribute that FujiBlock sets before calling this layer's forward.

        Args:
            hidden_states: Shape [n, S, B, D] when use_mhc=True, else [S, B, D].
            **kwargs:      All remaining kwargs are forwarded to the parent
                           TransformerLayer (attention_mask, rotary_pos_emb, etc.).

        Returns:
            Tuple (hidden_states, context) where hidden_states has the same leading
            shape as the input ([n,S,B,D] or [S,B,D]).
        """
        # Retrieve input_ids injected by FujiBlock (None if not set)
        input_ids: Optional[Tensor] = getattr(self, '_current_input_ids', None)

        if self.mhc is not None:
            # --- MHC-Lite path ---
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
