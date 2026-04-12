# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FujiBlock: TransformerBlock extended for mHC + Engram (Fuji architecture).

FujiBlock extends TransformerBlock with three additions:

1. **mHC stream expansion/collapse** (when config.use_mhc=True)
   - Expansion: [S, B, D] → [n, S, B, D] at the start of forward
   - Collapse: [n, S, B, D] → [S, B, D] at the end of forward via learned stream_proj
   - The final LayerNorm in the parent operates per-stream (dim=-1 = D), which is correct

2. **Engram module attachment** at specific layer positions
   - EngramModule instances are created and attached to MHCTransformerLayer objects
   - at the layer IDs specified in engram_config.engram_layer_ids

3. **input_ids propagation** for Engram
   - input_ids is temporarily stored on each MHCTransformerLayer before the parent forward
   - Cleared unconditionally in a finally block
   - Thread-safe for standard sequential Megatron training

Input/output contract:
  - forward() accepts and returns [S, B, D] when mHC is enabled, same as TransformerBlock
  - This keeps FujiBlock as a drop-in replacement for TransformerBlock in FujiModel

Pipeline parallel notes:
  - Per-stage mHC: streams are expanded at the start of each stage's forward and collapsed
    at the end. This simplifies inter-stage communication (no n× overhead) at the cost
    of re-initializing streams at every stage boundary.
  - Future work can pass [n, S, B, D] between stages for full cross-stage stream continuity.
"""

import math
from typing import Optional, Union, List

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.models.engram.engram_module import EngramConfig, EngramModule
from megatron.core.models.fuji.mhc_transformer_layer import MHCTransformerLayer
from megatron.core.models.fuji.fuji_hybrid_decoder_layer import FujiLinearAttentionDecoderLayer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.transformer_block import (
    LayerNormImpl,
    TransformerBlock,
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec


class FujiBlock(TransformerBlock):
    """TransformerBlock subclass for the Fuji architecture (mHC + Engram).

    The block's forward pass keeps the [S, B, D] interface identical to
    TransformerBlock; stream expansion/collapse is managed internally.

    Args:
        config:          TransformerConfig (must contain use_mhc / mhc_* / use_engram fields).
        spec:            Layer spec — should reference MHCTransformerLayer.
        engram_config:   Optional EngramConfig.  When provided and config.use_engram is True,
                         EngramModules are attached to layers listed in
                         engram_config.engram_layer_ids.
        post_layer_norm: Forwarded to TransformerBlock.
        pre_process:     Forwarded to TransformerBlock.
        post_process:    Forwarded to TransformerBlock.
        pg_collection:   Forwarded to TransformerBlock.
        vp_stage:        Forwarded to TransformerBlock.
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        engram_config: Optional[EngramConfig] = None,
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        # ----------------------------------------------------------------------
        # Hybrid layer logic (GatedDeltaNet + Full Attention)
        # ----------------------------------------------------------------------
        if getattr(config, 'experimental_attention_variant', None) == 'gated_delta_net':
            # Calculate layer offset and count for this stage
            offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
            num_layers = get_num_layers_to_build(config, vp_stage=vp_stage)

            # Determine hybrid pattern (default: one full attention every 4 layers)
            full_interval = getattr(config, 'full_attention_interval', 4)

            # Preserve the original block-level layer_norm so that the parent
            # TransformerBlock still constructs ``final_layernorm`` on the last
            # PP stage.  Without this, reconstructing ``TransformerBlockSubmodules``
            # below would drop ``layer_norm`` and the trained checkpoint would be
            # missing ``decoder.final_layernorm.weight``.
            if isinstance(spec, TransformerBlockSubmodules):
                original_layer_norm = spec.layer_norm
            else:
                original_layer_norm = LayerNormImpl

            # Convert input spec to a list of specs (one per layer)
            if isinstance(spec, TransformerBlockSubmodules):
                if spec.layer_specs is None:
                    # If generic spec, replicate for all layers (all Full Attn initially)
                    # We create a dummy list that we will overwrite
                    # This case assumes spec.layer_specs is NOT used yet.
                    # Actually spec contains submodules definition.
                    # We need to construct a list of ModuleSpecs.
                    # But wait, how do we get the ModuleSpec for the "standard" layer?
                    # TransformerBlockSubmodules doesn't contain the ModuleSpec of the layer class itself,
                    # just the submodules of that layer.
                    # Using TransformerBlockSubmodules directly implicitly uses the default layer class
                    # (which is MHCTransformerLayer in our case).
                    # We need to create specific ModuleSpecs.
                    base_full_attn_spec = ModuleSpec(module=MHCTransformerLayer, submodules=spec)
                    layer_specs = [base_full_attn_spec for _ in range(num_layers)]
                else:
                    layer_specs = list(spec.layer_specs)
            else:
                # specific ModuleSpec provided
                layer_specs = [spec for _ in range(num_layers)]

            # Override linear attention layers
            for i in range(num_layers):
                layer_idx = i + offset
                is_linear = ((layer_idx + 1) % full_interval) != 0

                if is_linear:
                    # Determine MoE usage for this layer
                    # Standard logic: MoE if (layer_idx + 1) % frequency == 0
                    use_moe = False
                    if getattr(config, 'num_moe_experts', 0) > 0:
                        freq = getattr(config, 'moe_layer_freq', 1)
                        if (layer_idx + 1) % freq == 0:
                            use_moe = True
                        
                        # Check explicitly excluded layers (mlp_only_layers)
                        mlp_only_layers = getattr(config, 'mlp_only_layers', [])
                        if layer_idx in mlp_only_layers:
                            use_moe = False

                    # Create linear attention spec
                    # FujiLinearAttentionDecoderLayer(config, layer_number, mlp_only=bool)
                    mlp_only = not use_moe
                    layer_specs[i] = ModuleSpec(
                        module=FujiLinearAttentionDecoderLayer,
                        params={"mlp_only": mlp_only},
                    )

            # Update spec to use the new per-layer specs.  Carry layer_norm
            # over so the parent block keeps the final layernorm builder.
            spec = TransformerBlockSubmodules(
                layer_specs=layer_specs,
                layer_norm=original_layer_norm,
            )

        # Auto-compute optimal recompute_num_layers (L_r*) when requested.
        # L_r* = round(sqrt(n * L / (n + 2))) minimises peak transient memory
        # during the backward pass under uniform recomputation.
        if getattr(config, 'mhc_auto_recompute_num_layers', False) and getattr(config, 'use_mhc', False):
            n = config.mhc_num_streams
            L = config.num_layers
            config.recompute_num_layers = max(1, round(math.sqrt(n * L / (n + 2))))

        super().__init__(
            config=config,
            spec=spec,
            post_layer_norm=post_layer_norm,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        # mHC stream-collapse projection: [n*D] → [D]
        # Initialised to equal-weight average so that at start of training the
        # collapsed output equals the embedding (all n streams start identical).
        # Sharded across TP on the D output dim; gather_output=True so the
        # collapsed hidden state is fully replicated for the next layer.
        self.stream_proj: Optional[ColumnParallelLinear] = None
        if getattr(config, 'use_mhc', False):
            n = config.mhc_num_streams
            D = config.hidden_size
            self.stream_proj = ColumnParallelLinear(
                n * D,
                D,
                config=config,
                init_method=config.init_method,
                bias=False,
                gather_output=True,
                skip_bias_add=True,
            )
            with torch.no_grad():
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                tp_rank = parallel_state.get_tensor_model_parallel_rank()
                assert D % tp_size == 0, (
                    f"hidden_size={D} must be divisible by TP={tp_size} for stream_proj"
                )
                D_local = D // tp_size
                row_start = tp_rank * D_local
                row_end = row_start + D_local
                w_full = torch.zeros(D, n * D)
                for s in range(n):
                    w_full[:, s * D:(s + 1) * D] = torch.eye(D) / n
                w_local = w_full[row_start:row_end]
                self.stream_proj.weight.copy_(
                    w_local.to(self.stream_proj.weight.device)
                )

        # Attach Engram modules to the appropriate layers
        if engram_config is not None and getattr(config, 'use_engram', False):
            self._attach_engram_modules(config, engram_config)

    # ------------------------------------------------------------------
    # Engram attachment helpers
    # ------------------------------------------------------------------

    def _attach_engram_modules(
        self, config: TransformerConfig, engram_config: EngramConfig
    ) -> None:
        """Instantiate and attach EngramModule to each eligible layer."""
        target_layer_ids = set(engram_config.engram_layer_ids)
        for layer in self.layers:
            if not isinstance(layer, (MHCTransformerLayer, FujiLinearAttentionDecoderLayer)):
                continue
            # layer.layer_number is 1-indexed; convert to 0-indexed global ID
            layer_idx = layer.layer_number - 1
            if layer_idx in target_layer_ids:
                layer.engram = EngramModule(
                    config=engram_config,
                    layer_id=layer_idx,
                    hidden_size=config.hidden_size,
                )

    # ------------------------------------------------------------------
    # Distributed checkpointing
    # ------------------------------------------------------------------

    def sharded_state_dict(self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None):
        """Override to give stream_proj a PP-rank-qualified key.

        stream_proj exists independently on every PP stage (each stage expands
        and collapses mHC streams on its own).  Without this override, the
        parent's sharded_state_dict emits the same key
        ``decoder.stream_proj.weight`` on every stage with
        ``replica_id=(0, 0, 0)``, which fails distributed-checkpoint
        sharding validation.  We rename it to
        ``decoder.stream_proj_ppN.weight`` so each stage has a unique key.
        """
        sharded_sd = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        if self.stream_proj is not None:
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            old_dict_key = f'{prefix}stream_proj.weight'
            new_dict_key = f'{prefix}stream_proj_pp{pp_rank}.weight'
            if old_dict_key in sharded_sd:
                st = sharded_sd.pop(old_dict_key)
                # ShardedTensor.key must also be updated to match the dict key.
                st.key = new_dict_key
                sharded_sd[new_dict_key] = st

        return sharded_sd

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        hidden_states: Tensor,
        attention_mask,
        input_ids: Optional[Tensor] = None,
        **kwargs,
    ):
        """Forward pass.

        Args:
            hidden_states: Shape [S, B, D] (same as standard TransformerBlock).
            attention_mask: Attention mask.
            input_ids:     Token IDs [B, S] for Engram lookup.  May be None if
                           no Engram module is active in this block.
            **kwargs:      Forwarded to TransformerBlock.forward.

        Returns:
            Output hidden states [S, B, D] (mHC streams managed internally).
        """
        use_mhc = getattr(self.config, 'use_mhc', False)

        # --- mHC: expand [S, B, D] → [n, S, B, D] ---
        if use_mhc:
            n = self.config.mhc_num_streams
            if self.pre_process:
                # Replicate hidden_states across n streams: [S,B,D] → [n,S,B,D]
                hidden_states = hidden_states.unsqueeze(0).expand(n, -1, -1, -1).contiguous()
            else:
                # In non-pre_process stages TransformerBlock.forward() uses
                # self.input_tensor; expand that instead.
                if self.input_tensor is not None:
                    self.input_tensor = (
                        self.input_tensor.unsqueeze(0).expand(n, -1, -1, -1).contiguous()
                    )

        # --- Inject input_ids for Engram ---
        if input_ids is not None:
            for layer in self.layers:
                if isinstance(layer, (MHCTransformerLayer, FujiLinearAttentionDecoderLayer)):
                    layer._current_input_ids = input_ids

        try:
            result = super().forward(hidden_states, attention_mask, **kwargs)
        finally:
            # Clear even on exception
            for layer in self.layers:
                if isinstance(layer, (MHCTransformerLayer, FujiLinearAttentionDecoderLayer)):
                    layer._current_input_ids = None

        # --- mHC: collapse [n, S, B, D] → [S, B, D] ---
        if use_mhc and self.stream_proj is not None:
            # result may be (hidden_states,) or (hidden_states, intermediate_states)
            if isinstance(result, tuple):
                h, extra = result[0], result[1:]
            else:
                h, extra = result, None

            n_actual = h.shape[0]
            S, B, D = h.shape[1], h.shape[2], h.shape[3]
            # Permute to [S, B, n, D] then flatten to [S, B, n*D]
            h = h.permute(1, 2, 0, 3).contiguous().reshape(S, B, n_actual * D)
            h, _bias = self.stream_proj(h)  # [S, B, D]

            if extra is not None:
                return (h,) + extra
            return h

        return result
