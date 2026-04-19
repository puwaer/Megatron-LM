# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hybrid decoder layer for the Susono / Qwen3-Next architecture.

Provides SusonoLinearAttentionDecoderLayer — a complete decoder layer that uses
GatedDeltaNet linear attention instead of softmax self-attention.  It mirrors
the interface of MHCTransformerLayer so that SusonoBlock can store both types
in a single nn.ModuleList and call them uniformly.

Layer structure (Qwen3-Next linear attention layer):
    RMSNorm  →  GatedDeltaNet  →  residual
    RMSNorm  →  MoE / MLP      →  residual
    (MHC-Lite aggregate / distribute wraps the above when use_mhc=True)
    (Engram memory added to x_in before GatedDeltaNet when attached)

Megatron tensor convention: [S, B, D] without MHC-Lite, [n, S, B, D] with MHC-Lite.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.models.engram.engram_module import EngramModule
from megatron.core.transformer.module import MegatronModule
from megatron.core.models.susono.susono_gated_delta_net import (
    SusonoGatedDeltaNet,
    GatedDeltaNetInferenceCache,
)
from megatron.core.models.susono.susono_moe import SusonoDenseMLP, SusonoSparseMoE
from megatron.core.transformer.mhc import ManifoldConstrainedHyperConnection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


# ──────────────────────────────────────────────────────────────────────────────
# Norm
# ──────────────────────────────────────────────────────────────────────────────

from megatron.core.fusions.susono_fused_norm import rmsnorm_1p


class _RMSNorm(nn.Module):
    """RMSNorm with (1 + w) scaling, matching Qwen3-Next's SusonoRMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))  # (1+w) init → zero

    def forward(self, x: Tensor) -> Tensor:
        return rmsnorm_1p(x, self.weight, self.eps)


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid linear attention decoder layer
# ──────────────────────────────────────────────────────────────────────────────

class SusonoLinearAttentionDecoderLayer(MegatronModule):
    """Complete decoder layer using GatedDeltaNet linear attention.

    Drop-in companion to MHCTransformerLayer inside SusonoBlock.  The public
    interface is identical:

        output, context = layer(hidden_states, **kwargs)

    where hidden_states is [n,S,B,D] with MHC-Lite or [S,B,D] without.

    Args:
        config:       TransformerConfig.
        layer_number: 1-indexed layer position (Megatron convention).
        mlp_only:     If True, use dense MLP instead of MoE for feed-forward.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        mlp_only: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(config)
        # Store GLOBAL layer number to match TransformerLayer convention.
        # Required for correct distributed checkpoint key generation in PP training.
        vp_stage = kwargs.get('vp_stage', None)
        pp_offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        self.layer_number = layer_number + pp_offset  # GLOBAL 1-indexed (Megatron convention)
        self.layer_idx = layer_number - 1             # LOCAL 0-indexed (used for GatedDeltaNet cache)

        D = config.hidden_size
        eps = getattr(config, 'layernorm_epsilon', 1e-6)

        # Norms
        self.post_attention_layernorm = _RMSNorm(D, eps=eps)

        # Linear attention.  When GDN was able to build the fused
        # ``input_ln_proj`` (TELayerNormColumnParallelLinear) module it owns
        # the pre-attention RMSNorm internally and we skip the external call.
        self.linear_attn = SusonoGatedDeltaNet(config, layer_idx=self.layer_idx)

        if getattr(self.linear_attn, 'input_ln_proj', None) is not None:
            self.input_layernorm = None           # norm lives inside linear_attn
        else:
            self.input_layernorm = _RMSNorm(D, eps=eps)

        # Feed-forward (MoE or dense)
        if mlp_only:
            self.mlp = SusonoDenseMLP(config)
        else:
            self.mlp = SusonoSparseMoE(config, layer_number=self.layer_number)

        # MHC-Lite connection (created if enabled in config)
        self.mhc: Optional[ManifoldConstrainedHyperConnection] = None
        if getattr(config, 'use_mhc', False):
            self.mhc = ManifoldConstrainedHyperConnection(
                hidden_size=D,
                num_streams=config.mhc_num_streams,
                layer_index=self.layer_number,
                sinkhorn_iterations=config.mhc_sinkhorn_iterations,
                use_fused_kernel=getattr(config, 'mhc_use_fused_kernel', False),
                auto_use_fused_kernel=getattr(config, 'mhc_auto_use_fused_kernel', True),
            )

        # Engram module — attached externally by SusonoBlock
        self.engram: Optional[EngramModule] = None

        # Placeholder for input_ids injected by SusonoBlock before forward
        self._current_input_ids: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Checkpoint migration: old format → B-6 fused format
    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate pre-B-6 checkpoints on-the-fly.

        Old layout (pre-B-6):
            {prefix}input_layernorm.weight                     [D]
            {prefix}linear_attn.in_proj_qkvz.weight            [proj_qkvz, D]
            {prefix}linear_attn.in_proj_ba.weight              [proj_ba, D]

        New layout (B-6 fused):
            {prefix}linear_attn.input_ln_proj.layer_norm_weight  [D]
            {prefix}linear_attn.input_ln_proj.weight             [proj_qkvz + proj_ba, D]

        Only runs the concat+rename when the old keys are present and the
        current module actually owns the fused path.  If the state_dict is
        already in the new format, this is a no-op.
        """
        old_ln = prefix + 'input_layernorm.weight'
        old_qkvz = prefix + 'linear_attn.in_proj_qkvz.weight'
        old_ba = prefix + 'linear_attn.in_proj_ba.weight'
        new_ln = prefix + 'linear_attn.input_ln_proj.layer_norm_weight'
        new_w = prefix + 'linear_attn.input_ln_proj.weight'

        has_fused = (
            self.input_layernorm is None
            and getattr(self.linear_attn, 'input_ln_proj', None) is not None
        )

        if has_fused:
            # Migrate layernorm weight
            if old_ln in state_dict and new_ln not in state_dict:
                state_dict[new_ln] = state_dict.pop(old_ln)
            # Migrate concatenated projection weight
            if (
                old_qkvz in state_dict
                and old_ba in state_dict
                and new_w not in state_dict
            ):
                state_dict[new_w] = torch.cat(
                    [state_dict.pop(old_qkvz), state_dict.pop(old_ba)],
                    dim=0,
                )

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    # ------------------------------------------------------------------
    # Core layer computation (single stream, [S, B, D])
    # ------------------------------------------------------------------

    def _layer_forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor],
        inference_cache: Optional[GatedDeltaNetInferenceCache],
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Single-stream forward: attention + MLP + residuals.

        Args:
            x:               [S, B, D]
            attention_mask:  Padding mask [B, S] (optional).
            inference_cache: GatedDeltaNet inference cache (optional).

        Returns:
            ([S, B, D], router_logits or None)
        """
        # ── Linear attention ──────────────────────────────────────────
        residual = x
        if self.input_layernorm is not None:
            # Legacy path: external RMSNorm feeds GDN's separate projections.
            x = self.input_layernorm(x)
        # Fused path: GDN's ``input_ln_proj`` (TELayerNormColumnParallelLinear)
        # applies the RMSNorm internally before the QKVZ+BA projection.
        x = self.linear_attn(
            x,
            inference_cache=inference_cache,
            attention_mask=attention_mask,
        )
        x = residual + x

        # ── Feed-forward ──────────────────────────────────────────────
        residual = x
        x = self.post_attention_layernorm(x)
        x, router_logits = self.mlp(x)
        x = residual + x

        return x, router_logits

    # ------------------------------------------------------------------
    # Public forward (handles MHC-Lite and Engram)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_cache: Optional[GatedDeltaNetInferenceCache] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass.

        When MHC-Lite is enabled, hidden_states is [n, S, B, D] and MHC-Lite aggregate /
        distribute is handled here (mirroring MHCTransformerLayer).

        input_ids for Engram is read from self._current_input_ids (set by SusonoBlock).

        Args:
            hidden_states:   [n, S, B, D] with MHC-Lite, [S, B, D] without.
            attention_mask:  Padding mask for GatedDeltaNet.
            inference_cache: Per-layer GatedDeltaNet cache.
            **kwargs:        Ignored (for interface compatibility with TransformerLayer).

        Returns:
            (hidden_states, router_logits) — router_logits is a Tensor for MoE layers,
            None for dense MLP layers.
        """
        input_ids: Optional[Tensor] = getattr(self, '_current_input_ids', None)

        if self.mhc is not None:
            # ── MHC-Lite path: [n,S,B,D] ────────────────────────────
            x_in = self.mhc.aggregate_streams(hidden_states)   # [S, B, D]

            # Engram memory injection
            if self.engram is not None and input_ids is not None:
                # Engram in Megatron uses [S,B,D] convention
                x_in = x_in + self.engram(input_ids, x_in)

            x_out, _router_logits = self._layer_forward(x_in, attention_mask, inference_cache, **kwargs)
            hidden_states = self.mhc.distribute_output(hidden_states, x_out)  # [n,S,B,D]

        else:
            # ── Standard path: [S,B,D] ───────────────────────────────
            if self.engram is not None and input_ids is not None:
                hidden_states = hidden_states + self.engram(input_ids, hidden_states)

            hidden_states, _router_logits = self._layer_forward(hidden_states, attention_mask, inference_cache, **kwargs)

        # Return None as context (matching TransformerLayer interface).
        # MoE auxiliary loss is handled via save_to_aux_losses_tracker() inside SusonoTopKRouter.
        return hidden_states, None
