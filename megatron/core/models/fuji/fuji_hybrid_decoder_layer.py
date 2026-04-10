# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hybrid decoder layer for the Fuji / Qwen3-Next architecture.

Provides FujiLinearAttentionDecoderLayer — a complete decoder layer that uses
GatedDeltaNet linear attention instead of softmax self-attention.  It mirrors
the interface of MHCTransformerLayer so that FujiBlock can store both types
in a single nn.ModuleList and call them uniformly.

Layer structure (Qwen3-Next linear attention layer):
    RMSNorm  →  GatedDeltaNet  →  residual
    RMSNorm  →  MoE / MLP      →  residual
    (mHC aggregate / distribute wraps the above when use_mhc=True)
    (Engram memory added to x_in before GatedDeltaNet when attached)

Megatron tensor convention: [S, B, D] without mHC, [n, S, B, D] with mHC.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.models.engram.engram_module import EngramModule
from megatron.core.transformer.module import MegatronModule
from megatron.core.models.fuji.fuji_gated_delta_net import (
    FujiGatedDeltaNet,
    GatedDeltaNetInferenceCache,
)
from megatron.core.models.fuji.fuji_moe import FujiDenseMLP, FujiSparseMoE
from megatron.core.transformer.mhc import ManifoldConstrainedHyperConnection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


# ──────────────────────────────────────────────────────────────────────────────
# Norm
# ──────────────────────────────────────────────────────────────────────────────

class _RMSNorm(nn.Module):
    """RMSNorm with (1 + w) scaling, matching Qwen3-Next's FujiRMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))  # (1+w) init → zero

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.float()
        out = x_f * torch.rsqrt((x_f * x_f).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid linear attention decoder layer
# ──────────────────────────────────────────────────────────────────────────────

class FujiLinearAttentionDecoderLayer(MegatronModule):
    """Complete decoder layer using GatedDeltaNet linear attention.

    Drop-in companion to MHCTransformerLayer inside FujiBlock.  The public
    interface is identical:

        output, context = layer(hidden_states, **kwargs)

    where hidden_states is [n,S,B,D] with mHC or [S,B,D] without.

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
        self.input_layernorm          = _RMSNorm(D, eps=eps)
        self.post_attention_layernorm = _RMSNorm(D, eps=eps)

        # Linear attention
        self.linear_attn = FujiGatedDeltaNet(config, layer_idx=self.layer_idx)

        # Feed-forward (MoE or dense)
        if mlp_only:
            self.mlp = FujiDenseMLP(config)
        else:
            self.mlp = FujiSparseMoE(config)

        # mHC connection (created if enabled in config)
        self.mhc: Optional[ManifoldConstrainedHyperConnection] = None
        if getattr(config, 'use_mhc', False):
            self.mhc = ManifoldConstrainedHyperConnection(
                hidden_size=D,
                num_streams=config.mhc_num_streams,
                layer_index=self.layer_number,
                sinkhorn_iterations=config.mhc_sinkhorn_iterations,
                use_fused_kernel=getattr(config, 'mhc_use_fused_kernel', False),
            )

        # Engram module — attached externally by FujiBlock
        self.engram: Optional[EngramModule] = None

        # Placeholder for input_ids injected by FujiBlock before forward
        self._current_input_ids: Optional[Tensor] = None

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
        x = self.input_layernorm(x)
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
    # Public forward (handles mHC and Engram)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_cache: Optional[GatedDeltaNetInferenceCache] = None,
        **kwargs,
    ) -> Tuple[Tensor, None]:
        """Forward pass.

        When mHC is enabled, hidden_states is [n, S, B, D] and mHC aggregate /
        distribute is handled here (mirroring MHCTransformerLayer).

        input_ids for Engram is read from self._current_input_ids (set by FujiBlock).

        Args:
            hidden_states:   [n, S, B, D] with mHC, [S, B, D] without.
            attention_mask:  Padding mask for GatedDeltaNet.
            inference_cache: Per-layer GatedDeltaNet cache.
            **kwargs:        Ignored (for interface compatibility with TransformerLayer).

        Returns:
            (hidden_states, None)  — context is None for linear attention.
        """
        input_ids: Optional[Tensor] = getattr(self, '_current_input_ids', None)

        if self.mhc is not None:
            # ── mHC path: [n,S,B,D] ─────────────────────────────────
            x_in = self.mhc.aggregate_streams(hidden_states)   # [S, B, D]

            # Engram memory injection
            if self.engram is not None and input_ids is not None:
                # Engram in Megatron uses [S,B,D] convention
                x_in = x_in + self.engram(input_ids, x_in)

            x_out, router_logits = self._layer_forward(x_in, attention_mask, inference_cache, **kwargs)
            hidden_states = self.mhc.distribute_output(hidden_states, x_out)  # [n,S,B,D]

        else:
            # ── Standard path: [S,B,D] ───────────────────────────────
            if self.engram is not None and input_ids is not None:
                hidden_states = hidden_states + self.engram(input_ids, hidden_states)

            hidden_states, router_logits = self._layer_forward(hidden_states, attention_mask, inference_cache, **kwargs)

        # Second return value carries router_logits (for MoE aux loss) instead of context
        return hidden_states, router_logits
