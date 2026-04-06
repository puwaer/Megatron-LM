# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GatedDeltaNet linear attention for the Fuji / Qwen3-Next architecture.

Megatron-LM uses sequence-first tensors [S, B, D].  This module transposes to
batch-first [B, S, D] internally for the core computation, then converts back.

The GatedDeltaNet algorithm (Qwen3-Next linear attention):
  1. Project input → Q, K, V, Z (gate), A (decay), B (beta)
  2. Depthwise causal conv1d on QKV (context fusion)
  3. Chunked delta-rule recurrence (training) or token-by-token (inference)
  4. RMS-norm gated output
  5. Project back to hidden_size

Optional FLA (flash-linear-attention) fast path:
  If `causal_conv1d` and `fla.ops.gated_delta_rule` are available the kernel
  implementations are used; otherwise the pure-PyTorch fallback is used.

References:
  Qwen3-Next modeling_qwen3_next.py  (Qwen3NextGatedDeltaNet)
  arXiv: Gated Delta Networks (Yang et al.)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )
except ImportError:
    chunk_gated_delta_rule = None
    fused_recurrent_gated_delta_rule = None

_FAST_PATH = all(
    [causal_conv1d_fn, causal_conv1d_update, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule]
)


# ──────────────────────────────────────────────────────────────────────────────
# Inference cache
# ──────────────────────────────────────────────────────────────────────────────

class GatedDeltaNetInferenceCache:
    """Per-layer inference state for GatedDeltaNet.

    Attributes:
        conv_state:      [B, conv_dim, kernel_size]  — convolution state.
        recurrent_state: [B, num_v_heads, k_head_dim, v_head_dim]  — delta-rule state.
    """

    def __init__(self):
        self.conv_state: Optional[Tensor] = None
        self.recurrent_state: Optional[Tensor] = None


# ──────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch fallbacks
# ──────────────────────────────────────────────────────────────────────────────

def _torch_causal_conv1d_update(
    hidden_states: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Single-step causal conv update for autoregressive inference."""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hs = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hs[:, :, -state_len:])
    out = F.conv1d(hs, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


def _l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    chunk_size: int = 64,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Pure-PyTorch chunked GatedDeltaRule recurrence."""
    init_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    # [B, S, H, D] → [B, H, S, D]
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    B, H, S, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - S % chunk_size) % chunk_size
    query  = F.pad(query,  (0, 0, 0, pad))
    key    = F.pad(key,    (0, 0, 0, pad))
    value  = F.pad(value,  (0, 0, 0, pad))
    beta   = F.pad(beta,   (0, pad))
    g      = F.pad(g,      (0, pad))
    T = S + pad
    scale = 1.0 / math.sqrt(k_dim)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key   * beta.unsqueeze(-1)
    # Reshape to chunks: [B, H, num_chunks, chunk_size, D]
    query, key, value, k_beta, v_beta = [
        x.reshape(B, H, -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(B, H, -1, chunk_size)
    triu_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    # A[i,j] = -(k_beta[i] @ key[j]) * decay[i,j]  for j < i, else 0
    # (strictly lower-triangular)
    A = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(triu_mask, 0)
    # The 63-iteration Python loop + `attn += eye` was computing (I - A)^{-1}
    # via forward substitution (proved by induction: result[i,j] = [(I-A)^{-1}]_{ij}).
    # Replace with two solve_triangular calls that apply (I - A)^{-1} directly to the
    # RHS matrices, avoiding both the Python loop and the full [B,H,C,C,C] inverse.
    _eye = torch.eye(chunk_size, dtype=A.dtype, device=A.device)
    I_minus_A = _eye - A          # unit lower-triangular [B, H, num_chunks, C, C]
    value = torch.linalg.solve_triangular(
        I_minus_A, v_beta.contiguous(), upper=False
    )                             # [B, H, num_chunks, C, v_dim]
    k_cumdecay = torch.linalg.solve_triangular(
        I_minus_A, (k_beta * g.exp().unsqueeze(-1)).contiguous(), upper=False
    )                             # [B, H, num_chunks, C, k_dim]
    state = (
        torch.zeros(B, H, k_dim, v_dim, device=query.device, dtype=torch.float32)
        if initial_state is None else initial_state.float()
    )
    out = torch.zeros_like(value)
    triu2 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for ci in range(T // chunk_size):
        q_i, k_i, v_i = query[:, :, ci], key[:, :, ci], value[:, :, ci]
        a_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, ci]).masked_fill_(triu2, 0)
        v_prime = k_cumdecay[:, :, ci] @ state
        v_new = v_i - v_prime
        out[:, :, ci] = (q_i * g[:, :, ci, :, None].exp()) @ state + a_i @ v_new
        state = (
            state * g[:, :, ci, -1, None, None].exp()
            + (k_i * (g[:, :, ci, -1, None] - g[:, :, ci]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    final_state = state if output_final_state else None
    out = out.reshape(B, H, T, -1)[:, :, :S]
    out = out.transpose(1, 2).contiguous().to(init_dtype)  # [B, S, H, v_dim]
    return out, final_state


def _recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Pure-PyTorch token-by-token GatedDeltaRule for inference."""
    init_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    B, H, S, k_dim = key.shape
    v_dim = value.shape[-1]
    scale = 1.0 / math.sqrt(k_dim)
    query = query * scale
    out = torch.zeros(B, H, S, v_dim, device=query.device, dtype=torch.float32)
    state = (
        torch.zeros(B, H, k_dim, v_dim, device=query.device, dtype=torch.float32)
        if initial_state is None else initial_state.float()
    )
    for t in range(S):
        q_t, k_t, v_t = query[:, :, t], key[:, :, t], value[:, :, t]
        g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        b_t = beta[:, :, t].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * b_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    final_state = state if output_final_state else None
    out = out.transpose(1, 2).contiguous().to(init_dtype)  # [B, S, H, v_dim]
    return out, final_state


# ──────────────────────────────────────────────────────────────────────────────
# RMSNorm with gating (output norm)
# ──────────────────────────────────────────────────────────────────────────────

class _RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = self.weight * x.to(dtype)
        return x * F.silu(gate.float()).to(dtype)


# ──────────────────────────────────────────────────────────────────────────────
# GatedDeltaNet module
# ──────────────────────────────────────────────────────────────────────────────

class FujiGatedDeltaNet(nn.Module):
    """GatedDeltaNet linear attention layer (Megatron [S, B, D] convention).

    Internally converts to batch-first [B, S, D] for computation then back.

    Args:
        config:     TransformerConfig.  Used fields:
                    hidden_size, linear_num_key_heads, linear_num_value_heads,
                    linear_key_head_dim, linear_value_head_dim,
                    linear_conv_kernel_dim, hidden_act, rms_norm_eps.
        layer_idx:  0-indexed layer position (used for cache addressing).
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Attention head configuration (reuse linear_* fields from TransformerConfig)
        self.num_k_heads = getattr(config, 'linear_num_key_heads', 16)
        self.num_v_heads = getattr(config, 'linear_num_value_heads', 32)
        self.k_head_dim  = getattr(config, 'linear_key_head_dim',   128)
        self.v_head_dim  = getattr(config, 'linear_value_head_dim', 128)
        self.key_dim   = self.num_k_heads * self.k_head_dim
        self.value_dim = self.num_v_heads * self.v_head_dim
        self.conv_kernel = getattr(config, 'linear_conv_kernel_dim', 4)
        self.act_name = getattr(config, 'hidden_act', 'silu')
        self.norm_eps = getattr(config, 'rms_norm_eps', 1e-6)

        # Convolution operates on [key_dim*2 + value_dim] channels
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim,
            kernel_size=self.conv_kernel,
            padding=self.conv_kernel - 1,
            groups=self.conv_dim,
            bias=False,
        )

        # Input projections
        proj_qkvz = self.key_dim * 2 + self.value_dim * 2
        proj_ba   = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, proj_qkvz, bias=False)
        self.in_proj_ba   = nn.Linear(self.hidden_size, proj_ba,   bias=False)

        # Learnable time-step and decay parameters.
        # A ~ U[0.001, 0.016] → per-step g ≈ -0.007, cumsum(64) ≈ -0.45,
        # exp(-0.45) ≈ 0.64: healthy cross-chunk state decay.
        # (Previously A ~ U[0, 16] → cumsum(64) ≈ -672, exp(-672) ≈ 0: state zeroed.)
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0.001, 0.016)
        self.A_log  = nn.Parameter(A.log())

        # Output normalisation (RMSNorm + gating)
        self.norm = _RMSNormGated(self.v_head_dim, eps=self.norm_eps)

        # Output projection
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # Select compute kernels
        self._causal_conv1d_update = causal_conv1d_update or _torch_causal_conv1d_update
        self._causal_conv1d_fn = causal_conv1d_fn
        self._chunk_fn  = chunk_gated_delta_rule   or _chunk_gated_delta_rule
        self._recur_fn  = fused_recurrent_gated_delta_rule or _recurrent_gated_delta_rule

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_qkvz_ba(
        self, qkvz: Tensor, ba: Tensor
    ):
        """Split projected tensors into Q, K, V, Z, beta, alpha."""
        B, S, _ = qkvz.shape
        # Reshape to [B, S, num_k_heads, per_head_dim]
        per_head = (
            self.k_head_dim,               # Q
            self.k_head_dim,               # K
            self.num_v_heads // self.num_k_heads * self.v_head_dim,   # V
            self.num_v_heads // self.num_k_heads * self.v_head_dim,   # Z
        )
        qkvz = qkvz.view(B, S, self.num_k_heads, sum(per_head))
        q, k, v, z = qkvz.split(list(per_head), dim=-1)
        # Reshape V and Z to [B, S, num_v_heads, v_head_dim]
        ratio = self.num_v_heads // self.num_k_heads
        v = v.reshape(B, S, -1, self.v_head_dim)
        z = z.reshape(B, S, -1, self.v_head_dim)

        per_ba = (ratio, ratio)
        ba = ba.view(B, S, self.num_k_heads, sum(per_ba))
        b, a = ba.split(list(per_ba), dim=-1)
        b = b.reshape(B, S, self.num_v_heads)
        a = a.reshape(B, S, self.num_v_heads)
        return q, k, v, z, b, a

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        inference_cache: Optional[GatedDeltaNetInferenceCache] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """GatedDeltaNet forward pass.

        Args:
            hidden_states:  [S, B, D]  (Megatron sequence-first).
            inference_cache: Per-layer cache for autoregressive inference.
                             If provided and populated, uses recurrent mode.
            attention_mask:  [B, S] padding mask (1 = keep, 0 = pad).
                             Applied to hidden_states before projection.

        Returns:
            output: [S, B, D]
        """
        # Megatron [S, B, D] → batch-first [B, S, D]
        hidden_states = hidden_states.transpose(0, 1)  # [B, S, D]
        B, S, D = hidden_states.shape

        # Zero-out padding tokens (matches Qwen3-Next's apply_mask_to_padding_states)
        if attention_mask is not None and attention_mask.shape[1] > 1 and B > 1:
            dtype = hidden_states.dtype
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        is_inference_step = (
            inference_cache is not None
            and inference_cache.conv_state is not None
            and S == 1
        )

        # Project input
        qkvz = self.in_proj_qkvz(hidden_states)   # [B, S, proj_qkvz]
        ba   = self.in_proj_ba(hidden_states)      # [B, S, proj_ba]
        q, k, v, z, b, a = self._split_qkvz_ba(qkvz, ba)

        # Flatten Q, K, V to channel dimension for conv1d: [B, S, conv_dim]
        q_flat = q.reshape(B, S, -1)
        k_flat = k.reshape(B, S, -1)
        v_flat = v.reshape(B, S, -1)
        mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1).transpose(1, 2)  # [B, conv_dim, S]

        # Causal convolution
        if is_inference_step:
            mixed_qkv = self._causal_conv1d_update(
                mixed_qkv,
                inference_cache.conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
            )
        else:
            if inference_cache is not None:
                inference_cache.conv_state = F.pad(
                    mixed_qkv, (self.conv_kernel - S, 0)
                )
            if self._causal_conv1d_fn is not None:
                mixed_qkv = self._causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation='silu',
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :S])

        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, S, conv_dim]
        q_flat, k_flat, v_flat = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        q = q_flat.reshape(B, S, self.num_k_heads, self.k_head_dim)
        k = k_flat.reshape(B, S, self.num_k_heads, self.k_head_dim)
        v = v_flat.reshape(B, S, self.num_v_heads, self.v_head_dim)

        # Beta and decay (g)
        beta = b.sigmoid()
        g = (-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())).to(a.dtype)

        # Expand Q/K if num_v_heads > num_k_heads (grouped)
        ratio = self.num_v_heads // self.num_k_heads
        if ratio > 1:
            q = q.repeat_interleave(ratio, dim=2)
            k = k.repeat_interleave(ratio, dim=2)

        # Recurrence
        if not is_inference_step:
            core_out, final_state = self._chunk_fn(
                q, k, v, g=g, beta=beta,
                initial_state=None,
                output_final_state=(inference_cache is not None),
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_out, final_state = self._recur_fn(
                q, k, v, g=g, beta=beta,
                initial_state=inference_cache.recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        if inference_cache is not None and final_state is not None:
            inference_cache.recurrent_state = final_state

        # Output normalisation: flatten [B, S, H, v_head_dim] → [B*S, H*v_head_dim]
        z_shape = z.shape
        core_out = core_out.reshape(-1, core_out.shape[-1])
        z_flat   = z.reshape(-1, z.shape[-1])
        core_out = self.norm(core_out, z_flat)
        core_out = core_out.reshape(z_shape[0], z_shape[1], -1)  # [B, S, value_dim]

        output = self.out_proj(core_out)  # [B, S, D]

        # Back to Megatron [S, B, D]
        return output.transpose(0, 1)
