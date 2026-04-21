# Copyright (c) 2026 Susono authors.
"""Fused Switch Transformer auxiliary load-balancing loss.

Replaces the multi-kernel sequence::

    routing_map = torch.zeros_like(routing_probs)     # alloc
    routing_map.scatter_(1, top_indices, 1.0)          # scatter
    tokens_per_expert = routing_map.sum(dim=0)         # reduce (T over E)
    f = tokens_per_expert / (T * top_k)
    P = routing_probs.mean(dim=0)                       # reduce (T over E)
    aux_loss = num_experts * torch.dot(f, P) * coeff

with one Triton kernel (per-expert parallelism) that computes both
``tokens_per_expert`` (counts) and ``mean_probs`` in one pass. The final small
dot product is done in PyTorch (tiny E-sized op).

Backward propagates gradient to ``routing_probs`` only — ``top_indices`` is
non-differentiable through ``torch.topk``, so ``f`` is treated as a constant.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except ImportError:
    _HAVE_TRITON = False


def _pytorch_aux_loss(
    routing_probs: Tensor,
    top_indices: Tensor,
    num_experts: int,
    top_k: int,
    coeff: float,
) -> Tensor:
    T = routing_probs.shape[0]
    routing_map = torch.zeros_like(routing_probs)
    routing_map.scatter_(1, top_indices, 1.0)
    f = routing_map.sum(dim=0) / (T * top_k)
    P = routing_probs.mean(dim=0)
    return num_experts * torch.dot(f, P) * coeff


if _HAVE_TRITON:

    @triton.jit
    def _fused_aux_loss_fwd_kernel(
        probs_ptr,      # [T, E]   fp32 routing_probs
        top_idx_ptr,    # [T, K]   int64 indices
        counts_ptr,     # [E]      fp32 output: tokens_per_expert
        mean_p_ptr,     # [E]      fp32 output: mean_probs
        T, E,
        K: tl.constexpr,
        T_BLOCK: tl.constexpr,
    ):
        e = tl.program_id(0)
        if e >= E:
            return

        count = tl.zeros([], dtype=tl.float32)
        sum_p = tl.zeros([], dtype=tl.float32)

        for t_blk in range(tl.cdiv(T, T_BLOCK)):
            offs = t_blk * T_BLOCK + tl.arange(0, T_BLOCK)
            mask = offs < T

            # Count occurrences of expert e in top_idx[offs, :]
            for k in tl.static_range(K):
                idx = tl.load(
                    top_idx_ptr + offs * K + k,
                    mask=mask, other=-1,
                ).to(tl.int64)
                matches = tl.where(mask & (idx == e), 1.0, 0.0)
                count = count + tl.sum(matches, axis=0)

            # Accumulate probs[:, e]
            p = tl.load(
                probs_ptr + offs * E + e,
                mask=mask, other=0.0,
            ).to(tl.float32)
            sum_p = sum_p + tl.sum(p, axis=0)

        tl.store(counts_ptr + e, count)
        tl.store(mean_p_ptr + e, sum_p / T)


    class FusedAuxLoss(torch.autograd.Function):
        """Fused Switch Transformer aux loss scalar (with backward to routing_probs)."""

        @staticmethod
        def forward(ctx, routing_probs, top_indices, num_experts, top_k, coeff):
            assert routing_probs.is_cuda and top_indices.is_cuda
            assert routing_probs.dtype == torch.float32, (
                "fused_aux_loss requires fp32 routing_probs (matches fused_router contract)"
            )
            assert top_indices.dtype == torch.int64
            T, E = routing_probs.shape
            K = int(top_k)

            probs_c = routing_probs.contiguous()
            idx_c = top_indices.contiguous()
            counts = torch.empty(E, dtype=torch.float32, device=routing_probs.device)
            mean_p = torch.empty(E, dtype=torch.float32, device=routing_probs.device)

            _fused_aux_loss_fwd_kernel[(E,)](
                probs_c, idx_c, counts, mean_p,
                T=T, E=E, K=K, T_BLOCK=1024,
            )

            f = counts / (T * K)
            aux_loss = num_experts * torch.dot(f, mean_p) * coeff

            ctx.save_for_backward(f)
            ctx.T = T
            ctx.E = E
            ctx.num_experts = num_experts
            ctx.coeff = coeff
            return aux_loss

        @staticmethod
        def backward(ctx, daux):
            (f,) = ctx.saved_tensors
            # dL/drouting_probs[t, e] = daux * (num_experts * coeff / T) * f[e]
            scale = ctx.num_experts * ctx.coeff / ctx.T
            dprobs = (daux * scale) * f                             # [E]
            dprobs = dprobs.unsqueeze(0).expand(ctx.T, ctx.E).contiguous()
            return dprobs, None, None, None, None


def fused_aux_loss(
    routing_probs: Tensor,
    top_indices: Tensor,
    num_experts: int,
    top_k: int,
    coeff: float,
) -> Tensor:
    """Fused Switch aux loss scalar: ``num_experts * sum(f * P) * coeff``.

    - ``f[e] = (count of e in top_indices) / (T * top_k)``
    - ``P[e] = mean_t(routing_probs[t, e])``

    ``coeff == 0`` short-circuits to a scalar zero. Falls back to PyTorch when
    Triton / CUDA unavailable or dtype mismatch.
    """
    if coeff == 0.0:
        return torch.zeros(
            (), dtype=routing_probs.dtype, device=routing_probs.device,
        )
    if (
        _HAVE_TRITON
        and routing_probs.is_cuda
        and top_indices.is_cuda
        and routing_probs.dtype == torch.float32
        and top_indices.dtype == torch.int64
    ):
        return FusedAuxLoss.apply(routing_probs, top_indices, num_experts, top_k, coeff)
    return _pytorch_aux_loss(routing_probs, top_indices, num_experts, top_k, coeff)
