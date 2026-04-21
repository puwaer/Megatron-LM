# Copyright (c) 2026 Susono authors.
"""Fused router: softmax + top-k + optional renormalize in a single Triton kernel.

Replaces the sequence::

    probs = F.softmax(logits, dim=-1, dtype=torch.float)
    top_w, top_idx = torch.topk(probs, k, dim=-1)
    if norm_topk_prob:
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + eps)

(~4 kernel launches: softmax + topk + sum + div) with one fused Triton kernel
per direction.

Constraints:
  - One program per token (parallelism = T).
  - Each token loads all ``E`` expert logits into SRAM (``E_BLOCK = next_pow2(E)``).
    Suitable for moderate expert counts (E up to ~1024).
  - K must be small and ``tl.constexpr`` (K ≤ 16). For Susono K=4 this is trivial.
  - Internally fp32; routing_probs output is always fp32 (matches original
    ``F.softmax(..., dtype=torch.float)`` contract); top_weights is input-dtype.

Fall back to PyTorch when Triton is unavailable or shape requirements unmet.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except ImportError:
    _HAVE_TRITON = False


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _pytorch_router(
    logits: Tensor, top_k: int, norm_topk_prob: bool, eps: float = 1e-6
) -> Tuple[Tensor, Tensor, Tensor]:
    # Mirrors the Triton kernel contract: probs stays fp32.
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    top_w, top_idx = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + eps)
    return probs, top_w.to(logits.dtype), top_idx


if _HAVE_TRITON:

    @triton.jit
    def _fused_router_fwd_kernel(
        logits_ptr,        # [T, E]   input dtype (bf16/fp16/fp32)
        probs_ptr,         # [T, E]   output: fp32 routing_probs
        top_w_ptr,         # [T, K]   output: top-K weights in input dtype
        top_idx_ptr,       # [T, K]   output: int64 indices
        norm_scale_ptr,    # [T]      output: fp32 1/(sum+eps), saved for backward
        T, E,
        K: tl.constexpr,
        E_BLOCK: tl.constexpr,
        EPS: tl.constexpr,
        NORM_TOPK: tl.constexpr,
    ):
        t = tl.program_id(0)
        if t >= T:
            return

        offs = tl.arange(0, E_BLOCK)
        mask = offs < E

        # Load logits, upcast to fp32 with masked positions set to -inf
        x = tl.load(logits_ptr + t * E + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x, -float('inf'))

        # Stable softmax
        m = tl.max(x, axis=0)
        e_v = tl.exp(x - m)
        e_v = tl.where(mask, e_v, 0.0)
        s = tl.sum(e_v, axis=0)
        p = e_v / s

        # Store full fp32 probs for aux loss use
        tl.store(
            probs_ptr + t * E + offs,
            p.to(probs_ptr.dtype.element_ty),
            mask=mask,
        )

        # Top-K via iterative argmax + masking (K small, unrolled)
        # NaN sanitization: NaN != NaN is true in IEEE 754, so this replaces NaNs with -inf.
        # Without this, argmax over a vector containing NaN has undefined semantics and may
        # return an index in the E_BLOCK padding region (>= E), which breaks downstream bincount.
        p_safe = tl.where(p == p, p, -float('inf'))
        p_work = tl.where(mask, p_safe, -float('inf'))
        top_sum = tl.zeros([], dtype=tl.float32)
        for k in tl.static_range(K):
            # Idempotent re-mask on every iteration as a belt-and-braces guard.
            p_work = tl.where(mask, p_work, -float('inf'))
            val = tl.max(p_work, axis=0)
            idx = tl.argmax(p_work, axis=0).to(tl.int64)
            # Hard clamp: even if argmax somehow returns OOB, force idx into [0, E).
            idx = tl.minimum(idx, tl.full([], E - 1, tl.int64))
            tl.store(top_w_ptr + t * K + k, val.to(top_w_ptr.dtype.element_ty))
            tl.store(top_idx_ptr + t * K + k, idx)
            p_work = tl.where(offs == idx, -float('inf'), p_work)
            top_sum = top_sum + val

        # Renormalize top-K (or leave as-is)
        if NORM_TOPK:
            inv_s = 1.0 / (top_sum + EPS)
        else:
            inv_s = tl.zeros([], dtype=tl.float32) + 1.0
        tl.store(norm_scale_ptr + t, inv_s)

        if NORM_TOPK:
            for k in tl.static_range(K):
                v = tl.load(top_w_ptr + t * K + k).to(tl.float32) * inv_s
                tl.store(top_w_ptr + t * K + k, v.to(top_w_ptr.dtype.element_ty))


    @triton.jit
    def _fused_router_bwd_kernel(
        probs_ptr,         # [T, E]   saved fp32 routing_probs
        top_w_ptr,         # [T, K]   saved post-norm top_w (input dtype)
        top_idx_ptr,       # [T, K]   saved indices (int64)
        norm_scale_ptr,    # [T]      saved 1/(s+eps), fp32
        dprobs_ext_ptr,    # [T, E]   external gradient (e.g. from aux loss path)
        dtop_w_ptr,        # [T, K]   gradient from downstream (expert path)
        dlogits_ptr,       # [T, E]   output gradient w.r.t. logits
        T, E,
        K: tl.constexpr,
        E_BLOCK: tl.constexpr,
        NORM_TOPK: tl.constexpr,
    ):
        t = tl.program_id(0)
        if t >= T:
            return

        offs = tl.arange(0, E_BLOCK)
        mask = offs < E

        inv_s = tl.load(norm_scale_ptr + t).to(tl.float32)

        # When NORM_TOPK, renorm backward: C = sum_k(dtop_w[k] * top_w_post[k])
        # When NORM_TOPK=False, top_w_post = top_w_pre (no renorm),
        #                      so dtop_w_pre[k] = dtop_w[k] (identity derivative).
        C = tl.zeros([], dtype=tl.float32)
        if NORM_TOPK:
            for k in tl.static_range(K):
                dtw_k = tl.load(dtop_w_ptr + t * K + k).to(tl.float32)
                tw_k = tl.load(top_w_ptr + t * K + k).to(tl.float32)
                C = C + dtw_k * tw_k

        # Scatter dw[k] into dp_scatter at top_idx[k] positions
        dp_scatter = tl.zeros([E_BLOCK], dtype=tl.float32)
        for k in tl.static_range(K):
            idx_k = tl.load(top_idx_ptr + t * K + k).to(tl.int64)
            dtw_k = tl.load(dtop_w_ptr + t * K + k).to(tl.float32)
            if NORM_TOPK:
                dw_k = (dtw_k - C) * inv_s
            else:
                dw_k = dtw_k
            dp_scatter = tl.where(offs == idx_k, dp_scatter + dw_k, dp_scatter)

        dp_ext = tl.load(dprobs_ext_ptr + t * E + offs, mask=mask, other=0.0).to(tl.float32)
        dp_total = dp_scatter + dp_ext

        # Softmax backward: dlogits[j] = p[j] * (dp_total[j] - sum_i(dp_total[i] * p[i]))
        p = tl.load(probs_ptr + t * E + offs, mask=mask, other=0.0).to(tl.float32)
        dp_valid = tl.where(mask, dp_total, 0.0)
        p_valid = tl.where(mask, p, 0.0)
        D = tl.sum(dp_valid * p_valid, axis=0)
        dlogits = p_valid * (dp_total - D)

        tl.store(
            dlogits_ptr + t * E + offs,
            dlogits.to(dlogits_ptr.dtype.element_ty),
            mask=mask,
        )


    class FusedRouter(torch.autograd.Function):
        """Fused softmax + top-k + renormalize with autograd."""

        @staticmethod
        def forward(ctx, logits, top_k, norm_topk_prob, eps=1e-6):
            assert logits.is_cuda, "fused_router requires CUDA"
            assert logits.dim() == 2, f"expected [T, E], got {logits.shape}"
            T, E = logits.shape
            K = int(top_k)
            E_BLOCK = _next_pow2(E)

            logits_c = logits.contiguous()
            # Match original F.softmax(dtype=torch.float) contract: fp32 probs
            probs = torch.empty(T, E, dtype=torch.float32, device=logits.device)
            top_w = torch.empty(T, K, dtype=logits.dtype, device=logits.device)
            top_idx = torch.empty(T, K, dtype=torch.int64, device=logits.device)
            norm_scale = torch.empty(T, dtype=torch.float32, device=logits.device)

            _fused_router_fwd_kernel[(T,)](
                logits_c, probs, top_w, top_idx, norm_scale,
                T=T, E=E,
                K=K, E_BLOCK=E_BLOCK,
                EPS=float(eps), NORM_TOPK=bool(norm_topk_prob),
            )

            ctx.save_for_backward(probs, top_w, top_idx, norm_scale)
            ctx.K = K
            ctx.E_BLOCK = E_BLOCK
            ctx.NORM_TOPK = bool(norm_topk_prob)
            return probs, top_w, top_idx

        @staticmethod
        def backward(ctx, dprobs_ext, dtop_w, dtop_idx_unused):
            probs, top_w, top_idx, norm_scale = ctx.saved_tensors
            T, E = probs.shape
            if dprobs_ext is None:
                dprobs_ext = torch.zeros_like(probs)
            if dtop_w is None:
                dtop_w = torch.zeros_like(top_w)
            dprobs_ext = dprobs_ext.contiguous()
            dtop_w = dtop_w.contiguous()

            dlogits = torch.empty_like(probs)
            _fused_router_bwd_kernel[(T,)](
                probs, top_w, top_idx, norm_scale,
                dprobs_ext, dtop_w, dlogits,
                T=T, E=E,
                K=ctx.K, E_BLOCK=ctx.E_BLOCK,
                NORM_TOPK=ctx.NORM_TOPK,
            )
            # dlogits is fp32 (probs dtype); caller may need to cast
            return dlogits.to(top_w.dtype), None, None, None


def fused_router(
    logits: Tensor, top_k: int, norm_topk_prob: bool = True, eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fused softmax + top-k + optional renormalize.

    Returns (routing_probs [T, E] fp32, top_weights [T, K] input-dtype, top_indices [T, K] int64).

    Falls back to PyTorch when Triton / CUDA unavailable.
    """
    if (
        _HAVE_TRITON
        and logits.is_cuda
        and logits.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and logits.dim() == 2
    ):
        return FusedRouter.apply(logits, top_k, norm_topk_prob, eps)
    return _pytorch_router(logits, top_k, norm_topk_prob, eps)
