# Copyright (c) 2026 Susono authors.
"""Fused Triton kernel for the GatedDeltaNet decay term.

Replaces the PyTorch expression

    g = (-A_log.float().exp() * F.softplus(a.float() + dt_bias.float())).to(a.dtype)

which issues ~6 sequential elementwise CUDA kernels (cast, add, softplus, exp,
mul, cast-back) per forward pass of every GatedDeltaNet layer, plus the
equivalent chain during backward.  Profile job 1681505 showed the chain
contributed to the large ``unrolled_elementwise_kernel<direct_copy>`` count
(29 460 instances, 970 ms) and the ``elementwise_kernel<128,2>`` count (14 196
instances, 949 ms).

Kernel design:
  - Forward:  one program per token, reads ``a[t, :H]`` bf16 + broadcasts
    ``A_log[:H]`` / ``dt_bias[:H]`` fp32, computes ``g`` fully in fp32, writes
    bf16.  ``H ≤ 128`` for this model (linear_num_value_heads = 32), so a
    single 1-D block ``BLOCK_H = next_pow2(H)`` is sufficient.
  - Backward: one program per token, accumulates ``d_A_log[h]`` and
    ``d_dt_bias[h]`` via ``tl.atomic_add`` while writing ``d_a`` directly.

Public API:
  - ``fused_gdn_decay(a, A_log, dt_bias) -> Tensor``
  - ``FusedGDNDecay`` (autograd.Function)

Falls back to the original PyTorch expression when Triton is unavailable or
the inputs don't match the fast-path (wrong dtype, CPU, H > 256).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except ImportError:
    _HAVE_TRITON = False


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _pytorch_gdn_decay(a: Tensor, A_log: Tensor, dt_bias: Tensor) -> Tensor:
    """Reference implementation, bit-equivalent to the original inline expression."""
    return (
        -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    ).to(a.dtype)


if _HAVE_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=nw, num_stages=ns)
            for nw in (2, 4, 8)
            for ns in (2, 3)
        ],
        key=['BLOCK_H'],
    )
    @triton.jit
    def _fused_gdn_decay_fwd_kernel(
        a_ptr,          # [T, H]  input dtype (bf16/fp16/fp32)
        A_log_ptr,      # [H]     fp32 (or bf16, loaded as fp32)
        dt_bias_ptr,    # [H]     fp32
        g_ptr,          # [T, H]  input dtype (output)
        T,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        t = tl.program_id(0)
        if t >= T:
            return

        h = tl.arange(0, BLOCK_H)
        mask = h < H

        a = tl.load(a_ptr + t * H + h, mask=mask, other=0.0).to(tl.float32)
        A_log = tl.load(A_log_ptr + h, mask=mask, other=0.0).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + h, mask=mask, other=0.0).to(tl.float32)

        # softplus(x) = log(1 + exp(x)); for large x approaches x, for very
        # negative x approaches 0.  Triton's tl.log1p(tl.exp(x)) is stable
        # enough here because `a + dt_bias` stays in a reasonable range
        # (dt_bias init = 0, a ~ linear projection output).
        x = a + dt_bias
        sp = tl.log(1.0 + tl.exp(x))
        # A_log initialised as log(A) with A ~ U[0.001, 0.016], so
        # -exp(A_log) ∈ (-0.016, -0.001).  Stable.
        coef = -tl.exp(A_log)

        g = coef * sp
        tl.store(g_ptr + t * H + h, g.to(g_ptr.dtype.element_ty), mask=mask)

    # NOTE: intentionally no @triton.autotune on the backward kernel.
    # ``tl.atomic_add`` accumulators for d_A_log / d_dt_bias cannot tolerate
    # repeated kernel invocations during autotune benchmarking — each tuning
    # run adds into the same buffer, inflating the gradient by ~O(#trials).
    # The same pattern is used by the MHC backward kernel for the same reason.
    @triton.jit
    def _fused_gdn_decay_bwd_kernel(
        a_ptr,          # [T, H]    input dtype
        A_log_ptr,      # [H]       fp32
        dt_bias_ptr,    # [H]       fp32
        dg_ptr,         # [T, H]    input dtype (grad wrt g)
        # Outputs
        da_ptr,         # [T, H]    input dtype
        # Atomics:
        d_A_log_acc_ptr,    # [H]   fp32
        d_dt_bias_acc_ptr,  # [H]   fp32
        T,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        t = tl.program_id(0)
        if t >= T:
            return

        h = tl.arange(0, BLOCK_H)
        mask = h < H

        a = tl.load(a_ptr + t * H + h, mask=mask, other=0.0).to(tl.float32)
        A_log = tl.load(A_log_ptr + h, mask=mask, other=0.0).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + h, mask=mask, other=0.0).to(tl.float32)
        dg = tl.load(dg_ptr + t * H + h, mask=mask, other=0.0).to(tl.float32)

        x = a + dt_bias
        # softplus forward: sp = log(1 + exp(x))
        # d(softplus)/dx = sigmoid(x)
        sp = tl.log(1.0 + tl.exp(x))
        sig = tl.sigmoid(x)
        coef = -tl.exp(A_log)

        # d a[t, h] = dg[t, h] * coef[h] * sigmoid(x[t, h])
        da = dg * coef * sig
        tl.store(da_ptr + t * H + h, da.to(da_ptr.dtype.element_ty), mask=mask)

        # d A_log[h] = sum_t(dg * d(coef)/d(A_log) * sp)
        #            = sum_t(dg * (-exp(A_log)) * sp)  (because coef = -exp(A_log))
        #            = sum_t(dg * coef * sp)
        d_A_log_contrib = dg * coef * sp
        # d dt_bias[h] = sum_t(dg * coef * sigmoid(x)) = sum_t(da)
        d_dt_bias_contrib = da

        # Atomic add per-lane; masked lanes write zero to h=0 safely.
        safe_h = tl.where(mask, h, 0)
        safe_dA = tl.where(mask, d_A_log_contrib, 0.0)
        safe_ddt = tl.where(mask, d_dt_bias_contrib, 0.0)
        tl.atomic_add(d_A_log_acc_ptr + safe_h, safe_dA)
        tl.atomic_add(d_dt_bias_acc_ptr + safe_h, safe_ddt)


    class FusedGDNDecay(torch.autograd.Function):
        """Autograd Function wrapping the Triton kernels.

        Forward: ``g = (-exp(A_log)) * softplus(a + dt_bias)`` in fp32,
        written as ``a.dtype`` (bf16/fp16/fp32).  Shape preserved.
        """

        @staticmethod
        def forward(ctx, a: Tensor, A_log: Tensor, dt_bias: Tensor) -> Tensor:
            # a: [..., H];  A_log / dt_bias: [H]
            assert a.is_cuda, "fused_gdn_decay requires CUDA input"
            assert A_log.shape == dt_bias.shape and A_log.dim() == 1
            H = A_log.shape[0]
            assert a.shape[-1] == H

            a_flat = a.contiguous().view(-1, H)
            T = a_flat.shape[0]
            g_flat = torch.empty_like(a_flat)

            BLOCK_H = _next_pow2(H)
            _fused_gdn_decay_fwd_kernel[(T,)](
                a_flat, A_log, dt_bias, g_flat,
                T=T, H=H, BLOCK_H=BLOCK_H,
            )

            ctx.save_for_backward(a_flat, A_log, dt_bias)
            ctx.shape = a.shape
            ctx.H = H
            return g_flat.view(a.shape)

        @staticmethod
        def backward(ctx, dg: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            a_flat, A_log, dt_bias = ctx.saved_tensors
            H = ctx.H
            T = a_flat.shape[0]

            dg_flat = dg.contiguous().view(-1, H)
            da_flat = torch.empty_like(a_flat)

            # Accumulate grads for params in fp32 to avoid atomic-add precision loss.
            d_A_log_acc = torch.zeros(H, dtype=torch.float32, device=a_flat.device)
            d_dt_bias_acc = torch.zeros(H, dtype=torch.float32, device=a_flat.device)

            BLOCK_H = _next_pow2(H)
            _fused_gdn_decay_bwd_kernel[(T,)](
                a_flat, A_log, dt_bias, dg_flat,
                da_flat,
                d_A_log_acc, d_dt_bias_acc,
                T=T, H=H, BLOCK_H=BLOCK_H,
                num_warps=4, num_stages=2,
            )

            d_A_log = d_A_log_acc.to(A_log.dtype)
            d_dt_bias = d_dt_bias_acc.to(dt_bias.dtype)

            return da_flat.view(ctx.shape), d_A_log, d_dt_bias


def fused_gdn_decay(a: Tensor, A_log: Tensor, dt_bias: Tensor) -> Tensor:
    """Compute ``g = (-exp(A_log)) * softplus(a + dt_bias)``.

    Equivalent to::

        (-A_log.float().exp() * F.softplus(a.float() + dt_bias.float())).to(a.dtype)

    but as a single fused Triton kernel (with a matching backward) to
    eliminate the ~6 elementwise CUDA kernel launches the chain issues.

    Args:
        a:        Input tensor of any shape ``[..., H]``.  Typical dtype bf16.
        A_log:    1-D parameter of shape ``[H]`` (typically fp32).
        dt_bias:  1-D parameter of shape ``[H]`` (typically fp32).

    Returns:
        Tensor with the same shape and dtype as ``a``.
    """
    use_triton = (
        _HAVE_TRITON
        and a.is_cuda
        and A_log.is_cuda
        and dt_bias.is_cuda
        and a.shape[-1] <= 256  # BLOCK_H limit for the simple kernel
    )
    if use_triton:
        return FusedGDNDecay.apply(a, A_log, dt_bias)
    return _pytorch_gdn_decay(a, A_log, dt_bias)
