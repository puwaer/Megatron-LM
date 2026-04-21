# Copyright (c) 2026 Susono authors.
"""Fused ``sigmoid(gate_input) * value`` Triton kernel.

Replaces the two-kernel PyTorch sequence

    gate = torch.sigmoid(gate_input)
    out  = gate * value

which appears in multiple hot-path locations:

  - Engram ``engram_module.py:513-514`` (context-aware gating).
  - MoE shared-expert gate in ``susono_moe.py:761-762``.

With bfloat16 activations each of the above produces 2 kernel launches per
invocation.  In the 19-MoE-layer / 2-Engram-layer configuration with 3 passes
per iteration (forward + recompute + backward) this is roughly
``19*2 + 2*2 = 42`` fused invocations replacing ``~126`` raw launches per
iteration.

Kernel:
  - Single pass over the flat-elementwise view ``[N]`` with ``BLOCK``-sized
    tiles.
  - Forward: ``out = sigmoid(x) * y``.
  - Backward: ``ds = s * (1 - s) * y * dout``; ``dy = s * dout`` where
    ``s = sigmoid(x)`` is recomputed (fp32) to avoid storing ``s`` itself.

Falls back to the original PyTorch expression when Triton is unavailable or
the inputs don't satisfy the fast-path preconditions (CPU, fp64, etc.).
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


def _pytorch_sigmoid_mul(
    gate_input: Tensor, value: Tensor, bias: Optional[Tensor] = None
) -> Tensor:
    if bias is not None:
        gate_input = gate_input + bias
    return torch.sigmoid(gate_input) * value


if _HAVE_TRITON:

    _SIGMOID_MUL_AUTOTUNE_CONFIGS = [
        triton.Config({'BLOCK': bs}, num_warps=nw, num_stages=ns)
        for bs in (1024, 2048, 4096)
        for nw in (4, 8)
        for ns in (2, 3)
    ]

    @triton.autotune(
        configs=_SIGMOID_MUL_AUTOTUNE_CONFIGS,
        key=['N'],
    )
    @triton.jit
    def _fused_sigmoid_mul_fwd_kernel(
        x_ptr, y_ptr, out_ptr,
        N,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        s = tl.sigmoid(x)
        out = s * y
        tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.autotune(
        configs=_SIGMOID_MUL_AUTOTUNE_CONFIGS,
        key=['N'],
    )
    @triton.jit
    def _fused_sigmoid_mul_bwd_kernel(
        x_ptr, y_ptr, dout_ptr,
        dx_ptr, dy_ptr,
        N,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dout = tl.load(dout_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        s = tl.sigmoid(x)
        # dy = s * dout
        dy = s * dout
        # dx = s * (1 - s) * y * dout   (chain: ds/dx = s * (1-s); dout/ds = y)
        dx = s * (1.0 - s) * y * dout

        tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
        tl.store(dy_ptr + offs, dy.to(dy_ptr.dtype.element_ty), mask=mask)


    class FusedSigmoidMul(torch.autograd.Function):
        """out = sigmoid(gate_input) * value."""

        @staticmethod
        def forward(ctx, gate_input: Tensor, value: Tensor) -> Tensor:
            assert gate_input.shape == value.shape, (
                f"shapes must match: gate_input {gate_input.shape}, value {value.shape}"
            )
            assert gate_input.is_cuda and value.is_cuda

            gate_input_c = gate_input.contiguous()
            value_c = value.contiguous()

            out = torch.empty_like(value_c)
            N = value_c.numel()

            def grid(meta):
                return (triton.cdiv(N, meta['BLOCK']),)

            _fused_sigmoid_mul_fwd_kernel[grid](
                gate_input_c, value_c, out,
                N=N,
            )

            ctx.save_for_backward(gate_input_c, value_c)
            ctx.shape = value.shape
            return out

        @staticmethod
        def backward(ctx, dout: Tensor) -> Tuple[Tensor, Tensor]:
            gate_input, value = ctx.saved_tensors
            dout_c = dout.contiguous()

            dx = torch.empty_like(gate_input)
            dy = torch.empty_like(value)
            N = value.numel()

            def grid(meta):
                return (triton.cdiv(N, meta['BLOCK']),)

            _fused_sigmoid_mul_bwd_kernel[grid](
                gate_input, value, dout_c,
                dx, dy,
                N=N,
            )

            return dx, dy


if _HAVE_TRITON:

    @triton.autotune(
        configs=_SIGMOID_MUL_AUTOTUNE_CONFIGS,
        key=['D'],
    )
    @triton.jit
    def _fused_sigmoid_row_gate_mul_fwd_kernel(
        gate_ptr,   # [T]    per-row scalar (post-squeeze)
        value_ptr,  # [T, D]
        out_ptr,    # [T, D]
        bias_ptr,   # [1] scalar tensor (same dtype as gate), unused if HAS_BIAS=False
        HAS_BIAS: tl.constexpr,
        T, D,
        BLOCK: tl.constexpr,
    ):
        t = tl.program_id(0)
        if t >= T:
            return

        gate = tl.load(gate_ptr + t).to(tl.float32)
        if HAS_BIAS:
            gate = gate + tl.load(bias_ptr).to(tl.float32)
        s = tl.sigmoid(gate)

        for d_blk in range(tl.cdiv(D, BLOCK)):
            offs = d_blk * BLOCK + tl.arange(0, BLOCK)
            mask = offs < D
            y = tl.load(value_ptr + t * D + offs, mask=mask, other=0.0).to(tl.float32)
            out = s * y
            tl.store(
                out_ptr + t * D + offs,
                out.to(out_ptr.dtype.element_ty),
                mask=mask,
            )

    @triton.autotune(
        configs=_SIGMOID_MUL_AUTOTUNE_CONFIGS,
        key=['D'],
    )
    @triton.jit
    def _fused_sigmoid_row_gate_mul_bwd_kernel(
        gate_ptr,   # [T]
        value_ptr,  # [T, D]
        dout_ptr,   # [T, D]
        dgate_ptr,  # [T]          fp32 (accumulated scalar per row)
        dvalue_ptr, # [T, D]
        bias_ptr,   # [1] scalar tensor, unused if HAS_BIAS=False
        HAS_BIAS: tl.constexpr,
        T, D,
        BLOCK: tl.constexpr,
    ):
        t = tl.program_id(0)
        if t >= T:
            return

        gate = tl.load(gate_ptr + t).to(tl.float32)
        if HAS_BIAS:
            gate = gate + tl.load(bias_ptr).to(tl.float32)
        s = tl.sigmoid(gate)
        sgrad = s * (1.0 - s)  # d sigmoid / d gate

        # dgate[t] = sum_d(sgrad * value[t, d] * dout[t, d])
        # dvalue[t, d] = s * dout[t, d]
        acc = tl.zeros([1], dtype=tl.float32)
        for d_blk in range(tl.cdiv(D, BLOCK)):
            offs = d_blk * BLOCK + tl.arange(0, BLOCK)
            mask = offs < D
            y = tl.load(value_ptr + t * D + offs, mask=mask, other=0.0).to(tl.float32)
            dout = tl.load(dout_ptr + t * D + offs, mask=mask, other=0.0).to(tl.float32)

            dv = s * dout
            tl.store(
                dvalue_ptr + t * D + offs,
                dv.to(dvalue_ptr.dtype.element_ty),
                mask=mask,
            )
            acc += tl.sum(sgrad * y * dout)

        dgate_val = tl.sum(acc, axis=0)
        tl.store(dgate_ptr + t, dgate_val)


    class FusedSigmoidRowGateMul(torch.autograd.Function):
        """out = sigmoid(gate + bias) * value, where gate is [T, 1] (per-row scalar).

        ``bias`` is an optional scalar Parameter (shape [1] or ()). When provided,
        it is added to ``gate`` inside the kernel before ``sigmoid``, saving one
        extra elementwise kernel vs. doing ``nn.Linear(bias=True)`` separately.
        """

        @staticmethod
        def forward(
            ctx, gate: Tensor, value: Tensor, bias: Optional[Tensor] = None
        ) -> Tensor:
            assert gate.is_cuda and value.is_cuda
            assert gate.dtype == value.dtype
            assert gate.shape[-1] == 1, (
                f"expected trailing dim 1 for gate, got {gate.shape}"
            )
            assert gate.shape[:-1] == value.shape[:-1], (
                f"leading dims mismatch: gate {gate.shape}, value {value.shape}"
            )

            gate_flat = gate.contiguous().view(-1)     # [T]
            value_flat = value.contiguous().view(-1, value.shape[-1])  # [T, D]
            T, D = value_flat.shape
            out = torch.empty_like(value_flat)

            has_bias = bias is not None
            # Triton requires a pointer even when HAS_BIAS=False; pass gate_flat as dummy.
            bias_ptr = bias.contiguous().view(-1) if has_bias else gate_flat

            _fused_sigmoid_row_gate_mul_fwd_kernel[(T,)](
                gate_flat, value_flat, out,
                bias_ptr,
                HAS_BIAS=has_bias,
                T=T, D=D,
            )

            ctx.save_for_backward(gate_flat, value_flat, bias_ptr if has_bias else None)
            ctx.shape_gate = gate.shape
            ctx.shape_value = value.shape
            ctx.has_bias = has_bias
            return out.view(value.shape)

        @staticmethod
        def backward(ctx, dout: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
            saved = ctx.saved_tensors
            gate_flat, value_flat = saved[0], saved[1]
            bias_ptr = saved[2] if ctx.has_bias else gate_flat
            T, D = value_flat.shape
            dout_flat = dout.contiguous().view(-1, D)

            dgate_f32 = torch.empty(T, dtype=torch.float32, device=gate_flat.device)
            dvalue_flat = torch.empty_like(value_flat)

            _fused_sigmoid_row_gate_mul_bwd_kernel[(T,)](
                gate_flat, value_flat, dout_flat,
                dgate_f32, dvalue_flat,
                bias_ptr,
                HAS_BIAS=ctx.has_bias,
                T=T, D=D,
            )

            dgate = dgate_f32.to(gate_flat.dtype).view(ctx.shape_gate)
            # bias grad == sum(dgate) since bias is a linear shift on gate
            dbias = dgate_f32.sum().to(gate_flat.dtype).view(1) if ctx.has_bias else None
            return dgate, dvalue_flat.view(ctx.shape_value), dbias


def fused_sigmoid_mul(
    gate_input: Tensor, value: Tensor, bias: Optional[Tensor] = None
) -> Tensor:
    """Compute ``sigmoid(gate_input + bias) * value`` as a single fused kernel.

    Equivalent to::

        torch.sigmoid(gate_input + bias) * value   (bias treated as 0 when None)

    Issues one CUDA kernel launch for forward and one for backward.
    ``bias`` is only honored for the row-gate pattern (broadcast gate).

    Supports two shape patterns:
      1. ``gate_input.shape == value.shape`` — full elementwise (bias ignored,
         falls back to PyTorch when bias given since this path has no bias kernel).
      2. ``gate_input.shape[-1] == 1`` and leading dims match — per-row scalar
         gate broadcast across the last dimension of ``value``.  This is the
         MoE shared-expert gating pattern; bias is a scalar added inside the
         kernel before ``sigmoid``.
    """
    if (
        _HAVE_TRITON
        and gate_input.is_cuda
        and value.is_cuda
        and gate_input.dtype == value.dtype
    ):
        if gate_input.shape == value.shape:
            if bias is not None:
                return _pytorch_sigmoid_mul(gate_input, value, bias)
            return FusedSigmoidMul.apply(gate_input, value)
        if (
            gate_input.dim() == value.dim()
            and gate_input.shape[-1] == 1
            and gate_input.shape[:-1] == value.shape[:-1]
        ):
            return FusedSigmoidRowGateMul.apply(gate_input, value, bias)
    return _pytorch_sigmoid_mul(gate_input, value, bias)
