# Copyright (c) 2026 Susono authors.
"""Unit tests for megatron.core.fusions.fused_gdn_decay.

Validates forward and backward parity against the pure-PyTorch reference

    g = (-A_log.float().exp() * F.softplus(a.float() + dt_bias.float())).to(a.dtype)
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_gdn_decay import (
    _pytorch_gdn_decay,
    fused_gdn_decay,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused_gdn_decay requires CUDA",
)


def _make_inputs(
    B: int = 4,
    S: int = 512,
    H: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    device: str = "cuda",
):
    gen = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(B, S, H, dtype=dtype, device=device, generator=gen) * 0.5
    # A_log and dt_bias live as fp32 params in practice.
    # A ~ U[0.001, 0.016] -> A_log = log(A)
    A = torch.empty(H, device=device, dtype=torch.float32).uniform_(
        0.001, 0.016, generator=gen,
    )
    A_log = A.log().clone().requires_grad_(True)
    dt_bias = torch.zeros(H, device=device, dtype=torch.float32, requires_grad=True)
    return a.requires_grad_(True), A_log, dt_bias


def test_forward_matches_reference_bf16():
    a, A_log, dt_bias = _make_inputs()

    ref = _pytorch_gdn_decay(a.detach(), A_log.detach(), dt_bias.detach())
    out = fused_gdn_decay(a.detach(), A_log.detach(), dt_bias.detach())

    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_forward_matches_reference_fp32():
    a, A_log, dt_bias = _make_inputs(dtype=torch.float32)

    ref = _pytorch_gdn_decay(a.detach(), A_log.detach(), dt_bias.detach())
    out = fused_gdn_decay(a.detach(), A_log.detach(), dt_bias.detach())

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_backward_matches_reference_bf16():
    a_r, A_log_r, dt_bias_r = _make_inputs(seed=13)
    a_t, A_log_t, dt_bias_t = _make_inputs(seed=13)

    ref = _pytorch_gdn_decay(a_r, A_log_r, dt_bias_r)
    out = fused_gdn_decay(a_t, A_log_t, dt_bias_t)

    # Forward parity first
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    grad = torch.randn_like(ref)
    ref.backward(grad)
    out.backward(grad)

    # bf16 gradient tolerance is loose — the params are summed over B*S tokens
    # which amplifies accumulation noise.
    torch.testing.assert_close(a_t.grad, a_r.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(A_log_t.grad, A_log_r.grad, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(dt_bias_t.grad, dt_bias_r.grad, rtol=5e-2, atol=5e-2)


def test_backward_matches_reference_fp32():
    a_r, A_log_r, dt_bias_r = _make_inputs(seed=77, dtype=torch.float32)
    a_t, A_log_t, dt_bias_t = _make_inputs(seed=77, dtype=torch.float32)

    ref = _pytorch_gdn_decay(a_r, A_log_r, dt_bias_r)
    out = fused_gdn_decay(a_t, A_log_t, dt_bias_t)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    grad = torch.randn_like(ref)
    ref.backward(grad)
    out.backward(grad)

    torch.testing.assert_close(a_t.grad, a_r.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(A_log_t.grad, A_log_r.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dt_bias_t.grad, dt_bias_r.grad, rtol=1e-4, atol=1e-4)


def test_shape_variants():
    """Kernel must handle both 2-D and 3-D inputs (only last dim = H matters)."""
    H = 32
    for shape in [(10, H), (3, 17, H), (2, 4, 8, H)]:
        a = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.5
        A_log = torch.randn(H, device="cuda", dtype=torch.float32) - 3.0
        dt_bias = torch.zeros(H, device="cuda", dtype=torch.float32)

        ref = _pytorch_gdn_decay(a, A_log, dt_bias)
        out = fused_gdn_decay(a, A_log, dt_bias)
        assert out.shape == a.shape
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
