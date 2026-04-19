# Copyright (c) 2026 Susono authors.
"""Unit tests for megatron.core.fusions.fused_sigmoid_mul.

Validates forward and backward parity against the reference

    out = torch.sigmoid(gate_input) * value

for both the same-shape case (Engram) and the row-scalar broadcast case
(MoE shared-expert gate with gate [T, 1] and value [T, D]).
"""

import pytest
import torch

from megatron.core.fusions.fused_sigmoid_mul import (
    _pytorch_sigmoid_mul,
    fused_sigmoid_mul,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused_sigmoid_mul requires CUDA",
)


# ---------------------------------------------------------------------------
# Same-shape (Engram) path
# ---------------------------------------------------------------------------

def test_forward_same_shape_bf16():
    torch.manual_seed(0)
    x = torch.randn(8, 128, 672, device="cuda", dtype=torch.bfloat16) * 0.5
    y = torch.randn(8, 128, 672, device="cuda", dtype=torch.bfloat16) * 0.5

    ref = _pytorch_sigmoid_mul(x, y)
    out = fused_sigmoid_mul(x, y)
    # bf16 1-ULP is ~0.0078; PyTorch sigmoid vs our fp32-internal sigmoid
    # can disagree by a single ULP on some elements.
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_backward_same_shape_bf16():
    torch.manual_seed(1)
    x_data = (torch.randn(4, 64, 128, device="cuda", dtype=torch.bfloat16)
              * 0.5).detach()
    y_data = (torch.randn(4, 64, 128, device="cuda", dtype=torch.bfloat16)
              * 0.5).detach()
    x1 = x_data.clone().requires_grad_(True)
    y1 = y_data.clone().requires_grad_(True)
    x2 = x_data.clone().requires_grad_(True)
    y2 = y_data.clone().requires_grad_(True)

    ref = _pytorch_sigmoid_mul(x1, y1)
    out = fused_sigmoid_mul(x2, y2)
    # bf16 1-ULP is ~0.0078; PyTorch sigmoid vs our fp32-internal sigmoid
    # can disagree by a single ULP on some elements.
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    g = torch.randn_like(ref)
    ref.backward(g)
    out.backward(g)

    # 1-ULP bf16 tolerance (~0.0156)
    torch.testing.assert_close(x2.grad, x1.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(y2.grad, y1.grad, rtol=2e-2, atol=2e-2)


def test_forward_same_shape_fp32():
    torch.manual_seed(2)
    x = torch.randn(4, 32, 64, device="cuda", dtype=torch.float32)
    y = torch.randn(4, 32, 64, device="cuda", dtype=torch.float32)

    ref = _pytorch_sigmoid_mul(x, y)
    out = fused_sigmoid_mul(x, y)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Row-gate broadcast (MoE) path
# ---------------------------------------------------------------------------

def test_forward_row_gate_bf16():
    torch.manual_seed(3)
    # [T, 1] gate × [T, D] value
    T, D = 256, 2048
    gate = torch.randn(T, 1, device="cuda", dtype=torch.bfloat16) * 0.5
    value = torch.randn(T, D, device="cuda", dtype=torch.bfloat16) * 0.5

    ref = _pytorch_sigmoid_mul(gate, value)
    out = fused_sigmoid_mul(gate, value)
    assert out.shape == value.shape
    # bf16 1-ULP is ~0.0078; PyTorch sigmoid vs our fp32-internal sigmoid
    # can disagree by a single ULP on some elements.
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_backward_row_gate_bf16():
    torch.manual_seed(4)
    T, D = 128, 512
    gate_data = (torch.randn(T, 1, device="cuda", dtype=torch.bfloat16)
                 * 0.5).detach()
    value_data = (torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
                  * 0.5).detach()
    gate_r = gate_data.clone().requires_grad_(True)
    value_r = value_data.clone().requires_grad_(True)
    gate_t = gate_data.clone().requires_grad_(True)
    value_t = value_data.clone().requires_grad_(True)

    ref = _pytorch_sigmoid_mul(gate_r, value_r)
    out = fused_sigmoid_mul(gate_t, value_t)
    # bf16 1-ULP is ~0.0078; PyTorch sigmoid vs our fp32-internal sigmoid
    # can disagree by a single ULP on some elements.
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    g = torch.randn_like(ref)
    ref.backward(g)
    out.backward(g)

    # Gate is [T, 1]; its gradient is a reduction over D and accumulates
    # bf16 noise, so loosen the tolerance for the gate.
    torch.testing.assert_close(gate_t.grad, gate_r.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(value_t.grad, value_r.grad, rtol=2e-2, atol=2e-2)


def test_row_gate_various_leading_dims():
    """3-D/4-D gate shapes should also dispatch to the row-gate kernel."""
    torch.manual_seed(5)
    D = 128
    for gshape, vshape in [
        ((5, 7, 1), (5, 7, D)),
        ((2, 3, 4, 1), (2, 3, 4, D)),
    ]:
        gate = torch.randn(*gshape, device="cuda", dtype=torch.bfloat16) * 0.5
        value = torch.randn(*vshape, device="cuda", dtype=torch.bfloat16) * 0.5

        ref = _pytorch_sigmoid_mul(gate, value)
        out = fused_sigmoid_mul(gate, value)
        assert out.shape == value.shape
        # bf16 1-ULP is ~0.0078; PyTorch sigmoid vs our fp32-internal sigmoid
    # can disagree by a single ULP on some elements.
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
