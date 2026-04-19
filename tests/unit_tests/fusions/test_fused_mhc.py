# Copyright (c) 2026 Susono authors.
"""Unit tests for megatron.core.fusions.fused_mhc_width_connection.

Covers:
  - Depth connection: forward & backward bit-compatibility with the
    pure-PyTorch reference (`beta ⊗ x_out + new_residuals` + permute).
  - Width connection forward: allclose with the PyTorch reference
    (small numerical noise from the fused RMSNorm + GEMM path in bf16).

GPU-only (Triton kernels require CUDA). Skipped on CPU.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused_mhc_width_connection requires CUDA",
)


# ---------------------------------------------------------------------------
# Depth connection
# ---------------------------------------------------------------------------


def _pytorch_depth(x_out, new_residuals, beta):
    return (
        beta.unsqueeze(-1) * x_out.unsqueeze(-2) + new_residuals
    ).permute(2, 0, 1, 3).contiguous()


def test_fused_depth_forward_allclose_bf16():
    from megatron.core.fusions.fused_mhc_width_connection import (
        fused_mhc_depth_connection,
    )

    torch.manual_seed(0)
    S, B, n, D = 8, 2, 4, 64
    x_out = torch.randn(S, B, D, dtype=torch.bfloat16, device="cuda")
    new_residuals = torch.randn(
        S, B, n, D, dtype=torch.bfloat16, device="cuda",
    )
    beta = torch.rand(S, B, n, dtype=torch.bfloat16, device="cuda") * 2

    out_triton = fused_mhc_depth_connection(x_out, new_residuals, beta)
    out_ref = _pytorch_depth(x_out, new_residuals, beta)

    assert out_triton.shape == out_ref.shape
    # bf16 ULP-level noise: Triton computes `beta * x + nr` in fp32 then
    # casts down; PyTorch does each op in bf16.  Allow ~1 bf16 ULP (rtol=1e-2).
    assert torch.allclose(
        out_triton.float(), out_ref.float(), rtol=1e-2, atol=1e-2,
    )


def test_fused_depth_backward_allclose_bf16():
    """d_x_out / d_new_residuals / d_beta must match PyTorch autograd path
    (bf16 numerical noise allowed)."""
    from megatron.core.fusions.fused_mhc_width_connection import (
        fused_mhc_depth_connection,
    )

    torch.manual_seed(1)
    S, B, n, D = 8, 2, 4, 64

    def make_inputs():
        x = torch.randn(S, B, D, dtype=torch.bfloat16, device="cuda")
        nr = torch.randn(S, B, n, D, dtype=torch.bfloat16, device="cuda")
        # Build beta range [0, 2] THEN make it a leaf tensor with grad,
        # so that `.grad` is populated by backward.
        b = (torch.rand(S, B, n, dtype=torch.bfloat16, device="cuda") * 2).detach()
        return (
            x.detach().requires_grad_(True),
            nr.detach().requires_grad_(True),
            b.requires_grad_(True),
        )

    # Reference path.
    torch.manual_seed(42)
    x_ref, nr_ref, b_ref = make_inputs()
    out_ref = _pytorch_depth(x_ref, nr_ref, b_ref)
    upstream = torch.randn_like(out_ref)
    out_ref.backward(upstream)

    # Triton path.
    torch.manual_seed(42)
    x_fus, nr_fus, b_fus = make_inputs()
    out_fus = fused_mhc_depth_connection(x_fus, nr_fus, b_fus)
    out_fus.backward(upstream.detach())

    # Compare gradients.
    # d_beta is a sum over D=64 bf16 terms; its noise is ~sqrt(D)*bf16-ULP
    # which on magnitudes ~10 can reach ~0.1.  Reference accumulates in bf16,
    # Triton in fp32 — they converge within rtol≈5%.
    assert torch.allclose(x_ref.grad.float(), x_fus.grad.float(), rtol=1e-2, atol=1e-2)
    assert torch.allclose(nr_ref.grad.float(), nr_fus.grad.float(), rtol=1e-2, atol=1e-2)
    assert torch.allclose(b_ref.grad.float(), b_fus.grad.float(), rtol=5e-2, atol=1e-1)


def test_fused_depth_fp32_fallback():
    """fp32 input should trigger the PyTorch fallback and return correct
    values (no Triton dispatch)."""
    from megatron.core.fusions.fused_mhc_width_connection import (
        fused_mhc_depth_connection,
    )

    torch.manual_seed(2)
    S, B, n, D = 4, 2, 4, 32
    x_out = torch.randn(S, B, D, dtype=torch.float32, device="cuda")
    new_residuals = torch.randn(S, B, n, D, dtype=torch.float32, device="cuda")
    beta = torch.rand(S, B, n, dtype=torch.float32, device="cuda") * 2

    out = fused_mhc_depth_connection(x_out, new_residuals, beta)
    ref = _pytorch_depth(x_out, new_residuals, beta)
    assert torch.allclose(out, ref, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Width connection forward (autotune sanity)
# ---------------------------------------------------------------------------


def test_fused_width_forward_matches_reference_bf16():
    """Triton fused forward matches the PyTorch reference within bf16 noise.

    This also exercises the newly-added @triton.autotune decorators.
    """
    import itertools
    from megatron.core.fusions.fused_mhc_width_connection import (
        _pytorch_width_connection,
        fused_mhc_width_connection,
    )

    torch.manual_seed(3)
    n, S, B, D = 4, 4, 2, 64
    n_perms = math.factorial(n)

    X = torch.randn(n, S, B, D, dtype=torch.bfloat16, device="cuda") * 0.1
    W_alpha = torch.randn(
        n * D, n + n_perms, dtype=torch.bfloat16, device="cuda",
    ) * 0.01
    W_beta = torch.randn(n * D, n, dtype=torch.bfloat16, device="cuda") * 0.01
    # static_alpha / static_beta / gamma are typically bf16 in the live model
    # (nn.Parameter under bf16 autocast); matching here keeps the reference
    # path's matmul dtypes consistent with the fused kernel's expectations.
    static_alpha = torch.randn(n + n_perms, dtype=torch.bfloat16, device="cuda")
    static_beta = torch.ones(n, dtype=torch.bfloat16, device="cuda") * -8.0
    gamma = torch.zeros(n * D, dtype=torch.bfloat16, device="cuda")
    pbs = torch.tensor([1e-2], dtype=torch.float32, device="cuda")
    rs = torch.tensor([1e-2], dtype=torch.float32, device="cuda")
    hps = torch.tensor([1e-2], dtype=torch.float32, device="cuda")

    # Build permutation matrices (same as mhc._get_permutation_matrices).
    perms = list(itertools.permutations(range(n)))
    eye = torch.eye(n, dtype=torch.float32, device="cuda")
    idx = torch.tensor(perms, dtype=torch.long, device="cuda")
    perms_tensor = eye[idx].to(torch.bfloat16)  # [n!, n, n]

    bi, nr, beta = fused_mhc_width_connection(
        X, W_alpha, W_beta, static_alpha, static_beta, gamma,
        perms_tensor, pbs, rs, hps,
    )

    # Reference on [T, n, D] layout.
    T = S * B
    X_flat = X.permute(1, 2, 0, 3).reshape(T, n, D).contiguous()
    bi_r, nr_r, beta_r = _pytorch_width_connection(
        X_flat, W_alpha, W_beta, static_alpha, static_beta, gamma,
        perms_tensor, pbs, rs, hps,
        T, n, D, n_perms,
    )
    bi_r = bi_r.view(S, B, D)
    nr_r = nr_r.view(S, B, n, D)
    beta_r = beta_r.view(S, B, n)

    assert torch.allclose(bi.float(), bi_r.float(), rtol=5e-2, atol=5e-2)
    assert torch.allclose(nr.float(), nr_r.float(), rtol=5e-2, atol=5e-2)
    assert torch.allclose(beta.float(), beta_r.float(), rtol=5e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# Width connection backward (Triton vs PyTorch-recompute reference)
# ---------------------------------------------------------------------------


def _make_width_inputs(n=4, S=4, B=2, D=64, seed=7):
    """Build a fresh set of MHC width-connection parameters + input."""
    import itertools
    torch.manual_seed(seed)
    n_perms = math.factorial(n)

    X = (torch.randn(n, S, B, D, dtype=torch.bfloat16, device="cuda") * 0.1)
    W_alpha = (torch.randn(
        n * D, n + n_perms, dtype=torch.bfloat16, device="cuda",
    ) * 0.01)
    W_beta = torch.randn(n * D, n, dtype=torch.bfloat16, device="cuda") * 0.01
    static_alpha = torch.randn(n + n_perms, dtype=torch.bfloat16, device="cuda")
    static_beta = torch.ones(n, dtype=torch.bfloat16, device="cuda") * -8.0
    gamma = torch.zeros(n * D, dtype=torch.bfloat16, device="cuda")
    pbs = torch.tensor([1e-2], dtype=torch.float32, device="cuda")
    rs = torch.tensor([1e-2], dtype=torch.float32, device="cuda")
    hps = torch.tensor([1e-2], dtype=torch.float32, device="cuda")

    perms = list(itertools.permutations(range(n)))
    eye = torch.eye(n, dtype=torch.float32, device="cuda")
    idx = torch.tensor(perms, dtype=torch.long, device="cuda")
    perms_tensor = eye[idx].to(torch.bfloat16)

    return {
        "X": X, "W_alpha": W_alpha, "W_beta": W_beta,
        "static_alpha": static_alpha, "static_beta": static_beta,
        "gamma": gamma, "pbs": pbs, "rs": rs, "hps": hps,
        "perms": perms_tensor, "n": n, "S": S, "B": B, "D": D,
    }


def _run_width_bwd(backend: str, inputs: dict):
    """Run forward+backward with the specified backend and return (grads, outputs)."""
    import os
    from megatron.core.fusions import fused_mhc_width_connection as mod

    # Configure backend BEFORE importing the module is too late — it reads env
    # at import time.  Override the module-level flag directly.
    prev_mode = mod._WIDTH_BWD_MODE
    mod._WIDTH_BWD_MODE = backend
    try:
        # Fresh leaf tensors with grad.
        X = inputs["X"].detach().clone().requires_grad_(True)
        W_alpha = inputs["W_alpha"].detach().clone().requires_grad_(True)
        W_beta = inputs["W_beta"].detach().clone().requires_grad_(True)
        static_alpha = inputs["static_alpha"].detach().clone().requires_grad_(True)
        static_beta = inputs["static_beta"].detach().clone().requires_grad_(True)
        gamma = inputs["gamma"].detach().clone().requires_grad_(True)
        pbs = inputs["pbs"].detach().clone().requires_grad_(True)
        rs = inputs["rs"].detach().clone().requires_grad_(True)
        hps = inputs["hps"].detach().clone().requires_grad_(True)

        bi, nr, beta = mod.fused_mhc_width_connection(
            X, W_alpha, W_beta, static_alpha, static_beta, gamma,
            inputs["perms"], pbs, rs, hps,
        )
        # Deterministic upstream: sum of each output (matches gradients=1 of unit).
        torch.manual_seed(101)
        up_bi = torch.randn_like(bi)
        up_nr = torch.randn_like(nr)
        up_beta = torch.randn_like(beta)
        loss = (bi * up_bi).sum() + (nr * up_nr).sum() + (beta * up_beta).sum()
        loss.backward()

        return {
            "X_grad": X.grad, "Wa_grad": W_alpha.grad, "Wb_grad": W_beta.grad,
            "sa_grad": static_alpha.grad, "sb_grad": static_beta.grad,
            "g_grad": gamma.grad,
            "pbs_grad": pbs.grad, "rs_grad": rs.grad, "hps_grad": hps.grad,
            "bi": bi.detach(), "nr": nr.detach(), "beta": beta.detach(),
        }
    finally:
        mod._WIDTH_BWD_MODE = prev_mode


def _grad_close(a, b, rtol, atol, name):
    diff = (a.float() - b.float()).abs()
    ref_scale = b.float().abs().max().item() + 1e-9
    assert torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol), (
        f"{name}: max diff {diff.max().item():.3e}, max ref {ref_scale:.3e}, "
        f"rtol={rtol}, atol={atol}"
    )


def test_fused_width_backward_matches_pytorch_bf16():
    """Triton width backward must match the PyTorch-recompute reference
    within bf16 numerical noise."""
    inputs = _make_width_inputs()

    out_py = _run_width_bwd("pytorch", inputs)
    out_triton = _run_width_bwd("triton", inputs)

    # Forward outputs are unrelated to backend selection but worth double-checking.
    _grad_close(out_py["bi"], out_triton["bi"], rtol=5e-2, atol=5e-2, name="bi")
    _grad_close(out_py["nr"], out_triton["nr"], rtol=5e-2, atol=5e-2, name="nr")
    _grad_close(out_py["beta"], out_triton["beta"], rtol=5e-2, atol=5e-2, name="beta")

    # Backward tolerances: bf16 accumulation noise can be sizeable, especially
    # for softmax backward of 24 perms.
    _grad_close(out_py["X_grad"], out_triton["X_grad"], rtol=5e-2, atol=5e-2, name="X_grad")
    _grad_close(out_py["Wa_grad"], out_triton["Wa_grad"], rtol=5e-2, atol=5e-2, name="Wa_grad")
    _grad_close(out_py["Wb_grad"], out_triton["Wb_grad"], rtol=5e-2, atol=5e-2, name="Wb_grad")
    _grad_close(out_py["sa_grad"], out_triton["sa_grad"], rtol=1e-1, atol=1e-1, name="sa_grad")
    _grad_close(out_py["sb_grad"], out_triton["sb_grad"], rtol=5e-2, atol=5e-2, name="sb_grad")
    _grad_close(out_py["g_grad"], out_triton["g_grad"], rtol=5e-2, atol=5e-2, name="g_grad")
    _grad_close(out_py["pbs_grad"], out_triton["pbs_grad"], rtol=1e-1, atol=1e-1, name="pbs_grad")
    _grad_close(out_py["rs_grad"], out_triton["rs_grad"], rtol=1e-1, atol=1e-1, name="rs_grad")
    _grad_close(out_py["hps_grad"], out_triton["hps_grad"], rtol=1e-1, atol=1e-1, name="hps_grad")
