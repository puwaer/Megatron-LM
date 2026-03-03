# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused Triton kernel for MHC-Lite width connection.

Fuses the following sequence from ``_width_connection`` into two kernels
so that intermediate tensors (normed, H_res, res_coeff) are kept in SRAM
and never written to HBM:

    normed        = RMSNorm(concat(X_l streams))               [T, n*D]
    wc            = normed @ W_alpha                            [T, n+n!]
    alpha_pre     = sigmoid(scale_pre * wc[:n]  + b_pre)       [T, n]
    res_coeff     = softmax(scale_res * wc[n:]  + b_res)       [T, n!]
    H_res         = Σ_r res_coeff[r] * P_r                     [T, n, n]
    new_residuals = H_res @ X_sb                               [T, n, D]
    branch_input  = Σ_s alpha_pre[s] * X_sb[:, s, :]           [T, D]
    dc            = normed @ W_beta                            [T, n]
    beta          = sigmoid(scale_post * dc + b_beta) * 2      [T, n]

Kernel design (two passes):
    Pass 1 — ``_fused_mhc_wc_proj_kernel``:
        Reads X [T, n, D]; fuses RMSNorm with W_alpha / W_beta projection.
        Writes small scratch buffers: wc [T, n+n!], dc [T, n].
        normed is NEVER written to HBM.

    Pass 2 — ``_fused_mhc_output_kernel``:
        Reads X [T, n, D], wc [T, 28], dc [T, 4].
        Computes alpha_pre, res_coeff, H_res entirely in registers.
        Writes new_residuals [T, n, D], branch_input [T, D], beta [T, n].
        H_res and res_coeff are NEVER written to HBM.

HBM I/O comparison (n=4, D=2048, n!=24, T=S*B tokens):
    Standard PyTorch: reads+writes normed (~8192·T), H_res (~16·T),
                      res_coeff (~24·T) in addition to inputs/outputs
    Triton fused:     only additional scratch: wc (~28·T), dc (~4·T)
                      — all other intermediates stay in SRAM/registers

Usage::

    branch_input, new_residuals, beta = fused_mhc_width_connection(
        X, W_alpha, W_beta, static_alpha, static_beta,
        gamma, perms, pre_branch_scale, residual_scale, h_post_scale,
    )

Requirements:
    - triton >= 2.2.0
    - n == 4  (kernel unrolls stream loops at compile time)
    - bfloat16 inputs on GPU
"""

import math
from typing import Tuple

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure-PyTorch reference implementation (fallback + backward)
# ---------------------------------------------------------------------------

def _pytorch_width_connection(
    X_flat: Tensor,       # [T, n, D]
    W_alpha: Tensor,      # [n*D, n+n_perms]
    W_beta: Tensor,       # [n*D, n]
    static_alpha: Tensor, # [n+n_perms]
    static_beta: Tensor,  # [n]
    gamma: Tensor,        # [n*D]
    perms: Tensor,        # [n_perms, n, n]
    pre_branch_scale: Tensor,
    residual_scale: Tensor,
    h_post_scale: Tensor,
    T: int,
    n: int,
    D: int,
    n_perms: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pure-PyTorch width connection on flattened tensors.

    Equivalent to ``ManifoldConstrainedHyperConnection._width_connection`` but
    operates on [T, n, D] layout instead of [n, S, B, D].  Used as the
    backward-pass reference and as a CPU/fallback path.
    """
    import torch.nn.functional as F

    X_sb = X_flat                                     # [T, n, D]

    # RMSNorm: x / rms(x) where rms = sqrt(mean(x^2))
    normed = X_sb.reshape(T, n * D)
    normed = F.normalize(normed, dim=-1) * math.sqrt(n * D) * (gamma + 1)

    # Width projection
    wc = normed @ W_alpha                              # [T, n+n_perms]
    alpha_pre = torch.sigmoid(
        pre_branch_scale * wc[..., :n] + static_alpha[:n]
    )                                                  # [T, n]
    res_coeff = torch.softmax(
        residual_scale * wc[..., n:] + static_alpha[n:], dim=-1
    )                                                  # [T, n_perms]

    H_res = torch.einsum('...r,rij->...ij', res_coeff, perms)   # [T, n, n]
    new_residuals = torch.einsum('...ij,...jd->...id', H_res, X_sb)  # [T, n, D]
    branch_input = (alpha_pre.unsqueeze(-1) * X_sb).sum(dim=-2)      # [T, D]

    # Depth projection
    dc = normed @ W_beta                               # [T, n]
    beta = torch.sigmoid(h_post_scale * dc + static_beta) * 2        # [T, n]

    return branch_input, new_residuals, beta


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_mhc_wc_proj_kernel(
        # Inputs
        X_ptr,            # [T, n*D]  (already permuted to T-first layout)
        W_alpha_ptr,      # [n*D, n_alpha]
        W_beta_ptr,       # [n*D, n_beta]
        gamma_ptr,        # [n*D]
        # Outputs
        wc_ptr,           # [T, n_alpha]
        dc_ptr,           # [T, n_beta]
        # Dimensions (all constexpr for register-array unrolling)
        T,
        nD: tl.constexpr,       # n * D
        n_alpha: tl.constexpr,  # n + n_perms
        n_beta: tl.constexpr,   # n
        BLOCK_ND: tl.constexpr,
    ):
        """Pass 1: fused RMSNorm + linear projection for wc and dc.

        Reads X once; normed is computed on-the-fly and never written to HBM.
        """
        token_id = tl.program_id(0)
        if token_id >= T:
            return

        x_base = token_id * nD

        # ---- Step 1: compute rms_scale (sum-of-squares pass) ----
        # rms_scale = rsqrt(mean(x^2) + eps) so that x * rms_scale = x / rms(x)
        # This gives F.normalize(x) * sqrt(nD), matching _RMSNorm.forward.
        sumsq = tl.zeros([1], dtype=tl.float32)
        for _blk in tl.static_range(tl.cdiv(nD, BLOCK_ND)):
            offs = _blk * BLOCK_ND + tl.arange(0, BLOCK_ND)
            mask = offs < nD
            x = tl.load(X_ptr + x_base + offs, mask=mask, other=0.0).to(tl.float32)
            sumsq += tl.sum(x * x)
        # rms_scale = 1 / sqrt(mean(x^2) + eps) = rsqrt(sum(x^2)/nD + eps)
        rms_scale = tl.rsqrt(sumsq / nD + 1e-12)

        # ---- Step 2: compute wc = normed @ W_alpha (output-stationary) ----
        # Load normed once per block, accumulate all n_alpha output columns.
        # acc_wc[col] accumulates the dot product for output column `col`.
        # Since n_alpha is constexpr, the loop is unrolled at compile time.
        acc_wc = tl.zeros([n_alpha], dtype=tl.float32)
        acc_dc = tl.zeros([n_beta], dtype=tl.float32)

        for _blk in tl.static_range(tl.cdiv(nD, BLOCK_ND)):
            offs = _blk * BLOCK_ND + tl.arange(0, BLOCK_ND)
            mask = offs < nD

            # Compute normed_chunk = x * rms_scale * (gamma + 1)
            x_chunk = tl.load(X_ptr + x_base + offs, mask=mask, other=0.0).to(tl.float32)
            g_chunk = tl.load(gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            normed_chunk = x_chunk * rms_scale * (g_chunk + 1.0)

            # wc accumulation: loop over n_alpha output columns (compile-time unroll)
            for col in tl.static_range(n_alpha):
                # W_alpha is row-major [nD, n_alpha]: W_alpha[row, col] at row*n_alpha+col
                w_col = tl.load(
                    W_alpha_ptr + offs * n_alpha + col,
                    mask=mask, other=0.0
                ).to(tl.float32)
                acc_wc[col] += tl.sum(normed_chunk * w_col)

            # dc accumulation: loop over n_beta output columns
            for col in tl.static_range(n_beta):
                w_col = tl.load(
                    W_beta_ptr + offs * n_beta + col,
                    mask=mask, other=0.0
                ).to(tl.float32)
                acc_dc[col] += tl.sum(normed_chunk * w_col)

        # Write wc and dc to scratch buffers (much smaller than normed)
        wc_base = token_id * n_alpha
        for col in tl.static_range(n_alpha):
            tl.store(wc_ptr + wc_base + col, acc_wc[col])

        dc_base = token_id * n_beta
        for col in tl.static_range(n_beta):
            tl.store(dc_ptr + dc_base + col, acc_dc[col])

    @triton.jit
    def _fused_mhc_output_kernel(
        # Inputs
        X_ptr,             # [T, n, D]  (T-first layout, n=4 streams each D elements)
        wc_ptr,            # [T, n_alpha]
        dc_ptr,            # [T, n_beta]
        static_alpha_ptr,  # [n_alpha]
        static_beta_ptr,   # [n_beta]
        perms_ptr,         # [n_perms * n * n]  (flattened, bfloat16 or float32)
        pre_branch_scale,  # scalar (float)
        residual_scale,    # scalar (float)
        h_post_scale,      # scalar (float)
        # Outputs
        branch_input_ptr,   # [T, D]
        new_residuals_ptr,  # [T, n, D]
        beta_ptr,           # [T, n]
        # Dimensions
        T,
        n: tl.constexpr,        # number of streams (must be 4)
        D: tl.constexpr,
        n_alpha: tl.constexpr,  # n + n_perms
        n_perms: tl.constexpr,  # n! = 24 for n=4
        n_beta: tl.constexpr,   # n
        BLOCK_D: tl.constexpr,
    ):
        """Pass 2: compute alpha_pre, H_res (in registers), new_residuals, branch_input, beta."""
        token_id = tl.program_id(0)
        if token_id >= T:
            return

        # ---- Load wc and dc from scratch (small: 28 + 4 elements) ----
        wc = tl.load(wc_ptr + token_id * n_alpha + tl.arange(0, n_alpha)).to(tl.float32)
        dc = tl.load(dc_ptr + token_id * n_beta + tl.arange(0, n_beta)).to(tl.float32)

        # ---- alpha_pre [n] = sigmoid(scale * wc[:n] + static_alpha[:n]) ----
        sa_pre = tl.load(static_alpha_ptr + tl.arange(0, n)).to(tl.float32)
        alpha_pre = tl.sigmoid(pre_branch_scale * wc[:n] + sa_pre)    # [n]

        # ---- res_coeff [n_perms] = softmax(scale * wc[n:] + static_alpha[n:]) ----
        sa_res = tl.load(static_alpha_ptr + n + tl.arange(0, n_perms)).to(tl.float32)
        logits = residual_scale * wc[n:] + sa_res                      # [n_perms]
        logits = logits - tl.max(logits, axis=0)                       # numeric stability
        exp_l = tl.exp(logits)
        res_coeff = exp_l / tl.sum(exp_l)                              # [n_perms]

        # ---- H_res [n*n] = Σ_r res_coeff[r] * P_r  (kept in registers) ----
        H_res = tl.zeros([n * n], dtype=tl.float32)
        for r in tl.static_range(n_perms):
            perm_r = tl.load(perms_ptr + r * n * n + tl.arange(0, n * n)).to(tl.float32)
            H_res = H_res + res_coeff[r] * perm_r

        # ---- beta [n] = sigmoid(scale * dc + static_beta) * 2 ----
        sb = tl.load(static_beta_ptr + tl.arange(0, n_beta)).to(tl.float32)
        beta_vec = tl.sigmoid(h_post_scale * dc + sb) * 2.0           # [n]
        tl.store(beta_ptr + token_id * n + tl.arange(0, n_beta),
                 beta_vec.to(tl.bfloat16))

        # ---- Process D in blocks: compute new_residuals and branch_input ----
        # X layout: [T, n, D] with n=4; X[token, s, d] at token*n*D + s*D + d
        x_base = token_id * n * D

        for d_blk in tl.static_range(tl.cdiv(D, BLOCK_D)):
            d_offs = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D

            # Load the four streams for this D block (n=4, hard-coded for unrolling)
            x0 = tl.load(X_ptr + x_base + 0 * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            x1 = tl.load(X_ptr + x_base + 1 * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            x2 = tl.load(X_ptr + x_base + 2 * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            x3 = tl.load(X_ptr + x_base + 3 * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)

            # branch_input[d] = Σ_s alpha_pre[s] * x_s[d]
            bi = alpha_pre[0] * x0 + alpha_pre[1] * x1 + alpha_pre[2] * x2 + alpha_pre[3] * x3
            tl.store(branch_input_ptr + token_id * D + d_offs,
                     bi.to(tl.bfloat16), mask=d_mask)

            # new_residuals[i, d] = Σ_j H_res[i*n+j] * x_j[d]
            x_streams = (x0, x1, x2, x3)
            for i in tl.static_range(n):
                nr_i = (H_res[i * n + 0] * x_streams[0]
                        + H_res[i * n + 1] * x_streams[1]
                        + H_res[i * n + 2] * x_streams[2]
                        + H_res[i * n + 3] * x_streams[3])
                tl.store(new_residuals_ptr + token_id * n * D + i * D + d_offs,
                         nr_i.to(tl.bfloat16), mask=d_mask)


# ---------------------------------------------------------------------------
# Autograd Function wrapping the two-pass Triton kernels
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    class _FusedMHCWidthFunction(torch.autograd.Function):
        """Autograd wrapper: Triton forward, PyTorch backward (via recompute)."""

        @staticmethod
        def forward(
            ctx,
            X_flat,           # [T, n, D]  contiguous bfloat16
            W_alpha,          # [n*D, n+n_perms]
            W_beta,           # [n*D, n]
            static_alpha,     # [n+n_perms]
            static_beta,      # [n]
            gamma,            # [n*D]
            perms_flat,       # [n_perms, n, n]  contiguous
            pre_branch_scale, # scalar tensor
            residual_scale,   # scalar tensor
            h_post_scale,     # scalar tensor
            T: int,
            n: int,
            D: int,
            n_perms: int,
        ):
            device = X_flat.device
            nD = n * D
            n_alpha = n + n_perms
            n_beta = n
            BLOCK_ND = min(128, nD)
            BLOCK_D = min(128, D)

            # Scratch buffers: wc [T, n_alpha], dc [T, n]  (float32 for precision)
            wc_scratch = torch.empty((T, n_alpha), dtype=torch.float32, device=device)
            dc_scratch = torch.empty((T, n_beta), dtype=torch.float32, device=device)

            # Output tensors
            branch_input = torch.empty((T, D), dtype=X_flat.dtype, device=device)
            new_residuals = torch.empty((T, n, D), dtype=X_flat.dtype, device=device)
            beta_out = torch.empty((T, n), dtype=X_flat.dtype, device=device)

            # X needs to be viewed as [T, nD] for Pass 1 (RMSNorm uses flattened)
            X_flat_nd = X_flat.reshape(T, nD)

            # Pass 1: fused RMSNorm + projection → wc, dc scratch buffers
            _fused_mhc_wc_proj_kernel[(T,)](
                X_flat_nd, W_alpha, W_beta, gamma,
                wc_scratch, dc_scratch,
                T=T, nD=nD, n_alpha=n_alpha, n_beta=n_beta,
                BLOCK_ND=BLOCK_ND,
            )

            # Pass 2: compute alpha_pre, H_res (in registers), outputs
            _fused_mhc_output_kernel[(T,)](
                X_flat,          # [T, n, D] for stream-aware access
                wc_scratch, dc_scratch,
                static_alpha, static_beta,
                perms_flat.reshape(-1),   # flatten to 1D for simple pointer math
                pre_branch_scale.item(),
                residual_scale.item(),
                h_post_scale.item(),
                branch_input, new_residuals, beta_out,
                T=T, n=n, D=D, n_alpha=n_alpha, n_perms=n_perms, n_beta=n_beta,
                BLOCK_D=BLOCK_D,
            )

            ctx.save_for_backward(
                X_flat, W_alpha, W_beta,
                static_alpha, static_beta, gamma,
                perms_flat,
                pre_branch_scale, residual_scale, h_post_scale,
            )
            ctx.T, ctx.n, ctx.D, ctx.n_perms = T, n, D, n_perms
            return branch_input, new_residuals, beta_out

        @staticmethod
        def backward(ctx, grad_branch, grad_res, grad_beta):
            """Backward: recompute forward in PyTorch, then differentiate."""
            (X_flat, W_alpha, W_beta,
             static_alpha, static_beta, gamma,
             perms_flat,
             pbs, rs, hps) = ctx.saved_tensors
            T, n, D, n_perms = ctx.T, ctx.n, ctx.D, ctx.n_perms

            def attach(t):
                return t.detach().requires_grad_(t.requires_grad)

            X_d, Wa_d, Wb_d = attach(X_flat), attach(W_alpha), attach(W_beta)
            sa_d, sb_d, g_d = attach(static_alpha), attach(static_beta), attach(gamma)
            pbs_d, rs_d, hps_d = attach(pbs), attach(rs), attach(hps)

            with torch.enable_grad():
                bi_r, nr_r, beta_r = _pytorch_width_connection(
                    X_d, Wa_d, Wb_d, sa_d, sb_d, g_d,
                    perms_flat, pbs_d, rs_d, hps_d,
                    T, n, D, n_perms,
                )
                torch.autograd.backward(
                    (bi_r, nr_r, beta_r),
                    (grad_branch, grad_res, grad_beta),
                )

            def g(t):
                return t.grad if t.requires_grad else None

            return (
                g(X_d), g(Wa_d), g(Wb_d),
                g(sa_d), g(sb_d), g(g_d),
                None,                   # perms_flat — constant
                g(pbs_d), g(rs_d), g(hps_d),
                None, None, None, None, # T, n, D, n_perms
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_mhc_width_connection(
    X: Tensor,
    W_alpha: Tensor,
    W_beta: Tensor,
    static_alpha: Tensor,
    static_beta: Tensor,
    gamma: Tensor,
    perms: Tensor,
    pre_branch_scale: Tensor,
    residual_scale: Tensor,
    h_post_scale: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fused mHC width connection forward pass.

    Uses two-pass Triton kernels when available (GPU + bfloat16 + n==4);
    falls back to pure PyTorch for other configurations (CPU, fp32, n≠4).

    Args:
        X:                Multi-stream hidden states [n, S, B, D].
        W_alpha:          Dynamic alpha projection  [n*D, n+n!].
        W_beta:           Dynamic beta projection   [n*D, n].
        static_alpha:     Static alpha biases       [n+n!].
        static_beta:      Static beta biases        [n].
        gamma:            RMSNorm learnable scale   [n*D]  (the ``gamma``
                          *parameter*, not ``gamma+1`` — the +1 offset is
                          applied inside this function, matching _RMSNorm).
        perms:            Permutation matrices      [n!, n, n].
        pre_branch_scale: Scalar parameter for H_pre gate.
        residual_scale:   Scalar parameter for H_res gate.
        h_post_scale:     Scalar parameter for H_post (beta) gate.

    Returns:
        branch_input:   [S, B, D]    — gated aggregation over streams.
        new_residuals:  [S, B, n, D] — streams after permutation mixing.
        beta:           [S, B, n]    — depth-connection gate weights.
    """
    n, S, B, D = X.shape
    n_perms = perms.shape[0]
    T = S * B

    # Rearrange to [T, n, D] (contiguous for kernel access pattern)
    X_flat = X.permute(1, 2, 0, 3).reshape(T, n, D).contiguous()
    perms_flat = perms.contiguous()

    use_triton = (
        _TRITON_AVAILABLE
        and X.is_cuda
        and X.dtype == torch.bfloat16
        and n == 4   # kernel hard-codes n=4 stream unrolling
    )

    if use_triton:
        branch_flat, res_flat, beta_flat = _FusedMHCWidthFunction.apply(
            X_flat, W_alpha, W_beta,
            static_alpha, static_beta, gamma,
            perms_flat,
            pre_branch_scale, residual_scale, h_post_scale,
            T, n, D, n_perms,
        )
    else:
        branch_flat, res_flat, beta_flat = _pytorch_width_connection(
            X_flat, W_alpha, W_beta,
            static_alpha, static_beta, gamma,
            perms_flat,
            pre_branch_scale, residual_scale, h_post_scale,
            T, n, D, n_perms,
        )

    # Reshape to [S, B, ...] layout expected by the caller
    branch_input = branch_flat.view(S, B, D)
    new_residuals = res_flat.view(S, B, n, D)
    beta = beta_flat.view(S, B, n)

    return branch_input, new_residuals, beta
