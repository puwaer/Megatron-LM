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


def _next_pow2(n: int) -> int:
    """Return the smallest power of 2 that is >= n."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


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
    ).to(X_sb.dtype)                                   # [T, n]
    
    res_coeff = torch.softmax(
        residual_scale * wc[..., n:] + static_alpha[n:], dim=-1
    ).to(X_sb.dtype)                                   # [T, n_perms]

    H_res = torch.einsum('...r,rij->...ij', res_coeff, perms.to(X_sb.dtype))   # [T, n, n]
    new_residuals = torch.einsum('...ij,...jd->...id', H_res, X_sb)  # [T, n, D]
    branch_input = (alpha_pre.unsqueeze(-1) * X_sb).sum(dim=-2)      # [T, D]

    # Depth projection
    dc = normed @ W_beta                               # [T, n]
    beta = (torch.sigmoid(h_post_scale * dc + static_beta) * 2.0).to(X_sb.dtype)        # [T, n]

    return branch_input, new_residuals, beta


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    _MHC_PROJ_AUTOTUNE_CONFIGS = [
        triton.Config({'BLOCK_ND': bn}, num_warps=nw, num_stages=ns)
        for bn in (64, 128, 256)
        for nw in (4, 8)
        for ns in (2, 3)
    ]

    _MHC_OUTPUT_AUTOTUNE_CONFIGS = [
        triton.Config({'BLOCK_D': bd}, num_warps=nw, num_stages=ns)
        for bd in (64, 128, 256)
        for nw in (4, 8)
        for ns in (2, 3)
    ]

    _MHC_DEPTH_AUTOTUNE_CONFIGS = [
        triton.Config({'BLOCK_D': bd}, num_warps=nw, num_stages=ns)
        for bd in (64, 128, 256, 512)
        for nw in (2, 4, 8)
        for ns in (2, 3)
    ]

    @triton.autotune(
        configs=_MHC_PROJ_AUTOTUNE_CONFIGS,
        key=['nD', 'n_alpha', 'N_ALPHA_PAD'],
    )
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
        nD: tl.constexpr,           # n * D
        n_alpha: tl.constexpr,      # n + n_perms  (actual count, e.g. 28)
        n_beta: tl.constexpr,       # n
        N_ALPHA_PAD: tl.constexpr,  # next pow2 >= n + N_PERMS_PAD  (e.g. 64)
        BLOCK_ND: tl.constexpr,     # autotuned
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
        for _blk in range(tl.cdiv(nD, BLOCK_ND)):
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
        acc_wc = tl.zeros([N_ALPHA_PAD], dtype=tl.float32)
        acc_dc = tl.zeros([n_beta], dtype=tl.float32)

        for _blk in range(tl.cdiv(nD, BLOCK_ND)):
            offs = _blk * BLOCK_ND + tl.arange(0, BLOCK_ND)
            mask = offs < nD

            # Compute normed_chunk = x * rms_scale * (gamma + 1)
            x_chunk = tl.load(X_ptr + x_base + offs, mask=mask, other=0.0).to(tl.float32)
            g_chunk = tl.load(gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            normed_chunk = x_chunk * rms_scale * (g_chunk + 1.0)

            # wc accumulation: 2D load [BLOCK_ND, N_ALPHA_PAD] → tl.sum(axis=0)
            cols_wc = tl.arange(0, N_ALPHA_PAD)
            w_ptrs = W_alpha_ptr + offs[:, None] * n_alpha + cols_wc[None, :]
            w_mask = mask[:, None] & (cols_wc[None, :] < n_alpha)
            w_chunk = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
            acc_wc += tl.sum(normed_chunk[:, None] * w_chunk, axis=0)

            # dc accumulation: 2D load [BLOCK_ND, n_beta] (n_beta=4, power-of-2)
            cols_dc = tl.arange(0, n_beta)
            w_ptrs_dc = W_beta_ptr + offs[:, None] * n_beta + cols_dc[None, :]
            w_chunk_dc = tl.load(w_ptrs_dc, mask=mask[:, None], other=0.0).to(tl.float32)
            acc_dc += tl.sum(normed_chunk[:, None] * w_chunk_dc, axis=0)

        # Vectorized stores (no scalar indexing)
        wc_base = token_id * N_ALPHA_PAD
        tl.store(
            wc_ptr + wc_base + tl.arange(0, N_ALPHA_PAD),
            acc_wc,
            mask=tl.arange(0, N_ALPHA_PAD) < n_alpha,
        )
        dc_base = token_id * n_beta
        tl.store(dc_ptr + dc_base + tl.arange(0, n_beta), acc_dc)

    @triton.autotune(
        configs=_MHC_OUTPUT_AUTOTUNE_CONFIGS,
        key=['D', 'n', 'n_perms', 'N_ALPHA_PAD', 'N_PERMS_PAD'],
    )
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
        n: tl.constexpr,            # number of streams (must be 4)
        D: tl.constexpr,
        n_alpha: tl.constexpr,      # n + n_perms  (actual, e.g. 28)
        n_perms: tl.constexpr,      # n! = 24 for n=4
        n_beta: tl.constexpr,       # n
        N_ALPHA_PAD: tl.constexpr,  # next pow2 >= n + N_PERMS_PAD  (e.g. 64)
        N_PERMS_PAD: tl.constexpr,  # next pow2 >= n_perms  (e.g. 32)
        BLOCK_D: tl.constexpr,      # autotuned
    ):
        """Pass 2: compute alpha_pre, H_res (in registers), new_residuals, branch_input, beta."""
        token_id = tl.program_id(0)
        if token_id >= T:
            return

        # ---- Load wc in two parts (slice notation unsupported in this Triton version) ----
        wc_base = token_id * N_ALPHA_PAD
        wc_pre = tl.load(wc_ptr + wc_base + tl.arange(0, n)).to(tl.float32)               # [n]
        wc_res = tl.load(wc_ptr + wc_base + n + tl.arange(0, N_PERMS_PAD)).to(tl.float32) # [N_PERMS_PAD]
        dc = tl.load(dc_ptr + token_id * n_beta + tl.arange(0, n_beta)).to(tl.float32)

        # ---- alpha_pre [n] = sigmoid(scale * wc_pre + static_alpha[:n]) ----
        sa_pre = tl.load(static_alpha_ptr + tl.arange(0, n)).to(tl.float32)
        alpha_pre = tl.sigmoid(pre_branch_scale * wc_pre + sa_pre)                         # [n]

        # ---- res_coeff [N_PERMS_PAD] = softmax(scale * wc_res + static_alpha[n:]) ----
        # sa_res: padded to N_PERMS_PAD; elements >= n_perms are 0
        sa_res = tl.load(
            static_alpha_ptr + n + tl.arange(0, N_PERMS_PAD),
            mask=tl.arange(0, N_PERMS_PAD) < n_perms, other=0.0,
        ).to(tl.float32)
        logits = residual_scale * wc_res + sa_res                                           # [N_PERMS_PAD]
        # Mask padded positions to large negative value so softmax gives them zero weight
        logits = tl.where(tl.arange(0, N_PERMS_PAD) < n_perms, logits, -1e9)
        logits = logits - tl.max(logits, axis=0)                       # numeric stability
        exp_l = tl.exp(logits)
        res_coeff = exp_l / tl.sum(exp_l)                              # [N_PERMS_PAD]

        # ---- Precompute h_rows: one [n] weight-vector per output stream ----
        # Tritonのリスト制約を回避するため、n=4 を前提に手動でアンロールして展開
        rows_p = tl.arange(0, N_PERMS_PAD)
        cols_n = tl.arange(0, n)
        
        p0 = tl.load(perms_ptr + rows_p[:, None] * (n * n) + 0 * n + cols_n[None, :], mask=rows_p[:, None] < n_perms, other=0.0).to(tl.float32)
        h_row_0 = tl.sum(res_coeff[:, None] * p0, axis=0)
        
        p1 = tl.load(perms_ptr + rows_p[:, None] * (n * n) + 1 * n + cols_n[None, :], mask=rows_p[:, None] < n_perms, other=0.0).to(tl.float32)
        h_row_1 = tl.sum(res_coeff[:, None] * p1, axis=0)
        
        p2 = tl.load(perms_ptr + rows_p[:, None] * (n * n) + 2 * n + cols_n[None, :], mask=rows_p[:, None] < n_perms, other=0.0).to(tl.float32)
        h_row_2 = tl.sum(res_coeff[:, None] * p2, axis=0)
        
        p3 = tl.load(perms_ptr + rows_p[:, None] * (n * n) + 3 * n + cols_n[None, :], mask=rows_p[:, None] < n_perms, other=0.0).to(tl.float32)
        h_row_3 = tl.sum(res_coeff[:, None] * p3, axis=0)

        # ---- beta [n] = sigmoid(scale * dc + static_beta) * 2 ----
        sb = tl.load(static_beta_ptr + tl.arange(0, n_beta)).to(tl.float32)
        beta_vec = tl.sigmoid(h_post_scale * dc + sb) * 2.0           # [n]
        tl.store(beta_ptr + token_id * n + tl.arange(0, n_beta),
                 beta_vec.to(tl.bfloat16))

        # ---- Process D in blocks: compute new_residuals and branch_input ----
        # X layout: [T, n, D] with n=4; X[token, s, d] at token*n*D + s*D + d
        x_base = token_id * n * D
        n_idx = tl.arange(0, n)

        for d_blk in range(tl.cdiv(D, BLOCK_D)):
            d_offs = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D

            # 2D load all n streams at once: x_chunk [n, BLOCK_D]
            x_ptrs = X_ptr + x_base + n_idx[:, None] * D + d_offs[None, :]
            x_chunk = tl.load(x_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)

            # branch_input[d] = Σ_s alpha_pre[s] * x_s[d]
            bi = tl.sum(alpha_pre[:, None] * x_chunk, axis=0)           # [BLOCK_D]
            tl.store(branch_input_ptr + token_id * D + d_offs,
                     bi.to(tl.bfloat16), mask=d_mask)

            # new_residuals[i, d] = Σ_j h_rows_i[j] * x_chunk[j, d]
            # こちらもリストを使わず個別のテンソルで計算
            nr_0 = tl.sum(h_row_0[:, None] * x_chunk, axis=0)
            nr_1 = tl.sum(h_row_1[:, None] * x_chunk, axis=0)
            nr_2 = tl.sum(h_row_2[:, None] * x_chunk, axis=0)
            nr_3 = tl.sum(h_row_3[:, None] * x_chunk, axis=0)

            # メモリへの書き込み
            out_ptr_base = new_residuals_ptr + token_id * n * D + d_offs
            tl.store(out_ptr_base + 0 * D, nr_0.to(tl.bfloat16), mask=d_mask)
            tl.store(out_ptr_base + 1 * D, nr_1.to(tl.bfloat16), mask=d_mask)
            tl.store(out_ptr_base + 2 * D, nr_2.to(tl.bfloat16), mask=d_mask)
            tl.store(out_ptr_base + 3 * D, nr_3.to(tl.bfloat16), mask=d_mask)

    # -----------------------------------------------------------------------
    # Depth connection kernels:
    #   output[s, t, d] = beta[t, s] * x_out[t, d] + new_residuals[t, s, d]
    #   Stored in [n, T, D] layout directly (== permute(2,0,1,3) of [T,n,D]).
    # -----------------------------------------------------------------------

    @triton.autotune(
        configs=_MHC_DEPTH_AUTOTUNE_CONFIGS,
        key=['D', 'n'],
    )
    @triton.jit
    def _fused_mhc_depth_fwd_kernel(
        x_out_ptr,            # [T, D]
        new_residuals_ptr,    # [T, n, D]
        beta_ptr,             # [T, n]
        output_ptr,           # [n, T, D]
        T,
        n: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        t = tl.program_id(0)
        s = tl.program_id(1)
        if t >= T:
            return

        beta_s = tl.load(beta_ptr + t * n + s).to(tl.float32)

        for d_blk in range(tl.cdiv(D, BLOCK_D)):
            d_offs = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D

            x = tl.load(
                x_out_ptr + t * D + d_offs, mask=d_mask, other=0.0
            ).to(tl.float32)
            nr = tl.load(
                new_residuals_ptr + t * n * D + s * D + d_offs,
                mask=d_mask, other=0.0,
            ).to(tl.float32)

            out = beta_s * x + nr
            tl.store(
                output_ptr + s * T * D + t * D + d_offs,
                out.to(tl.bfloat16),
                mask=d_mask,
            )

    @triton.autotune(
        configs=_MHC_DEPTH_AUTOTUNE_CONFIGS,
        key=['D', 'n'],
    )
    @triton.jit
    def _fused_mhc_depth_bwd_kernel(
        grad_output_ptr,      # [n, T, D]
        x_out_ptr,            # [T, D]
        beta_ptr,             # [T, n]
        # Outputs
        d_x_out_ptr,          # [T, D]
        d_new_residuals_ptr,  # [T, n, D]
        d_beta_ptr,           # [T, n]
        T,
        n: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per token t. Computes d_x_out, d_new_residuals, d_beta for all n streams."""
        t = tl.program_id(0)
        if t >= T:
            return

        beta = tl.load(beta_ptr + t * n + tl.arange(0, n)).to(tl.float32)  # [n]

        # Accumulate d_beta across D blocks.
        d_beta_acc = tl.zeros([n], dtype=tl.float32)

        n_idx = tl.arange(0, n)

        for d_blk in range(tl.cdiv(D, BLOCK_D)):
            d_offs = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D

            # Load grad_output [n, BLOCK_D] for this t.
            go_ptrs = (
                grad_output_ptr
                + n_idx[:, None] * (T * D)
                + t * D
                + d_offs[None, :]
            )
            go = tl.load(
                go_ptrs, mask=d_mask[None, :], other=0.0
            ).to(tl.float32)  # [n, BLOCK_D]

            # Load x_out [BLOCK_D] for this t.
            x = tl.load(
                x_out_ptr + t * D + d_offs, mask=d_mask, other=0.0
            ).to(tl.float32)  # [BLOCK_D]

            # d_beta[s] += sum_d(go[s, d] * x[d])
            d_beta_acc += tl.sum(go * x[None, :], axis=1)  # [n]

            # d_x_out[d] = sum_s(beta[s] * go[s, d])
            dx = tl.sum(beta[:, None] * go, axis=0)  # [BLOCK_D]
            tl.store(
                d_x_out_ptr + t * D + d_offs,
                dx.to(tl.bfloat16),
                mask=d_mask,
            )

            # d_new_residuals[s, d] = go[s, d]
            dnr_ptrs = (
                d_new_residuals_ptr
                + t * n * D
                + n_idx[:, None] * D
                + d_offs[None, :]
            )
            tl.store(
                dnr_ptrs, go.to(tl.bfloat16), mask=d_mask[None, :]
            )

        # Store d_beta [n] for this t.
        tl.store(
            d_beta_ptr + t * n + tl.arange(0, n),
            d_beta_acc.to(tl.bfloat16),
        )


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
            N_PERMS_PAD = _next_pow2(n_perms)           # 24 → 32
            N_ALPHA_PAD = _next_pow2(n + N_PERMS_PAD)   # 36 → 64

            # Scratch buffers: wc uses padded size; zeros() ensures padding elements are 0
            wc_scratch = torch.zeros((T, N_ALPHA_PAD), dtype=torch.float32, device=device)
            dc_scratch = torch.empty((T, n_beta), dtype=torch.float32, device=device)

            # Output tensors
            branch_input = torch.empty((T, D), dtype=X_flat.dtype, device=device)
            new_residuals = torch.empty((T, n, D), dtype=X_flat.dtype, device=device)
            beta_out = torch.empty((T, n), dtype=X_flat.dtype, device=device)

            # X needs to be viewed as [T, nD] for Pass 1 (RMSNorm uses flattened)
            X_flat_nd = X_flat.reshape(T, nD)

            # Pass 1: fused RMSNorm + projection → wc, dc scratch buffers
            # BLOCK_ND / num_warps / num_stages are picked by @triton.autotune.
            _fused_mhc_wc_proj_kernel[(T,)](
                X_flat_nd, W_alpha, W_beta, gamma,
                wc_scratch, dc_scratch,
                T=T, nD=nD, n_alpha=n_alpha, n_beta=n_beta,
                N_ALPHA_PAD=N_ALPHA_PAD,
            )

            # Pass 2: compute alpha_pre, H_res (in registers), outputs
            # BLOCK_D / num_warps / num_stages are picked by @triton.autotune.
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
                N_ALPHA_PAD=N_ALPHA_PAD, N_PERMS_PAD=N_PERMS_PAD,
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
# Depth-connection autograd Function (Triton forward & backward)
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    class _FusedMHCDepthFunction(torch.autograd.Function):
        """Fused depth connection.

        Forward:  output[s, t, d] = beta[t, s] * x_out[t, d] + new_residuals[t, s, d]
        Layout:   output is returned in [n, T, D] = permute(2, 0, 1, 3) order,
                  which is what ``_depth_connection`` needs to feed back into
                  SusonoBlock as ``[n, S, B, D]``.

        Backward computes d_x_out, d_new_residuals, d_beta directly from
        grad_output in a single per-token kernel (no PyTorch recompute).
        """

        @staticmethod
        def forward(ctx, x_out, new_residuals, beta):
            # x_out:         [T, D]
            # new_residuals: [T, n, D]
            # beta:          [T, n]
            T, n, D = new_residuals.shape
            output = torch.empty(
                (n, T, D), dtype=x_out.dtype, device=x_out.device,
            )
            _fused_mhc_depth_fwd_kernel[(T, n)](
                x_out, new_residuals, beta, output,
                T=T, n=n, D=D,
            )
            ctx.save_for_backward(x_out, beta)
            ctx.T, ctx.n, ctx.D = T, n, D
            return output

        @staticmethod
        def backward(ctx, grad_output):
            x_out, beta = ctx.saved_tensors
            T, n, D = ctx.T, ctx.n, ctx.D

            grad_output_c = grad_output.contiguous()

            d_x_out = torch.empty_like(x_out)
            d_new_residuals = torch.empty(
                (T, n, D), dtype=x_out.dtype, device=x_out.device,
            )
            d_beta = torch.empty_like(beta)

            _fused_mhc_depth_bwd_kernel[(T,)](
                grad_output_c, x_out, beta,
                d_x_out, d_new_residuals, d_beta,
                T=T, n=n, D=D,
            )
            return d_x_out, d_new_residuals, d_beta


def fused_mhc_depth_connection(
    x_out: Tensor,
    new_residuals: Tensor,
    beta: Tensor,
) -> Tensor:
    """Fused MHC depth connection with Triton forward & backward.

    Args:
        x_out:         [S, B, D]       — transformer layer output (single-stream).
        new_residuals: [S, B, n, D]    — permutation-mixed residuals from width.
        beta:          [S, B, n]       — per-stream gate weights.

    Returns:
        [n, S, B, D] multi-stream hidden states, ready to feed the next layer.
    """
    S, B, n, D = new_residuals.shape
    T = S * B

    use_triton = (
        _TRITON_AVAILABLE
        and x_out.is_cuda
        and x_out.dtype == torch.bfloat16
        and n == 4
    )

    if use_triton:
        x_out_t = x_out.reshape(T, D).contiguous()
        new_residuals_t = new_residuals.reshape(T, n, D).contiguous()
        beta_t = beta.reshape(T, n).contiguous()
        output = _FusedMHCDepthFunction.apply(x_out_t, new_residuals_t, beta_t)
        return output.view(n, S, B, D)

    # Pure-PyTorch fallback (unchanged from previous behaviour).
    output = beta.unsqueeze(-1) * x_out.unsqueeze(-2) + new_residuals
    return output.permute(2, 0, 1, 3).contiguous()


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
