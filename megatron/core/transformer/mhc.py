# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Manifold-Constrained Hyper-Connections (MHC-Lite).

Implements MHC-Lite: efficient mHC using a learnable convex combination of
permutation matrices instead of iterative Sinkhorn-Knopp projection.

Algorithm (per layer boundary):
    normed        = RMSNorm(concat(X_l streams))              # [S,B,n*D]
    alpha_pre     = sigmoid(scale * normed @ W_pre + b_pre)   # [S,B,n]
    res_coeff     = softmax(scale * normed @ W_res + b_res)   # [S,B,n!]
    H_res         = Σ_r res_coeff[r] * P_r                    # [S,B,n,n]  (doubly stochastic)
    x_in          = Σ_s alpha_pre[s] * X_l[s]                 # [S,B,D]
    new_residuals = H_res @ X_l                               # [S,B,n,D]
    x_out         = F(x_in)                                   # [S,B,D]    (layer function)
    beta          = sigmoid(scale * normed @ W_beta + b_beta) * 2  # [S,B,n]
    X_{l+1}       = beta ⊗ x_out + new_residuals              # [n,S,B,D]

Key differences from Sinkhorn-Knopp mHC:
- H_res is parameterised as a convex combination of all n! permutation matrices.
  By Birkhoff's theorem this spans the same Birkhoff polytope (doubly stochastic
  matrices) as Sinkhorn-Knopp, but requires no iterative projection.
- alpha (H_pre) and beta (H_post) are input-dependent (static + dynamic weights).
- forward() returns a (branch_input, add_residual_fn) closure so that intermediate
  state is captured without mutating instance variables.

Reference: MHC-Lite https://arxiv.org/abs/2601.05732
"""

import itertools
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Module-level permutation-matrix cache  (shared across all instances)
# ---------------------------------------------------------------------------
_perm_mats_cache: dict = {}


def _get_permutation_matrices(
    n: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Return all n! permutation matrices as a tensor of shape [n!, n, n].

    Results are cached per (n, device, dtype) so that the allocation and dtype
    conversion happen only once per unique combination.
    The identity permutation is always the first entry (index 0), matching
    the initialisation heuristic in ``ManifoldConstrainedHyperConnection``.
    """
    key = (n, str(device), dtype)
    if key not in _perm_mats_cache:
        # Build float32 base first (reused across dtype variants on same device)
        base_key = (n, str(device), torch.float32)
        if base_key not in _perm_mats_cache:
            perms = list(itertools.permutations(range(n)))   # identity first
            idx = torch.tensor(perms, dtype=torch.long)
            eye = torch.eye(n, dtype=torch.float32)
            perm_mats = eye[idx]                             # [n!, n, n]
            _perm_mats_cache[base_key] = perm_mats.to(device)
        _perm_mats_cache[key] = _perm_mats_cache[base_key].to(dtype)
    return _perm_mats_cache[key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """RMSNorm used for stream normalisation inside MHC-Lite."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.float()
        out = x_f * torch.rsqrt((x_f * x_f).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class ManifoldConstrainedHyperConnection(nn.Module):
    """MHC-Lite: Manifold-Constrained Hyper-Connections via permutation matrices.

    More memory and compute efficient than Sinkhorn-Knopp based mHC:

    - H_res is a learnable convex combination of n! permutation matrices,
      which spans the Birkhoff polytope without iterative projection
      (O(n²·n!) instead of O(k·n²) per forward for small n, and avoids
      the backward through k Sinkhorn steps entirely).
    - alpha (H_pre) and beta (H_post) are input-dependent via lightweight
      linear projections with small-magnitude dynamic components.

    Args:
        hidden_size:          Dimension D of each residual stream.
        num_streams:          Number of parallel residual streams n.
        layer_index:          Layer index for initialisation (optional).
                              Determines which stream is favoured at init.
        sinkhorn_iterations:  Unused; retained for API backward compatibility.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int,
        layer_index: Optional[int] = None,
        sinkhorn_iterations: int = 20,   # unused, kept for API compat
        use_fused_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.use_fused_kernel = use_fused_kernel
        n = num_streams
        num_perms = math.factorial(n)
        self.num_perms = num_perms

        # Which stream to favour at initialisation
        init_idx = (layer_index if layer_index is not None else 0) % n

        # Normalise all streams concatenated: [S, B, n*D] → [S, B, n*D]
        self.norm = _RMSNorm(hidden_size * n)

        # ------------------------------------------------------------------
        # Width-connection parameters
        #   static_alpha[:n]   → H_pre   (sigmoid, per-stream input gate)
        #   static_alpha[n:]   → H_res   (softmax over n! permutations)
        # ------------------------------------------------------------------
        init_alpha_pre = torch.ones(n) * -1.0
        init_alpha_pre[init_idx] = 1.0

        init_alpha_res = torch.ones(num_perms) * -8.0
        init_alpha_res[0] = 0.0   # identity permutation (index 0) dominates

        self.static_alpha = nn.Parameter(torch.cat([init_alpha_pre, init_alpha_res]))

        # Dynamic (input-dependent) component: [n*D] → [n + n!]
        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(hidden_size * n, n + num_perms)
        )
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale   = nn.Parameter(torch.ones(1) * 1e-2)

        # ------------------------------------------------------------------
        # Depth-connection parameters (beta / H_post)
        #   beta = sigmoid(h_post_scale * normed @ W_beta + b_beta) * 2
        # ------------------------------------------------------------------
        # Initialize beta ≈ 0 so depth connection starts as pure residual.
        # sigmoid(-8)*2 ≈ 0.00067 → per-layer eigenvalue ≈ 1.001
        # Over 48 layers: 1.001^48 ≈ 1.05 (stable), vs 2.503^48 ≈ 1.6E19 before.
        init_beta = torch.ones(n) * -8.0
        self.static_beta = nn.Parameter(init_beta)

        # Dynamic component: [n*D] → [n]
        self.dynamic_beta_fn = nn.Parameter(torch.zeros(hidden_size * n, n))
        self.h_post_scale = nn.Parameter(torch.ones(1) * 1e-2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _width_connection(
        self, X: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute branch input, permutation-mixed residuals, and beta.

        When ``self.use_fused_kernel`` is True and Triton is available, uses
        the fused kernel from :mod:`megatron.core.fusions.fused_mhc_width_connection`
        to keep intermediate tensors (normed, wc, H_res, res_coeff) in SRAM.

        Args:
            X: Multi-stream hidden states [n, S, B, D].

        Returns:
            branch_input:   [S, B, D]    — gated aggregation over streams.
            new_residuals:  [S, B, n, D] — streams after permutation mixing.
            beta:           [S, B, n]    — depth-connection gate weights.
        """
        n, S, B, D = X.shape
        perms = _get_permutation_matrices(n, X.device, X.dtype)  # [n!, n, n], dtype-cached

        if self.use_fused_kernel:
            from megatron.core.fusions.fused_mhc_width_connection import (
                fused_mhc_width_connection,
            )
            return fused_mhc_width_connection(
                X,
                self.dynamic_alpha_fn,
                self.dynamic_beta_fn,
                self.static_alpha,
                self.static_beta,
                self.norm.weight,
                perms,
                self.pre_branch_scale,
                self.residual_scale,
                self.h_post_scale,
            )

        # ---- Standard PyTorch path ----------------------------------------
        # Rearrange to [S, B, n, D] and flatten streams for norm
        X_sb = X.permute(1, 2, 0, 3)                 # [S, B, n, D]
        normed = X_sb.reshape(S, B, n * D)            # [S, B, n*D]
        normed = self.norm(normed)                    # [S, B, n*D]

        # ---- Width weights (alpha) ----------------------------------------
        wc = normed @ self.dynamic_alpha_fn           # [S, B, n + n!]
        dynamic_pre = wc[..., :n]                     # [S, B, n]
        dynamic_res = wc[..., n:]                     # [S, B, n!]

        # H_pre: input-gated per-stream weights (sigmoid ∈ (0,1))
        alpha_pre = torch.sigmoid(
            self.pre_branch_scale * dynamic_pre + self.static_alpha[:n]
        )                                             # [S, B, n]

        # H_res: convex combination of permutation matrices (no Sinkhorn)
        res_coeff = torch.softmax(
            self.residual_scale * dynamic_res + self.static_alpha[n:],
            dim=-1,
        )                                             # [S, B, n!]
        H_res = torch.einsum('...r, rij -> ...ij', res_coeff, perms)  # [S, B, n, n]

        # Apply H_res: new_residuals[..., i, :] = Σ_j H_res[..., i, j] * X[..., j, :]
        new_residuals = torch.einsum(
            '...ij, ...jd -> ...id', H_res, X_sb
        )                                             # [S, B, n, D]

        # Branch input: gated weighted sum over streams (einsum avoids [S,B,n,D] intermediate)
        branch_input = torch.einsum('...n,...nd->...d', alpha_pre, X_sb)  # [S, B, D]

        # ---- Depth weights (beta) ----------------------------------------
        dc = normed @ self.dynamic_beta_fn            # [S, B, n]
        beta = torch.sigmoid(
            self.h_post_scale * dc + self.static_beta
        ) * 2                                         # [S, B, n]  (range [0, 2])

        return branch_input, new_residuals, beta

    def _depth_connection(
        self,
        x_out: Tensor,
        new_residuals: Tensor,
        beta: Tensor,
    ) -> Tensor:
        """Combine layer output with permutation-mixed residuals.

        Args:
            x_out:         Layer output [S, B, D].
            new_residuals: Permutation-mixed streams [S, B, n, D].
            beta:          Per-stream gate weights [S, B, n].

        Returns:
            Updated multi-stream hidden states [n, S, B, D].
        """
        # beta: [S,B,n] → [S,B,n,1];  x_out: [S,B,D] → [S,B,1,D]
        output = (
            beta.unsqueeze(-1) * x_out.unsqueeze(-2) + new_residuals
        )                                             # [S, B, n, D]
        return output.permute(2, 0, 1, 3)            # [n, S, B, D]

    # ------------------------------------------------------------------
    # Public API  (closure pattern — no instance-variable mutation)
    # ------------------------------------------------------------------

    def forward(self, X: Tensor) -> Tuple[Tensor, Callable[[Tensor], Tensor]]:
        """Width connection: returns branch input and a depth-connection closure.

        Usage::

            branch_input, add_residual = mhc(hidden_states)   # [S,B,D], closure
            x_out = transformer_layer(branch_input)            # [S,B,D]
            hidden_states = add_residual(x_out)                # [n,S,B,D]

        Args:
            X: Multi-stream hidden states [n, S, B, D].

        Returns:
            branch_input:  Aggregated input [S, B, D].
            add_residual:  Closure ``(x_out: [S,B,D]) → [n,S,B,D]``.
        """
        branch_input, new_residuals, beta = self._width_connection(X)

        def add_residual(x_out: Tensor) -> Tensor:
            return self._depth_connection(x_out, new_residuals, beta)

        return branch_input, add_residual

    # ------------------------------------------------------------------
    # Legacy interface (aggregate_streams / distribute_output)
    # kept so that existing callers continue to work.
    # NOTE: not thread-safe — do not use in multi-threaded inference.
    # ------------------------------------------------------------------

    def aggregate_streams(self, X: Tensor) -> Tensor:
        """[Legacy] Aggregate n streams → single branch input.

        Internally caches mixed residuals and beta for the subsequent
        :meth:`distribute_output` call.  Prefer the closure API
        (:meth:`forward`) for new code.
        """
        branch_input, new_residuals, beta = self._width_connection(X)
        self._cached_residuals = new_residuals
        self._cached_beta = beta
        return branch_input

    def distribute_output(self, X: Tensor, x_out: Tensor) -> Tensor:
        """[Legacy] Update n streams using cached width-connection state.

        Must be called immediately after :meth:`aggregate_streams`.
        """
        new_residuals = self._cached_residuals
        beta = self._cached_beta
        self._cached_residuals = None
        self._cached_beta = None
        return self._depth_connection(x_out, new_residuals, beta)
