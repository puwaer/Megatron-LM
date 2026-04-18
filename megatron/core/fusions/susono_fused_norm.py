# Copyright (c) 2026 Susono authors.
"""Fused RMSNorm helpers for the Susono architecture.

Two variants are provided:
  - ``rmsnorm_1p``: RMSNorm with a zero-centered gamma scaling ``(1 + weight)``.
    Used in MHC-Lite stream normalisation and the hybrid linear-attention
    decoder layer (matches Qwen3-Next "SusonoRMSNorm" convention).
  - ``rmsnorm_gated``: standard RMSNorm followed by a SiLU-gated multiplication,
    used as the output norm of ``SusonoGatedDeltaNet``.

When ``liger_kernel`` is installed, the underlying RMSNorm is performed by
``LigerRMSNormFunction`` (Triton, fwd+bwd fused, zero intermediate materialisation).
Otherwise a pure-PyTorch fallback is used; the fallback is bit-wise equivalent
to the previous inline implementations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    _HAVE_LIGER_RMSNORM = True
except ImportError:
    LigerRMSNormFunction = None
    _HAVE_LIGER_RMSNORM = False


def _rmsnorm_pytorch(x: Tensor, weight: Tensor, eps: float, offset: float) -> Tensor:
    """Pure-PyTorch RMSNorm with optional (offset + weight) scaling."""
    dtype = x.dtype
    x_f = x.float()
    out = x_f * torch.rsqrt((x_f * x_f).mean(-1, keepdim=True) + eps)
    if offset != 0.0:
        scale = (offset + weight.float()).to(dtype)
    else:
        scale = weight.to(dtype)
    return out.to(dtype) * scale


def rmsnorm_1p(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm with (1 + weight) zero-centered gamma scaling.

    Matches the prior ``_RMSNorm.forward`` bit-for-bit:
        out = x * rsqrt(mean(x^2) + eps)    # computed in fp32
        y   = out * (1 + weight)            # scale, then cast back to x.dtype
    """
    if _HAVE_LIGER_RMSNORM:
        # Liger's offset adds `offset` to weight inside the fused kernel:
        # final gamma = offset + weight. casting_mode="llama" matches
        # our fp32-compute / input-dtype-output convention.
        # in_place=False: the input x is typically used again for the residual
        # add downstream, so we must not overwrite it.
        return LigerRMSNormFunction.apply(
            x, weight, eps, 1.0, "llama", False,
        )
    return _rmsnorm_pytorch(x, weight, eps, offset=1.0)


def rmsnorm_gated(x: Tensor, weight: Tensor, gate: Tensor, eps: float = 1e-6) -> Tensor:
    """Standard RMSNorm (gamma = weight) followed by ``* SiLU(gate)``.

    Matches the prior ``_RMSNormGated.forward``.  The RMSNorm is the expensive
    part (kernel launch + intermediate tensor) and is what Liger fuses.  The
    SiLU-gate multiplication is a small pointwise op applied after.
    """
    if _HAVE_LIGER_RMSNORM:
        # in_place=False to keep x untouched for any downstream residual use.
        normed = LigerRMSNormFunction.apply(
            x, weight, eps, 0.0, "llama", False,
        )
    else:
        normed = _rmsnorm_pytorch(x, weight, eps, offset=0.0)
    # Apply SiLU-gated mul in fp32 for numerical parity with the
    # original implementation, then cast back.
    dtype = normed.dtype
    return normed * F.silu(gate.float()).to(dtype)
