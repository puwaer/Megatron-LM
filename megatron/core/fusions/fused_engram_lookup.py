# Copyright (c) 2026 Susono authors.
"""Fused Triton kernel for Engram hash + gather.

Replaces the two Python-dispatched steps in ``EngramModule.forward``::

    indices = ngram_hash(compressed_ids)       # [B, S, total_heads]   int64
    emb     = multi_head_emb(indices)           # [B, S, total_heads, E] bf16
    emb_flat = emb.view(B, S, total_heads * E)

with a single fused kernel that computes all N-gram hashes in-register
and performs the embedding-table gather directly into the flattened
``[B, S, total_heads * E]`` output.  The intermediate ``indices`` tensor
is still produced (small, int64) because it is required by backward to
scatter-add gradients into ``table.weight``.

Numerical semantics (must match the reference PyTorch implementation):

    hash_val = XOR_i (t_i * m_{k,h,i})    # int64 arithmetic
    index    = (hash_val % prime + prime) % prime + head_base

where ``prime`` is the table modulus per N-gram order and ``head_base``
is the precomputed ``offsets[order] + head_idx * prime`` base.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAVE_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernel: hash + gather → [B, S, total_heads * E] flat output
# ---------------------------------------------------------------------------

if _HAVE_TRITON:

    @triton.jit
    def _fused_engram_hash_gather_kernel(
        # Inputs
        compressed_ptr,      # [B, S]                     int64
        table_ptr,           # [total_rows, E]            bf16
        mults_ptr,           # [NUM_ORDERS, H, MAX_K]     int64
        primes_ptr,          # [NUM_ORDERS]               int64
        head_base_ptr,       # [NUM_ORDERS * H]           int64
        # Outputs
        indices_ptr,         # [B, S, NUM_ORDERS * H]     int64
        out_ptr,             # [B, S, NUM_ORDERS * H, E]  bf16  (flat-viewed by caller)
        # Scalars / shape
        B,
        S,
        NUM_ORDERS: tl.constexpr,
        H_PER_ORDER: tl.constexpr,
        MAX_K: tl.constexpr,
        E: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Compute one (b, s, global_head) hash + gather per program.

        grid = (B * S, NUM_ORDERS * H_PER_ORDER)
        """
        bs = tl.program_id(0)
        gh = tl.program_id(1)

        b = bs // S
        s = bs % S

        order_idx = gh // H_PER_ORDER
        head_idx = gh % H_PER_ORDER
        k = order_idx + 2  # N-gram orders start at 2

        prime = tl.load(primes_ptr + order_idx)
        head_base = tl.load(head_base_ptr + gh)

        # ---- Hash computation (int64 XOR cascade) -----------------------------
        hash_val = tl.zeros([1], dtype=tl.int64)
        s_last = S - 1
        for i in tl.static_range(MAX_K):
            active = i < k
            # padding: clamp s+i to the last position (matches "last-token repeat")
            s_clamped = tl.minimum(s + i, s_last)
            tok = tl.load(
                compressed_ptr + b * S + s_clamped,
                mask=active,
                other=0,
            )
            m = tl.load(
                mults_ptr + (order_idx * H_PER_ORDER + head_idx) * MAX_K + i,
                mask=active,
                other=0,
            )
            prod = tok * m
            hash_val = tl.where(active, hash_val ^ prod, hash_val)

        # hash_val % prime with PyTorch "floor mod" semantics (always non-negative)
        rem = hash_val % prime
        rem = tl.where(rem < 0, rem + prime, rem)
        index = rem + head_base

        # Reduce [1]-block → scalar so we can (a) store to a scalar address
        # (`indices_ptr + ...`) and (b) use it as a scalar offset for the
        # subsequent gather.  Triton's `tl.store` rejects block-typed values
        # when the pointer is a scalar, so the reduction is mandatory here.
        idx_scalar = tl.sum(index, axis=0)

        # Store index for backward scatter-add
        tl.store(indices_ptr + bs * (NUM_ORDERS * H_PER_ORDER) + gh, idx_scalar)

        # ---- Gather row of table[index, :] directly into flat output ----------
        row_base = idx_scalar * E
        out_row_base = (bs * (NUM_ORDERS * H_PER_ORDER) + gh) * E

        for e_start in range(0, E, BLOCK_E):
            offs = e_start + tl.arange(0, BLOCK_E)
            mask = offs < E
            vals = tl.load(table_ptr + row_base + offs, mask=mask, other=0.0)
            tl.store(out_ptr + out_row_base + offs, vals, mask=mask)


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------


class FusedEngramHashAndGather(torch.autograd.Function):
    """Fused hash + embedding gather.

    Forward:  (compressed_ids, table_weight, mults, primes, head_base)
              -> flat_emb [B, S, total_heads * E]
    Backward: d_table = scatter_add_(dim=0, indices, d_flat_emb.view(-1, E))

    The gradient w.r.t. ``compressed_ids`` and hash params is None
    (integer ops, non-differentiable).
    """

    @staticmethod
    def forward(
        ctx,
        compressed_ids: Tensor,    # [B, S] int64
        table_weight: Tensor,       # [total_rows, E] bf16
        multipliers: Tensor,        # [NUM_ORDERS, H, MAX_K] int64
        primes: Tensor,             # [NUM_ORDERS] int64
        head_base: Tensor,          # [NUM_ORDERS * H] int64
    ) -> Tensor:
        assert _HAVE_TRITON, "Triton is required for FusedEngramHashAndGather"
        assert compressed_ids.is_cuda and table_weight.is_cuda
        assert compressed_ids.dtype == torch.int64
        assert multipliers.dtype == torch.int64 and primes.dtype == torch.int64
        assert head_base.dtype == torch.int64

        B, S = compressed_ids.shape
        num_orders, H_per_order, MAX_K = multipliers.shape
        total_heads = num_orders * H_per_order
        total_rows, E = table_weight.shape

        device = compressed_ids.device
        indices = torch.empty(
            (B, S, total_heads), dtype=torch.int64, device=device
        )
        out_flat = torch.empty(
            (B, S, total_heads * E), dtype=table_weight.dtype, device=device
        )

        compressed_c = compressed_ids.contiguous()
        table_c = table_weight.contiguous()
        mults_c = multipliers.contiguous()
        primes_c = primes.contiguous()
        head_base_c = head_base.contiguous()

        # Tile size over the embedding feature dim. 128 is a reasonable default
        # for hopper; kernel loops internally if E exceeds BLOCK_E.
        BLOCK_E = 128 if E >= 128 else triton.next_power_of_2(E)

        grid = (B * S, total_heads)
        _fused_engram_hash_gather_kernel[grid](
            compressed_c, table_c, mults_c, primes_c, head_base_c,
            indices,
            out_flat,
            B, S,
            NUM_ORDERS=num_orders,
            H_PER_ORDER=H_per_order,
            MAX_K=MAX_K,
            E=E,
            BLOCK_E=BLOCK_E,
        )

        # Save only what's needed for backward.  compressed_ids is not saved
        # because indices already contains the resolved addresses.
        ctx.save_for_backward(indices, table_weight)
        ctx.table_shape = table_weight.shape
        return out_flat

    @staticmethod
    def backward(ctx, grad_out_flat: Tensor) -> Tuple[
        None, Tensor, None, None, None
    ]:
        indices, table_weight = ctx.saved_tensors
        total_rows, E = ctx.table_shape

        # grad_out_flat: [B, S, total_heads * E]
        # reshape: [B * S * total_heads, E]
        grad_rows = grad_out_flat.reshape(-1, E)
        flat_indices = indices.reshape(-1)

        d_table = torch.zeros(
            (total_rows, E),
            dtype=grad_out_flat.dtype,
            device=grad_out_flat.device,
        )
        d_table.index_add_(0, flat_indices, grad_rows)

        # Gradients: compressed_ids is int (None), table_weight is d_table,
        # multipliers/primes/head_base are integer buffers (None).
        return None, d_table, None, None, None


def fused_engram_hash_and_gather(
    compressed_ids: Tensor,
    table_weight: Tensor,
    multipliers: Tensor,
    primes: Tensor,
    head_base: Tensor,
) -> Tensor:
    """Run the fused kernel if Triton is available; otherwise fall back.

    Returns a flattened embedding tensor of shape
    ``[B, S, total_heads * embed_dim]`` equivalent to::

        indices = ngram_hash(compressed_ids)         # [B, S, total_heads]
        emb     = F.embedding(indices, table_weight) # [B, S, total_heads, E]
        emb_flat = emb.view(B, S, total_heads * E)

    but computed in a single GPU kernel with no intermediate
    ``products``, ``hash_val`` or int64 ``indices`` arrays living across
    kernel launches.
    """
    if _HAVE_TRITON and compressed_ids.is_cuda and table_weight.is_cuda:
        return FusedEngramHashAndGather.apply(
            compressed_ids, table_weight, multipliers, primes, head_base,
        )
    return _fallback_hash_and_gather(
        compressed_ids, table_weight, multipliers, primes, head_base,
    )


def _fallback_hash_and_gather(
    compressed_ids: Tensor,
    table_weight: Tensor,
    multipliers: Tensor,
    primes: Tensor,
    head_base: Tensor,
) -> Tensor:
    """Pure-PyTorch reference implementation.

    Used when Triton is unavailable, and as ground truth for unit tests.
    Computes the same hash algorithm as the original NgramHashMapping,
    then gathers rows of table_weight, returning a flat
    ``[B, S, total_heads * E]`` tensor.
    """
    B, S = compressed_ids.shape
    num_orders, H_per_order, MAX_K = multipliers.shape
    total_heads = num_orders * H_per_order
    total_rows, E = table_weight.shape

    # Compute indices per order (matches NgramHashMapping.forward).
    all_indices = []
    for order_idx in range(num_orders):
        k = order_idx + 2
        prime = int(primes[order_idx].item())

        # Padding: replicate last token.
        last_tok = compressed_ids[:, -1:].expand(-1, k - 1)
        padded = torch.cat([compressed_ids, last_tok], dim=1)
        ngrams = padded.unfold(1, k, 1)  # [B, S, k]

        mults = multipliers[order_idx, :, :k]  # [H, k]
        # broadcast: [B, S, 1, k] * [1, 1, H, k] -> [B, S, H, k]
        products = ngrams.unsqueeze(2) * mults.unsqueeze(0).unsqueeze(0)
        # XOR cascade over the k dim.
        hash_val = products[..., 0]
        for i in range(1, k):
            hash_val = torch.bitwise_xor(hash_val, products[..., i])

        head_base_o = head_base[order_idx * H_per_order:
                                (order_idx + 1) * H_per_order]  # [H]
        indices_o = (hash_val % prime) + head_base_o  # [B, S, H]
        all_indices.append(indices_o)

    indices = torch.cat(all_indices, dim=-1)  # [B, S, total_heads]
    emb = torch.nn.functional.embedding(indices, table_weight)  # [B, S, th, E]
    return emb.reshape(B, S, total_heads * E)
