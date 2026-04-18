# Copyright (c) 2026 Susono authors.
"""Unit tests for megatron.core.fusions.fused_engram_lookup.

Validates that ``fused_engram_hash_and_gather`` is numerically equivalent
to the reference path through ``NgramHashMapping`` + ``MultiHeadEmbedding``
used by ``EngramModule``.

The tests are GPU-only because the fused Triton kernel requires CUDA.
They are skipped gracefully on CPU-only machines.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused_engram_lookup requires CUDA",
)


def _run_with_config(
    B: int = 2,
    S: int = 32,
    base_vocab_size: int = 4096,
    max_ngram_size: int = 3,
    n_head_per_ngram: int = 4,
    n_embed_per_ngram: int = 9973,  # prime
    n_embed_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    """Build a NgramHashMapping + table, run both paths, return (ref, fused, indices_ref)."""
    from megatron.core.models.engram.engram_module import (
        EngramConfig,
        MultiHeadEmbedding,
        NgramHashMapping,
    )
    from megatron.core.fusions.fused_engram_lookup import (
        _fallback_hash_and_gather,
        fused_engram_hash_and_gather,
    )

    config = EngramConfig(
        max_ngram_size=max_ngram_size,
        n_embed_per_ngram=n_embed_per_ngram,
        n_head_per_ngram=n_head_per_ngram,
        n_embed_dim=n_embed_dim,
        base_vocab_size=base_vocab_size,
        seed=seed,
    )

    torch.manual_seed(seed)
    hash_mod = NgramHashMapping(config, layer_id=3).cuda()
    total_rows = sum(hash_mod.vocab_sizes)
    emb_mod = MultiHeadEmbedding(total_rows, n_embed_dim).cuda()
    emb_mod.table.weight.data = emb_mod.table.weight.data.to(dtype)

    # Random compressed token ids.
    compressed = torch.randint(
        0, base_vocab_size, (B, S), dtype=torch.long, device="cuda",
    )

    # Reference path: compute indices explicitly then gather.
    indices_ref = hash_mod(compressed)                        # [B,S,total_heads]
    emb_ref = emb_mod(indices_ref)                            # [B,S,total_heads,E]
    emb_flat_ref = emb_ref.reshape(B, S, -1).contiguous()

    # Fused path.
    emb_flat_fused = fused_engram_hash_and_gather(
        compressed,
        emb_mod.table.weight,
        hash_mod.multipliers,
        hash_mod.primes_tensor,
        hash_mod.head_base_flat,
    )

    # Fallback path for sanity (must exactly match the reference path).
    emb_flat_fallback = _fallback_hash_and_gather(
        compressed,
        emb_mod.table.weight,
        hash_mod.multipliers,
        hash_mod.primes_tensor,
        hash_mod.head_base_flat,
    )

    return {
        "indices_ref": indices_ref,
        "emb_flat_ref": emb_flat_ref,
        "emb_flat_fused": emb_flat_fused,
        "emb_flat_fallback": emb_flat_fallback,
        "hash_mod": hash_mod,
        "emb_mod": emb_mod,
        "compressed": compressed,
        "config": config,
    }


def test_fallback_matches_reference():
    """The pure-PyTorch fallback must be bit-wise identical to the reference
    produced by NgramHashMapping + MultiHeadEmbedding."""
    r = _run_with_config()
    assert torch.equal(r["emb_flat_ref"], r["emb_flat_fallback"])


def test_fused_indices_bit_exact():
    """The fused kernel must produce bit-wise identical indices to the reference
    (hash computation is pure integer arithmetic)."""
    r = _run_with_config()

    # Recover indices from the fused path by matching embedding rows.
    # Easier: re-run the fused kernel and inspect its saved ``indices`` by
    # calling the underlying Function directly.  We do it indirectly by
    # comparing embedding outputs for bf16 (allows small noise on table
    # gather due to reshape ordering, but indices must match exactly for
    # identical gather results).
    assert torch.equal(r["emb_flat_ref"], r["emb_flat_fused"])


def test_fused_forward_allclose_bf16():
    """bf16 forward outputs must match the reference exactly (same indices →
    same table rows → same values)."""
    r = _run_with_config(dtype=torch.bfloat16)
    assert r["emb_flat_fused"].dtype == torch.bfloat16
    assert torch.equal(r["emb_flat_ref"], r["emb_flat_fused"])


def test_fused_backward_gradient():
    """d_table from the fused backward (scatter_add) must match the reference
    ``F.embedding`` backward."""
    from megatron.core.fusions.fused_engram_lookup import (
        fused_engram_hash_and_gather,
    )

    r = _run_with_config(dtype=torch.float32, S=16, n_embed_dim=32)
    hash_mod = r["hash_mod"]
    emb_mod = r["emb_mod"]
    compressed = r["compressed"]

    # Reference path with gradient.
    table_ref = emb_mod.table.weight.detach().clone().requires_grad_(True)
    indices = hash_mod(compressed)                                # int64 [B,S,th]
    emb_ref = torch.nn.functional.embedding(indices, table_ref)   # [B,S,th,E]
    loss_ref = emb_ref.sum()
    loss_ref.backward()

    # Fused path with gradient.
    table_fused = emb_mod.table.weight.detach().clone().requires_grad_(True)
    emb_flat = fused_engram_hash_and_gather(
        compressed,
        table_fused,
        hash_mod.multipliers,
        hash_mod.primes_tensor,
        hash_mod.head_base_flat,
    )
    loss_fused = emb_flat.sum()
    loss_fused.backward()

    # index_add_ / scatter_add on CUDA uses atomicAdd which is non-deterministic,
    # so the sum order may differ between two invocations even with identical inputs.
    # Allow a tiny absolute tolerance in fp32.
    assert torch.allclose(table_ref.grad, table_fused.grad, rtol=1e-4, atol=1e-4)


def test_fused_backward_random_grad():
    """d_table matches for a non-uniform upstream gradient."""
    from megatron.core.fusions.fused_engram_lookup import (
        fused_engram_hash_and_gather,
    )

    r = _run_with_config(dtype=torch.float32, S=16, n_embed_dim=32)
    hash_mod = r["hash_mod"]
    emb_mod = r["emb_mod"]
    compressed = r["compressed"]

    torch.manual_seed(123)

    # Reference path.
    table_ref = emb_mod.table.weight.detach().clone().requires_grad_(True)
    indices = hash_mod(compressed)
    emb_ref = torch.nn.functional.embedding(indices, table_ref)
    B, S, th, E = emb_ref.shape
    upstream = torch.randn_like(emb_ref)
    emb_ref.backward(upstream)

    # Fused path with the same upstream (reshaped to flat layout).
    table_fused = emb_mod.table.weight.detach().clone().requires_grad_(True)
    emb_flat = fused_engram_hash_and_gather(
        compressed,
        table_fused,
        hash_mod.multipliers,
        hash_mod.primes_tensor,
        hash_mod.head_base_flat,
    )
    emb_flat.backward(upstream.reshape(B, S, th * E))

    assert torch.allclose(table_ref.grad, table_fused.grad, rtol=1e-4, atol=1e-4)
