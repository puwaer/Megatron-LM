# Copyright (c) 2026 Susono authors.
"""Unit tests for the Approach-B fused Engram path
(``fused_engram_hash_gather_headproj``).

Verifies numerical equivalence with the split path
``fused_engram_hash_and_gather → head_proj`` and the pure-PyTorch reference.
GPU-only (Triton required).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused_engram_lookup requires CUDA",
)


def _build(B=2, S=32, base_vocab=4096, max_ngram=3, n_head=4,
           n_embed=9973, embed_dim=64, out_dim=40, seed=123):
    from megatron.core.models.engram.engram_module import (
        EngramConfig, MultiHeadEmbedding, NgramHashMapping,
    )

    config = EngramConfig(
        max_ngram_size=max_ngram,
        n_embed_per_ngram=n_embed,
        n_head_per_ngram=n_head,
        n_embed_dim=embed_dim,
        base_vocab_size=base_vocab,
        seed=seed,
    )

    torch.manual_seed(seed)
    hash_mod = NgramHashMapping(config, layer_id=3).cuda()
    total_rows = sum(hash_mod.vocab_sizes)
    emb_mod = MultiHeadEmbedding(total_rows, embed_dim).cuda()
    emb_mod.table.weight.data = emb_mod.table.weight.data.to(torch.bfloat16)

    # head_proj: [out_dim, total_heads * embed_dim]
    total_heads = (max_ngram - 1) * n_head
    head_proj_w = (
        torch.randn(out_dim, total_heads * embed_dim, dtype=torch.bfloat16, device="cuda") * 0.01
    )

    compressed = torch.randint(
        0, base_vocab, (B, S), dtype=torch.long, device="cuda",
    )

    return {
        "compressed": compressed,
        "table": emb_mod.table.weight,
        "head_proj_w": head_proj_w,
        "multipliers": hash_mod.multipliers,
        "primes": hash_mod.primes_tensor,
        "head_base": hash_mod.head_base_flat,
        "hash_mod": hash_mod,
        "emb_mod": emb_mod,
    }


def test_approach_b_forward_matches_split_path_bf16():
    """Approach-B fused forward must match the split fused_hash_and_gather →
    head_proj path within bf16 precision."""
    from megatron.core.fusions.fused_engram_lookup import (
        fused_engram_hash_and_gather,
        fused_engram_hash_gather_headproj,
    )

    r = _build()

    # Split path: hash+gather then matmul.
    emb_flat = fused_engram_hash_and_gather(
        r["compressed"], r["table"],
        r["multipliers"], r["primes"], r["head_base"],
    )
    out_split = emb_flat @ r["head_proj_w"].t()

    # Fused path.
    out_fused = fused_engram_hash_gather_headproj(
        r["compressed"], r["table"], r["head_proj_w"],
        r["multipliers"], r["primes"], r["head_base"],
    )

    assert out_split.shape == out_fused.shape
    # Both use the same kernel + cuBLAS GEMM, so output should match bit-for-bit.
    assert torch.equal(out_split, out_fused)


def test_approach_b_backward_d_table_matches_split(dtype=torch.float32):
    """d_table accumulated by Approach-B backward must match the autograd-
    driven split path."""
    from megatron.core.fusions.fused_engram_lookup import (
        fused_engram_hash_and_gather,
        fused_engram_hash_gather_headproj,
    )

    r = _build()
    # Promote to fp32 for a tighter comparison.
    table_ref = r["table"].detach().clone().to(dtype).requires_grad_(True)
    hp_ref = r["head_proj_w"].detach().clone().to(dtype).requires_grad_(True)

    # ---- split path ----
    emb_flat = fused_engram_hash_and_gather(
        r["compressed"], table_ref,
        r["multipliers"], r["primes"], r["head_base"],
    )
    out_ref = emb_flat @ hp_ref.t()
    torch.manual_seed(42)
    upstream = torch.randn_like(out_ref)
    out_ref.backward(upstream)

    # ---- Approach-B path ----
    table_fus = r["table"].detach().clone().to(dtype).requires_grad_(True)
    hp_fus = r["head_proj_w"].detach().clone().to(dtype).requires_grad_(True)
    out_fus = fused_engram_hash_gather_headproj(
        r["compressed"], table_fus, hp_fus,
        r["multipliers"], r["primes"], r["head_base"],
    )
    out_fus.backward(upstream.detach())

    # scatter_add on CUDA is non-deterministic → allow tight but non-zero tolerance.
    assert torch.allclose(
        table_ref.grad, table_fus.grad, rtol=1e-4, atol=1e-4,
    ), (
        f"d_table max diff = "
        f"{(table_ref.grad - table_fus.grad).abs().max().item():.3e}"
    )
    assert torch.allclose(
        hp_ref.grad, hp_fus.grad, rtol=1e-4, atol=1e-4,
    ), (
        f"d_head_proj_w max diff = "
        f"{(hp_ref.grad - hp_fus.grad).abs().max().item():.3e}"
    )


def test_approach_b_matches_pytorch_reference_bf16():
    """Full autograd reference (pytorch fallback + nn.Linear) versus fused."""
    from megatron.core.fusions.fused_engram_lookup import (
        _fallback_hash_and_gather,
        fused_engram_hash_gather_headproj,
    )

    r = _build()

    # Reference: pure PyTorch hash+gather then matmul via autograd.
    table_ref = r["table"].detach().clone().requires_grad_(True)
    hp_ref = r["head_proj_w"].detach().clone().requires_grad_(True)

    emb_flat_ref = _fallback_hash_and_gather(
        r["compressed"], table_ref,
        r["multipliers"], r["primes"], r["head_base"],
    )
    out_ref = emb_flat_ref @ hp_ref.t()
    torch.manual_seed(99)
    upstream = torch.randn_like(out_ref)
    out_ref.backward(upstream)

    # Fused.
    table_fus = r["table"].detach().clone().requires_grad_(True)
    hp_fus = r["head_proj_w"].detach().clone().requires_grad_(True)
    out_fus = fused_engram_hash_gather_headproj(
        r["compressed"], table_fus, hp_fus,
        r["multipliers"], r["primes"], r["head_base"],
    )
    out_fus.backward(upstream.detach())

    assert torch.equal(out_ref, out_fus), "forward mismatch"
    # bf16 atomic_add noise in d_table → relax tolerance a bit.
    assert torch.allclose(table_ref.grad, table_fus.grad, rtol=1e-2, atol=1e-2)
    assert torch.allclose(hp_ref.grad, hp_fus.grad, rtol=1e-2, atol=1e-2)
