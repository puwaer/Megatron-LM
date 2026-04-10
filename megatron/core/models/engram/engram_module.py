# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Engram: Conditional Memory via Scalable Lookup.

Implements the Engram memory module from "Conditional Memory via Scalable Lookup:
A New Axis of Sparsity for Large Language Models" (arXiv:2601.07372).

Engram augments LLMs with a deterministic N-gram hash-based memory:

    token_ids → CompressedTokenizer → NgramHashMapping → MultiHeadEmbedding
              → ShortConv → context-aware gating → fuse with hidden_states

Key design principles:
  - O(1) deterministic addressing (no learned routing)
  - Vocabulary compression reduces effective vocab size by ~23%
  - Multiple hash heads for collision resistance
  - Layer-specific hash multipliers to avoid cross-layer pattern collisions
  - U-shaped capacity allocation: ~20-30% Engram, remainder in MoE

Tensor parallel notes:
  - CompressedTokenizer: replicated across all TP ranks (small lookup table)
  - NgramHashMapping: replicated (deterministic, no weights)
  - MultiHeadEmbedding: replicated in this implementation; future work can shard
    via VocabParallelEmbedding for very large tables
  - ShortConv / gate_proj / out_proj: replicated; out_proj uses RowParallelLinear
    when TP > 1 to keep the output consistent with the rest of the hidden state
"""

import unicodedata
from dataclasses import dataclass, field
from functools import reduce
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EngramConfig:
    """Hyperparameters for the Engram conditional memory module.

    Attributes
    ----------
    max_ngram_size : int
        Largest N-gram order to index (inclusive). The module generates
        N-grams for n = 2 .. max_ngram_size.
    n_embed_per_ngram : int
        Embedding dimension for each N-gram order's hash table.
    n_head_per_ngram : int
        Number of independent hash functions (heads) per N-gram order.
        Multiple heads improve collision resistance.
    engram_layer_ids : List[int]
        0-indexed transformer layer IDs at which Engram is applied.
    seed : int
        Base seed for generating layer- and order-specific hash multipliers.
    base_vocab_size : int
        Vocabulary size of the tokenizer before compression.
    """

    max_ngram_size: int = 3
    n_embed_per_ngram: int = 99991
    """Hash table size (prime) per N-gram order and head. Used as the modulus in
    NgramHashMapping. Should be a prime number for good hashing properties."""

    n_embed_dim: int = 512
    """Embedding dimension for each entry in the multi-head embedding table."""

    n_head_per_ngram: int = 8
    engram_layer_ids: List[int] = field(default_factory=lambda: [0, 14])
    seed: int = 0
    base_vocab_size: int = 129280


# ---------------------------------------------------------------------------
# Vocabulary compression
# ---------------------------------------------------------------------------

class CompressedTokenizer(nn.Module):
    """Maps raw token IDs to a compressed vocabulary via NFKC normalisation.

    Normalisation pipeline (applied to the *string representation* of each
    token at model build-time, then materialised as a lookup table):
        NFKC decomposition → NFD → strip combining marks → lower-case

    The resulting lookup table is a fixed integer buffer (not a parameter).

    Args:
        base_vocab_size: Total number of tokens in the original vocabulary.
        seed:            Unused here but kept for interface symmetry.
    """

    def __init__(self, base_vocab_size: int, seed: int = 0) -> None:
        super().__init__()
        self.base_vocab_size = base_vocab_size

        # Build compression mapping: token_id → compressed_id.
        # We use a simple canonical normalisation on the byte-level repr of
        # token indices (since we don't have access to the actual token strings
        # at this layer). In a real deployment, pass the tokenizer's vocab and
        # call _build_from_vocab. Here we default to identity for portability.
        mapping = torch.arange(base_vocab_size, dtype=torch.long)
        self.register_buffer('mapping', mapping)

    @classmethod
    def from_vocab(cls, vocab: dict, base_vocab_size: int, seed: int = 0) -> 'CompressedTokenizer':
        """Build a CompressedTokenizer from a {token_id: token_string} dict.

        The normalisation pipeline matches the Engram paper:
            NFKC → NFD → remove combining characters → lower-case

        Two token IDs that map to the same normalised form receive the same
        compressed ID, reducing the effective vocabulary.

        Args:
            vocab:            {int token_id: str token_string}
            base_vocab_size:  Total vocabulary size.
            seed:             Unused, kept for interface symmetry.

        Returns:
            A CompressedTokenizer with a non-trivial mapping buffer.
        """
        obj = cls(base_vocab_size, seed)

        # Map each normalised string to its canonical compressed ID
        norm_to_cid: dict = {}
        mapping = torch.arange(base_vocab_size, dtype=torch.long)
        for tid, text in vocab.items():
            if not isinstance(text, str) or tid >= base_vocab_size:
                continue
            norm = unicodedata.normalize('NFKC', text)
            norm = unicodedata.normalize('NFD', norm)
            norm = ''.join(c for c in norm if unicodedata.category(c) != 'Mn')
            norm = norm.lower()
            if norm not in norm_to_cid:
                norm_to_cid[norm] = tid
            mapping[tid] = norm_to_cid[norm]

        obj.mapping = mapping
        return obj

    def forward(self, token_ids: Tensor) -> Tensor:
        """Compress token IDs using the normalisation lookup table.

        Args:
            token_ids: Integer tensor of shape [B, S] (batch × sequence).

        Returns:
            Compressed token IDs of shape [B, S], dtype=torch.long.
        """
        return self.mapping[token_ids]


# ---------------------------------------------------------------------------
# N-gram hash mapping
# ---------------------------------------------------------------------------

class NgramHashMapping(nn.Module):
    """Deterministic N-gram hashing using XOR-mix hash functions.

    For each N-gram order k (2 ≤ k ≤ max_ngram_size) and each hash head h,
    the hash is:

        hash = (t_0 * m_{k,h,0}) XOR (t_1 * m_{k,h,1}) XOR ... XOR (t_{k-1} * m_{k,h,k-1})
        index = hash mod prime_k

    where the multipliers m are derived from a seeded RNG specific to the
    (layer_id, k, h) triple, ensuring cross-layer independence.

    The final index into the embedding table is offset to account for all
    smaller N-gram tables laid out contiguously in MultiHeadEmbedding.

    Args:
        config:    EngramConfig instance.
        layer_id:  0-indexed layer number (used to seed layer-specific hashes).
    """

    # Small primes used as hash table sizes per N-gram order.
    # These are chosen to be large enough for diversity but manageable in memory.
    # In the paper, the sizes are vocab_size * 5 per head; here we use fixed
    # defaults that can be overridden.
    _DEFAULT_PRIMES = {2: 999983, 3: 1999993, 4: 3999971}

    def __init__(self, config: EngramConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        # Determine prime hash-table size per N-gram order
        self.primes: List[int] = [
            config.n_embed_per_ngram
            for k in range(2, config.max_ngram_size + 1)
        ]

        # Precompute cumulative table offsets for MultiHeadEmbedding indexing
        # offsets[i] = total heads × rows before N-gram order (i+2)
        self.vocab_sizes: List[int] = [p * config.n_head_per_ngram for p in self.primes]
        offsets = [0]
        for vs in self.vocab_sizes[:-1]:
            offsets.append(offsets[-1] + vs)
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))

        # Precompute per-order head offsets: [num_orders, n_head_per_ngram]
        # Avoids `torch.arange(...) * prime` allocation on every forward call.
        all_ho = []
        for o_idx, prime in enumerate(self.primes):
            ho = offsets[o_idx] + torch.arange(config.n_head_per_ngram, dtype=torch.long) * prime
            all_ho.append(ho)
        # persistent=False: このバッファは計算用途のみ。
        # checkpoint への保存・読み込み対象外にする。
        self.register_buffer('head_offsets_per_order', torch.stack(all_ho), persistent=False)  # [num_orders, n_head]

        # Precompute hash multipliers: [num_ngram_orders, n_head, max_ngram_size]
        multipliers = self._build_multipliers(config, layer_id)
        self.register_buffer('multipliers', multipliers)

    @staticmethod
    def _build_multipliers(config: EngramConfig, layer_id: int) -> Tensor:
        """Generate deterministic odd multipliers for each (ngram_order, head, position)."""
        num_orders = config.max_ngram_size - 1  # orders 2..max_ngram_size
        shape = (num_orders, config.n_head_per_ngram, config.max_ngram_size)
        gen = torch.Generator()
        gen.manual_seed(config.seed * 10007 + layer_id * 1009)
        # Use random 32-bit odd integers as multipliers (odd ensures invertibility mod 2^32)
        mults = torch.randint(1, 2**31, shape, generator=gen, dtype=torch.long)
        mults = mults * 2 + 1  # ensure odd
        return mults

    def forward(self, compressed_ids: Tensor) -> Tensor:
        """Compute N-gram hash indices for all orders and heads.

        Args:
            compressed_ids: Compressed token IDs of shape [B, S].

        Returns:
            Hash indices tensor of shape [B, S, total_heads] where
            total_heads = (max_ngram_size - 1) * n_head_per_ngram.
            Values are offsets into the flat MultiHeadEmbedding table.
        """
        B, S = compressed_ids.shape
        device = compressed_ids.device
        all_indices = []

        for order_idx, k in enumerate(range(2, self.config.max_ngram_size + 1)):
            prime = self.primes[order_idx]
            offset = self.offsets[order_idx]  # scalar

            # Build k-gram sequences: ngrams[b, s, i] = compressed_ids[b, s+i]
            # Single pad + unfold replaces k separate F.pad+slice calls
            padded = F.pad(compressed_ids, (0, k - 1), value=0)  # [B, S+k-1]
            ngrams = padded.unfold(1, k, 1)  # [B, S, k]

            # Hash per head: multipliers [n_head, k] (trimmed to k positions)
            mults = self.multipliers[order_idx, :, :k]  # [n_head, k]

            # XOR-mix hash: sum of (token * multiplier) then XOR cascade
            # Broadcast: ngrams [B, S, k] × mults [n_head, k] → [B, S, n_head, k]
            products = ngrams.unsqueeze(2) * mults.unsqueeze(0).unsqueeze(0)  # [B,S,H,k]
            # XOR across token positions
            hash_val = reduce(torch.bitwise_xor, products.unbind(-1))

            # Map to table index with offset for MultiHeadEmbedding flat layout
            head_offset = self.head_offsets_per_order[order_idx]  # [n_head] — precomputed
            indices = (hash_val % prime) + head_offset  # [B, S, n_head]
            all_indices.append(indices)

        # Concatenate all orders: [B, S, total_heads]
        return torch.cat(all_indices, dim=-1)


# ---------------------------------------------------------------------------
# Multi-head embedding
# ---------------------------------------------------------------------------

class MultiHeadEmbedding(nn.Module):
    """A single flat embedding table covering all N-gram orders and hash heads.

    The table layout (in row dimension) is:
        [head_0_ngram2 | head_1_ngram2 | ... | head_0_ngram3 | ...]

    Each row is addressed by the offset indices produced by NgramHashMapping.

    Args:
        total_rows:  Total number of rows = sum(prime_k * n_head for each k).
        embed_dim:   Embedding dimension per row.
    """

    def __init__(self, total_rows: int, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.table = nn.Embedding(total_rows, embed_dim)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, indices: Tensor) -> Tensor:
        """Look up embeddings for N-gram hash indices.

        Args:
            indices: Shape [B, S, total_heads], integer indices into the table.

        Returns:
            Embeddings of shape [B, S, total_heads, embed_dim].
        """
        return self.table(indices)


# ---------------------------------------------------------------------------
# Short convolution
# ---------------------------------------------------------------------------

class ShortConv(nn.Module):
    """1-D depthwise convolution along the sequence dimension to fuse adjacent
    N-gram embeddings (captures local sequential patterns).

    Args:
        channels:    Number of input/output channels (= total_heads * embed_dim
                     after flattening head and embed dims, or embed_dim if heads
                     are averaged first).
        kernel_size: Convolution kernel size (typically 3 or 4).
    """

    def __init__(self, channels: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=kernel_size - 1, groups=channels,  # depthwise
        )
        self._trim = -(kernel_size - 1) if kernel_size > 1 else None

    def forward(self, x: Tensor) -> Tensor:
        """Apply short convolution.

        Args:
            x: Shape [B, S, C].

        Returns:
            Shape [B, S, C] (sequence dimension preserved via padding trim).
        """
        # Conv1d expects [B, C, S]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # Trim extra padding to keep length S
        if self._trim is not None:
            x = x[..., :self._trim]
        return x.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Main Engram module
# ---------------------------------------------------------------------------

class EngramModule(nn.Module):
    """Engram conditional memory module.

    Retrieves static N-gram memory and fuses it with the current hidden states
    via context-aware gating.

    The module is designed to be inserted at specific transformer layers
    (typically early layers) to offload static pattern reconstruction.

    Args:
        config:      EngramConfig with all Engram hyperparameters.
        layer_id:    0-indexed ID of the transformer layer this module belongs to.
        hidden_size: Hidden size D of the transformer (for gating projection).
    """

    def __init__(
        self,
        config: EngramConfig,
        layer_id: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = hidden_size

        # --- Vocabulary compression ---
        self.tokenizer = CompressedTokenizer(config.base_vocab_size, config.seed)

        # --- N-gram hash mapping ---
        self.ngram_hash = NgramHashMapping(config, layer_id)

        # Total rows in the flat embedding table
        total_rows = sum(self.ngram_hash.vocab_sizes)

        # --- Multi-head embedding ---
        self.multi_head_emb = MultiHeadEmbedding(total_rows, config.n_embed_dim)

        # Number of hash heads total: (max_ngram_size - 1) * n_head_per_ngram
        num_total_heads = (config.max_ngram_size - 1) * config.n_head_per_ngram

        # --- Short convolution ---
        # Operate on head-averaged embedding: [B, S, n_embed_dim]
        self.short_conv = ShortConv(channels=config.n_embed_dim, kernel_size=4)

        # --- Head aggregation projection ---
        # Collapse total_heads dimension to a single embedding vector
        self.head_proj = nn.Linear(
            num_total_heads * config.n_embed_dim,
            config.n_embed_dim,
            bias=False,
        )

        # --- Context-aware gating (attention-like) ---
        # gate_proj: hidden_states → gate weights  (D → n_embed_dim)
        self.gate_proj = nn.Linear(hidden_size, config.n_embed_dim, bias=False)

        # --- Output projection: n_embed_dim → hidden_size ---
        self.out_proj = nn.Linear(config.n_embed_dim, hidden_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # zero-init: Engram starts as identity

    def forward(self, input_ids: Tensor, hidden_states: Tensor) -> Tensor:
        """Compute the Engram memory increment.

        Args:
            input_ids:     Token IDs, shape [B, S] (batch × sequence).
            hidden_states: Current hidden states, shape [S, B, D] (Megatron convention).

        Returns:
            Memory increment of shape [S, B, D] to be added to hidden_states.
        """
        # Megatron uses [S, B, D]; convert to [B, S, D] for batch-first ops
        hidden_bsf = hidden_states.permute(1, 0, 2)  # [B, S, D]
        B, S, D = hidden_bsf.shape

        # 1. Compress vocabulary
        compressed = self.tokenizer(input_ids)  # [B, S]

        # 2. Generate N-gram hash indices
        indices = self.ngram_hash(compressed)  # [B, S, total_heads]

        # 3. Look up multi-head embeddings
        emb = self.multi_head_emb(indices)  # [B, S, total_heads, n_embed]

        # 4. Flatten heads and project to single embedding
        emb_flat = emb.view(B, S, -1)       # [B, S, total_heads * n_embed]
        emb = self.head_proj(emb_flat)       # [B, S, n_embed]

        # 5. Short convolution (local sequence fusion)
        emb = self.short_conv(emb)           # [B, S, n_embed]

        # 6. Context-aware gating
        gate = torch.sigmoid(self.gate_proj(hidden_bsf))  # [B, S, n_embed]
        emb = gate * emb                                   # [B, S, n_embed]

        # 7. Project to hidden size
        out = self.out_proj(emb)             # [B, S, D]

        # Convert back to Megatron [S, B, D]
        return out.permute(1, 0, 2)
