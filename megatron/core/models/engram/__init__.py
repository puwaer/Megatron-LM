# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Engram: Conditional Memory via Scalable Lookup (arXiv:2601.07372)."""

from megatron.core.models.engram.engram_module import (
    CompressedTokenizer,
    EngramConfig,
    EngramModule,
    MultiHeadEmbedding,
    NgramHashMapping,
    ShortConv,
)

__all__ = [
    "EngramConfig",
    "EngramModule",
    "CompressedTokenizer",
    "NgramHashMapping",
    "MultiHeadEmbedding",
    "ShortConv",
]
