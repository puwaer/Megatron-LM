# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fuji architecture: mHC (Manifold-Constrained Hyper-Connections) + Engram."""

from megatron.core.models.fuji.fuji_block import FujiBlock
from megatron.core.models.fuji.fuji_layer_specs import (
    get_fuji_layer_local_spec,
    get_fuji_layer_with_transformer_engine_spec,
)
from megatron.core.models.fuji.fuji_model import FujiModel, build_fuji_model
from megatron.core.models.fuji.mhc_transformer_layer import MHCTransformerLayer

__all__ = [
    "FujiModel",
    "FujiBlock",
    "MHCTransformerLayer",
    "build_fuji_model",
    "get_fuji_layer_local_spec",
    "get_fuji_layer_with_transformer_engine_spec",
]
