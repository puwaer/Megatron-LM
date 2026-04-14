# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Susono architecture: mHC (Manifold-Constrained Hyper-Connections) + Engram."""

from megatron.core.models.susono.susono_block import SusonoBlock
from megatron.core.models.susono.susono_layer_specs import (
    get_susono_layer_local_spec,
    get_susono_layer_with_transformer_engine_spec,
)
from megatron.core.models.susono.susono_model import SusonoModel, build_susono_model
from megatron.core.models.susono.mhc_transformer_layer import MHCTransformerLayer

__all__ = [
    "SusonoModel",
    "SusonoBlock",
    "MHCTransformerLayer",
    "build_susono_model",
    "get_susono_layer_local_spec",
    "get_susono_layer_with_transformer_engine_spec",
]
