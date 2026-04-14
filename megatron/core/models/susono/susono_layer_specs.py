# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Layer specifications for the Susono architecture (mHC + Engram).

Provides builder functions that return ModuleSpec objects wrapping
MHCTransformerLayer with the appropriate submodule configurations.

The pattern mirrors the existing GPT layer spec builders in
``megatron.core.models.gpt.gpt_layer_specs``, but uses MHCTransformerLayer
as the outer module so that SusonoBlock/TransformerBlock will instantiate
the mHC-capable layer class.

Usage::

    from megatron.core.models.susono.susono_layer_specs import (
        get_susono_layer_local_spec,
        get_susono_layer_with_transformer_engine_spec,
    )

    # Local (Megatron-Core native, no Transformer Engine required)
    spec = get_susono_layer_local_spec()

    # Transformer Engine (recommended for FP8 training)
    spec = get_susono_layer_with_transformer_engine_spec()
"""

import warnings
from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.models.susono.mhc_transformer_layer import MHCTransformerLayer
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

try:
    import transformer_engine as te  # type: ignore[import-untyped]  # noqa: F401

    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # type: ignore[import-untyped]  # noqa: F401

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    LNImpl = WrappedTorchNorm
    HAVE_APEX = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_local_submodules(
    backend: BackendSpecProvider,
    normalization: str = "LayerNorm",
    qk_layernorm: bool = False,
) -> TransformerLayerSubmodules:
    """Build TransformerLayerSubmodules using the given backend."""
    rms_norm = normalization == "RMSNorm"
    layer_norm = backend.layer_norm(rms_norm=rms_norm, for_qk=False)
    qk_norm = backend.layer_norm(rms_norm=rms_norm, for_qk=True)

    return TransformerLayerSubmodules(
        input_layernorm=layer_norm,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_linear(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
                q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                k_layernorm=qk_norm if qk_layernorm else IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=layer_norm,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=backend.column_parallel_linear(),
                linear_fc2=backend.row_parallel_linear(),
            ),
        ),
        mlp_bda=get_bias_dropout_add,
    )


# ---------------------------------------------------------------------------
# Public spec builders
# ---------------------------------------------------------------------------

def get_susono_layer_local_spec(
    normalization: str = "LayerNorm",
    qk_layernorm: bool = False,
) -> ModuleSpec:
    """Return a ModuleSpec for MHCTransformerLayer using Megatron-Core native modules.

    This spec does NOT require Transformer Engine and is suitable for prototyping
    and for hardware without TE support.

    Args:
        normalization: ``"LayerNorm"`` or ``"RMSNorm"``.
        qk_layernorm:  Apply layer-norm to queries and keys.

    Returns:
        ``ModuleSpec(module=MHCTransformerLayer, submodules=...)``
    """
    backend = LocalSpecProvider()
    submodules = _build_local_submodules(
        backend=backend,
        normalization=normalization,
        qk_layernorm=qk_layernorm,
    )
    return ModuleSpec(module=MHCTransformerLayer, submodules=submodules)


def get_susono_layer_with_transformer_engine_spec(
    normalization: str = "LayerNorm",
    qk_layernorm: bool = False,
    fp8: Optional[str] = None,
) -> ModuleSpec:
    """Return a ModuleSpec for MHCTransformerLayer backed by Transformer Engine.

    This spec is recommended for FP8 training and production use.

    Args:
        normalization: ``"LayerNorm"`` or ``"RMSNorm"``.
        qk_layernorm:  Apply layer-norm to queries and keys.
        fp8:           Deprecated, kept for compatibility.

    Returns:
        ``ModuleSpec(module=MHCTransformerLayer, submodules=...)``
    """
    assert HAVE_TE, (
        "get_susono_layer_with_transformer_engine_spec requires TransformerEngine. "
        "Install it or use get_susono_layer_local_spec() instead."
    )
    if fp8 is not None:
        warnings.warn(
            "The fp8 argument in get_susono_layer_with_transformer_engine_spec is deprecated."
        )
    backend = TESpecProvider()
    rms_norm = normalization == "RMSNorm"
    layer_norm = backend.layer_norm(rms_norm=rms_norm, for_qk=False)
    qk_norm = backend.layer_norm(rms_norm=rms_norm, for_qk=True)

    submodules = TransformerLayerSubmodules(
        input_layernorm=layer_norm,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_layer_norm_linear(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
                q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                k_layernorm=qk_norm if qk_layernorm else IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=layer_norm,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=backend.column_parallel_linear(),
                linear_fc2=backend.row_parallel_linear(),
            ),
        ),
        mlp_bda=get_bias_dropout_add,
    )
    return ModuleSpec(module=MHCTransformerLayer, submodules=submodules)
