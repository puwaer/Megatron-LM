# Copyright (c) 2026 Third Intelligence. All rights reserved.
"""Reference-model lifecycle helpers for Susono DPO.

The ref model shares the exact same builder / spec / config as the policy model
(``pretrain_susono.susono_builder``), is loaded from ``args.ref_load`` via the
standard Megatron distributed checkpoint loader, is frozen, and is held as a
module-level singleton accessed from the DPO ``forward_step``.

This mirrors the design in ms-swift's ``MegatronRLHFTrainer.prepare_model`` /
``null_ref_context`` (``swift/megatron/trainers/rlhf_mixin.py:29-70``) but
without depending on swift's bridge or its trainer harness.
"""
from contextlib import contextmanager
from typing import List, Optional

import torch

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.training import get_model


_REF_MODELS: Optional[List[torch.nn.Module]] = None


def build_and_load_ref_model(model_provider) -> List[torch.nn.Module]:
    """Build the reference model and load weights from ``args.ref_load``.

    The returned list mirrors the per-virtual-pipeline layout that Megatron
    expects (one entry per VP rank; for VP=1 it is a single-element list).
    """
    args = get_args()
    if args.ref_load is None:
        raise ValueError(
            "DPO requires --ref-load <checkpoint dir> pointing at the frozen "
            "reference model (typically the SFT-stage output)."
        )

    print_rank_0(f"[Susono DPO] Building reference model from {args.ref_load}")
    ref_model = get_model(
        model_provider,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
    )
    if not isinstance(ref_model, list):
        ref_model = [ref_model]

    # Use the standard distributed-checkpoint loader, but force "weights only"
    # semantics: no optimizer state, no RNG state. We swap a few args
    # transiently so Megatron's loader takes the "finetune" path internally.
    saved_load = args.load
    saved_no_load_optim = getattr(args, "no_load_optim", False)
    saved_no_load_rng = getattr(args, "no_load_rng", False)
    saved_finetune = getattr(args, "finetune", False)
    saved_pretrained = getattr(args, "pretrained_checkpoint", None)
    try:
        args.no_load_optim = True
        args.no_load_rng = True
        args.finetune = True
        args.pretrained_checkpoint = None
        load_checkpoint(
            ref_model,
            optimizer=None,
            opt_param_scheduler=None,
            load_arg="ref_load",
            strict=True,
        )
    finally:
        args.no_load_optim = saved_no_load_optim
        args.no_load_rng = saved_no_load_rng
        args.finetune = saved_finetune
        args.pretrained_checkpoint = saved_pretrained
        args.load = saved_load

    for module in ref_model:
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)

    global _REF_MODELS
    _REF_MODELS = ref_model
    print_rank_0(
        f"[Susono DPO] Reference model ready: "
        f"{sum(p.numel() for m in ref_model for p in m.parameters())} params, frozen."
    )
    return ref_model


def get_ref_model() -> torch.nn.Module:
    """Return the ref model for the current VP stage (defaults to index 0)."""
    if _REF_MODELS is None:
        raise RuntimeError(
            "Reference model has not been built. Call build_and_load_ref_model() "
            "after Megatron initialization but before the first forward_step."
        )
    vp_stage = mpu.get_virtual_pipeline_model_parallel_rank() or 0
    if vp_stage >= len(_REF_MODELS):
        vp_stage = 0
    return _REF_MODELS[vp_stage]


@contextmanager
def null_ref_context():
    """Context manager that yields the ref-model list.

    Provided for API parity with ms-swift's MegatronRLHFTrainer; for the
    Susono ``tuner_type='full'`` path this is just a nullcontext over the
    pre-built ref models.
    """
    if _REF_MODELS is None:
        raise RuntimeError("Reference model not initialized.")
    yield _REF_MODELS
