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
import warnings
from contextlib import contextmanager
from typing import List, Optional

import torch

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.training import get_model


_REF_MODELS: Optional[List[torch.nn.Module]] = None


class ShardedRefModel(torch.nn.Module):
    """ZeRO-3 style DP sharding for a frozen ref model.

    Each rank holds 1/world_size of every nn.Parameter (where world_size is
    the size of ``shard_group``). Just before forward, params are all-gathered
    into their original shapes; the gathered tensors are released after
    forward returns so the steady-state footprint is the per-rank shard only.
    """

    def __init__(self, ref_model: torch.nn.Module, shard_group):
        super().__init__()
        self._ref = ref_model
        self._group = shard_group
        self._world = torch.distributed.get_world_size(shard_group)
        self._rank = torch.distributed.get_rank(shard_group)
        self._shards = {}
        self._meta = {}
        self._partition()

    def _partition(self) -> None:
        for name, p in self._ref.named_parameters():
            full_numel = p.numel()
            if full_numel == 0:
                continue
            shard_size = (full_numel + self._world - 1) // self._world
            start = self._rank * shard_size
            end = min(start + shard_size, full_numel)
            flat = p.data.contiguous().view(-1)
            shard = torch.empty(shard_size, dtype=p.dtype, device=p.device)
            if end > start:
                shard[: end - start].copy_(flat[start:end])
            if end - start < shard_size:
                shard[end - start :].zero_()
            self._shards[name] = shard
            self._meta[name] = (p.shape, full_numel)
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)

    def _gather(self) -> None:
        for name, p in self._ref.named_parameters():
            if name not in self._meta:
                continue
            shape, full_numel = self._meta[name]
            shard = self._shards[name]
            buf = torch.empty(
                shard.numel() * self._world, dtype=shard.dtype, device=shard.device
            )
            torch.distributed.all_gather_into_tensor(buf, shard, group=self._group)
            p.data = buf[:full_numel].view(shape)

    def _release(self) -> None:
        for _, p in self._ref.named_parameters():
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)

    def forward(self, *args, **kwargs):
        self._gather()
        try:
            with torch.no_grad():
                return self._ref(*args, **kwargs)
        finally:
            self._release()

    def set_input_tensor(self, input_tensor) -> None:
        self._ref.set_input_tensor(input_tensor)


class LayerWiseShardedRefModel(torch.nn.Module):
    """FSDP-style layer-wise sharded ref model.

    Each parameter is sharded across ``shard_group`` and held either on CPU
    (``cpu_offload=True``) or GPU. Forward hooks on every entry of
    ``ref_model.decoder.layers`` all-gather just that layer's params before
    the layer runs and release them right after. Non-layer params (embed,
    final norm, output projection) are gathered once at outer forward entry
    and released at outer exit. Peak gathered GPU memory during ref forward
    is roughly one transformer layer worth (~ full_size / num_layers).
    """

    def __init__(self, ref_model: torch.nn.Module, shard_group, cpu_offload: bool = False):
        super().__init__()
        self._ref = ref_model
        self._group = shard_group
        if shard_group is None:
            self._world = 1
            self._rank = 0
        else:
            self._world = torch.distributed.get_world_size(shard_group)
            self._rank = torch.distributed.get_rank(shard_group)
        self._cpu = cpu_offload
        self._shards = {}    # id(param) -> 1D shard tensor (CPU or GPU)
        self._meta = {}      # id(param) -> (shape, full_numel, shard_size)
        self._partition_all()

        # Unwrap Megatron wrappers (Float16Module is added unconditionally
        # under --bf16/--fp16) to reach the underlying model with decoder.layers.
        inner = ref_model
        for _ in range(5):
            if hasattr(inner, "decoder") and hasattr(getattr(inner, "decoder"), "layers"):
                break
            if hasattr(inner, "module"):
                inner = inner.module
            else:
                break
        assert hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"), (
            "LayerWiseShardedRefModel could not locate decoder.layers on the ref "
            "model (checked up to 5 wrapper levels)."
        )
        self._blocks = list(inner.decoder.layers)
        block_param_ids = {id(p) for blk in self._blocks for p in blk.parameters()}
        self._global_params = [
            p for _, p in ref_model.named_parameters()
            if id(p) in self._meta and id(p) not in block_param_ids
        ]
        self._setup_gather_buffers()
        for block_idx, blk in enumerate(self._blocks):
            blk.register_forward_pre_hook(self._make_pre(block_idx))
            blk.register_forward_hook(self._make_post(block_idx))

    def _partition_all(self) -> None:
        ref_dtype = None
        for _, p in self._ref.named_parameters():
            full_numel = p.numel()
            if full_numel == 0:
                continue
            if ref_dtype is None:
                ref_dtype = p.dtype
            elif ref_dtype != p.dtype:
                raise AssertionError(
                    f"LayerWiseShardedRefModel assumes all ref params share dtype "
                    f"(got {ref_dtype} and {p.dtype}); the shared gather buffer "
                    f"design needs a per-dtype extension to support mixed dtypes."
                )
            shard_size = (full_numel + self._world - 1) // self._world
            start = self._rank * shard_size
            end = min(start + shard_size, full_numel)
            shard = torch.zeros(shard_size, dtype=p.dtype, device="cpu")
            if end > start:
                shard[: end - start].copy_(
                    p.data.detach().contiguous().view(-1)[start:end].cpu()
                )
            if not self._cpu:
                shard = shard.to(p.device)
            else:
                # Page-lock so async H2D copy via NVLink-C2C uses DMA at full
                # bandwidth. Fall back silently to pageable if pin fails (e.g.
                # kernel rejects locking that much memory).
                try:
                    shard = shard.pin_memory()
                except RuntimeError as exc:
                    if self._rank == 0:
                        warnings.warn(
                            f"[Susono DPO] pin_memory() failed on ref shard "
                            f"(numel={shard_size}): {exc}; falling back to "
                            f"pageable memory (H2D will be synchronous)."
                        )
            self._shards[id(p)] = shard
            self._meta[id(p)] = (p.shape, full_numel, shard_size)
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)

        # ``_ref_dtype`` and per-shard sizes are needed later by
        # ``_setup_gather_buffers``; stash them here.
        self._ref_dtype = ref_dtype

    def _setup_gather_buffers(self) -> None:
        """Build per-block / per-global offset layouts and pre-allocate the
        reusable gather buffers.

        Each param gets a distinct offset within its layout so multiple
        params live side-by-side in the buffer rather than overwriting the
        same prefix.
        """
        if not self._meta:
            self._global_buf = None
            self._block_buf = None
            self._h2d_buf = None
            self._global_offsets = {}
            self._block_layouts = [{} for _ in self._blocks]
            return

        device = torch.cuda.current_device()
        ref_dtype = self._ref_dtype

        # Globals: persistent buffer, fixed offsets.
        self._global_offsets = {}
        global_total = 0
        for p in self._global_params:
            shard_size = self._meta[id(p)][2]
            gathered = shard_size * self._world
            self._global_offsets[id(p)] = global_total
            global_total += gathered
        self._global_buf = (
            torch.empty(global_total, dtype=ref_dtype, device=device)
            if global_total > 0 else None
        )

        # Per-block layouts. Block buffer is reused across blocks (post_hook
        # releases all params before the next block's pre_hook overwrites).
        self._block_layouts = []
        max_block_total = 0
        for blk in self._blocks:
            layout = {}
            cur = 0
            for p in blk.parameters():
                if id(p) not in self._meta:
                    continue
                shard_size = self._meta[id(p)][2]
                gathered = shard_size * self._world
                layout[id(p)] = cur
                cur += gathered
            max_block_total = max(max_block_total, cur)
            self._block_layouts.append(layout)
        self._block_buf = (
            torch.empty(max_block_total, dtype=ref_dtype, device=device)
            if max_block_total > 0 else None
        )

        # H2D scratch: reused param-by-param within a block (next H2D is
        # serialized after the previous all_gather on the same CUDA stream).
        max_shard_numel = max(shard_size for _, _, shard_size in self._meta.values())
        self._h2d_buf = (
            torch.empty(max_shard_numel, dtype=ref_dtype, device=device)
            if self._cpu else None
        )

    def _gather_one(self, p, buf, offset_in_buf) -> None:
        if id(p) not in self._meta:
            return
        shape, full_numel, shard_size = self._meta[id(p)]
        shard = self._shards[id(p)]
        if shard.is_cpu:
            # H2D into the pre-allocated scratch (async DMA when shard is pinned).
            h2d_view = self._h2d_buf[:shard_size]
            h2d_view.copy_(shard, non_blocking=True)
            shard_gpu = h2d_view
        else:
            shard_gpu = shard

        gathered_numel = shard_size * self._world
        buf_view = buf[offset_in_buf : offset_in_buf + gathered_numel]
        if self._world == 1:
            buf_view[:shard_size].copy_(shard_gpu)
        else:
            torch.distributed.all_gather_into_tensor(
                buf_view, shard_gpu, group=self._group
            )
        p.data = buf_view[:full_numel].view(shape)

    def _release_one(self, p) -> None:
        if id(p) in self._meta:
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)

    def _make_pre(self, block_idx):
        def hook(module, args):
            layout = self._block_layouts[block_idx]
            for p in self._blocks[block_idx].parameters():
                if id(p) in self._meta:
                    self._gather_one(p, self._block_buf, layout[id(p)])
            return None
        return hook

    def _make_post(self, block_idx):
        def hook(module, args, output):
            for p in self._blocks[block_idx].parameters():
                self._release_one(p)
            return None
        return hook

    def forward(self, *args, **kwargs):
        if self._global_buf is not None:
            for p in self._global_params:
                self._gather_one(p, self._global_buf, self._global_offsets[id(p)])
        try:
            with torch.no_grad():
                return self._ref(*args, **kwargs)
        finally:
            for p in self._global_params:
                self._release_one(p)

    def set_input_tensor(self, input_tensor) -> None:
        self._ref.set_input_tensor(input_tensor)


def _get_ref_shard_group(args):
    """Resolve the ProcessGroup used to shard ref params.

    Returns ``None`` when sharding is disabled (``--ref-shard-size 1``).
    Reuses Megatron's intra-instance DP group when the user-specified shard
    size matches ``DP_SIZE / num_distributed_optimizer_instances``.
    """
    shard_size = args.ref_shard_size
    dp_size = mpu.get_data_parallel_world_size()
    opt_n = getattr(args, "num_distributed_optimizer_instances", 1) or 1
    intra_policy_size = dp_size // opt_n

    if dp_size % shard_size != 0:
        raise ValueError(
            f"DP size {dp_size} must be divisible by --ref-shard-size {shard_size}"
        )

    if shard_size == 1:
        return None
    if shard_size == dp_size:
        return mpu.get_data_parallel_group()
    if shard_size == intra_policy_size and opt_n > 1:
        return mpu.get_data_parallel_group(
            partial_data_parallel=True, with_context_parallel=True
        )

    world_rank = torch.distributed.get_rank()
    dp_global_ranks = torch.distributed.get_process_group_ranks(
        mpu.get_data_parallel_group()
    )
    instance_id = dp_global_ranks.index(world_rank) // shard_size
    start = instance_id * shard_size
    end = start + shard_size
    return torch.distributed.new_group(ranks=dp_global_ranks[start:end])


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

    print_rank_0(
        f"[Susono DPO] Reference model ready: "
        f"{sum(p.numel() for m in ref_model for p in m.parameters())} params, frozen."
    )

    shard_group = _get_ref_shard_group(args)
    cpu_offload = bool(getattr(args, "ref_cpu_offload", False))
    if cpu_offload or shard_group is not None:
        ref_model = [
            LayerWiseShardedRefModel(m, shard_group, cpu_offload=cpu_offload)
            for m in ref_model
        ]
        world = 1 if shard_group is None else torch.distributed.get_world_size(shard_group)
        storage = "CPU" if cpu_offload else "GPU"
        print_rank_0(
            f"[Susono DPO] Ref params layer-wise sharded ({storage}) across {world} "
            f"ranks (ref-shard-size={args.ref_shard_size}, "
            f"ref-cpu-offload={cpu_offload})."
        )
    else:
        print_rank_0("[Susono DPO] Ref params not sharded (full GPU replica per rank).")

    global _REF_MODELS
    _REF_MODELS = ref_model
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
