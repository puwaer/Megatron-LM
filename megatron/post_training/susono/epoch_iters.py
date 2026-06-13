# Copyright (c) 2026 Third Intelligence. All rights reserved.
"""Derive ``--train-iters`` for ONE full pass over a pre-tokenized post dataset.

Why this lives here
-------------------
The Susono SFT / DPO runtime datasets size themselves purely by
``num_samples = train_iters * global_batch_size`` and *cycle* the underlying
rows once exhausted:

  * ``SusonoSFTDataset._pack_one``              -> ``self._raw_index = 0`` on overflow
  * ``SusonoDPOPreferenceDataset.__getitem__``  -> ``idx % len(self._raw)``

So ``train_iters`` alone decides how many epochs are actually seen — a hardcoded
cap that exceeds one epoch silently re-trains the data N times, and one too small
never reaches the end. To train on ALL data exactly ~once we derive ``train_iters``
from the real on-disk dataset size, keeping the epoch math next to the dataset
classes whose packing / cycling semantics it depends on.

  DPO (1 preference pair == 1 sample, no packing):
      units       = num_rows
  SFT (rows packed into ``seq_length``-token sequences):
      total_tok   = sum(len(input_ids)) + num_rows   # +1 inter-sample eos each
      units       = ceil(total_tok / seq_length)     # packed samples per epoch

  train_iters = max(ceil(units / global_batch_size) * num_epochs, min_iters)

The whole (un-sharded) dataset is counted: ``train_iters * global_batch_size`` is
the GLOBAL sample count, which the runtime dataset then shards across DP ranks.

Token totals are summed from the Arrow list-offset buffers only
(``pyarrow.compute.list_value_length``), so multi-GB datasets are scanned in a
few seconds without materialising the token values. The dataset is only
memory-mapped (no CUDA), so this is safe to call before distributed init.
"""
from __future__ import annotations

import math


def _load_split(dataset_path: str, split: str):
    from datasets import DatasetDict, load_from_disk

    ds = load_from_disk(dataset_path)
    if isinstance(ds, DatasetDict):
        ds = ds[split]
    return ds


def _sft_units(ds, seq_length: int) -> int:
    """Packed-sample count for one epoch = ceil(total_tokens / seq_length)."""
    import pyarrow.compute as pc

    col = ds.data.column("input_ids")
    chunks = col.chunks if hasattr(col, "chunks") else [col]
    total = 0
    for chunk in chunks:
        total += int(pc.sum(pc.list_value_length(chunk)).as_py() or 0)
    # +1 inter-sample eos separator per raw sample (sft_dataset._process_one).
    total += ds.num_rows
    return math.ceil(total / seq_length)


def dataset_units(kind: str, dataset_path: str, split: str, seq_length: int) -> int:
    """Number of training units in one epoch (packed samples for SFT, rows for DPO)."""
    ds = _load_split(dataset_path, split)
    if kind == "dpo":
        return ds.num_rows
    if kind == "sft":
        return _sft_units(ds, seq_length)
    raise ValueError(f"unknown kind {kind!r} (expected 'sft' or 'dpo')")


def compute_train_iters(
    kind: str,
    dataset_path: str,
    split: str,
    global_batch_size: int,
    seq_length: int,
    num_epochs: int,
    min_iters: int = 50,
) -> int:
    """train_iters for ``num_epochs`` full passes over ``dataset_path``."""
    units = dataset_units(kind, dataset_path, split, seq_length)
    return max(math.ceil(units / global_batch_size) * num_epochs, min_iters)


def compute_wsd_decay_iters(train_iters: int, fraction: float) -> int:
    """WSD annealing length = ``fraction`` of the run, clamped to [1, train_iters].

    Megatron only accepts an absolute ``--lr-wsd-decay-iters``; there is no
    fraction flag, so we resolve it here once ``train_iters`` is known.
    """
    return max(1, min(train_iters, math.ceil(train_iters * fraction)))


def resolve_iter_defaults(kind: str) -> dict:
    """Compute ``args_defaults`` for ``pretrain()`` from the real dataset size.

    Reads the relevant flags off ``sys.argv`` (the rest are ignored), and — when
    ``--train-iters`` is NOT given on the command line — returns a defaults dict
    ``{"train_iters": N[, "lr_wsd_decay_iters": M]}``. Megatron's ``validate_args``
    applies these only to args the user did not pass, so an explicit ``--train-iters``
    always wins (the escape hatch). Returns ``{}`` when nothing should be injected.

    Deterministic given the dataset, so every rank produces identical values
    (required for Megatron's cross-rank argument consistency).
    """
    import argparse
    import os
    import sys

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--finetune-hf-dataset", type=str, default=None)
    p.add_argument("--finetune-data-split", type=str, default="train")
    p.add_argument("--global-batch-size", type=int, default=None)
    p.add_argument("--seq-length", type=int, default=None)
    p.add_argument("--finetune-num-epochs", type=int, default=1)
    p.add_argument("--lr-wsd-decay-fraction", type=float, default=0.0)
    p.add_argument("--train-iters", type=int, default=None)
    p.add_argument("--lr-wsd-decay-iters", type=int, default=None)
    a, _ = p.parse_known_args(sys.argv[1:])

    # Respect a manually pinned --train-iters; inject nothing.
    if a.train_iters is not None:
        return {}
    if not a.finetune_hf_dataset or a.global_batch_size is None or a.seq_length is None:
        # Misconfigured; let Megatron raise its own "train_iters required" error.
        return {}

    train_iters = compute_train_iters(
        kind=kind,
        dataset_path=a.finetune_hf_dataset,
        split=a.finetune_data_split,
        global_batch_size=a.global_batch_size,
        seq_length=a.seq_length,
        num_epochs=a.finetune_num_epochs,
    )
    defaults = {"train_iters": train_iters}
    if a.lr_wsd_decay_iters is None and a.lr_wsd_decay_fraction > 0.0:
        defaults["lr_wsd_decay_iters"] = compute_wsd_decay_iters(
            train_iters, a.lr_wsd_decay_fraction
        )

    if os.environ.get("RANK", "0") == "0":
        print(
            f"[epoch_iters] kind={kind} dataset={a.finetune_hf_dataset} "
            f"gbs={a.global_batch_size} seq={a.seq_length} "
            f"epochs={a.finetune_num_epochs} -> {defaults}",
            file=sys.stderr,
            flush=True,
        )
    return defaults
