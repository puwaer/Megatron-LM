# Copyright (c) 2026 Third Intelligence. All rights reserved.
"""Batch construction for Susono SFT / DPO.

Mirrors the design of ms-swift's `MegatronTrainer.forward_step` data plumbing,
but cut down to Susono's current operating point (CP=1, PP=1, non-padding-free).

Input dataset items are expected to already contain HF-style labels with
`-100` at positions that should not contribute to the loss (prompt / pad tokens).
The first token's label is dropped so that `labels[t]` aligns with `logits[t]`
(predicting position `t+1` from `tokens[:t+1]`).
"""
from typing import Dict, Optional

import torch

from megatron.core import tensor_parallel
from megatron.training.utils import get_ltor_masks_and_position_ids


def _broadcast_to_tp(data, seq_length, dtype=torch.int64):
    keys = ["input_ids", "labels"]
    return tensor_parallel.broadcast_data(keys, data, dtype)


def get_batch(data_iterator, *, seq_length: int, eos_token_id: int,
              reset_position_ids: bool = True,
              reset_attention_mask: bool = True) -> Optional[Dict[str, torch.Tensor]]:
    """Pull one micro-batch from the iterator and prepare model inputs.

    Returns ``None`` on middle pipeline stages (they receive activations via P2P
    instead of reading data). The returned ``labels`` already carry the
    HF-style ``-100`` ignore index for non-loss tokens.
    """
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = _broadcast_to_tp(data, seq_length)

    # Shift: tokens predict labels[1:] from tokens[:-1].
    tokens = data_b["input_ids"][:, : seq_length].contiguous()
    labels = data_b["labels"][:, 1 : seq_length + 1].contiguous()

    attention_mask, _loss_mask_unused, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        eos_token_id,
        eos_token_id,
        reset_position_ids,
        reset_attention_mask,
        eod_mask_loss=False,
        pad_mask_loss=False,
    )

    return {
        "tokens": tokens.contiguous(),
        "labels": labels.contiguous(),
        "position_ids": position_ids.contiguous(),
        "attention_mask": attention_mask,
    }


def get_dpo_batch(data_iterator, *, seq_length: int, eos_token_id: int,
                  reset_position_ids: bool = True,
                  reset_attention_mask: bool = True) -> Optional[Dict[str, torch.Tensor]]:
    """Pull one preference pair and prepare the [2, S] model input.

    Each item from the DPO dataset is a paired tensor of shape
    ``[2, seq_length + 1]`` where row 0 is the chosen and row 1 is the
    rejected. With ``micro_batch_size = 1`` (enforced by the entry script)
    the broadcast tensor is ``[1, 2, S+1]``; we squeeze the leading mbs
    dimension so the effective per-step batch fed to the model is
    ``[2, S]``. The DPO loss is responsible for chunking ref vs. policy
    after the forward.
    """
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = _broadcast_to_tp(data, seq_length)

    # Expected mbs == 1. Drop the outer (mbs) dim and keep the [2, S+1] pair.
    ids = data_b["input_ids"]
    if ids.dim() == 3:
        ids = ids[0]
    lbl = data_b["labels"]
    if lbl.dim() == 3:
        lbl = lbl[0]

    tokens = ids[:, : seq_length].contiguous()
    labels = lbl[:, 1 : seq_length + 1].contiguous()

    attention_mask, _loss_mask_unused, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        eos_token_id,
        eos_token_id,
        reset_position_ids,
        reset_attention_mask,
        eod_mask_loss=False,
        pad_mask_loss=False,
    )

    return {
        "tokens": tokens.contiguous(),
        "labels": labels.contiguous(),
        "position_ids": position_ids.contiguous(),
        "attention_mask": attention_mask,
    }
