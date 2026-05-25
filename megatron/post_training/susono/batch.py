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
    """Pull one preference micro-batch and prepare the ``[2*mbs, S]`` input.

    Each item from the DPO dataset is a paired tensor of shape
    ``[2, seq_length + 1]`` where row 0 is the chosen and row 1 is the
    rejected. With ``micro_batch_size = mbs`` the broadcast tensor is
    ``[mbs, 2, S+1]``; we transpose the leading two dims and flatten so
    the first ``mbs`` rows are all chosen and the next ``mbs`` are all
    rejected — matching the layout that :func:`dpo_loss_func` expects
    (see ``loss.py:49-62``).
    """
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = _broadcast_to_tp(data, seq_length)

    ids = data_b["input_ids"]
    lbl = data_b["labels"]
    if ids.dim() == 3:
        # ``[mbs, 2, S+1]`` -> ``[2, mbs, S+1]`` -> ``[2*mbs, S+1]``.
        # The naive ``ids[0]`` we used to do here silently dropped every
        # sample beyond index 0 once mbs>1, making mbs=2/4 train on only
        # 1/mbs of the samples actually consumed by the dataloader.
        mbs, two, sp1 = ids.shape
        assert two == 2, (
            f"Expected paired DPO sample [mbs, 2, S+1], got {tuple(ids.shape)}"
        )
        ids = ids.transpose(0, 1).contiguous().view(2 * mbs, sp1)
        lbl = lbl.transpose(0, 1).contiguous().view(2 * mbs, sp1)

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
