# Copyright (c) 2026 Third Intelligence. All rights reserved.
"""HuggingFace preference dataset for Susono DPO.

Each ``__getitem__`` returns one paired sample of shape ``[2, seq_length + 1]``:
row 0 carries the chosen response, row 1 carries the rejected response. The
prompt portion of each row is masked with ``-100`` so only response tokens
contribute to the DPO log-probabilities.

Supported source schemas (auto-detected):
- ``{"prompt": str, "chosen": str, "rejected": str}`` — ``trl-lib/ultrafeedback_binarized``
- ``{"chosen": [...], "rejected": [...]}`` (OpenAI-style messages list) — e.g. ``HuggingFaceH4/ultrafeedback_binarized``
- ``{"prompt": [...], "chosen": [...], "rejected": [...]}`` (DPO-Mix schemas)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers


def _as_messages(content: Any, role: str) -> List[Dict[str, str]]:
    """Normalise ``content`` (string or list of messages) into chat messages."""
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if isinstance(content, list):
        return [
            {
                "role": m.get("role") or m.get("from") or role,
                "content": m.get("content") or m.get("value") or "",
            }
            for m in content
        ]
    raise TypeError(f"Unsupported message payload of type {type(content)}.")


def _extract_pair(example: Dict[str, Any]) -> Optional[Tuple[List[Dict[str, str]],
                                                              List[Dict[str, str]]]]:
    """Return ``(chosen_messages, rejected_messages)`` or ``None`` if malformed."""
    chosen = example.get("chosen")
    rejected = example.get("rejected")
    if chosen is None or rejected is None:
        return None

    prompt = example.get("prompt")
    if prompt is None:
        # `chosen` / `rejected` are expected to already contain the full dialogue.
        chosen_msgs = _as_messages(chosen, role="assistant")
        rejected_msgs = _as_messages(rejected, role="assistant")
    else:
        prompt_msgs = _as_messages(prompt, role="user")
        chosen_msgs = prompt_msgs + _as_messages(chosen, role="assistant")
        rejected_msgs = prompt_msgs + _as_messages(rejected, role="assistant")

    if not chosen_msgs or not rejected_msgs:
        return None
    return chosen_msgs, rejected_msgs


def _encode_with_mask(messages: List[Dict[str, str]],
                      tokenizer: transformers.PreTrainedTokenizerBase
                      ) -> Tuple[List[int], List[int]]:
    """Tokenize a dialogue and mark assistant spans for loss supervision."""
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    labels = [-100] * len(full_ids)

    prev_len = 0
    for turn_idx in range(len(messages)):
        prefix = messages[: turn_idx + 1]
        prefix_ids = tokenizer.apply_chat_template(
            prefix, tokenize=True, add_generation_prompt=False
        )
        cur_len = len(prefix_ids)
        if cur_len <= prev_len:
            prev_len = cur_len
            continue
        if messages[turn_idx]["role"] == "assistant":
            for j in range(prev_len, cur_len):
                labels[j] = full_ids[j]
        prev_len = cur_len

    if len(full_ids) != len(labels):
        return [], []
    return full_ids, labels


class SusonoDPOPreferenceDataset(torch.utils.data.Dataset):
    """Yield ``(chosen, rejected)`` token / label pairs padded to ``seq_length + 1``."""

    def __init__(
        self,
        num_samples: int,
        hf_dataset: str,
        split: str,
        tokenizer: transformers.PreTrainedTokenizerBase,
        seq_length: int,
        num_shards: int = 1,
        shard_index: int = 0,
    ):
        if not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            raise ValueError("SusonoDPOPreferenceDataset requires an HF tokenizer.")
        if tokenizer.chat_template is None:
            raise ValueError("Tokenizer has no chat_template; cannot encode DPO data.")
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id for DPO padding.")
        if tokenizer.pad_token_id is None:
            # Falling back to eos as pad is the standard HF idiom for causal LMs.
            tokenizer.pad_token_id = tokenizer.eos_token_id

        from datasets import load_dataset

        self.num_samples = num_samples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.num_shards = num_shards
        self.shard_index = shard_index
        self._eos = int(tokenizer.eos_token_id)
        self._pad = int(tokenizer.pad_token_id)
        self._raw = load_dataset(
            hf_dataset,
            split=split,
            token=os.environ.get("HF_TOKEN", None),
        )
        if num_shards > 1:
            self._raw = self._raw.shard(num_shards=num_shards, index=shard_index)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world = 1
            rank = 0
        print(
            f"[SusonoDPOPreferenceDataset] rank {rank}/{world} shard "
            f"{shard_index}/{num_shards} dataset={hf_dataset} split={split} "
            f"raw_pairs={len(self._raw)}",
            flush=True,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.LongTensor]:
        attempts = 0
        max_attempts = max(len(self._raw), 1)
        while attempts < max_attempts:
            raw_idx = (idx + attempts) % len(self._raw)
            pair = _extract_pair(self._raw[raw_idx])
            if pair is None:
                attempts += 1
                continue
            chosen_msgs, rejected_msgs = pair
            c_ids, c_labels = _encode_with_mask(chosen_msgs, self.tokenizer)
            r_ids, r_labels = _encode_with_mask(rejected_msgs, self.tokenizer)
            if not c_ids or not r_ids:
                attempts += 1
                continue

            c_ids, c_labels = self._pad_or_trim(c_ids, c_labels)
            r_ids, r_labels = self._pad_or_trim(r_ids, r_labels)

            input_ids = torch.tensor([c_ids, r_ids], dtype=torch.long)
            labels = torch.tensor([c_labels, r_labels], dtype=torch.long)
            return {"input_ids": input_ids, "labels": labels}
        raise RuntimeError("Exhausted DPO dataset without finding a usable pair.")

    def _pad_or_trim(self, ids: List[int], labels: List[int]) -> Tuple[List[int], List[int]]:
        target = self.seq_length + 1
        if len(ids) >= target:
            return ids[:target], labels[:target]
        pad_len = target - len(ids)
        ids = ids + [self._pad] * pad_len
        labels = labels + [-100] * pad_len
        return ids, labels


def build_dpo_train_valid_test(num_samples_train_val_test, *, args, tokenizer,
                                shard_world, shard_index):
    if args.finetune_hf_dataset is None:
        raise ValueError("--finetune-hf-dataset must point at a preference dataset for DPO.")
    kwargs = dict(
        hf_dataset=args.finetune_hf_dataset,
        split=args.finetune_data_split,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        num_shards=shard_world,
        shard_index=shard_index,
    )
    train_ds = SusonoDPOPreferenceDataset(num_samples_train_val_test[0], **kwargs)
    valid_ds = SusonoDPOPreferenceDataset(max(num_samples_train_val_test[1], 1), **kwargs)
    test_ds = SusonoDPOPreferenceDataset(max(num_samples_train_val_test[2], 1), **kwargs)
    return train_ds, valid_ds, test_ds
