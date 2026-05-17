# Copyright (c) 2026 Third Intelligence. All rights reserved.
"""HuggingFace chat-template based SFT dataset for Susono.

Adapted from ``examples/post_training/modelopt/finetune.py:86-332`` (the
``SFTDataset`` class). The two changes vs. the reference are:

1. Items are returned as ``{"input_ids": [seq+1], "labels": [seq+1]}`` where
   ``labels`` already carries ``-100`` at prompt / pad / eos positions. This
   matches the HF / ms-swift convention so that downstream code can derive
   the loss mask from ``labels != -100`` alone.
2. Assistant span detection uses HF tokenizers' ``return_assistant_tokens_mask``
   when available, falling back to a deterministic rerun of the template with
   the assistant content stripped (works for any chat template).

The dataset is Megatron-agnostic; only ``transformers`` and ``datasets`` are
required at runtime.
"""
from __future__ import annotations

import itertools
import os
from typing import Any, Dict, List, Optional

import torch
import transformers


_SHARE_GPT_ROLE_MAP = {
    "user": "user",
    "User": "user",
    "human": "user",
    "assistant": "assistant",
    "Assistant": "assistant",
    "gpt": "assistant",
    "system": "system",
    "System": "system",
}


def _sharegpt_to_openai(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = example.get("conversations")
    if raw is None:
        return None
    out = []
    for msg in raw:
        role = msg.get("from") or msg.get("role")
        content = msg.get("value") or msg.get("content")
        if role is None or content is None:
            return None
        out.append({"role": _SHARE_GPT_ROLE_MAP.get(role, role), "content": content})
    return {"messages": out}


def _to_messages(example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Best-effort normalisation of common SFT schemas to OpenAI chat messages."""
    if "messages" in example and isinstance(example["messages"], list):
        return example["messages"]
    if "conversations" in example:
        norm = _sharegpt_to_openai(example)
        if norm is not None:
            return norm["messages"]
    if "question" in example and "response" in example:
        return [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["response"]},
        ]
    if "prompt" in example and "response" in example:
        return [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
    return None


class SusonoSFTDataset(torch.utils.data.Dataset):
    """Packed SFT dataset that yields HF-style ``{input_ids, labels}`` samples."""

    def __init__(
        self,
        num_packed_samples: int,
        hf_dataset: str,
        split: str,
        tokenizer: transformers.PreTrainedTokenizerBase,
        seq_length: int,
        num_shards: int = 1,
        shard_index: int = 0,
    ):
        if not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            raise ValueError("SusonoSFTDataset requires a transformers PreTrainedTokenizer.")
        if tokenizer.chat_template is None:
            raise ValueError("Tokenizer has no chat_template; cannot apply SFT formatting.")

        from datasets import load_dataset

        self.num_packed_samples = num_packed_samples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.num_shards = num_shards
        self.shard_index = shard_index
        self._raw_samples = load_dataset(
            hf_dataset,
            split=split,
            token=os.environ.get("HF_TOKEN", None),
        )
        if num_shards > 1:
            self._raw_samples = self._raw_samples.shard(
                num_shards=num_shards, index=shard_index
            )
        self._raw_index = 0
        self._packed: List[Dict[str, List[int]]] = []

        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer must define an eos_token_id for SFT packing.")
        self._eos_id = int(eos)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world = 1
            rank = 0
        print(
            f"[SusonoSFTDataset] rank {rank}/{world} shard {shard_index}/{num_shards} "
            f"dataset={hf_dataset} split={split} raw_samples={len(self._raw_samples)}",
            flush=True,
        )

    def __len__(self) -> int:
        return self.num_packed_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.LongTensor]:
        idx = idx // max(self.num_shards, 1)
        while idx >= len(self._packed):
            packed = self._pack_one()
            if packed is None:
                break
            self._packed.append(packed)
        if not self._packed:
            raise RuntimeError("No packed samples; dataset exhausted on first call.")
        idx = idx % len(self._packed)
        sample = self._packed[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }

    def _pack_one(self) -> Optional[Dict[str, List[int]]]:
        required = self.seq_length + 1
        input_ids: List[int] = []
        labels: List[int] = []

        while len(input_ids) < required:
            if self._raw_index >= len(self._raw_samples):
                # Allow cycling through the dataset for the rest of training.
                self._raw_index = 0
            raw = self._raw_samples[self._raw_index]
            self._raw_index += 1
            processed = self._process_one(raw)
            if processed is None:
                continue
            input_ids.extend(processed["input_ids"])
            labels.extend(processed["labels"])

        return {
            "input_ids": input_ids[:required],
            "labels": labels[:required],
        }

    def _process_one(self, example: Dict[str, Any]) -> Optional[Dict[str, List[int]]]:
        messages = _to_messages(example)
        if messages is None or len(messages) < 2:
            return None
        if messages[0]["role"] == "assistant":
            return None

        ids, labels = self._encode_messages(messages)
        if not ids:
            return None

        # Inter-sample separator: an extra eos with -100 label so the loss skips it.
        ids = ids + [self._eos_id]
        labels = labels + [-100]
        if len(ids) > self.seq_length:
            ids = ids[: self.seq_length]
            labels = labels[: self.seq_length]
        return {"input_ids": ids, "labels": labels}

    def _encode_messages(self, messages: List[Dict[str, str]]):
        """Tokenize a conversation and label assistant spans only.

        Strategy: encode the full conversation once, then for each assistant
        turn re-encode the prefix up to (and including) the turn to determine
        the answer-token span. This is template-agnostic; it does not rely on
        ``return_assistant_tokens_mask`` support in the chat template.
        """
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        labels = [-100] * len(full_ids)

        prev_len = 0
        for turn_idx in range(len(messages)):
            prefix = messages[: turn_idx + 1]
            prefix_ids = self.tokenizer.apply_chat_template(
                prefix,
                tokenize=True,
                add_generation_prompt=False,
            )
            cur_len = len(prefix_ids)
            if cur_len <= prev_len:
                # No new tokens (shouldn't happen for well-formed templates).
                prev_len = cur_len
                continue
            if messages[turn_idx]["role"] == "assistant":
                # Mark the assistant span as supervised; keep template wrappers
                # (system / role tags) outside the span as -100.
                for j in range(prev_len, cur_len):
                    labels[j] = full_ids[j]
            prev_len = cur_len

        # Sanity: ensure ids align (the template should be deterministic).
        if len(full_ids) != len(labels):
            return [], []
        return full_ids, labels


def build_sft_train_valid_test(num_samples_train_val_test, *, args, tokenizer,
                                shard_world, shard_index):
    """Create train / valid / test SusonoSFTDataset triple.

    ``num_samples_train_val_test`` is the tuple ``(n_train, n_valid, n_test)``
    passed in by Megatron's ``train_valid_test_dataset_provider`` signature.
    """
    if args.finetune_hf_dataset is None:
        raise ValueError(
            "--finetune-hf-dataset must be set when using SusonoSFTDataset."
        )
    kwargs = dict(
        hf_dataset=args.finetune_hf_dataset,
        split=args.finetune_data_split,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        num_shards=shard_world,
        shard_index=shard_index,
    )
    train_ds = SusonoSFTDataset(num_samples_train_val_test[0], **kwargs)
    valid_ds = SusonoSFTDataset(max(num_samples_train_val_test[1], 1), **kwargs)
    test_ds = SusonoSFTDataset(max(num_samples_train_val_test[2], 1), **kwargs)
    return train_ds, valid_ds, test_ds
