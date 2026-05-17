# Copyright (c) 2026 Third Intelligence. All rights reserved.
"""Supervised fine-tuning entry point for Susono.

Mirrors :mod:`pretrain_susono` (same model builder, same checkpointing
wrapper) but swaps in an HF chat-template SFT dataset and a label-masked
cross-entropy loss. The design follows ms-swift's MegatronTrainer
(``swift/megatron/trainers/trainer.py``) so the loss-mask source is purely
``labels != -100``.
"""
import os
from functools import partial

import transformers

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0, pretrain

# Re-use Susono builder and the save-time CPU memory wrapper.
# Importing pretrain_susono installs the save_checkpoint memory-cleanup wrapper
# as a side effect (see pretrain_susono.py:240-258), matching the pretrain run.
from pretrain_susono import susono_builder  # noqa: F401  (side-effects required)
import pretrain_susono  # noqa: F401

from megatron.post_training.susono.arguments import add_susono_sft_args
from megatron.post_training.susono.batch import get_batch
from megatron.post_training.susono.loss import bind_sft_loss
from megatron.post_training.susono.sft_dataset import build_sft_train_valid_test


def model_provider(pre_process=True, post_process=True, config=None, pg_collection=None):
    args = get_args()
    return susono_builder(
        args,
        pre_process=pre_process,
        post_process=post_process,
        config=config,
        pg_collection=pg_collection,
    )


def train_valid_test_datasets_provider(num_samples_train_val_test):
    args = get_args()
    print_rank_0("> building Susono SFT datasets ...")
    tokenizer = get_tokenizer()
    hf_tokenizer = getattr(tokenizer, "_tokenizer", None)
    if not isinstance(hf_tokenizer, transformers.PreTrainedTokenizerBase):
        raise ValueError(
            "Susono SFT requires --tokenizer-type HuggingFaceTokenizer "
            "(underlying object must be transformers.PreTrainedTokenizerBase)."
        )
    if args.micro_batch_size != 1:
        raise ValueError(
            "Susono SFT enforces --micro-batch-size 1 (samples are pre-packed)."
        )

    shard_world = mpu.get_expert_data_parallel_world_size()
    shard_index = mpu.get_expert_data_parallel_rank()
    train_ds, valid_ds, test_ds = build_sft_train_valid_test(
        num_samples_train_val_test,
        args=args,
        tokenizer=hf_tokenizer,
        shard_world=shard_world,
        shard_index=shard_index,
    )
    print_rank_0("> finished building Susono SFT datasets")
    return train_ds, valid_ds, test_ds


def forward_step(data_iterator, model):
    args = get_args()
    timers = get_timers()

    timers("batch-generator", log_level=2).start()
    tokenizer = get_tokenizer()
    eos_id = int(tokenizer._tokenizer.eos_token_id)
    batch = get_batch(
        data_iterator,
        seq_length=args.seq_length,
        eos_token_id=eos_id,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
    )
    timers("batch-generator").stop()

    tokens = batch["tokens"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    position_ids = batch["position_ids"]

    output_tensor = model(
        tokens,
        position_ids,
        attention_mask,
        labels=labels,
    )
    return output_tensor, bind_sft_loss(labels)


if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
        extra_args_provider=add_susono_sft_args,
    )
