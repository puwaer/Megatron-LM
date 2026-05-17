# Copyright (c) 2026 Third Intelligence. All rights reserved.
"""DPO entry point for Susono.

Mirrors ms-swift's ``MegatronDPOTrainer.forward_step``
(``swift/megatron/trainers/dpo_trainer.py:77-95``):

    1. Pull a paired ``[2, S+1]`` preference batch.
    2. Run the frozen reference model under ``torch.no_grad()``.
    3. Run the policy model with gradients enabled.
    4. Concatenate ``[ref_out; policy_out]`` along the batch dim and hand
       the joint tensor to the DPO loss for splitting and logit computation.

The reference model is built lazily on first ``forward_step`` call (after
Megatron's distributed init and policy ``setup_model_and_optimizer``), so this
script needs no extra hook into Megatron's training pipeline.
"""
import torch
import transformers

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0, pretrain

# Re-use Susono builder and save-time CPU memory wrapper.
from pretrain_susono import susono_builder  # noqa: F401
import pretrain_susono  # noqa: F401

from megatron.post_training.susono.arguments import add_susono_dpo_args
from megatron.post_training.susono.batch import get_dpo_batch
from megatron.post_training.susono.loss import bind_dpo_loss
from megatron.post_training.susono.dpo_dataset import build_dpo_train_valid_test
from megatron.post_training.susono.ref_model import (
    build_and_load_ref_model,
    get_ref_model,
)


_REF_INITIALIZED = False


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
    print_rank_0("> building Susono DPO preference datasets ...")
    tokenizer = get_tokenizer()
    hf_tokenizer = getattr(tokenizer, "_tokenizer", None)
    if not isinstance(hf_tokenizer, transformers.PreTrainedTokenizerBase):
        raise ValueError(
            "Susono DPO requires --tokenizer-type HuggingFaceTokenizer "
            "(underlying object must be transformers.PreTrainedTokenizerBase)."
        )
    if args.micro_batch_size != 1:
        raise ValueError(
            "Susono DPO enforces --micro-batch-size 1 (each sample is a paired tensor)."
        )

    shard_world = mpu.get_expert_data_parallel_world_size()
    shard_index = mpu.get_expert_data_parallel_rank()
    train_ds, valid_ds, test_ds = build_dpo_train_valid_test(
        num_samples_train_val_test,
        args=args,
        tokenizer=hf_tokenizer,
        shard_world=shard_world,
        shard_index=shard_index,
    )
    print_rank_0("> finished building Susono DPO datasets")
    return train_ds, valid_ds, test_ds


def _ensure_ref_initialized():
    global _REF_INITIALIZED
    if _REF_INITIALIZED:
        return
    build_and_load_ref_model(model_provider)
    _REF_INITIALIZED = True


def forward_step(data_iterator, model):
    args = get_args()
    timers = get_timers()

    _ensure_ref_initialized()

    timers("batch-generator", log_level=2).start()
    tokenizer = get_tokenizer()
    eos_id = int(tokenizer._tokenizer.eos_token_id)
    batch = get_dpo_batch(
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

    ref_model = get_ref_model()
    with torch.no_grad():
        ref_out = ref_model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
        )
    policy_out = model(
        tokens,
        position_ids,
        attention_mask,
        labels=labels,
    )

    combined = torch.cat([ref_out, policy_out], dim=0)
    return combined, bind_dpo_loss(
        labels,
        beta=args.dpo_beta,
        label_smoothing=args.dpo_label_smoothing,
        rpo_alpha=args.rpo_alpha,
    )


if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
        extra_args_provider=add_susono_dpo_args,
    )
