# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain Susono."""

import os
import torch
from functools import partial
from typing import Union

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.susono.susono_model import SusonoModel
from megatron.core.models.susono.susono_layer_specs import (
    get_susono_layer_with_transformer_engine_spec,
    get_susono_layer_local_spec,
)
from megatron.core.models.engram.engram_module import EngramConfig
from megatron.training import (
    get_args,
    get_timers,
    print_rank_0,
    pretrain
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import get_batch_on_this_tp_rank, is_first_or_last_pipeline_stage, average_losses_across_data_parallel_group
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.utils import StragglerDetector

stimer = StragglerDetector()

def susono_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build the Susono model."""
    print_rank_0('building Susono model ...')

    if config is None:
        config = core_transformer_config_from_args(args)

    # 1. Select the appropriate layer spec
    if args.transformer_impl == 'transformer_engine':
        transformer_layer_spec = get_susono_layer_with_transformer_engine_spec(
            normalization=args.normalization,
            qk_layernorm=args.qk_layernorm,
        )
    else:
        transformer_layer_spec = get_susono_layer_local_spec(
            normalization=args.normalization,
            qk_layernorm=args.qk_layernorm,
        )

    # 2. Configure Engram if enabled
    engram_config = None
    if config.use_engram:
        # Defaults if not specified in args (though usually they should be in config/args)
        # Note: We rely on the config object having these attributes populated from args
        engram_layer_ids_val = getattr(config, 'engram_layer_ids', None)
        engram_config = EngramConfig(
            max_ngram_size=getattr(config, 'engram_max_ngram_size', 3),
            n_embed_per_ngram=getattr(config, 'engram_n_embed_per_ngram', 99991),
            n_embed_dim=getattr(config, 'engram_embed_dim', 672),
            n_head_per_ngram=getattr(config, 'engram_n_head_per_ngram', 8),
            **({"engram_layer_ids": engram_layer_ids_val} if engram_layer_ids_val is not None else {}),
            seed=getattr(config, 'engram_seed', 0),
            base_vocab_size=getattr(config, 'engram_base_vocab_size', args.padded_vocab_size),
        )

    # 3. Instantiate SusonoModel
    model = SusonoModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        engram_config=engram_config,
    )

    return model

def model_provider(pre_process=True, post_process=True, config=None, pg_collection=None):
    """Wrapper for the model builder."""
    args = get_args()
    return susono_builder(
        args,
        pre_process=pre_process,
        post_process=post_process,
        config=config,
        pg_collection=pg_collection,
    )

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
    from megatron.training.utils import get_blend_and_blend_per_split

    tokenizer = build_tokenizer(args)
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    gpt_config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )

    print_rank_0("> building train, validation, and test datasets "
                 "for Susono ...")

    # We use GPTDataset/GPTDatasetConfig as Susono is GPT-compatible for data
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        lambda: is_first_or_last_pipeline_stage(None) and parallel_state.get_tensor_model_parallel_rank() == 0,
        gpt_config
    ).build()

    print_rank_0("> finished creating Susono datasets ...")
    return train_ds, valid_ds, test_ds

def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    # Forward model.
    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)

def get_batch(data_iterator):
    """Generate a batch"""
    if data_iterator is None:
        # Middle pipeline stages receive hidden states via P2P; they don't read data.
        return None, None, None, None, None
    batch = get_batch_on_this_tp_rank(data_iterator)
    return (
        batch['tokens'],
        batch['labels'],
        batch['loss_mask'],
        batch['attention_mask'],
        batch['position_ids'],
    )

def loss_func(loss_mask, output_tensor):
    """Loss function."""
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum().clamp(min=1.0)

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}

def add_susono_args(parser):
    # All Susono-specific arguments (use_mhc, mhc_*, use_engram, engram_*) are already
    # registered automatically via ArgumentGroupFactory(TransformerConfig) in
    # megatron/training/arguments.py. Re-registering them here would cause a conflict.
    return parser

if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_susono_args
    )
