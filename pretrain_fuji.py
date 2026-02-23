# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain Fuji."""

import os
import torch
from functools import partial
from typing import Union

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.fuji.fuji_model import FujiModel
from megatron.core.models.fuji.fuji_layer_specs import (
    get_fuji_layer_with_transformer_engine_spec,
    get_fuji_layer_local_spec,
)
from megatron.core.models.engram.engram_module import EngramConfig
from megatron.training import (
    get_args,
    get_timers,
    print_rank_0,
    pretrain
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.utils import StragglerDetector

stimer = StragglerDetector()

def fuji_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build the Fuji model."""
    print_rank_0('building Fuji model ...')

    if config is None:
        config = core_transformer_config_from_args(args)

    # 1. Select the appropriate layer spec
    if args.transformer_impl == 'transformer_engine':
        transformer_layer_spec = get_fuji_layer_with_transformer_engine_spec(
            normalization=args.normalization,
            qk_layernorm=args.qk_layernorm,
        )
    else:
        transformer_layer_spec = get_fuji_layer_local_spec(
            normalization=args.normalization,
            qk_layernorm=args.qk_layernorm,
        )

    # 2. Configure Engram if enabled
    engram_config = None
    if config.use_engram:
        # Defaults if not specified in args (though usually they should be in config/args)
        # Note: We rely on the config object having these attributes populated from args
        engram_config = EngramConfig(
            max_ngram_size=getattr(config, 'engram_max_ngram_size', 3),
            n_embed_per_ngram=getattr(config, 'engram_n_embed_per_ngram', 512),
            n_head_per_ngram=getattr(config, 'engram_n_head_per_ngram', 8),
            engram_layer_ids=getattr(config, 'engram_layer_ids', None), # Model handles None default
            seed=getattr(config, 'engram_seed', 0),
            base_vocab_size=getattr(config, 'engram_base_vocab_size', args.padded_vocab_size),
        )

    # 3. Instantiate FujiModel
    model = FujiModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=args.share_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        engram_config=engram_config,
    )

    return model

def model_provider(pre_process=True, post_process=True):
    """Wrapper for the model builder."""
    args = get_args()
    return fuji_builder(
        args,
        pre_process=pre_process,
        post_process=post_process,
    )

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    config = core_transformer_config_from_args(args)
    
    print_rank_0("> building train, validation, and test datasets "
                 "for Fuji ...")
    
    # We use GPTDatasetConfig/Builder as Fuji is GPT-compatible for data
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDatasetConfig,
        train_val_test_num_samples,
        lambda: parallel_state.is_pipeline_last_stage(),
        config
    ).build()
    
    print_rank_0("> finished creating Fuji datasets ...")
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
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = parallel_state.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    """Loss function."""
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}

def add_fuji_args(parser):
    group = parser.add_argument_group(title='fuji')

    group.add_argument('--use-mhc', action='store_true',
                       help='Enable mHC (Manifold-Constrained Hyper-Connections).')
    group.add_argument('--mhc-num-streams', type=int, default=4,
                       help='Number of parallel streams for mHC.')
    group.add_argument('--mhc-sinkhorn-iterations', type=int, default=20,
                       help='Number of Sinkhorn iterations for mHC.')

    group.add_argument('--use-engram', action='store_true',
                       help='Enable Engram specific layers.')
    group.add_argument('--engram-max-ngram-size', type=int, default=3,
                       help='Maximum n-gram size for Engram.')
    group.add_argument('--engram-n-embed-per-ngram', type=int, default=512,
                       help='Embedding size per n-gram for Engram.')
    group.add_argument('--engram-n-head-per-ngram', type=int, default=8,
                       help='Number of heads per n-gram for Engram.')
    group.add_argument('--engram-layer-ids', nargs='+', type=int, default=None,
                       help='Layer IDs where Engram is enabled.')
    group.add_argument('--engram-seed', type=int, default=0,
                       help='Seed for deterministic Engram hash multiplier generation.')
    group.add_argument('--engram-base-vocab-size', type=int, default=None,
                       help='Vocabulary size used by Engram tokenizer compression. Defaults to padded_vocab_size.')

    return parser

from megatron.training import get_tokenizer
from megatron.core.utils import get_ltor_masks_and_position_ids
from megatron.training.utils import average_losses_across_data_parallel_group

if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_fuji_args
    )
