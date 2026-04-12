# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain Fuji."""

import torch
from functools import partial

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.fuji.fuji_model import FujiModel
from megatron.core.models.fuji.fuji_layer_specs import (
    get_fuji_layer_with_transformer_engine_spec,
    get_fuji_layer_local_spec,
)
from megatron.core.models.engram.engram_module import EngramConfig
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.training import (
    get_args,
    get_timers,
    inprocess_restart,
    print_rank_0,
    pretrain,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)

stimer = StragglerDetector()


def _apply_fuji_args_to_config(args, config):
    """Directly set Fuji-specific arguments to the TransformerConfig object.

    Since core_transformer_config_from_args() only targets the fields of
    TransformerConfig, Fuji-specific attributes (mHC / Engram / hybrid)
    must be attached manually.
    """
    # mHC (Manifold-Constrained Hyper-Connections)
    config.use_mhc                   = getattr(args, 'use_mhc', False)
    config.mhc_num_streams           = getattr(args, 'mhc_num_streams', 4)
    config.mhc_selective_recompute   = getattr(args, 'mhc_selective_recompute', False)
    config.mhc_use_fused_kernel      = getattr(args, 'mhc_use_fused_kernel', False)
    config.mhc_async_pp_overlap      = getattr(args, 'mhc_async_pp_overlap', False)
    config.mhc_sinkhorn_iterations   = getattr(args, 'mhc_sinkhorn_iterations', 20)
    config.mhc_auto_recompute_num_layers = getattr(args, 'mhc_auto_recompute_num_layers', False)

    # Engram (Conditional Memory)
    config.use_engram                = getattr(args, 'use_engram', False)
    config.engram_max_ngram_size     = getattr(args, 'engram_max_ngram_size', 3)
    config.engram_n_embed_per_ngram  = getattr(args, 'engram_n_embed_per_ngram', 99991)
    config.engram_n_head_per_ngram   = getattr(args, 'engram_n_head_per_ngram', 8)
    config.engram_seed               = getattr(args, 'engram_seed', 0)
    config.engram_layer_ids          = getattr(args, 'engram_layer_ids', None)
    config.engram_base_vocab_size    = getattr(args, 'engram_base_vocab_size', None)
    config.engram_embed_dim          = getattr(args, 'engram_embed_dim', 672)

    # Hybrid linear attention variant
    config.experimental_attention_variant = getattr(args, 'experimental_attention_variant', None)

    # Map --linear-attention-freq (Megatron standard argument) to full_attention_interval read by FujiBlock
    lin_freq = getattr(args, 'linear_attention_freq', None)
    if lin_freq is not None:
        config.full_attention_interval = lin_freq if isinstance(lin_freq, int) else 4
    else:
        config.full_attention_interval = getattr(config, 'full_attention_interval', 4)

    return config


def fuji_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build the Fuji model."""
    print_rank_0('building Fuji model ...')

    if config is None:
        config = core_transformer_config_from_args(args)

    # Attach Fuji-specific arguments to config (manual setup as they are outside TransformerConfig fields)
    config = _apply_fuji_args_to_config(args, config)

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
    if getattr(config, 'use_engram', False):
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
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        vp_stage=vp_stage,
        engram_config=engram_config,
    )

    return model


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    """Wrapper for the model builder called by the pretrain framework."""
    args = get_args()
    return fuji_builder(
        args,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build train, valid, and test datasets."""
    args = get_args()

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

    print_rank_0("> building train, validation, and test datasets for Fuji ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        lambda: is_first_or_last_pipeline_stage(vp_stage)
            and parallel_state.get_tensor_model_parallel_rank() == 0,
        gpt_config,
    ).build()

    print_rank_0("> finished creating Fuji datasets ...")
    return train_ds, valid_ds, test_ds


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""
    if not is_first_or_last_pipeline_stage(vp_stage):
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


def loss_func(loss_mask, output_tensor, model=None):
    """Loss function."""
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask)

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # Check for NaN / Inf
    rerun_state_machine = get_rerun_state_machine()
    if get_args().check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator, vp_stage
        )
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(
            tokens, position_ids, attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )

    return output_tensor, partial(loss_func, loss_mask)


def add_fuji_args(parser):
    """Register Fuji-specific arguments.

    Fields that do not exist in Megatron-LM's standard TransformerConfig are added here.
    Note: --linear-attention-type / --linear-attention-freq / --linear-*-head-* / --linear-conv-kernel-dim
    are already registered in Megatron's core _add_linear_attention_args(). Duplicate registration is not allowed.
    """
    group = parser.add_argument_group(title='fuji-mhc',
                                      description='Manifold-Constrained Hyper-Connections (mHC)')
    group.add_argument('--use-mhc', action='store_true', default=False,
                       help='Enable mHC multi-stream residual connections.')
    group.add_argument('--mhc-num-streams', type=int, default=4,
                       help='Number of parallel streams n in mHC.')
    group.add_argument('--mhc-selective-recompute', action='store_true', default=False,
                       help='Discard intermediate tensors in width/depth connections and recompute them during backward pass.')
    group.add_argument('--mhc-use-fused-kernel', action='store_true', default=False,
                       help='Use Triton fused kernels for width/depth connections.')
    group.add_argument('--mhc-async-pp-overlap', action='store_true', default=False,
                       help='Run mHC SM kernels concurrently with NCCL on a dedicated CUDA stream.')
    group.add_argument('--mhc-sinkhorn-iterations', type=int, default=20,
                       help='(Unused) Kept for API compatibility.')
    group.add_argument('--mhc-auto-recompute-num-layers', action='store_true', default=False,
                       help='Automatically calculate L_r* = round(sqrt(n*L/(n+2))) and set it as recompute_num_layers.')

    group = parser.add_argument_group(title='fuji-engram',
                                      description='Engram Conditional Memory')
    group.add_argument('--use-engram', action='store_true', default=False,
                       help='Enable Engram conditional memory.')
    group.add_argument('--engram-max-ngram-size', type=int, default=3,
                       help='Maximum n-gram size for Engram.')
    group.add_argument('--engram-n-embed-per-ngram', type=int, default=99991,
                       help='Number of embedding slots per n-gram size.')
    group.add_argument('--engram-n-head-per-ngram', type=int, default=8,
                       help='Number of heads per n-gram.')
    group.add_argument('--engram-seed', type=int, default=0,
                       help='Random seed for Engram hash initialization.')
    group.add_argument('--engram-layer-ids', nargs='+', type=int, default=None,
                       help='Layer IDs (0-indexed) to attach Engram modules. Multiple values allowed.')
    group.add_argument('--engram-base-vocab-size', type=int, default=None,
                       help='Base vocabulary size for Engram.')
    group.add_argument('--engram-embed-dim', type=int, default=672,
                       help='Embedding dimension for Engram.')

    group = parser.add_argument_group(title='fuji-hybrid',
                                      description='Fuji Hybrid Linear Attention')
    group.add_argument('--experimental-attention-variant', type=str, default=None,
                       choices=['gated_delta_net'],
                       help='Enable GatedDeltaNet hybrid layer. '
                            'Synonymous with "--linear-attention-type gated_delta_net".')

    return parser


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_fuji_args,
        store=store,
    )