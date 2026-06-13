# Copyright (c) 2026 Third Intelligence. All rights reserved.
"""CLI arguments for Susono SFT / DPO entry points."""


def add_susono_sft_args(parser):
    group = parser.add_argument_group(title="susono-sft")
    group.add_argument(
        "--finetune-hf-dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name or local path used for SFT.",
    )
    group.add_argument(
        "--finetune-data-split",
        type=str,
        default="train",
        help="HuggingFace dataset split.",
    )
    group.add_argument(
        "--finetune-num-shards-from-dp",
        action="store_true",
        help="If set, shard the HF dataset by (expert_)data_parallel_world_size.",
    )
    group.add_argument(
        "--finetune-num-epochs",
        type=int,
        default=1,
        help="Number of full passes over --finetune-hf-dataset. When --train-iters "
             "is not given, train_iters is derived from the real dataset size so "
             "every sample is seen ~num_epochs times "
             "(see megatron.post_training.susono.epoch_iters).",
    )
    group.add_argument(
        "--lr-wsd-decay-fraction",
        type=float,
        default=0.0,
        help="WSD annealing length as a fraction of the (auto-derived) train_iters. "
             "Resolved to an absolute --lr-wsd-decay-iters at startup when that flag "
             "is not given explicitly. 0 = no WSD decay tail.",
    )
    return parser


def add_susono_dpo_args(parser):
    parser = add_susono_sft_args(parser)
    group = parser.add_argument_group(title="susono-dpo")
    group.add_argument(
        "--ref-load",
        type=str,
        default=None,
        help="Checkpoint directory used to initialize the frozen reference model for DPO.",
    )
    group.add_argument(
        "--dpo-beta",
        type=float,
        default=0.1,
        help="DPO temperature beta. Smaller = tighter to reference.",
    )
    group.add_argument(
        "--dpo-label-smoothing",
        type=float,
        default=0.0,
        help="cDPO label smoothing factor in [0, 0.5).",
    )
    group.add_argument(
        "--dpo-loss-type",
        type=str,
        default="sigmoid",
        choices=["sigmoid"],
        help="DPO loss variant. Currently only 'sigmoid' (vanilla DPO).",
    )
    group.add_argument(
        "--rpo-alpha",
        type=float,
        default=0.0,
        help="RPO auxiliary NLL coefficient on the chosen response (0 = disabled).",
    )
    group.add_argument(
        "--ref-shard-size",
        type=int,
        default=1,
        help="Number of GPUs (DP ranks) across which each ref parameter is "
             "sharded (ZeRO-3 style). 1 = no sharding (full replica per rank); "
             "DP_SIZE = sharded across the entire DP group. When equal to "
             "DP_SIZE / num-distributed-optimizer-instances (= the intra-instance "
             "DP group size used by policy params), the existing Megatron group "
             "is reused so ref and policy shards align on the same node set.",
    )
    group.add_argument(
        "--ref-cpu-offload",
        action="store_true",
        help="Store ref-model parameter shards on CPU (NVLink-C2C H2D on GH200) "
             "instead of GPU. Combine with --ref-shard-size N to control the "
             "per-rank CPU footprint. Switches the ref wrapper to FSDP-style "
             "layer-wise gather, so peak GPU memory during ref forward stays "
             "around one transformer layer instead of the full model.",
    )
    return parser
