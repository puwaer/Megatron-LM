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
    return parser
