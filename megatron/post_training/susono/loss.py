# Copyright (c) 2026 Third Intelligence. All rights reserved.
"""SFT and DPO loss helpers.

Mirrors ms-swift `swift/megatron/trainers/trainer.py:50-77` for SFT and
`swift/megatron/trainers/dpo_trainer.py:34-75` for DPO, but kept dependency-free.
``output_tensor`` is the per-token cross-entropy returned by GPTModel when
``labels`` is passed at forward time, so ``logp = -CE``.
"""
from functools import partial
from typing import Tuple

import torch
import torch.nn.functional as F

from megatron.training.utils import average_losses_across_data_parallel_group


def sft_loss_func(labels: torch.Tensor, output_tensor: torch.Tensor):
    """Standard cross-entropy SFT loss with HF ``-100`` ignore convention.

    Returns the Megatron 2-tuple ``(loss, {metrics})`` — the same protocol used
    by ``pretrain_susono.loss_func`` and :func:`dpo_loss_func`. ``loss`` is the
    per-token mean over answer tokens; the pipeline schedule then divides only by
    ``num_microbatches`` (see ``schedules.py:267-272``).

    NOTE: do NOT return the 3-tuple ``(loss, num_tokens, metrics)`` here. With
    ``calculate_per_token_loss=False`` the 3-tuple branch expects ``loss`` to be
    the per-token *sum* and divides it by ``num_tokens`` itself; returning an
    already-averaged loss in that branch divides the gradient a second time by
    ~num_tokens, collapsing it to ~1/num_tokens and breaking SFT optimization.
    """
    losses = output_tensor.float()
    loss_mask = (labels != -100).float()
    denom = loss_mask.sum().clamp(min=1.0)
    loss = (losses * loss_mask).sum() / denom

    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"lm loss": averaged_loss[0]}


def get_sequence_logps(output_tensor: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return summed log-probabilities over answer tokens, shape ``[B]``."""
    per_token_logp = -output_tensor.float()
    loss_mask = (labels != -100).float()
    return (per_token_logp * loss_mask).sum(dim=-1)


def dpo_loss_func(labels: torch.Tensor,
                  beta: float,
                  label_smoothing: float,
                  rpo_alpha: float,
                  output_tensor: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """Vanilla sigmoid DPO loss.

    ``output_tensor`` is the concatenation ``[ref_out; policy_out]`` along the
    batch dimension (see :mod:`dpo_susono`). ``labels`` is the per-(chosen|rejected)
    label tensor for one side, of shape ``[2*num_samples, S]`` where the first
    ``num_samples`` rows are chosen and the remaining are rejected.
    """
    batch_size = output_tensor.shape[0]
    assert batch_size % 2 == 0, "Expected concatenated [ref; policy] output."
    half = batch_size // 2

    ref_out = output_tensor[:half].detach()
    policy_out = output_tensor[half:]

    ref_logps = get_sequence_logps(ref_out, labels)
    policy_logps = get_sequence_logps(policy_out, labels)

    assert ref_logps.shape[0] % 2 == 0, "Expected paired chosen/rejected rows."
    num_samples = ref_logps.shape[0] // 2
    pc, pr = policy_logps[:num_samples], policy_logps[num_samples:]
    rc, rr = ref_logps[:num_samples], ref_logps[num_samples:]

    logits = beta * ((pc - rc) - (pr - rr))
    loss = (
        -(1.0 - label_smoothing) * F.logsigmoid(logits)
        - label_smoothing * F.logsigmoid(-logits)
    ).mean()

    chosen_reward = (beta * (pc - rc)).detach()
    rejected_reward = (beta * (pr - rr)).detach()
    margins = chosen_reward - rejected_reward
    accuracies = (chosen_reward > rejected_reward).float()

    if rpo_alpha and rpo_alpha > 0.0:
        # Auxiliary NLL on chosen-side answer tokens only.
        chosen_labels = labels[:num_samples]
        chosen_losses = policy_out[:num_samples].float()
        chosen_mask = (chosen_labels != -100).float()
        denom = chosen_mask.sum().clamp(min=1.0)
        nll_loss = (chosen_losses * chosen_mask).sum() / denom
        loss = loss + rpo_alpha * nll_loss
    else:
        nll_loss = None

    raw_metrics = {
        "loss": loss.detach(),
        "logps/chosen": pc.detach().mean(),
        "logps/rejected": pr.detach().mean(),
        "rewards/chosen": chosen_reward.mean(),
        "rewards/rejected": rejected_reward.mean(),
        "rewards/accuracies": accuracies.mean(),
        "rewards/margins": margins.mean(),
    }
    if nll_loss is not None:
        raw_metrics["nll_loss"] = nll_loss.detach()

    reduced = average_losses_across_data_parallel_group(list(raw_metrics.values()))
    metrics = {k: reduced[i] for i, k in enumerate(raw_metrics.keys())}
    return loss, metrics


def bind_sft_loss(labels):
    return partial(sft_loss_func, labels)


def bind_dpo_loss(labels, *, beta, label_smoothing, rpo_alpha):
    return partial(dpo_loss_func, labels, beta, label_smoothing, rpo_alpha)
