# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3-Next-style Sparse Mixture-of-Experts for the Fuji architecture.

Design (from Qwen3-Next):
  - Top-k routing with softmax probabilities and optional probability normalisation.
  - Each MoE block contains:
      shared_expert   — always active, gated by a sigmoid scalar
      routed_experts  — top-k selected from num_experts candidates
  - Both expert types use a SwiGLU MLP.

Megatron-LM tensor convention: [S, B, D] (sequence-first).
Internally the module reshapes to [S*B, D] for expert computation then back.

Note: This implementation is intended for correctness and research iteration.
      For high-throughput production use, the standard Megatron MoE
      (SwitchMLP / GroupedMLP) with proper expert parallelism should be
      integrated.  That integration is left as future work.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class _SwiGLUMLP(nn.Module):
    """Standard SwiGLU MLP used for both shared and routed experts."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj  = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FujiTopKRouter(nn.Module):
    """Softmax-based top-k token router.

    Computes per-token routing probabilities, selects top-k experts, and
    optionally normalises the selected weights to sum to 1.

    Args:
        num_experts:      Total number of routed expert slots.
        num_experts_per_tok: Number of experts selected per token (top-k).
        hidden_size:      Input feature dimension.
        norm_topk_prob:   If True, normalise selected probabilities.
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.top_k = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: [T, D]  (T = num_tokens after reshaping)

        Returns:
            router_logits:  [T, num_experts]  — full probability vector (for aux loss).
            routing_weights:[T, top_k]        — selected and optionally normalised weights.
            selected_experts:[T, top_k]        — expert indices (long).
        """
        router_logits = F.softmax(F.linear(hidden_states, self.weight), dim=-1, dtype=torch.float)
        top_weights, top_indices = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)
        return router_logits, top_weights.to(hidden_states.dtype), top_indices


class FujiRoutedExperts(nn.Module):
    """Batched routed experts (num_experts × SwiGLU MLPs).

    Parameters are stored as 3-D weight tensors for efficient per-expert lookup.

    Args:
        num_experts:       Total number of routed experts.
        hidden_size:       Model hidden dimension D.
        intermediate_size: Per-expert intermediate dimension.
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        nn.init.kaiming_uniform_(self.gate_up_proj.view(-1, hidden_size), a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.down_proj.view(-1, intermediate_size), a=5 ** 0.5)

    def forward(
        self,
        hidden_states: Tensor,
        top_k_indices: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        """Compute weighted sum of selected expert outputs.

        Args:
            hidden_states: [T, D]
            top_k_indices: [T, k]  — selected expert IDs.
            top_k_weights: [T, k]  — normalised routing weights.

        Returns:
            [T, D]  — accumulated expert output.
        """
        T, D = hidden_states.shape
        output = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_indices, self.num_experts)  # [T, k, E]
            expert_mask = expert_mask.permute(2, 1, 0)                 # [E, k, T]
            active_experts = expert_mask.sum(dim=(1, 2)).nonzero(as_tuple=False)

        for row in active_experts:
            eid = row[0].item()
            k_pos, tok_idx = torch.where(expert_mask[eid])
            x = hidden_states[tok_idx]                                  # [n_tok, D]
            gate_proj, up_proj = F.linear(x, self.gate_up_proj[eid]).chunk(2, dim=-1)
            x = F.silu(gate_proj) * up_proj
            x = F.linear(x, self.down_proj[eid])                       # [n_tok, D]
            x = x * top_k_weights[tok_idx, k_pos, None]
            output.index_add_(0, tok_idx, x.to(output.dtype))

        return output


# ──────────────────────────────────────────────────────────────────────────────
# Full MoE block
# ──────────────────────────────────────────────────────────────────────────────

class FujiSparseMoE(nn.Module):
    """Qwen3-Next-style Sparse Mixture-of-Experts block.

    Structure:
        shared_expert_output = shared_expert(x)
        gated_shared         = sigmoid(shared_gate(x)) * shared_expert_output
        router_logits, weights, indices = router(x)
        routed_output = routed_experts(x, indices, weights)
        output = gated_shared + routed_output

    The router operates over the flattened token dimension [S*B, D].

    Args:
        config:  TransformerConfig.  Used fields:
                 hidden_size, moe_intermediate_size, shared_expert_intermediate_size,
                 num_experts, num_experts_per_tok, norm_topk_prob.
    """

    def __init__(self, config) -> None:
        super().__init__()
        hidden          = config.hidden_size
        moe_inter       = getattr(config, 'moe_intermediate_size', 512)
        shared_inter    = getattr(config, 'shared_expert_intermediate_size', 512)
        num_experts     = getattr(config, 'num_experts', 64)
        num_experts_tok = getattr(config, 'num_experts_per_tok', 2)
        norm_topk       = getattr(config, 'norm_topk_prob', True)

        self.gate = FujiTopKRouter(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_tok,
            hidden_size=hidden,
            norm_topk_prob=norm_topk,
        )
        self.experts = FujiRoutedExperts(
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=moe_inter,
        )
        self.shared_expert      = _SwiGLUMLP(hidden, shared_inter)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """MoE forward pass.

        Args:
            hidden_states: [S, B, D]  (Megatron sequence-first).

        Returns:
            output:       [S, B, D]
            router_logits:[T, num_experts]  for auxiliary load-balancing loss.
        """
        S, B, D = hidden_states.shape
        x = hidden_states.reshape(-1, D)                   # [T, D]

        # Shared expert (always active, sigmoid-gated)
        shared_out  = self.shared_expert(x)                # [T, D]
        shared_gate = torch.sigmoid(self.shared_expert_gate(x))  # [T, 1]
        shared_out  = shared_gate * shared_out

        # Routed experts
        router_logits, routing_weights, selected_experts = self.gate(x)
        expert_out = self.experts(x, selected_experts, routing_weights)

        output = (shared_out + expert_out).reshape(S, B, D)
        return output, router_logits


# ──────────────────────────────────────────────────────────────────────────────
# Dense MLP fallback (for mlp_only_layers)
# ──────────────────────────────────────────────────────────────────────────────

class FujiDenseMLP(nn.Module):
    """Dense SwiGLU MLP used for layers in mlp_only_layers.

    Args:
        config: TransformerConfig.  Uses hidden_size and ffn_hidden_size.
    """

    def __init__(self, config) -> None:
        super().__init__()
        inter = getattr(config, 'ffn_hidden_size', None) or 4 * config.hidden_size
        self.mlp = _SwiGLUMLP(config.hidden_size, inter)

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, None]:
        S, B, D = hidden_states.shape
        x = hidden_states.reshape(-1, D)
        out = self.mlp(x).reshape(S, B, D)
        return out, None
