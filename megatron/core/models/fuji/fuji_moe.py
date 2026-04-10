# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3-Next-style Sparse Mixture-of-Experts for the Fuji architecture.

Design (from Qwen3-Next):
  - Top-k routing with softmax probabilities and optional probability normalisation.
  - Each MoE block contains:
      shared_expert   — always active, gated by a sigmoid scalar
      routed_experts  — top-k selected from num_experts candidates
  - Both expert types use a SwiGLU MLP.

Expert Parallelism (EP):
  - FujiRoutedExperts shards experts across EP ranks.
  - Each GPU holds num_experts / EP local experts.
  - EP > 1: tokens are dispatched via all-to-all before expert compute and
    gathered back via all-to-all after. Routing weights are applied inside
    the expert compute step.
  - Expert compute uses GroupedGEMM when available (grouped_gemm package),
    falling back to a Python loop otherwise.
  - Checkpoint format: always saved as full [num_experts, ...] tensors regardless
    of EP size (sharded_state_dict all-gathers before saving).  _load_from_state_dict
    automatically slices the correct shard when loading into any EP configuration.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from megatron.core import parallel_state as _parallel_state
    _HAVE_PARALLEL_STATE = True
except ImportError:
    _HAVE_PARALLEL_STATE = False

try:
    from megatron.core.transformer.moe import grouped_gemm_util as gg
    _HAVE_GROUPED_GEMM = gg.grouped_gemm_is_available()
except Exception:
    gg = None
    _HAVE_GROUPED_GEMM = False

try:
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    _HAVE_SHARDED_TENSOR = True
except ImportError:
    _HAVE_SHARDED_TENSOR = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_ep_info():
    """Return (ep_size, ep_rank, ep_group) from parallel_state, or (1, 0, None)."""
    if not _HAVE_PARALLEL_STATE:
        return 1, 0, None
    try:
        ep_size = _parallel_state.get_expert_model_parallel_world_size()
        ep_rank = _parallel_state.get_expert_model_parallel_rank()
        ep_group = _parallel_state.get_expert_model_parallel_group()
        # world_size returns 0 when distributed is not initialised
        if ep_size <= 0:
            return 1, 0, None
        return ep_size, ep_rank, ep_group
    except Exception:
        return 1, 0, None


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
    """EP-aware batched routed experts.

    Each GPU holds ``num_local_experts = num_experts / EP`` experts.
    When EP > 1, tokens are dispatched via all-to-all before expert computation
    and gathered back after.  Expert computation uses GroupedGEMM when the
    ``grouped_gemm`` package is available, otherwise falls back to a Python loop.

    Weights (``gate_up_proj``, ``down_proj``) keep their original shapes so that
    checkpoints produced without EP (all experts on one GPU) can be loaded
    without conversion: ``_load_from_state_dict`` automatically slices the
    correct shard.

    Args:
        num_experts:       Total number of routed experts (global).
        hidden_size:       Model hidden dimension D.
        intermediate_size: Per-expert intermediate dimension.
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        ep_size, ep_rank, _ep_group = _get_ep_info()

        assert num_experts % ep_size == 0, (
            f"num_experts ({num_experts}) must be divisible by EP size ({ep_size})"
        )

        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.num_local_experts = num_experts // ep_size
        self.local_expert_offset = ep_rank * self.num_local_experts

        # Only allocate local experts on this rank.
        # Shapes kept identical to the original full-expert tensors (per expert),
        # so existing checkpoint keys remain valid with shard-aware loading.
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, intermediate_size)
        )
        nn.init.kaiming_uniform_(self.gate_up_proj.view(-1, hidden_size), a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.down_proj.view(-1, intermediate_size), a=5 ** 0.5)

        # Expert weights must NOT be all-reduced across DP when EP > 1.
        # Megatron uses the `allreduce` attribute to decide this.
        _expert_parallel = ep_size > 1
        setattr(self.gate_up_proj, 'allreduce', not _expert_parallel)
        setattr(self.down_proj,    'allreduce', not _expert_parallel)

    # ------------------------------------------------------------------
    # Checkpoint compatibility: load from full (EP=1) or sharded state
    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load local expert shard from a full (EP=1) or matching shard checkpoint.

        If the saved tensor has shape ``[num_experts, ...]`` (full checkpoint),
        we slice ``[local_expert_offset : local_expert_offset + num_local_experts]``.
        If the saved tensor already has shape ``[num_local_experts, ...]``, we
        load it directly.
        """
        for param_name in ('gate_up_proj', 'down_proj'):
            key = prefix + param_name
            if key not in state_dict:
                if strict:
                    missing_keys.append(key)
                continue

            saved = state_dict[key]
            param = getattr(self, param_name)

            if saved.shape == param.shape:
                # Already the correct shard shape — load directly.
                param.data.copy_(saved)
            elif saved.shape[0] == self.num_experts:
                # Full checkpoint (EP=1): slice the local shard.
                shard = saved[
                    self.local_expert_offset : self.local_expert_offset + self.num_local_experts
                ]
                if shard.shape != param.shape:
                    error_msgs.append(
                        f"Size mismatch for {key}: sliced shard {shard.shape} "
                        f"!= parameter {param.shape}"
                    )
                else:
                    param.data.copy_(shard)
            else:
                error_msgs.append(
                    f"Cannot load {key}: saved shape {saved.shape} is incompatible "
                    f"with num_local_experts={self.num_local_experts}, "
                    f"num_experts={self.num_experts}."
                )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Dist-checkpointing sharded state dict — always saves in EP=1 format.

        Even when EP > 1, the expert weights are gathered across all EP ranks
        via all_gather_into_tensor so that the saved tensor always has the full
        [num_experts, ...] shape (identical to an EP=1 checkpoint).

        replica_id encodes (ep_rank, dp_rank) so that exactly one rank writes:
          EP=1:  replica_id = dp_rank
          EP>1:  replica_id = ep_rank * dp_size + dp_rank
        Only the rank with replica_id == 0 (i.e. ep_rank=0, dp_rank=0) writes.

        This guarantees that checkpoints are interchangeable regardless of the
        EP size used during training, and that _load_from_state_dict can always
        slice the correct shard when loading into any EP configuration.
        """
        ep_size, ep_rank, ep_group = _get_ep_info()

        # DP rank/size — needed so that only one DP replica writes.
        try:
            dp_rank = _parallel_state.get_data_parallel_rank(with_context_parallel=True)
            dp_size = _parallel_state.get_data_parallel_world_size(with_context_parallel=True)
        except Exception:
            dp_rank, dp_size = 0, 1

        if not _HAVE_SHARDED_TENSOR:
            # Fallback for environments without dist_checkpointing.
            # Gather to full [num_experts, ...] even in this path.
            sharded_sd = {}
            for param_name in ('gate_up_proj', 'down_proj'):
                param = getattr(self, param_name)
                if ep_size > 1:
                    full_shape = (self.num_experts, *param.data.shape[1:])
                    full = torch.empty(full_shape, dtype=param.data.dtype, device=param.data.device)
                    dist.all_gather_into_tensor(full, param.data.contiguous(), group=ep_group)
                    sharded_sd[prefix + param_name] = full
                else:
                    sharded_sd[prefix + param_name] = param.data
            return sharded_sd

        prepend_axis_num = len(sharded_offsets)
        sharded_sd = {}

        for param_name in ('gate_up_proj', 'down_proj'):
            key   = prefix + param_name
            param = getattr(self, param_name)

            if ep_size > 1:
                # All-gather local shard → full [num_experts, ...] tensor.
                # Ranks are ordered by ep_rank, matching local_expert_offset layout.
                full_shape = (self.num_experts, *param.data.shape[1:])
                full_tensor = torch.empty(
                    full_shape, dtype=param.data.dtype, device=param.data.device
                )
                dist.all_gather_into_tensor(
                    full_tensor, param.data.contiguous(), group=ep_group
                )
                # replica_id = ep_rank * dp_size + dp_rank
                # → 0 only when ep_rank=0 and dp_rank=0 (sole writer).
                replica_id = ep_rank * dp_size + dp_rank
                sharded_sd[key] = ShardedTensor.from_rank_offsets(
                    key,
                    full_tensor,
                    *sharded_offsets,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                )
            else:
                # EP=1: expert weights are replicated across DP ranks.
                # replica_id = dp_rank → only DP rank 0 writes.
                sharded_sd[key] = ShardedTensor.from_rank_offsets(
                    key,
                    param.data,
                    *sharded_offsets,
                    replica_id=dp_rank,
                    prepend_axis_num=prepend_axis_num,
                )
        return sharded_sd

    # ------------------------------------------------------------------
    # Internal: grouped-GEMM expert compute on locally-owned tokens
    # ------------------------------------------------------------------

    def _compute_local(
        self,
        tokens: Tensor,        # [N, D] — tokens sorted by local expert id
        local_ids: Tensor,     # [N]   — local expert id in [0, num_local_experts)
        weights: Tensor,       # [N]   — routing weight per token
        tokens_per_expert: Tensor,  # [num_local_experts] int
    ) -> Tensor:
        """Compute expert outputs for the locally-owned experts.

        Uses GroupedGEMM when available (much faster), otherwise falls back to
        the original Python loop.

        Returns:
            [N, D]  — expert outputs with routing weights applied.
        """
        N = tokens.shape[0]
        if N == 0:
            return torch.zeros_like(tokens)

        if _HAVE_GROUPED_GEMM:
            return self._compute_local_grouped_gemm(tokens, tokens_per_expert, weights)
        else:
            return self._compute_local_loop(tokens, local_ids, weights)

    def _compute_local_grouped_gemm(
        self,
        tokens: Tensor,             # [N, D]  sorted by local expert
        tokens_per_expert: Tensor,  # [E_local] int
        weights: Tensor,            # [N]
    ) -> Tensor:
        """GroupedGEMM path for local expert computation.

        Weight convention (kept from original FujiRoutedExperts):
          gate_up_proj: [E_local, 2*I, D]  → gmm with trans_b=True → [N, D] @ [D, 2I]
          down_proj:    [E_local, D, I]    → gmm with trans_b=True → [N, I] @ [I, D]
        """
        # fc1: [N, D] @ gate_up_proj[e].T = [N, 2I]
        fc1 = gg.ops.gmm(tokens, self.gate_up_proj, tokens_per_expert, trans_b=True)

        # SwiGLU
        gate, up = fc1.chunk(2, dim=-1)       # each [N, I]
        h = F.silu(gate) * up                  # [N, I]

        # Apply routing weights before down_proj (saves one pass)
        h = h * weights.unsqueeze(-1)          # [N, I]

        # fc2: [N, I] @ down_proj[e].T = [N, D]
        out = gg.ops.gmm(h, self.down_proj, tokens_per_expert, trans_b=True)

        return out

    def _compute_local_loop(
        self,
        tokens: Tensor,    # [N, D]  sorted by local expert
        local_ids: Tensor, # [N]
        weights: Tensor,   # [N]
    ) -> Tensor:
        """Fallback Python-loop path (slow, but no extra dependency)."""
        output = torch.zeros_like(tokens)

        expert_mask = F.one_hot(local_ids, self.num_local_experts)  # [N, E_local]
        expert_mask = expert_mask.T                                   # [E_local, N]
        active = expert_mask.any(dim=1).nonzero(as_tuple=False)

        for row in active:
            eid = row[0].item()
            tok_idx = torch.where(expert_mask[eid])[0]
            x = tokens[tok_idx]
            gate_out, up_out = F.linear(x, self.gate_up_proj[eid]).chunk(2, dim=-1)
            x = F.silu(gate_out) * up_out
            x = x * weights[tok_idx, None]
            x = F.linear(x, self.down_proj[eid])
            output.index_add_(0, tok_idx, x.to(output.dtype))

        return output

    # ------------------------------------------------------------------
    # All-to-all dispatch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _exchange_counts(
        send_counts: Tensor,   # [EP] int64 on device
        ep_group,
    ) -> Tensor:
        """All-to-all exchange of per-rank token counts. Returns recv_counts [EP]."""
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
        return recv_counts

    @staticmethod
    def _all_to_all_tokens(
        data: Tensor,           # [N_send, ...]
        send_counts: Tensor,    # [EP] int
        recv_counts: Tensor,    # [EP] int
        ep_group,
    ) -> Tensor:
        """Send/receive variable-length token data across EP ranks."""
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()
        total_recv  = int(recv_counts.sum().item())
        shape_rest  = data.shape[1:]
        out = torch.empty(
            (total_recv, *shape_rest), dtype=data.dtype, device=data.device
        )
        dist.all_to_all_single(
            out, data.contiguous(),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=ep_group,
        )
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,      # [T, D]
        top_k_indices: Tensor,      # [T, k]  global expert ids
        top_k_weights: Tensor,      # [T, k]
    ) -> Tensor:
        """Compute weighted sum of selected expert outputs.

        Args:
            hidden_states:  [T, D]
            top_k_indices:  [T, k]  — selected expert IDs (global).
            top_k_weights:  [T, k]  — normalised routing weights.

        Returns:
            [T, D]  — accumulated expert output.
        """
        ep_size, ep_rank, ep_group = _get_ep_info()

        if ep_size == 1:
            return self._forward_no_ep(hidden_states, top_k_indices, top_k_weights)
        else:
            return self._forward_ep(hidden_states, top_k_indices, top_k_weights,
                                    ep_size, ep_group)

    # ------------------------------------------------------------------
    # EP = 1 path  (no communication)
    # ------------------------------------------------------------------

    def _forward_no_ep(
        self,
        hidden_states: Tensor,  # [T, D]
        top_k_indices: Tensor,  # [T, k]
        top_k_weights: Tensor,  # [T, k]
    ) -> Tensor:
        T, D = hidden_states.shape
        k = top_k_indices.shape[1]

        # Flatten: one entry per (token, expert-slot) pair.
        flat_ids  = top_k_indices.reshape(-1)                              # [T*k]
        flat_w    = top_k_weights.reshape(-1)                              # [T*k]
        flat_x    = hidden_states.unsqueeze(1).expand(-1, k, -1).reshape(-1, D)  # [T*k, D]

        # Sort by expert so GroupedGEMM can process contiguous segments.
        sort_idx = flat_ids.argsort(stable=True)
        flat_x_s = flat_x[sort_idx]
        flat_ids_s = flat_ids[sort_idx]   # local ids == global ids when EP=1
        flat_w_s = flat_w[sort_idx]

        tokens_per_expert = torch.bincount(
            flat_ids_s, minlength=self.num_local_experts
        ).to(dtype=torch.int64, device='cpu')

        out_sorted = self._compute_local(flat_x_s, flat_ids_s, flat_w_s, tokens_per_expert)

        # Restore original order.
        unsort_idx = sort_idx.argsort()
        out = out_sorted[unsort_idx]       # [T*k, D]

        # Accumulate k expert outputs per token.
        return out.reshape(T, k, D).sum(dim=1)  # [T, D]

    # ------------------------------------------------------------------
    # EP > 1 path  (all-to-all dispatch + combine)
    # ------------------------------------------------------------------

    def _forward_ep(
        self,
        hidden_states: Tensor,  # [T, D]
        top_k_indices: Tensor,  # [T, k]
        top_k_weights: Tensor,  # [T, k]
        ep_size: int,
        ep_group,
    ) -> Tensor:
        T, D = hidden_states.shape
        k = top_k_indices.shape[1]
        device = hidden_states.device

        # ── 1. Flatten (T, k) → T*k ──────────────────────────────────
        flat_ids = top_k_indices.reshape(-1)                               # [T*k]
        flat_w   = top_k_weights.reshape(-1)                               # [T*k]
        flat_x   = hidden_states.unsqueeze(1).expand(-1, k, -1).reshape(-1, D)  # [T*k, D]

        # ── 2. Sort by destination EP rank ───────────────────────────
        dst_rank = flat_ids // self.num_local_experts                      # [T*k]
        sort_idx = dst_rank.argsort(stable=True)
        flat_x_s   = flat_x[sort_idx]
        flat_ids_s = flat_ids[sort_idx]
        flat_w_s   = flat_w[sort_idx]

        # Save inverse permutation for the combine step.
        unsort_idx = sort_idx.argsort()

        # ── 3. Exchange token counts (metadata only) ──────────────────
        dst_rank_s = dst_rank[sort_idx]
        send_counts = torch.bincount(
            dst_rank_s.to(torch.long), minlength=ep_size
        ).to(dtype=torch.int64, device=device)

        recv_counts = self._exchange_counts(send_counts, ep_group)

        # ── 4. Dispatch tokens, expert ids, weights via all-to-all ───
        recv_x   = self._all_to_all_tokens(flat_x_s,   send_counts, recv_counts, ep_group)
        recv_ids = self._all_to_all_tokens(flat_ids_s, send_counts, recv_counts, ep_group)
        recv_w   = self._all_to_all_tokens(flat_w_s,   send_counts, recv_counts, ep_group)

        # ── 5. Convert global → local expert ids ─────────────────────
        local_ids = recv_ids - self.local_expert_offset    # [N_recv]

        # Sort received tokens by local expert for GroupedGEMM.
        local_sort_idx = local_ids.argsort(stable=True)
        recv_x_s   = recv_x[local_sort_idx]
        local_ids_s = local_ids[local_sort_idx]
        recv_w_s   = recv_w[local_sort_idx]

        tokens_per_expert = torch.bincount(
            local_ids_s.to(torch.long), minlength=self.num_local_experts
        ).to(dtype=torch.int64, device='cpu')

        # ── 6. Expert computation (GroupedGEMM or loop) ───────────────
        local_out_s = self._compute_local(
            recv_x_s, local_ids_s, recv_w_s, tokens_per_expert
        )

        # Restore received-order before combine.
        local_unsort = local_sort_idx.argsort()
        local_out = local_out_s[local_unsort]              # [N_recv, D]

        # ── 7. Combine: send results back to originating ranks ────────
        combined = self._all_to_all_tokens(
            local_out,
            recv_counts,   # now these are our send sizes
            send_counts,   # and these are our recv sizes
            ep_group,
        )                                                  # [T*k, D]

        # ── 8. Restore original (token, slot) order ──────────────────
        combined = combined[unsort_idx]                    # [T*k, D]

        # ── 9. Accumulate k expert contributions per token ────────────
        return combined.reshape(T, k, D).sum(dim=1)        # [T, D]


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
        # Megatron stores these as num_moe_experts / moe_router_topk; fall back to
        # the legacy attribute names for configs that use them directly.
        num_experts     = getattr(config, 'num_moe_experts', getattr(config, 'num_experts', 64))
        num_experts_tok = getattr(config, 'moe_router_topk', getattr(config, 'num_experts_per_tok', 2))
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

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharded state dict for dist_checkpointing.

        Delegates to each child's sharded_state_dict (or fallback) so that
        FujiRoutedExperts.sharded_state_dict is correctly invoked for EP sharding.
        Without this override, the default traversal calls state_dict() on this
        entire module and wraps expert weights as non-sharded tensors, causing a
        global shape mismatch when the checkpoint has EP=1 shapes.
        """
        from megatron.core.transformer.utils import sharded_state_dict_default
        sharded_sd = {}
        for name, module in self.named_children():
            sharded_sd.update(
                sharded_state_dict_default(
                    module, f'{prefix}{name}.', sharded_offsets, metadata
                )
            )
        return sharded_sd

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
