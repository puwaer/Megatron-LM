# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Manifold-Constrained Hyper-Connections (mHC).

Implements mHC from "mHC: Manifold-Constrained Hyper-Connections" (arXiv:2512.24880).

mHC extends the standard residual connection x_{l+1} = x_l + F(x_l) to n parallel streams:

    x_in   = softmax(H_pre) · X_l           # aggregate n streams → single input  [S,B,D]
    x_out  = F(x_in)                         # layer function output                [S,B,D]
    H_res  = SinkhornKnopp(H_res_raw)        # doubly stochastic matrix             [n,n]
    X_{l+1} = H_res · X_l + H_post ⊗ x_out  # update n streams                   [n,S,B,D]

The Sinkhorn-Knopp projection constrains H_res to the Birkhoff polytope (doubly stochastic
matrices), restoring the identity-mapping property and ensuring training stability at scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinkhornKnopp(nn.Module):
    """Projects a square matrix to the doubly stochastic manifold (Birkhoff polytope).

    Alternately normalises rows and columns until convergence. Starting from exp(H)
    ensures all entries are positive. About 20 iterations are sufficient in practice.

    Args:
        iterations (int): Number of alternating row/column normalisation steps.
    """

    def __init__(self, iterations: int = 20) -> None:
        super().__init__()
        self.iterations = iterations

    def forward(self, H: Tensor) -> Tensor:
        """Project H onto the doubly stochastic manifold.

        Args:
            H: Square matrix of shape [n, n] (raw, unconstrained parameters).

        Returns:
            Doubly stochastic matrix of shape [n, n] where every row and column
            sums to 1 and all entries are non-negative.
        """
        M = torch.exp(H)
        for _ in range(self.iterations):
            M = M / M.sum(dim=1, keepdim=True)  # row normalisation
            M = M / M.sum(dim=0, keepdim=True)  # column normalisation
        return M


class ManifoldConstrainedHyperConnection(nn.Module):
    """Layer-level Manifold-Constrained Hyper-Connection (mHC) module.

    Manages n parallel residual streams at layer boundaries. The internal residuals
    within a TransformerLayer (attention residual + MLP residual) are kept standard;
    mHC adds multi-stream connectivity between layers.

    Parameters
    ----------
    hidden_size : int
        Dimension D of each residual stream.
    num_streams : int
        Number of parallel residual streams n.
    sinkhorn_iterations : int
        Number of Sinkhorn-Knopp iterations for H_res projection.

    Attributes
    ----------
    H_pre_raw : nn.Parameter [n]
        Raw parameters for stream aggregation weights. Applied via softmax so that
        weights are non-negative and sum to 1.
    H_post_raw : nn.Parameter [n]
        Raw parameters for output distribution weights. Applied via softmax.
    H_res_raw : nn.Parameter [n, n]
        Raw parameters for residual stream mixing. Projected to doubly stochastic
        matrix via Sinkhorn-Knopp during forward pass.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int,
        sinkhorn_iterations: int = 20,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.sinkhorn = SinkhornKnopp(iterations=sinkhorn_iterations)

        # H_pre: weights to aggregate n streams into one layer input.
        # Initialised uniformly so all streams contribute equally at the start.
        self.H_pre_raw = nn.Parameter(torch.zeros(num_streams))

        # H_post: weights to distribute the layer output back to n streams.
        # Initialised uniformly.
        self.H_post_raw = nn.Parameter(torch.zeros(num_streams))

        # H_res: residual stream mixing matrix.
        # Initialised as identity so each stream is its own residual at the start.
        self.H_res_raw = nn.Parameter(torch.eye(num_streams))

    def aggregate_streams(self, X: Tensor) -> Tensor:
        """Aggregate n residual streams into a single input for the layer function.

        Args:
            X: Multi-stream hidden states of shape [n, S, B, D].

        Returns:
            Aggregated tensor of shape [S, B, D].
        """
        H_pre = F.softmax(self.H_pre_raw, dim=0)  # [n]
        # Weighted sum over the stream dimension
        return torch.einsum('n,n...->...', H_pre, X)

    def distribute_output(self, X: Tensor, x_out: Tensor) -> Tensor:
        """Update n residual streams using the layer output and the Sinkhorn-projected
        residual mixing matrix.

        Args:
            X:     Current multi-stream hidden states, shape [n, S, B, D].
            x_out: Output of the standard TransformerLayer, shape [S, B, D].

        Returns:
            Updated multi-stream hidden states of shape [n, S, B, D].
        """
        # Project H_res to doubly stochastic matrix
        H_res = self.sinkhorn(self.H_res_raw)   # [n, n]
        H_post = F.softmax(self.H_post_raw, dim=0)  # [n]

        # H_res @ X: residual mixing  (i,j),(j,...) -> (i,...)
        x_res = torch.einsum('ij,j...->i...', H_res, X)   # [n, S, B, D]
        # H_post ⊗ x_out: broadcast output to n streams
        x_new = x_res + torch.einsum('i,...->i...', H_post, x_out)  # [n, S, B, D]
        return x_new

    def forward(self, X: Tensor, layer_fn) -> Tensor:
        """Convenience method: aggregate → layer_fn → distribute.

        In practice, MHCTransformerLayer calls aggregate_streams and distribute_output
        separately so that it can pass extra kwargs to the underlying TransformerLayer.
        This method is provided for testing and standalone use.

        Args:
            X:        Multi-stream hidden states of shape [n, S, B, D].
            layer_fn: Callable that maps [S, B, D] → [S, B, D].

        Returns:
            Updated multi-stream hidden states of shape [n, S, B, D].
        """
        x_in = self.aggregate_streams(X)
        x_out = layer_fn(x_in)
        return self.distribute_output(X, x_out)
