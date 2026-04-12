# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MHC Selective Recompute: custom autograd Function for mHC width/depth connections.

Saves only the inputs (X, x_out) during the forward pass and recomputes
width_connection during the backward pass, eliminating the storage of:
  - new_residuals [S, B, n, D]
  - beta          [S, B, n]
  - H_res         [S, B, n, n]  (intermediate)
  - normed        [S, B, n*D]   (intermediate)
  - res_coeff     [S, B, n!]    (intermediate)
  - wc            [S, B, n+n!]  (intermediate)
"""

import torch


class MHCSelectiveCheckpoint(torch.autograd.Function):
    """Custom autograd Function implementing selective recompute for mHC.

    Instead of retaining all intermediate tensors from width_connection and
    depth_connection in the autograd graph, this Function:
      - forward:  runs both connections under no_grad and saves only X + x_out
      - backward: recomputes width_connection with enable_grad to rebuild the
                  graph on-the-fly, then calls backward through it

    Memory savings vs. standard closure pattern (n=4, D=2048, T = S*B tokens):
      - new_residuals: 8192·T elements eliminated
      - beta:             4·T elements eliminated
      - All width-connection intermediates (H_res, normed, wc, ...) eliminated

    Usage::

        X_next = MHCSelectiveCheckpoint.apply(X, x_out, mhc_module)
    """

    @staticmethod
    def forward(ctx, X, x_out, mhc_module):
        """Compute depth connection result, saving only X and x_out.

        Args:
            X:          Multi-stream hidden states [n, S, B, D].
            x_out:      Layer output [S, B, D].
            mhc_module: ManifoldConstrainedHyperConnection instance.

        Returns:
            X_next: Updated multi-stream hidden states [n, S, B, D].
        """
        with torch.no_grad():
            _, new_residuals, beta = mhc_module._width_connection(X)
            X_next = mhc_module._depth_connection(x_out, new_residuals, beta)

        ctx.save_for_backward(X, x_out)
        ctx.mhc_module = mhc_module
        return X_next

    @staticmethod
    def backward(ctx, grad_X_next):
        """Recompute width_connection and differentiate through both connections.

        Args:
            grad_X_next: Gradient w.r.t. X_next [n, S, B, D].

        Returns:
            grad_X:      Gradient w.r.t. X [n, S, B, D].
            grad_x_out:  Gradient w.r.t. x_out [S, B, D].
            None:        Gradient w.r.t. mhc_module (not differentiable).

        Implementation note — why autograd.grad instead of backward():
            MHCTransformerLayer with selective recompute uses mhc params in
            TWO places:
              1. _width_connection → branch_input (x_in) — kept in main graph
              2. _width_connection/_depth_connection — run under no_grad in
                 forward, recomputed here.

            If we called X_next_r.backward(), PyTorch's AccumulateGrad would
            fire the DDP backward-post-hook for each mHC param (1st time).
            Later, the main backward traverses branch_input back through
            _width_connection and AccumulateGrad fires again (2nd time) →
            Megatron DDP asserts "Cannot set grad twice".

            torch.autograd.grad() returns gradients WITHOUT calling
            AccumulateGrad or any registered hooks. We then manually
            accumulate the depth-connection gradients into param.grad. When
            the main backward subsequently reaches the mHC params via the
            branch_input path, AccumulateGrad fires exactly once (adding the
            branch_input gradient on top of the depth-connection gradient
            already in param.grad), and the DDP hook fires exactly once.
        """
        X, x_out = ctx.saved_tensors
        mhc_module = ctx.mhc_module

        # Detach and re-attach requires_grad so autograd can track this subgraph
        X_d = X.detach().requires_grad_(True)
        x_out_d = x_out.detach().requires_grad_(True)

        with torch.enable_grad():
            # Recompute width connection (rebuilds intermediate graph in SRAM)
            _, new_residuals, beta = mhc_module._width_connection(X_d)
            # Recompute depth connection
            X_next_r = mhc_module._depth_connection(x_out_d, new_residuals, beta)

        # Collect mHC parameters that participate in the depth-connection path.
        mhc_params = [p for p in mhc_module.parameters() if p.requires_grad]

        # Compute gradients via autograd.grad (does NOT trigger AccumulateGrad
        # or DDP backward-post-hooks on the mHC parameters).
        all_inputs = [X_d, x_out_d] + mhc_params
        all_grads = torch.autograd.grad(
            outputs=X_next_r,
            inputs=all_inputs,
            grad_outputs=grad_X_next,
            allow_unused=True,
        )

        grad_X = all_grads[0]
        grad_x_out = all_grads[1]
        param_grads = all_grads[2:]

        # Accumulate depth-connection gradients into param.grad WITHOUT
        # triggering AccumulateGrad.  When the main backward later reaches
        # AccumulateGrad for these params (via the branch_input path), it will
        # add the branch_input gradient on top, and the DDP hook fires once.
        for param, grad in zip(mhc_params, param_grads):
            if grad is not None:
                if param.grad is not None:
                    param.grad.add_(grad)
                else:
                    param.grad = grad.clone()

        return grad_X, grad_x_out, None
