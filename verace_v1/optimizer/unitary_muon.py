"""
Unitary Muon Optimizer Module
Implements exact Stiefel-manifold orthogonal updates for every 2D parameter (square,
tall, or wide) via SVD-based polar decomposition: G = U diag(S) V^T, Q = U V^T.
This guarantees Q^T Q = I (or Q Q^T = I) to floating-point precision regardless of G's
singular value spectrum -- unlike iterative Newton-Schulz methods, which were tried
first here and found to enter a persistent oscillation (never converging) for matrices
with small singular values, which is the common case for real weight-matrix gradients.
"""

import torch
from torch.optim import Optimizer

def stiefel_orthogonalize(G: torch.Tensor) -> torch.Tensor:
    """
    Projects 2D matrix G (square or rectangular M x N) onto the Stiefel manifold via
    exact SVD-based polar decomposition.
    Guarantees Q^T Q = I (M >= N) or Q Q^T = I (M <= N), exact to floating-point
    precision in a single pass -- no iteration, no convergence dependence on G's scale.
    """
    if G.ndim != 2:
        return G

    U, _, Vh = torch.linalg.svd(G.float(), full_matrices=False)
    return (U @ Vh).to(G.dtype)


class UnitaryMuon(Optimizer):
    """
    Unitary Muon Optimizer.
    Applies exact Stiefel-manifold orthogonalization to every 2D parameter's momentum update.
    """
    def __init__(
        self,
        params,
        lr: float = 0.03,
        momentum: float = 0.98,
        weight_decay: float = 0.05
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                if p.ndim == 2:
                    update = stiefel_orthogonalize(buf)
                else:
                    update = buf

                if weight_decay > 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                p.data.add_(update, alpha=-lr)

        return loss
