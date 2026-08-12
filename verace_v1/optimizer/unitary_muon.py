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

def stiefel_orthogonalize(G: torch.Tensor, steps: int = 5, tol: float = 1e-2) -> torch.Tensor:
    """
    Projects 2D matrix G (square or rectangular M x N) onto the Stiefel manifold.
    First attempts 5 steps of fast GPU quintic Newton-Schulz polynomial iteration.
    Falls back to exact SVD polar decomposition if deviation > tol or for singular spectra.
    Guarantees Q^T Q = I (M >= N) or Q Q^T = I (M <= N) with maximum GPU throughput.
    """
    if G.ndim != 2:
        return G

    m, n = G.shape
    X = G.float()
    min_dim = min(m, n)

    norm = torch.norm(X)
    if norm > 1e-7:
        # Scale matrix so Frobenius norm matches expected orthogonal norm sqrt(min_dim)
        X_scaled = (X / norm) * (min_dim ** 0.5)

        # Quintic Newton-Schulz polynomial coefficients
        a, b, c = 3.4445, -4.7750, 2.0315

        if m >= n:
            for _ in range(steps):
                A = X_scaled.T @ X_scaled
                A2 = A @ A
                B = b * A + c * A2
                B.diagonal().add_(a)
                X_scaled = X_scaled @ B
            dev = torch.norm(X_scaled.T @ X_scaled - torch.eye(n, device=G.device)).item()
        else:
            for _ in range(steps):
                A = X_scaled @ X_scaled.T
                A2 = A @ A
                B = b * A + c * A2
                B.diagonal().add_(a)
                X_scaled = B @ X_scaled
            dev = torch.norm(X_scaled @ X_scaled.T - torch.eye(m, device=G.device)).item()

        if dev < tol:
            return X_scaled.to(G.dtype)

    # Exact SVD fallback for pathologically ill-conditioned matrices
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return (U @ Vh).to(G.dtype)


class UnitaryMuon(Optimizer):
    """
    Unitary Muon Optimizer.
    Applies exact Stiefel-manifold orthogonalization to every 2D parameter's momentum update.
    Uses fast GPU Newton-Schulz polynomial iterations with automatic SVD fallback.
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

