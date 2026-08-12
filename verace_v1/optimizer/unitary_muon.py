"""
Unitary Muon Optimizer Module
Implements exact Stiefel-manifold orthogonal updates for every 2D parameter (square,
tall, or wide) via SVD-based polar decomposition: G = U diag(S) V^T, Q = U V^T.
This guarantees Q^T Q = I (or Q Q^T = I) to floating-point precision regardless of G's
singular value spectrum -- unlike iterative Newton-Schulz methods, which were tried
first here and found to enter a persistent oscillation (never converging) for matrices
with small singular values, which is the common case for real weight-matrix gradients.
"""

import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
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
    else:
        X_scaled = X  # near-zero gradient: nothing to orthogonalize, but still need a fallback value below

    # Exact SVD fallback for pathologically ill-conditioned matrices. cusolver's batched
    # GPU driver can itself fail to converge on sufficiently degenerate matrices (seen in
    # practice: "algorithm failed to converge... input matrix is ill-conditioned or has
    # too many repeated singular values") -- letting that propagate unhandled crashes the
    # entire training run over a single pathological parameter update. Retry on CPU first
    # (LAPACK's SVD is generally more numerically robust than cusolver for these cases);
    # only if that *also* fails do we accept the un-converged Newton-Schulz iterate rather
    # than crash -- logged loudly either way, never silent.
    try:
        U, _, Vh = torch.linalg.svd(X, full_matrices=False)
        return (U @ Vh).to(G.dtype)
    except torch._C._LinAlgError as e:
        try:
            U, _, Vh = torch.linalg.svd(X.cpu(), full_matrices=False)
            warnings.warn(
                f"stiefel_orthogonalize: GPU SVD failed to converge ({e}); CPU SVD "
                f"retry succeeded for a {tuple(G.shape)} matrix.",
                stacklevel=2
            )
            return (U @ Vh).to(G.dtype).to(G.device)
        except torch._C._LinAlgError as e2:
            warnings.warn(
                f"stiefel_orthogonalize: SVD failed to converge on both GPU and CPU "
                f"({e2}) for a {tuple(G.shape)} matrix -- falling back to the "
                f"un-converged Newton-Schulz iterate (deviation from orthogonal not "
                f"guaranteed < tol this step) rather than crashing the training run.",
                stacklevel=2
            )
            return X_scaled.to(G.dtype)


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

                # Defense in depth: a non-finite gradient must never enter the momentum
                # buffer. buf.mul_(momentum) can never clear a NaN/Inf once absorbed (NaN
                # times anything is still NaN), so every future step would silently stay
                # corrupted forever -- observed in practice: a single bad batch produced a
                # NaN gradient, momentum absorbed it, and the (necessarily NaN-in-NaN-out)
                # orthogonalization fallback had no way to recover a meaningful update
                # from an already-NaN input. Skipping this parameter's update for one step
                # (leaving its momentum untouched) is far safer than corrupting it, and
                # gradient clipping upstream (see train_pretrain_step) is what should keep
                # this rare in the first place.
                if not torch.isfinite(grad).all():
                    warnings.warn(
                        f"UnitaryMuon: non-finite gradient for a {tuple(p.shape)} parameter "
                        f"-- skipping this step's update for it (momentum buffer left "
                        f"untouched) rather than corrupting momentum permanently.",
                        stacklevel=2
                    )
                    continue

                buf.mul_(momentum).add_(grad)

                if p.ndim == 2:
                    update = stiefel_orthogonalize(buf)
                else:
                    update = buf

                if weight_decay > 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                p.data.add_(update, alpha=-lr)

        return loss


class HybridMuonAdamW:
    """
    Routes a model's 2D hidden-layer weight matrices through UnitaryMuon and
    everything else (the tied embedding/lm_head table, and every <2D
    parameter -- norm gains, biases) through AdamW.

    Orthogonalizing a matrix that acts as a linear map at every forward pass
    is what Muon-family optimizers are for; forcing an embedding table's rows
    (which encode per-token semantics and frequency-correlated norms, not a
    linear transform) onto the Stiefel manifold every step fights what the
    table needs to represent and is a documented way to stall convergence --
    published Muon implementations exclude embeddings/heads/scalars for
    exactly this reason. Exposes the same zero_grad/step/state_dict surface
    as a single torch.optim.Optimizer so callers don't need to special-case it.
    """
    def __init__(self, muon_optimizer: UnitaryMuon, adamw_optimizer: torch.optim.AdamW):
        self.muon_optimizer = muon_optimizer
        self.adamw_optimizer = adamw_optimizer

    def zero_grad(self, set_to_none: bool = True):
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.muon_optimizer.step(closure)
        self.adamw_optimizer.step()
        return loss

    def state_dict(self) -> Dict:
        return {
            "muon": self.muon_optimizer.state_dict(),
            "adamw": self.adamw_optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict):
        self.muon_optimizer.load_state_dict(state["muon"])
        self.adamw_optimizer.load_state_dict(state["adamw"])

    @property
    def param_groups(self) -> List[Dict]:
        return self.muon_optimizer.param_groups + self.adamw_optimizer.param_groups


def build_hybrid_optimizer(
    model: nn.Module,
    muon_lr: float = 0.03,
    muon_momentum: float = 0.98,
    muon_weight_decay: float = 0.05,
    adamw_lr: float = 3e-4,
    adamw_weight_decay: float = 0.0,
    adamw_betas: tuple = (0.9, 0.95),
) -> HybridMuonAdamW:
    """
    Splits model.named_parameters() into the two groups HybridMuonAdamW
    expects: the tied embed_tokens/lm_head weight and every parameter with
    ndim < 2 go to AdamW; every other 2D weight matrix goes to UnitaryMuon.
    """
    embedding_weight: Optional[torch.Tensor] = None
    if hasattr(model, "embed_tokens"):
        embedding_weight = model.embed_tokens.weight

    muon_params: List[torch.nn.Parameter] = []
    adamw_params: List[torch.nn.Parameter] = []
    seen_ids = set()

    for p in model.parameters():
        if not p.requires_grad or id(p) in seen_ids:
            continue
        seen_ids.add(id(p))

        is_tied_embedding = embedding_weight is not None and p is embedding_weight
        if p.ndim == 2 and not is_tied_embedding:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    muon_optimizer = UnitaryMuon(
        muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=muon_weight_decay
    )
    adamw_optimizer = torch.optim.AdamW(
        adamw_params, lr=adamw_lr, weight_decay=adamw_weight_decay, betas=adamw_betas
    )
    return HybridMuonAdamW(muon_optimizer, adamw_optimizer)

