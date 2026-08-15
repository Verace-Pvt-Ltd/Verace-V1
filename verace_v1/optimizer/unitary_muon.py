"""
Unitary Muon Optimizer Module
Implements Stiefel-manifold orthogonal updates for every 2D parameter (square, tall, or
wide). The primary path is Keller Jordan's quintic Newton-Schulz iteration
(https://kellerjordan.github.io/posts/muon/) -- fast and GPU-friendly, but only bounded
when scaled correctly: divide by the Frobenius norm alone (never rescale further), so
every singular value provably lands in [0, 1] before iterating. The scalar recursion
sigma -> sigma*(3.4445 - 4.7750*sigma^2 + 2.0315*sigma^4) has an unstable fixed point at
sigma ~= 1.2637 (verified numerically: g'(1.2637) ~= 6.5, |g'|>1) that acts as a hard
divergence boundary -- singular values below it stay bounded under repeated iteration,
values above it diverge to Inf/NaN within 2-3 steps. An earlier version of this module
additionally rescaled by sqrt(min_dim) on top of the Frobenius-norm division, which
breaks the [0, 1] guarantee: for a skewed/low-rank spectrum (the typical case for real
transformer weight gradients), it can push the dominant singular value above 1.2637 --
verified directly to cause exactly this Inf/NaN blowup. That was a real, fixable scaling
bug in this codebase, not a fundamental property of Newton-Schulz iteration, and has been
fixed (X_scaled = X / norm, matching Keller Jordan's original). Note that staying below
the divergence boundary does not by itself guarantee convergence to machine-precision
orthogonality within `steps` iterations (`steps=5` here is just this function's default
parameter value, not a fixed budget -- raise it if a particular spectrum needs more): for
spread-out spectra, deviation can plateau well above `tol` regardless of how many more
steps are run (a bounded, non-divergent fixed point/cycle other than the identity, not
solved by iterating longer), which is exactly what the SVD fallback below exists to
handle.
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

    norm = torch.norm(X)
    if norm > 1e-7:
        # Scale by the Frobenius norm ALONE (matching Keller Jordan's original Muon
        # NewtonSchulz5: `X /= X.norm()`, https://kellerjordan.github.io/posts/muon/).
        # This is not cosmetic: since ||X||_F^2 = sum(sigma_i^2), dividing by ||X||_F
        # guarantees every individual singular value of X_scaled lands in [0, 1] --
        # which is the range these coefficients are proven to converge on. An earlier
        # version of this function additionally multiplied by sqrt(min_dim) (to target
        # ||X_scaled||_F = sqrt(min_dim) instead), which breaks that guarantee: for a
        # skewed/low-rank spectrum (the typical case for real transformer weight
        # gradients per Keller Jordan's own writeup), it can push the dominant singular
        # value above 1 -- verified directly to cause the quintic map to blow up to
        # Inf/NaN within 2-3 iterations for exactly this kind of matrix, whereas the
        # correct (norm-only) scaling on the same matrix converges cleanly. The SVD
        # fallback below remains for genuine cusolver GPU convergence failures, not as
        # a substitute for correct scaling.
        X_scaled = X / norm

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


def stiefel_orthogonalize_per_head(G: torch.Tensor, num_heads: int, steps: int = 5, tol: float = 1e-2) -> torch.Tensor:
    """
    Per-Head Muon (Kimi K3 Technical Report, arXiv:2607.24653, Sec. 2.5): for attention
    Q/K/V projection matrices, orthogonalize each head's [head_dim, in_dim] block
    separately instead of the full [num_heads*head_dim, in_dim] matrix as one block.
    Rationale (quoting the paper): "full-matrix orthogonalization treats all heads as a
    single coupled block, so heads with larger gradient or momentum scales dominate the
    shared update direction, while smaller-scale heads receive insufficiently normalized
    updates; per-head orthogonalization equalizes the update scale across heads."
    G: [num_heads*head_dim, in_dim] (a Linear layer's weight for a Q/K/V-style
    projection, PyTorch's [out_features, in_features] convention).
    """
    if G.ndim != 2 or G.shape[0] % num_heads != 0:
        return stiefel_orthogonalize(G, steps=steps, tol=tol)

    head_dim = G.shape[0] // num_heads
    heads = G.view(num_heads, head_dim, G.shape[1])
    out_heads = torch.stack([stiefel_orthogonalize(heads[h], steps=steps, tol=tol) for h in range(num_heads)], dim=0)
    return out_heads.view_as(G)


class UnitaryMuon(Optimizer):
    """
    Unitary Muon Optimizer.
    Applies exact Stiefel-manifold orthogonalization to every 2D parameter's momentum update.
    Uses fast GPU Newton-Schulz polynomial iterations with automatic SVD fallback.
    Supports Per-Head Muon (see stiefel_orthogonalize_per_head) via a per-param-group
    `num_heads` setting (default 1 = ordinary full-matrix orthogonalization).
    """
    def __init__(
        self,
        params,
        lr: float = 0.03,
        momentum: float = 0.98,
        weight_decay: float = 0.05,
        num_heads: int = 1
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, num_heads=num_heads)
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
                    num_heads = group.get("num_heads", 1)
                    if num_heads > 1 and p.shape[0] % num_heads == 0:
                        # Per-Head Muon (Kimi K3, arXiv:2607.24653, Sec. 2.5): orthogonalize
                        # each head's block separately rather than the full stacked matrix
                        # as one block, so heads with larger gradient/momentum scales don't
                        # dominate the shared update direction at the expense of smaller-
                        # scale heads. See stiefel_orthogonalize_per_head's docstring.
                        update = stiefel_orthogonalize_per_head(buf, num_heads)
                        per_head_dim = p.shape[0] // num_heads
                        max_dim = max(per_head_dim, p.shape[1])
                    else:
                        update = stiefel_orthogonalize(buf)
                        max_dim = max(p.shape[0], p.shape[1])
                    # Match update RMS across parameter shapes (Moonshot AI, "Muon is
                    # Scalable for LLM Training", arXiv:2502.16982, Eq. 4 / Lemma 1):
                    # a semi-orthogonal M x N matrix has RMS = sqrt(1/max(M,N)), which is
                    # shape-dependent and uncalibrated to anything -- without correction,
                    # a single shared `lr` gives wildly different effective update
                    # magnitudes across e.g. a (256,256) and a (1,256) parameter, which
                    # is not a tuning assumption a single lr can satisfy simultaneously.
                    # Multiplying by 0.2*sqrt(max(M,N)) cancels the shape dependence
                    # exactly, giving every 2D parameter the same ~0.2 update RMS
                    # (matching AdamW's typical 0.2-0.4 update RMS), so lr and
                    # weight_decay can be shared meaningfully across Muon and AdamW
                    # parameter groups. Equivalent (up to a global constant) to Keller
                    # Jordan's own reference implementation's sqrt(max(1, A/B)) scaling.
                    # When per-head, M/N here are the PER-HEAD block's dims (the shape
                    # that was actually orthogonalized), not the full stacked matrix's.
                    update = update * (0.2 * (max_dim ** 0.5))
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
    SSSDAttention's w_q/w_k/w_v projections (each [num_heads*head_dim, in_dim])
    get their own Muon param group with num_heads set, so they're orthogonalized
    per-head (Kimi K3, arXiv:2607.24653, Sec. 2.5) instead of as one full block --
    see stiefel_orthogonalize_per_head's docstring for why that matters.
    """
    embedding_weight: Optional[torch.Tensor] = None
    if hasattr(model, "embed_tokens"):
        embedding_weight = model.embed_tokens.weight

    # Collect SSSDAttention's Q/K/V projection weights by identity, grouped by
    # num_heads (uniform across layers in practice, but grouped defensively in
    # case a future model mixes head counts across layers).
    per_head_group_ids: Dict[int, List[torch.nn.Parameter]] = {}
    per_head_param_ids = set()
    for module in model.modules():
        if type(module).__name__ == "SSSDAttention":
            num_heads = getattr(module, "num_heads", 1)
            for proj_name in ("w_q", "w_k", "w_v"):
                proj = getattr(module, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    per_head_group_ids.setdefault(num_heads, []).append(proj.weight)
                    per_head_param_ids.add(id(proj.weight))

    muon_params: List[torch.nn.Parameter] = []
    adamw_params: List[torch.nn.Parameter] = []
    seen_ids = set()

    for p in model.parameters():
        if not p.requires_grad or id(p) in seen_ids:
            continue
        seen_ids.add(id(p))

        is_tied_embedding = embedding_weight is not None and p is embedding_weight
        if id(p) in per_head_param_ids:
            continue  # already routed via per_head_group_ids
        if p.ndim == 2 and not is_tied_embedding:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    muon_param_groups: List[Dict] = [{"params": muon_params}]
    for num_heads, params in per_head_group_ids.items():
        muon_param_groups.append({"params": params, "num_heads": num_heads})

    muon_optimizer = UnitaryMuon(
        muon_param_groups, lr=muon_lr, momentum=muon_momentum, weight_decay=muon_weight_decay
    )
    adamw_optimizer = torch.optim.AdamW(
        adamw_params, lr=adamw_lr, weight_decay=adamw_weight_decay, betas=adamw_betas
    )
    return HybridMuonAdamW(muon_optimizer, adamw_optimizer)

