"""
Spectral State-Space Differential Attention (SSSD) Module
Implements Exact Lie-Algebra Skew-Symmetric Unitary State Updates.
Uses Cayley Transform R_t = (I - 0.5 * delta * A)^{-1} (I + 0.5 * delta * A) where A = k v^T - v k^T (Skew-Symmetric),
strictly conserving state norm and energy over infinite sequence horizons: ||Psi_t|| = ||Psi_0||.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    from verace_v1.serving.triton_kernels import HAS_TRITON
except ImportError:
    HAS_TRITON = False


def _solve_cayley(right: torch.Tensor, left: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    """
    Solves right @ R = left for R (the Cayley transform R_t = (I-0.5*delta*A)^-1(I+0.5*delta*A)).
    `right` is provably non-singular in exact arithmetic (its eigenvalues are 1 + 0.5*delta*i*mu
    for skew-symmetric A's purely-imaginary eigenvalues i*mu, so the real part is always exactly
    1) -- but cusolver's batched GPU LU solve can still occasionally return non-finite output for
    a batch of many small matrices (same class of driver edge case observed in
    stiefel_orthogonalize's SVD fallback; see verace_v1/optimizer/unitary_muon.py). A single such
    failure previously produced a hard, instant jump from healthy to fully-NaN hidden states mid
    -training with no gradual warning.

    Patching only the forward *output* of a bad solve (e.g. torch.where'ing in a safe value
    after the fact) is NOT sufficient: autograd differentiates the original solve node
    regardless of what happens to its output downstream, and torch.linalg.solve's backward is
    itself an independent adjoint solve that reuses the (already non-finite) primal result --
    so NaN leaks straight into d(loss)/d(right) and d(loss)/d(left) even when every consumer of
    the forward value only ever sees the cleaned-up output. (Confirmed directly: a near-singular
    `right` produces a NaN gradient on `right` even after fully replacing the forward output with
    a clean value via torch.where -- same class of bug as torch.norm's gradient at zero, one level
    removed.) The only robust fix is to substitute the *input* for the affected positions before
    solving at all, with a trivially well-conditioned pair (right=I, left=I => R=I, matching what
    R_t already reduces to when delta=0) so torch.linalg.solve's backward never touches whatever
    made the original matrix numerically pathological, for either the forward or the backward pass.
    """
    with torch.no_grad():
        probe = torch.linalg.solve(right, left)
        bad = ~torch.isfinite(probe)

    if not bad.any():
        return torch.linalg.solve(right, left)

    warnings.warn(
        f"SSSD: torch.linalg.solve returned non-finite values for {bad.any(dim=(-1, -2)).sum().item()} "
        f"of {right.shape[:-2].numel()} matrices (GPU solver edge case) -- substituting a "
        f"well-conditioned identity pair for those positions' INPUT (right=I, left=I => R=I, "
        f"equivalent to delta=0) and re-solving, so neither the forward value nor the gradient "
        f"ever touches the pathological matrix.",
        stacklevel=2
    )
    bad_matrix = bad.any(dim=-1, keepdim=True).any(dim=-2, keepdim=True).expand_as(right)
    safe_right = torch.where(bad_matrix, eye.expand_as(right), right)
    safe_left = torch.where(bad_matrix, eye.expand_as(left), left)
    R_seq = torch.linalg.solve(safe_right, safe_left)

    if not torch.isfinite(R_seq).all():
        warnings.warn(
            "SSSD: torch.linalg.solve still produced non-finite output after input "
            "substitution -- forcing remaining non-finite entries to 0 as a last resort.",
            stacklevel=2
        )
        R_seq = torch.nan_to_num(R_seq, nan=0.0, posinf=0.0, neginf=0.0)
    return R_seq


class SSSDAttention(nn.Module):
    """
    Spectral State-Space Differential Attention (SSSD).
    Guarantees strict norm conservation via Lie-Algebra Skew-Symmetric updates R_t in SO(d).
    """
    def __init__(
        self,
        hidden_dim: int = 16384,
        num_heads: int = 128,
        head_dim: int = 128,
        spectral_dim: int = 256
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.spectral_dim = spectral_dim

        self.w_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)

        # Write Coupling
        self.w_delta = nn.Linear(hidden_dim, num_heads, bias=False)

        self.head_rmsnorm = nn.RMSNorm(head_dim)
        self.w_out = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
        active_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [batch, seq_len, hidden_dim]
        active_mask: [batch, seq_len] - True for active tokens, False for halted tokens
        """
        b, s, d = x.shape

        q = F.normalize(self.w_q(x).view(b, s, self.num_heads, self.head_dim), p=2, dim=-1)
        k = F.normalize(self.w_k(x).view(b, s, self.num_heads, self.head_dim), p=2, dim=-1)
        v = F.normalize(self.w_v(x).view(b, s, self.num_heads, self.head_dim), p=2, dim=-1)

        delta = 0.1 * torch.sigmoid(self.w_delta(x)).view(b, s, self.num_heads, 1)

        # Zero-Delta Identity Recurrence for Halted Tokens: delta = 0 when active_mask is False
        if active_mask is not None:
            while active_mask.ndim < delta.ndim:
                active_mask = active_mask.unsqueeze(-1)
            delta = delta * active_mask.to(delta.dtype)

        # GPU Triton Path (Inference fast path)
        if x.is_cuda and HAS_TRITON and not x.requires_grad:
            from verace_v1.serving.triton_kernels import launch_sssd_triton_scan
            o_norm_triton, Psi_state = launch_sssd_triton_scan(q, k, v, delta.squeeze(-1), initial_state)
            o_norm = self.head_rmsnorm(o_norm_triton).view(b, s, self.num_heads * self.head_dim)
            output = self.w_out(o_norm)
            if return_state:
                return output, Psi_state
            return output, None

        # Exact Lie-Algebra Skew-Symmetric Unitary Update Path via O(log S) Parallel Prefix Scan
        # Forced to fp32 with autocast disabled: this path's entire purpose is an exact
        # orthogonality guarantee (||Psi_t|| conserved to floating-point precision), which
        # bf16's ~3 decimal digits cannot hold, and under active autocast torch.linalg.solve
        # is forced to fp32 while the subsequent torch.matmul calls are cast back down to
        # bf16, producing a dtype mismatch on the parallel scan's in-place writes.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q32, k32, v32, delta32 = q.float(), k.float(), v.float(), delta.float()

            if initial_state is not None:
                Psi_0 = initial_state.float()
            else:
                Psi_0 = torch.eye(self.head_dim, device=x.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(b, self.num_heads, 1, 1)

            eye = torch.eye(self.head_dim, device=x.device, dtype=torch.float32)

            # 1. Parallel construction of all skew-symmetric generators A_t and Cayley rotation matrices R_t
            kt_col = k32.transpose(1, 2).unsqueeze(-1) # [b, h, s, d_k, 1]
            vt_row = v32.transpose(1, 2).unsqueeze(-2) # [b, h, s, 1, d_k]
            vt_col = v32.transpose(1, 2).unsqueeze(-1)
            kt_row = k32.transpose(1, 2).unsqueeze(-2)

            A_seq = torch.matmul(kt_col, vt_row) - torch.matmul(vt_col, kt_row) # [b, h, s, d_k, d_k]
            d_seq = delta32.transpose(1, 2).unsqueeze(-1) # [b, h, s, 1, 1]

            scale_A = 0.5 * d_seq * A_seq
            left = eye - scale_A
            right = eye + scale_A
            R_seq = _solve_cayley(right, left, eye) # [b, h, s, d_k, d_k] Exact Orthogonal R_t

            # 2. Logarithmic O(log2 S) Associative Parallel Prefix Scan over matrix multiplications
            P_seq = parallel_prefix_scan(R_seq) # [b, h, s, d_k, d_k] P_t = R_t * R_{t-1} * ... * R_1

            # 3. Parallel state & read output computation across all timesteps
            # Psi_t = P_t * Psi_0
            Psi_seq = torch.matmul(P_seq, Psi_0.unsqueeze(2)) # [b, h, s, d_k, d_k]
            q_col = q32.transpose(1, 2).unsqueeze(-1) # [b, h, s, d_k, 1]
            o_seq = torch.matmul(Psi_seq, q_col).squeeze(-1).transpose(1, 2) # [b, s, h, d_k]

        o_norm = self.head_rmsnorm(o_seq.contiguous()).reshape(b, s, self.num_heads * self.head_dim)
        output = self.w_out(o_norm)

        if return_state:
            return output, Psi_seq[:, :, -1]
        return output, None


def parallel_prefix_scan(R_seq: torch.Tensor) -> torch.Tensor:
    """
    Computes associative prefix products P_t = R_t @ R_{t-1} ... @ R_1 in O(log2 S) parallel depth.
    R_seq shape: [batch, heads, seq_len, dim, dim]
    """
    b, h, s, d, _ = R_seq.shape
    if s == 1:
        return R_seq

    s_pad = 1 << (s - 1).bit_length()
    if s_pad > s:
        pad = torch.eye(d, device=R_seq.device, dtype=R_seq.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(b, h, s_pad - s, 1, 1)
        R_seq = torch.cat([R_seq, pad], dim=2)

    P = R_seq.clone()

    # Up-sweep (Reduce) phase
    step = 1
    while step < s_pad:
        idx_dst = torch.arange(2 * step - 1, s_pad, 2 * step, device=R_seq.device)
        idx_src = idx_dst - step
        P[:, :, idx_dst] = torch.matmul(P[:, :, idx_dst], P[:, :, idx_src])
        step *= 2

    # Down-sweep phase
    step = s_pad // 4
    while step > 0:
        idx_dst = torch.arange(3 * step - 1, s_pad, 2 * step, device=R_seq.device)
        idx_src = idx_dst - step
        P[:, :, idx_dst] = torch.matmul(P[:, :, idx_dst], P[:, :, idx_src])
        step //= 2

    return P[:, :, :s]
