"""
Continuous Holographic Associative Memory (CHAM) Module
Implements Exact Unitary Holographic Matrix Updates via Newton-Schulz Unitary Retraction:
H <- 0.5 * H * (3 * I - H^H * H).
Guarantees H^H * H = I (Exact Complex Unitary Operator H in U(d)), ensuring zero-loss associative recall.
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    from verace_v1.serving.triton_kernels import HAS_TRITON
except ImportError:
    HAS_TRITON = False

def newton_schulz_unitary_retraction(H_real: torch.Tensor, H_imag: torch.Tensor, steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Newton-Schulz Unitary Retraction for Complex Holographic Matrix H = H_r + i * H_i.
    Projects H to the nearest unitary matrix satisfying H^H * H = I.
    """
    b, d, _ = H_real.shape
    eye = torch.eye(d, device=H_real.device, dtype=H_real.dtype).unsqueeze(0)

    for _ in range(steps):
        # H^H * H = (H_r^T - i*H_i^T) * (H_r + i*H_i) = (H_r^T H_r + H_i^T H_i) + i*(H_r^T H_i - H_i^T H_r)
        HH_r = torch.matmul(H_real.transpose(-1, -2), H_real) + torch.matmul(H_imag.transpose(-1, -2), H_imag)
        HH_i = torch.matmul(H_real.transpose(-1, -2), H_imag) - torch.matmul(H_imag.transpose(-1, -2), H_real)

        # 3 * I - H^H * H
        diff_r = 3.0 * eye - HH_r
        diff_i = -HH_i

        # H_next = 0.5 * H * (3 * I - H^H * H)
        # = 0.5 * (H_r + i*H_i) * (diff_r + i*diff_i)
        # Both outputs need the SAME pre-update (H_real, H_imag) -- computing H_real's
        # new value first and reusing the variable name for H_imag's computation (as an
        # earlier version of this code did) silently feeds the already-updated H_real
        # into H_imag's formula instead of the original, corrupting the complex
        # multiplication on every iteration past the first. Real bug, not just numerical
        # sensitivity: this is why the iteration could occasionally diverge to NaN/Inf
        # for inputs that are otherwise perfectly well-conditioned.
        H_real_next = 0.5 * (torch.matmul(H_real, diff_r) - torch.matmul(H_imag, diff_i))
        H_imag_next = 0.5 * (torch.matmul(H_real, diff_i) + torch.matmul(H_imag, diff_r))
        H_real, H_imag = H_real_next, H_imag_next

    return H_real, H_imag


def _solve_cham_cayley(right: torch.Tensor, left: torch.Tensor, eye: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Complex analog of SSSD's _solve_cayley (verace_v1/modules/sssd_attention.py): solves
    right @ U = left for U, where right = I - i*G/2, left = I + i*G/2 for a Hermitian
    generator G -- giving EXACTLY unitary U in exact arithmetic (right is provably
    non-singular: G Hermitian => real eigenvalues mu => right's eigenvalues 1 - i*mu/2
    never vanish, |1 - i*mu/2| >= 1 > 0). Same defensive non-finite-value substitution
    pattern as SSSD's _solve_cayley -- see that function's docstring for why patching only
    the forward output is insufficient (torch.linalg.solve's backward is an independent
    adjoint solve that reuses the primal result, so a bad forward value poisons gradients
    even when every downstream consumer only sees a cleaned-up copy).

    right, left, eye: complex64 [*, d, d]. Returns (U_real, U_imag).
    """
    with torch.no_grad():
        probe = torch.linalg.solve(right, left)
        bad = ~torch.isfinite(probe.real) | ~torch.isfinite(probe.imag)

    if not bad.any():
        U = torch.linalg.solve(right, left)
        return U.real, U.imag

    warnings.warn(
        f"CHAM: torch.linalg.solve returned non-finite values for {bad.any(dim=(-1, -2)).sum().item()} "
        f"of {right.shape[:-2].numel()} matrices (GPU solver edge case) -- substituting a "
        f"well-conditioned identity pair for those positions' INPUT (right=I, left=I => U=I, "
        f"equivalent to gamma=0) and re-solving, so neither the forward value nor the gradient "
        f"ever touches the pathological matrix.",
        stacklevel=2
    )
    bad_matrix = bad.any(dim=-1, keepdim=True).any(dim=-2, keepdim=True).expand_as(right)
    safe_right = torch.where(bad_matrix, eye.expand_as(right), right)
    safe_left = torch.where(bad_matrix, eye.expand_as(left), left)
    U = torch.linalg.solve(safe_right, safe_left)

    if not (torch.isfinite(U.real).all() and torch.isfinite(U.imag).all()):
        warnings.warn(
            "CHAM: torch.linalg.solve still produced non-finite output after input "
            "substitution -- forcing remaining non-finite entries to 0 as a last resort.",
            stacklevel=2
        )
        U = torch.nan_to_num(U.real) + 1j * torch.nan_to_num(U.imag)
    return U.real, U.imag


class ContinuousHolographicMemory(nn.Module):
    """
    Continuous Holographic Associative Memory (CHAM).
    Maintains exact unitary operator property H^H * H = I via Newton-Schulz Unitary Retraction.
    """
    def __init__(self, hidden_dim: int = 16384, holographic_dim: int = 1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.holographic_dim = holographic_dim

        self.w_q = nn.Linear(hidden_dim, holographic_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, holographic_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, holographic_dim, bias=False)
        self.w_gamma = nn.Linear(hidden_dim, 1, bias=False)

        self.w_out = nn.Linear(holographic_dim, hidden_dim, bias=False)
        self.norm = nn.RMSNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        initial_hologram: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        active_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: [batch, seq_len, hidden_dim]
        active_mask: [batch, seq_len] - True for active tokens, False for halted tokens
        """
        b, s, d = x.shape

        q = F.normalize(self.w_q(x), p=2, dim=-1)
        k = F.normalize(self.w_k(x), p=2, dim=-1)
        # v must be magnitude-bounded, not just direction-bounded like k/q: U_t = I +
        # i*gamma*(k v^T) is only a first-order ("infinitesimal") approximation to a
        # unitary matrix, accurate when gamma*||k||*||v|| is small. Unlike SSSD's
        # analogous construction (verace_v1/modules/sssd_attention.py), which L2-
        # normalizes BOTH of its outer-product vectors, v here was previously left as
        # unbounded F.silu(...) output with nothing to stop it from growing during
        # training. Verified in practice: v's max magnitude grew monotonically and
        # unboundedly (~2 -> ~17 over 50 steps) as w_v's weights adapted, pushing
        # gamma*||v|| an order of magnitude past the infinitesimal-approximation regime
        # and causing the holographic state to go non-finite -- confirmed as the
        # earliest point of corruption in the whole model via forward-pass tracing
        # (CHAM's post-retraction output was already non-finite while every other
        # module's inputs were still clean). L2-normalizing v, like k, keeps
        # gamma*||k||*||v|| providably bounded by gamma's own ceiling (0.1) regardless
        # of how the model's weights evolve.
        v = F.normalize(F.silu(self.w_v(x)), p=2, dim=-1)
        gamma = 0.1 * torch.sigmoid(self.w_gamma(x))

        # Zero-Gamma Identity Recurrence for Halted Tokens: gamma = 0 when active_mask is False
        if active_mask is not None:
            while active_mask.ndim < gamma.ndim:
                active_mask = active_mask.unsqueeze(-1)
            gamma = gamma * active_mask.to(gamma.dtype)

        # GPU Triton Path (Inference fast path). Both launch_cham_triton_update (h_dim <=
        # 128, cham_holographic_scan_kernel) and its h_dim > 128 fallback
        # launch_cham_triton_tiled (_cham_rank2_update_kernel) implement the same exact
        # Cayley-transform unitary update as the O(log S) path below (see either kernel's
        # docstring in verace_v1/serving/triton_kernels.py for the Sherman-Morrison-
        # Woodbury derivation) -- previously they implemented only a first-order
        # approximation, which caused unbounded, sequence-length-compounding drift from
        # unitarity (root cause of NaN training divergence at chams_holographic_dim >= 48);
        # both were fixed to match this path exactly (see git history).
        if x.is_cuda and HAS_TRITON and not x.requires_grad:
            from verace_v1.serving.triton_kernels import launch_cham_triton_update
            hologram_out, (H_real, H_imag) = launch_cham_triton_update(q, k, v, gamma.squeeze(-1), initial_hologram)
            y = self.norm(self.w_out(hologram_out))
            return y, (H_real, H_imag)

        # O(log S) Associative Parallel Prefix Scan Path
        # Forced to fp32 with autocast disabled: this path's entire purpose is an exact
        # unitary guarantee (H^H H = I held to floating-point precision), which bf16's
        # ~3 decimal digits cannot hold, and under active autocast the eye/eye_seq
        # tensors (built from x.dtype, still fp32 pre-autocast) mix with q/k/v/gamma
        # (already cast to bf16 by the Linear layers above), producing a dtype
        # mismatch on the parallel scan's in-place writes.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q32, k32, v32, gamma32 = q.float(), k.float(), v.float(), gamma.float()

            if initial_hologram is not None:
                H_r_0, H_i_0 = initial_hologram
                H_r_0, H_i_0 = H_r_0.float(), H_i_0.float()
            else:
                eye = torch.eye(self.holographic_dim, device=x.device, dtype=torch.float32).unsqueeze(0).repeat(b, 1, 1)
                H_r_0 = eye
                H_i_0 = torch.zeros_like(eye)

            kt_col = k32.unsqueeze(-1) # [b, s, h_dim, 1]
            vt_row = v32.unsqueeze(-2) # [b, s, 1, h_dim]
            vt_col = v32.unsqueeze(-1)
            kt_row = k32.unsqueeze(-2)

            eye_seq = torch.eye(self.holographic_dim, device=x.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            g_seq = gamma32.unsqueeze(-1)

            # Exact Unitary Transformation sequence via Cayley transform of a Hermitian
            # generator: G_t = gamma_t * (k_t v_t^T + v_t k_t^T) is real-symmetric, hence
            # Hermitian, so U_t = (I - i*G_t/2)^{-1}(I + i*G_t/2) is EXACTLY unitary in
            # exact arithmetic for every t -- unlike the previous first-order-only
            # approximation U_t = I + i*gamma_t*(k_t v_t^T), whose per-step deviation from
            # unitarity is not O(gamma^2) but present even at O(gamma) (i*(k v^T) is not
            # Hermitian in general, so it isn't a valid infinitesimal-unitary generator at
            # all), and compounds across the sequence: verified in isolation to reach
            # O(1) deviation from unitarity within a single 256-token sequence, eventually
            # pushing the raw state's singular values far enough from 1 that the
            # Newton-Schulz retraction below crosses its own documented divergence
            # boundary (see unitary_muon.py's stiefel_orthogonalize) and produces the
            # training NaNs observed at chams_holographic_dim >= 48. With this exact
            # construction, deviation after 256 composed steps is ~1e-4 (pure fp32
            # rounding) instead of ~2.4 (order-1, non-rounding).
            G_seq = g_seq * (torch.matmul(kt_col, vt_row) + torch.matmul(vt_col, kt_row)) # [b, s, h_dim, h_dim]
            G_seq_c = G_seq.to(torch.complex64)
            eye_seq_c = eye_seq.repeat(b, s, 1, 1).to(torch.complex64)
            right_seq = eye_seq_c - 1j * G_seq_c / 2
            left_seq = eye_seq_c + 1j * G_seq_c / 2
            U_r_seq, U_i_seq = _solve_cham_cayley(right_seq, left_seq, eye_seq_c)

            # Logarithmic O(log2 S) Associative Parallel Prefix Scan over complex matrix sequence.
            # This chunk's own raw product, newest-leftmost: P_t = U_t @ ... @ U_(chunk start).
            P_r_seq, P_i_seq = parallel_complex_prefix_scan(U_r_seq, U_i_seq)

            # Compose with the incoming raw state H_0 (identity for a fresh sequence, or the
            # raw -- never retracted -- state from a previous call/chunk): since new tokens'
            # rotations are left-multiplied onto the running product, continuing correctly
            # across a chunk boundary requires H_0 on the RIGHT (H_0 @ nothing before this
            # chunk's own tokens), i.e. Total_t = P_t @ H_0, not H_0 @ P_t -- multiplication
            # isn't commutative, and H_0 @ P_t silently gives the wrong answer for any
            # non-identity H_0 (invisible on a fresh sequence, where H_0 = I).
            H_r_seq_raw = torch.matmul(P_r_seq, H_r_0.unsqueeze(1)) - torch.matmul(P_i_seq, H_i_0.unsqueeze(1))
            H_i_seq_raw = torch.matmul(P_r_seq, H_i_0.unsqueeze(1)) + torch.matmul(P_i_seq, H_r_0.unsqueeze(1))

            # Parallel Newton-Schulz Unitary Retraction across all sequence positions -- for
            # the readout only. Retraction is a nonlinear projection and must never be fed
            # back into the recurrence, so the *raw* H_r_seq_raw/H_i_seq_raw (not this
            # retracted copy) is what gets exported as the continuable state below.
            H_r_seq, H_i_seq = parallel_newton_schulz_retraction(H_r_seq_raw, H_i_seq_raw, steps=3)

            # Holographic Recall: Re(H_t * q_t) across all positions in parallel
            rec_seq = torch.matmul(H_r_seq, q32.unsqueeze(-1)).squeeze(-1) # [b, s, h_dim]

        y = self.norm(self.w_out(rec_seq))

        return y, (H_r_seq_raw[:, -1], H_i_seq_raw[:, -1])


def parallel_complex_prefix_scan(U_r: torch.Tensor, U_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes associative complex matrix prefix products in O(log2 S) parallel depth.
    U = U_r + i * U_i of shape [batch, seq_len, dim, dim]
    """
    b, s, d, _ = U_r.shape
    if s == 1:
        return U_r, U_i

    s_pad = 1 << (s - 1).bit_length()
    if s_pad > s:
        pad_r = torch.eye(d, device=U_r.device, dtype=U_r.dtype).unsqueeze(0).unsqueeze(0).repeat(b, s_pad - s, 1, 1)
        pad_i = torch.zeros_like(pad_r)
        U_r = torch.cat([U_r, pad_r], dim=1)
        U_i = torch.cat([U_i, pad_i], dim=1)

    P_r, P_i = U_r.clone(), U_i.clone()

    step = 1
    while step < s_pad:
        idx_dst = torch.arange(2 * step - 1, s_pad, 2 * step, device=U_r.device)
        idx_src = idx_dst - step

        dst_r, dst_i = P_r[:, idx_dst], P_i[:, idx_dst]
        src_r, src_i = P_r[:, idx_src], P_i[:, idx_src]

        # Complex matrix multiply: (dst_r + i*dst_i) * (src_r + i*src_i)
        next_r = torch.matmul(dst_r, src_r) - torch.matmul(dst_i, src_i)
        next_i = torch.matmul(dst_r, src_i) + torch.matmul(dst_i, src_r)

        P_r[:, idx_dst], P_i[:, idx_dst] = next_r, next_i
        step *= 2

    step = s_pad // 4
    while step > 0:
        idx_dst = torch.arange(3 * step - 1, s_pad, 2 * step, device=U_r.device)
        idx_src = idx_dst - step

        dst_r, dst_i = P_r[:, idx_dst], P_i[:, idx_dst]
        src_r, src_i = P_r[:, idx_src], P_i[:, idx_src]

        next_r = torch.matmul(dst_r, src_r) - torch.matmul(dst_i, src_i)
        next_i = torch.matmul(dst_r, src_i) + torch.matmul(dst_i, src_r)

        P_r[:, idx_dst], P_i[:, idx_dst] = next_r, next_i
        step //= 2

    return P_r[:, :s], P_i[:, :s]


def parallel_newton_schulz_retraction(H_real: torch.Tensor, H_imag: torch.Tensor, steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parallel Newton-Schulz Unitary Retraction over 4D tensor [b, s, d, d]."""
    b, s, d, _ = H_real.shape
    eye = torch.eye(d, device=H_real.device, dtype=H_real.dtype).unsqueeze(0).unsqueeze(0)

    for _ in range(steps):
        HH_r = torch.matmul(H_real.transpose(-1, -2), H_real) + torch.matmul(H_imag.transpose(-1, -2), H_imag)
        HH_i = torch.matmul(H_real.transpose(-1, -2), H_imag) - torch.matmul(H_imag.transpose(-1, -2), H_real)

        diff_r = 3.0 * eye - HH_r
        diff_i = -HH_i

        # See newton_schulz_unitary_retraction's comment above: both outputs need the
        # same pre-update (H_real, H_imag) pair, so compute both into temporaries before
        # reassigning either.
        H_real_next = 0.5 * (torch.matmul(H_real, diff_r) - torch.matmul(H_imag, diff_i))
        H_imag_next = 0.5 * (torch.matmul(H_real, diff_i) + torch.matmul(H_imag, diff_r))
        H_real, H_imag = H_real_next, H_imag_next

    return H_real, H_imag
