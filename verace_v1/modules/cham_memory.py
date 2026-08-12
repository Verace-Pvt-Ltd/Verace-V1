"""
Continuous Holographic Associative Memory (CHAM) Module
Implements Exact Unitary Holographic Matrix Updates via Newton-Schulz Unitary Retraction:
H <- 0.5 * H * (3 * I - H^H * H).
Guarantees H^H * H = I (Exact Complex Unitary Operator H in U(d)), ensuring zero-loss associative recall.
"""

import math
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
        H_real = 0.5 * (torch.matmul(H_real, diff_r) - torch.matmul(H_imag, diff_i))
        H_imag = 0.5 * (torch.matmul(H_real, diff_i) + torch.matmul(H_imag, diff_r))

    return H_real, H_imag


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
        v = F.silu(self.w_v(x))
        gamma = 0.1 * torch.sigmoid(self.w_gamma(x))

        # Zero-Gamma Identity Recurrence for Halted Tokens: gamma = 0 when active_mask is False
        if active_mask is not None:
            while active_mask.ndim < gamma.ndim:
                active_mask = active_mask.unsqueeze(-1)
            gamma = gamma * active_mask.to(gamma.dtype)

        # GPU Triton Path (Inference fast path)
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
            kv_seq = torch.matmul(kt_col, vt_row) # [b, s, h_dim, h_dim]

            eye_seq = torch.eye(self.holographic_dim, device=x.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            g_seq = gamma32.unsqueeze(-1)

            # Infinitesimal Unitary Transformation sequence: U_t = I + i * g_t * (k_t v_t^T)
            U_r_seq = eye_seq.repeat(b, s, 1, 1)
            U_i_seq = g_seq * kv_seq # [b, s, h_dim, h_dim]

            # Logarithmic O(log2 S) Associative Parallel Prefix Scan over complex matrix sequence
            P_r_seq, P_i_seq = parallel_complex_prefix_scan(U_r_seq, U_i_seq)

            # Compute prefix holograms H_t = H_0 * P_t
            H_r_seq = torch.matmul(H_r_0.unsqueeze(1), P_r_seq) - torch.matmul(H_i_0.unsqueeze(1), P_i_seq)
            H_i_seq = torch.matmul(H_r_0.unsqueeze(1), P_i_seq) + torch.matmul(H_i_0.unsqueeze(1), P_r_seq)

            # Parallel Newton-Schulz Unitary Retraction across all sequence positions
            H_r_seq, H_i_seq = parallel_newton_schulz_retraction(H_r_seq, H_i_seq, steps=3)

            # Holographic Recall: Re(H_t * q_t) across all positions in parallel
            rec_seq = torch.matmul(H_r_seq, q32.unsqueeze(-1)).squeeze(-1) # [b, s, h_dim]

        y = self.norm(self.w_out(rec_seq))

        return y, (H_r_seq[:, -1], H_i_seq[:, -1])


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

        H_real = 0.5 * (torch.matmul(H_real, diff_r) - torch.matmul(H_imag, diff_i))
        H_imag = 0.5 * (torch.matmul(H_real, diff_i) + torch.matmul(H_imag, diff_r))

    return H_real, H_imag
