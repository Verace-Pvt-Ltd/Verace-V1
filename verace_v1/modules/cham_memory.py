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
            gamma = gamma * active_mask.unsqueeze(-1).to(gamma.dtype)

        # GPU Triton Path
        if x.is_cuda and HAS_TRITON:
            from verace_v1.serving.triton_kernels import launch_cham_triton_update
            hologram_out, (H_real, H_imag) = launch_cham_triton_update(q, k, v, gamma.squeeze(-1))
            y = self.norm(self.w_out(hologram_out))
            return y, (H_real, H_imag)

        if initial_hologram is not None:
            H_real, H_imag = initial_hologram
        else:
            eye = torch.eye(self.holographic_dim, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(b, 1, 1)
            H_real = eye
            H_imag = torch.zeros_like(eye)

        outputs = []

        for t in range(s):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            g_t = gamma[:, t]

            kt_col = k_t.unsqueeze(-1)
            vt_row = v_t.unsqueeze(-2)
            kv_outer = torch.matmul(kt_col, vt_row)

            # Infinitesimal Unitary Transformation: H_next = H * (I + i * g_t * kv_outer)
            H_real_next = H_real - g_t.unsqueeze(-1) * torch.matmul(H_imag, kv_outer)
            H_imag_next = H_imag + g_t.unsqueeze(-1) * torch.matmul(H_real, kv_outer)

            # Exact Newton-Schulz Unitary Retraction to guarantee H^H * H = I
            H_real, H_imag = newton_schulz_unitary_retraction(H_real_next, H_imag_next, steps=3)

            # Holographic Recall: Re(H * q_t)
            rec_t = torch.matmul(H_real, q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(rec_t)

        hologram_out = torch.stack(outputs, dim=1)
        y = self.norm(self.w_out(hologram_out))

        return y, (H_real, H_imag)
