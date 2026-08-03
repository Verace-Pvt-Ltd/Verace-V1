"""
Spectral State-Space Differential Attention (SSSD) Module
Implements Exact Lie-Algebra Skew-Symmetric Unitary State Updates.
Uses Cayley Transform R_t = (I - 0.5 * delta * A)^{-1} (I + 0.5 * delta * A) where A = k v^T - v k^T (Skew-Symmetric),
strictly conserving state norm and energy over infinite sequence horizons: ||Psi_t|| = ||Psi_0||.
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

        # Dynamic Phase Frequency Generator & Write Coupling
        self.w_omega = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
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

        omega = math.pi * torch.tanh(self.w_omega(x).view(b, s, self.num_heads, self.head_dim))
        delta = 0.1 * torch.sigmoid(self.w_delta(x)).view(b, s, self.num_heads, 1)

        # Zero-Delta Identity Recurrence for Halted Tokens: delta = 0 when active_mask is False
        if active_mask is not None:
            delta = delta * active_mask.unsqueeze(-1).unsqueeze(-1).to(delta.dtype)

        # GPU Triton Path
        if x.is_cuda and HAS_TRITON:
            from verace_v1.serving.triton_kernels import launch_sssd_triton_scan
            o_norm_triton, Psi_state = launch_sssd_triton_scan(q, k, v, delta.squeeze(-1), omega)
            o_norm = self.head_rmsnorm(o_norm_triton).view(b, s, self.num_heads * self.head_dim)
            output = self.w_out(o_norm)
            if return_state:
                return output, Psi_state
            return output, None

        # Exact Lie-Algebra Skew-Symmetric Unitary Update Path (PyTorch)
        if initial_state is not None:
            Psi = initial_state
        else:
            # Initialize to identity state (norm = 1.0 per column)
            Psi = torch.eye(self.head_dim, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0).repeat(b, self.num_heads, 1, 1)

        eye = torch.eye(self.head_dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(s):
            q_t = q[:, t]       # [b, h, d_k]
            k_t = k[:, t]       # [b, h, d_k]
            v_t = v[:, t]       # [b, h, d_k]
            d_t = delta[:, t]   # [b, h, 1]

            # 1. Construct Skew-Symmetric Lie Generator: A_t = k_t v_t^T - v_t k_t^T
            kt_col = k_t.unsqueeze(-1) # [b, h, d_k, 1]
            vt_row = v_t.unsqueeze(-2) # [b, h, 1, d_k]
            vt_col = v_t.unsqueeze(-1)
            kt_row = k_t.unsqueeze(-2)

            A_t = torch.matmul(kt_col, vt_row) - torch.matmul(vt_col, kt_row) # [b, h, d_k, d_k] (A^T = -A)

            # 2. Cayley Transform for Exact Unitary Rotation Matrix R_t \in SO(d_k):
            # R_t = (I - 0.5 * delta * A_t)^{-1} * (I + 0.5 * delta * A_t)
            scale_A = 0.5 * d_t.unsqueeze(-1) * A_t
            left = eye - scale_A
            right = eye + scale_A
            R_t = torch.linalg.solve(right, left) # R_t^T * R_t = I (Exact Orthogonal/Unitary)

            # 3. Exact Norm-Preserving State Rotation: Psi_t = R_t * Psi_{t-1}
            Psi = torch.matmul(R_t, Psi) # ||Psi_t||_F == ||Psi_{t-1}||_F (Exact Conservation!)

            # Read Output
            o_t = torch.matmul(Psi, q_t.unsqueeze(-1)).squeeze(-1) # [b, h, d_k]
            outputs.append(o_t)

        o_tilde = torch.stack(outputs, dim=1)
        o_norm = self.head_rmsnorm(o_tilde).view(b, s, self.num_heads * self.head_dim)
        output = self.w_out(o_norm)

        if return_state:
            return output, Psi
        return output, None
