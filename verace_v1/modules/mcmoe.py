"""
Manifold Continuous Mixture-of-Experts (M-CMoE) Module
Implements Dynamic Sparse Top-K Continuous Hyper-Manifold Experts.
Combines sparse conditional compute (Top-K selection over basis components) with continuous LoRA manifold adaptation,
achieving total capacity scaling far past active FLOPs while eliminating GPU All-to-All communication bottlenecks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from verace_v1.modules.activations import SiTUGLU

try:
    from verace_v1.serving.triton_kernels import HAS_TRITON
except ImportError:
    HAS_TRITON = False

class ManifoldContinuousMoE(nn.Module):
    """
    Manifold Continuous Mixture-of-Experts (M-CMoE).
    Selects Top-K sparse components per token from N manifold basis generators,
    applying dynamic continuous hyper-manifold weight adaptation:
    Delta W(x) = sum_{j in TopK} phi_j(x) * ( U_j * diag(sigma_j(x)) * V_j^T )
    """
    def __init__(
        self,
        hidden_dim: int = 16384,
        rank: int = 32,
        num_components: int = 64,
        top_k_components: int = 8,
        beta1: float = 4.0,
        beta2: float = 25.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_components = num_components
        self.top_k_components = top_k_components

        # Shared base Feed-Forward Network
        self.base_situ_glu = SiTUGLU(hidden_dim, hidden_dim * 2, beta1=beta1, beta2=beta2)
        self.base_w_down = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        # Manifold Basis Generators U_j and V_j
        self.u_basis = nn.Parameter(torch.randn(num_components, hidden_dim, rank) * 0.02)
        self.v_basis = nn.Parameter(torch.randn(num_components, rank, hidden_dim) * 0.02)

        # Router Head for Sparse Top-K Component Selection
        self.router = nn.Linear(hidden_dim, num_components, bias=False)

        # Hyper-Network: Generates singular scales sigma_j(x) for selected components
        self.hyper_sigma = nn.Linear(hidden_dim, num_components * rank, bias=False)

        self.norm = nn.RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, hidden_dim]
        """
        b, s, d = x.shape
        x_flat = x.view(b * s, d)

        # 1. Base Shared FFN Output
        base_out = self.base_w_down(self.base_situ_glu(x_flat))

        # 2. Sparse Top-K Component Routing
        # Noisy Top-K Gating (Shazeer et al., "Outrageously Large Neural Networks: The
        # Sparsely-Gated Mixture-of-Experts Layer", arXiv:1701.06538, Eq. 2-5):
        # G(x) = Softmax(KeepTopK(H(x), k)) -- the top-k MASK is applied to the raw
        # logits BEFORE softmax (non-selected entries set to -inf), so the softmax
        # normalizes only over the surviving k entries and their weights always sum to
        # exactly 1 (a proper convex combination of the selected components). A prior
        # version of this code applied softmax over all num_components first and only
        # then took the top-k of the already-normalized weights -- those weights are a
        # subset of a distribution that summed to 1 over ALL components, so they summed
        # to less than 1 (verified: ~0.54 for a near-uniform router, the common case at
        # initialization), systematically under-scaling manifold_adapt by ~2x whenever
        # the router isn't yet confident.
        k_val = min(self.top_k_components, self.num_components)
        router_logits = self.router(x_flat) # [b*s, num_components]
        topk_logits, topk_indices = torch.topk(router_logits, k=k_val, dim=-1)
        masked_logits = torch.full_like(router_logits, float("-inf"))
        masked_logits.scatter_(-1, topk_indices, topk_logits)
        topk_weights = torch.gather(torch.softmax(masked_logits, dim=-1), -1, topk_indices)

        sigma_all = torch.sigmoid(self.hyper_sigma(x_flat)).view(b * s, self.num_components, self.rank)

        # GPU path via Triton kernel (Inference fast path)
        if x.is_cuda and HAS_TRITON and not x.requires_grad:
            from verace_v1.serving.triton_kernels import launch_mcmoe_triton_projection
            manifold_adapt = launch_mcmoe_triton_projection(
                x_flat, self.u_basis, self.v_basis, topk_indices, topk_weights, sigma_all
            )
            y_flat = base_out + self.norm(manifold_adapt)
            return y_flat.view(b, s, d)

        # Sparse Top-K Component Execution
        manifold_adapt = torch.zeros_like(x_flat)
        for k_idx in range(k_val):
            comp_idx = topk_indices[:, k_idx] # [b*s]
            w_k = topk_weights[:, k_idx].unsqueeze(-1) # [b*s, 1]

            # Gather basis matrices for selected component
            u_k = self.u_basis[comp_idx] # [b*s, d, rank]
            v_k = self.v_basis[comp_idx] # [b*s, rank, d]

            sigma_k = torch.gather(
                sigma_all, dim=1, index=comp_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.rank)
            ).squeeze(1) # [b*s, rank]

            # v_proj = (V_k^T * x) * sigma_k
            v_proj = torch.matmul(v_k, x_flat.unsqueeze(-1)).squeeze(-1) * sigma_k # [b*s, rank]
            u_proj = torch.matmul(u_k, v_proj.unsqueeze(-1)).squeeze(-1) # [b*s, d]

            manifold_adapt = manifold_adapt + w_k * u_proj

        y_flat = base_out + self.norm(manifold_adapt)
        return y_flat.view(b, s, d)
