"""
Latent Tree-Diffusion & Energy Critic Module
Implements Energy-Based Latent Tree Verification using quadratic energy:
E(x, y) = ||h_cand - W_energy * x||_2^2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class LatentEnergyCritic(nn.Module):
    """
    Quadratic Energy Critic for Latent Thought Branches.
    Computes E(x, y) = ||h_cand - W_energy * x||_2^2.
    Lower energy score indicates higher logical consistency.
    """
    def __init__(self, hidden_dim: int = 16384):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_energy = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def compute_energy(self, x_prompt: torch.Tensor, h_cand: torch.Tensor) -> torch.Tensor:
        """
        x_prompt: [batch, seq_len, hidden_dim]
        h_cand: [batch, num_branches, seq_len, hidden_dim]
        Returns energy scores: [batch, num_branches]
        """
        proj_x = self.w_energy(x_prompt).unsqueeze(1) # [b, 1, s, d]
        diff = h_cand - proj_x # [b, branches, s, d]
        # ||diff||^2 computed directly as sum(diff^2), NOT torch.norm(diff)**2: norm()'s
        # gradient is diff/||diff||, which is 0/0 = NaN whenever diff is exactly (or
        # numerically very close to) zero -- observed in practice during real training,
        # ~95 steps in, once a candidate's predicted state started closely matching the
        # target. Mathematically identical forward value, but sum-of-squares has a
        # gradient (2*diff) that's well-defined everywhere, including at diff=0.
        energy_tokens = (diff ** 2).sum(dim=-1) # [b, branches, s]
        energy_score = torch.mean(energy_tokens, dim=-1) # [b, branches]
        return energy_score

    def select_best_thought_branch(
        self,
        x_prompt: torch.Tensor,
        candidate_thoughts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        energies = self.compute_energy(x_prompt, candidate_thoughts)
        best_branch_idx = torch.argmin(energies, dim=-1)

        b, branches, s, d = candidate_thoughts.shape
        best_thought = torch.stack([
            candidate_thoughts[i, best_branch_idx[i]]
            for i in range(b)
        ], dim=0)

        return best_thought, best_branch_idx
