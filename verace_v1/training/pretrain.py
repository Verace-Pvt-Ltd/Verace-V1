"""
Verace V1 Pre-Training Module
Reference pretraining step: next-token cross-entropy, an adaptive-depth compute
penalty, and a latent energy-consistency penalty (via LatentEnergyCritic) between
consecutive hidden states — optimized with the Unitary Muon optimizer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.optimizer.unitary_muon import UnitaryMuon

def train_pretrain_step(
    model: VeraceV1Model,
    optimizer: UnitaryMuon,
    batch: dict,
    depth_penalty_weight: float = 0.001,
    energy_penalty_weight: float = 0.01
) -> Tuple[float, float]:
    """
    Pre-training step minimizing:
        ce_loss + depth_penalty_weight * mean(depth_counts)
                + energy_penalty_weight * mean(E(h_{t-1}, h_t))
    where E is LatentEnergyCritic.compute_energy, applied between each position's
    hidden state and the next — see docs/modules/energy-critic.md and
    docs/training/pretraining.md.
    """
    model.train()
    optimizer.zero_grad()

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    images = batch.get("image", None)

    logits, depth_counts, hidden = model(
        input_ids, images=images, use_adaptive_depth=True, return_hidden=True
    )

    # Next token prediction loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )

    # Adaptive Cognitive Depth Penalty Loss (encourages compute efficiency)
    depth_loss = depth_penalty_weight * torch.mean(depth_counts.float())

    # Latent energy consistency loss: E(h_{t-1}, h_t) = || h_t - W_energy * h_{t-1} ||^2,
    # averaged over positions and batch. Encourages consecutive hidden states to be
    # consistent under the same projection LatentEnergyCritic uses to score generation
    # branches at inference time.
    prior_h = hidden[:, :-1, :]
    next_h = hidden[:, 1:, :].unsqueeze(1)  # [batch, 1, seq_len - 1, hidden_dim]
    energy_per_example = model.energy_critic.compute_energy(prior_h, next_h)  # [batch, 1]
    energy_loss = energy_penalty_weight * energy_per_example.mean()

    total_loss = ce_loss + depth_loss + energy_loss
    total_loss.backward()

    optimizer.step()

    return ce_loss.item(), torch.mean(depth_counts.float()).item()
