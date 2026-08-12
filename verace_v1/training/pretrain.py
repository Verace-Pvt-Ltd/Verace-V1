"""
Verace V1 Production Pre-Training & Distributed Training Module
Implements multi-GPU FSDP2 sharding, activation checkpointing, AMP bfloat16 mixed precision,
Cosine LR scheduling with warmup, and fault-tolerant checkpointing.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Any

from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model, VeraceV1Layer
from verace_v1.optimizer.unitary_muon import UnitaryMuon

try:
    import torch.distributed as dist
    from torch.distributed._composable.fsdp import fully_shard
    from torch.utils.checkpoint import checkpoint
    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False


def setup_distributed_model(
    model: VeraceV1Model,
    use_activation_checkpointing: bool = True,
    use_fsdp: bool = True
) -> nn.Module:
    """
    Prepares a VeraceV1Model for production multi-GPU training.
    Enables FSDP2 per-layer sharding and gradient/activation checkpointing.
    """
    if use_activation_checkpointing:
        for layer in model.layers:
            # Wrap layer forward pass with activation checkpointing to save VRAM
            orig_forward = layer.forward
            def make_ckpt_forward(l_fn):
                def custom_forward(*args, **kwargs):
                    return checkpoint(l_fn, *args, use_reentrant=False, **kwargs)
                return custom_forward
            layer.forward = make_ckpt_forward(orig_forward)

    if use_fsdp and HAS_DISTRIBUTED and dist.is_initialized():
        # Apply FSDP2 per-layer sharding to achieve linear scale-out across GPUs
        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)

    return model


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Creates a Cosine Learning Rate Schedule with Linear Warmup. Pass
    last_epoch=resumed_step - 1 when resuming from a checkpoint so the
    schedule continues from the right point instead of restarting warmup.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: str
):
    """Saves training checkpoint for fault tolerance."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"verace_v1_step_{step}.pt")
    state_dict = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state_dict, ckpt_path)
    print(f"[Verace V1 Checkpoint] Saved checkpoint at step {step} to {ckpt_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    map_location: str = "cpu"
) -> int:
    """Loads a training checkpoint saved by save_checkpoint. Returns the step it was saved at."""
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state_dict["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer"])
    step = state_dict.get("step", 0)
    print(f"[Verace V1 Checkpoint] Loaded checkpoint from {checkpoint_path} at step {step}")
    return step


def train_pretrain_step(
    model: VeraceV1Model,
    optimizer: UnitaryMuon,
    batch: dict,
    depth_penalty_weight: float = 0.001,
    energy_penalty_weight: float = 0.01,
    use_amp: bool = True
) -> Tuple[float, float]:
    """
    Production pre-training step minimizing:
        ce_loss + depth_penalty_weight * mean(depth_counts)
                + energy_penalty_weight * mean(E(h_{t-1}, h_t))
    Supports AMP bfloat16 mixed precision.
    """
    model.train()
    optimizer.zero_grad()

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    images = batch.get("image", None)

    device_type = "cuda" if input_ids.is_cuda else "cpu"
    amp_dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
        logits, depth_counts, hidden = model(
            input_ids, images=images, use_adaptive_depth=True, return_hidden=True
        )

        if logits.shape[1] != labels.shape[1]:
            logits = logits[:, -labels.shape[1]:, :]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        depth_loss = depth_penalty_weight * torch.mean(depth_counts.float())

        prior_h = hidden[:, :-1, :]
        next_h = hidden[:, 1:, :].unsqueeze(1)
        energy_per_example = model.energy_critic.compute_energy(prior_h, next_h)
        energy_loss = energy_penalty_weight * energy_per_example.mean()

        total_loss = ce_loss + depth_loss + energy_loss

    total_loss.backward()
    optimizer.step()

    return ce_loss.item(), torch.mean(depth_counts.float()).item()

