"""
Lightweight training-time diagnostics for Verace V1.
Captures per-step invariant/health signals without touching the model's
forward() signature or hot path:
  - CHAM unitary deviation (||H^H H - I||) via a forward hook on every layer's
    CHAM module, reading real training-distribution activations at zero extra
    forward-pass cost (it just observes what train_pretrain_step already computes).
  - ACDE halting-depth distribution from the depth_counts tensor the model
    already returns.
"""
from typing import Dict, List, Optional

import torch


class CHAMInvariantProbe:
    """
    Registers a forward hook on every layer's CHAM module to capture its
    (H_real, H_imag) hologram output during the real training forward pass --
    zero extra compute, just reads what was already produced. Call
    pop_mean_deviation() once per step (after backward/step, before the next
    forward) to read and clear the buffer.

    Note: cham_memory.py's forward() returns the RAW (never-retracted) state by
    design -- that's what composes correctly if fed back in as initial_hologram
    for incremental decoding (see its docstring). So the deviation measured here
    is the raw prefix product's actual drift from unitary, NOT the freshly
    -retracted value CHAM's own output read (`y`) is always computed from at
    every position -- that one stays close to exact by construction and
    wouldn't show any signal. This raw-drift number is expected to grow with
    sequence length (a documented property of the fp32 O(log S) scan, see
    tests/test_triton_kernels.py::test_cham_eager_path_numerical_fragility_at_long_sequences_is_known)
    -- that growth is exactly what makes it a useful live signal, not a bug.
    """
    def __init__(self, model: torch.nn.Module):
        self._captured: List[tuple] = []
        self._handles = [
            layer.cham_memory.register_forward_hook(self._hook)
            for layer in model.layers
        ]

    def _hook(self, module, inputs, output):
        _, (H_real, H_imag) = output
        self._captured.append((H_real.detach(), H_imag.detach()))

    def pop_mean_deviation(self) -> Optional[float]:
        """
        Mean ||H^H H - I|| of the raw carried-forward state across every layer
        invocation captured since the last call (clears the buffer) -- see the
        class docstring for why this is raw drift, not a post-retraction
        readout. Returns None if nothing was captured (e.g. every token halted
        before any layer ran).
        """
        if not self._captured:
            return None
        deviations = []
        for H_real, H_imag in self._captured:
            d = H_real.shape[-1]
            eye = torch.eye(d, device=H_real.device, dtype=H_real.dtype).unsqueeze(0)
            HH_r = torch.matmul(H_real.transpose(-1, -2), H_real) + torch.matmul(H_imag.transpose(-1, -2), H_imag)
            deviations.append(torch.norm(HH_r - eye, dim=(-2, -1)).mean().item())
        self._captured.clear()
        return sum(deviations) / len(deviations)

    def remove(self):
        for handle in self._handles:
            handle.remove()


def depth_distribution_stats(depth_counts: torch.Tensor) -> Dict[str, float]:
    """
    Summary stats of ACDE's per-token depth_counts (min/mean/max/std) --
    watches for halting collapsing to min_depth or max_depth instead of
    spreading meaningfully across the allowed range.
    """
    flat = depth_counts.float().flatten()
    return {
        "depth_mean": flat.mean().item(),
        "depth_std": flat.std().item() if flat.numel() > 1 else 0.0,
        "depth_min": flat.min().item(),
        "depth_max": flat.max().item(),
    }
