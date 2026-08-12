"""
Adaptive Cognitive Depth Engine (ACDE) Module
Implements per-example variable-length token gathering for early-exit computation.
Active tokens are gathered per batch item independently (shape [1, S_active_i, d]),
so halted tokens stop consuming compute without affecting other items in the batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class AdaptiveCognitiveDepthEngine(nn.Module):
    """
    Adaptive Cognitive Depth Engine (ACDE).
    Implements dynamic per-token early exit (ACT-style halting):
    1. Per-example variable-length gathering: processes each batch item i independently on [1, S_active_i, d].
    2. Reduced FLOPs: linear layers and MoE execute matmuls only on active tokens.
    3. No cross-batch contamination: batch items never mix tokens.
    4. Gated halting accumulation: a halted token writes its output once and freezes.
    """
    def __init__(self, hidden_dim: int = 16384, energy_threshold: float = 0.99):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.energy_threshold = energy_threshold

        self.w_halting = nn.Linear(hidden_dim, 1)
        # Initialize bias to -3.0 so base halting prob per layer is ~0.047,
        # preventing immediate collapse to min_depth=2 and allowing dynamic depth unrolling.
        nn.init.constant_(self.w_halting.bias, -3.0)
        nn.init.normal_(self.w_halting.weight, std=0.01)

    def compute_halting_probability(self, h_l: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.w_halting(h_l)).squeeze(-1)

    def execute_adaptive_recurrent_loop(
        self,
        layers: nn.ModuleList,
        h_in: torch.Tensor,
        max_depth: int = 128,
        min_depth: int = 2,
        initial_layer_states: Optional[List[Optional[List[Optional[Tuple]]]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[List[Optional[Tuple]]]]]:
        """
        h_in: [batch, seq_len, hidden_dim]
        initial_layer_states: per-layer list (length min(len(layers), max_depth)) of
            per-batch-item lists (length b) of (sssd_state, cham_hologram) or None --
            the SSSD/CHAM recurrent state each layer had reached for each batch item
            as of a previous call, as returned by this method. Pass None (default) for
            a fresh sequence; pass the prior return value back in, together with only
            the *new* tokens in h_in, to continue incrementally (see
            verace_v1/serving/hyper_generate.py). A batch item/layer that never
            received real tokens yet stays None, which SSSDAttention/CHAM treat as
            fresh identity init -- identical behavior to the very first call.
        Returns:
          final_hidden: [batch, seq_len, hidden_dim]
          depth_counts: [batch, seq_len]
          layer_states: per-layer, per-batch-item state as described above, to pass
            back in on the next call.
        """
        b, s, d = h_in.shape
        device = h_in.device

        h_curr = h_in.clone()
        final_h = torch.zeros_like(h_in)
        accumulated_prob = torch.zeros(b, s, device=device, dtype=h_in.dtype)
        depth_counts = torch.zeros(b, s, device=device, dtype=torch.int32)
        active_mask = torch.ones(b, s, device=device, dtype=torch.bool)

        num_layers = min(len(layers), max_depth)
        block_residuals = [h_in.new_zeros(s, 0, d) for _ in range(b)]

        # Per-layer, per-batch-item state, seeded from initial_layer_states (shallow-copied
        # so branching multiple candidates off the same cached state -- e.g. tree search --
        # never mutates the shared input).
        layer_states_out: List[List[Optional[Tuple]]] = [
            (list(initial_layer_states[l]) if initial_layer_states is not None and l < len(initial_layer_states)
             and initial_layer_states[l] is not None else [None] * b)
            for l in range(num_layers)
        ]

        for l_idx in range(num_layers):
            if not active_mask.any():
                break

            layer = layers[l_idx]
            h_next = h_curr.clone()

            # Per-example variable-length gathering: process each batch item i independently.
            for i in range(b):
                active_indices_i = torch.nonzero(active_mask[i], as_tuple=True)[0]
                if len(active_indices_i) == 0:
                    continue

                # Gather active tokens ONLY for batch item i -> shape [1, S_active_i, d]
                h_active_i = h_curr[i, active_indices_i].unsqueeze(0)
                block_res_i = block_residuals[i][active_indices_i]

                prev_state_i = layer_states_out[l_idx][i]
                sssd_state_i = prev_state_i[0] if prev_state_i is not None else None
                cham_hologram_i = prev_state_i[1] if prev_state_i is not None else None

                # Execute the layer only on active tokens for item i
                layer_out_i, new_block_res_i, new_sssd_state_i, new_cham_hologram_i = layer(
                    h_active_i,
                    block_residual=block_res_i,
                    sssd_state=sssd_state_i,
                    cham_hologram=cham_hologram_i
                )

                # Scatter layer output back to item i
                h_next[i, active_indices_i] = layer_out_i.squeeze(0)
                block_residuals[i][active_indices_i] = new_block_res_i
                layer_states_out[l_idx][i] = (new_sssd_state_i, new_cham_hologram_i)

            # Compute Halting Probabilities for active tokens
            p_l = self.compute_halting_probability(h_next)

            if l_idx < min_depth:
                p_l = torch.zeros_like(p_l)

            # Check halting threshold for active tokens
            new_accum = accumulated_prob + p_l
            newly_halted = (new_accum >= self.energy_threshold) & active_mask
            still_active = active_mask & (~newly_halted)

            # Weight calculation: newly halted tokens get remainder weight; active tokens get p_l
            remainder_weight = (1.0 - accumulated_prob).clamp(min=0.0)
            weight_layer = torch.where(newly_halted, remainder_weight, p_l)

            # Gated halting accumulation: accumulate contribution only for active tokens
            update_mask = (active_mask | newly_halted)
            final_h = final_h + (weight_layer * update_mask.to(h_in.dtype)).unsqueeze(-1) * h_next
            depth_counts = depth_counts + active_mask.to(torch.int32)

            # Update global state buffers
            accumulated_prob = new_accum
            active_mask = still_active
            h_curr = h_next

        # Any remaining active tokens take full final representation
        if active_mask.any():
            remainder_weight = (1.0 - accumulated_prob).clamp(min=0.0)
            final_h = final_h + (remainder_weight * active_mask.to(h_in.dtype)).unsqueeze(-1) * h_curr

        return final_h, depth_counts, layer_states_out
