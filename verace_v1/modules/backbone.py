"""
Verace V1 Backbone & Layer Integration
Integrates Spectral State-Space Differential Attention (SSSD), Continuous Holographic
Associative Memory (CHAM), Manifold Continuous Mixture-of-Experts (M-CMoE),
and Adaptive Cognitive Depth Engine (ACDE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from verace_v1.config import VeraceV1Config
from verace_v1.modules.sssd_attention import SSSDAttention
from verace_v1.modules.cham_memory import ContinuousHolographicMemory
from verace_v1.modules.mcmoe import ManifoldContinuousMoE
from verace_v1.modules.acd_engine import AdaptiveCognitiveDepthEngine
from verace_v1.modules.energy_critic import LatentEnergyCritic
from verace_v1.modules.vision import VeraceVisionEncoder

class VeraceV1Layer(nn.Module):
    """
    Single Verace V1 decoder layer.
    Combines SSSD attention, CHAM holographic memory, and manifold continuous MoE.
    """
    def __init__(self, layer_idx: int, config: VeraceV1Config):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.sssd_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.sssd_attn = SSSDAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            spectral_dim=config.spectral_dim
        )

        self.cham_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.cham_memory = ContinuousHolographicMemory(
            hidden_dim=config.hidden_dim,
            holographic_dim=config.chams_holographic_dim
        )

        self.mcmoe_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.mcmoe = ManifoldContinuousMoE(
            hidden_dim=config.hidden_dim,
            rank=config.mcmoe_rank,
            num_components=config.mcmoe_num_components,
            beta1=4.0, beta2=25.0
        )

    def forward(
        self,
        h_in: torch.Tensor,
        block_residual: Optional[torch.Tensor] = None,
        sssd_state: Optional[torch.Tensor] = None,
        cham_hologram: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        active_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        h_in: [batch, seq_len, hidden_dim]
        block_residual: [total_tokens, num_blocks, hidden_dim]
        active_mask: [batch, seq_len]
        """
        b, s, h = h_in.shape
        flat_h = h_in.reshape(b * s, h)

        if block_residual is None:
            block_residual = flat_h.new_zeros(b * s, 0, h)

        # AttnRes mixing if candidates exist
        if block_residual.shape[1] > 0:
            attn_res_weight = torch.softmax(torch.matmul(flat_h.unsqueeze(1), block_residual.transpose(-1, -2)).squeeze(1), dim=-1)
            mixed_h = flat_h + torch.matmul(attn_res_weight.unsqueeze(1), block_residual).squeeze(1)
            h_in = mixed_h.view(b, s, h)

        # Commit current snapshot every 12 layers
        if self.layer_idx % 12 == 0:
            block_residual = torch.cat([block_residual, flat_h.unsqueeze(1)], dim=1)

        # 1. SSSD Attention Sub-layer with active_mask
        norm_sssd = self.sssd_norm(h_in)
        sssd_out, new_sssd_state = self.sssd_attn(
            norm_sssd, initial_state=sssd_state, return_state=True, active_mask=active_mask
        )
        h_mid1 = h_in + sssd_out

        # 2. CHAM Holographic Associative Memory Sub-layer with active_mask
        norm_cham = self.cham_norm(h_mid1)
        cham_out, new_cham_hologram = self.cham_memory(
            norm_cham, initial_hologram=cham_hologram, active_mask=active_mask
        )
        h_mid2 = h_mid1 + cham_out

        # 3. Manifold Continuous MoE Sub-layer
        norm_moe = self.mcmoe_norm(h_mid2)
        moe_out = self.mcmoe(norm_moe)
        h_out = h_mid2 + moe_out

        return h_out, block_residual, new_sssd_state


class VeraceV1Model(nn.Module):
    """
    Complete Verace V1 model architecture.
    Combines Adaptive Cognitive Depth routing, SSSD attention, CHAM associative memory,
    and M-CMoE expert computation into a single decoder stack.
    """
    def __init__(self, config: VeraceV1Config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.embed_tokens.weight = self.lm_head.weight # Weight tying

        # Vision Encoder
        self.vision_encoder = VeraceVisionEncoder(
            embed_dim=config.vision_config.embed_dim,
            num_layers=config.vision_config.num_layers,
            num_heads=config.vision_config.num_heads,
            patch_size=config.vision_config.patch_size,
            projector_dim=config.hidden_dim
        )

        # Cognitive Layers
        self.layers = nn.ModuleList([
            VeraceV1Layer(layer_idx=i, config=config)
            for i in range(config.num_layers)
        ])

        # Adaptive Cognitive Depth Engine
        self.acd_engine = AdaptiveCognitiveDepthEngine(
            hidden_dim=config.hidden_dim,
            energy_threshold=config.energy_halting_threshold
        )

        # Latent Energy Critic
        self.energy_critic = LatentEnergyCritic(hidden_dim=config.hidden_dim)

        self.final_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        use_adaptive_depth: bool = True,
        return_hidden: bool = False
    ):
        """
        input_ids: [batch, seq_len]
        Returns: (logits [batch, seq_len, vocab_size], depth_counts [batch, seq_len])
        if return_hidden=True, also returns the post-final-norm hidden state
        [batch, seq_len, hidden_dim] as a third element — the representation
        consumed by LatentEnergyCritic for branch scoring (see
        verace_v1/serving/hyper_generate.py and verace_v1/training/pretrain.py).
        """
        b, s = input_ids.shape
        token_embeds = self.embed_tokens(input_ids) # [b, s, d]

        if images is not None:
            visual_tokens = self.vision_encoder(images)
            v_len = min(visual_tokens.shape[1], s)
            token_embeds[:, :v_len] = visual_tokens[:, :v_len]

        if use_adaptive_depth:
            # Dynamic layer unrolling per token via Adaptive Cognitive Depth Engine
            final_h, depth_counts = self.acd_engine.execute_adaptive_recurrent_loop(
                layers=self.layers,
                h_in=token_embeds,
                max_depth=self.config.max_cognitive_depth,
                min_depth=self.config.min_cognitive_depth
            )
        else:
            # Static unrolling fallback
            h_curr = token_embeds
            for layer in self.layers:
                h_curr, _, _ = layer(h_curr)
            final_h = h_curr
            depth_counts = torch.full((b, s), len(self.layers), device=input_ids.device, dtype=torch.int32)

        norm_final_h = self.final_norm(final_h)
        logits = self.lm_head(norm_final_h)

        if return_hidden:
            return logits, depth_counts, norm_final_h
        return logits, depth_counts
