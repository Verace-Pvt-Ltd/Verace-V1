"""
Verace V1 Configuration Module
Defines the architecture and hyperparameter specification for:
1. Spectral State-Space Differential Attention (SSSD) with unitary phase rotations
2. Adaptive Cognitive Depth Engine (ACDE) for dynamic per-token compute allocation
3. Manifold Continuous Mixture-of-Experts (M-CMoE), avoiding all-to-all communication
4. Continuous Holographic Associative Memory (CHAM) for O(1) context retrieval
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class VeraceV1Config:
    """
    Verace V1 architecture and hyperparameter specification.

    Defaults below describe a small, immediately-runnable reference size (matches
    the README quickstart) so `VeraceV1Config()` with no arguments never silently
    tries to build a many-billion-parameter model. Pass explicit dimensions for a
    production-scale run.
    """
    # Core Model Dimensions
    vocab_size: int = 163840        # Vocabulary size (matches official Moonshot Kimi K3 tokenizer)
    hidden_dim: int = 1024          # Hidden state dimension
    num_layers: int = 12            # Maximum number of decoder layers
    num_heads: int = 8              # Number of attention heads
    head_dim: int = 128

    # Spectral State-Space Differential Attention (SSSD)
    spectral_dim: int = 256         # Unitary rotation manifold dimension
    sssd_phase_scale: float = 1.0   # Phase angle scaling factor

    # Continuous Holographic Associative Memory (CHAM)
    chams_holographic_dim: int = 1024
    chams_memory_decay: float = 0.001 # Decorrelation noise applied to the hologram

    # Manifold Continuous Mixture-of-Experts (M-CMoE)
    mcmoe_rank: int = 32            # Continuous low-rank adaptation rank
    mcmoe_num_components: int = 64  # Number of basis manifold generators

    # Adaptive Cognitive Depth Engine (ACDE)
    max_cognitive_depth: int = 12   # Maximum layer unrolling per token (clamped to num_layers)
    min_cognitive_depth: int = 2    # Minimum layers for trivial tokens
    energy_halting_threshold: float = 0.99 # ACT cumulative halting threshold

    # Latent Tree Search & Energy Critic
    tree_branches: int = 4          # Parallel latent reasoning rollout branches

    # Context & Memory
    max_context_length: int = 10000000 # Target context length; CHAM cost is O(1) in sequence length

    # Optimization
    learning_rate: float = 0.03
    unitary_muon_momentum: float = 0.98
    weight_decay: float = 0.05
    rms_norm_eps: float = 1e-6
    
    vision_config: Optional[Any] = None
    media_placeholder_token_id: Optional[int] = None  # if set, visual tokens replace
    # embeddings only at positions in input_ids equal to this id; must be < vocab_size.
    # If None, or if the id doesn't appear in a given input_ids batch, visual tokens
    # are prepended instead -- text is never overwritten either way (see backbone.py).

    def __post_init__(self):
        if self.vision_config is None:
            @dataclass
            class DefaultVisionConfig:
                embed_dim: int = 1152
                num_layers: int = 27
                num_heads: int = 12
                patch_size: int = 14
            self.vision_config = DefaultVisionConfig()
