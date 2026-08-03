# Configuration Reference

`VeraceV1Config` (`verace_v1/config.py`) is a single dataclass that holds
every architecture and hyperparameter setting. Construct it with keyword
arguments; any field you don't set keeps its default.

```python
from verace_v1 import VeraceV1Config

config = VeraceV1Config(hidden_dim=1024, num_layers=12, num_heads=8, head_dim=128)
```

The defaults describe a large-scale configuration (16384 hidden dim, 128
layers). For local development and the test suite, override the dimensions
down — see `tests/test_end2end.py` for a small working configuration.

## Core Model Dimensions

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `vocab_size` | `262144` | Vocabulary size (embedding table and LM head width) |
| `hidden_dim` | `16384` | Hidden state width threaded through every layer |
| `num_layers` | `128` | Number of decoder layers instantiated; also the ceiling for `max_cognitive_depth` |
| `num_heads` | `128` | Number of SSSD attention heads |
| `head_dim` | `128` | Per-head dimension for SSSD attention |

## SSSD Attention

See [SSSD Attention](modules/sssd-attention.md).

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `spectral_dim` | `256` | Declared for the unitary rotation manifold; not currently read by `SSSDAttention`, which derives its rotation dimension from `head_dim` |
| `sssd_phase_scale` | `1.0` | Reserved for scaling the phase-rotation angle; not yet wired into `SSSDAttention.forward` |

## CHAM Memory

See [CHAM Memory](modules/cham-memory.md).

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `chams_holographic_dim` | `1024` | Width of the holographic matrix `H` (passed to `ContinuousHolographicMemory`) |
| `chams_memory_decay` | `0.001` | Reserved for decorrelation-noise scheduling; not yet wired into `ContinuousHolographicMemory.forward` |

## M-CMoE

See [M-CMoE](modules/mcmoe.md).

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `mcmoe_rank` | `32` | Rank of each expert's low-rank `U_j`, `V_j` basis matrices |
| `mcmoe_num_components` | `64` | Number of basis manifold generators to select the top-K from |

`ManifoldContinuousMoE.top_k_components` (default `8`) and the SiTU-GLU
`beta1`/`beta2` shape parameters are constructor arguments on the module
itself, not `VeraceV1Config` fields — see [`backbone.py`](modules/backbone.md)
for how `VeraceV1Layer` currently hardcodes them (`beta1=4.0, beta2=25.0`).

## ACD Engine

See [ACD Engine](modules/acd-engine.md).

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `max_cognitive_depth` | `128` | Upper bound on layers a token can run through before being forced to halt |
| `min_cognitive_depth` | `2` | Every token runs at least this many layers before halting is allowed |
| `energy_halting_threshold` | `0.01` | Accumulated halting probability at which a token stops (passed to `AdaptiveCognitiveDepthEngine` as `energy_threshold`) |

## Latent Tree Search

See [Energy Critic](modules/energy-critic.md) and [Generation](serving/generation.md).

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `tree_branches` | `4` | Default number of candidate branches `VeraceV1Generator.generate` scores per generated token when `use_tree_search=True` (its `num_branches` argument overrides this per call) |

## Context & Memory

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `max_context_length` | `10000000` | Documents the design target enabled by CHAM's O(1) memory cost; not an enforced limit anywhere in the code |

## Optimization

These fields describe the intended optimizer settings but are **not**
automatically read when constructing `UnitaryMuon` — you pass them
explicitly, as in `tests/test_end2end.py`:

```python
optimizer = UnitaryMuon(model.parameters(), lr=0.01)
```

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `learning_rate` | `0.03` | Reference value for `UnitaryMuon(lr=...)` |
| `unitary_muon_momentum` | `0.98` | Reference value for `UnitaryMuon(momentum=...)` |
| `weight_decay` | `0.05` | Reference value for `UnitaryMuon(weight_decay=...)` |
| `rms_norm_eps` | `1e-6` | Passed to every `nn.RMSNorm` in `VeraceV1Layer` / `VeraceV1Model` |

## Vision

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `vision_config` | `None` | If left `None`, `__post_init__` fills in a default vision config (`embed_dim=1152, num_layers=27, num_heads=12, patch_size=14`) consumed by `VeraceVisionEncoder` — see [Vision Encoder](modules/vision-encoder.md) |
| `media_placeholder_token_id` | `None` | If set to a valid token id (must be `< vocab_size`) that appears in `input_ids`, visual tokens replace embeddings only at those positions, preserving sequence length. If `None` (default), or if the id doesn't appear in a given batch, visual tokens are prepended instead — sequence length grows, but text is never overwritten. See [Backbone](modules/backbone.md#_fuse_visual_tokens) and [Pretraining](training/pretraining.md) for the `logits`/`labels` alignment this implies. |
