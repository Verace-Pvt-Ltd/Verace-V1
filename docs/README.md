# Verace V1 Documentation

This directory documents Verace V1 module by module, mirroring the
`verace_v1/` package layout 1:1. Start with **Getting Started**, then read
**Architecture Overview** for how the pieces fit together, then drill into
whichever module you're working on.

## Start Here

| Doc | Covers |
| :--- | :--- |
| [Getting Started](getting-started.md) | Install, environment requirements, quickstart, running the tests |
| [Configuration](configuration.md) | Every `VeraceV1Config` field: what it controls and its default |
| [Testing](testing.md) | How the test suite maps to the invariant each module guarantees |

## Architecture (`verace_v1/modules/`)

| Doc | Module | What it does |
| :--- | :--- | :--- |
| [Overview](modules/overview.md) | — | System diagram, design philosophy, how the modules compose |
| [SSSD Attention](modules/sssd-attention.md) | `sssd_attention.py` | Sequence mixing — replaces softmax self-attention |
| [CHAM Memory](modules/cham-memory.md) | `cham_memory.py` | Long-range memory — replaces the KV cache |
| [M-CMoE](modules/mcmoe.md) | `mcmoe.py` | Expert computation — replaces discrete routed MoE |
| [ACD Engine](modules/acd-engine.md) | `acd_engine.py` | Per-token compute allocation — replaces fixed depth |
| [Energy Critic](modules/energy-critic.md) | `energy_critic.py` | Latent branch scoring for generation |
| [Vision Encoder](modules/vision-encoder.md) | `vision.py` | Patch-based image encoder feeding the decoder |
| [Backbone](modules/backbone.md) | `backbone.py` | Assembles the above into `VeraceV1Layer` / `VeraceV1Model` |

## Optimization (`verace_v1/optimizer/`)

| Doc | Module |
| :--- | :--- |
| [Unitary Muon](optimizer/unitary-muon.md) | `unitary_muon.py` |

## Chat Template (`verace_v1/chat_template/`)

| Doc | Module |
| :--- | :--- |
| [Hyper-XTML](chat_template/hyper-xtml.md) | `hyper_xtml.py` |

## Serving (`verace_v1/serving/`)

| Doc | Module |
| :--- | :--- |
| [Generation](serving/generation.md) | `hyper_generate.py` |
| [Triton Kernels](serving/triton-kernels.md) | `sssd_triton.py`, `triton_kernels.py` |

## Training (`verace_v1/training/`)

| Doc | Module |
| :--- | :--- |
| [Pretraining](training/pretraining.md) | `pretrain.py` |

## Evaluation (`verace_v1/eval/`)

| Doc | Module |
| :--- | :--- |
| [Benchmark Runner](eval/benchmark-runner.md) | `benchmark_runner.py` |

## Status

This is a from-scratch reference implementation — module code, configuration,
and a unit/integration test suite. It does not include pretrained weights.
See [Getting Started](getting-started.md) for what you can and can't do with
it out of the box.
