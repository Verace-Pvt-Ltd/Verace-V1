# Getting Started

## Requirements

- Python 3.10+
- PyTorch 2.2+
- Triton 2.2+ (optional — only required for the GPU kernel path; the pure
  PyTorch path runs on CPU or GPU without it)

## Install

```bash
git clone https://github.com/Verace-Pvt-Ltd/verace-v1.git
cd verace-v1
pip install -e .
```

This installs the `verace_v1` package in editable mode along with its
dependencies (`torch`, `numpy`, `triton`).

## CPU vs. GPU

Every module has a pure PyTorch forward path that runs on CPU. Three modules
(`SSSDAttention`, `ContinuousHolographicMemory`, `ManifoldContinuousMoE`) also
have a fused Triton GPU kernel path (`verace_v1/serving/triton_kernels.py`,
`verace_v1/serving/sssd_triton.py`) that is used automatically when the input
tensor is on CUDA and Triton is importable (`HAS_TRITON`). There is no code
path difference required from the caller — the same `forward()` call
dispatches to whichever path applies.

## Quickstart

```python
import torch
from verace_v1 import VeraceV1Config, VeraceV1Model, VeraceV1Generator

config = VeraceV1Config(
    vocab_size=32000,
    hidden_dim=1024,
    num_layers=12,
    num_heads=8,
    head_dim=128,
)

model = VeraceV1Model(config)

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (1, 32))
logits, depth_counts = model(input_ids, use_adaptive_depth=True)
print(logits.shape, depth_counts.float().mean())

# Generation
generator = VeraceV1Generator(model, config)
print(generator.generate("Explain the Verace V1 architecture", max_new_tokens=32))
```

See [Configuration](configuration.md) for what every `VeraceV1Config` field
controls, and [Architecture Overview](modules/overview.md) for how `logits`
and `depth_counts` relate to the Adaptive Cognitive Depth Engine.

## Running the Tests

```bash
python3 -m pytest tests/ -v
```

Each test file exercises one module and checks the specific mathematical
invariant that module is supposed to guarantee (norm conservation, unitary
constraint, batch independence, etc.) rather than just "it runs." See
[Testing](testing.md) for the full mapping.

## What This Repository Does Not Include

- **Pretrained weights.** `VeraceV1Model(config)` gives you randomly
  initialized parameters. There is nothing to download.
- **A production training pipeline.** `verace_v1/training/pretrain.py`
  (documented in [Training](training/pretraining.md)) is a minimal, correct
  single-step reference — it wires the model, loss, and optimizer together
  correctly, but has no data loading, distributed training, checkpointing,
  or learning-rate scheduling.
- **A tokenizer.** `VeraceV1Generator` encodes prompts as raw UTF-8 bytes
  modulo `vocab_size` for demonstration purposes (see
  [Generation](serving/generation.md)); production use requires wiring in a
  real tokenizer.
