# Testing

```bash
python3 -m pytest tests/ -v
```

Each test file targets one module and checks the specific invariant that
module claims to guarantee — not just "it runs without erroring."

| Test file | Module under test | Invariant checked |
| :--- | :--- | :--- |
| `tests/test_sssd.py` | [SSSD Attention](modules/sssd-attention.md) | `\|\|Psi_t\|\|_F == \|\|Psi_0\|\|_F` — state norm conservation via the Cayley transform, within `1e-4` |
| `tests/test_cham.py` | [CHAM Memory](modules/cham-memory.md) | `\|\|H^H H - I\|\| < 1e-3` — the holographic matrix stays unitary after a forward pass |
| `tests/test_mcmoe.py` | [M-CMoE](modules/mcmoe.md) | Output shape correctness and absence of `NaN` |
| `tests/test_acd.py` | [ACD Engine](modules/acd-engine.md) | (1) Batch independence — identical item, different batchmates, max logit diff `< 1e-5`; (2) per-example gathering runs correctly under forced early halting |
| `tests/test_unitary_muon.py` | [Unitary Muon](optimizer/unitary-muon.md) | SVD-based orthogonalization deviation `< 1e-3` across square, tall, wide, and pathologically-small-singular-value inputs; `step()` runs cleanly on a rectangular parameter |
| `tests/test_vision_fusion.py` | [Backbone](modules/backbone.md#_fuse_visual_tokens) | Fallback (prepend) path never alters existing text embeddings; placeholder-token path preserves sequence length and touches only placeholder positions; [`train_pretrain_step`](training/pretraining.md) correctly aligns `logits`/`labels` when images grow the sequence |
| `tests/test_end2end.py` | Full pipeline | Model construction → forward pass → [pretraining step](training/pretraining.md) → [generation](serving/generation.md) → [evaluation](eval/benchmark-runner.md), wired together and asserted at each stage |

## Why These Invariants, Specifically

SSSD, CHAM, and the [Unitary Muon optimizer](optimizer/unitary-muon.md) each
make an *exact* algebraic claim (norm conservation, unitarity,
orthogonality) rather than an approximate one — see
[modules/overview.md](modules/overview.md#design-principle-common-to-sssd-cham-and-unitary-muon)
for why this is a shared design principle across the three. Testing these
directly (rather than only testing end-to-end task accuracy, which a broken
invariant might not visibly affect at small scale) is what catches a
regression in the underlying math before it becomes a subtle training
instability at scale.

ACDE's tests check a correctness property (batch independence) that a naive
masking-based implementation would silently violate — see
[modules/acd-engine.md](modules/acd-engine.md#per-example-gathering-why-this-isnt-just-a-mask)
for why gathering, not masking, is required.

## Adding a New Test

Follow the existing pattern: construct a small `VeraceV1Config` (see any
existing test for realistic small dimensions), instantiate only the module
under test where possible (not the full model) to keep the test fast and
its failure mode specific, and assert the invariant with an explicit
numeric tolerance rather than an exact equality where floating-point
arithmetic is involved.
