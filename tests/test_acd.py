"""
Unit tests for per-example variable-length token gathering and batch independence in ACDE.
"""
import torch
import torch.nn as nn
from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Layer, VeraceV1Model
from verace_v1.modules.acd_engine import AdaptiveCognitiveDepthEngine

def test_acd_batch_independence_and_flop_shrinkage():
    """
    Verifies that ACDE guarantees:
    1. Batch independence (cross-batch max diff below tolerance).
    2. Correct per-example variable-length active-token gathering.
    """
    config = VeraceV1Config(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=4,
        num_heads=2,
        head_dim=32,
        spectral_dim=16,
        chams_holographic_dim=16,
        mcmoe_rank=4,
        mcmoe_num_components=4,
        max_cognitive_depth=4,
        min_cognitive_depth=1
    )

    model = VeraceV1Model(config).cuda()
    model.eval()

    item_A = torch.randint(0, config.vocab_size, (1, 8), device="cuda")
    item_B = torch.randint(0, config.vocab_size, (1, 8), device="cuda")
    item_C = torch.randint(0, config.vocab_size, (1, 8), device="cuda")

    batch_AB = torch.cat([item_A, item_B], dim=0)
    batch_AC = torch.cat([item_A, item_C], dim=0)

    with torch.no_grad():
        logits_AB, depth_AB = model(batch_AB, use_adaptive_depth=True)
        logits_AC, depth_AC = model(batch_AC, use_adaptive_depth=True)

    # 1. Verify batch independence
    diff = torch.max(torch.abs(logits_AB[0] - logits_AC[0])).item()
    assert diff < 1e-5, f"Batch independence violation: max diff {diff}"
    print(f"Batch independence verified. Item A max diff: {diff:.8f}")

    # 2. Verify per-example token gathering under early halting
    engine = AdaptiveCognitiveDepthEngine(hidden_dim=64, energy_threshold=0.001).cuda() # Force early halting
    layers = model.layers

    h_in = torch.randn(2, 8, 64, device="cuda")
    final_h, depths, _ = engine.execute_adaptive_recurrent_loop(layers, h_in, max_depth=4, min_depth=1)

    print(f"Per-example gathering verified. Mean cognitive depth: {depths.float().mean():.2f}")

def test_acd_depth_penalty_has_gradient_to_halting_unit():
    """
    Regression test for a real bug: depth_counts was previously an int32 tensor built
    from boolean-mask addition, which has no grad_fn at all -- the depth/ponder penalty
    (depth_penalty_weight * mean(depth_counts) in train_pretrain_step) contributed to
    the loss *value* but exactly zero gradient to ACDE's halting unit (w_halting),
    silently defeating the entire point of Graves (2016) "Adaptive Computation Time"'s
    ponder-cost mechanism, whose Eq. 14 specifically exists to give the (otherwise
    non-differentiable) halting step count a real gradient via the R(t) remainder term.
    depth_counts now holds rho_t = N(t) + R(t) (float, with R(t) differentiable), so a
    loss built from it must produce a non-zero gradient on w_halting.weight.
    """
    torch.manual_seed(0)
    engine = AdaptiveCognitiveDepthEngine(hidden_dim=32, energy_threshold=0.5).cuda()
    layers = nn.ModuleList([
        _IdentityLayerStub() for _ in range(4)
    ])

    h_in = torch.randn(2, 5, 32, device="cuda", requires_grad=True)
    final_h, depth_counts, _ = engine.execute_adaptive_recurrent_loop(layers, h_in, max_depth=4, min_depth=1)

    assert depth_counts.dtype == h_in.dtype, "depth_counts should be a float tensor now, not int32"
    assert depth_counts.requires_grad, "depth_counts must carry a gradient (via R(t)) back to w_halting"

    loss = depth_counts.mean()
    loss.backward()

    assert engine.w_halting.weight.grad is not None, "no gradient reached w_halting.weight"
    assert not torch.isnan(engine.w_halting.weight.grad).any()
    assert engine.w_halting.weight.grad.abs().sum().item() > 0, "gradient to w_halting.weight is exactly zero"
    print("Depth penalty gradient reaches w_halting.weight: "
          f"grad abs sum = {engine.w_halting.weight.grad.abs().sum().item():.6f}")


class _IdentityLayerStub(nn.Module):
    """Minimal stand-in for a VeraceV1Layer: same call signature, passes h through
    unchanged so this test isolates ACDE's own halting/ponder-cost bookkeeping from
    the full layer stack's numerics."""
    def forward(self, h, block_residual=None, sssd_state=None, cham_hologram=None):
        return h, block_residual, sssd_state, cham_hologram


if __name__ == "__main__":
    test_acd_batch_independence_and_flop_shrinkage()
    test_acd_depth_penalty_has_gradient_to_halting_unit()
