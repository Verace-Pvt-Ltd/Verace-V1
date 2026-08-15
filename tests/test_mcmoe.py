"""
Unit tests for Manifold Continuous Mixture-of-Experts (M-CMoE)
"""
import torch
from verace_v1.modules.mcmoe import ManifoldContinuousMoE

def test_mcmoe():
    b, s, d = 2, 8, 256
    rank = 16
    num_components = 8

    moe = ManifoldContinuousMoE(hidden_dim=d, rank=rank, num_components=num_components).cuda()
    x = torch.randn(b, s, d, device="cuda")

    y = moe(x)
    assert y.shape == (b, s, d)
    assert not torch.isnan(y).any()
    print("M-CMoE test passed.")


def test_mcmoe_topk_gate_weights_sum_to_one():
    """
    Regression test for a real gating-normalization bug: the router previously applied
    softmax over ALL num_components first, then took the top-k of the already-normalized
    weights -- those weights are a subset of a distribution that summed to 1 over every
    component, so they summed to LESS than 1 (verified: ~0.54 for a near-uniform router,
    the common case at initialization), systematically under-scaling manifold_adapt by
    ~2x. The correct noisy top-k gating (Shazeer et al. 2017, Eq. 2-5) masks non-top-k
    LOGITS to -inf before softmax, so the surviving top-k weights are a proper convex
    combination that always sums to exactly 1.
    """
    torch.manual_seed(0)
    d = 256
    num_components, k = 16, 8
    moe = ManifoldContinuousMoE(hidden_dim=d, rank=16, num_components=num_components, top_k_components=k).cuda()

    x_flat = torch.randn(1000, d, device="cuda") * 0.1  # small activations -> near-uniform router at init
    router_logits = moe.router(x_flat)
    topk_logits, topk_indices = torch.topk(router_logits, k=k, dim=-1)
    masked_logits = torch.full_like(router_logits, float("-inf"))
    masked_logits.scatter_(-1, topk_indices, topk_logits)
    topk_weights = torch.gather(torch.softmax(masked_logits, dim=-1), -1, topk_indices)

    weight_sums = topk_weights.sum(-1)
    torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), rtol=1e-4, atol=1e-4)
    print(f"M-CMoE top-k gate weights sum to 1.0 (mean sum = {weight_sums.mean().item():.6f}).")


if __name__ == "__main__":
    test_mcmoe()
    test_mcmoe_topk_gate_weights_sum_to_one()
