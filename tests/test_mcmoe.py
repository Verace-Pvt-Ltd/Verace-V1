"""
Unit tests for Manifold Continuous Mixture-of-Experts (M-CMoE)
"""
import torch
from verace_v1.modules.mcmoe import ManifoldContinuousMoE

def test_mcmoe():
    b, s, d = 2, 8, 256
    rank = 16
    num_components = 8
    
    moe = ManifoldContinuousMoE(hidden_dim=d, rank=rank, num_components=num_components)
    x = torch.randn(b, s, d)
    
    y = moe(x)
    assert y.shape == (b, s, d)
    assert not torch.isnan(y).any()
    print("M-CMoE test passed.")

if __name__ == "__main__":
    test_mcmoe()
