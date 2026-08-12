"""
Unit tests for LatentEnergyCritic's quadratic energy E(x, y) = ||h_cand - W_energy * x||_2^2.
"""
import torch

from verace_v1.modules.energy_critic import LatentEnergyCritic


def test_compute_energy_matches_norm_squared_formula():
    """The sum-of-squares implementation must be numerically identical (not just
    approximately equal) to the mathematically-defining ||diff||_2^2 formula --
    the fix changes only the gradient path, never the forward value."""
    torch.manual_seed(0)
    critic = LatentEnergyCritic(hidden_dim=16)
    x_prompt = torch.randn(2, 4, 16)
    h_cand = torch.randn(2, 3, 4, 16)

    energy = critic.compute_energy(x_prompt, h_cand)

    proj_x = critic.w_energy(x_prompt).unsqueeze(1)
    diff = h_cand - proj_x
    reference = torch.mean(torch.norm(diff, dim=-1) ** 2, dim=-1)

    torch.testing.assert_close(energy, reference, rtol=1e-5, atol=1e-6)
    print("compute_energy matches the ||diff||_2^2 reference formula exactly.")


def test_compute_energy_gradient_is_finite_when_diff_is_exactly_zero():
    """
    Regression test for a real training-divergence root cause: torch.norm(diff)'s
    gradient is diff/||diff||, which is 0/0 = NaN when diff is exactly zero -- this
    happened in practice ~95 steps into a real pretraining run, once a candidate's
    predicted state started exactly matching the target. sum(diff**2) has gradient
    2*diff, well-defined at zero, so no NaN should appear here.
    """
    critic = LatentEnergyCritic(hidden_dim=8)
    # Force h_cand == W_energy(x_prompt) exactly, i.e. diff == 0, by constructing
    # h_cand directly from the same linear projection.
    x_prompt = torch.randn(1, 2, 8, requires_grad=True)
    with torch.no_grad():
        proj = critic.w_energy(x_prompt)
    h_cand = proj.unsqueeze(1).clone().requires_grad_(True)  # [1, 1, 2, 8], diff == 0 exactly

    energy = critic.compute_energy(x_prompt, h_cand)
    assert torch.equal(energy, torch.zeros_like(energy)), "energy should be exactly 0 when diff is exactly 0"

    energy.sum().backward()

    assert x_prompt.grad is not None and not torch.isnan(x_prompt.grad).any(), "x_prompt gradient is NaN at diff=0"
    assert h_cand.grad is not None and not torch.isnan(h_cand.grad).any(), "h_cand gradient is NaN at diff=0"
    print("compute_energy's gradient is finite at diff=0 (the exact failure mode observed in training).")


if __name__ == "__main__":
    test_compute_energy_matches_norm_squared_formula()
    test_compute_energy_gradient_is_finite_when_diff_is_exactly_zero()
