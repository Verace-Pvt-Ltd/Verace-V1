"""
Unit tests for UnitaryMuon's Stiefel-manifold orthogonalization across matrix shapes.
"""
import torch
from verace_v1.optimizer.unitary_muon import stiefel_orthogonalize, UnitaryMuon

def _deviation_from_orthogonal(X: torch.Tensor, shape: tuple) -> float:
    m, n = shape
    if m <= n:
        return torch.norm(X @ X.T - torch.eye(m)).item()
    return torch.norm(X.T @ X - torch.eye(n)).item()

def test_stiefel_orthogonalize_square_tall_wide():
    torch.manual_seed(0)
    shapes = [(64, 64), (128, 32), (32, 128)]
    for shape in shapes:
        G = torch.randn(*shape) * 0.02
        X = stiefel_orthogonalize(G)
        dev = _deviation_from_orthogonal(X, shape)
        assert dev < 1e-3, f"shape {shape}: deviation from orthogonal = {dev}"
        print(f"shape={shape}: deviation from orthogonal = {dev:.8f}")

def test_stiefel_orthogonalize_small_singular_values():
    """
    Regression test: the previous Newton-Schulz-based implementation entered a
    persistent oscillation (never converging) when singular values were small after
    Frobenius normalization -- the common case for real gradient/momentum matrices.
    SVD-based polar decomposition has no such failure mode.
    """
    torch.manual_seed(0)
    shape = (128, 32)
    G = torch.randn(*shape) * 0.02
    G[:, 0] *= 1e-6  # pathologically small singular value
    X = stiefel_orthogonalize(G)
    dev = _deviation_from_orthogonal(X, shape)
    assert dev < 1e-3, f"deviation from orthogonal with a near-zero singular value = {dev}"
    print(f"Near-zero singular value case: deviation from orthogonal = {dev:.8f}")

def test_unitary_muon_step_runs_on_rectangular_params():
    torch.manual_seed(0)
    linear = torch.nn.Linear(32, 128, bias=False)
    optimizer = UnitaryMuon(linear.parameters(), lr=0.01)

    x = torch.randn(4, 32)
    loss = linear(x).sum()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(linear.weight).any()
    print("UnitaryMuon.step() on a rectangular (128, 32) parameter completed cleanly.")

if __name__ == "__main__":
    test_stiefel_orthogonalize_square_tall_wide()
    test_stiefel_orthogonalize_small_singular_values()
    test_unitary_muon_step_runs_on_rectangular_params()
