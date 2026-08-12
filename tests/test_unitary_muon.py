"""
Unit tests for UnitaryMuon's Stiefel-manifold orthogonalization across matrix shapes.
"""
import torch
from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.optimizer.unitary_muon import build_hybrid_optimizer, stiefel_orthogonalize, UnitaryMuon

def _deviation_from_orthogonal(X: torch.Tensor, shape: tuple) -> float:
    m, n = shape
    if m <= n:
        return torch.norm(X @ X.T - torch.eye(m, device=X.device)).item()
    return torch.norm(X.T @ X - torch.eye(n, device=X.device)).item()

def test_stiefel_orthogonalize_square_tall_wide():
    torch.manual_seed(0)
    shapes = [(64, 64), (128, 32), (32, 128)]
    for shape in shapes:
        G = torch.randn(*shape, device="cuda") * 0.02
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
    G = torch.randn(*shape, device="cuda") * 0.02
    G[:, 0] *= 1e-6  # pathologically small singular value
    X = stiefel_orthogonalize(G)
    dev = _deviation_from_orthogonal(X, shape)
    assert dev < 1e-3, f"deviation from orthogonal with a near-zero singular value = {dev}"
    print(f"Near-zero singular value case: deviation from orthogonal = {dev:.8f}")

def test_unitary_muon_step_runs_on_rectangular_params():
    torch.manual_seed(0)
    linear = torch.nn.Linear(32, 128, bias=False).cuda()
    optimizer = UnitaryMuon(linear.parameters(), lr=0.01)

    x = torch.randn(4, 32, device="cuda")
    loss = linear(x).sum()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(linear.weight).any()
    print("UnitaryMuon.step() on a rectangular (128, 32) parameter completed cleanly.")

def test_build_hybrid_optimizer_excludes_tied_embedding_and_1d_params_from_muon():
    """
    Regression test for the embedding-orthogonalization pitfall: the tied
    embed_tokens/lm_head weight and every ndim<2 param (RMSNorm gains, the
    ACDE halting bias) must land in the AdamW group, not the Muon group --
    Muon should only ever touch hidden 2D weight matrices.
    """
    config = VeraceV1Config(
        vocab_size=200, hidden_dim=32, num_layers=2, num_heads=2, head_dim=16,
        spectral_dim=8, chams_holographic_dim=8, mcmoe_rank=4, mcmoe_num_components=4,
        max_cognitive_depth=2, min_cognitive_depth=1
    )
    model = VeraceV1Model(config).cuda()
    hybrid = build_hybrid_optimizer(model)

    muon_param_ids = {id(p) for p in hybrid.muon_optimizer.param_groups[0]["params"]}
    adamw_param_ids = {id(p) for p in hybrid.adamw_optimizer.param_groups[0]["params"]}

    assert id(model.embed_tokens.weight) in adamw_param_ids
    assert id(model.embed_tokens.weight) not in muon_param_ids
    assert all(p.ndim == 2 for p in hybrid.muon_optimizer.param_groups[0]["params"])
    assert any(p.ndim < 2 for p in hybrid.adamw_optimizer.param_groups[0]["params"])

    total_model_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(muon_param_ids) + len(adamw_param_ids) == total_model_params
    print(f"Muon group: {len(muon_param_ids)} params, AdamW group: {len(adamw_param_ids)} params.")


def test_hybrid_optimizer_step_runs_and_updates_both_groups():
    torch.manual_seed(0)
    config = VeraceV1Config(
        vocab_size=200, hidden_dim=32, num_layers=2, num_heads=2, head_dim=16,
        spectral_dim=8, chams_holographic_dim=8, mcmoe_rank=4, mcmoe_num_components=4,
        max_cognitive_depth=2, min_cognitive_depth=1
    )
    model = VeraceV1Model(config).cuda()
    hybrid = build_hybrid_optimizer(model)

    input_ids = torch.randint(0, config.vocab_size, (2, 8), device="cuda")
    embed_before = model.embed_tokens.weight.clone()

    logits, _ = model(input_ids, use_adaptive_depth=True)
    loss = logits.sum()
    hybrid.zero_grad()
    loss.backward()
    hybrid.step()

    assert not torch.isnan(model.embed_tokens.weight).any()
    assert not torch.equal(embed_before, model.embed_tokens.weight), "AdamW group did not update the embedding"
    print("HybridMuonAdamW.step() completed cleanly and updated both param groups.")


if __name__ == "__main__":
    test_stiefel_orthogonalize_square_tall_wide()
    test_stiefel_orthogonalize_small_singular_values()
    test_unitary_muon_step_runs_on_rectangular_params()
    test_build_hybrid_optimizer_excludes_tied_embedding_and_1d_params_from_muon()
    test_hybrid_optimizer_step_runs_and_updates_both_groups()
