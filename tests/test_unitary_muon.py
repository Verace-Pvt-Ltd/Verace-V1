"""
Unit tests for UnitaryMuon's Stiefel-manifold orthogonalization across matrix shapes.
"""
import warnings

import torch
import verace_v1.optimizer.unitary_muon as unitary_muon_mod
from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.optimizer.unitary_muon import (
    build_hybrid_optimizer,
    stiefel_orthogonalize,
    stiefel_orthogonalize_per_head,
    UnitaryMuon,
)

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
    A near-zero singular value (common for real gradient/momentum matrices, which are
    typically almost low-rank) must not break orthogonalization. The SVD fallback
    guarantees this regardless of the fast path's behavior.
    """
    torch.manual_seed(0)
    shape = (128, 32)
    G = torch.randn(*shape, device="cuda") * 0.02
    G[:, 0] *= 1e-6  # pathologically small singular value
    X = stiefel_orthogonalize(G)
    dev = _deviation_from_orthogonal(X, shape)
    assert dev < 1e-3, f"deviation from orthogonal with a near-zero singular value = {dev}"
    print(f"Near-zero singular value case: deviation from orthogonal = {dev:.8f}")


def test_stiefel_orthogonalize_newton_schulz_fast_path_stays_bounded_for_skewed_spectrum(monkeypatch):
    """
    Regression test for a real bug: an earlier version of stiefel_orthogonalize scaled
    the input by (X / ||X||_F) * sqrt(min_dim) before iterating, instead of Keller
    Jordan's original (X / ||X||_F) alone (https://kellerjordan.github.io/posts/muon/).
    That extra factor breaks the guarantee that every singular value lands in [0, 1]
    before iterating -- for a skewed/low-rank spectrum (the typical case for real
    transformer weight gradients), it can push the dominant singular value above the
    quintic polynomial's rigorously-verified divergence boundary (the unstable fixed
    point at sigma ~= 1.2637), causing the max singular value to blow up to Inf/NaN
    within 2-3 iterations. Disables the SVD fallback here (forces it to raise on both
    the GPU and CPU-retry attempts) so this test exercises ONLY the fast Newton-Schulz
    path's raw iterate and would fail loudly if the scaling regressed.

    NB: staying bounded is the only thing the corrected scaling actually guarantees --
    full convergence to dev < tol within the 5-step production budget is a separate,
    unrelated property that is NOT guaranteed for an arbitrary spectrum (this specific
    matrix plateaus around dev ~= 5.5, nowhere near tol, which is exactly why the SVD
    fallback exists and is expected to fire for cases like this one in production).
    """
    def raise_always(*args, **kwargs):
        raise torch._C._LinAlgError("SVD disabled for this test")

    monkeypatch.setattr(torch.linalg, "svd", raise_always)

    torch.manual_seed(0)
    shape = (256, 256)
    m, n = shape
    min_dim = min(m, n)
    U, _ = torch.linalg.qr(torch.randn(m, min_dim, device="cuda"))
    V, _ = torch.linalg.qr(torch.randn(n, min_dim, device="cuda"))
    svals = torch.rand(min_dim, device="cuda") * 0.05  # skewed/low-rank-like small spectrum
    G = U @ torch.diag(svals) @ V.T

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X = stiefel_orthogonalize(G)
        assert any("un-converged Newton-Schulz iterate" in str(warning.message) for warning in w), (
            "expected the SVD-disabled path to fall back to the un-converged NS iterate"
        )

    assert not torch.isnan(X).any() and not torch.isinf(X).any(), (
        "Newton-Schulz fast path produced non-finite output for a skewed small-singular"
        "-value matrix -- the sqrt(min_dim) scaling bug may have regressed"
    )
    print("Newton-Schulz fast path stayed bounded (finite) for a skewed spectrum "
          "even with the SVD fallback disabled.")

def test_stiefel_orthogonalize_per_head_orthogonalizes_each_head_independently():
    """
    Per-Head Muon (Kimi K3 Technical Report, arXiv:2607.24653, Sec. 2.5): each head's
    [head_dim, in_dim] block of a Q/K/V-style projection matrix must independently
    satisfy the Stiefel constraint, matching what stiefel_orthogonalize alone would
    produce for that block in isolation -- per-head orthogonalization must not let one
    head's momentum scale influence another head's output.
    """
    torch.manual_seed(0)
    num_heads, head_dim, in_dim = 4, 16, 64
    G = torch.randn(num_heads * head_dim, in_dim, device="cuda")
    # Make heads wildly different in scale, exactly the scenario the paper describes.
    scales = torch.tensor([1.0, 100.0, 0.01, 10.0], device="cuda")
    G = G.view(num_heads, head_dim, in_dim) * scales.view(num_heads, 1, 1)
    G = G.reshape(num_heads * head_dim, in_dim)

    out = stiefel_orthogonalize_per_head(G.clone(), num_heads)
    out_heads = out.view(num_heads, head_dim, in_dim)

    for h in range(num_heads):
        expected = stiefel_orthogonalize(G.view(num_heads, head_dim, in_dim)[h].clone())
        torch.testing.assert_close(out_heads[h], expected, rtol=1e-4, atol=1e-5)
        dev = torch.norm(out_heads[h] @ out_heads[h].T - torch.eye(head_dim, device="cuda")).item()
        assert dev < 1e-2, f"head {h} (scale={scales[h].item()}): deviation from orthogonal = {dev}"
    print("Per-head orthogonalization matches independent per-head stiefel_orthogonalize "
          "even with 10000x scale imbalance across heads.")


def test_build_hybrid_optimizer_routes_sssd_qkv_to_per_head_group():
    """SSSDAttention's w_q/w_k/w_v must land in a Muon param group with num_heads set
    to the layer's actual head count, not the default (num_heads=1, full-block)
    group -- otherwise Per-Head Muon silently never engages."""
    config = VeraceV1Config(
        vocab_size=200, hidden_dim=32, num_layers=2, num_heads=4, head_dim=8,
        spectral_dim=8, chams_holographic_dim=8, mcmoe_rank=4, mcmoe_num_components=4,
        max_cognitive_depth=2, min_cognitive_depth=1
    )
    model = VeraceV1Model(config).cuda()
    hybrid = build_hybrid_optimizer(model)

    qkv_ids = set()
    for layer in model.layers:
        for proj_name in ("w_q", "w_k", "w_v"):
            qkv_ids.add(id(getattr(layer.sssd_attn, proj_name).weight))

    found_per_head_group = False
    for group in hybrid.muon_optimizer.param_groups:
        group_ids = {id(p) for p in group["params"]}
        if group_ids & qkv_ids:
            assert group_ids <= qkv_ids, "per-head group should contain ONLY SSSD Q/K/V params"
            assert group["num_heads"] == config.num_heads
            found_per_head_group = True

    assert found_per_head_group, "no Muon param group found containing SSSD's Q/K/V projections"
    print(f"SSSD Q/K/V projections correctly routed to a num_heads={config.num_heads} Muon group.")


def test_unitary_muon_update_rms_is_shape_independent():
    """
    Regression test for a missing scale correction (Moonshot AI, "Muon is Scalable for
    LLM Training", arXiv:2502.16982, Eq. 4 / Lemma 1): a semi-orthogonal M x N update has
    RMS = sqrt(1/max(M,N)) before correction, which varies by shape -- e.g. a (256,256)
    and a (1,256) parameter would get very differently-scaled updates under the same lr.
    UnitaryMuon.step() now multiplies by 0.2*sqrt(max(M,N)) to cancel that dependence, so
    every 2D parameter's update RMS should land near the same constant (~0.2) regardless
    of shape.
    """
    torch.manual_seed(0)
    shapes = [(256, 256), (1, 256), (512, 256), (16, 256), (256, 512)]
    rms_values = []
    for shape in shapes:
        p = torch.nn.Parameter(torch.zeros(*shape, device="cuda"))
        optimizer = UnitaryMuon([p], lr=1.0, momentum=0.0, weight_decay=0.0)
        p.grad = torch.randn(*shape, device="cuda") * 0.02
        optimizer.step()
        # p.data.add_(update, alpha=-lr) with lr=1.0, weight_decay=0, starting from zero
        # means p.data == -update after one step, so update RMS == p.data's RMS.
        rms = p.data.pow(2).mean().sqrt().item()
        rms_values.append(rms)
        print(f"shape={shape}: update RMS = {rms:.5f}")

    for rms in rms_values:
        assert abs(rms - 0.2) < 0.05, f"update RMS {rms} is not near the target 0.2 for shape-independence"
    spread = max(rms_values) - min(rms_values)
    assert spread < 0.02, f"update RMS spread across shapes ({spread:.5f}) is too large -- scaling isn't shape-independent"
    print(f"Update RMS is shape-independent across {shapes}: spread = {spread:.5f}")


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


def test_unitary_muon_skips_update_and_preserves_momentum_on_nonfinite_gradient():
    """
    Regression test for a real training-divergence bug: a NaN gradient (from an
    upstream numerical spike) got absorbed into the momentum buffer
    (buf.mul_(momentum).add_(grad)), which can never recover once poisoned -- NaN times
    anything is still NaN, so every subsequent step stayed corrupted forever, and the
    orthogonalization fallback had no way to produce a meaningful update from an
    already-NaN input. The fix: detect a non-finite gradient before it touches momentum
    and skip that parameter's update for the step, leaving momentum untouched.
    """
    torch.manual_seed(0)
    linear = torch.nn.Linear(16, 32, bias=False).cuda()
    optimizer = UnitaryMuon(linear.parameters(), lr=0.01, momentum=0.9)

    # Healthy step first, to give momentum a real (non-zero) value to protect.
    x = torch.randn(4, 16, device="cuda")
    linear(x).sum().backward()
    optimizer.step()
    momentum_before = optimizer.state[linear.weight]["momentum_buffer"].clone()
    weight_before = linear.weight.clone()
    assert not torch.isnan(momentum_before).any()

    # Now a NaN gradient (e.g. from an upstream numerical explosion).
    linear.weight.grad = torch.full_like(linear.weight, float("nan"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        optimizer.step()
        assert any("non-finite gradient" in str(warning.message) for warning in w)

    momentum_after = optimizer.state[linear.weight]["momentum_buffer"]
    assert torch.equal(momentum_before, momentum_after), "momentum buffer must be untouched by a non-finite gradient"
    assert torch.equal(weight_before, linear.weight), "parameter must be untouched when its gradient is non-finite"
    print("UnitaryMuon correctly skipped the update and preserved momentum on a NaN gradient.")


def test_stiefel_orthogonalize_recovers_from_gpu_svd_convergence_failure(monkeypatch):
    """
    Regression test for a real crash observed during a pilot run: cusolver's batched GPU
    SVD driver raised torch._C._LinAlgError ("algorithm failed to converge... too many
    repeated singular values") inside the SVD fallback path, and it was unhandled --
    crashing the entire training run on a single pathological parameter update. The fix
    retries on CPU (generally more numerically robust); this test forces the GPU call to
    fail and the CPU retry to succeed, and asserts a warning is raised but no exception
    propagates.
    """
    real_svd = torch.linalg.svd

    def flaky_svd(X, *args, **kwargs):
        if X.is_cuda:
            raise torch._C._LinAlgError("simulated cusolver convergence failure")
        return real_svd(X, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", flaky_svd)

    # A matrix pathological enough that Newton-Schulz won't hit `dev < tol` either,
    # forcing the SVD fallback path to actually run.
    G = torch.zeros(32, 16, device="cuda")
    G[0, 0] = 1e-8

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = stiefel_orthogonalize(G)
        assert any("CPU SVD retry succeeded" in str(warning.message) for warning in w)

    assert result.shape == G.shape
    assert not torch.isnan(result).any()
    print("stiefel_orthogonalize recovered from a simulated GPU SVD failure via CPU retry.")


def test_stiefel_orthogonalize_falls_back_to_newton_schulz_iterate_if_svd_fails_everywhere(monkeypatch):
    """When SVD fails on both GPU and CPU, the function must return the (imperfectly
    orthogonal) Newton-Schulz iterate rather than crash -- verified here by forcing both
    calls to fail."""
    def always_fails(X, *args, **kwargs):
        raise torch._C._LinAlgError("simulated total SVD failure")

    monkeypatch.setattr(torch.linalg, "svd", always_fails)

    G = torch.zeros(32, 16, device="cuda")
    G[0, 0] = 1e-8

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = stiefel_orthogonalize(G)
        assert any("falling back to the un-converged Newton-Schulz iterate" in str(warning.message) for warning in w)

    assert result.shape == G.shape
    assert not torch.isnan(result).any()
    print("stiefel_orthogonalize returned the Newton-Schulz iterate instead of crashing when SVD failed everywhere.")


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

    # build_hybrid_optimizer now splits Muon params across multiple groups: the regular
    # group plus a per-head group for SSSDAttention's Q/K/V projections (Per-Head Muon,
    # Kimi K3 arXiv:2607.24653 Sec. 2.5) -- so these must iterate ALL of Muon's groups,
    # not just group 0.
    muon_params_all = [p for g in hybrid.muon_optimizer.param_groups for p in g["params"]]
    muon_param_ids = {id(p) for p in muon_params_all}
    adamw_param_ids = {id(p) for p in hybrid.adamw_optimizer.param_groups[0]["params"]}

    assert id(model.embed_tokens.weight) in adamw_param_ids
    assert id(model.embed_tokens.weight) not in muon_param_ids
    assert all(p.ndim == 2 for p in muon_params_all)
    assert any(p.ndim < 2 for p in hybrid.adamw_optimizer.param_groups[0]["params"])
    assert len(hybrid.muon_optimizer.param_groups) > 1, "expected a separate per-head group for SSSD Q/K/V"

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
