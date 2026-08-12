"""
Unit tests for training-time diagnostics (CHAM invariant probe, depth stats).
"""
import torch

from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.modules.cham_memory import newton_schulz_unitary_retraction
from verace_v1.optimizer.unitary_muon import build_hybrid_optimizer
from verace_v1.training.diagnostics import CHAMInvariantProbe, depth_distribution_stats
from verace_v1.training.pretrain import train_pretrain_step


def _small_config():
    return VeraceV1Config(
        vocab_size=200, hidden_dim=32, num_layers=3, num_heads=2, head_dim=16,
        spectral_dim=8, chams_holographic_dim=8, mcmoe_rank=4, mcmoe_num_components=4,
        max_cognitive_depth=3, min_cognitive_depth=1
    )


def test_cham_invariant_probe_captures_raw_state_during_real_training_step():
    """
    The probe hooks CHAM's real forward output during training -- confirming
    the hook wiring actually captures live output (not silently nothing) and
    that train_pretrain_step's optional `diagnostics` dict gets populated as a
    side effect. The captured state is the RAW (never-retracted) carried
    -forward state by design (see CHAMInvariantProbe's docstring), so its
    deviation from unitary is a finite, sane, non-negative number -- not
    necessarily near-zero. Separately, retracting that SAME captured state
    (what a caller reading the hologram right now would do) must land close
    to unitary, proving retraction itself still works correctly on real
    training-distribution activations, not just on random test tensors.
    """
    torch.manual_seed(0)
    config = _small_config()
    model = VeraceV1Model(config).cuda()
    optimizer = build_hybrid_optimizer(model)
    probe = CHAMInvariantProbe(model)

    input_ids = torch.randint(0, config.vocab_size, (2, 8), device="cuda")
    batch = {"input_ids": input_ids, "labels": input_ids.clone()}

    diagnostics = {}
    train_pretrain_step(model, optimizer, batch, use_amp=False, diagnostics=diagnostics)
    deviation = probe.pop_mean_deviation()

    assert deviation is not None
    assert deviation >= 0.0 and deviation < 10.0, f"CHAM raw-state deviation implausible: {deviation}"
    assert "depth_mean" in diagnostics and "depth_std" in diagnostics
    print(f"CHAM probe captured raw-state deviation={deviation:.6f} during a real training step.")

    # Buffer clears itself each call -- nothing new captured since last pop.
    assert probe.pop_mean_deviation() is None
    probe.remove()


def test_cham_invariant_probe_captured_state_retracts_cleanly():
    """Retracting the raw state the probe captures (what a reader would do)
    must land close to unitary -- proving retraction itself is healthy on
    real training-distribution activations, complementing the raw-drift check
    above."""
    torch.manual_seed(1)
    config = _small_config()
    model = VeraceV1Model(config).cuda()
    optimizer = build_hybrid_optimizer(model)
    probe = CHAMInvariantProbe(model)

    input_ids = torch.randint(0, config.vocab_size, (2, 8), device="cuda")
    batch = {"input_ids": input_ids, "labels": input_ids.clone()}
    train_pretrain_step(model, optimizer, batch, use_amp=False)

    assert len(probe._captured) > 0
    for H_real, H_imag in probe._captured:
        H_real_ro, H_imag_ro = newton_schulz_unitary_retraction(H_real, H_imag)
        d = H_real_ro.shape[-1]
        eye = torch.eye(d, device=H_real_ro.device).unsqueeze(0)
        HH_r = torch.matmul(H_real_ro.transpose(-1, -2), H_real_ro) + torch.matmul(H_imag_ro.transpose(-1, -2), H_imag_ro)
        diff = torch.norm(HH_r - eye, dim=(-2, -1)).max().item()
        assert diff < 1e-2, f"Retracted CHAM state is not unitary: {diff}"
    probe.remove()
    print("Retracting the probe's captured raw state lands close to unitary.")


def test_depth_distribution_stats_matches_manual_computation():
    depth_counts = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    stats = depth_distribution_stats(depth_counts)

    assert abs(stats["depth_mean"] - 3.5) < 1e-6
    assert stats["depth_min"] == 1.0
    assert stats["depth_max"] == 6.0
    assert stats["depth_std"] > 0.0
    print(f"depth_distribution_stats: {stats}")


if __name__ == "__main__":
    test_cham_invariant_probe_captures_raw_state_during_real_training_step()
    test_cham_invariant_probe_captured_state_retracts_cleanly()
    test_depth_distribution_stats_matches_manual_computation()
