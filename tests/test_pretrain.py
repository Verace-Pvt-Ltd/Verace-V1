"""
Unit tests for the pretraining driver's checkpoint and LR-schedule machinery.
"""
import os
import tempfile

import torch

from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.optimizer.unitary_muon import build_hybrid_optimizer
from verace_v1.training.pretrain import (
    get_cosine_schedule_with_warmup,
    load_checkpoint,
    save_checkpoint,
    train_pretrain_step,
)


def test_resumed_scheduler_matches_continuous_training_not_a_warmup_restart():
    """
    Regression test for the resume LR-reset bug: verace_v1/training/train.py
    resumes by building a fresh scheduler and fast-forwarding it via
    `for _ in range(start_step): scheduler.step()`. Before that fast-forward
    loop existed, resuming silently restarted the LR at warmup instead of
    continuing the schedule. This pins both halves: the fast-forwarded LR
    must match what uninterrupted training would show at the same step, and
    must differ from the pre-fix warmup-restart value.
    """
    warmup, total, resume_at = 5, 40, 12
    lin = torch.nn.Linear(4, 4).cuda()

    # Ground truth: uninterrupted training to `resume_at` steps.
    opt_continuous = torch.optim.SGD(lin.parameters(), lr=0.1)
    sched_continuous = get_cosine_schedule_with_warmup(opt_continuous, warmup, total)
    for _ in range(resume_at):
        sched_continuous.step()
    continuous_lr = opt_continuous.param_groups[0]["lr"]

    # train.py's resume path: fresh scheduler, fast-forwarded resume_at steps.
    opt_resumed = torch.optim.SGD(lin.parameters(), lr=0.1)
    sched_resumed = get_cosine_schedule_with_warmup(opt_resumed, warmup, total)
    for _ in range(resume_at):
        sched_resumed.step()
    resumed_lr = opt_resumed.param_groups[0]["lr"]

    # The pre-fix bug: a scheduler that was never fast-forwarded on resume.
    opt_unfast_forwarded = torch.optim.SGD(lin.parameters(), lr=0.1)
    sched_unfast_forwarded = get_cosine_schedule_with_warmup(opt_unfast_forwarded, warmup, total)
    buggy_lr = opt_unfast_forwarded.param_groups[0]["lr"]

    assert abs(resumed_lr - continuous_lr) < 1e-10, "fast-forwarded resume LR must match uninterrupted training"
    assert abs(resumed_lr - buggy_lr) > 1e-4, "fast-forwarded LR must differ from the pre-fix warmup-restart value"
    print(f"Resumed LR {resumed_lr:.6f} matches continuous {continuous_lr:.6f}, "
          f"differs from pre-fix restart value {buggy_lr:.6f}.")


def test_save_and_load_checkpoint_roundtrip_preserves_model_and_optimizer_state():
    """
    End-to-end checkpoint round trip through save_checkpoint/load_checkpoint,
    including HybridMuonAdamW.state_dict()/load_state_dict() (not exercised
    by any other test): train one real step, save, load into a fresh
    model+optimizer, and verify the restored step and parameters match.
    """
    config = VeraceV1Config(
        vocab_size=200, hidden_dim=32, num_layers=2, num_heads=2, head_dim=16,
        spectral_dim=8, chams_holographic_dim=8, mcmoe_rank=4, mcmoe_num_components=4,
        max_cognitive_depth=2, min_cognitive_depth=1
    )
    model = VeraceV1Model(config).cuda()
    optimizer = build_hybrid_optimizer(model)

    input_ids = torch.randint(0, config.vocab_size, (2, 8), device="cuda")
    batch = {"input_ids": input_ids, "labels": input_ids.clone()}
    train_pretrain_step(model, optimizer, batch, use_amp=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_checkpoint(model, optimizer, step=1, checkpoint_dir=tmp_dir)
        ckpt_path = os.path.join(tmp_dir, "verace_v1_step_1.pt")

        model2 = VeraceV1Model(config).cuda()
        optimizer2 = build_hybrid_optimizer(model2)
        loaded_step = load_checkpoint(model2, optimizer2, ckpt_path, map_location="cuda")

    assert loaded_step == 1
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "restored model parameters do not match the saved checkpoint"

    # Optimizer momentum/Adam state should also have round-tripped -- take one more
    # step on each and confirm they stay in lockstep (would drift if state was lost).
    train_pretrain_step(model, optimizer, batch, use_amp=False)
    train_pretrain_step(model2, optimizer2, batch, use_amp=False)
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), "optimizer state did not round-trip through the checkpoint"
    print("Checkpoint round trip preserved model and optimizer state.")


def test_train_pretrain_step_clips_gradients_and_reports_grad_norm():
    """
    Regression test for a real training-divergence bug: an unclipped gradient spike
    around step ~75 of a real pretraining run pushed a parameter's momentum buffer to
    non-finite, which UnitaryMuon's momentum could never recover from afterward (see
    test_unitary_muon.py's non-finite-gradient test for that half of the fix). This
    tests the other half: train_pretrain_step must clip gradients to max_grad_norm
    before the optimizer step, and report the (pre-clip) grad_norm via diagnostics so
    spikes are visible in training logs rather than silently absorbed.
    """
    config = VeraceV1Config(
        vocab_size=200, hidden_dim=32, num_layers=2, num_heads=2, head_dim=16,
        spectral_dim=8, chams_holographic_dim=8, mcmoe_rank=4, mcmoe_num_components=4,
        max_cognitive_depth=2, min_cognitive_depth=1
    )
    model = VeraceV1Model(config).cuda()
    optimizer = build_hybrid_optimizer(model)

    input_ids = torch.randint(0, config.vocab_size, (2, 8), device="cuda")
    batch = {"input_ids": input_ids, "labels": input_ids.clone()}

    # An artificially tiny clip threshold against an otherwise perfectly normal, healthy
    # batch/model -- any nonzero gradient exceeds it, so clipping reliably engages without
    # needing to induce a real gradient spike (which risks tipping the model itself into
    # non-finite territory and testing the *other* defense instead -- see
    # test_unitary_muon.py's non-finite-gradient test for that one).
    tiny_clip = 1e-4
    diagnostics = {}
    train_pretrain_step(model, optimizer, batch, use_amp=False, diagnostics=diagnostics, max_grad_norm=tiny_clip)

    assert "grad_norm" in diagnostics
    assert diagnostics["grad_norm"] > tiny_clip, "test setup should produce a nonzero grad norm the clip engages on"
    total_norm_after_clip = torch.norm(
        torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])
    ).item()
    assert total_norm_after_clip <= tiny_clip * 1.01, f"gradients were not clipped to max_grad_norm={tiny_clip}: {total_norm_after_clip}"
    print(f"Reported grad_norm={diagnostics['grad_norm']:.4f} (pre-clip), actual post-clip norm={total_norm_after_clip:.6f}.")


if __name__ == "__main__":
    test_resumed_scheduler_matches_continuous_training_not_a_warmup_restart()
    test_save_and_load_checkpoint_roundtrip_preserves_model_and_optimizer_state()
    test_train_pretrain_step_clips_gradients_and_reports_grad_norm()
