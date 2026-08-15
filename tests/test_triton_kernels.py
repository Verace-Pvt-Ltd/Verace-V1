"""
GPU correctness tests for Verace V1 Triton kernels.

Each module (SSSDAttention, ContinuousHolographicMemory, ManifoldContinuousMoE) exposes two
computation paths that must agree: the eager PyTorch "exact" path (used whenever the input
tensor requires grad) and the Triton fast inference path (used on CUDA when the input does
not require grad). These tests run both paths through the *same* module instance (identical
weights) on the *same* device and assert the outputs match, which is what actually exercises
each launch_* Triton kernel against its reference recurrence.

Requires a CUDA device.
"""

import pytest
import torch
import torch.nn.functional as F

from verace_v1.modules.sssd_attention import SSSDAttention
from verace_v1.modules.cham_memory import ContinuousHolographicMemory, newton_schulz_unitary_retraction
from verace_v1.modules.mcmoe import ManifoldContinuousMoE

TOL = dict(rtol=1e-3, atol=1e-4)
FP64_TOL = dict(rtol=1e-4, atol=1e-4)


def _cham_fp64_reference(model, x, active_mask=None):
    """
    Independent fp64 sequential reference for CHAM (deliberately not sharing code with
    either the eager parallel-scan path or the Triton kernel). Implements the *deferred
    retraction* semantics the eager path (cham_memory.py's parallel_complex_prefix_scan +
    a single per-position Newton-Schulz retraction of the raw prefix product) actually
    specifies: the raw prefix product P_t = U_t @ ... @ U_1 is carried forward WITHOUT ever
    being retracted internally (it is only first-order-unitary per step and is allowed to
    drift); H_0 @ P_t is retracted fresh, independently, at every readout. This is NOT the
    same function as retracting every step and feeding the retracted value back in -- that
    was tried and verified (against this same fp64 reference, and against the live eager
    module) to diverge increasingly with sequence length; this formulation matches the
    eager module directly at short/medium sequences. At long sequences (s >~ 50) the eager
    path's own fp32 O(log S) tree-scan realization of this formula has substantial
    numerical error relative to this fp64 ground truth (its raw P drifts far from unitary
    and its particular fp32 accumulation order handles that drift poorly) -- this is a
    property of the eager implementation, not of this reference or of the Triton kernel,
    both of which track this fp64 answer to ~1e-6 regardless of sequence length.
    """
    with torch.no_grad():
        q = F.normalize(model.w_q(x), p=2, dim=-1).double()
        k = F.normalize(model.w_k(x), p=2, dim=-1).double()
        v = F.normalize(F.silu(model.w_v(x)), p=2, dim=-1).double()
        gamma = (0.1 * torch.sigmoid(model.w_gamma(x))).double()
        if active_mask is not None:
            am = active_mask
            while am.ndim < gamma.ndim:
                am = am.unsqueeze(-1)
            gamma = gamma * am.to(gamma.dtype)

        b, s, h_dim = q.shape
        eye = torch.eye(h_dim, dtype=torch.float64, device=x.device)
        rec = torch.zeros(b, s, h_dim, dtype=torch.float64, device=x.device)
        for bi in range(b):
            Pr, Pi = eye.clone(), torch.zeros(h_dim, h_dim, dtype=torch.float64, device=x.device)
            for t in range(s):
                kt, vt, qt, gt = k[bi, t], v[bi, t], q[bi, t], gamma[bi, t, 0]
                KV = torch.outer(kt, vt)
                Pr, Pi = Pr - gt * (KV @ Pi), Pi + gt * (KV @ Pr)
                Hr, Hi = Pr.clone(), Pi.clone()  # H_0 = I -> H_0 @ P_t == P_t
                for _ in range(3):
                    HHr = Hr.T @ Hr + Hi.T @ Hi
                    HHi = Hr.T @ Hi - Hi.T @ Hr
                    dr, di = 3.0 * eye - HHr, -HHi
                    Hr, Hi = 0.5 * (Hr @ dr - Hi @ di), 0.5 * (Hr @ di + Hi @ dr)
                rec[bi, t] = Hr @ qt
        return model.norm(model.w_out(rec.to(torch.float32)))


# =====================================================================
# SSSD (Cayley-transform unitary scan)
# =====================================================================
@pytest.mark.parametrize("b,s,head_dim,num_heads", [
    (2, 8, 16, 2),
    (1, 1, 16, 1),
    (2, 13, 16, 2),  # non-power-of-2 seq_len
    (2, 8, 24, 2),   # non-power-of-2 head_dim -> exercises boundary masking
])
def test_sssd_triton_matches_reference(b, s, head_dim, num_heads):
    torch.manual_seed(0)
    hidden_dim = num_heads * head_dim
    model = SSSDAttention(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim).cuda()
    x = torch.randn(b, s, hidden_dim, device="cuda")

    out_ref, _ = model(x.clone().requires_grad_(True))
    with torch.no_grad():
        out_tri, _ = model(x.clone())

    assert out_tri.shape == out_ref.shape
    assert not torch.isnan(out_tri).any()
    torch.testing.assert_close(out_tri, out_ref.detach(), **TOL)


def test_sssd_triton_halted_tokens_match_reference():
    """active_mask=False forces delta=0 for the back half -- exercises the Woodbury
    kernel's near-zero-delta boundary branch (must fall back to identity, not NaN/Inf)."""
    torch.manual_seed(1)
    b, s, head_dim, num_heads = 2, 10, 16, 2
    hidden_dim = num_heads * head_dim
    model = SSSDAttention(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim).cuda()
    x = torch.randn(b, s, hidden_dim, device="cuda")
    active_mask = torch.ones(b, s, dtype=torch.bool, device="cuda")
    active_mask[:, s // 2:] = False

    out_ref, _ = model(x.clone().requires_grad_(True), active_mask=active_mask)
    with torch.no_grad():
        out_tri, _ = model(x.clone(), active_mask=active_mask)

    assert not torch.isnan(out_tri).any()
    torch.testing.assert_close(out_tri, out_ref.detach(), **TOL)


def test_sssd_triton_initial_state_roundtrip():
    """Verifies the Triton path actually honors initial_state (chunked/incremental
    decoding), which the previous kernel silently ignored."""
    torch.manual_seed(2)
    b, s, head_dim, num_heads = 2, 6, 16, 2
    hidden_dim = num_heads * head_dim
    model = SSSDAttention(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim).cuda()
    x1 = torch.randn(b, s, hidden_dim, device="cuda")
    x2 = torch.randn(b, s, hidden_dim, device="cuda")

    with torch.no_grad():
        _, state1_tri = model(x1.clone(), return_state=True)
        out2_tri, _ = model(x2.clone(), initial_state=state1_tri, return_state=True)

    _, state1_ref = model(x1.clone().requires_grad_(True), return_state=True)
    out2_ref, _ = model(
        x2.clone().requires_grad_(True), initial_state=state1_ref.detach(), return_state=True
    )

    torch.testing.assert_close(state1_tri, state1_ref.detach().to(state1_tri.dtype), **TOL)
    torch.testing.assert_close(out2_tri, out2_ref.detach(), **TOL)


# =====================================================================
# CHAM (holographic unitary update + Newton-Schulz retraction)
# =====================================================================
@pytest.mark.parametrize("b,s,holographic_dim", [
    (2, 8, 16),
    (1, 1, 16),
    (2, 5, 16),    # non-power-of-2 seq_len
    (2, 8, 20),    # non-power-of-2 holographic_dim -> boundary masking + tl.dot padding
    (2, 6, 160),   # > 128 -> exercises the single-CTA-too-large parallel-scan fallback path
])
def test_cham_triton_matches_reference(b, s, holographic_dim):
    torch.manual_seed(3)
    hidden_dim = 32
    model = ContinuousHolographicMemory(hidden_dim=hidden_dim, holographic_dim=holographic_dim).cuda()
    x = torch.randn(b, s, hidden_dim, device="cuda")

    with torch.no_grad():
        out_tri, _ = model(x.clone())
    out_gold = _cham_fp64_reference(model, x)

    assert out_tri.shape == out_gold.shape
    assert not torch.isnan(out_tri).any()
    torch.testing.assert_close(out_tri, out_gold, **FP64_TOL)


def test_cham_triton_preserves_unitarity_with_halted_tokens():
    torch.manual_seed(4)
    b, s, holographic_dim = 2, 10, 16
    hidden_dim = 32
    model = ContinuousHolographicMemory(hidden_dim=hidden_dim, holographic_dim=holographic_dim).cuda()
    x = torch.randn(b, s, hidden_dim, device="cuda")
    active_mask = torch.ones(b, s, dtype=torch.bool, device="cuda")
    active_mask[:, s // 2:] = False

    with torch.no_grad():
        out_tri, (Hr_tri, Hi_tri) = model(x.clone(), active_mask=active_mask)
    out_gold = _cham_fp64_reference(model, x, active_mask=active_mask)

    torch.testing.assert_close(out_tri, out_gold, **FP64_TOL)

    # model()'s second return value is the RAW (never-retracted) state, by design --
    # see cham_memory.py's forward() docstring. Newton-Schulz retraction's exactness
    # guarantee (H^H H = I) applies on read, so retract before checking it.
    Hr_ro, Hi_ro = newton_schulz_unitary_retraction(Hr_tri, Hi_tri)
    HH_r = Hr_ro.transpose(-1, -2) @ Hr_ro + Hi_ro.transpose(-1, -2) @ Hi_ro
    eye = torch.eye(holographic_dim, device="cuda").unsqueeze(0).expand(b, -1, -1)
    torch.testing.assert_close(HH_r, eye, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("s", [8, 16, 32])
def test_cham_triton_matches_live_eager_at_practical_lengths(s):
    """Direct trained-vs-served check (no fp64 helper involved): at the sequence lengths
    this is actually likely to run at, Triton and the live eager forward() must agree
    tightly, since eager is what the model is trained against."""
    torch.manual_seed(6)
    b, holographic_dim, hidden_dim = 2, 16, 32
    model = ContinuousHolographicMemory(hidden_dim=hidden_dim, holographic_dim=holographic_dim).cuda()
    x = torch.randn(b, s, hidden_dim, device="cuda")

    out_eager, _ = model(x.clone().requires_grad_(True))
    with torch.no_grad():
        out_tri, _ = model(x.clone())

    torch.testing.assert_close(out_tri, out_eager.detach(), rtol=5e-2, atol=2e-2)


def test_cham_eager_path_matches_fp64_ground_truth_at_long_sequences():
    """
    cham_memory.py's parallel_newton_schulz_retraction had a real bug (not a numerical
    precision limitation): it computed H_real's updated value, then reused that
    already-updated H_real (instead of the pre-update value) to compute H_imag,
    corrupting the complex multiplication on every iteration past the first. This
    compounded over long sequences, previously producing >5e-3 error against a fp64
    ground truth at s=100 here (formerly documented as "known eager-path fragility" --
    it wasn't fragility, it was a wrong formula). Fixed by computing both outputs from
    the same pre-update (H_real, H_imag) pair; the eager path now tracks fp64 ground
    truth as tightly as the Triton kernel (which never had this bug) at long sequences.
    """
    torch.manual_seed(7)
    b, s, holographic_dim, hidden_dim = 2, 100, 16, 32
    model = ContinuousHolographicMemory(hidden_dim=hidden_dim, holographic_dim=holographic_dim).cuda()
    x = torch.randn(b, s, hidden_dim, device="cuda")

    out_eager, _ = model(x.clone().requires_grad_(True))
    with torch.no_grad():
        out_tri, _ = model(x.clone())
    out_gold = _cham_fp64_reference(model, x)

    tri_err = (out_tri.double() - out_gold.double()).abs().max().item()
    eager_err = (out_eager.detach().double() - out_gold.double()).abs().max().item()

    assert tri_err < 1e-4, f"Triton should stay near fp64 ground truth at s={s}, got {tri_err:.2e}"
    assert eager_err < 1e-4, f"Eager should now stay near fp64 ground truth at s={s} too, got {eager_err:.2e}"


# =====================================================================
# M-CMoE (sparse Top-K manifold projection)
# =====================================================================
@pytest.mark.parametrize("b,s,hidden_dim,num_components,top_k,rank", [
    (2, 8, 32, 8, 2, 4),
    (1, 1, 32, 8, 2, 4),
    (2, 8, 40, 6, 3, 4),  # non-power-of-2 hidden_dim -> boundary masking
    (2, 8, 32, 8, 8, 4),  # top_k == num_components (dense limit)
])
def test_mcmoe_triton_matches_reference(b, s, hidden_dim, num_components, top_k, rank):
    torch.manual_seed(5)
    model = ManifoldContinuousMoE(
        hidden_dim=hidden_dim, rank=rank, num_components=num_components, top_k_components=top_k
    ).cuda()
    x = torch.randn(b, s, hidden_dim, device="cuda")

    out_ref = model(x.clone().requires_grad_(True))
    with torch.no_grad():
        out_tri = model(x.clone())

    assert out_tri.shape == out_ref.shape
    assert not torch.isnan(out_tri).any()
    torch.testing.assert_close(out_tri, out_ref.detach(), **TOL)
