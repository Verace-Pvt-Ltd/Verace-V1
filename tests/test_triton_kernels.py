"""
GPU correctness tests for Verace V1 Triton kernels.

Each module (SSSDAttention, ContinuousHolographicMemory, ManifoldContinuousMoE) exposes two
computation paths that must agree: the eager PyTorch "exact" path (used whenever the input
tensor requires grad) and the Triton fast inference path (used on CUDA when the input does
not require grad). These tests run both paths through the *same* module instance (identical
weights) on the *same* device and assert the outputs match, which is what actually exercises
each launch_* Triton kernel against its reference recurrence.

CHAM's Triton kernels (launch_cham_triton_update for holographic_dim <= 128, its
launch_cham_triton_tiled fallback above that) implement an exact Cayley-transform unitary
update via a Sherman-Morrison-Woodbury closed form (see either kernel's docstring in
verace_v1/serving/triton_kernels.py) -- previously they implemented only a first-order
approximation matching the eager path's old formula, which caused unbounded,
sequence-length-compounding drift from unitarity (root cause of NaN training divergence at
chams_holographic_dim >= 48); both the eager path and the Triton kernels were fixed to the
exact construction (see git history).

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
    Independent fp64 sequential reference for CHAM's exact-Cayley unitary update
    (deliberately not sharing code with the eager parallel-scan path), using native
    complex128 throughout for precision. Per-token factor U_t = (I - i*G_t/2)^{-1}(I +
    i*G_t/2) for Hermitian generator G_t = gamma_t*(k_t v_t^T + v_t k_t^T) -- matching
    cham_memory.py's forward() exactly (see that module for why this construction is
    EXACTLY unitary, unlike the first-order approximation U_t = I + i*gamma_t*(k_t v_t^T)
    it replaced, whose deviation from unitarity compounded unboundedly with sequence
    length and was the root cause of NaN training divergence at chams_holographic_dim
    >= 48). Implements the *deferred retraction* semantics the eager path actually
    specifies: the raw prefix product P_t = U_t @ ... @ U_1 is carried forward WITHOUT
    ever being retracted internally; H_0 @ P_t is retracted fresh, independently, at
    every readout (see cham_memory.py's forward() docstring/comments).
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
        eye_c = torch.eye(h_dim, dtype=torch.complex128, device=x.device)
        eye_r = torch.eye(h_dim, dtype=torch.float64, device=x.device)
        rec = torch.zeros(b, s, h_dim, dtype=torch.float64, device=x.device)
        for bi in range(b):
            P = eye_c.clone()
            for t in range(s):
                kt, vt, qt, gt = k[bi, t], v[bi, t], q[bi, t], gamma[bi, t, 0]
                G = (gt * (torch.outer(kt, vt) + torch.outer(vt, kt))).to(torch.complex128)
                right = eye_c - 0.5j * G
                left = eye_c + 0.5j * G
                U = torch.linalg.solve(right, left)
                P = U @ P
                # Fresh independent retraction at readout (H_0 = I -> H_0 @ P_t == P_t)
                Hr, Hi = P.real.clone(), P.imag.clone()
                for _ in range(3):
                    HHr = Hr.T @ Hr + Hi.T @ Hi
                    HHi = Hr.T @ Hi - Hi.T @ Hr
                    dr, di = 3.0 * eye_r - HHr, -HHi
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
    (2, 6, 160),   # > 128 -> exercises the tiled (launch_cham_triton_tiled) fallback path
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


def test_cham_triton_zero_gamma_is_exact_noop_for_halted_tokens():
    """
    Regression test for the Woodbury closed-form update's removable singularity at
    gamma=0 (halted tokens): the analytic N11/N12 -> 0 smoothly as gamma -> 0, but the
    derivation's intermediate b_i = -2/gamma term is a literal division by zero at
    gamma=0 exactly, which real training/inference hits deliberately (active_mask=False
    forces gamma=0, not just gamma-near-0). Both Triton kernels clamp gamma before that
    division and explicitly zero the resulting update when gamma is ~0, rather than
    relying on the singularity being "removable" to also be safe in fp32. This forces
    gamma=0 for every token and asserts the hologram is untouched (H stays exactly I).
    """
    torch.manual_seed(5)
    b, s, holographic_dim, hidden_dim = 2, 12, 16, 32
    model = ContinuousHolographicMemory(hidden_dim=hidden_dim, holographic_dim=holographic_dim).cuda()
    x = torch.randn(b, s, hidden_dim, device="cuda")
    active_mask = torch.zeros(b, s, dtype=torch.bool, device="cuda")  # every token halted

    with torch.no_grad():
        out_tri, (Hr_tri, Hi_tri) = model(x.clone(), active_mask=active_mask)

    assert not torch.isnan(out_tri).any() and not torch.isinf(out_tri).any()
    eye = torch.eye(holographic_dim, device="cuda").unsqueeze(0).expand(b, -1, -1)
    torch.testing.assert_close(Hr_tri, eye, **TOL)
    torch.testing.assert_close(Hi_tri, torch.zeros_like(Hi_tri), **TOL)


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
    corrupting the complex multiplication on every iteration past the first. Separately,
    the per-token update itself was only a first-order approximation to unitary, whose
    deviation compounded unboundedly with sequence length (root cause of NaN training
    divergence at chams_holographic_dim >= 48) -- fixed via an exact Cayley-transform
    construction. This checks both eager and Triton track the (now-matching) fp64 ground
    truth tightly even at long sequences, where either bug would have shown up.
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
    assert eager_err < 1e-4, f"Eager should stay near fp64 ground truth at s={s} too, got {eager_err:.2e}"


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
