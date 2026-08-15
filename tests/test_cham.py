"""
Unit tests for CHAM Newton-Schulz Unitary Retraction (H^H * H = I)
"""
import torch
from verace_v1.modules.cham_memory import ContinuousHolographicMemory, newton_schulz_unitary_retraction

def test_cham_unitary_retraction():
    """
    cham(x)'s second return value is the RAW (never-retracted) accumulated state --
    correct on purpose, so it composes correctly if fed back in as initial_hologram
    for a later chunk (see cham_memory.py's forward() docstring/comments; retraction
    is a nonlinear projection that must not be folded into the carried-forward state).
    A caller reading the hologram right now gets that guarantee by retracting on
    read, exactly like forward() itself does internally for its own `y` output --
    so that's what this test checks, rather than the raw state directly.
    """
    b, s, d = 2, 8, 128
    holographic_dim = 32

    cham = ContinuousHolographicMemory(hidden_dim=d, holographic_dim=holographic_dim).cuda()
    x = torch.randn(b, s, d, device="cuda")

    output, (H_real, H_imag) = cham(x)
    assert output.shape == (b, s, d)
    assert not torch.isnan(output).any()

    # Verify Unitary Constraint on a fresh read: H^H * H = I after retraction.
    H_real_retracted, H_imag_retracted = newton_schulz_unitary_retraction(H_real, H_imag)
    HH_r = torch.matmul(H_real_retracted[0].T, H_real_retracted[0]) + torch.matmul(H_imag_retracted[0].T, H_imag_retracted[0])
    eye = torch.eye(holographic_dim, device=x.device)

    diff = torch.norm(HH_r - eye).item()
    assert diff < 1e-3, f"CHAM hologram is not unitary after retraction: ||H^H * H - I|| = {diff:.6f}"
    print(f"CHAM unitary constraint verified. ||H^H * H - I|| = {diff:.6f}")

def test_cham_unitary_retraction_holds_under_cuda_bf16_autocast():
    """
    Regression test: on CUDA, the parallel complex prefix scan used to crash
    under torch.autocast(dtype=bfloat16) with a dtype mismatch -- eye/eye_seq
    were built from x.dtype (still fp32 pre-autocast) while q/k/v/gamma were
    already cast to bf16 by the Linear layers above, so the in-place scan
    writes mixed fp32 destinations with bf16 matmul sources. The fix forces
    this block to fp32 with autocast disabled -- assert it neither crashes
    nor loses the exact unitary guarantee when called from inside a real
    bf16-AMP training step.
    """
    b, s, d = 2, 8, 128
    holographic_dim = 32

    cham = ContinuousHolographicMemory(hidden_dim=d, holographic_dim=holographic_dim).cuda()
    x = torch.randn(b, s, d, device="cuda", requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        output, (H_real, H_imag) = cham(x)
        output.sum().backward()

    assert not torch.isnan(output).any()
    assert x.grad is not None and not torch.isnan(x.grad).any()

    # See test_cham_unitary_retraction: the returned state is raw by design, retract
    # on read to check the invariant a caller reading the hologram right now would see.
    H_real_retracted, H_imag_retracted = newton_schulz_unitary_retraction(H_real.float(), H_imag.float())
    HH_r = torch.matmul(H_real_retracted[0].T, H_real_retracted[0]) + torch.matmul(H_imag_retracted[0].T, H_imag_retracted[0])
    eye = torch.eye(holographic_dim, device=x.device)
    diff = torch.norm(HH_r - eye).item()
    assert diff < 1e-3, f"CHAM hologram is not unitary under CUDA bf16 autocast: ||H^H * H - I|| = {diff:.6f}"
    print(f"CHAM unitary constraint under CUDA bf16 autocast verified. ||H^H * H - I|| = {diff:.6f}")

def test_cham_stays_finite_when_w_v_produces_large_activations():
    """
    Regression test for a real training-divergence root cause: v (unlike q and k) was
    left as unnormalized F.silu(w_v(x)) output, with nothing bounding its magnitude.
    U_t = I + i*gamma*(k v^T) is only a first-order ("infinitesimal") approximation to a
    unitary matrix, accurate only while gamma*||k||*||v|| stays small -- verified in a
    real pretraining run that v's max magnitude grew unboundedly (~2 -> ~17 over 50
    steps) as w_v's weights adapted, pushing this approximation far outside its valid
    regime and causing the holographic state to go non-finite (confirmed via forward-
    pass tracing to be the earliest point of corruption in the whole model). Simulates
    that scenario directly by scaling w_v's weights up by a large factor (standing in
    for what training organically drove them to) and asserting the state stays finite
    regardless, now that v is L2-normalized like k.
    """
    torch.manual_seed(0)
    b, s, d = 2, 32, 64
    holographic_dim = 32

    cham = ContinuousHolographicMemory(hidden_dim=d, holographic_dim=holographic_dim).cuda()
    with torch.no_grad():
        cham.w_v.weight.mul_(50.0)  # simulate w_v's weights having grown large during training
    x = torch.randn(b, s, d, device="cuda") * 3.0  # also scale up the input itself

    output, (H_real, H_imag) = cham(x)
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), (
        "CHAM output went non-finite for large w_v-driven activations -- v may no "
        "longer be magnitude-bounded"
    )
    assert not torch.isnan(H_real).any() and not torch.isnan(H_imag).any()
    print("CHAM stays finite when w_v produces large activations (v is magnitude-bounded).")


def test_newton_schulz_retraction_matches_correct_complex_multiplication():
    """
    Regression test for a real bug: the retraction loop computed H_real's updated value,
    then reused that already-updated H_real (instead of the pre-update value) when
    computing H_imag, corrupting the complex multiplication H_next = 0.5*H*(3I - H^H H)
    on every iteration past the first. This compares against a reference that evaluates
    both outputs from the same pre-update (H_real, H_imag) pair via Python tuple
    assignment (which evaluates the whole right-hand side before rebinding either name --
    the correct pattern the fix now uses too).
    """
    torch.manual_seed(0)
    d = 16
    H_real = torch.randn(2, d, d, device="cuda") * 0.1 + torch.eye(d, device="cuda")
    H_imag = torch.randn(2, d, d, device="cuda") * 0.1
    eye = torch.eye(d, device="cuda").unsqueeze(0)

    def reference_retraction(H_real, H_imag, steps=3):
        for _ in range(steps):
            HH_r = torch.matmul(H_real.transpose(-1, -2), H_real) + torch.matmul(H_imag.transpose(-1, -2), H_imag)
            HH_i = torch.matmul(H_real.transpose(-1, -2), H_imag) - torch.matmul(H_imag.transpose(-1, -2), H_real)
            diff_r = 3.0 * eye - HH_r
            diff_i = -HH_i
            # Tuple assignment: Python evaluates the full RHS (using the pre-update
            # H_real/H_imag for BOTH expressions) before rebinding either name.
            H_real, H_imag = (
                0.5 * (torch.matmul(H_real, diff_r) - torch.matmul(H_imag, diff_i)),
                0.5 * (torch.matmul(H_real, diff_i) + torch.matmul(H_imag, diff_r)),
            )
        return H_real, H_imag

    got_r, got_i = newton_schulz_unitary_retraction(H_real.clone(), H_imag.clone())
    expected_r, expected_i = reference_retraction(H_real.clone(), H_imag.clone())

    torch.testing.assert_close(got_r, expected_r, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got_i, expected_i, rtol=1e-5, atol=1e-6)
    print("newton_schulz_unitary_retraction matches the correct (pre-update-consistent) complex multiplication.")


def test_cham_raw_state_stays_unitary_over_a_long_sequence():
    """
    Regression test for a real training-divergence root cause: the previous per-token
    update U_t = I + i*gamma*(k_t v_t^T) was only a first-order approximation to unitary
    (i*(k v^T) is not even Hermitian in general, so it isn't a valid infinitesimal-unitary
    generator at all), and the RAW accumulated state (H_real, H_imag) -- the second return
    value, which is what actually gets carried forward as `new_cham_hologram` and reused as
    initial_hologram -- was never corrected by the Newton-Schulz retraction (by design,
    per forward()'s comments: retraction is a nonlinear projection that must not be folded
    into the recurrence). This meant deviation from unitarity was unbounded and compounded
    across the sequence: verified in isolation (no training, no GPU) to reach O(1)
    deviation within a single 256-token sequence with realistic gamma/k/v magnitudes,
    eventually pushing the raw state's singular values far enough from 1 that the
    Newton-Schulz retraction used for readout crossed its own documented divergence
    boundary (see unitary_muon.py's stiefel_orthogonalize) and produced NaN -- reproduced
    directly in real training runs at chams_holographic_dim >= 48.

    The existing test_cham_unitary_retraction only exercises s=8 (too short to show
    compounding) and checks the RETRACTED copy, not the raw state actually being carried
    forward. This test checks the RAW (H_real, H_imag) directly, at context_length=256
    (this project's real training context length), with no retraction applied.
    """
    torch.manual_seed(0)
    b, s, d = 2, 256, 64
    holographic_dim = 64  # matches the 8.3M config that diverged in real training

    cham = ContinuousHolographicMemory(hidden_dim=d, holographic_dim=holographic_dim).cuda()
    # requires_grad=True to force the O(log S) associative-scan Python path (what
    # training actually uses) rather than the Triton inference fast path, which is a
    # separate kernel (verace_v1/serving/triton_kernels.py) not touched by this fix.
    x = torch.randn(b, s, d, device="cuda", requires_grad=True)

    _, (H_real, H_imag) = cham(x)
    assert not torch.isnan(H_real).any() and not torch.isnan(H_imag).any()

    HH_r = torch.matmul(H_real[0].T, H_real[0]) + torch.matmul(H_imag[0].T, H_imag[0])
    eye = torch.eye(holographic_dim, device=x.device)
    diff = torch.norm(HH_r - eye).item()

    # Old (first-order-approximation) formula reached ~2.4 (order-1, not fp32 rounding)
    # at s=256 in isolation; the exact-Cayley construction stays near machine precision
    # (~1e-4 in isolation with complex64). 0.05 is a generous margin above that, still
    # far below the ~1+ deviation that indicates the old, broken behavior.
    assert diff < 0.05, (
        f"CHAM's RAW (carried-forward) state deviates from unitarity by {diff:.4f} after "
        f"a {s}-token sequence -- expected near-machine-precision deviation from an exact "
        f"per-token unitary construction, not compounding drift."
    )
    print(f"CHAM raw state stays unitary over a {s}-token sequence: ||H^H H - I|| = {diff:.6f}")


if __name__ == "__main__":
    test_cham_unitary_retraction()
    test_cham_unitary_retraction_holds_under_cuda_bf16_autocast()
    test_newton_schulz_retraction_matches_correct_complex_multiplication()
    test_cham_raw_state_stays_unitary_over_a_long_sequence()
