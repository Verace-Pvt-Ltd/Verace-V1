"""
Unit tests for SSSD Lie-Algebra Skew-Symmetric Unitary State Recurrence
"""
import math
import warnings

import torch
from verace_v1.modules.sssd_attention import SSSDAttention, _solve_cayley

def test_sssd_unitary_norm_conservation():
    b, s, d = 2, 8, 128
    num_heads, head_dim = 2, 64

    attn = SSSDAttention(hidden_dim=d, num_heads=num_heads, head_dim=head_dim).cuda()
    x = torch.randn(b, s, d, device="cuda")

    output, Psi = attn(x, return_state=True)
    assert output.shape == (b, s, d)
    assert not torch.isnan(output).any()

    # Verify Lie-Algebra Norm Conservation: ||Psi_t||_F per head
    norm_initial = math.sqrt(head_dim) # Identity matrix norm
    norm_final = torch.norm(Psi[0, 0], p="fro").item()

    assert abs(norm_final - norm_initial) < 1e-4, f"SSSD state norm was not conserved: initial {norm_initial:.4f}, final {norm_final:.4f}"
    print(f"SSSD norm conservation verified. Initial: {norm_initial:.4f}, final: {norm_final:.4f}")

def test_sssd_norm_conservation_holds_under_cuda_bf16_autocast():
    """
    Regression test: the exact Cayley-transform scan used to crash under
    torch.autocast(dtype=bfloat16) with a dtype mismatch on the in-place
    parallel-scan writes (fp32 destination, bf16 matmul source) -- autocast
    forces torch.linalg.solve to fp32 but casts the following matmuls back
    down to bf16. The fix forces this block to run in fp32 with autocast
    disabled -- assert it neither crashes nor loses the exact
    norm-conservation guarantee when called from inside a real bf16-AMP
    training step on GPU.
    """
    b, s, d = 2, 8, 128
    num_heads, head_dim = 2, 64

    attn = SSSDAttention(hidden_dim=d, num_heads=num_heads, head_dim=head_dim).cuda()
    x = torch.randn(b, s, d, device="cuda", requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        output, Psi = attn(x, return_state=True)
        output.sum().backward()

    assert not torch.isnan(output).any()
    assert x.grad is not None and not torch.isnan(x.grad).any()

    norm_initial = math.sqrt(head_dim)
    norm_final = torch.norm(Psi[0, 0].float(), p="fro").item()
    assert abs(norm_final - norm_initial) < 1e-4, (
        f"SSSD state norm was not conserved under CUDA bf16 autocast: initial {norm_initial:.4f}, final {norm_final:.4f}"
    )
    print(f"SSSD norm conservation under CUDA bf16 autocast verified. Initial: {norm_initial:.4f}, final: {norm_final:.4f}")


def test_solve_cayley_recovers_from_gpu_solve_returning_nonfinite(monkeypatch):
    """
    Regression test for a real training-divergence root cause: cusolver's batched GPU
    torch.linalg.solve occasionally returned non-finite values for the Cayley transform's
    `right` matrix (provably non-singular in exact arithmetic: right = I + 0.5*delta*A for
    skew-symmetric A has eigenvalues with real part exactly 1), causing a hard, instant
    jump from healthy hidden states to fully-NaN mid-training with no gradual warning.
    Forces solve to fail on the original (non-identity) matrix but succeed on the
    identity-substituted one, matching what the fix actually does: asserts a warning is
    raised but no NaN propagates in the forward output.
    """
    real_solve = torch.linalg.solve
    d = 8
    eye_mat = torch.eye(d, device="cuda")

    def flaky_solve(A, B, *args, **kwargs):
        if torch.allclose(A.reshape(-1, d, d)[0], eye_mat):
            return real_solve(A, B, *args, **kwargs)
        return torch.full_like(B, float("nan"))

    monkeypatch.setattr(torch.linalg, "solve", flaky_solve)

    eye = eye_mat.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # A non-identity `right` so flaky_solve fails on the first (probe) call.
    right = (eye_mat * 2.0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    left = eye.clone()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _solve_cayley(right, left, eye)
        assert any("substituting a well-conditioned identity pair" in str(warning.message) for warning in w)

    assert not torch.isnan(result).any()
    torch.testing.assert_close(result, eye.expand_as(result))
    print("_solve_cayley recovered from a simulated GPU solve failure via input substitution.")


def test_solve_cayley_gradient_is_finite_when_gpu_solve_returns_nonfinite(monkeypatch):
    """
    Regression test for the actual root cause behind test_solve_cayley_recovers_from_gpu_solve_
    returning_nonfinite continuing to diverge even after that fix landed: patching only the
    *forward* output of a bad solve (e.g. via torch.where) does not stop NaN from leaking into
    the *gradient* of `right`/`left`, because torch.linalg.solve's backward is an independent
    adjoint solve that reuses the (still non-finite) primal result regardless of what happens to
    the output downstream -- confirmed directly against a genuinely near-singular `right` (not a
    monkeypatch) that produces NaN forward output. The fix must substitute the *input* before
    solving so the backward pass never touches the pathological matrix either.
    """
    d = 8
    right = torch.eye(d, device="cuda").clone()
    with torch.no_grad():
        right[-1, -1] = 1e-40  # near-singular, not exactly singular (avoids the strict
                                # LinAlgError check) but numerically catastrophic: 1/1e-40
                                # overflows fp32's ~3.4e38 max -> inf/nan in solve's output.
    right.requires_grad_(True)
    left = torch.eye(d, device="cuda", requires_grad=True)
    eye = torch.eye(d, device="cuda")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _solve_cayley(right, left, eye)
        assert any("substituting a well-conditioned identity pair" in str(warning.message) for warning in w)

    assert not torch.isnan(result).any() and not torch.isinf(result).any()

    result.sum().backward()
    assert right.grad is not None and not torch.isnan(right.grad).any() and not torch.isinf(right.grad).any(), (
        "right.grad is non-finite -- input substitution failed to protect the backward pass"
    )
    assert left.grad is not None and not torch.isnan(left.grad).any() and not torch.isinf(left.grad).any(), (
        "left.grad is non-finite -- input substitution failed to protect the backward pass"
    )
    print("_solve_cayley's gradient stays finite even when the raw solve's forward is non-finite.")


def test_solve_cayley_falls_back_to_zero_if_solve_fails_even_after_input_substitution(monkeypatch):
    """Last-resort case: if solve fails even on the trivially well-conditioned
    identity-substituted input (should essentially never happen in practice), must
    zero the remaining non-finite entries rather than propagate NaN."""
    def always_nan(A, B, *args, **kwargs):
        return torch.full_like(B, float("nan"))

    monkeypatch.setattr(torch.linalg, "solve", always_nan)

    d = 8
    eye = torch.eye(d, device="cuda").unsqueeze(0).unsqueeze(0).unsqueeze(0)
    right = eye.clone()
    left = eye.clone()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _solve_cayley(right, left, eye)
        assert any("forcing remaining non-finite entries to 0" in str(warning.message) for warning in w)

    assert not torch.isnan(result).any()
    torch.testing.assert_close(result, torch.zeros_like(result))
    print("_solve_cayley zeroed remaining non-finite entries as a last resort.")


if __name__ == "__main__":
    import math
    test_sssd_unitary_norm_conservation()
    test_sssd_norm_conservation_holds_under_cuda_bf16_autocast()
