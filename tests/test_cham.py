"""
Unit tests for CHAM Newton-Schulz Unitary Retraction (H^H * H = I)
"""
import torch
from verace_v1.modules.cham_memory import ContinuousHolographicMemory

def test_cham_unitary_retraction():
    b, s, d = 2, 8, 128
    holographic_dim = 32

    cham = ContinuousHolographicMemory(hidden_dim=d, holographic_dim=holographic_dim).cuda()
    x = torch.randn(b, s, d, device="cuda")

    output, (H_real, H_imag) = cham(x)
    assert output.shape == (b, s, d)
    assert not torch.isnan(output).any()

    # Verify Unitary Constraint: H^H * H = I
    HH_r = torch.matmul(H_real[0].T, H_real[0]) + torch.matmul(H_imag[0].T, H_imag[0])
    eye = torch.eye(holographic_dim, device=x.device)

    diff = torch.norm(HH_r - eye).item()
    assert diff < 1e-3, f"CHAM hologram is not unitary: ||H^H * H - I|| = {diff:.6f}"
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

    HH_r = torch.matmul(H_real[0].float().T, H_real[0].float()) + torch.matmul(H_imag[0].float().T, H_imag[0].float())
    eye = torch.eye(holographic_dim, device=x.device)
    diff = torch.norm(HH_r - eye).item()
    assert diff < 1e-3, f"CHAM hologram is not unitary under CUDA bf16 autocast: ||H^H * H - I|| = {diff:.6f}"
    print(f"CHAM unitary constraint under CUDA bf16 autocast verified. ||H^H * H - I|| = {diff:.6f}")

if __name__ == "__main__":
    test_cham_unitary_retraction()
    test_cham_unitary_retraction_holds_under_cuda_bf16_autocast()
