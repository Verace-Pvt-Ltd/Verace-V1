"""
Unit tests for CHAM Newton-Schulz Unitary Retraction (H^H * H = I)
"""
import torch
from verace_v1.modules.cham_memory import ContinuousHolographicMemory

def test_cham_unitary_retraction():
    b, s, d = 2, 8, 128
    holographic_dim = 32

    cham = ContinuousHolographicMemory(hidden_dim=d, holographic_dim=holographic_dim)
    x = torch.randn(b, s, d)

    output, (H_real, H_imag) = cham(x)
    assert output.shape == (b, s, d)
    assert not torch.isnan(output).any()

    # Verify Unitary Constraint: H^H * H = I
    HH_r = torch.matmul(H_real[0].T, H_real[0]) + torch.matmul(H_imag[0].T, H_imag[0])
    eye = torch.eye(holographic_dim, device=x.device)

    diff = torch.norm(HH_r - eye).item()
    assert diff < 1e-3, f"CHAM hologram is not unitary: ||H^H * H - I|| = {diff:.6f}"
    print(f"CHAM unitary constraint verified. ||H^H * H - I|| = {diff:.6f}")

if __name__ == "__main__":
    test_cham_unitary_retraction()
