"""
Unit tests for SSSD Lie-Algebra Skew-Symmetric Unitary State Recurrence
"""
import math
import torch
from verace_v1.modules.sssd_attention import SSSDAttention

def test_sssd_unitary_norm_conservation():
    b, s, d = 2, 8, 128
    num_heads, head_dim = 2, 64

    attn = SSSDAttention(hidden_dim=d, num_heads=num_heads, head_dim=head_dim)
    x = torch.randn(b, s, d)

    output, Psi = attn(x, return_state=True)
    assert output.shape == (b, s, d)
    assert not torch.isnan(output).any()

    # Verify Lie-Algebra Norm Conservation: ||Psi_t||_F per head
    norm_initial = math.sqrt(head_dim) # Identity matrix norm
    norm_final = torch.norm(Psi[0, 0], p="fro").item()

    assert abs(norm_final - norm_initial) < 1e-4, f"SSSD state norm was not conserved: initial {norm_initial:.4f}, final {norm_final:.4f}"
    print(f"SSSD norm conservation verified. Initial: {norm_initial:.4f}, final: {norm_final:.4f}")

if __name__ == "__main__":
    import math
    test_sssd_unitary_norm_conservation()
