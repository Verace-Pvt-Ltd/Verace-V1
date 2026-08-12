"""
Unit tests for SSSD Lie-Algebra Skew-Symmetric Unitary State Recurrence
"""
import math
import torch
from verace_v1.modules.sssd_attention import SSSDAttention

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

if __name__ == "__main__":
    import math
    test_sssd_unitary_norm_conservation()
    test_sssd_norm_conservation_holds_under_cuda_bf16_autocast()
