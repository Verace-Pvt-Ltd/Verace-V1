"""
Triton GPU Kernels for Verace V1
Implements fused Triton GPU kernels for:
1. SSSD exact Lie-algebra Cayley-transform unitary state scan
2. CHAM fused holographic associative memory update (with Newton-Schulz unitary retraction)
3. M-CMoE fused sparse Top-K continuous manifold projection
The Unitary Muon optimizer (verace_v1/optimizer/unitary_muon.py) runs its SVD-based
orthogonalization via torch.linalg.svd rather than a kernel in this module.
These kernels require a CUDA device; there is no CPU fallback path in this module.

Every kernel below implements the *same* recurrence as the eager PyTorch reference path
in its corresponding verace_v1/modules/*.py file (this is a hard correctness requirement,
verified by tests/test_triton_kernels.py against the reference on-GPU). Each per-step
recurrence is sequential across the sequence dimension (one Triton program per batch, or
per batch*head, holding the full state tile for the duration of the scan) -- this favors
low-latency incremental / prefill inference over the O(log S) parallel-prefix-scan training
path used by the eager fallback. Because each program holds a dense (head_dim x head_dim)
or (holographic_dim x holographic_dim) state tile, this design has a practical ceiling on
state dimension set by the GPU's register/shared-memory budget; it has been validated at
the dimensions used by tests/test_end2end.py and tests/test_triton_kernels.py. Scaling to
the full production chams_holographic_dim (1024) would require a tiled multi-block
redesign of the CHAM kernel and is out of scope here.
"""

import torch
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# =====================================================================
# 1. SSSD Exact Cayley-Transform Unitary Scan Triton GPU Kernel
# =====================================================================
if HAS_TRITON:
    @triton.jit
    def sssd_cayley_scan_kernel(
        q_ptr, k_ptr, v_ptr, delta_ptr, psi_init_ptr,
        out_ptr, psi_out_ptr,
        num_heads, seq_len, head_dim,
        BLOCK_D: tl.constexpr
    ):
        """
        Sequential exact Cayley-transform unitary state scan, one program per (batch, head).

        R_t = (I - 0.5*delta_t*A_t)^{-1} (I + 0.5*delta_t*A_t),  A_t = k_t v_t^T - v_t k_t^T
        Psi_t = R_t @ Psi_{t-1},   o_t = Psi_t @ q_t

        A_t is skew-symmetric of rank <= 2 (A_t = U C U^T, U = [k_t, v_t], C = [[0,1],[-1,0]]).
        The rank-2 Sherman-Morrison-Woodbury identity is used to apply R_t to Psi_{t-1}
        exactly, without ever forming or inverting a (head_dim x head_dim) matrix:

            Y            = Psi + U (0.5*delta*C) (U^T Psi)                     -- (I + X) Psi
            Psi_new       = Y + U S^{-1} (U^T Y),  S = (0.5*delta*C)^{-1} - U^T U   -- (I - X)^{-1} Y

        S is a 2x2 matrix, so S^{-1} is closed-form. This is exact (not a series
        approximation) to floating-point precision for delta > EPS; for delta <= EPS
        (halted tokens, where the reference model forces delta = 0) the update is skipped
        entirely and Psi is left unchanged, matching the true delta -> 0 limit R_t -> I.
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs = tl.arange(0, BLOCK_D)
        mask_d = offs < head_dim
        mask_2d = mask_d[:, None] & mask_d[None, :]

        # q/k/v/out are [batch, seq_len, num_heads, head_dim] contiguous (NOT
        # [batch, num_heads, seq_len, head_dim]) -- strides must reflect that layout.
        qkv_batch_stride = seq_len * num_heads * head_dim
        qkv_step_stride = num_heads * head_dim
        h_base = pid_b * qkv_batch_stride + pid_h * head_dim

        # delta is [batch, seq_len, num_heads] contiguous.
        delta_batch_stride = seq_len * num_heads
        delta_h_base = pid_b * delta_batch_stride + pid_h

        bh = pid_b * num_heads + pid_h
        state_base = bh * head_dim * head_dim

        Psi = tl.load(
            psi_init_ptr + state_base + offs[:, None] * head_dim + offs[None, :],
            mask=mask_2d, other=0.0
        ).to(tl.float32)

        EPS: tl.constexpr = 1e-3

        for t in range(seq_len):
            t_off = h_base + t * qkv_step_stride
            q = tl.load(q_ptr + t_off + offs, mask=mask_d, other=0.0).to(tl.float32)
            k = tl.load(k_ptr + t_off + offs, mask=mask_d, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + t_off + offs, mask=mask_d, other=0.0).to(tl.float32)
            delta_t = tl.load(delta_ptr + delta_h_base + t * num_heads).to(tl.float32)

            active = tl.where(delta_t > EPS, 1.0, 0.0)
            delta_safe = tl.where(delta_t > EPS, delta_t, 1.0)
            # The eager reference computes R_t = torch.linalg.solve(I + X, I - X) = (I+X)^{-1}(I-X)
            # (note: opposite operand order from the module's docstring). This is algebraically
            # R_t(X) with X -> -X of the (I-X)^{-1}(I+X) form the Woodbury identity below was
            # derived for, so delta is negated here to match the reference exactly.
            delta_eff = -delta_safe

            # Y = (I + X) Psi, X = U (0.5*delta_eff*C) U^T
            M0 = tl.sum(k[:, None] * Psi, axis=0)
            M1 = tl.sum(v[:, None] * Psi, axis=0)
            cm0 = 0.5 * delta_eff * M1
            cm1 = -0.5 * delta_eff * M0
            Y = Psi + k[:, None] * cm0[None, :] + v[:, None] * cm1[None, :]

            # Psi_candidate = (I - X)^{-1} Y = Y + U S^{-1} (U^T Y)
            N0 = tl.sum(k[:, None] * Y, axis=0)
            N1 = tl.sum(v[:, None] * Y, axis=0)

            rho = tl.sum(k * v)
            cinv01 = -2.0 / delta_eff
            cinv10 = 2.0 / delta_eff
            s00 = -1.0
            s01 = cinv01 - rho
            s10 = cinv10 - rho
            s11 = -1.0
            det = s00 * s11 - s01 * s10
            sinv00 = s11 / det
            sinv01 = -s01 / det
            sinv10 = -s10 / det
            sinv11 = s00 / det

            sn0 = sinv00 * N0 + sinv01 * N1
            sn1 = sinv10 * N0 + sinv11 * N1
            Psi_candidate = Y + k[:, None] * sn0[None, :] + v[:, None] * sn1[None, :]

            Psi = Psi + active * (Psi_candidate - Psi)
            Psi = tl.where(mask_2d, Psi, 0.0)

            o = tl.sum(Psi * q[None, :], axis=1)
            tl.store(out_ptr + t_off + offs, o.to(out_ptr.dtype.element_ty), mask=mask_d)

        tl.store(
            psi_out_ptr + state_base + offs[:, None] * head_dim + offs[None, :],
            Psi, mask=mask_2d
        )


# =====================================================================
# 2. CHAM Holographic Unitary Update Triton GPU Kernel
# =====================================================================
if HAS_TRITON:
    @triton.jit
    def cham_holographic_scan_kernel(
        q_ptr, k_ptr, v_ptr, gamma_ptr,
        h_real_init_ptr, h_imag_init_ptr,
        out_ptr, h_real_out_ptr, h_imag_out_ptr,
        seq_len, h_dim,
        BLOCK_H: tl.constexpr
    ):
        """
        Sequential complex unitary holographic scan, one program per batch element.

        The reference builds this as P_t = U_t @ U_{t-1} @ ... @ U_1 (a RAW, never-retracted
        prefix product, via cham_memory.py's parallel_complex_prefix_scan), then composes it
        with the incoming state H_0 as Total_t = P_t @ H_0 (H_0 on the RIGHT -- new tokens'
        rotations are left-multiplied onto the running product, so continuing correctly
        across a call boundary requires H_0 to sit after this chunk's own P, not before it;
        H_0 @ P_t is only equivalent when H_0 = I, which is why this was invisible on a
        single fresh sequence) and retracts (Newton-Schulz) each position's Total_t
        independently for the readout only. Retraction is a nonlinear projection, so this is
        NOT the same function as retracting after every step and feeding the retracted value
        back into the recurrence (that alternative was tried and verified to diverge from the
        reference, growing with sequence length, since it corrects drift the reference's
        formula does not correct until the very end of each position's own prefix). To match
        exactly: P (raw, unitary only to first order per step, never corrected) is what's
        actually carried across steps AND exported as the final state for a caller to
        continue from; H_0 is composed in and NS-retracted fresh at every position for the
        readout only, and next step's rank-1 update is applied to the raw P, not to the
        retracted readout, and next step's rank-2 update is applied to the raw P, not to
        the retracted readout.

        U_t = I + W N W^T (W=[k_t,v_t], N a 2x2 complex matrix computed per-token from
        k_t.v_t and gamma_t via Sherman-Morrison-Woodbury) is the EXACT Cayley transform
        of the Hermitian generator G_t = gamma_t*(k_t v_t^T + v_t k_t^T) -- see this
        function's per-token comment below, and cham_memory.py's forward()/module
        docstring, for the derivation and why exactness (not just a first-order
        approximation) matters. Since W has rank <= 2, U_t @ P is applied in O(h_dim^2)
        without a dense h_dim x h_dim matmul: W N W^T @ P reduces to two rank-1 outer-
        product updates driven by the reductions k^T P and v^T P. P @ H_0 and the
        Newton-Schulz retraction matmuls are genuine dense products (tl.dot).
        """
        pid_b = tl.program_id(0)

        offs = tl.arange(0, BLOCK_H)
        mask_d = offs < h_dim
        mask_2d = mask_d[:, None] & mask_d[None, :]
        eye = tl.where((offs[:, None] == offs[None, :]) & mask_2d, 1.0, 0.0)

        seq_base = pid_b * seq_len * h_dim
        state_base = pid_b * h_dim * h_dim

        H0_r = tl.load(
            h_real_init_ptr + state_base + offs[:, None] * h_dim + offs[None, :],
            mask=mask_2d, other=0.0
        ).to(tl.float32)
        H0_i = tl.load(
            h_imag_init_ptr + state_base + offs[:, None] * h_dim + offs[None, :],
            mask=mask_2d, other=0.0
        ).to(tl.float32)

        P_r = eye
        P_i = tl.zeros((BLOCK_H, BLOCK_H), dtype=tl.float32)

        NS_STEPS: tl.constexpr = 3
        # Hraw_r/Hraw_i: this chunk's local raw P combined with H_0 (Total_t = P_t @ H_0),
        # BEFORE retraction -- this, not the purely-local P_r/P_i and not the retracted
        # Hro_r/Hro_i, is what must be exported as the continuable state (see module
        # docstring: H_0 belongs on the right, and retraction must never feed back in).
        Hraw_r = H0_r
        Hraw_i = H0_i

        for t in range(seq_len):
            t_off = seq_base + t * h_dim
            q = tl.load(q_ptr + t_off + offs, mask=mask_d, other=0.0).to(tl.float32)
            k = tl.load(k_ptr + t_off + offs, mask=mask_d, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + t_off + offs, mask=mask_d, other=0.0).to(tl.float32)
            g = tl.load(gamma_ptr + pid_b * seq_len + t).to(tl.float32)

            # Raw prefix update: P <- U_t @ P. Never retracted.
            # U_t = I + W N W^T (W=[k,v]) is the EXACT Cayley transform of the Hermitian
            # generator G_t = g*(k v^T + v k^T), via Sherman-Morrison-Woodbury (only 2x2
            # complex arithmetic -- no dense d x d solve needed, since G_t is rank <= 2).
            # Matches verace_v1/modules/cham_memory.py's forward() exactly; replaces the
            # old first-order-only approximation U_t = I + i*g*(k v^T), whose deviation
            # from unitarity compounded unboundedly across the sequence (root cause of
            # NaN training divergence at holographic_dim >= 48 -- see git history).
            # Derivation/validation: see cham_memory.py's module docstring.
            kv = tl.sum(k * v)
            eps: tl.constexpr = 1e-7
            is_active = tl.where(tl.abs(g) < eps, 0.0, 1.0)
            g_safe = tl.where(tl.abs(g) < eps, eps, g)

            b_r = -kv
            b_i = -2.0 / g_safe
            denom_r = 1.0 - b_r * b_r + b_i * b_i
            denom_i = -2.0 * b_r * b_i
            dmag2 = denom_r * denom_r + denom_i * denom_i
            inv_r = denom_r / dmag2
            inv_i = -denom_i / dmag2
            M11_r = -inv_r
            M11_i = -inv_i
            M12_r = -b_r * inv_r + b_i * inv_i
            M12_i = -(b_r * inv_i + b_i * inv_r)
            mp_diag_r = M11_r + kv * M12_r
            mp_diag_i = M11_i + kv * M12_i
            mp_off_r = kv * M11_r + M12_r
            mp_off_i = kv * M11_i + M12_i
            N11_r = (M11_r - (g_safe / 2.0) * mp_off_i) * is_active
            N11_i = (M11_i + (g_safe / 2.0) * mp_off_r) * is_active
            N12_r = (M12_r - (g_safe / 2.0) * mp_diag_i) * is_active
            N12_i = (M12_i + (g_safe / 2.0) * (1.0 + mp_diag_r)) * is_active
            # N21 = N12, N22 = N11 (symmetric, since G_t is symmetric in k, v)

            kT_Pr = tl.sum(k[:, None] * P_r, axis=0)
            kT_Pi = tl.sum(k[:, None] * P_i, axis=0)
            vT_Pr = tl.sum(v[:, None] * P_r, axis=0)
            vT_Pi = tl.sum(v[:, None] * P_i, axis=0)

            C_r = N11_r * kT_Pr - N11_i * kT_Pi + N12_r * vT_Pr - N12_i * vT_Pi
            C_i = N11_r * kT_Pi + N11_i * kT_Pr + N12_r * vT_Pi + N12_i * vT_Pr
            D_r = N12_r * kT_Pr - N12_i * kT_Pi + N11_r * vT_Pr - N11_i * vT_Pi
            D_i = N12_r * kT_Pi + N12_i * kT_Pr + N11_r * vT_Pi + N11_i * vT_Pr

            P_r = P_r + k[:, None] * C_r[None, :] + v[:, None] * D_r[None, :]
            P_i = P_i + k[:, None] * C_i[None, :] + v[:, None] * D_i[None, :]

            # Combined raw readout: Total_t = P_t @ H_0 (H_0 on the right -- see docstring).
            Hraw_r = tl.dot(P_r, H0_r, allow_tf32=False) - tl.dot(P_i, H0_i, allow_tf32=False)
            Hraw_i = tl.dot(P_r, H0_i, allow_tf32=False) + tl.dot(P_i, H0_r, allow_tf32=False)

            # Retract a separate copy for this step's output only -- discarded after the
            # readout, does NOT feed back into P or Hraw.
            Hro_r, Hro_i = Hraw_r, Hraw_i
            for _ in range(NS_STEPS):
                Ht_r = tl.trans(Hro_r)
                Ht_i = tl.trans(Hro_i)
                HH_r = tl.dot(Ht_r, Hro_r, allow_tf32=False) + tl.dot(Ht_i, Hro_i, allow_tf32=False)
                HH_i = tl.dot(Ht_r, Hro_i, allow_tf32=False) - tl.dot(Ht_i, Hro_r, allow_tf32=False)
                diff_r = 3.0 * eye - HH_r
                diff_i = -HH_i
                Hro_r_new = 0.5 * (tl.dot(Hro_r, diff_r, allow_tf32=False) - tl.dot(Hro_i, diff_i, allow_tf32=False))
                Hro_i_new = 0.5 * (tl.dot(Hro_r, diff_i, allow_tf32=False) + tl.dot(Hro_i, diff_r, allow_tf32=False))
                Hro_r, Hro_i = Hro_r_new, Hro_i_new

            Hro_r = tl.where(mask_2d, Hro_r, 0.0)
            Hro_i = tl.where(mask_2d, Hro_i, 0.0)

            rec = tl.sum(Hro_r * q[None, :], axis=1)
            tl.store(out_ptr + t_off + offs, rec.to(out_ptr.dtype.element_ty), mask=mask_d)

        # Final returned state is the last position's RAW (never-retracted) combined total
        # Hraw_r/Hraw_i, matching the reference's H_r_seq_raw[:, -1] / H_i_seq_raw[:, -1] --
        # NOT the retracted Hro_r/Hro_i, and NOT the H_0-agnostic local P_r/P_i, either of
        # which would silently break correctness for a caller that feeds this back in as
        # the next call's H_0.
        tl.store(h_real_out_ptr + state_base + offs[:, None] * h_dim + offs[None, :], Hraw_r, mask=mask_2d)
        tl.store(h_imag_out_ptr + state_base + offs[:, None] * h_dim + offs[None, :], Hraw_i, mask=mask_2d)


# =====================================================================
# 3. M-CMoE Sparse Top-K Continuous Manifold Projection Triton GPU Kernel
# =====================================================================
if HAS_TRITON:
    @triton.jit
    def mcmoe_manifold_gpu_kernel(
        x_ptr, u_basis_ptr, v_basis_ptr, topk_idx_ptr, topk_w_ptr, sigma_ptr, out_ptr,
        hidden_dim, num_components, rank,
        TOP_K: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        """
        Fused sparse Top-K M-CMoE projection, one program per token.
        Only the TOP_K selected components (gathered via topk_idx/topk_w, computed by the
        router+softmax+topk in Python, identical to the eager reference path) are visited --
        this must NOT loop over all num_components or use unnormalized router logits, both
        of which silently defeat the sparse-MoE routing this kernel exists to accelerate.

        delta_w(x) = sum_{j in TopK} phi_j(x) * (U_j diag(sigma_j(x)) V_j^T) @ x
        """
        pid_t = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask_d = offs < hidden_dim

        x = tl.load(x_ptr + pid_t * hidden_dim + offs, mask=mask_d, other=0.0).to(tl.float32)
        acc_out = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for kk in range(TOP_K):
            comp_idx = tl.load(topk_idx_ptr + pid_t * TOP_K + kk)
            w = tl.load(topk_w_ptr + pid_t * TOP_K + kk).to(tl.float32)

            u_proj = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for r in range(rank):
                v_k_r = tl.load(
                    v_basis_ptr + (comp_idx * rank + r) * hidden_dim + offs, mask=mask_d, other=0.0
                ).to(tl.float32)
                sig_k_r = tl.load(sigma_ptr + (pid_t * num_components + comp_idx) * rank + r).to(tl.float32)
                dp = tl.sum(x * v_k_r) * sig_k_r
                u_k_r = tl.load(
                    u_basis_ptr + (comp_idx * hidden_dim + offs) * rank + r, mask=mask_d, other=0.0
                ).to(tl.float32)
                u_proj += dp * u_k_r

            acc_out += w * u_proj

        tl.store(out_ptr + pid_t * hidden_dim + offs, acc_out.to(out_ptr.dtype.element_ty), mask=mask_d)


# =====================================================================
# 2b. Tiled CHAM kernels (holographic_dim > 128): the fused per-timestep kernel above
#     requires one CTA to hold the whole (h_dim x h_dim) state in registers, which is
#     infeasible past roughly h_dim=128 (4MB for a single fp32 matrix at h_dim=1024, far
#     beyond any GPU's register/shared-memory budget). These decompose every matmul the
#     recurrence needs into a genuine multi-block tiled GEMM, at the price of driving the
#     sequential (batch, timestep, Newton-Schulz-iteration) loop from Python with many
#     kernel launches instead of one fused kernel -- no zero-DRAM-roundtrip property here.
# =====================================================================
if HAS_TRITON:
    @triton.jit
    def _tiled_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        """Standard blocked C = A @ B, fp32 accumulation, masked on all three dims.
        A/B may be logical transposes of a contiguous tensor (pass a `.transpose(-1,-2)`
        view's strides directly) -- there is no separate transpose flag or code path."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k0 < K)
            b_mask = (offs_k[:, None] + k0 < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)

    @triton.jit
    def _cham_rank2_update_kernel(
        hr_ptr, hi_ptr, k_ptr, v_ptr,
        kth_r_ptr, kth_i_ptr, vth_r_ptr, vth_i_ptr,
        n11_r_ptr, n11_i_ptr, n12_r_ptr, n12_i_ptr,
        d,
        stride_hr_m, stride_hr_n, stride_hi_m, stride_hi_n,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        """In-place H += k(C) + v(D) over a (row,col) tile grid, where C = N11*(k^T H) +
        N12*(v^T H), D = N12*(k^T H) + N11*(v^T H) (all complex; N21=N12, N22=N11 since
        the generator is symmetric in k,v) -- the exact rank-2 Cayley-transform update,
        matching cham_holographic_scan_kernel's per-token comment (same math, same
        derivation) but with N11/N12 precomputed host-side in _cham_tiled_step (cheap
        2x2 scalar arithmetic, not worth a separate kernel launch) and passed in as
        single-element tensors. Needs no K-loop/reduction here, only the already-reduced
        k^T H / v^T H vectors (computed by separate tiled matmul calls with M=1, since
        the correction is rank <= 2)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < d
        mask_n = offs_n < d
        mask = mask_m[:, None] & mask_n[None, :]

        k_tile = tl.load(k_ptr + offs_m, mask=mask_m, other=0.0)
        v_tile = tl.load(v_ptr + offs_m, mask=mask_m, other=0.0)
        kthr = tl.load(kth_r_ptr + offs_n, mask=mask_n, other=0.0)
        kthi = tl.load(kth_i_ptr + offs_n, mask=mask_n, other=0.0)
        vthr = tl.load(vth_r_ptr + offs_n, mask=mask_n, other=0.0)
        vthi = tl.load(vth_i_ptr + offs_n, mask=mask_n, other=0.0)
        n11_r = tl.load(n11_r_ptr)
        n11_i = tl.load(n11_i_ptr)
        n12_r = tl.load(n12_r_ptr)
        n12_i = tl.load(n12_i_ptr)

        c_r = n11_r * kthr - n11_i * kthi + n12_r * vthr - n12_i * vthi
        c_i = n11_r * kthi + n11_i * kthr + n12_r * vthi + n12_i * vthr
        d_r = n12_r * kthr - n12_i * kthi + n11_r * vthr - n11_i * vthi
        d_i = n12_r * kthi + n12_i * kthr + n11_r * vthi + n11_i * vthr

        hr_ptrs = hr_ptr + offs_m[:, None] * stride_hr_m + offs_n[None, :] * stride_hr_n
        hi_ptrs = hi_ptr + offs_m[:, None] * stride_hi_m + offs_n[None, :] * stride_hi_n
        hr = tl.load(hr_ptrs, mask=mask, other=0.0)
        hi = tl.load(hi_ptrs, mask=mask, other=0.0)

        hr_new = hr + k_tile[:, None] * c_r[None, :] + v_tile[:, None] * d_r[None, :]
        hi_new = hi + k_tile[:, None] * c_i[None, :] + v_tile[:, None] * d_i[None, :]

        tl.store(hr_ptrs, hr_new, mask=mask)
        tl.store(hi_ptrs, hi_new, mask=mask)


def _tiled_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """C = A @ B via the tiled Triton GEMM kernel. A, B must be 2D CUDA fp32 tensors
    (possibly non-contiguous transposed views -- strides are passed through as-is)."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"shape mismatch for matmul: {tuple(A.shape)} @ {tuple(B.shape)}"
    BLOCK_M = min(64, max(16, triton.next_power_of_2(M)))
    BLOCK_N = min(64, max(16, triton.next_power_of_2(N)))
    BLOCK_K = min(32, max(16, triton.next_power_of_2(K)))
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _tiled_matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


def _cham_tiled_step(
    Pr: torch.Tensor, Pi: torch.Tensor, H0r: torch.Tensor, H0i: torch.Tensor,
    k: torch.Tensor, v: torch.Tensor, q: torch.Tensor, g: torch.Tensor, d: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One step: raw EXACT rank-2 Cayley-transform prefix update (never retracted, carried
    forward as Pr/Pi -- see _cham_rank2_update_kernel and cham_holographic_scan_kernel's
    docstring for the math), then a fresh combined-raw readout Hraw = P @ H0 (H0 on the
    RIGHT -- see cham_holographic_scan_kernel's docstring for why: new tokens' rotations
    left-multiply onto the running product, so H0 must be composed in after this chunk's
    own P, not before it), retracted via 3x Newton-Schulz into a SEPARATE copy for this
    step's output. Returns (Pr, Pi, Hraw_r, Hraw_i, rec); Hraw_r/Hraw_i (the combined raw
    value, BEFORE retraction) -- not the purely-local Pr/Pi and not the retracted output
    -- is what the caller exports as the final continuable state if this is the last
    timestep."""
    # N11, N12 (2x2 Cayley-transform coefficients, symmetric: N21=N12, N22=N11) computed
    # host-side -- cheap scalar arithmetic, not worth a separate kernel launch. See
    # cham_holographic_scan_kernel's per-token comment for the derivation; validated to
    # match a direct d x d complex solve to machine precision.
    kv = torch.dot(k, v)
    eps = 1e-7
    is_active = torch.where(g.abs() < eps, torch.zeros_like(g), torch.ones_like(g))
    g_safe = torch.where(g.abs() < eps, torch.full_like(g, eps), g)

    b_r, b_i = -kv, -2.0 / g_safe
    denom_r = 1.0 - b_r * b_r + b_i * b_i
    denom_i = -2.0 * b_r * b_i
    dmag2 = denom_r * denom_r + denom_i * denom_i
    inv_r, inv_i = denom_r / dmag2, -denom_i / dmag2
    m11_r, m11_i = -inv_r, -inv_i
    m12_r = -b_r * inv_r + b_i * inv_i
    m12_i = -(b_r * inv_i + b_i * inv_r)
    mp_diag_r, mp_diag_i = m11_r + kv * m12_r, m11_i + kv * m12_i
    mp_off_r, mp_off_i = kv * m11_r + m12_r, kv * m11_i + m12_i
    n11_r = (m11_r - (g_safe / 2.0) * mp_off_i) * is_active
    n11_i = (m11_i + (g_safe / 2.0) * mp_off_r) * is_active
    n12_r = (m12_r - (g_safe / 2.0) * mp_diag_i) * is_active
    n12_i = (m12_i + (g_safe / 2.0) * (1.0 + mp_diag_r)) * is_active

    k_row, v_row = k.view(1, d), v.view(1, d)
    kT_Pr = _tiled_matmul(k_row, Pr).view(d)
    kT_Pi = _tiled_matmul(k_row, Pi).view(d)
    vT_Pr = _tiled_matmul(v_row, Pr).view(d)
    vT_Pi = _tiled_matmul(v_row, Pi).view(d)

    BLOCK = min(64, max(16, triton.next_power_of_2(d)))
    grid = (triton.cdiv(d, BLOCK), triton.cdiv(d, BLOCK))
    _cham_rank2_update_kernel[grid](
        Pr, Pi, k, v,
        kT_Pr, kT_Pi, vT_Pr, vT_Pi,
        n11_r.reshape(1), n11_i.reshape(1), n12_r.reshape(1), n12_i.reshape(1),
        d,
        Pr.stride(0), Pr.stride(1), Pi.stride(0), Pi.stride(1),
        BLOCK_M=BLOCK, BLOCK_N=BLOCK
    )

    Hraw_r = _tiled_matmul(Pr, H0r) - _tiled_matmul(Pi, H0i)
    Hraw_i = _tiled_matmul(Pr, H0i) + _tiled_matmul(Pi, H0r)

    eye = torch.eye(d, device=Pr.device, dtype=torch.float32)
    Hro_r, Hro_i = Hraw_r, Hraw_i
    Hro_r_t, Hro_i_t = Hro_r.transpose(-1, -2), Hro_i.transpose(-1, -2)
    for _ in range(3):
        HH_r = _tiled_matmul(Hro_r_t, Hro_r) + _tiled_matmul(Hro_i_t, Hro_i)
        HH_i = _tiled_matmul(Hro_r_t, Hro_i) - _tiled_matmul(Hro_i_t, Hro_r)
        diff_r = 3.0 * eye - HH_r
        diff_i = -HH_i
        Hro_r_new = 0.5 * (_tiled_matmul(Hro_r, diff_r) - _tiled_matmul(Hro_i, diff_i))
        Hro_i_new = 0.5 * (_tiled_matmul(Hro_r, diff_i) + _tiled_matmul(Hro_i, diff_r))
        Hro_r, Hro_i = Hro_r_new.contiguous(), Hro_i_new.contiguous()
        Hro_r_t, Hro_i_t = Hro_r.transpose(-1, -2), Hro_i.transpose(-1, -2)

    rec = _tiled_matmul(Hro_r, q.view(d, 1)).view(d)
    return Pr, Pi, Hraw_r, Hraw_i, rec


def launch_cham_triton_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
    initial_hologram: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Tiled-GEMM CHAM scan for holographic_dim too large to fuse into one CTA (see
    cham_holographic_scan_kernel's 128-dim ceiling). Same math, same deferred-retraction
    semantics, decomposed into standard blocked GEMMs so no single kernel instance ever
    holds the full (h_dim x h_dim) state. The price: this drives many Python-level kernel
    launches per (batch, timestep) rather than one fused kernel, and is not expected to
    match the fused kernel's throughput at dims where both are usable -- it exists purely
    to remove the dimension ceiling."""
    assert q.is_cuda, "This kernel requires CUDA tensors"
    b, s, d = q.shape
    out_dtype = q.dtype
    q_f, k_f, v_f, gamma_f = q.float(), k.float(), v.float(), gamma.float()

    if initial_hologram is None:
        H0r_batch = torch.eye(d, device=q.device, dtype=torch.float32).unsqueeze(0).expand(b, d, d).contiguous()
        H0i_batch = torch.zeros(b, d, d, device=q.device, dtype=torch.float32)
    else:
        H0r_batch = initial_hologram[0].float().contiguous()
        H0i_batch = initial_hologram[1].float().contiguous()

    out = torch.empty(b, s, d, device=q.device, dtype=torch.float32)
    Hr_final = torch.empty(b, d, d, device=q.device, dtype=torch.float32)
    Hi_final = torch.empty(b, d, d, device=q.device, dtype=torch.float32)

    eye_d = torch.eye(d, device=q.device, dtype=torch.float32)
    for bi in range(b):
        H0r, H0i = H0r_batch[bi].contiguous(), H0i_batch[bi].contiguous()
        Pr, Pi = eye_d.clone(), torch.zeros(d, d, device=q.device, dtype=torch.float32)
        for t in range(s):
            Pr, Pi, Hraw_r, Hraw_i, rec = _cham_tiled_step(
                Pr, Pi, H0r, H0i, k_f[bi, t], v_f[bi, t], q_f[bi, t], gamma_f[bi, t], d
            )
            out[bi, t] = rec
        # Raw (never-retracted) combined total P_last @ H0 -- not the purely-local Pr/Pi and
        # not the retracted per-step output -- see _cham_tiled_step's docstring for why
        # either of those would silently break correctness for the next call.
        Hr_final[bi] = Hraw_r
        Hi_final[bi] = Hraw_i

    return out.to(out_dtype), (Hr_final, Hi_final)


# =====================================================================
# GPU kernel launchers (CUDA required)
# =====================================================================
def launch_sssd_triton_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k, v: [batch, seq_len, num_heads, head_dim] (L2-normalized)
    delta:   [batch, seq_len, num_heads]
    initial_state: optional [batch, num_heads, head_dim, head_dim], defaults to identity.
    Returns (out [batch, seq_len, num_heads, head_dim], final_state [batch, num_heads, head_dim, head_dim]).
    """
    assert q.is_cuda, "This kernel requires CUDA tensors"
    b, s, h, d_k = q.shape

    q_c, k_c, v_c, delta_c = q.contiguous(), k.contiguous(), v.contiguous(), delta.contiguous()

    if initial_state is None:
        psi_init = torch.eye(d_k, device=q.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(b, h, d_k, d_k).contiguous()
    else:
        psi_init = initial_state.to(torch.float32).contiguous()

    out = torch.empty_like(q_c)
    psi_out = torch.empty(b, h, d_k, d_k, device=q.device, dtype=torch.float32)

    grid = (b, h)
    sssd_cayley_scan_kernel[grid](
        q_c, k_c, v_c, delta_c, psi_init,
        out, psi_out,
        h, s, d_k,
        BLOCK_D=triton.next_power_of_2(d_k)
    )
    return out, psi_out


def launch_cham_triton_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
    initial_hologram: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    q, k, v: [batch, seq_len, holographic_dim]
    gamma:   [batch, seq_len]
    initial_hologram: optional (H_real, H_imag), each [batch, holographic_dim, holographic_dim],
                       defaults to (I, 0).
    """
    assert q.is_cuda, "This kernel requires CUDA tensors"
    b, s, h_dim = q.shape

    # Hardware SRAM/Register Guard:
    # Single CTA Triton block holding (h_dim x h_dim) FP32 state requires 4MB at h_dim=1024,
    # exceeding max GPU shared memory/register capacity (228KB). Routes to the genuine tiled
    # multi-block GEMM implementation (launch_cham_triton_tiled) for h_dim > 128 -- same math,
    # same per-step order, decomposed so no single kernel instance holds the full state; see
    # that function's docstring for the price (many small kernel launches, Python-driven loop).
    if h_dim > 128:
        return launch_cham_triton_tiled(q, k, v, gamma, initial_hologram)

    q_c, k_c, v_c, gamma_c = q.contiguous(), k.contiguous(), v.contiguous(), gamma.contiguous()

    if initial_hologram is None:
        h_r_init = torch.eye(h_dim, device=q.device, dtype=torch.float32).unsqueeze(0).expand(b, h_dim, h_dim).contiguous()
        h_i_init = torch.zeros(b, h_dim, h_dim, device=q.device, dtype=torch.float32)
    else:
        h_r_init = initial_hologram[0].to(torch.float32).contiguous()
        h_i_init = initial_hologram[1].to(torch.float32).contiguous()

    out = torch.empty_like(q_c)
    h_r_out = torch.empty(b, h_dim, h_dim, device=q.device, dtype=torch.float32)
    h_i_out = torch.empty(b, h_dim, h_dim, device=q.device, dtype=torch.float32)

    grid = (b,)
    cham_holographic_scan_kernel[grid](
        q_c, k_c, v_c, gamma_c,
        h_r_init, h_i_init,
        out, h_r_out, h_i_out,
        s, h_dim,
        BLOCK_H=max(16, triton.next_power_of_2(h_dim))
    )
    return out, (h_r_out, h_i_out)


def launch_mcmoe_triton_projection(
    x_flat: torch.Tensor,
    u_basis: torch.Tensor,
    v_basis: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    x_flat: [n_tokens, hidden_dim]
    u_basis: [num_components, hidden_dim, rank]
    v_basis: [num_components, rank, hidden_dim]
    topk_indices, topk_weights: [n_tokens, top_k] -- output of the same router
        softmax+topk used by the eager reference path.
    sigma: [n_tokens, num_components, rank]
    """
    assert x_flat.is_cuda, "This kernel requires CUDA tensors"
    n_tokens, d = x_flat.shape
    num_components, _, rank = u_basis.shape
    top_k = topk_indices.shape[1]

    x_c = x_flat.contiguous()
    u_c = u_basis.contiguous()
    v_c = v_basis.contiguous()
    idx_c = topk_indices.contiguous().to(torch.int32)
    w_c = topk_weights.contiguous()
    sigma_c = sigma.contiguous()

    out = torch.empty_like(x_c)
    grid = (n_tokens,)

    mcmoe_manifold_gpu_kernel[grid](
        x_c, u_c, v_c, idx_c, w_c, sigma_c, out,
        d, num_components, rank,
        TOP_K=top_k, BLOCK_D=triton.next_power_of_2(d)
    )
    return out
