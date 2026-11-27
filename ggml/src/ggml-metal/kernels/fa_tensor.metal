#include "common.h"
#include "dequantize.h"

#ifdef GGML_METAL_HAS_TENSOR

constant bool FC_flash_attn_ext_tensor_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 0)]];
constant bool FC_flash_attn_ext_tensor_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 1)]];
constant bool FC_flash_attn_ext_tensor_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 2)]];
constant bool FC_flash_attn_ext_tensor_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 3)]];

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<
    typename T,                                 // Q, K, V and P types
    typename T4,
    short DK,                                   // K head size
    short DV,                                   // V head size
    short Q   = OP_FLASH_ATTN_EXT_TENSOR_NQPSG, // queries per threadgroup
    short C   = OP_FLASH_ATTN_EXT_TENSOR_NCPSG, // cache items per threadgroup
    short NSG = OP_FLASH_ATTN_EXT_TENSOR_NSG>   // number of simd groups
kernel void kernel_flash_attn_ext_tensor(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * blk,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NT = NW*NSG;
    constexpr short NQ = Q/NSG;
    constexpr short NC = C/NW; // columns per thread

    static_assert(DK % 4 == 0,  "DK must be divisible by 4");
    static_assert(Q % NSG == 0, "Q must be divisible by NSG");
    static_assert(C % NW == 0,  "C must be divisible by NW");

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0]*Q;

    const short tiitg = sgitg*NW + tiisg;

    threadgroup T     * sq = (threadgroup T     *) shmem;       // [Q, DK] queries
    threadgroup float * ss = (threadgroup float *) (sq + Q*DK); // [Q, C]  scores
    threadgroup T     * sp = (threadgroup T     *) (ss + Q*C);  // [Q, C]  probabilities
    threadgroup float * sr = (threadgroup float *) (sp + Q*C);  // [Q]     per-row scale of O
    threadgroup int   * sf = (threadgroup int   *) (sr + Q);    // [1]     last iteration (ic0 + 1) that rescaled O

    q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

    {
        const int ikv2 = iq2/(args.ne02/args.ne_12_2);
        const int ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // with softcap the scale is small (scale/softcap), so it is applied to the scores to keep the precision of Q
    const float qscale = FC_flash_attn_ext_tensor_has_scap ? 1.0f : args.scale;

    // load the queries, with the scale folded in
    for (int i = tiitg; i < Q*DK/4; i += NT) {
        const int j = i/(DK/4);

        float4 q4 = 0.0f;
        if (iq1 + j < args.ne01) {
            q4 = ((device const float4 *) (q + j*args.nb01))[i%(DK/4)];
        }

        ((threadgroup T4 *) sq)[i] = (T4) (q4*qscale);
    }

    device const half * pm[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm[jj] = (device const half *) (mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int nblk1 = (args.ne01 + Q - 1)/Q;
        const int nblk0 = (args.ne11 + C - 1)/C;

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    float M[NQ];
    float S[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        M[jj] = -FLT_MAX/2;
        S[jj] = 0.0f;
    }

    float slope = 1.0f;

    // ALiBi
    if (FC_flash_attn_ext_tensor_has_bias) {
        const short h = iq2;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exph);
    }

    const int sk = args.ns10;
    const int sv = args.ns20;

    auto tq = tensor(sq, dextents<int32_t, 2>(DK, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C,  Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C,  Q));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, DV, C, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    auto tv0 = tensor((device T *) v, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

    // the O matrix from the paper
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    if (tiitg == 0) {
        sf[0] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // the host guarantees ne11 % C == 0
    for (int ic0 = 0, ic = 0; ic < args.ne11; ++ic0, ic += C) {
        char blk_cur = 1;

        if (FC_flash_attn_ext_tensor_has_mask) {
            blk_cur = blk[ic0];

            if (blk_cur == 0) {
                continue;
            }
        }

        // Q*K^T
        {
            auto tk = tensor((device T *) (k + (uint64_t) ic*args.nb11), dextents<int32_t, 2>(DK, C), array<int, 2>({1, sk}));

            mm_qk.run(tq, tk, ts);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;

            float s[NC];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                s[ii] = ss[j*C + ii*NW + tiisg];
            }

            if (FC_flash_attn_ext_tensor_has_scap) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] = args.logit_softcap*precise::tanh(s[ii]*args.scale);
                }
            }

            if (FC_flash_attn_ext_tensor_has_mask && blk_cur != 2 && iq1 + j < args.ne31) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] += slope*(float) pm[jj][ic + ii*NW + tiisg];
                }
            }

            float m = M[jj];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                m = max(m, s[ii]);
            }

            m = simd_max(m);

            // lazy rescaling: move the running max only when it grows by more than 8 (e^8 fits in half)
            float ms = 1.0f;

            if (m > M[jj] + 8.0f) {
                ms    = exp(M[jj] - m);
                M[jj] = m;

                if (tiisg == 0) {
                    sf[0] = ic0 + 1;
                }
            }

            float sum = 0.0f;

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                // the sum uses the same rounded values as P*V
                const T p = (T) exp(s[ii] - M[jj]);

                sp[j*C + ii*NW + tiisg] = p;

                sum += (float) p;
            }

            S[jj] = S[jj]*ms + simd_sum(sum);

            if (tiisg == 0) {
                sr[j] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(ms)*O + P*V
        if (sf[0] == ic0 + 1) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        {
            auto tv = tensor((device T *) (v + (uint64_t) ic*args.nb21), dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

            mm_pv.run(tp, tv, co);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        // the sink only adds to the denominator - its rescale of O is folded into the final scale
        float ms = 1.0f;

        if (FC_flash_attn_ext_tensor_has_sinks) {
            const float s = ((device const float *) sinks)[iq2];
            const float m = max(M[jj], s);

            ms = exp(M[jj] - m);

            S[jj] = S[jj]*ms + exp(s - m);
        }

        if (tiisg == 0) {
            sr[j] = S[jj] == 0.0f ? 0.0f : ms/S[jj];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
    }

    // store to global memory - rows past ne01 are clipped by the tensor extents
    device float * pdst = (device float *) dst + ((uint64_t) iq3*args.ne2*args.ne1 + iq2 + (uint64_t) iq1*args.ne1)*DV;

    auto td = tensor(pdst, dextents<int32_t, 2>(DV, args.ne01 - iq1), array<int, 2>({1, args.ne1*DV}));

    co.store(td);
}

typedef decltype(kernel_flash_attn_ext_tensor<half, half4, 64, 64>) flash_attn_ext_tensor_t;

template [[host_name("kernel_flash_attn_ext_tensor_f16_dk64_dv64"  )]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 64,  64>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk128_dv128")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 128, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk192_dv128")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 192, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk256_dv256")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 256, 256>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk512_dv512")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 512, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk576_dv512")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 576, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;

template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk64_dv64"  )]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 64,  64>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk128_dv128")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 128, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk192_dv128")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 192, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk256_dv256")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 256, 256>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk512_dv512")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 512, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk576_dv512")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 576, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;

// Inkling on-the-fly relative-bias FlashAttention using Metal 4 tensor ops (matmul2d).
// Eliminates the O(n_kv * n_q * n_head) kq_b mask tensor completely and scales to 1M context.
// Supports F16 (direct device tensor), Q8_0, and Q4_0 (threadgroup staged to half) KV caches.
struct inkling_fa_tensor_args {
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t extent;
    uint32_t n_swa;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb21;
    uint32_t nb22;
    float    scale;
};

template <short C = 64>
inline void inkling_fa_stage_kv_q8_0(
        threadgroup half * dst,
        device const char * src,
        uint ic,
        uint kv_head,
        uint nb_head,
        uint nb_row,
        ushort tiitg) {
    for (uint bi = tiitg; bi < (uint)C * 4u; bi += 128u) {
        const uint row = bi >> 2;
        const uint blk = bi & 3u;
        device const block_q8_0 * b = (device const block_q8_0 *)(src + (uint64_t)(ic + row) * nb_row + (uint64_t)kv_head * nb_head) + blk;
        threadgroup half4 * out4 = (threadgroup half4 *)(dst + row * 128u + blk * 32u);
        half4x4 r0, r1;
        dequantize_q8_0(b, 0, r0);
        dequantize_q8_0(b, 1, r1);
        out4[0] = r0[0]; out4[1] = r0[1]; out4[2] = r0[2]; out4[3] = r0[3];
        out4[4] = r1[0]; out4[5] = r1[1]; out4[6] = r1[2]; out4[7] = r1[3];
    }
}

template <short C = 64>
inline void inkling_fa_stage_kv_q4_0(
        threadgroup half * dst,
        device const char * src,
        uint ic,
        uint kv_head,
        uint nb_head,
        uint nb_row,
        ushort tiitg) {
    for (uint bi = tiitg; bi < (uint)C * 4u; bi += 128u) {
        const uint row = bi >> 2;
        const uint blk = bi & 3u;
        device const block_q4_0 * b = (device const block_q4_0 *)(src + (uint64_t)(ic + row) * nb_row + (uint64_t)kv_head * nb_head) + blk;
        threadgroup half4 * out4 = (threadgroup half4 *)(dst + row * 128u + blk * 32u);
        half4x4 r0, r1;
        dequantize_q4_0(b, 0, r0);
        dequantize_q4_0(b, 1, r1);
        out4[0] = r0[0]; out4[1] = r0[1]; out4[2] = r0[2]; out4[3] = r0[3];
        out4[4] = r1[0]; out4[5] = r1[1]; out4[6] = r1[2]; out4[7] = r1[3];
    }
}

template <short Q, int K_KIND, int V_KIND>
kernel void kernel_inkling_fa_tensor_impl(
        constant inkling_fa_tensor_args & args [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const char  * k       [[buffer(2)]],
        device const char  * v       [[buffer(3)]],
        device const float * table   [[buffer(4)]],
        device const float * sinks   [[buffer(5)]],
        device const int   * q_pos   [[buffer(6)]],
        device const int   * k_pos   [[buffer(7)]],
        device       float * out     [[buffer(8)]],
        threadgroup  char  * shmem   [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short DK  = 128;
    constexpr short DV  = 128;
    constexpr short C   = 64;
    constexpr short NSG = 4;
    constexpr short NW  = 32;
    constexpr short NT  = NW * NSG;
    constexpr short NQ  = Q / NSG;
    constexpr short NC  = C / NW;

    const uint iq1     = tgpig.x * Q;
    const uint head    = tgpig.y;
    const uint kv_head = head / 3u;
    const ushort tiitg = sgitg * NW + tiisg;
    const uint extent    = args.extent;
    const int  n_swa     = (int)args.n_swa;

    // Layout must stay in sync with ggml_metal_inkling_fa_tensor_smem() in ggml-metal-expert.h.
    // Every region starts on a 16-byte boundary (Q in {16, 32}, DK=128, C=64).
    threadgroup half  * sq      = (threadgroup half  *) shmem;             // [Q, 128]
    threadgroup float * ss      = (threadgroup float *) (sq + Q * DK);     // [Q, 64]
    threadgroup half  * sp      = (threadgroup half  *) (ss + Q * C);      // [Q, 64]
    threadgroup float * sr      = (threadgroup float *) (sp + Q * C);      // [Q]
    threadgroup int   * sf      = (threadgroup int   *) (sr + Q);          // [4]
    threadgroup int   * kpos_tg = (threadgroup int   *) (sf + 4);          // [64]
    threadgroup ulong * kseq_tg = (threadgroup ulong *) (kpos_tg + C);     // [64]
    threadgroup half  * skv     = (threadgroup half  *) (kseq_tg + C);     // [64, 128] (KV_KIND != 0)

    for (uint i = tiitg; i < (uint)(Q * DK / 4); i += NT) {
        const uint j = i / (DK / 4);
        const uint d4 = i % (DK / 4);
        float4 q4 = 0.0f;
        if (iq1 + j < args.n_q) {
            q4 = ((device const float4 *)(q + ((uint64_t)(iq1 + j) * 24u + head) * DK))[d4];
        }
        ((threadgroup half4 *) sq)[i] = (half4)(q4 * args.scale);
    }

    int   qpos_r[NQ];
    ulong qbit_r[NQ];
    float sink_r[NQ];
    float far_bias_r[NQ];
    float M[NQ];
    float S[NQ];
    int   my_min_qp = INT_MAX;
    int   my_max_qp = INT_MIN;
    ulong my_seq_mask = 0ULL;
    int   my_seq_invalid = 0;

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const uint j = (uint)(jj * NSG + sgitg);
        const uint q_idx = iq1 + j;
        if (q_idx < args.n_q) {
            const int qp = q_pos[q_idx * 2u];
            const uint qseq = (uint)q_pos[q_idx * 2u + 1u];
            const ulong qbit = qseq < 64u ? (1ULL << qseq) : 0ULL;
            if (qseq >= 64u) { my_seq_invalid = 1; }
            qpos_r[jj] = qp;
            qbit_r[jj] = qbit;
            my_seq_mask |= qbit;
            my_min_qp = min(my_min_qp, qp);
            my_max_qp = max(my_max_qp, qp);
            const uint qh = q_idx * 24u + head;
            const float sk_val = sinks[qh];
            sink_r[jj] = sk_val;
            far_bias_r[jj] = -sk_val;
        } else {
            qpos_r[jj] = -1;
            qbit_r[jj] = 0ULL;
            sink_r[jj] = 0.0f;
            far_bias_r[jj] = 0.0f;
        }
        M[jj] = -FLT_MAX / 2.0f;
        S[jj] = 0.0f;
    }

    if (tiisg == 0) {
        kpos_tg[sgitg]      = my_min_qp;
        kpos_tg[sgitg + 4]  = my_max_qp;
        kpos_tg[sgitg + 8]  = my_seq_invalid;
        kseq_tg[sgitg]      = my_seq_mask;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int   tg_min_qp     = min(min(kpos_tg[0], kpos_tg[1]), min(kpos_tg[2], kpos_tg[3]));
    const int   tg_max_qp     = max(max(kpos_tg[4], kpos_tg[5]), max(kpos_tg[6], kpos_tg[7]));
    const bool  tg_any_inval  = (kpos_tg[8] | kpos_tg[9] | kpos_tg[10] | kpos_tg[11]) != 0;
    const ulong tg_seq_mask   = kseq_tg[0] | kseq_tg[1] | kseq_tg[2] | kseq_tg[3];
    const bool  tg_single_seq = !tg_any_inval && (tg_seq_mask != 0ULL) && ((tg_seq_mask & (tg_seq_mask - 1ULL)) == 0ULL);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tq = tensor(sq, dextents<int32_t, 2>(DK, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C,  Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C,  Q));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, DV, C, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    const int sk_f16 = (int)(args.nb12 / sizeof(half));
    const int sv_f16 = (int)(args.nb22 / sizeof(half));
    device const char * k_base = k + (uint64_t)kv_head * args.nb11;
    device const char * v_base = v + (uint64_t)kv_head * args.nb21;

    auto tv_dev0 = tensor((device half *)v_base, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv_f16}));
    auto tv_tg0  = tensor(skv, dextents<int32_t, 2>(DV, C));
    using tv_ref_t = conditional_t<V_KIND == 0, decltype(tv_dev0), decltype(tv_tg0)>;
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), tv_ref_t, float>();
    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    for (uint ic0 = 0, ic = 0; ic < args.n_kv; ++ic0, ic += C) {
        if (tiitg < 64u) {
            const uint kbase = (ic + tiitg) * 3u;
            const int kp = k_pos[kbase];
            const ulong kmask = ulong(uint(k_pos[kbase + 1u])) | (ulong(uint(k_pos[kbase + 2u])) << 32);
            kpos_tg[tiitg] = kp;
            kseq_tg[tiitg] = kmask;
            const bool seq_hit = (kmask & tg_seq_mask) != 0ULL;
            const bool vis = (kp >= 0) && seq_hit && (kp <= tg_max_qp) && (n_swa == 0 || kp > tg_min_qp - n_swa);
            const bool far_ok = (n_swa == 0) && tg_single_seq && (kp >= 0) && seq_hit && (kp <= tg_min_qp - (int)extent);
            const bool sg_any_vis = simd_any(vis);
            const bool sg_all_far = simd_all(far_ok);
            if (tiisg == 0) {
                sf[1 + sgitg] = (sg_any_vis ? 1 : 0) | (sg_all_far ? 2 : 0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int flags = (sf[1] | sf[2]);
        if ((flags & 1) == 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }
        const bool all_far = ((sf[1] & 2) != 0) && ((sf[2] & 2) != 0);

        if constexpr (K_KIND == 0) {
            auto tk = tensor((device half *)(k_base + (uint64_t)ic * args.nb12), dextents<int32_t, 2>(DK, C), array<int, 2>({1, sk_f16}));
            mm_qk.run(tq, tk, ts);
        } else if constexpr (K_KIND == 1) {
            inkling_fa_stage_kv_q8_0(skv, k, ic, kv_head, args.nb11, args.nb12, tiitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto tk = tensor(skv, dextents<int32_t, 2>(DK, C));
            mm_qk.run(tq, tk, ts);
        } else {
            inkling_fa_stage_kv_q4_0(skv, k, ic, kv_head, args.nb11, args.nb12, tiitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto tk = tensor(skv, dextents<int32_t, 2>(DK, C));
            mm_qk.run(tq, tk, ts);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if constexpr (V_KIND == 1) {
            inkling_fa_stage_kv_q8_0(skv, v, ic, kv_head, args.nb21, args.nb22, tiitg);
        } else if constexpr (V_KIND == 2) {
            inkling_fa_stage_kv_q4_0(skv, v, ic, kv_head, args.nb21, args.nb22, tiitg);
        }

        int sg_upd = 0;
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const uint j = (uint)(jj * NSG + sgitg);
            const uint q_idx = iq1 + j;
            const bool q_valid = q_idx < args.n_q;

            float s[NC];
            if (!q_valid) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) { s[ii] = -INFINITY; }
            } else if (all_far) {
                const float fb = far_bias_r[jj];
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] = ss[j * C + ii * NW + tiisg] + fb;
                }
            } else {
                const int qp = qpos_r[jj];
                const ulong qbit = qbit_r[jj];
                const float sk_val = sink_r[jj];
                const uint64_t tbl_off = (uint64_t)(q_idx * 24u + head) * extent;
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    const uint col = ii * NW + tiisg;
                    const int kp = kpos_tg[col];
                    const int d = qp - kp;
                    if (kp < 0 || (kseq_tg[col] & qbit) == 0ULL || d < 0 || (n_swa > 0 && d >= n_swa)) {
                        s[ii] = -INFINITY;
                    } else {
                        const float rel = (uint)d < extent ? table[tbl_off + (uint)d] : 0.0f;
                        s[ii] = ss[j * C + col] + (rel - sk_val);
                    }
                }
            }

            float m = M[jj];
            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                m = max(m, s[ii]);
            }
            m = simd_max(m);

            float ms = 1.0f;
            if (m > M[jj]) {
                ms = exp(M[jj] - m);
                M[jj] = m;
                sg_upd = 1;
            }

            float sum = 0.0f;
            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                const half p = (half)exp(s[ii] - M[jj]);
                sp[j * C + ii * NW + tiisg] = p;
                sum += (float)p;
            }
            S[jj] = S[jj] * ms + simd_sum(sum);

            if (tiisg == 0) {
                sr[j] = ms;
            }
        }
        if (tiisg == 0) {
            sf[sgitg] = sg_upd;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((sf[0] | sf[1] | sf[2] | sf[3]) != 0) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        if constexpr (V_KIND == 0) {
            auto tv = tensor((device half *)(v_base + (uint64_t)ic * args.nb22), dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv_f16}));
            mm_pv.run(tp, tv, co);
        } else {
            auto tv = tensor(skv, dextents<int32_t, 2>(DV, C));
            mm_pv.run(tp, tv, co);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const uint j = (uint)(jj * NSG + sgitg);
        const float m = max(M[jj], 0.0f);
        const float corr = exp(M[jj] - m);
        const float denom = S[jj] * corr + exp(-m);
        if (tiisg == 0) {
            sr[j] = denom == 0.0f ? 0.0f : corr / denom;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
    }

    if (iq1 < args.n_q) {
        device float * pdst = out + (uint64_t)iq1 * (24u * DV) + (uint64_t)head * DV;
        const int rows_valid = (int)min((uint)Q, args.n_q - iq1);
        auto td = tensor(pdst, dextents<int32_t, 2>(DV, rows_valid), array<int, 2>({1, 24 * DV}));
        co.store(td);
    }
}

template [[host_name("kernel_inkling_fa_tensor_f16")]]       kernel void kernel_inkling_fa_tensor_impl<32, 0, 0>(constant inkling_fa_tensor_args &, device const float *, device const char *, device const char *, device const float *, device const float *, device const int *, device const int *, device float *, threadgroup char *, uint2, ushort, ushort);
template [[host_name("kernel_inkling_fa_tensor_q8_0")]]      kernel void kernel_inkling_fa_tensor_impl<16, 1, 1>(constant inkling_fa_tensor_args &, device const float *, device const char *, device const char *, device const float *, device const float *, device const int *, device const int *, device float *, threadgroup char *, uint2, ushort, ushort);
template [[host_name("kernel_inkling_fa_tensor_q4_0")]]      kernel void kernel_inkling_fa_tensor_impl<16, 2, 2>(constant inkling_fa_tensor_args &, device const float *, device const char *, device const char *, device const float *, device const float *, device const int *, device const int *, device float *, threadgroup char *, uint2, ushort, ushort);
template [[host_name("kernel_inkling_fa_tensor_q8_0_q4_0")]] kernel void kernel_inkling_fa_tensor_impl<16, 1, 2>(constant inkling_fa_tensor_args &, device const float *, device const char *, device const char *, device const float *, device const float *, device const int *, device const int *, device float *, threadgroup char *, uint2, ushort, ushort);
template [[host_name("kernel_inkling_fa_tensor_f16_q8_0")]]  kernel void kernel_inkling_fa_tensor_impl<16, 0, 1>(constant inkling_fa_tensor_args &, device const float *, device const char *, device const char *, device const float *, device const float *, device const int *, device const int *, device float *, threadgroup char *, uint2, ushort, ushort);
template [[host_name("kernel_inkling_fa_tensor_f16_q4_0")]]  kernel void kernel_inkling_fa_tensor_impl<16, 0, 2>(constant inkling_fa_tensor_args &, device const float *, device const char *, device const char *, device const float *, device const float *, device const int *, device const int *, device float *, threadgroup char *, uint2, ushort, ushort);

// GQA-packed (3 Q heads per KV head), split-K Metal 4 Tensor decode attention for Inkling2.
// Packs BQ queries x 3 GQA heads into one Q-row tile (Q=16 for BQ=5, Q=24 for BQ=8, Q=32 for BQ=10)
// with C=64 (direct F16) or C=32 (high-occupancy 2-TG/core quantized KV staging) and optional
// Quest-style exact-upper-bound sparse page-skip indexing.
struct inkling_fa_decode_tensor_args {
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t extent;
    uint32_t n_swa;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nwg;
    uint32_t has_gate;
    uint32_t max_pages;
    float    scale;
};

struct inkling_fa_page_meta {
    ulong sig;
    ulong kcheck;
    ulong seq_mask;
    int   min_kp;
    int   max_kp;
    uint  epoch;
    uint  _pad[3];
};

struct inkling_fa_page_update_args {
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t n_pages;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t epoch;
};

struct inkling_fa_page_score_args {
    uint32_t n_q;
    uint32_t n_pages;
    uint32_t sink_tokens;
    uint32_t local_tokens;
    float    scale;
};

struct inkling_fa_page_topk_args {
    uint32_t n_pages;
    uint32_t k_pages;
};

inline uint inkling_fa_f2u(float f) {
    if (isnan(f)) {
        return 0x007FFFFFu; // map NaN to -INFINITY
    }
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <int K_KIND>
kernel void kernel_inkling_fa_page_update_impl(
        constant inkling_fa_page_update_args & args [[buffer(0)]],
        device const char           * k           [[buffer(1)]],
        device const int            * q_pos       [[buffer(2)]],
        device const int            * k_pos       [[buffer(3)]],
        device       half           * page_minmax [[buffer(4)]],
        device inkling_fa_page_meta * page_meta   [[buffer(5)]],
        threadgroup  char           * shmem       [[threadgroup(0)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint ip = tgpig;
    if (ip >= args.n_pages) {
        return;
    }
    const uint ic = ip * 64u;
    const ushort tiitg = sgitg * 32u + tiisg;

    threadgroup uchar kvalid_tg[64];
    threadgroup uint  s_hlo[2], s_hhi[2], s_mlo[2], s_mhi[2];
    threadgroup int   s_min[2], s_max[2];
    threadgroup uint  s_dirty[1];

    if (tiitg < 64u) {
        const uint kbase = (ic + tiitg) * 3u;
        const int  kp = k_pos[kbase];
        const ulong kmask = ulong(uint(k_pos[kbase + 1u])) | (ulong(uint(k_pos[kbase + 2u])) << 32);
        const bool valid = (kp >= 0) && (kmask != 0ULL);
        kvalid_tg[tiitg] = valid ? 1u : 0u;

        ulong h = (ulong(uint(kp)) ^ 0x9e3779b97f4a7c15ULL) * (ulong(tiitg) * 0xbf58476d1ce4e5b9ULL + 1ULL);
        h ^= (kmask * 0x94d049bb133111ebULL);
        h ^= (h >> 30);

        const uint hlo = simd_xor(uint(h));
        const uint hhi = simd_xor(uint(h >> 32));
        const uint mlo = simd_or(valid ? uint(kmask) : 0u);
        const uint mhi = simd_or(valid ? uint(kmask >> 32) : 0u);
        const int  mn  = simd_min(valid ? kp : INT_MAX);
        const int  mx  = simd_max(valid ? kp : -1);

        if (tiisg == 0) {
            s_hlo[sgitg] = hlo;
            s_hhi[sgitg] = hhi;
            s_mlo[sgitg] = mlo;
            s_mhi[sgitg] = mhi;
            s_min[sgitg] = mn;
            s_max[sgitg] = mx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiitg == 0) {
        const ulong sig      = ulong(s_hlo[0] ^ s_hlo[1]) | (ulong(s_hhi[0] ^ s_hhi[1]) << 32);
        const ulong seq_mask = ulong(s_mlo[0] | s_mlo[1]) | (ulong(s_mhi[0] | s_mhi[1]) << 32);
        const int   min_kp   = min(s_min[0], s_min[1]);
        const int   max_kp   = max(s_max[0], s_max[1]);

        int tg_min_qp = INT_MAX;
        for (uint qi = 0; qi < args.n_q; ++qi) {
            tg_min_qp = min(tg_min_qp, q_pos[qi * 2u]);
        }

        // Spot-check 4 rows x 2 kv_heads (64 B/page) to catch in-place buffer overwrites at identical k_pos.
        ulong kcheck = 0ULL;
        const uint sample_rows[4] = { 0u, 21u, 42u, 63u };
        const uint sample_hds[2]  = { 0u, 4u };
        for (uint ri = 0; ri < 4u; ++ri) {
            for (uint hi = 0; hi < 2u; ++hi) {
                device const uint * p = (device const uint *)(k + (uint64_t)(ic + sample_rows[ri]) * args.nb12 + (uint64_t)sample_hds[hi] * args.nb11);
                ulong w = ulong(p[0]) | (ulong(p[1]) << 32);
                kcheck = (kcheck * 0xbf58476d1ce4e5b9ULL) ^ (w + 0x9e3779b97f4a7c15ULL);
            }
        }

        const inkling_fa_page_meta prev = page_meta[ip];
        const bool frontier_overlap = (max_kp >= 0) && (max_kp >= tg_min_qp - 64);
        const bool dirty = (prev.epoch != args.epoch) ||
                           (prev.sig != sig) ||
                           (prev.kcheck != kcheck) ||
                           frontier_overlap;
        s_dirty[0] = dirty ? 1u : 0u;
        if (dirty) {
            inkling_fa_page_meta cur;
            cur.sig      = sig;
            cur.kcheck   = kcheck;
            cur.seq_mask = seq_mask;
            cur.min_kp   = min_kp;
            cur.max_kp   = max_kp;
            cur.epoch    = args.epoch;
            cur._pad[0]  = 0u;
            cur._pad[1]  = 0u;
            cur._pad[2]  = 0u;
            page_meta[ip] = cur;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (s_dirty[0] == 0u) {
        return;
    }

    const int max_kp = max(s_max[0], s_max[1]);
    threadgroup half * skv = (threadgroup half *) shmem;
    const uint row_stride_h = args.nb12 / sizeof(half);

    for (uint kv_head = 0; kv_head < 8u; ++kv_head) {
        const uint64_t dst_base = ((uint64_t)ip * 8u + kv_head) * 2u * 128u;
        if (max_kp < 0) {
            page_minmax[dst_base + tiitg]        = (half)0.0h;
            page_minmax[dst_base + 128u + tiitg] = (half)0.0h;
            continue;
        }
        if constexpr (K_KIND == 0) {
            device const half * kh = (device const half *)(k + (uint64_t)ic * args.nb12 + (uint64_t)kv_head * args.nb11);
            half vmin = (half)65504.0h;
            half vmax = (half)-65504.0h;
            for (uint r = 0; r < 64u; ++r) {
                if (kvalid_tg[r]) {
                    const half val = kh[r * row_stride_h + tiitg];
                    vmin = min(vmin, val);
                    vmax = max(vmax, val);
                }
            }
            page_minmax[dst_base + tiitg]        = vmin;
            page_minmax[dst_base + 128u + tiitg] = vmax;
        } else {
            if constexpr (K_KIND == 1) {
                inkling_fa_stage_kv_q8_0<64>(skv, k, ic, kv_head, args.nb11, args.nb12, tiitg);
            } else {
                inkling_fa_stage_kv_q4_0<64>(skv, k, ic, kv_head, args.nb11, args.nb12, tiitg);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            half vmin = (half)65504.0h;
            half vmax = (half)-65504.0h;
            for (uint r = 0; r < 64u; ++r) {
                if (kvalid_tg[r]) {
                    const half val = skv[r * 128u + tiitg];
                    vmin = min(vmin, val);
                    vmax = max(vmax, val);
                }
            }
            page_minmax[dst_base + tiitg]        = vmin;
            page_minmax[dst_base + 128u + tiitg] = vmax;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

template [[host_name("kernel_inkling_fa_page_update_f16")]]  kernel void kernel_inkling_fa_page_update_impl<0>(constant inkling_fa_page_update_args &, device const char *, device const int *, device const int *, device half *, device inkling_fa_page_meta *, threadgroup char *, uint, ushort, ushort);
template [[host_name("kernel_inkling_fa_page_update_q8_0")]] kernel void kernel_inkling_fa_page_update_impl<1>(constant inkling_fa_page_update_args &, device const char *, device const int *, device const int *, device half *, device inkling_fa_page_meta *, threadgroup char *, uint, ushort, ushort);
template [[host_name("kernel_inkling_fa_page_update_q4_0")]] kernel void kernel_inkling_fa_page_update_impl<2>(constant inkling_fa_page_update_args &, device const char *, device const int *, device const int *, device half *, device inkling_fa_page_meta *, threadgroup char *, uint, ushort, ushort);

kernel void kernel_inkling_fa_page_score(
        constant inkling_fa_page_score_args & args [[buffer(0)]],
        device const float                * q           [[buffer(1)]],
        device const float                * sinks       [[buffer(2)]],
        device const int                  * q_pos       [[buffer(3)]],
        device const half                 * page_minmax [[buffer(4)]],
        device const inkling_fa_page_meta * page_meta   [[buffer(5)]],
        device       float                * page_scores [[buffer(6)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint kv_head   = tgpig.y;
    const ushort tiitg   = sgitg * 32u + tiisg;
    const uint n_act_q   = args.n_q * 3u; // <= 45 (since n_q < 16)

    threadgroup half  sq[48 * 128];
    threadgroup float ssink[48];

    for (uint i = tiitg; i < n_act_q * 32u; i += 128u) {
        const uint j  = i >> 5;
        const uint d4 = i & 31u;
        const uint qi = j / 3u;
        const uint g  = j % 3u;
        const uint head = kv_head * 3u + g;
        const float4 q4 = ((device const float4 *)(q + ((uint64_t)qi * 24u + head) * 128u))[d4];
        ((threadgroup half4 *) sq)[i] = (half4)(q4 * args.scale);
    }
    if (tiitg < n_act_q) {
        const uint qi = tiitg / 3u;
        const uint g  = tiitg % 3u;
        ssink[tiitg] = sinks[qi * 24u + kv_head * 3u + g];
    }

    int   tg_min_qp   = INT_MAX;
    int   tg_max_qp   = INT_MIN;
    ulong tg_seq_mask = 0ULL;
    for (uint qi = 0; qi < args.n_q; ++qi) {
        const int  qp   = q_pos[qi * 2u];
        const uint qseq = (uint)q_pos[qi * 2u + 1u];
        tg_min_qp = min(tg_min_qp, qp);
        tg_max_qp = max(tg_max_qp, qp);
        if (qseq < 64u) {
            tg_seq_mask |= (1ULL << qseq);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pi = 0; pi < 4u; ++pi) {
        const uint ip = tgpig.x * 16u + sgitg * 4u + pi;
        if (ip >= args.n_pages) {
            break;
        }
        const inkling_fa_page_meta meta = page_meta[ip];
        if (meta.max_kp < 0 || meta.min_kp > tg_max_qp || (meta.seq_mask & tg_seq_mask) == 0ULL) {
            if (tiisg == 0) {
                page_scores[(uint64_t)kv_head * args.n_pages + ip] = -INFINITY;
            }
            continue;
        }
        if (meta.min_kp < (int)args.sink_tokens || meta.max_kp >= tg_min_qp - (int)args.local_tokens) {
            if (tiisg == 0) {
                page_scores[(uint64_t)kv_head * args.n_pages + ip] = INFINITY;
            }
            continue;
        }
        const uint64_t mm_base = ((uint64_t)ip * 8u + kv_head) * 2u * 128u;
        const float4 kmin4 = float4(((device const half4 *)(page_minmax + mm_base))[tiisg]);
        const float4 kmax4 = float4(((device const half4 *)(page_minmax + mm_base + 128u))[tiisg]);
        float best_ub = -INFINITY;
        for (uint j = 0; j < n_act_q; ++j) {
            const float4 q4 = float4(((threadgroup const half4 *)(sq + j * 128u))[tiisg]);
            const float4 hi = max(q4 * kmin4, q4 * kmax4);
            const float ub = simd_sum(hi.x + hi.y + hi.z + hi.w) - ssink[j];
            best_ub = max(best_ub, ub);
        }
        if (tiisg == 0) {
            page_scores[(uint64_t)kv_head * args.n_pages + ip] = best_ub;
        }
    }
}

kernel void kernel_inkling_fa_page_topk(
        constant inkling_fa_page_topk_args & args [[buffer(0)]],
        device const float * page_scores  [[buffer(1)]],
        device       uint  * active_pages [[buffer(2)]],
        device       uint  * active_count [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    const uint kv_head = tgpig;
    const uint n_pages = args.n_pages;
    const uint k_pages = min(args.k_pages, n_pages);
    device const float * scores = page_scores + (uint64_t)kv_head * n_pages;
    device       uint  * out_p  = active_pages + (uint64_t)kv_head * n_pages;

    threadgroup atomic_uint hist[256];
    threadgroup atomic_uint cnt_tg[2];
    threadgroup uint        sp[4];

    atomic_store_explicit(&hist[tiitg], 0u, memory_order_relaxed);
    if (tiitg < 2u) {
        atomic_store_explicit(&cnt_tg[tiitg], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 1: count valid pages & top-byte histogram (bits 31..24)
    uint local_valid = 0u;
    for (uint ip = tiitg; ip < n_pages; ip += 256u) {
        const uint u = inkling_fa_f2u(scores[ip]);
        if (u > 0x007FFFFFu) {
            local_valid += 1u;
            atomic_fetch_add_explicit(&hist[u >> 24], 1u, memory_order_relaxed);
        }
    }
    if (local_valid > 0u) {
        atomic_fetch_add_explicit(&cnt_tg[0], local_valid, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_valid = atomic_load_explicit(&cnt_tg[0], memory_order_relaxed);
    if (n_valid <= k_pages) {
        for (uint ip = tiitg; ip < n_pages; ip += 256u) {
            const uint u = inkling_fa_f2u(scores[ip]);
            if (u > 0x007FFFFFu) {
                const uint idx = atomic_fetch_add_explicit(&cnt_tg[1], 1u, memory_order_relaxed);
                if (idx < k_pages) {
                    out_p[idx] = ip;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg == 0) {
            active_count[kv_head] = min(atomic_load_explicit(&cnt_tg[1], memory_order_relaxed), k_pages);
        }
        return;
    }

    if (tiitg == 0) {
        uint cum = 0u;
        uint b0  = 0u;
        for (int b = 255; b >= 0; --b) {
            const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
            if (cum + c >= k_pages) {
                b0 = (uint)b;
                break;
            }
            cum += c;
        }
        sp[0] = b0;
        sp[1] = k_pages - cum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: refine second byte (bits 23..16) inside bin b0
    const uint b0   = sp[0];
    const uint rem0 = sp[1];
    atomic_store_explicit(&hist[tiitg], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ip = tiitg; ip < n_pages; ip += 256u) {
        const uint u = inkling_fa_f2u(scores[ip]);
        if (u > 0x007FFFFFu && (u >> 24) == b0) {
            atomic_fetch_add_explicit(&hist[(u >> 16) & 0xFFu], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiitg == 0) {
        uint cum = 0u;
        uint b1  = 0u;
        for (int b = 255; b >= 0; --b) {
            const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
            if (cum + c >= rem0) {
                b1 = (uint)b;
                break;
            }
            cum += c;
        }
        sp[2] = (b0 << 8) | b1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint thresh16 = sp[2];

    // Collect pass 1: strictly above threshold (always includes all forced +INFINITY sink & local pages)
    for (uint ip = tiitg; ip < n_pages; ip += 256u) {
        const uint u = inkling_fa_f2u(scores[ip]);
        if (u > 0x007FFFFFu && (u >> 16) > thresh16) {
            const uint idx = atomic_fetch_add_explicit(&cnt_tg[1], 1u, memory_order_relaxed);
            if (idx < k_pages) {
                out_p[idx] = ip;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Collect pass 2: fill remaining budget from threshold bin
    for (uint ip = tiitg; ip < n_pages; ip += 256u) {
        const uint u = inkling_fa_f2u(scores[ip]);
        if (u > 0x007FFFFFu && (u >> 16) == thresh16) {
            const uint idx = atomic_fetch_add_explicit(&cnt_tg[1], 1u, memory_order_relaxed);
            if (idx < k_pages) {
                out_p[idx] = ip;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiitg == 0) {
        active_count[kv_head] = min(atomic_load_explicit(&cnt_tg[1], memory_order_relaxed), k_pages);
    }
}

template <short Q, short C, int K_KIND, int V_KIND, bool HAS_SPARSE = false>
kernel void kernel_inkling_fa_decode_tensor_impl(
        constant inkling_fa_decode_tensor_args & args [[buffer(0)]],
        device const float  * q            [[buffer(1)]],
        device const char   * k            [[buffer(2)]],
        device const char   * v            [[buffer(3)]],
        device const float  * table        [[buffer(4)]],
        device const float  * sinks        [[buffer(5)]],
        device const int    * q_pos        [[buffer(6)]],
        device const int    * k_pos        [[buffer(7)]],
        device       float  * dst          [[buffer(8)]],
        device       float  * tmp          [[buffer(9)]],
        device const float4 * gate         [[buffer(10)]],
        device const uint   * active_pages [[buffer(11)]],
        device const uint   * active_count [[buffer(12)]],
        threadgroup  char   * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short DK  = 128;
    constexpr short DV  = 128;
    constexpr short NSG = 4;
    constexpr short NW  = 32;
    constexpr short NT  = NW * NSG;
    constexpr short NQ  = Q / NSG;
    constexpr short NC  = C / NW;
    constexpr uint  BQ  = (uint)(Q / 3); // 5 for Q=16 (15 rows), 8 for Q=24 (24 rows), 10 for Q=32 (30 rows)
    constexpr uint  SPLIT_STRIDE = 132u;
    constexpr float LOG2E = 1.4426950408889634f;

    const uint q0      = tgpig.x * BQ;
    const uint kv_head = tgpig.y;
    const uint iwg     = tgpig.z;
    const uint nwg     = args.nwg;
    const ushort tiitg = sgitg * NW + tiisg;
    const uint extent  = args.extent;
    const int  n_swa   = (int)args.n_swa;
    const float q_scale = args.scale * LOG2E;

    // Identical layout to ggml_metal_inkling_fa_decode_tensor_smem(Q, C, has_skv).
    threadgroup half  * sq      = (threadgroup half  *) shmem;             // [Q, 128]
    threadgroup float * ss      = (threadgroup float *) (sq + Q * DK);     // [Q, C]
    threadgroup half  * sp      = (threadgroup half  *) (ss + Q * C);      // [Q, C]
    threadgroup float * sr      = (threadgroup float *) (sp + Q * C);      // [Q]
    threadgroup int   * sf      = (threadgroup int   *) (sr + Q);          // [4]
    threadgroup int   * kpos_tg = (threadgroup int   *) (sf + 4);          // [C]
    threadgroup ulong * kseq_tg = (threadgroup ulong *) (kpos_tg + C);     // [C]
    threadgroup half  * skv     = (threadgroup half  *) (kseq_tg + C);     // [C, 128] (KV_KIND != 0)

    for (uint i = tiitg; i < (uint)(Q * DK / 4); i += NT) {
        const uint j  = i / (DK / 4);
        const uint d4 = i % (DK / 4);
        const uint qi = j / 3u;
        const uint g  = j % 3u;
        const uint q_idx = q0 + qi;
        float4 q4 = 0.0f;
        if (qi < BQ && q_idx < args.n_q) {
            const uint head = kv_head * 3u + g;
            q4 = ((device const float4 *)(q + ((uint64_t)q_idx * 24u + head) * DK))[d4];
        }
        ((threadgroup half4 *) sq)[i] = (half4)(q4 * q_scale);
    }

    int   qpos_r[NQ];
    ulong qbit_r[NQ];
    float sink_r[NQ];
    float far_bias_r[NQ];
    float M[NQ];
    float S[NQ];
    int   my_min_qp = INT_MAX;
    int   my_max_qp = INT_MIN;
    ulong my_seq_mask = 0ULL;
    int   my_seq_invalid = 0;

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const uint j  = (uint)(jj * NSG + sgitg);
        const uint qi = j / 3u;
        const uint g  = j % 3u;
        const uint q_idx = q0 + qi;
        if (qi < BQ && q_idx < args.n_q) {
            const int qp = q_pos[q_idx * 2u];
            const uint qseq = (uint)q_pos[q_idx * 2u + 1u];
            const ulong qbit = qseq < 64u ? (1ULL << qseq) : 0ULL;
            if (qseq >= 64u) { my_seq_invalid = 1; }
            qpos_r[jj] = qp;
            qbit_r[jj] = qbit;
            my_seq_mask |= qbit;
            my_min_qp = min(my_min_qp, qp);
            my_max_qp = max(my_max_qp, qp);
            const uint head = kv_head * 3u + g;
            const uint qh = q_idx * 24u + head;
            const float sk_val = sinks[qh] * LOG2E;
            sink_r[jj] = sk_val;
            far_bias_r[jj] = -sk_val;
        } else {
            qpos_r[jj] = -1;
            qbit_r[jj] = 0ULL;
            sink_r[jj] = 0.0f;
            far_bias_r[jj] = 0.0f;
        }
        M[jj] = -FLT_MAX / 2.0f;
        S[jj] = 0.0f;
    }

    if (tiisg == 0) {
        kpos_tg[sgitg]     = my_min_qp;
        kpos_tg[sgitg + 4] = my_max_qp;
        kpos_tg[sgitg + 8] = my_seq_invalid;
        kseq_tg[sgitg]     = my_seq_mask;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int   tg_min_qp     = min(min(kpos_tg[0], kpos_tg[1]), min(kpos_tg[2], kpos_tg[3]));
    const int   tg_max_qp     = max(max(kpos_tg[4], kpos_tg[5]), max(kpos_tg[6], kpos_tg[7]));
    const bool  tg_any_inval  = (kpos_tg[8] | kpos_tg[9] | kpos_tg[10] | kpos_tg[11]) != 0;
    const ulong tg_seq_mask   = kseq_tg[0] | kseq_tg[1] | kseq_tg[2] | kseq_tg[3];
    const bool  tg_single_seq = !tg_any_inval && (tg_seq_mask != 0ULL) && ((tg_seq_mask & (tg_seq_mask - 1ULL)) == 0ULL);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tq = tensor(sq, dextents<int32_t, 2>(DK, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C,  Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C,  Q));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, DV, C, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    const int sk_f16 = (int)(args.nb12 / sizeof(half));
    const int sv_f16 = (int)(args.nb22 / sizeof(half));
    device const char * k_base = k + (uint64_t)kv_head * args.nb11;
    device const char * v_base = v + (uint64_t)kv_head * args.nb21;

    auto tv_dev0 = tensor((device half *)v_base, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv_f16}));
    auto tv_tg0  = tensor(skv, dextents<int32_t, 2>(DV, C));
    using tv_ref_t = conditional_t<V_KIND == 0, decltype(tv_dev0), decltype(tv_tg0)>;
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), tv_ref_t, float>();
    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    // Contiguous split-K over dense or Quest-selected 64-key pages (args.n_kv is multiple of 64).
    const uint n_tiles = HAS_SPARSE
        ? (active_count[kv_head] * (64u / (uint)C))
        : (args.n_kv / (uint)C);
    const uint tiles_per_wg = (n_tiles + nwg - 1u) / nwg;
    const uint t_beg        = iwg * tiles_per_wg;
    const uint t_end        = min(n_tiles, t_beg + tiles_per_wg);

    for (uint it = t_beg; it < t_end; ++it) {
        uint ic;
        if constexpr (HAS_SPARSE) {
            if constexpr (C == 64) {
                ic = active_pages[(uint64_t)kv_head * args.max_pages + it] * 64u;
            } else {
                ic = active_pages[(uint64_t)kv_head * args.max_pages + (it >> 1)] * 64u + (it & 1u) * 32u;
            }
        } else {
            ic = it * (uint)C;
        }

        if (tiitg < (uint)C) {
            const uint kbase = (ic + tiitg) * 3u;
            const int kp = k_pos[kbase];
            const ulong kmask = ulong(uint(k_pos[kbase + 1u])) | (ulong(uint(k_pos[kbase + 2u])) << 32);
            kpos_tg[tiitg] = kp;
            kseq_tg[tiitg] = kmask;
            const bool seq_hit = (kmask & tg_seq_mask) != 0ULL;
            const bool vis = (kp >= 0) && seq_hit && (kp <= tg_max_qp) && (n_swa == 0 || kp > tg_min_qp - n_swa);
            const bool far_ok = (n_swa == 0) && tg_single_seq && (kp >= 0) && seq_hit && (kp <= tg_min_qp - (int)extent);
            const bool sg_any_vis = simd_any(vis);
            const bool sg_all_far = simd_all(far_ok);
            if (tiisg == 0) {
                sf[1 + sgitg] = (sg_any_vis ? 1 : 0) | (sg_all_far ? 2 : 0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int flags = (C == 32) ? sf[1] : (sf[1] | sf[2]);
        if ((flags & 1) == 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }
        const bool all_far = (C == 32) ? ((sf[1] & 2) != 0) : (((sf[1] & 2) != 0) && ((sf[2] & 2) != 0));

        if constexpr (K_KIND == 0) {
            auto tk = tensor((device half *)(k_base + (uint64_t)ic * args.nb12), dextents<int32_t, 2>(DK, C), array<int, 2>({1, sk_f16}));
            mm_qk.run(tq, tk, ts);
        } else if constexpr (K_KIND == 1) {
            inkling_fa_stage_kv_q8_0<C>(skv, k, ic, kv_head, args.nb11, args.nb12, tiitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto tk = tensor(skv, dextents<int32_t, 2>(DK, C));
            mm_qk.run(tq, tk, ts);
        } else {
            inkling_fa_stage_kv_q4_0<C>(skv, k, ic, kv_head, args.nb11, args.nb12, tiitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto tk = tensor(skv, dextents<int32_t, 2>(DK, C));
            mm_qk.run(tq, tk, ts);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if constexpr (V_KIND == 1) {
            inkling_fa_stage_kv_q8_0<C>(skv, v, ic, kv_head, args.nb21, args.nb22, tiitg);
        } else if constexpr (V_KIND == 2) {
            inkling_fa_stage_kv_q4_0<C>(skv, v, ic, kv_head, args.nb21, args.nb22, tiitg);
        }

        int sg_upd = 0;
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const uint j  = (uint)(jj * NSG + sgitg);
            const uint qi = j / 3u;
            const uint g  = j % 3u;
            const uint q_idx = q0 + qi;
            const bool q_valid = (qi < BQ) && (q_idx < args.n_q);

            float s[NC];
            if (!q_valid) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) { s[ii] = -INFINITY; }
            } else if (all_far) {
                const float fb = far_bias_r[jj];
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] = ss[j * C + ii * NW + tiisg] + fb;
                }
            } else {
                const int qp = qpos_r[jj];
                const ulong qbit = qbit_r[jj];
                const float sk_val = sink_r[jj];
                const uint head = kv_head * 3u + g;
                const uint64_t tbl_off = (uint64_t)(q_idx * 24u + head) * extent;
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    const uint col = ii * NW + tiisg;
                    const int kp = kpos_tg[col];
                    const int d = qp - kp;
                    if (kp < 0 || (kseq_tg[col] & qbit) == 0ULL || d < 0 || (n_swa > 0 && d >= n_swa)) {
                        s[ii] = -INFINITY;
                    } else {
                        const float rel = (uint)d < extent ? table[tbl_off + (uint)d] * LOG2E : 0.0f;
                        s[ii] = ss[j * C + col] + (rel - sk_val);
                    }
                }
            }

            float m = M[jj];
            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                m = max(m, s[ii]);
            }
            m = simd_max(m);

            float ms = 1.0f;
            if (m > M[jj]) {
                ms = exp2(M[jj] - m);
                M[jj] = m;
                sg_upd = 1;
            }

            float sum = 0.0f;
            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                const half p = (half)exp2(s[ii] - M[jj]);
                sp[j * C + ii * NW + tiisg] = p;
                sum += (float)p;
            }
            S[jj] = S[jj] * ms + simd_sum(sum);

            if (tiisg == 0) {
                sr[j] = ms;
            }
        }
        if (tiisg == 0) {
            sf[sgitg] = sg_upd;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((sf[0] | sf[1] | sf[2] | sf[3]) != 0) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        if constexpr (V_KIND == 0) {
            auto tv = tensor((device half *)(v_base + (uint64_t)ic * args.nb22), dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv_f16}));
            mm_pv.run(tp, tv, co);
        } else {
            auto tv = tensor(skv, dextents<int32_t, 2>(DV, C));
            mm_pv.run(tp, tv, co);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reuse shmem to unpack cooperative tensor [DV, Q] + M/S into coalesced float4 writes.
    // ggml_metal_inkling_fa_decode_tensor_smem() guarantees shmem >= Q*(DV+2)*4 bytes.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * so     = (threadgroup float *) shmem;
    threadgroup float * sm_out = so + Q * DV;
    threadgroup float * ss_out = sm_out + Q;

    auto to = tensor(so, dextents<int32_t, 2>(DV, Q));
    co.store(to);
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        if (tiisg == 0) {
            const uint j = (uint)(jj * NSG + sgitg);
            sm_out[j] = M[jj];
            ss_out[j] = S[jj];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = sgitg; j < BQ * 3u; j += (uint)NSG) {
        const uint qi = j / 3u;
        const uint g  = j % 3u;
        const uint q_idx = q0 + qi;
        if (q_idx >= args.n_q) { continue; }
        const uint head = kv_head * 3u + g;
        const float4 o4 = ((threadgroup const float4 *)(so + j * DV))[tiisg];
        const float m_all = sm_out[j];
        const float s_all = ss_out[j];
        if (nwg == 1u) {
            const float mf = max(m_all, 0.0f);
            const float corr = exp2(m_all - mf);
            const float denom = s_all * corr + exp2(-mf);
            const float inv = denom == 0.0f ? 0.0f : corr / denom;
            const uint64_t out_off4 = ((uint64_t)q_idx * 24u + head) * 32u + tiisg;
            float4 out4 = o4 * inv;
            if (args.has_gate) {
                const float4 g4 = gate[out_off4];
                out4 *= 1.0f / (1.0f + exp(-g4));
            }
            ((device float4 *)dst)[out_off4] = out4;
        } else {
            device float * out_part = tmp + (((uint64_t)q_idx * 24u + head) * nwg + iwg) * SPLIT_STRIDE;
            ((device float4 *)out_part)[tiisg] = o4;
            if (tiisg == 0) {
                out_part[128] = m_all;
                out_part[129] = s_all;
            }
        }
    }
}

#define INKLING_FA_DEC_TENSOR_SIG (constant inkling_fa_decode_tensor_args &, device const float *, device const char *, device const char *, device const float *, device const float *, device const int *, device const int *, device float *, device float *, device const float4 *, device const uint *, device const uint *, threadgroup char *, uint3, ushort, ushort)

// Dense F16/F16 (C=64: Q=16, Q=24 exact 8x3, Q=32; plus C=32 Q=16 for occupancy A/B)
template [[host_name("kernel_inkling_fa_decode_tensor_f16_q16")]]               kernel void kernel_inkling_fa_decode_tensor_impl<16, 64, 0, 0, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_f16_q24")]]               kernel void kernel_inkling_fa_decode_tensor_impl<24, 64, 0, 0, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_f16_q32")]]               kernel void kernel_inkling_fa_decode_tensor_impl<32, 64, 0, 0, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_f16_c32_q16")]]           kernel void kernel_inkling_fa_decode_tensor_impl<16, 32, 0, 0, false> INKLING_FA_DEC_TENSOR_SIG;

// Dense Quantized KV C=32 (15.8 KB smem -> 2 TGs/core for Q=16; single-pass Q=24 for n_q=6..8)
template [[host_name("kernel_inkling_fa_decode_tensor_q8_0_c32_q16")]]          kernel void kernel_inkling_fa_decode_tensor_impl<16, 32, 1, 1, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_q4_0_c32_q16")]]          kernel void kernel_inkling_fa_decode_tensor_impl<16, 32, 2, 2, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_q8_0_q4_0_c32_q16")]]     kernel void kernel_inkling_fa_decode_tensor_impl<16, 32, 1, 2, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_f16_q8_0_c32_q16")]]      kernel void kernel_inkling_fa_decode_tensor_impl<16, 32, 0, 1, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_f16_q4_0_c32_q16")]]      kernel void kernel_inkling_fa_decode_tensor_impl<16, 32, 0, 2, false> INKLING_FA_DEC_TENSOR_SIG;

template [[host_name("kernel_inkling_fa_decode_tensor_q8_0_c32_q24")]]          kernel void kernel_inkling_fa_decode_tensor_impl<24, 32, 1, 1, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_q4_0_c32_q24")]]          kernel void kernel_inkling_fa_decode_tensor_impl<24, 32, 2, 2, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_q8_0_q4_0_c32_q24")]]     kernel void kernel_inkling_fa_decode_tensor_impl<24, 32, 1, 2, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_f16_q8_0_c32_q24")]]      kernel void kernel_inkling_fa_decode_tensor_impl<24, 32, 0, 1, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_f16_q4_0_c32_q24")]]      kernel void kernel_inkling_fa_decode_tensor_impl<24, 32, 0, 2, false> INKLING_FA_DEC_TENSOR_SIG;

// Legacy C=64 quantized KV (kept for C=32 vs C=64 A/B comparison)
template [[host_name("kernel_inkling_fa_decode_tensor_q8_0_q16")]]              kernel void kernel_inkling_fa_decode_tensor_impl<16, 64, 1, 1, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_q4_0_q16")]]              kernel void kernel_inkling_fa_decode_tensor_impl<16, 64, 2, 2, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_q8_0_q4_0_q16")]]         kernel void kernel_inkling_fa_decode_tensor_impl<16, 64, 1, 2, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_f16_q8_0_q16")]]          kernel void kernel_inkling_fa_decode_tensor_impl<16, 64, 0, 1, false> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_f16_q4_0_q16")]]          kernel void kernel_inkling_fa_decode_tensor_impl<16, 64, 0, 2, false> INKLING_FA_DEC_TENSOR_SIG;

// Quest-style Exact-Upper-Bound Sparse Page-Skip Global Decode (F16 C=64 and Quantized C=32)
template [[host_name("kernel_inkling_fa_decode_tensor_sparse_f16_q16")]]        kernel void kernel_inkling_fa_decode_tensor_impl<16, 64, 0, 0, true>  INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_sparse_f16_q24")]]        kernel void kernel_inkling_fa_decode_tensor_impl<24, 64, 0, 0, true>  INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_sparse_f16_q32")]]        kernel void kernel_inkling_fa_decode_tensor_impl<32, 64, 0, 0, true>  INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_sparse_q8_0_c32_q16")]]   kernel void kernel_inkling_fa_decode_tensor_impl<16, 32, 1, 1, true>  INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_sparse_q4_0_c32_q16")]]   kernel void kernel_inkling_fa_decode_tensor_impl<16, 32, 2, 2, true>  INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_sparse_q8_0_q4_0_c32_q16")]] kernel void kernel_inkling_fa_decode_tensor_impl<16, 32, 1, 2, true> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_sparse_f16_q8_0_c32_q16")]]  kernel void kernel_inkling_fa_decode_tensor_impl<16, 32, 0, 1, true> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_sparse_f16_q4_0_c32_q16")]]  kernel void kernel_inkling_fa_decode_tensor_impl<16, 32, 0, 2, true> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_sparse_q8_0_c32_q24")]]   kernel void kernel_inkling_fa_decode_tensor_impl<24, 32, 1, 1, true>  INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_sparse_q4_0_c32_q24")]]   kernel void kernel_inkling_fa_decode_tensor_impl<24, 32, 2, 2, true>  INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_sparse_q8_0_q4_0_c32_q24")]] kernel void kernel_inkling_fa_decode_tensor_impl<24, 32, 1, 2, true> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_sparse_f16_q8_0_c32_q24")]]  kernel void kernel_inkling_fa_decode_tensor_impl<24, 32, 0, 1, true> INKLING_FA_DEC_TENSOR_SIG;
template [[host_name("kernel_inkling_fa_decode_tensor_sparse_f16_q4_0_c32_q24")]]  kernel void kernel_inkling_fa_decode_tensor_impl<24, 32, 0, 2, true> INKLING_FA_DEC_TENSOR_SIG;

#undef INKLING_FA_DEC_TENSOR_SIG

#endif // GGML_METAL_HAS_TENSOR
