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

inline void inkling_fa_stage_kv_q8_0(
        threadgroup half * dst,
        device const char * src,
        uint ic,
        uint kv_head,
        uint nb_head,
        uint nb_row,
        ushort tiitg) {
    for (uint bi = tiitg; bi < 64u * 4u; bi += 128u) {
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

inline void inkling_fa_stage_kv_q4_0(
        threadgroup half * dst,
        device const char * src,
        uint ic,
        uint kv_head,
        uint nb_head,
        uint nb_row,
        ushort tiitg) {
    for (uint bi = tiitg; bi < 64u * 4u; bi += 128u) {
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

template <short Q, int KV_KIND>
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
    const uint extent_p1 = extent + 1u;
    const int  n_swa     = (int)args.n_swa;

    threadgroup half  * sq      = (threadgroup half  *) shmem;             // [Q, 128]
    threadgroup float * ss      = (threadgroup float *) (sq + Q * DK);     // [Q, 64]
    threadgroup half  * sp      = (threadgroup half  *) (ss + Q * C);      // [Q, 64]
    threadgroup float * sr      = (threadgroup float *) (sp + Q * C);      // [Q]
    threadgroup int   * sf      = (threadgroup int   *) (sr + Q);          // [4]
    threadgroup int   * kpos_tg = (threadgroup int   *) (sf + 4);          // [64]
    threadgroup half  * skv     = (threadgroup half  *) (kpos_tg + C);     // [64, 128] (KV_KIND != 0)

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
    float sink_r[NQ];
    float far_bias_r[NQ];
    float M[NQ];
    float S[NQ];
    int my_min_qp = INT_MAX;
    int my_max_qp = INT_MIN;

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const uint j = (uint)(jj * NSG + sgitg);
        const uint q_idx = iq1 + j;
        if (q_idx < args.n_q) {
            const int qp = q_pos[q_idx];
            qpos_r[jj] = qp;
            my_min_qp = min(my_min_qp, qp);
            my_max_qp = max(my_max_qp, qp);
            const uint qh = q_idx * 24u + head;
            const float sk_val = sinks[qh];
            sink_r[jj] = sk_val;
            far_bias_r[jj] = table[(uint64_t)qh * extent_p1 + extent] - sk_val;
        } else {
            qpos_r[jj] = -1;
            sink_r[jj] = 0.0f;
            far_bias_r[jj] = 0.0f;
        }
        M[jj] = -FLT_MAX / 2.0f;
        S[jj] = 0.0f;
    }

    if (tiisg == 0) {
        kpos_tg[sgitg]     = my_min_qp;
        kpos_tg[sgitg + 4] = my_max_qp;
    }
    if (tiitg == 0) {
        sf[0] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int tg_min_qp = min(min(kpos_tg[0], kpos_tg[1]), min(kpos_tg[2], kpos_tg[3]));
    const int tg_max_qp = max(max(kpos_tg[4], kpos_tg[5]), max(kpos_tg[6], kpos_tg[7]));

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
    using tv_ref_t = conditional_t<KV_KIND == 0, decltype(tv_dev0), decltype(tv_tg0)>;
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), tv_ref_t, float>();
    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    for (uint ic0 = 0, ic = 0; ic < args.n_kv; ++ic0, ic += C) {
        if (tiitg < 64u) {
            const int kp = k_pos[ic + tiitg];
            kpos_tg[tiitg] = kp;
            const bool vis = (kp >= 0) && (kp <= tg_max_qp) && (n_swa == 0 || kp > tg_min_qp - n_swa);
            const bool far_ok = (n_swa == 0) && (kp >= 0) && (kp <= tg_min_qp - (int)extent);
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

        if constexpr (KV_KIND == 0) {
            auto tk = tensor((device half *)(k_base + (uint64_t)ic * args.nb12), dextents<int32_t, 2>(DK, C), array<int, 2>({1, sk_f16}));
            mm_qk.run(tq, tk, ts);
        } else if constexpr (KV_KIND == 1) {
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

        if constexpr (KV_KIND == 1) {
            inkling_fa_stage_kv_q8_0(skv, v, ic, kv_head, args.nb21, args.nb22, tiitg);
        } else if constexpr (KV_KIND == 2) {
            inkling_fa_stage_kv_q4_0(skv, v, ic, kv_head, args.nb21, args.nb22, tiitg);
        }

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
                const float sk_val = sink_r[jj];
                const uint64_t tbl_off = (uint64_t)(q_idx * 24u + head) * extent_p1;
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    const uint col = ii * NW + tiisg;
                    const int kp = kpos_tg[col];
                    const int d = qp - kp;
                    if (kp < 0 || d < 0 || (n_swa > 0 && d >= n_swa)) {
                        s[ii] = -INFINITY;
                    } else {
                        const uint b = (uint)d < extent ? (uint)d : extent;
                        s[ii] = ss[j * C + col] + (table[tbl_off + b] - sk_val);
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
                if (tiisg == 0) {
                    sf[0] = (int)(ic0 + 1u);
                }
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

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sf[0] == (int)(ic0 + 1u)) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        if constexpr (KV_KIND == 0) {
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

template [[host_name("kernel_inkling_fa_tensor_f16")]]  kernel void kernel_inkling_fa_tensor_impl<32, 0>(constant inkling_fa_tensor_args &, device const float *, device const char *, device const char *, device const float *, device const float *, device const int *, device const int *, device float *, threadgroup char *, uint2, ushort, ushort);
template [[host_name("kernel_inkling_fa_tensor_q8_0")]] kernel void kernel_inkling_fa_tensor_impl<16, 1>(constant inkling_fa_tensor_args &, device const float *, device const char *, device const char *, device const float *, device const float *, device const int *, device const int *, device float *, threadgroup char *, uint2, ushort, ushort);
template [[host_name("kernel_inkling_fa_tensor_q4_0")]] kernel void kernel_inkling_fa_tensor_impl<16, 2>(constant inkling_fa_tensor_args &, device const float *, device const char *, device const char *, device const float *, device const float *, device const int *, device const int *, device float *, threadgroup char *, uint2, ushort, ushort);

#endif // GGML_METAL_HAS_TENSOR
