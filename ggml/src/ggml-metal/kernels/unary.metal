#include "common.h"

kernel void kernel_inkling_control(
        device const float * cur, device const float * message,
        device const float * gain, device const float * forget,
        device float * output, constant uint & n_elems, uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    if (i >= n_elems) { return; }
    const float x = cur[i];
    const float forgotten = precise::tanh(forget[i]) * x;
    const float gained = precise::tanh(gain[i]) * message[i];
    output[i] = (x - forgotten) + gained;
}

kernel void kernel_inkling_situ(
        constant uint & width,
        constant float4 & scales,
        device const float4 * gate_up,
        device float4 * dst,
        uint2 p [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    if (p.x >= width) {
        return;
    }
    const uint src = p.y * 2 * width + p.x;
    const float4 gate = gate_up[src];
    float4 up = gate_up[src + width];
    const float4 curved = fma(scales.x, precise::tanh(fma(scales.y, gate, 0.0f)), 0.0f);
    const float4 sigmoid = 1.0f / (1.0f + exp(-gate));
    if (scales.z > 0.0f) {
        up = fma(scales.z, precise::tanh(fma(scales.w, up, 0.0f)), 0.0f);
    }
    dst[p.y * width + p.x] = (curved * sigmoid) * up;
}

// Fused Inkling depth-attention residual mixer: replaces O(K) concat + rms_norm +
// repeat + mul_mat + softmax + cont(transpose) + mul_mat with a single pass.
// Output layout is [n_embd, n_tokens, n_q] so each query slice is contiguous.
struct inkling_mix_args {
    uint  n_embd;
    uint  n_tokens;
    uint  n_states;
    uint  n_q;
    float eps;
    float scale;
};

kernel void kernel_inkling_mix(
        constant inkling_mix_args & args [[buffer(0)]],
        device const float4 * query      [[buffer(1)]],
        device       float4 * dst        [[buffer(2)]],
        device const float4 * s0         [[buffer(3)]],
        device const float4 * s1         [[buffer(4)]],
        device const float4 * s2         [[buffer(5)]],
        device const float4 * s3         [[buffer(6)]],
        device const float4 * s4         [[buffer(7)]],
        device const float4 * s5         [[buffer(8)]],
        device const float4 * s6         [[buffer(9)]],
        device const float4 * s7         [[buffer(10)]],
        device const float4 * s8         [[buffer(11)]],
        uint   t     [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    if (t >= args.n_tokens) { return; }
    device const float4 * st[9] = { s0, s1, s2, s3, s4, s5, s6, s7, s8 };
    const uint n4 = args.n_embd / 4u;
    const ulong row_off = ulong(t) * n4;

    threadgroup float sh_red[5][8];
    threadgroup float sh_inv[9];
    threadgroup float sh_w[4][9];

    for (uint k = 0; k < args.n_states; ++k) {
        device const float4 * sk = st[k] + row_off;
        float sq = 0.0f;
        float dq[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (uint i = tid; i < n4; i += 256u) {
            const float4 v = sk[i];
            sq += dot(v, v);
            for (uint q = 0; q < args.n_q; ++q) {
                dq[q] += dot(v, query[ulong(q) * n4 + i]);
            }
        }
        sq = simd_sum(sq);
        if (lane == 0) { sh_red[0][sg] = sq; }
        for (uint q = 0; q < args.n_q; ++q) {
            const float s = simd_sum(dq[q]);
            if (lane == 0) { sh_red[1 + q][sg] = s; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float tot_sq = 0.0f;
            for (uint j = 0; j < 8u; ++j) { tot_sq += sh_red[0][j]; }
            const float inv = rsqrt(tot_sq / float(args.n_embd) + args.eps);
            sh_inv[k] = inv;
            for (uint q = 0; q < args.n_q; ++q) {
                float tot_d = 0.0f;
                for (uint j = 0; j < 8u; ++j) { tot_d += sh_red[1 + q][j]; }
                sh_w[q][k] = tot_d * inv * args.scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        for (uint q = 0; q < args.n_q; ++q) {
            float mx = sh_w[q][0];
            for (uint k = 1; k < args.n_states; ++k) { mx = max(mx, sh_w[q][k]); }
            float denom = 0.0f;
            for (uint k = 0; k < args.n_states; ++k) {
                const float e = exp(sh_w[q][k] - mx);
                sh_w[q][k] = e;
                denom += e;
            }
            const float inv_d = 1.0f / denom;
            for (uint k = 0; k < args.n_states; ++k) {
                sh_w[q][k] = (sh_w[q][k] * inv_d) * sh_inv[k];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong stride_q = ulong(args.n_tokens) * n4;
    for (uint i = tid; i < n4; i += 256u) {
        float4 acc[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };
        for (uint k = 0; k < args.n_states; ++k) {
            const float4 v = st[k][row_off + i];
            for (uint q = 0; q < args.n_q; ++q) {
                acc[q] = fma(float4(sh_w[q][k]), v, acc[q]);
            }
        }
        for (uint q = 0; q < args.n_q; ++q) {
            dst[ulong(q) * stride_q + row_off + i] = acc[q];
        }
    }
}

// Fused Inkling loop controller down-projection: rms_norm(cur) + rms_norm(msg) + concat + Q8_0 matvec.
struct inkling_block_q8_0 {
    half   d;
    int8_t qs[32];
};
static_assert(sizeof(inkling_block_q8_0) == 34, "inkling_block_q8_0 must match GGML block_q8_0");

struct inkling_ctrl_args {
    uint  n_embd;
    uint  rank;
    uint  n_tokens;
    float eps;
};

kernel void kernel_inkling_ctrl_down_q8_0(
        constant inkling_ctrl_args & args        [[buffer(0)]],
        device const float * cur                 [[buffer(1)]],
        device const float * message             [[buffer(2)]],
        device const inkling_block_q8_0 * w_down [[buffer(3)]],
        device       float * dst                 [[buffer(4)]],
        uint2  group [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    const uint t = group.y;
    if (t >= args.n_tokens) { return; }

    const uint n4 = args.n_embd / 4u;
    device const float4 * cur4 = (device const float4 *)(cur + ulong(t) * args.n_embd);
    device const float4 * msg4 = (device const float4 *)(message + ulong(t) * args.n_embd);

    threadgroup float sh_sq[2][8];
    threadgroup float sh_inv[2];

    float sq_c = 0.0f;
    float sq_m = 0.0f;
    for (uint i = tid; i < n4; i += 256u) {
        const float4 vc = cur4[i];
        const float4 vm = msg4[i];
        sq_c += dot(vc, vc);
        sq_m += dot(vm, vm);
    }
    sq_c = simd_sum(sq_c);
    sq_m = simd_sum(sq_m);
    if (lane == 0) {
        sh_sq[0][sg] = sq_c;
        sh_sq[1][sg] = sq_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tc = 0.0f, tm = 0.0f;
        for (uint j = 0; j < 8u; ++j) {
            tc += sh_sq[0][j];
            tm += sh_sq[1][j];
        }
        sh_inv[0] = rsqrt(tc / float(args.n_embd) + args.eps);
        sh_inv[1] = rsqrt(tm / float(args.n_embd) + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = group.x * 8u + sg;
    if (row >= args.rank) { return; }

    const uint nb_half = args.n_embd / 32u;
    device const inkling_block_q8_0 * w_row = w_down + ulong(row) * (2u * nb_half);
    device const float * cur_row = cur + ulong(t) * args.n_embd;
    device const float * msg_row = message + ulong(t) * args.n_embd;

    const uint g = lane / 4u;
    const uint p = (lane % 4u) * 8u;
    float sum_c = 0.0f;
    float sum_m = 0.0f;

    for (uint ib = g; ib < nb_half; ib += 8u) {
        const device inkling_block_q8_0 & bc = w_row[ib];
        const device inkling_block_q8_0 & bm = w_row[nb_half + ib];
        const device float4 * xc = (const device float4 *)(cur_row + ib * 32u + p);
        const device float4 * xm = (const device float4 *)(msg_row + ib * 32u + p);
        const float4 qc0 = float4(bc.qs[p + 0], bc.qs[p + 1], bc.qs[p + 2], bc.qs[p + 3]);
        const float4 qc1 = float4(bc.qs[p + 4], bc.qs[p + 5], bc.qs[p + 6], bc.qs[p + 7]);
        const float4 qm0 = float4(bm.qs[p + 0], bm.qs[p + 1], bm.qs[p + 2], bm.qs[p + 3]);
        const float4 qm1 = float4(bm.qs[p + 4], bm.qs[p + 5], bm.qs[p + 6], bm.qs[p + 7]);
        sum_c = fma(float(bc.d), dot(qc0, xc[0]) + dot(qc1, xc[1]), sum_c);
        sum_m = fma(float(bm.d), dot(qm0, xm[0]) + dot(qm1, xm[1]), sum_m);
    }

    const float total = simd_sum(sum_c) * sh_inv[0] + simd_sum(sum_m) * sh_inv[1];
    if (lane == 0) {
        dst[ulong(t) * args.rank + row] = total;
    }
}

// Fused Inkling loop controller up-projection + gating: Q8_0 gain & forget + precise tanh residual mix.
kernel void kernel_inkling_ctrl_apply_q8_0(
        constant inkling_ctrl_args & args          [[buffer(0)]],
        device const float * cur                   [[buffer(1)]],
        device const float * message               [[buffer(2)]],
        device const float * gate                  [[buffer(3)]],
        device const inkling_block_q8_0 * w_gain   [[buffer(4)]],
        device const inkling_block_q8_0 * w_forget [[buffer(5)]],
        device       float * dst                   [[buffer(6)]],
        uint2  group [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    const uint t = group.y;
    if (t >= args.n_tokens) { return; }

    threadgroup float sh_gate[256];
    if (tid < args.rank) {
        sh_gate[tid] = gate[ulong(t) * args.rank + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint r = group.x * 256u + tid;
    if (r >= args.n_embd) { return; }

    const uint nb_r = args.rank / 32u;
    device const inkling_block_q8_0 * bg_row = w_gain   + ulong(r) * nb_r;
    device const inkling_block_q8_0 * bf_row = w_forget + ulong(r) * nb_r;

    float g_acc = 0.0f;
    float f_acc = 0.0f;
    for (uint ib = 0; ib < nb_r; ++ib) {
        const device inkling_block_q8_0 & bg = bg_row[ib];
        const device inkling_block_q8_0 & bf = bf_row[ib];
        float dg = 0.0f;
        float df = 0.0f;
        const uint base = ib * 32u;
        for (uint j = 0; j < 32u; ++j) {
            const float gv = sh_gate[base + j];
            dg += float(bg.qs[j]) * gv;
            df += float(bf.qs[j]) * gv;
        }
        g_acc += float(bg.d) * dg;
        f_acc += float(bf.d) * df;
    }

    const ulong idx = ulong(t) * args.n_embd + r;
    const float x = cur[idx];
    const float forgotten = precise::tanh(f_acc) * x;
    const float gained = precise::tanh(g_acc) * message[idx];
    dst[idx] = (x - forgotten) + gained;
}

// Fused single-pass Inkling DSpark Markov head + strict first-maximum argmax.
// Replaces w2*latent + w2_b + base + softcap + 6-pass tie-break + CPU logit readback.
struct inkling_dspark_part {
    float val;
    uint  idx;
};

struct inkling_dspark_argmax_args {
    uint rank;
    uint vocab;
    uint has_shortlist;
};

inline bool inkling_dspark_better(float val_a, uint idx_a, float val_b, uint idx_b) {
    if (idx_a == 0xFFFFFFFFu) { return false; }
    if (idx_b == 0xFFFFFFFFu) { return true; }
    return (val_a > val_b) || (val_a == val_b && idx_a < idx_b);
}

kernel void kernel_inkling_dspark_argmax_s1_q8_0(
        constant inkling_dspark_argmax_args & args [[buffer(0)]],
        device const inkling_block_q8_0 * w2       [[buffer(1)]],
        device const float * w2_b                  [[buffer(2)]],
        device const float * base_row              [[buffer(3)]],
        device const float * latent                [[buffer(4)]],
        device inkling_dspark_part * parts         [[buffer(5)]],
        uint   group [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh_lat[512];
    for (uint k = tid; k < args.rank; k += 256u) {
        sh_lat[k] = latent[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint nb = args.rank / 32u;
    const uint g_blk = lane / 4u;
    const uint p = (lane % 4u) * 8u;

    float best_val = -INFINITY;
    uint  best_idx = 0xFFFFFFFFu;

    for (uint v = group * 8u + sg; v < args.vocab; v += 2048u) {
        device const inkling_block_q8_0 * w_row = w2 + ulong(v) * nb;
        float acc = 0.0f;
        for (uint ib = g_blk; ib < nb; ib += 8u) {
            const device inkling_block_q8_0 & b = w_row[ib];
            const uint k0 = ib * 32u + p;
            const float4 l0 = float4(sh_lat[k0 + 0], sh_lat[k0 + 1], sh_lat[k0 + 2], sh_lat[k0 + 3]);
            const float4 l1 = float4(sh_lat[k0 + 4], sh_lat[k0 + 5], sh_lat[k0 + 6], sh_lat[k0 + 7]);
            const float4 q0 = float4(b.qs[p + 0], b.qs[p + 1], b.qs[p + 2], b.qs[p + 3]);
            const float4 q1 = float4(b.qs[p + 4], b.qs[p + 5], b.qs[p + 6], b.qs[p + 7]);
            acc = fma(float(b.d), dot(q0, l0) + dot(q1, l1), acc);
        }
        const float d = simd_sum(acc);
        if (lane == 0) {
            const float s = d + w2_b[v] + base_row[v];
            if (!isnan(s) && inkling_dspark_better(s, v, best_val, best_idx)) {
                best_val = s;
                best_idx = v;
            }
        }
    }

    threadgroup inkling_dspark_part sh_sg[8];
    if (lane == 0) {
        sh_sg[sg] = { best_val, best_idx };
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        inkling_dspark_part win = sh_sg[0];
        for (uint j = 1u; j < 8u; ++j) {
            if (inkling_dspark_better(sh_sg[j].val, sh_sg[j].idx, win.val, win.idx)) {
                win = sh_sg[j];
            }
        }
        parts[group] = win;
    }
}

#if defined(GGML_METAL_HAS_BF16)
kernel void kernel_inkling_dspark_argmax_s1_bf16(
        constant inkling_dspark_argmax_args & args [[buffer(0)]],
        device const bfloat4 * w2                  [[buffer(1)]],
        device const float * w2_b                  [[buffer(2)]],
        device const float * base_row              [[buffer(3)]],
        device const float * latent                [[buffer(4)]],
        device inkling_dspark_part * parts         [[buffer(5)]],
        uint   group [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh_lat[512];
    for (uint k = tid; k < args.rank; k += 256u) {
        sh_lat[k] = latent[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint r4 = args.rank / 4u;
    float best_val = -INFINITY;
    uint  best_idx = 0xFFFFFFFFu;

    for (uint v = group * 8u + sg; v < args.vocab; v += 2048u) {
        device const bfloat4 * w_row = w2 + ulong(v) * r4;
        float acc = 0.0f;
        for (uint i = lane; i < r4; i += 32u) {
            const float4 wv = float4(w_row[i]);
            const uint k0 = i * 4u;
            const float4 lv = float4(sh_lat[k0 + 0], sh_lat[k0 + 1], sh_lat[k0 + 2], sh_lat[k0 + 3]);
            acc += dot(wv, lv);
        }
        const float d = simd_sum(acc);
        if (lane == 0) {
            const float s = d + w2_b[v] + base_row[v];
            if (!isnan(s) && inkling_dspark_better(s, v, best_val, best_idx)) {
                best_val = s;
                best_idx = v;
            }
        }
    }

    threadgroup inkling_dspark_part sh_sg[8];
    if (lane == 0) {
        sh_sg[sg] = { best_val, best_idx };
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        inkling_dspark_part win = sh_sg[0];
        for (uint j = 1u; j < 8u; ++j) {
            if (inkling_dspark_better(sh_sg[j].val, sh_sg[j].idx, win.val, win.idx)) {
                win = sh_sg[j];
            }
        }
        parts[group] = win;
    }
}
#endif

kernel void kernel_inkling_dspark_argmax_s2(
        constant inkling_dspark_argmax_args & args [[buffer(0)]],
        device const inkling_dspark_part * parts   [[buffer(1)]],
        device const int * shortlist_ids           [[buffer(2)]],
        device       int * dst                     [[buffer(3)]],
        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup inkling_dspark_part sh[256];
    sh[tid] = parts[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            if (inkling_dspark_better(sh[tid + stride].val, sh[tid + stride].idx, sh[tid].val, sh[tid].idx)) {
                sh[tid] = sh[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const uint local_idx = (sh[0].idx < args.vocab) ? sh[0].idx : 0u;
        dst[0] = args.has_shortlist ? shortlist_ids[local_idx] : int(local_idx);
    }
}

constant short FC_unary_op [[function_constant(FC_UNARY + 0)]];
constant bool  FC_unary_cnt[[function_constant(FC_UNARY + 1)]];

template <typename T, typename U>
inline constexpr T ggml_metal_unary_cast(U x) {
    return (T) x;
}

#if defined(GGML_METAL_HAS_BF16)
// MSL has no C-style cast from float vectors to bfloat vectors
template <>
inline bfloat4 ggml_metal_unary_cast<bfloat4, float4>(float4 x) {
    return bfloat4(bfloat(x[0]), bfloat(x[1]), bfloat(x[2]), bfloat(x[3]));
}

template <>
inline bfloat4 ggml_metal_unary_cast<bfloat4, float>(float x) {
    return bfloat4(bfloat(x), bfloat(x), bfloat(x), bfloat(x));
}
#endif

template <typename T0, typename T, typename TC>
kernel void kernel_unary_impl(
        constant ggml_metal_kargs_unary & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
#define FC_OP  FC_unary_op
#define FC_CNT FC_unary_cnt

    device const T0 * src0_ptr;
    device       T  * dst_ptr;

    int i0;

    if (FC_CNT) {
        i0 = tgpig.x*ntg.x + tpitg.x;

        src0_ptr = (device const T0 *) (src0);
        dst_ptr  = (device       T  *) (dst);
    } else {
        const int i03 = tgpig.z;
        const int i02 = tgpig.y;
        const int k0  = tgpig.x/args.ne01;
        const int i01 = tgpig.x - k0*args.ne01;

        i0 = k0*ntg.x + tpitg.x;

        src0_ptr = (device const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
        dst_ptr  = (device       T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1 );
    }

    {
        //threadgroup_barrier(mem_flags::mem_none);

        if (!FC_CNT) {
            if (i0 >= args.ne0) {
                return;
            }
        } else {
            if (i0 >= args.ne00*args.ne01*args.ne02*args.ne03) {
                return;
            }
        }

        const TC x = (TC) src0_ptr[i0];

        if (FC_OP == OP_UNARY_NUM_SCALE) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(args.scale * x + args.bias);
        }

        if (FC_OP == OP_UNARY_NUM_FILL) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(args.val);
        }

        if (FC_OP == OP_UNARY_NUM_CLAMP) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(clamp(x, args.min, args.max));
        }

        if (FC_OP == OP_UNARY_NUM_SQR) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(x * x);
        }

        if (FC_OP == OP_UNARY_NUM_SQRT) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(sqrt(x));
        }

        if (FC_OP == OP_UNARY_NUM_SIN) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(sin(x));
        }

        if (FC_OP == OP_UNARY_NUM_COS) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(cos(x));
        }

        if (FC_OP == OP_UNARY_NUM_LOG) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(log(x));
        }

        if (FC_OP == OP_UNARY_NUM_LEAKY_RELU) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(TC(x > 0)*x + TC(x <= 0)*(x * args.slope));
        }

        if (FC_OP == OP_UNARY_NUM_TANH) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(precise::tanh(x));
        }

        if (FC_OP == OP_UNARY_NUM_RELU) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(fmax(0, x));
        }

        if (FC_OP == OP_UNARY_NUM_SIGMOID) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(1 / (1 + exp(-x)));
        }

        if (FC_OP == OP_UNARY_NUM_GELU) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(0.5*x*(1 + precise::tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x))));
        }

        if (FC_OP == OP_UNARY_NUM_GELU_ERF) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(0.5*x*(1 + erf_approx(SQRT_2_INV*x)));
        }

        if (FC_OP == OP_UNARY_NUM_GELU_QUICK) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(x * (1/(1 + exp(GELU_QUICK_COEF*x))));
        }

        if (FC_OP == OP_UNARY_NUM_SILU) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(x / (1 + exp(-x)));
        }

        if (FC_OP == OP_UNARY_NUM_ELU) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(elu_approx(x));
        }

        if (FC_OP == OP_UNARY_NUM_NEG) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(-x);
        }

        if (FC_OP == OP_UNARY_NUM_ABS) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(fabs(x));
        }

        if (FC_OP == OP_UNARY_NUM_SGN) {
            dst_ptr[i0] = T(x > 0) - T(x < 0);
        }

        if (FC_OP == OP_UNARY_NUM_STEP) {
            dst_ptr[i0] = T(x > 0);
        }

        if (FC_OP == OP_UNARY_NUM_HARDSWISH) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(x * fmax(0, fmin(1, x/6 + 0.5)));
        }

        if (FC_OP == OP_UNARY_NUM_HARDSIGMOID) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(fmax(0, fmin(1, x/6 + 0.5)));
        }

        if (FC_OP == OP_UNARY_NUM_EXP) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(exp(x));
        }

        if (FC_OP == OP_UNARY_NUM_SOFTPLUS) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(select(log(1 + exp(x)), x, x > 20));
        }

        if (FC_OP == OP_UNARY_NUM_EXPM1) {
            // TODO: precise implementation
            dst_ptr[i0] = ggml_metal_unary_cast<T>(exp(x) - 1);
        }

        if (FC_OP == OP_UNARY_NUM_FLOOR) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(floor(x));
        }

        if (FC_OP == OP_UNARY_NUM_CEIL) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(ceil(x));
        }

        if (FC_OP == OP_UNARY_NUM_ROUND) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(round(x));
        }

        if (FC_OP == OP_UNARY_NUM_TRUNC) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(trunc(x));
        }

        if (FC_OP == OP_UNARY_NUM_XIELU) {
            const TC xi      = x;
            const TC gate    = TC(xi > TC(0.0f));
            const TC clamped = fmin(xi, TC(args.val));
            const TC y_pos   = TC(args.scale) * xi * xi + TC(args.bias) * xi;
            const TC y_neg   = (exp(clamped) - TC(1.0f) - xi) * TC(args.slope) + TC(args.bias) * xi;
            dst_ptr[i0] = ggml_metal_unary_cast<T>(gate * y_pos + (TC(1.0f) - gate) * y_neg);
        }
    }

#undef FC_OP
#undef FC_CNT
}

typedef decltype(kernel_unary_impl<float, float, float>) kernel_unary_t;

template [[host_name("kernel_unary_f32_f32")]]   kernel kernel_unary_t kernel_unary_impl<float,  float,  float>;
template [[host_name("kernel_unary_f32_f32_4")]] kernel kernel_unary_t kernel_unary_impl<float4, float4, float4>;
template [[host_name("kernel_unary_f16_f16")]]   kernel kernel_unary_t kernel_unary_impl<half,   half,   float>;
template [[host_name("kernel_unary_f16_f16_4")]] kernel kernel_unary_t kernel_unary_impl<half4,  half4,  float4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_unary_bf16_bf16")]]   kernel kernel_unary_t kernel_unary_impl<bfloat,  bfloat,  float>;
template [[host_name("kernel_unary_bf16_bf16_4")]] kernel kernel_unary_t kernel_unary_impl<bfloat4, bfloat4, float4>;
#endif

kernel void kernel_silu_back_f32(
        constant ggml_metal_kargs_silu_back & args,
        device const float * dy,
        device const float * x,
        device       float * dx,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.ne) {
        return;
    }

    const float s = 1.0f / (1.0f + exp(-x[gid]));
    dx[gid] = dy[gid] * s * (1.0f + x[gid] * (1.0f - s));
}

template<typename T>
kernel void kernel_reglu(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        dst_row[i0] = (T)(x0*x1*(x0 > 0.0f));
    }
}

typedef decltype(kernel_reglu<float>) kernel_reglu_t;

template [[host_name("kernel_reglu_f32")]] kernel kernel_reglu_t kernel_reglu<float>;
template [[host_name("kernel_reglu_f16")]] kernel kernel_reglu_t kernel_reglu<half>;

template<typename T>
kernel void kernel_geglu(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu = 0.5f*x0*(1.0f + precise::tanh(SQRT_2_OVER_PI*x0*(1.0f + GELU_COEF_A*x0*x0)));

        dst_row[i0] = (T)(gelu*x1);
    }
}

typedef decltype(kernel_geglu<float>) kernel_geglu_t;

template [[host_name("kernel_geglu_f32")]] kernel kernel_geglu_t kernel_geglu<float>;
template [[host_name("kernel_geglu_f16")]] kernel kernel_geglu_t kernel_geglu<half>;

template<typename T>
kernel void kernel_swiglu(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float silu = x0 / (1.0f + exp(-x0));

        dst_row[i0] = (T)(silu*x1);
    }
}

typedef decltype(kernel_swiglu<float>) kernel_swiglu_t;

template [[host_name("kernel_swiglu_f32")]] kernel kernel_swiglu_t kernel_swiglu<float>;
template [[host_name("kernel_swiglu_f16")]] kernel kernel_swiglu_t kernel_swiglu<half>;

template<typename T>
kernel void kernel_swiglu_oai(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
        float x0 = src0_row[i0];
        float x1 = src1_row[i0];

        x0 = min(x0, args.limit);
        x1 = max(min(x1, args.limit), -args.limit);

        float out_glu = x0 / (1.0f + exp(-x0 * args.alpha));
        out_glu = out_glu * (1.0f + x1);

        dst_row[i0] = (T)out_glu;
    }
}

typedef decltype(kernel_swiglu_oai<float>) kernel_swiglu_oai_t;

template [[host_name("kernel_swiglu_oai_f32")]] kernel kernel_swiglu_oai_t kernel_swiglu_oai<float>;
template [[host_name("kernel_swiglu_oai_f16")]] kernel kernel_swiglu_oai_t kernel_swiglu_oai<half>;

template<typename T>
kernel void kernel_swiglu_clamp(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
        const float gate = min((float) src0_row[i0], args.limit);
        const float up = clamp((float) src1_row[i0], -args.limit, args.limit);

        dst_row[i0] = (T)(gate / (1.0f + exp(-gate)) * up);
    }
}

typedef decltype(kernel_swiglu_clamp<float>) kernel_swiglu_clamp_t;

template [[host_name("kernel_swiglu_clamp_f32")]] kernel kernel_swiglu_clamp_t kernel_swiglu_clamp<float>;
template [[host_name("kernel_swiglu_clamp_f16")]] kernel kernel_swiglu_clamp_t kernel_swiglu_clamp<half>;

template<typename T>
kernel void kernel_geglu_erf(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu_erf = 0.5f*x0*(1.0f+erf_approx<float>(x0*SQRT_2_INV));

        dst_row[i0] = (T)(gelu_erf*x1);
    }
}

typedef decltype(kernel_geglu_erf<float>) kernel_geglu_erf_t;

template [[host_name("kernel_geglu_erf_f32")]] kernel kernel_geglu_erf_t kernel_geglu_erf<float>;
template [[host_name("kernel_geglu_erf_f16")]] kernel kernel_geglu_erf_t kernel_geglu_erf<half>;

template<typename T>
kernel void kernel_geglu_quick(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu_quick = x0*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x0)));

        dst_row[i0] = (T)(gelu_quick*x1);
    }
}

typedef decltype(kernel_geglu_quick<float>) kernel_geglu_quick_t;

template [[host_name("kernel_geglu_quick_f32")]] kernel kernel_geglu_quick_t kernel_geglu_quick<float>;
template [[host_name("kernel_geglu_quick_f16")]] kernel kernel_geglu_quick_t kernel_geglu_quick<half>;
