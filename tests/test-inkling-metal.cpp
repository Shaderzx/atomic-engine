// Per-op Metal parity & safety tests for Inkling2 fused custom kernels.
// Exercises every custom Metal op against a pure C++ F32 reference implementation
// (run with MTL_SHADER_VALIDATION=1 to verify zero misaligned/OOB GPU accesses).
#include "testing.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

using fa_ready_fn = bool (*)(ggml_backend_t, int64_t, ggml_type);
using fa_create_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *,
        ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, float, int32_t, int32_t);
using fa_gated_create_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *,
        ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, float, int32_t, int32_t);
using mix_create_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor * const *, int, ggml_tensor *);
using mix_norm_create_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor * const *, int, ggml_tensor *, ggml_tensor * const *, float);
using conv_fused_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *,
        ggml_tensor *, ggml_tensor *, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, float);
using conv_fused_v2_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *,
        ggml_tensor *, ggml_tensor *, ggml_tensor *, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, float);
using moe_ffn_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, float, float);
using ctrl_down_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *);
using ctrl_apply_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *);
using mul_sigmoid_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *);
using softcap_trim_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *, int64_t, float);
using dspark_ready_fn = bool (*)(ggml_backend_t, int64_t, ggml_type);
using dspark_argmax_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, float);

static std::vector<float> rand_vec(size_t n, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> out(n);
    for (float & v : out) { v = dist(rng); }
    return out;
}

static void quantize_and_dequant(ggml_type type, const std::vector<float> & src, int64_t n_rows, int64_t n_cols,
        std::vector<uint8_t> & qbytes, std::vector<float> & ref_f32) {
    const size_t row_bytes = ggml_row_size(type, n_cols);
    qbytes.resize((size_t) n_rows * row_bytes);
    ref_f32.resize((size_t) n_rows * (size_t) n_cols);

    const auto * traits = ggml_get_type_traits(type);
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src.data(), reinterpret_cast<ggml_fp16_t *>(qbytes.data()), n_rows * n_cols);
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(qbytes.data()), ref_f32.data(), n_rows * n_cols);
    } else if (type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(src.data(), reinterpret_cast<ggml_bf16_t *>(qbytes.data()), n_rows * n_cols);
        ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(qbytes.data()), ref_f32.data(), n_rows * n_cols);
    } else {
        std::vector<float> imat((size_t) n_cols, 1.0f);
        ggml_quantize_chunk(type, src.data(), qbytes.data(), 0, n_rows, n_cols, imat.data());
        for (int64_t r = 0; r < n_rows; ++r) {
            traits->to_float(qbytes.data() + (size_t) r * row_bytes, ref_f32.data() + (size_t) r * (size_t) n_cols, n_cols);
        }
    }
}

static std::vector<float> cpu_ref_inkling_fa(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & table,
        const std::vector<float> & sinks,
        const std::vector<int32_t> & q_pos,
        const std::vector<int32_t> & k_pos,
        int64_t n_q, int64_t n_kv, int32_t extent, int32_t n_swa, float scale) {
    constexpr int64_t DK = 128;
    constexpr int64_t NH = 24;
    constexpr int64_t NKV = 8;
    std::vector<float> out((size_t) n_q * NH * DK, 0.0f);

    for (int64_t iq = 0; iq < n_q; ++iq) {
        const int32_t qp = q_pos[(size_t) iq * 2u + 0u];
        const uint32_t qseq = (uint32_t) q_pos[(size_t) iq * 2u + 1u];
        const uint64_t qbit = qseq < 64u ? (1ULL << qseq) : 0ULL;

        for (int64_t h = 0; h < NH; ++h) {
            const int64_t kvh = h / 3;
            const float * qptr = q.data() + ((size_t) iq * NH + (size_t) h) * DK;
            const float sink = sinks[(size_t) iq * NH + (size_t) h];
            const float * tbl = table.data() + ((size_t) iq * NH + (size_t) h) * (size_t) extent;

            float max_s = -std::numeric_limits<float>::infinity();
            std::vector<float> scores((size_t) n_kv, -std::numeric_limits<float>::infinity());
            for (int64_t ik = 0; ik < n_kv; ++ik) {
                const int32_t kp = k_pos[(size_t) ik * 3u + 0u];
                const uint64_t kmask = (uint64_t) (uint32_t) k_pos[(size_t) ik * 3u + 1u] |
                                       ((uint64_t) (uint32_t) k_pos[(size_t) ik * 3u + 2u] << 32);
                if (kp < 0 || (kmask & qbit) == 0ULL) { continue; }
                const int32_t d = qp - kp;
                if (d < 0 || (n_swa > 0 && d >= n_swa)) { continue; }
                const float * kptr = k.data() + ((size_t) ik * NKV + (size_t) kvh) * DK;
                double dot = 0.0;
                for (int64_t j = 0; j < DK; ++j) {
                    dot += (double) qptr[j] * (double) kptr[j];
                }
                const float rel = d < extent ? tbl[d] : 0.0f;
                const float s = (float) (dot * (double) scale) + (rel - sink);
                scores[(size_t) ik] = s;
                if (s > max_s) { max_s = s; }
            }

            const float mf = std::max(max_s, 0.0f);
            double denom = std::exp((double) (-mf));
            std::vector<double> acc(DK, 0.0);
            for (int64_t ik = 0; ik < n_kv; ++ik) {
                if (!std::isfinite(scores[(size_t) ik])) { continue; }
                const double w = std::exp((double) (scores[(size_t) ik] - mf));
                denom += w;
                const float * vptr = v.data() + ((size_t) ik * NKV + (size_t) kvh) * DK;
                for (int64_t j = 0; j < DK; ++j) {
                    acc[(size_t) j] += w * (double) vptr[j];
                }
            }
            float * optr = out.data() + ((size_t) iq * NH + (size_t) h) * DK;
            for (int64_t j = 0; j < DK; ++j) {
                optr[j] = denom > 0.0 ? (float) (acc[(size_t) j] / denom) : 0.0f;
            }
        }
    }
    return out;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<float>::infinity();
        }
        mx = std::max(mx, std::fabs(a[i] - b[i]));
    }
    return mx;
}

} // namespace

int main(int argc, char ** argv) {
    testing t;
    for (int i = 1; i < argc; ++i) { t.set_filter(argv[i]); }

    ggml_backend_load_all();
    ggml_backend_dev_t metal_dev = ggml_backend_dev_by_name("MTL0");
    ggml_backend_ptr metal(metal_dev ? ggml_backend_dev_init(metal_dev, nullptr) : nullptr);
    if (!metal) {
        std::fprintf(stderr, "Metal backend unavailable; skipping test-inkling-metal\n");
        return 0;
    }
    ggml_backend_t backend = metal.get();
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));

    auto run_graph = [&](ggml_context * ctx, ggml_tensor * out,
                         const std::vector<std::pair<ggml_tensor *, std::pair<const void *, size_t>>> & uploads,
                         void * dst, size_t dst_bytes) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        t.assert_true(ggml_gallocr_alloc_graph(allocr, gf));
        for (const auto & u : uploads) {
            ggml_backend_tensor_set(u.first, u.second.first, 0, u.second.second);
        }
        t.assert_true(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, dst, 0, dst_bytes);
        ggml_gallocr_free(allocr);
    };

    t.test("inkling_flash_attn parity across KV types, partial tiles, split-K, SWA, all-far, all-masked, and multi-seq", [&](testing & t) {
        auto ready = reinterpret_cast<fa_ready_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_flash_attn_ready_v1"));
        auto create = reinterpret_cast<fa_create_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_flash_attn_v1"));
        t.assert_true(ready != nullptr && create != nullptr);

        struct fa_case {
            const char * label;
            ggml_type k_type;
            ggml_type v_type;
            int64_t n_q;
            int64_t n_kv;
            int32_t extent;
            int32_t n_swa;
            int32_t n_seqs;
            bool include_all_masked;
        };
        const std::vector<fa_case> cases = {
            // Decode (n_q < 16), BQ in {1..8}, nwg=1 (64), 4 (640), 16 (4160), 32 (33024), 64 (65600), 128 (131200), and mixed K/V
            {"dec_q1_f16_nwg1",     GGML_TYPE_F16,  GGML_TYPE_F16,   1,    64,  32,    0, 1, false},
            {"dec_q2_f16_split4",   GGML_TYPE_F16,  GGML_TYPE_F16,   2,   640,  64,    0, 1, false},
            {"dec_q3_f16_split4",   GGML_TYPE_F16,  GGML_TYPE_F16,   3,   640,  64,    0, 1, false},
            {"dec_q4_q8_split4",    GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,  4,   640,  64,  256, 1, false},
            {"dec_q5_f16_split4",   GGML_TYPE_F16,  GGML_TYPE_F16,   5,   640,  64,    0, 1, false},
            {"dec_q6_q8q4_split16", GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,  6,  4160,  64,    0, 1, false},
            {"dec_q7_f16q4_split4", GGML_TYPE_F16,  GGML_TYPE_Q4_0,  7,   640,  64,  256, 1, false},
            {"dec_q8_q4_split16",   GGML_TYPE_Q4_0, GGML_TYPE_Q4_0,  8,  4160,  64,    0, 2, true },
            {"dec_q8_f16_split64",  GGML_TYPE_F16,  GGML_TYPE_F16,   8, 65600,  64,    0, 1, false},
            {"dec_q5_f16_split128", GGML_TYPE_F16,  GGML_TYPE_F16,   5,131200,  64,    0, 1, false},
            {"dec_q3_q8q4_split16", GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,  3,  4160,  64,    0, 1, false},
            {"dec_q3_q4_split16",   GGML_TYPE_Q4_0, GGML_TYPE_Q4_0,  3,  4160,  64,    0, 1, false},
            {"dec_q1_f16_split32",  GGML_TYPE_F16,  GGML_TYPE_F16,   1, 33024,  64,    0, 1, false},
            {"dec_q3_q8q4_split64", GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,  3, 65600,  64,    0, 1, false},
            {"dec_q3_q4_split32",   GGML_TYPE_Q4_0, GGML_TYPE_Q4_0,  3, 33024,  64,    0, 2, true },
            {"dec_q15_f16_multiseq",GGML_TYPE_F16,  GGML_TYPE_F16,  15,   640,  64,    0, 4, true },
            // Prefill (n_q >= 16, partial tiles n_q=17, 48), all-far tiles + SWA + multi-seq + mixed K/V
            {"pre_q17_f16_global",  GGML_TYPE_F16,  GGML_TYPE_F16,  17,   640,  64,    0, 1, false},
            {"pre_q48_f16_multiseq",GGML_TYPE_F16,  GGML_TYPE_F16,  48,   640,  64,    0, 3, true },
            {"pre_q17_q8_swa",      GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 17,   640,  64,  128, 2, false},
            {"pre_q17_q8q4_global", GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 17,   640,  64,    0, 2, false},
            {"pre_q17_f16q4_global",GGML_TYPE_F16,  GGML_TYPE_Q4_0, 17,   640,  64,    0, 1, false},
            {"pre_q17_q4_global",   GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, 17,   640,  64,    0, 2, false},
        };

        const float scale = 1.0f / std::sqrt(128.0f);
        uint32_t seed = 100;
        for (const auto & tc : cases) {
            if (!ready(backend, tc.n_q, tc.k_type) || !ready(backend, tc.n_q, tc.v_type)) {
                continue;
            }
            ++seed;
            const auto q_f32     = rand_vec((size_t) tc.n_q * 24 * 128, seed * 10 + 1, -0.8f, 0.8f);
            const auto k_raw     = rand_vec((size_t) tc.n_kv * 8 * 128, seed * 10 + 2, -0.8f, 0.8f);
            const auto v_raw     = rand_vec((size_t) tc.n_kv * 8 * 128, seed * 10 + 3, -0.8f, 0.8f);
            const auto table_f32 = rand_vec((size_t) tc.n_q * 24 * tc.extent, seed * 10 + 4, -0.5f, 0.5f);
            const auto sinks_f32 = rand_vec((size_t) tc.n_q * 24, seed * 10 + 5, -0.5f, 0.5f);

            std::vector<uint8_t> k_bytes, v_bytes;
            std::vector<float> k_ref, v_ref;
            quantize_and_dequant(tc.k_type, k_raw, tc.n_kv * 8, 128, k_bytes, k_ref);
            quantize_and_dequant(tc.v_type, v_raw, tc.n_kv * 8, 128, v_bytes, v_ref);

            // Populate multi-sequence KV cells: interleaved seqs + shared prefix + empty holes
            std::vector<int32_t> k_pos((size_t) tc.n_kv * 3u, 0);
            std::vector<int32_t> seq_len((size_t) tc.n_seqs, 0);
            for (int64_t ik = 0; ik < tc.n_kv; ++ik) {
                if (ik >= tc.n_kv - 17) {
                    // Empty tail cells
                    k_pos[(size_t) ik * 3u + 0u] = -1;
                    continue;
                }
                if (tc.n_seqs > 1 && ik < 32) {
                    // Shared prefix across all sequences
                    k_pos[(size_t) ik * 3u + 0u] = (int32_t) ik;
                    uint64_t mask = (1ULL << tc.n_seqs) - 1ULL;
                    k_pos[(size_t) ik * 3u + 1u] = (int32_t) (uint32_t) mask;
                    k_pos[(size_t) ik * 3u + 2u] = (int32_t) (uint32_t) (mask >> 32);
                    for (int32_t s = 0; s < tc.n_seqs; ++s) { seq_len[s] = (int32_t) ik + 1; }
                } else {
                    const int32_t s = (int32_t) (ik % tc.n_seqs);
                    k_pos[(size_t) ik * 3u + 0u] = seq_len[s]++;
                    const uint64_t mask = 1ULL << s;
                    k_pos[(size_t) ik * 3u + 1u] = (int32_t) (uint32_t) mask;
                    k_pos[(size_t) ik * 3u + 2u] = (int32_t) (uint32_t) (mask >> 32);
                }
            }

            std::vector<int32_t> q_pos((size_t) tc.n_q * 2u, 0);
            for (int64_t iq = 0; iq < tc.n_q; ++iq) {
                const int32_t s = (int32_t) (iq % tc.n_seqs);
                int32_t pos = std::max<int32_t>(0, seq_len[s] - (int32_t) tc.n_q + (int32_t) iq);
                if (tc.include_all_masked && iq == tc.n_q - 1) {
                    // Belongs to an unused sequence -> all keys masked -> output must be exactly 0
                    q_pos[(size_t) iq * 2u + 0u] = 500;
                    q_pos[(size_t) iq * 2u + 1u] = 63;
                } else {
                    q_pos[(size_t) iq * 2u + 0u] = pos;
                    q_pos[(size_t) iq * 2u + 1u] = s;
                }
            }

            ggml_context_ptr ctx(ggml_init({4 * 1024 * 1024, nullptr, true}));
            auto * t_q     = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 24, tc.n_q);
            auto * t_k     = ggml_new_tensor_3d(ctx.get(), tc.k_type, 128, 8, tc.n_kv);
            auto * t_v     = ggml_new_tensor_3d(ctx.get(), tc.v_type, 128, 8, tc.n_kv);
            auto * t_tbl   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, tc.extent, 24, tc.n_q);
            auto * t_sinks = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 24, tc.n_q);
            auto * t_qpos  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, tc.n_q);
            auto * t_kpos  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 3, tc.n_kv);

            auto * out = create(ctx.get(), t_q, t_k, t_v, t_tbl, t_sinks, t_qpos, t_kpos, scale, tc.extent, tc.n_swa);
            t.assert_true(out != nullptr);
            t.assert_true(ggml_backend_supports_op(backend, out));

            std::vector<float> got((size_t) tc.n_q * 24 * 128, -999.0f);
            run_graph(ctx.get(), out, {
                {t_q,     {q_f32.data(),     q_f32.size() * sizeof(float)}},
                {t_k,     {k_bytes.data(),   k_bytes.size()}},
                {t_v,     {v_bytes.data(),   v_bytes.size()}},
                {t_tbl,   {table_f32.data(), table_f32.size() * sizeof(float)}},
                {t_sinks, {sinks_f32.data(), sinks_f32.size() * sizeof(float)}},
                {t_qpos,  {q_pos.data(),     q_pos.size() * sizeof(int32_t)}},
                {t_kpos,  {k_pos.data(),     k_pos.size() * sizeof(int32_t)}},
            }, got.data(), got.size() * sizeof(float));

            const auto ref = cpu_ref_inkling_fa(q_f32, k_ref, v_ref, table_f32, sinks_f32, q_pos, k_pos,
                    tc.n_q, tc.n_kv, tc.extent, tc.n_swa, scale);
            const float err = max_abs_diff(got, ref);
            const float tol = tc.n_q >= 16 ? 3e-3f : 2e-3f;
            if (err > tol) {
                std::fprintf(stderr, "FAIL %s: max_abs_diff=%.6f > %.6f\n", tc.label, err, tol);
            }
            t.assert_true(err <= tol);
        }
    });

    t.test("inkling_mix parity and GGML_MAX_SRC-2 state cap", [&](testing & t) {
        auto create = reinterpret_cast<mix_create_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_mix_v1"));
        t.assert_true(create != nullptr);

        // Verify 9 states is cleanly rejected (GGML_MAX_SRC == 10, max 8 states + 1 query = 9 < 10)
        {
            ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}));
            ggml_tensor * st[9];
            for (int i = 0; i < 9; ++i) { st[i] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 64, 2); }
            auto * q = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 64, 1);
            t.assert_true(create(ctx.get(), st, 9, q) == nullptr);
        }

        for (int n_st : {2, 4, 8}) {
            for (int n_q : {1, 3}) {
                for (int n_tok : {1, 3, 17}) {
                    const int64_t n_embd = 256;
                    ggml_context_ptr ctx(ggml_init({2 * 1024 * 1024, nullptr, true}));
                    std::vector<ggml_tensor *> st_t((size_t) n_st);
                    std::vector<std::vector<float>> st_v((size_t) n_st);
                    std::vector<std::pair<ggml_tensor *, std::pair<const void *, size_t>>> ups;
                    for (int k = 0; k < n_st; ++k) {
                        st_t[(size_t) k] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_tok);
                        st_v[(size_t) k] = rand_vec((size_t) n_embd * n_tok, 200u + k * 7u + n_tok);
                        ups.push_back({st_t[(size_t) k], {st_v[(size_t) k].data(), st_v[(size_t) k].size() * sizeof(float)}});
                    }
                    auto * q_t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_q);
                    auto q_v = rand_vec((size_t) n_embd * n_q, 999u + n_st + n_q);
                    ups.push_back({q_t, {q_v.data(), q_v.size() * sizeof(float)}});

                    auto * out = create(ctx.get(), st_t.data(), n_st, q_t);
                    t.assert_true(out != nullptr && ggml_backend_supports_op(backend, out));

                    std::vector<float> got((size_t) n_embd * n_tok * n_q);
                    run_graph(ctx.get(), out, ups, got.data(), got.size() * sizeof(float));

                    // CPU F32 reference
                    std::vector<float> ref(got.size(), 0.0f);
                    const float inv_sqrt = 1.0f / std::sqrt((float) n_embd);
                    for (int tok = 0; tok < n_tok; ++tok) {
                        std::vector<float> inv((size_t) n_st);
                        std::vector<std::vector<float>> sc((size_t) n_q, std::vector<float>((size_t) n_st, 0.0f));
                        for (int k = 0; k < n_st; ++k) {
                            double sq = 0.0;
                            for (int i = 0; i < n_embd; ++i) {
                                const float v = st_v[(size_t) k][(size_t) tok * n_embd + i];
                                sq += (double) v * v;
                            }
                            inv[(size_t) k] = 1.0f / std::sqrt((float) (sq / n_embd) + 1e-6f);
                            for (int qi = 0; qi < n_q; ++qi) {
                                double d = 0.0;
                                for (int i = 0; i < n_embd; ++i) {
                                    d += (double) st_v[(size_t) k][(size_t) tok * n_embd + i] * q_v[(size_t) qi * n_embd + i];
                                }
                                sc[(size_t) qi][(size_t) k] = (float) d * inv[(size_t) k] * inv_sqrt;
                            }
                        }
                        for (int qi = 0; qi < n_q; ++qi) {
                            float mx = *std::max_element(sc[(size_t) qi].begin(), sc[(size_t) qi].end());
                            double denom = 0.0;
                            std::vector<double> w((size_t) n_st);
                            for (int k = 0; k < n_st; ++k) {
                                w[(size_t) k] = std::exp((double) (sc[(size_t) qi][(size_t) k] - mx));
                                denom += w[(size_t) k];
                            }
                            for (int i = 0; i < n_embd; ++i) {
                                double acc = 0.0;
                                for (int k = 0; k < n_st; ++k) {
                                    acc += (w[(size_t) k] / denom) * (double) inv[(size_t) k] * st_v[(size_t) k][(size_t) tok * n_embd + i];
                                }
                                ref[((size_t) qi * n_tok + tok) * n_embd + i] = (float) acc;
                            }
                        }
                    }
                    t.assert_true(max_abs_diff(got, ref) < 1e-4f);
                }
            }
        }
    });

    t.test("inkling_ctrl_down & ctrl_apply Q8_0 parity", [&](testing & t) {
        auto down_fn  = reinterpret_cast<ctrl_down_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_ctrl_down_v1"));
        auto apply_fn = reinterpret_cast<ctrl_apply_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_ctrl_apply_v1"));
        t.assert_true(down_fn != nullptr && apply_fn != nullptr);

        const int64_t n_embd = 256, rank = 64, n_tok = 5;
        auto cur_v = rand_vec(n_embd * n_tok, 301);
        auto msg_v = rand_vec(n_embd * n_tok, 302);
        auto wd_raw = rand_vec(2 * n_embd * rank, 303, -0.2f, 0.2f);
        auto wg_raw = rand_vec(rank * n_embd, 304, -0.2f, 0.2f);
        auto wf_raw = rand_vec(rank * n_embd, 305, -0.2f, 0.2f);

        std::vector<uint8_t> wd_q, wg_q, wf_q;
        std::vector<float> wd_ref, wg_ref, wf_ref;
        quantize_and_dequant(GGML_TYPE_Q8_0, wd_raw, rank, 2 * n_embd, wd_q, wd_ref);
        quantize_and_dequant(GGML_TYPE_Q8_0, wg_raw, n_embd, rank, wg_q, wg_ref);
        quantize_and_dequant(GGML_TYPE_Q8_0, wf_raw, n_embd, rank, wf_q, wf_ref);

        ggml_context_ptr ctx(ggml_init({2 * 1024 * 1024, nullptr, true}));
        auto * t_cur = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_tok);
        auto * t_msg = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_tok);
        auto * t_wd  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, 2 * n_embd, rank);
        auto * t_wg  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, rank, n_embd);
        auto * t_wf  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, rank, n_embd);

        auto * t_gate = down_fn(ctx.get(), t_cur, t_msg, t_wd);
        auto * t_out  = apply_fn(ctx.get(), t_cur, t_msg, t_gate, t_wg, t_wf);
        t.assert_true(t_gate != nullptr && t_out != nullptr);
        t.assert_true(ggml_backend_supports_op(backend, t_gate) && ggml_backend_supports_op(backend, t_out));

        std::vector<float> got(n_embd * n_tok);
        run_graph(ctx.get(), t_out, {
            {t_cur, {cur_v.data(), cur_v.size() * sizeof(float)}},
            {t_msg, {msg_v.data(), msg_v.size() * sizeof(float)}},
            {t_wd,  {wd_q.data(),  wd_q.size()}},
            {t_wg,  {wg_q.data(),  wg_q.size()}},
            {t_wf,  {wf_q.data(),  wf_q.size()}},
        }, got.data(), got.size() * sizeof(float));

        std::vector<float> ref(n_embd * n_tok);
        for (int64_t tok = 0; tok < n_tok; ++tok) {
            double sc = 0.0, sm = 0.0;
            for (int64_t i = 0; i < n_embd; ++i) {
                sc += (double) cur_v[tok * n_embd + i] * cur_v[tok * n_embd + i];
                sm += (double) msg_v[tok * n_embd + i] * msg_v[tok * n_embd + i];
            }
            const float ic = 1.0f / std::sqrt((float) (sc / n_embd) + 1e-6f);
            const float im = 1.0f / std::sqrt((float) (sm / n_embd) + 1e-6f);
            std::vector<float> gate(rank, 0.0f);
            for (int64_t r = 0; r < rank; ++r) {
                double ac = 0.0, am = 0.0;
                for (int64_t i = 0; i < n_embd; ++i) {
                    ac += (double) wd_ref[r * 2 * n_embd + i] * cur_v[tok * n_embd + i];
                    am += (double) wd_ref[r * 2 * n_embd + n_embd + i] * msg_v[tok * n_embd + i];
                }
                gate[r] = (float) (ac * ic + am * im);
            }
            for (int64_t i = 0; i < n_embd; ++i) {
                double g = 0.0, f = 0.0;
                for (int64_t r = 0; r < rank; ++r) {
                    g += (double) wg_ref[i * rank + r] * gate[r];
                    f += (double) wf_ref[i * rank + r] * gate[r];
                }
                const float x = cur_v[tok * n_embd + i];
                const float m = msg_v[tok * n_embd + i];
                ref[tok * n_embd + i] = (x - std::tanh((float) f) * x) + std::tanh((float) g) * m;
            }
        }
        t.assert_true(max_abs_diff(got, ref) < 1e-4f);
    });

    t.test("inkling_mul_sigmoid & softcap_trim parity", [&](testing & t) {
        auto ms_fn = reinterpret_cast<mul_sigmoid_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_mul_sigmoid_v1"));
        auto sc_fn = reinterpret_cast<softcap_trim_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_softcap_trim_v1"));
        t.assert_true(ms_fn != nullptr && sc_fn != nullptr);

        // mul_sigmoid
        {
            const int64_t n = 3072 * 5;
            auto x = rand_vec(n, 401, -4.0f, 4.0f);
            auto g = rand_vec(n, 402, -4.0f, 4.0f);
            ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}));
            auto * tx = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3072, 5);
            auto * tg = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3072, 5);
            auto * out = ms_fn(ctx.get(), tx, tg);
            t.assert_true(out != nullptr && ggml_backend_supports_op(backend, out));
            std::vector<float> got(n), ref(n);
            run_graph(ctx.get(), out, {{tx, {x.data(), n * sizeof(float)}}, {tg, {g.data(), n * sizeof(float)}}}, got.data(), n * sizeof(float));
            for (int64_t i = 0; i < n; ++i) { ref[i] = x[i] * (1.0f / (1.0f + std::exp(-g[i]))); }
            t.assert_true(max_abs_diff(got, ref) < 1e-5f);
        }

        // softcap_trim
        {
            const int64_t n_in = 1024, n_out = 997, n_rows = 3;
            const float cap = 30.0f;
            auto src = rand_vec(n_in * n_rows, 403, -60.0f, 60.0f);
            ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}));
            auto * ts = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_in, n_rows);
            auto * out = sc_fn(ctx.get(), ts, n_out, cap);
            t.assert_true(out != nullptr && ggml_backend_supports_op(backend, out));
            std::vector<float> got(n_out * n_rows), ref(n_out * n_rows);
            run_graph(ctx.get(), out, {{ts, {src.data(), src.size() * sizeof(float)}}}, got.data(), got.size() * sizeof(float));
            for (int64_t r = 0; r < n_rows; ++r) {
                for (int64_t c = 0; c < n_out; ++c) {
                    ref[r * n_out + c] = cap * std::tanh(src[r * n_in + c] / cap);
                }
            }
            t.assert_true(max_abs_diff(got, ref) < 1e-5f);
        }
    });

    t.test("inkling_dspark_argmax Q4_0/Q8_0/BF16, shortlist, softcap first-max tie, top1-top2 margin, and all-NaN -> -1", [&](testing & t) {
        auto ready  = reinterpret_cast<dspark_ready_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_dspark_argmax_ready_v2"));
        auto create = reinterpret_cast<dspark_argmax_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_dspark_argmax_v2"));
        t.assert_true(ready != nullptr && create != nullptr);

        // Unsupported rank must return false cleanly
        t.assert_true(!ready(backend, 513, GGML_TYPE_Q8_0));
        t.assert_true(!ready(backend, 48,  GGML_TYPE_Q4_0));

        for (ggml_type w2_type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_BF16}) {
            if (!ready(backend, 128, w2_type)) { continue; }
            const int64_t rank = 128, vocab = 5003;
            auto w2_raw = rand_vec(rank * vocab, 501, -0.3f, 0.3f);
            auto w2_b   = rand_vec(vocab, 502, -0.5f, 0.5f);
            auto base   = rand_vec(vocab, 503, -2.0f, 2.0f);
            auto lat    = rand_vec(rank, 504, -1.0f, 1.0f);

            // Force a softcap-saturation tie: index 777 gets +500, index 2999 gets +900.
            // After cap=30*tanh(s/30), both saturate to exact 30.0f so first-max must pick 777 and margin == 0!
            base[777]  = 500.0f;
            base[2999] = 900.0f;
            // Also inject NaN at index 100 to verify NaN is ignored.
            base[100]  = std::numeric_limits<float>::quiet_NaN();

            std::vector<uint8_t> w2_bytes;
            std::vector<float> w2_ref;
            quantize_and_dequant(w2_type, w2_raw, vocab, rank, w2_bytes, w2_ref);

            // 1. Direct softcap first-max tie test (margin must be 0.0f)
            {
                ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}));
                auto * tw2   = ggml_new_tensor_2d(ctx.get(), w2_type, rank, vocab);
                auto * tw2b  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tbase = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tlat  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, rank);
                auto * out   = create(ctx.get(), tw2, tw2b, tbase, tlat, nullptr, 30.0f);
                t.assert_true(out != nullptr && ggml_backend_supports_op(backend, out));

                int32_t out_ids[3] = {-999, -999, -999};
                run_graph(ctx.get(), out, {
                    {tw2,   {w2_bytes.data(), w2_bytes.size()}},
                    {tw2b,  {w2_b.data(),     w2_b.size() * sizeof(float)}},
                    {tbase, {base.data(),     base.size() * sizeof(float)}},
                    {tlat,  {lat.data(),      lat.size() * sizeof(float)}},
                }, out_ids, sizeof(out_ids));
                float margin = -1.0f;
                std::memcpy(&margin, &out_ids[2], sizeof(float));
                t.assert_true(out_ids[0] == 777 && out_ids[1] == 777 && margin == 0.0f);
            }

            // 1b. Exact top-1/top-2 margin parity against CPU dequant reference (no tie)
            {
                auto base_m = rand_vec(vocab, 513, -2.0f, 2.0f);
                base_m[1234] = 8.5f;
                base_m[3456] = 5.0f;
                base_m[42]   = std::numeric_limits<float>::quiet_NaN();
                double best1 = -1e30, best2 = -1e30;
                int64_t best_idx = -1;
                for (int64_t v = 0; v < vocab; ++v) {
                    if (std::isnan(base_m[v])) { continue; }
                    double dot = 0.0;
                    for (int64_t k = 0; k < rank; ++k) { dot += (double) w2_ref[v * rank + k] * lat[k]; }
                    double s = 30.0 * std::tanh((dot + w2_b[v] + base_m[v]) / 30.0);
                    if (s > best1) { best2 = best1; best1 = s; best_idx = v; }
                    else if (s > best2) { best2 = s; }
                }
                const float ref_margin = (float) std::max(0.0, best1 - best2);

                ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}));
                auto * tw2   = ggml_new_tensor_2d(ctx.get(), w2_type, rank, vocab);
                auto * tw2b  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tbase = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tlat  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, rank);
                auto * out   = create(ctx.get(), tw2, tw2b, tbase, tlat, nullptr, 30.0f);
                t.assert_true(out != nullptr && ggml_backend_supports_op(backend, out));

                int32_t out_ids[3] = {-999, -999, -999};
                run_graph(ctx.get(), out, {
                    {tw2,   {w2_bytes.data(), w2_bytes.size()}},
                    {tw2b,  {w2_b.data(),     w2_b.size() * sizeof(float)}},
                    {tbase, {base_m.data(),   base_m.size() * sizeof(float)}},
                    {tlat,  {lat.data(),      lat.size() * sizeof(float)}},
                }, out_ids, sizeof(out_ids));
                float got_margin = -1.0f;
                std::memcpy(&got_margin, &out_ids[2], sizeof(float));
                t.assert_true(out_ids[0] == best_idx && out_ids[1] == best_idx && std::fabs(got_margin - ref_margin) < 1e-3f);
            }

            // 2. Shortlist remapping test
            {
                std::vector<int32_t> sl(vocab);
                for (int64_t i = 0; i < vocab; ++i) { sl[i] = (int32_t) (i * 3 + 11); }
                ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}));
                auto * tw2   = ggml_new_tensor_2d(ctx.get(), w2_type, rank, vocab);
                auto * tw2b  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tbase = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tlat  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, rank);
                auto * tsl   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, vocab);
                auto * out   = create(ctx.get(), tw2, tw2b, tbase, tlat, tsl, 30.0f);
                t.assert_true(out != nullptr && ggml_backend_supports_op(backend, out));

                int32_t out_ids[3] = {-999, -999, -999};
                run_graph(ctx.get(), out, {
                    {tw2,   {w2_bytes.data(), w2_bytes.size()}},
                    {tw2b,  {w2_b.data(),     w2_b.size() * sizeof(float)}},
                    {tbase, {base.data(),     base.size() * sizeof(float)}},
                    {tlat,  {lat.data(),      lat.size() * sizeof(float)}},
                    {tsl,   {sl.data(),       sl.size() * sizeof(int32_t)}},
                }, out_ids, sizeof(out_ids));
                t.assert_true(out_ids[0] == 777 * 3 + 11 && out_ids[1] == 777 * 3 + 11);
            }

            // 3. All-NaN logits must return raw=-1 (stop host drafting) and safe=0 (clamped for chained get_rows)
            {
                std::vector<float> nan_base(vocab, std::numeric_limits<float>::quiet_NaN());
                ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}));
                auto * tw2   = ggml_new_tensor_2d(ctx.get(), w2_type, rank, vocab);
                auto * tw2b  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tbase = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tlat  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, rank);
                auto * out   = create(ctx.get(), tw2, tw2b, tbase, tlat, nullptr, 30.0f);
                t.assert_true(out != nullptr);

                int32_t out_ids[3] = {12345, 12345, 12345};
                run_graph(ctx.get(), out, {
                    {tw2,   {w2_bytes.data(), w2_bytes.size()}},
                    {tw2b,  {w2_b.data(),     w2_b.size() * sizeof(float)}},
                    {tbase, {nan_base.data(), nan_base.size() * sizeof(float)}},
                    {tlat,  {lat.data(),      lat.size() * sizeof(float)}},
                }, out_ids, sizeof(out_ids));
                t.assert_true(out_ids[0] == -1 && out_ids[1] == 0 && out_ids[2] == 0);
            }

            // 4. 2-step chained DSpark graph where step 0 has all-NaN logits -> elem 1 (0) feeds ggml_get_rows safely
            {
                std::vector<float> nan_base(vocab, std::numeric_limits<float>::quiet_NaN());
                auto w1_f32 = rand_vec(rank * vocab, 505, -0.5f, 0.5f);
                ggml_context_ptr ctx(ggml_init({2 * 1024 * 1024, nullptr, true}));
                auto * tw1    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, rank, vocab);
                auto * tw2    = ggml_new_tensor_2d(ctx.get(), w2_type, rank, vocab);
                auto * tw2b   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tbase0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tbase1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tlat0  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, rank);

                auto * step0  = create(ctx.get(), tw2, tw2b, tbase0, tlat0, nullptr, 30.0f);
                auto * raw0   = ggml_view_1d(ctx.get(), step0, 1, 0);
                auto * safe0  = ggml_view_1d(ctx.get(), step0, 1, sizeof(int32_t));
                // Step 1 reads row `safe0` (0, never -1) from w1 -> must not OOB-fault under MTL_SHADER_VALIDATION=1
                auto * row1   = ggml_get_rows(ctx.get(), tw1, safe0);
                auto * lat1   = ggml_reshape_1d(ctx.get(), row1, rank);
                auto * step1  = create(ctx.get(), tw2, tw2b, tbase1, lat1, nullptr, 30.0f);
                auto * raw1   = ggml_view_1d(ctx.get(), step1, 1, 0);
                auto * ids    = ggml_concat(ctx.get(), raw0, raw1, 0);
                t.assert_true(step0 != nullptr && step1 != nullptr && ids != nullptr);

                int32_t chained[2] = {999, 999};
                run_graph(ctx.get(), ids, {
                    {tw1,    {w1_f32.data(),   w1_f32.size() * sizeof(float)}},
                    {tw2,    {w2_bytes.data(), w2_bytes.size()}},
                    {tw2b,   {w2_b.data(),     w2_b.size() * sizeof(float)}},
                    {tbase0, {nan_base.data(), nan_base.size() * sizeof(float)}},
                    {tbase1, {base.data(),     base.size() * sizeof(float)}},
                    {tlat0,  {lat.data(),      lat.size() * sizeof(float)}},
                }, chained, sizeof(chained));
                t.assert_true(chained[0] == -1 && chained[1] == 777);
            }
        }
    });

    t.test("inkling_flash_attn_v2 gated decode (nwg=1 & nwg=4, f16/q8_0/q4_0)", [&](testing & t) {
        auto create_gated = reinterpret_cast<fa_gated_create_fn>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_flash_attn_v2"));
        t.assert_true(create_gated != nullptr);

        for (ggml_type kv_type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
            for (auto [n_q, n_kv] : std::vector<std::pair<int64_t, int64_t>>{{1, 256}, {5, 1024}, {8, 1024}}) {
                const int32_t extent = 64;
                const int32_t n_swa = 0;
                const float scale = 1.0f / std::sqrt(128.0f);
                auto q_v   = rand_vec(128 * 24 * n_q, 601, -0.5f, 0.5f);
                auto k_raw = rand_vec(128 * 8 * n_kv, 602, -0.5f, 0.5f);
                auto v_raw = rand_vec(128 * 8 * n_kv, 603, -0.5f, 0.5f);
                auto tbl_v = rand_vec(extent * 24 * n_q, 604, -0.5f, 0.5f);
                auto snk_v = rand_vec(24 * n_q, 605, -0.5f, 0.5f);
                auto gat_v = rand_vec(3072 * n_q, 606, -3.0f, 3.0f);

                std::vector<int32_t> q_pos(2 * n_q);
                for (int64_t i = 0; i < n_q; ++i) {
                    q_pos[2 * i + 0] = (int32_t) (n_kv - n_q + i);
                    q_pos[2 * i + 1] = 0;
                }
                std::vector<int32_t> k_pos(3 * n_kv);
                for (int64_t i = 0; i < n_kv; ++i) {
                    k_pos[3 * i + 0] = (int32_t) i;
                    k_pos[3 * i + 1] = 1;
                    k_pos[3 * i + 2] = 0;
                }

                std::vector<uint8_t> k_bytes, v_bytes;
                std::vector<float> k_ref, v_ref;
                quantize_and_dequant(kv_type, k_raw, 8 * n_kv, 128, k_bytes, k_ref);
                quantize_and_dequant(kv_type, v_raw, 8 * n_kv, 128, v_bytes, v_ref);

                ggml_context_ptr ctx(ggml_init({2 * 1024 * 1024, nullptr, true}));
                auto * tq   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 24, n_q);
                auto * tk   = ggml_new_tensor_3d(ctx.get(), kv_type, 128, 8, n_kv);
                auto * tv   = ggml_new_tensor_3d(ctx.get(), kv_type, 128, 8, n_kv);
                auto * ttbl = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, extent, 24, n_q);
                auto * tsnk = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 24, n_q);
                auto * tqp  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, n_q);
                auto * tkp  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 3, n_kv);
                auto * tgat = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3072, n_q);
                auto * out  = create_gated(ctx.get(), tq, tk, tv, ttbl, tsnk, tqp, tkp, tgat, scale, extent, n_swa);
                t.assert_true(out != nullptr && ggml_backend_supports_op(backend, out));

                std::vector<float> got(3072 * n_q);
                run_graph(ctx.get(), out, {
                    {tq,   {q_v.data(),     q_v.size() * sizeof(float)}},
                    {tk,   {k_bytes.data(), k_bytes.size()}},
                    {tv,   {v_bytes.data(), v_bytes.size()}},
                    {ttbl, {tbl_v.data(),   tbl_v.size() * sizeof(float)}},
                    {tsnk, {snk_v.data(),   snk_v.size() * sizeof(float)}},
                    {tqp,  {q_pos.data(),   q_pos.size() * sizeof(int32_t)}},
                    {tkp,  {k_pos.data(),   k_pos.size() * sizeof(int32_t)}},
                    {tgat, {gat_v.data(),   gat_v.size() * sizeof(float)}},
                }, got.data(), got.size() * sizeof(float));

                auto ref = cpu_ref_inkling_fa(q_v, k_ref, v_ref, tbl_v, snk_v, q_pos, k_pos, n_q, n_kv, extent, n_swa, scale);
                for (size_t i = 0; i < ref.size(); ++i) {
                    ref[i] *= 1.0f / (1.0f + std::exp(-gat_v[i]));
                }
                const float tol = kv_type == GGML_TYPE_F16 ? 2e-3f : 5e-3f;
                t.assert_true(max_abs_diff(got, ref) < tol);
            }
        }
    });

    t.test("inkling_mix_norm_v1 parity (n_q=1 and n_q=3, n_embd=3072)", [&](testing & t) {
        auto create = reinterpret_cast<mix_norm_create_fn>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_mix_norm_v1"));
        t.assert_true(create != nullptr);

        for (auto [n_st, n_q] : std::vector<std::pair<int, int>>{{5, 1}, {4, 3}}) {
            const int64_t n_embd = 3072, n_tok = 4;
            const float norm_eps = 1e-6f;
            std::vector<std::vector<float>> st_v(n_st);
            for (int k = 0; k < n_st; ++k) {
                st_v[k] = rand_vec(n_embd * n_tok, 700 + k, -2.0f, 2.0f);
            }
            auto q_v = rand_vec(n_embd * n_q, 750, -0.5f, 0.5f);
            std::vector<std::vector<float>> nw_v(n_q);
            for (int q = 0; q < n_q; ++q) {
                nw_v[q] = rand_vec(n_embd, 760 + q, 0.5f, 1.5f);
            }

            ggml_context_ptr ctx(ggml_init({2 * 1024 * 1024, nullptr, true}));
            std::vector<ggml_tensor *> st_t(n_st), nw_t(n_q);
            std::vector<std::pair<ggml_tensor *, std::pair<const void *, size_t>>> uploads;
            for (int k = 0; k < n_st; ++k) {
                st_t[k] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_tok);
                uploads.push_back({st_t[k], {st_v[k].data(), st_v[k].size() * sizeof(float)}});
            }
            auto * tq = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_q);
            uploads.push_back({tq, {q_v.data(), q_v.size() * sizeof(float)}});
            for (int q = 0; q < n_q; ++q) {
                nw_t[q] = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_embd);
                uploads.push_back({nw_t[q], {nw_v[q].data(), nw_v[q].size() * sizeof(float)}});
            }
            auto * out = create(ctx.get(), st_t.data(), n_st, tq, nw_t.data(), norm_eps);
            t.assert_true(out != nullptr && ggml_backend_supports_op(backend, out));

            std::vector<float> got(n_embd * n_tok * n_q);
            run_graph(ctx.get(), out, uploads, got.data(), got.size() * sizeof(float));

            // CPU reference
            std::vector<float> ref(n_embd * n_tok * n_q, 0.0f);
            const float inv_sqrt_d = 1.0f / std::sqrt((float) n_embd);
            for (int64_t tok = 0; tok < n_tok; ++tok) {
                std::vector<std::vector<float>> normed(n_st, std::vector<float>(n_embd));
                for (int k = 0; k < n_st; ++k) {
                    double sq = 0.0;
                    for (int64_t i = 0; i < n_embd; ++i) {
                        double v = st_v[k][tok * n_embd + i];
                        sq += v * v;
                    }
                    float inv = 1.0f / std::sqrt((float) (sq / n_embd) + 1e-6f);
                    for (int64_t i = 0; i < n_embd; ++i) { normed[k][i] = st_v[k][tok * n_embd + i] * inv; }
                }
                for (int q = 0; q < n_q; ++q) {
                    std::vector<float> sc(n_st);
                    float mx = -1e30f;
                    for (int k = 0; k < n_st; ++k) {
                        double d = 0.0;
                        for (int64_t i = 0; i < n_embd; ++i) { d += (double) normed[k][i] * (double) q_v[q * n_embd + i]; }
                        sc[k] = (float) d * inv_sqrt_d;
                        mx = std::max(mx, sc[k]);
                    }
                    float sum_e = 0.0f;
                    for (int k = 0; k < n_st; ++k) { sc[k] = std::exp(sc[k] - mx); sum_e += sc[k]; }
                    std::vector<float> mixed(n_embd, 0.0f);
                    double sq = 0.0;
                    for (int64_t i = 0; i < n_embd; ++i) {
                        double acc = 0.0;
                        for (int k = 0; k < n_st; ++k) { acc += (double) (sc[k] / sum_e) * (double) normed[k][i]; }
                        mixed[i] = (float) acc;
                        sq += acc * acc;
                    }
                    float inv_n = 1.0f / std::sqrt((float) (sq / n_embd) + norm_eps);
                    for (int64_t i = 0; i < n_embd; ++i) {
                        ref[(q * n_tok + tok) * n_embd + i] = mixed[i] * inv_n * nw_v[q][i];
                    }
                }
            }
            t.assert_true(max_abs_diff(got, ref) < 1e-4f);
        }
    });

    t.test("inkling_conv_fused_v1 parity (rollback slots, dynamic taps, residual add, n_seqs=2)", [&](testing & t) {
        auto create = reinterpret_cast<conv_fused_fn>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_conv_fused_v1"));
        t.assert_true(create != nullptr);

        // Test 1: n_seqs=2, n_seq_tokens=5, kw=5 (rollback slots 0..4), dynamic taps + residual add + non-zero ch_offset & head
        {
            const int32_t channels = 256, hist_ch = 512, ch_offset = 128;
            const int32_t n_seq_tokens = 5, n_seqs = 2, n_tokens = n_seq_tokens * n_seqs;
            const int32_t n_heads = 4, kw = 5, head = 1, n_cells = 4, K_slots = 6;
            const int32_t row_floats = hist_ch * 3;
            const float   scale = 1.0f / std::sqrt(9.0f);

            auto cur_v  = rand_vec(channels * n_tokens, 801, -1.0f, 1.0f);
            auto w_v    = rand_vec(4 * channels, 802, -0.5f, 0.5f);
            auto hist_v = rand_vec(3 * hist_ch * n_seqs, 803, -1.0f, 1.0f);
            auto dyn_v  = rand_vec(4 * n_heads * n_tokens, 804, -0.5f, 0.5f);
            auto res_v  = rand_vec(channels * n_tokens, 805, -1.0f, 1.0f);
            std::vector<float> state_init(row_floats * n_cells * K_slots, -7.25f);
            std::vector<float> state_ref = state_init;

            ggml_context_ptr ctx(ggml_init({4 * 1024 * 1024, nullptr, true}));
            auto * tcur  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, channels, n_tokens);
            auto * tw    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, channels);
            auto * thist = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 3, hist_ch, n_seqs);
            auto * tst   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, row_floats, n_cells * K_slots);
            auto * tdyn  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4 * n_heads, n_tokens);
            auto * tres  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, channels, n_tokens);

            auto * out = create(ctx.get(), tcur, tw, thist, tst, tdyn, tres,
                    ch_offset, n_seq_tokens, n_seqs, kw, head, n_cells, scale);
            t.assert_true(out != nullptr && ggml_backend_supports_op(backend, out));

            ggml_cgraph * gf = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(gf, out);
            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            t.assert_true(ggml_gallocr_alloc_graph(allocr, gf));
            ggml_backend_tensor_set(tcur,  cur_v.data(),      0, cur_v.size() * sizeof(float));
            ggml_backend_tensor_set(tw,    w_v.data(),        0, w_v.size() * sizeof(float));
            ggml_backend_tensor_set(thist, hist_v.data(),     0, hist_v.size() * sizeof(float));
            ggml_backend_tensor_set(tst,   state_init.data(), 0, state_init.size() * sizeof(float));
            ggml_backend_tensor_set(tdyn,  dyn_v.data(),      0, dyn_v.size() * sizeof(float));
            ggml_backend_tensor_set(tres,  res_v.data(),      0, res_v.size() * sizeof(float));
            t.assert_true(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

            std::vector<float> got_out(channels * n_tokens);
            std::vector<float> got_state(state_init.size());
            ggml_backend_tensor_get(out, got_out.data(),   0, got_out.size() * sizeof(float));
            ggml_backend_tensor_get(tst, got_state.data(), 0, got_state.size() * sizeof(float));
            ggml_gallocr_free(allocr);

            // CPU reference matching build_conv_state_store + convolve + add(residual)
            std::vector<float> ref_out(channels * n_tokens);
            for (int32_t s = 0; s < n_seqs; ++s) {
                for (int32_t ch = 0; ch < channels; ++ch) {
                    const int32_t h_idx = ch / (channels / n_heads);
                    const size_t hoff = ((size_t) s * hist_ch + (size_t) (ch_offset + ch)) * 3u;
                    float h0 = hist_v[hoff + 0], h1 = hist_v[hoff + 1], h2 = hist_v[hoff + 2];
                    for (int32_t step = 0; step < n_seq_tokens; ++step) {
                        const int32_t tok = s * n_seq_tokens + step;
                        const size_t idx = (size_t) tok * channels + ch;
                        const float xt = cur_v[idx];
                        float acc = h0 * w_v[ch * 4 + 0] + h1 * w_v[ch * 4 + 1] + h2 * w_v[ch * 4 + 2] + xt * w_v[ch * 4 + 3];
                        const size_t dbase = (size_t) tok * (4 * n_heads) + (size_t) h_idx * 4;
                        acc += xt * dyn_v[dbase + 0] + h2 * dyn_v[dbase + 1] + h1 * dyn_v[dbase + 2] + h0 * dyn_v[dbase + 3];
                        ref_out[idx] = (xt + acc) * scale + res_v[idx];
                        h0 = h1; h1 = h2; h2 = xt;
                        if (step + kw >= n_seq_tokens) {
                            const int32_t s_idx = (n_seq_tokens - 1) - step;
                            const size_t dst_st = ((size_t) s_idx * n_cells + (size_t) head + (size_t) s) * row_floats
                                    + (size_t) (ch_offset + ch) * 3u;
                            state_ref[dst_st + 0] = h0;
                            state_ref[dst_st + 1] = h1;
                            state_ref[dst_st + 2] = h2;
                        }
                    }
                }
            }
            t.assert_true(max_abs_diff(got_out, ref_out) < 1e-5f);
            t.assert_true(max_abs_diff(got_state, state_ref) == 0.0f);
        }

        // Test 2: no dynamic taps, WITH residual (tests packed src[4]=residual without null hole!), kw=1, n_seq_tokens=1
        {
            const int32_t channels = 128, hist_ch = 256, ch_offset = 0;
            const int32_t n_seq_tokens = 1, n_seqs = 1, kw = 1, head = 0, n_cells = 1;
            const int32_t row_floats = hist_ch * 3;
            const float   scale = 1.0f / std::sqrt(5.0f);
            auto cur_v  = rand_vec(channels, 811, -1.0f, 1.0f);
            auto w_v    = rand_vec(4 * channels, 812, -0.5f, 0.5f);
            auto hist_v = rand_vec(3 * hist_ch, 813, -1.0f, 1.0f);
            auto res_v  = rand_vec(channels, 814, -1.0f, 1.0f);
            std::vector<float> state_init(row_floats, 0.0f);

            ggml_context_ptr ctx(ggml_init({2 * 1024 * 1024, nullptr, true}));
            auto * tcur  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, channels, 1);
            auto * tw    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, channels);
            auto * thist = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 3, hist_ch, 1);
            auto * tst   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, row_floats, 1);
            auto * tres  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, channels, 1);

            auto * out = create(ctx.get(), tcur, tw, thist, tst, nullptr, tres,
                    ch_offset, n_seq_tokens, n_seqs, kw, head, n_cells, scale);
            t.assert_true(out != nullptr && ggml_backend_supports_op(backend, out));

            std::vector<float> got_out(channels);
            run_graph(ctx.get(), out, {
                {tcur,  {cur_v.data(),      cur_v.size() * sizeof(float)}},
                {tw,    {w_v.data(),        w_v.size() * sizeof(float)}},
                {thist, {hist_v.data(),     hist_v.size() * sizeof(float)}},
                {tst,   {state_init.data(), state_init.size() * sizeof(float)}},
                {tres,  {res_v.data(),      res_v.size() * sizeof(float)}},
            }, got_out.data(), got_out.size() * sizeof(float));

            std::vector<float> ref_out(channels);
            for (int32_t ch = 0; ch < channels; ++ch) {
                float acc = hist_v[ch * 3 + 0] * w_v[ch * 4 + 0] + hist_v[ch * 3 + 1] * w_v[ch * 4 + 1]
                          + hist_v[ch * 3 + 2] * w_v[ch * 4 + 2] + cur_v[ch] * w_v[ch * 4 + 3];
                ref_out[ch] = (cur_v[ch] + acc) * scale + res_v[ch];
            }
            t.assert_true(max_abs_diff(got_out, ref_out) < 1e-5f);
        }

        // Test 3: conv_fused_v2 with fused 8-expert weighted reduction + residual add + rollback slots (kw=4, n_seq_tokens=4)
        {
            auto create_v2 = reinterpret_cast<conv_fused_v2_fn>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_conv_fused_v2"));
            t.assert_true(create_v2 != nullptr);

            const int32_t channels = 256, n_exp = 8, hist_ch = 512, ch_offset = 64;
            const int32_t n_seq_tokens = 4, n_seqs = 1, n_tokens = 4, kw = 4, head = 0, n_cells = 2;
            const int32_t row_floats = hist_ch * 3;
            const float   scale = 1.0f / std::sqrt(5.0f);

            auto exp_v  = rand_vec(channels * n_exp * n_tokens, 821, -1.0f, 1.0f);
            auto ew_v   = rand_vec(n_exp * n_tokens, 822, 0.0f, 0.5f);
            auto w_v    = rand_vec(4 * channels, 823, -0.5f, 0.5f);
            auto hist_v = rand_vec(3 * hist_ch, 824, -1.0f, 1.0f);
            auto res_v  = rand_vec(channels * n_tokens, 825, -1.0f, 1.0f);
            std::vector<float> state_init(row_floats * n_cells * 6, -3.5f);
            std::vector<float> state_ref = state_init;

            ggml_context_ptr ctx(ggml_init({4 * 1024 * 1024, nullptr, true}));
            auto * texp  = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, channels, n_exp, n_tokens);
            auto * tew   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, n_exp, n_tokens);
            auto * tw    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, channels);
            auto * thist = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 3, hist_ch, n_seqs);
            auto * tst   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, row_floats, n_cells * 6);
            auto * tres  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, channels, n_tokens);

            auto * out = create_v2(ctx.get(), texp, tw, thist, tst, nullptr, tres, tew,
                    ch_offset, n_seq_tokens, n_seqs, kw, head, n_cells, scale);
            t.assert_true(out != nullptr && ggml_backend_supports_op(backend, out));

            ggml_cgraph * gf = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(gf, out);
            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            t.assert_true(ggml_gallocr_alloc_graph(allocr, gf));
            ggml_backend_tensor_set(texp,  exp_v.data(),      0, exp_v.size() * sizeof(float));
            ggml_backend_tensor_set(tew,   ew_v.data(),       0, ew_v.size() * sizeof(float));
            ggml_backend_tensor_set(tw,    w_v.data(),        0, w_v.size() * sizeof(float));
            ggml_backend_tensor_set(thist, hist_v.data(),     0, hist_v.size() * sizeof(float));
            ggml_backend_tensor_set(tst,   state_init.data(), 0, state_init.size() * sizeof(float));
            ggml_backend_tensor_set(tres,  res_v.data(),      0, res_v.size() * sizeof(float));
            t.assert_true(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

            std::vector<float> got_out(channels * n_tokens), got_state(state_init.size());
            ggml_backend_tensor_get(out, got_out.data(),   0, got_out.size() * sizeof(float));
            ggml_backend_tensor_get(tst, got_state.data(), 0, got_state.size() * sizeof(float));
            ggml_gallocr_free(allocr);

            std::vector<float> ref_out(channels * n_tokens);
            for (int32_t ch = 0; ch < channels; ++ch) {
                const size_t hoff = (size_t) (ch_offset + ch) * 3u;
                float h0 = hist_v[hoff + 0], h1 = hist_v[hoff + 1], h2 = hist_v[hoff + 2];
                for (int32_t step = 0; step < n_seq_tokens; ++step) {
                    float xt = 0.0f;
                    for (int32_t e = 0; e < n_exp; ++e) {
                        xt += exp_v[((size_t) step * n_exp + (size_t) e) * channels + ch] * ew_v[(size_t) step * n_exp + (size_t) e];
                    }
                    float acc = h0 * w_v[ch * 4 + 0] + h1 * w_v[ch * 4 + 1] + h2 * w_v[ch * 4 + 2] + xt * w_v[ch * 4 + 3];
                    ref_out[(size_t) step * channels + ch] = (xt + acc) * scale + res_v[(size_t) step * channels + ch];
                    h0 = h1; h1 = h2; h2 = xt;
                    const int32_t s_idx = (n_seq_tokens - 1) - step;
                    const size_t dst_st = ((size_t) s_idx * n_cells + (size_t) head) * row_floats + (size_t) (ch_offset + ch) * 3u;
                    state_ref[dst_st + 0] = h0;
                    state_ref[dst_st + 1] = h1;
                    state_ref[dst_st + 2] = h2;
                }
            }
            t.assert_true(max_abs_diff(got_out, ref_out) < 1e-5f);
            t.assert_true(max_abs_diff(got_state, state_ref) < 1e-5f);
        }
    });

    t.test("inkling_moe_ffn_v1 coalesced parity across Q4_0/Q4_K/IQ2_XXS + Q4_0/Q5_0/Q2_0/IQ4_NL (top-6 & top-8, M=1 & M=6)", [&](testing & t) {
        auto create = reinterpret_cast<moe_ffn_fn>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_moe_ffn_v1"));
        t.assert_true(create != nullptr);

        const int64_t n_embd = 256, n_ff = 128, n_exp = 16;
        const float beta = 2.0f, linear_beta = 4.0f;
        const std::vector<std::pair<ggml_type, ggml_type>> combos = {
            { GGML_TYPE_Q4_0,    GGML_TYPE_Q4_0   },
            { GGML_TYPE_Q4_K,    GGML_TYPE_Q5_0   },
            { GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_0   },
            { GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_NL },
        };

        for (auto [gu_type, dn_type] : combos) {
            auto gu_raw = rand_vec(n_embd * (2 * n_ff) * n_exp, 901, -0.3f, 0.3f);
            auto dn_raw = rand_vec(n_ff * n_embd * n_exp, 902, -0.3f, 0.3f);
            std::vector<uint8_t> gu_bytes, dn_bytes;
            std::vector<float> gu_ref, dn_ref;
            quantize_and_dequant(gu_type, gu_raw, (2 * n_ff) * n_exp, n_embd, gu_bytes, gu_ref);
            quantize_and_dequant(dn_type, dn_raw, n_embd * n_exp, n_ff, dn_bytes, dn_ref);

            for (int64_t top_k : {6, 8}) {
                for (int64_t n_tok : {1, 6}) {
                    for (bool with_skipped : {false, true}) {
                    auto x_v = rand_vec(n_embd * n_tok, 910 + (uint32_t)(n_tok * 10 + top_k + (with_skipped ? 100 : 0)), -1.0f, 1.0f);
                    std::vector<int32_t> ids_v((size_t)(top_k * n_tok));
                    for (int64_t tok = 0; tok < n_tok; ++tok) {
                        std::vector<bool> used(n_exp, false);
                        for (int64_t k = 0; k < top_k; ++k) {
                            if (with_skipped && ((tok == 0 && k >= top_k - 2) || ((tok & 1) == 1 && k == top_k - 1))) {
                                ids_v[(size_t)(tok * top_k + k)] = -1;
                                continue;
                            }
                            int32_t e = (int32_t)((tok * 3 + k * 2 + (k == 0 ? 0 : 1)) % n_exp);
                            while (used[e]) { e = (e + 1) % (int32_t)n_exp; }
                            used[e] = true;
                            ids_v[(size_t)(tok * top_k + k)] = e;
                        }
                    }

                    ggml_context_ptr ctx(ggml_init({4 * 1024 * 1024, nullptr, true}));
                    auto * tx   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_embd, 1, n_tok);
                    auto * tids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, top_k, n_tok);
                    auto * tgu  = ggml_new_tensor_3d(ctx.get(), gu_type, n_embd, 2 * n_ff, n_exp);
                    auto * tdn  = ggml_new_tensor_3d(ctx.get(), dn_type, n_ff, n_embd, n_exp);
                    auto * out  = create(ctx.get(), tx, tids, tgu, tdn, beta, linear_beta);
                    t.assert_true(out != nullptr && ggml_backend_supports_op(backend, out));

                    std::vector<float> got((size_t)(n_embd * top_k * n_tok), -999.0f);
                    run_graph(ctx.get(), out, {
                        {tx,   {x_v.data(),      x_v.size() * sizeof(float)}},
                        {tids, {ids_v.data(),    ids_v.size() * sizeof(int32_t)}},
                        {tgu,  {gu_bytes.data(), gu_bytes.size()}},
                        {tdn,  {dn_bytes.data(), dn_bytes.size()}},
                    }, got.data(), got.size() * sizeof(float));

                    std::vector<float> ref((size_t)(n_embd * top_k * n_tok), 0.0f);
                    std::vector<float> h(n_ff);
                    for (int64_t tok = 0; tok < n_tok; ++tok) {
                        const float * xp = x_v.data() + tok * n_embd;
                        for (int64_t k = 0; k < top_k; ++k) {
                            const int64_t e = ids_v[(size_t)(tok * top_k + k)];
                            if (e < 0) { continue; }
                            const float * gup = gu_ref.data() + e * (2 * n_ff) * n_embd;
                            for (int64_t r = 0; r < n_ff; ++r) {
                                double g = 0.0, u = 0.0;
                                const float * wg = gup + r * n_embd;
                                const float * wu = gup + (r + n_ff) * n_embd;
                                for (int64_t i = 0; i < n_embd; ++i) {
                                    g += (double)wg[i] * (double)xp[i];
                                    u += (double)wu[i] * (double)xp[i];
                                }
                                const float gf = (float)g, uf = (float)u;
                                const float act = (beta * std::tanh(gf / beta)) * (1.0f / (1.0f + std::exp(-gf))) *
                                        (linear_beta * std::tanh(uf / linear_beta));
                                h[r] = act;
                            }
                            const float * dnp = dn_ref.data() + e * n_embd * n_ff;
                            float * dp = ref.data() + (tok * top_k + k) * n_embd;
                            for (int64_t r = 0; r < n_embd; ++r) {
                                double acc = 0.0;
                                const float * wd = dnp + r * n_ff;
                                for (int64_t i = 0; i < n_ff; ++i) { acc += (double)wd[i] * (double)h[i]; }
                                dp[r] = (float)acc;
                            }
                        }
                    }
                    const float diff = max_abs_diff(got, ref);
                    t.assert_true(diff < 5e-4f);
                    }
                }
            }
        }
    });

    return t.summary();
}
