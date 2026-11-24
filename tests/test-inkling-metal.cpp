// Per-op Metal parity & safety tests for Inkling2 fused custom kernels.
// Exercises every custom Metal op against a pure C++ F32 reference implementation
// (run with MTL_SHADER_VALIDATION=1 to verify zero misaligned/OOB GPU accesses).
#include "testing.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-metal.h"

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
using mix_create_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor * const *, int, ggml_tensor *);
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
        ggml_quantize_chunk(type, src.data(), qbytes.data(), 0, n_rows, n_cols, nullptr);
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

    ggml_backend_ptr metal(ggml_backend_metal_init());
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
            ggml_type kv_type;
            int64_t n_q;
            int64_t n_kv;
            int32_t extent;
            int32_t n_swa;
            int32_t n_seqs;
            bool include_all_masked;
        };
        const std::vector<fa_case> cases = {
            // Decode (n_q < 16), nwg=1 (n_kv=64) and split-K nwg=4 (n_kv=640) & nwg=16 (n_kv=4160)
            {"dec_q1_f16_nwg1",     GGML_TYPE_F16,  1,   64,  32,    0, 1, false},
            {"dec_q3_f16_split4",   GGML_TYPE_F16,  3,  640,  64,    0, 1, false},
            {"dec_q4_q8_split4",    GGML_TYPE_Q8_0, 4,  640,  64,  256, 1, false},
            {"dec_q3_q4_split16",   GGML_TYPE_Q4_0, 3, 4160,  64,    0, 1, false},
            {"dec_q15_f16_multiseq",GGML_TYPE_F16, 15,  640,  64,    0, 4, true },
            // Prefill (n_q >= 16, partial tiles n_q=17, 48), all-far tiles + SWA + multi-seq
            {"pre_q17_f16_global",  GGML_TYPE_F16, 17,  640,  64,    0, 1, false},
            {"pre_q48_f16_multiseq",GGML_TYPE_F16, 48,  640,  64,    0, 3, true },
            {"pre_q17_q8_swa",      GGML_TYPE_Q8_0,17,  640,  64,  128, 2, false},
            {"pre_q17_q4_global",   GGML_TYPE_Q4_0,17,  640,  64,    0, 2, false},
        };

        const float scale = 1.0f / std::sqrt(128.0f);
        uint32_t seed = 100;
        for (const auto & tc : cases) {
            if (!ready(backend, tc.n_q, tc.kv_type)) {
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
            quantize_and_dequant(tc.kv_type, k_raw, tc.n_kv * 8, 128, k_bytes, k_ref);
            quantize_and_dequant(tc.kv_type, v_raw, tc.n_kv * 8, 128, v_bytes, v_ref);

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
            auto * t_k     = ggml_new_tensor_3d(ctx.get(), tc.kv_type, 128, 8, tc.n_kv);
            auto * t_v     = ggml_new_tensor_3d(ctx.get(), tc.kv_type, 128, 8, tc.n_kv);
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

    t.test("inkling_dspark_argmax Q8_0/BF16, shortlist, softcap first-max tie, and all-NaN -> -1", [&](testing & t) {
        auto ready  = reinterpret_cast<dspark_ready_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_dspark_argmax_ready_v1"));
        auto create = reinterpret_cast<dspark_argmax_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_dspark_argmax_v1"));
        t.assert_true(ready != nullptr && create != nullptr);

        // Unsupported rank must return false cleanly
        t.assert_true(!ready(backend, 513, GGML_TYPE_Q8_0));
        t.assert_true(!ready(backend, 48,  GGML_TYPE_Q8_0));

        for (ggml_type w2_type : {GGML_TYPE_Q8_0, GGML_TYPE_BF16}) {
            if (!ready(backend, 128, w2_type)) { continue; }
            const int64_t rank = 128, vocab = 5003;
            auto w2_raw = rand_vec(rank * vocab, 501, -0.3f, 0.3f);
            auto w2_b   = rand_vec(vocab, 502, -0.5f, 0.5f);
            auto base   = rand_vec(vocab, 503, -2.0f, 2.0f);
            auto lat    = rand_vec(rank, 504, -1.0f, 1.0f);

            // Force a softcap-saturation tie: index 777 gets +500, index 2999 gets +900.
            // After cap=30*tanh(s/30), both saturate to exact 30.0f so first-max must pick 777!
            base[777]  = 500.0f;
            base[2999] = 900.0f;
            // Also inject NaN at index 100 to verify NaN is ignored.
            base[100]  = std::numeric_limits<float>::quiet_NaN();

            std::vector<uint8_t> w2_bytes;
            std::vector<float> w2_ref;
            quantize_and_dequant(w2_type, w2_raw, vocab, rank, w2_bytes, w2_ref);

            // 1. Direct softcap first-max tie test
            {
                ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}));
                auto * tw2   = ggml_new_tensor_2d(ctx.get(), w2_type, rank, vocab);
                auto * tw2b  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tbase = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tlat  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, rank);
                auto * out   = create(ctx.get(), tw2, tw2b, tbase, tlat, nullptr, 30.0f);
                t.assert_true(out != nullptr && ggml_backend_supports_op(backend, out));

                int32_t winner = -999;
                run_graph(ctx.get(), out, {
                    {tw2,   {w2_bytes.data(), w2_bytes.size()}},
                    {tw2b,  {w2_b.data(),     w2_b.size() * sizeof(float)}},
                    {tbase, {base.data(),     base.size() * sizeof(float)}},
                    {tlat,  {lat.data(),      lat.size() * sizeof(float)}},
                }, &winner, sizeof(winner));
                t.assert_true(winner == 777);
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

                int32_t winner = -999;
                run_graph(ctx.get(), out, {
                    {tw2,   {w2_bytes.data(), w2_bytes.size()}},
                    {tw2b,  {w2_b.data(),     w2_b.size() * sizeof(float)}},
                    {tbase, {base.data(),     base.size() * sizeof(float)}},
                    {tlat,  {lat.data(),      lat.size() * sizeof(float)}},
                    {tsl,   {sl.data(),       sl.size() * sizeof(int32_t)}},
                }, &winner, sizeof(winner));
                t.assert_true(winner == 777 * 3 + 11);
            }

            // 3. All-NaN logits must return -1 (not token 0)
            {
                std::vector<float> nan_base(vocab, std::numeric_limits<float>::quiet_NaN());
                ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}));
                auto * tw2   = ggml_new_tensor_2d(ctx.get(), w2_type, rank, vocab);
                auto * tw2b  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tbase = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
                auto * tlat  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, rank);
                auto * out   = create(ctx.get(), tw2, tw2b, tbase, tlat, nullptr, 30.0f);
                t.assert_true(out != nullptr);

                int32_t winner = 12345;
                run_graph(ctx.get(), out, {
                    {tw2,   {w2_bytes.data(), w2_bytes.size()}},
                    {tw2b,  {w2_b.data(),     w2_b.size() * sizeof(float)}},
                    {tbase, {nan_base.data(), nan_base.size() * sizeof(float)}},
                    {tlat,  {lat.data(),      lat.size() * sizeof(float)}},
                }, &winner, sizeof(winner));
                t.assert_true(winner == -1);
            }
        }
    });

    return t.summary();
}
