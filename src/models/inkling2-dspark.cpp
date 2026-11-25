// Inkling2 DSpark graph from llama.cpp PR #32218, revision 8ea4363119c0d8d205e0cce264d004c017cbd402.
// Keep the target visit ordering, shared output weights and full noncausal draft block in sync with upstream.
#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <vector>

void llama_model_dflash::load_hparams_inkling2(llama_model_loader & ml) {
    // PR #32218 at e4b1cb5318f1 exports W2 over the true vocabulary and includes
    // the anchor position in the serialized sliding window.
    const auto * w2 = ml.get_tensor_meta("markov_w2.weight");
    if (w2) {
        if (w2->ne[1] <= 0 || w2->ne[1] > (1 << 24)) {
            throw std::runtime_error("invalid Inkling2 DSpark output vocabulary");
        }
        hparams.n_vocab_padded = hparams.n_vocab_unpadded = w2->ne[1];
    } else {
        ml.get_key(LLM_KV_VOCAB_SIZE, hparams.n_vocab_padded);
        ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);
    }
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer());

    if (hparams.dflash_block_size != 8 || !hparams.n_swa || hparams.n_swa >= INT32_MAX) {
        throw std::runtime_error("invalid Inkling2 DSpark block size or sliding window");
    }
    if (target_layer_ids.empty() || target_layer_ids.front() < 0 || target_layer_ids.back() > LLAMA_MAX_LAYERS ||
            std::adjacent_find(target_layer_ids.begin(), target_layer_ids.end(), std::greater_equal<int32_t>()) != target_layer_ids.end()) {
        throw std::runtime_error("Inkling2 DSpark target layers must be in range and strictly increasing");
    }
    if (target_layer_ids.size() * (uint64_t) hparams.n_embd > UINT32_MAX) {
        throw std::runtime_error("Inkling2 DSpark target features exceed the input width");
    }
    if (!std::isfinite(hparams.f_norm_rms_eps) || hparams.f_norm_rms_eps <= 0 ||
            !std::isfinite(hparams.f_final_logit_softcapping) || hparams.f_final_logit_softcapping <= 0) {
        throw std::runtime_error("invalid Inkling2 DSpark RMS epsilon or logit softcap");
    }
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
                hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_ff(il) ||
                hparams.n_rot(il) != hparams.n_embd_head_k(il) || hparams.n_rot(il) % 2) {
            throw std::runtime_error("invalid Inkling2 DSpark attention dimensions");
        }
    }
    // The anchor row includes the anchor and the window before it.
    if (!w2) { hparams.n_swa += 1; }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.rope_freq_base_train_swa = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
}

void llama_model_dflash::load_tensors_inkling2() {
    LLAMA_LOAD_LOCALS;

    const int64_t rank = ml->require_tensor_meta("markov_w1.weight")->ne[0];
    if (rank <= 0 || rank > n_embd) {
        throw std::runtime_error("invalid Inkling2 DSpark recurrent rank");
    }
    // The greedy tie-break stores token ranks in F32.
    if (n_vocab <= 0 || n_vocab != hparams.n_vocab_unpadded || n_vocab > hparams.n_vocab_padded || n_vocab > (1 << 24)) {
        throw std::runtime_error("invalid Inkling2 DSpark vocabulary size");
    }
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    const bool projected = ml->get_tensor_meta("markov_w2.weight") != nullptr;
    if (projected) {
        dspark_mask_embd = create_tensor(tn(LLM_TENSOR_DSPARK_MASK_EMBD, "weight"), {n_embd, hparams.dflash_block_size - 1}, 0);
        tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
        dspark_fuse = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_CTX_MIX, "weight"), {(int64_t) target_layer_ids.size()}, 0);
        dspark_norm = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_CTX_NORM, "weight"), {n_embd}, 0);
        dspark_markov_w2 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), {rank, n_vocab}, 0);
        dspark_markov_w2_b = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "bias"), {n_vocab}, 0);
    } else {
        dspark_mask_embd = create_tensor(tn(LLM_TENSOR_DSPARK_MASK_EMBD, "weight"), {n_embd}, 0);
        dspark_fuse = create_tensor(tn(LLM_TENSOR_DSPARK_FUSE, "weight"), {(int64_t) target_layer_ids.size(), n_layer + 1}, 0);
        dspark_norm = create_tensor(tn(LLM_TENSOR_DSPARK_NORM, "weight"), {n_embd}, 0);
    }
    dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), {rank, hparams.n_vocab_padded}, 0);
    dspark_joint_proj = create_tensor(tn(LLM_TENSOR_DSPARK_JOINT_PROJ, "weight"), {2*rank + 2*n_embd, 3*rank}, 0);
    dspark_joint_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_JOINT_PROJ, "bias"), {3*rank}, 0);
    if (!projected) {
        dspark_hidden_out = create_tensor(tn(LLM_TENSOR_DSPARK_HIDDEN_OUT, "weight"), {rank, n_embd}, 0);
        dspark_hidden_out_b = create_tensor(tn(LLM_TENSOR_DSPARK_HIDDEN_OUT, "bias"), {n_embd}, 0);
    }
    const auto folded_name = tn(LLM_TENSOR_DSPARK_FOLDED_OUTPUT, "weight");
    if (const auto * folded = ml->get_tensor_meta(folded_name.str().c_str())) {
        if (folded->type != GGML_TYPE_F16) {
            throw std::runtime_error("Inkling2 DSpark folded output must use F16");
        }
        dspark_folded_output = create_tensor(folded_name, {GGML_PAD(rank + 1, 32), hparams.n_vocab_padded}, 0);
    }
    for (int i = 0; i < n_layer; ++i) {
        auto & l = layers[i];
        const int64_t hd = hparams.n_embd_head_k(i);
        const int64_t q = hd*hparams.n_head(i);
        const int64_t kv = hd*hparams.n_head_kv(i);
        l.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        l.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        if (projected) {
            l.dflash_ctx_mix = create_tensor(tn(LLM_TENSOR_DFLASH_CTX_MIX, "weight", i), {(int64_t) target_layer_ids.size()}, 0);
        }
        l.dflare_norm = create_tensor(tn(projected ? LLM_TENSOR_DFLASH_CTX_NORM : LLM_TENSOR_DFLARE_NORM, "weight", i), {n_embd}, 0);
        l.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, q}, 0);
        l.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, kv}, 0);
        l.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, kv}, 0);
        l.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {q, n_embd}, 0);
        l.attn_k_ctx = create_tensor(tn(LLM_TENSOR_ATTN_K_CTX, "weight", i), {n_embd, kv}, 0);
        l.attn_v_ctx = create_tensor(tn(LLM_TENSOR_ATTN_V_CTX, "weight", i), {n_embd, kv}, 0);
        l.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {hd}, 0);
        l.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {hd}, 0);
        l.attn_k_ctx_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_CTX_NORM, "weight", i), {hd}, 0);
        l.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, hparams.n_ff(i)}, 0);
        l.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, hparams.n_ff(i)}, 0);
        l.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {hparams.n_ff(i), n_embd}, 0);
    }
}

using inkling_mark_add = void (*)(ggml_tensor *);
using inkling_dspark_argmax_ready_fn = bool (*)(ggml_backend_t, int64_t, ggml_type);
using inkling_dspark_argmax_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, float);

// Returns an immutable cached head entry kept alive for the entire lifetime of `dmodel`.
// Existing entries are never freed or mutated while `dmodel` is alive, so cached graphs
// and in-flight GPU work referencing their tensors remain permanently valid.
static std::shared_ptr<const llama_model_dflash::inkling_dspark_fast_head_entry> ensure_inkling_fast_head(
        const llama_model_dflash & dmodel,
        const llama_model & target,
        int64_t req_vocab,
        ggml_type want_w2_type) {
    const int64_t n_vocab_full = dmodel.hparams.n_vocab_unpadded;
    const int64_t rank         = dmodel.dspark_markov_w1 ? dmodel.dspark_markov_w1->ne[0] : 0;
    const int64_t n_embd       = dmodel.hparams.n_embd;

    if (req_vocab <= 0 || req_vocab >= n_vocab_full) {
        req_vocab = n_vocab_full;
    } else if (req_vocab < 1024) {
        req_vocab = std::min<int64_t>(1024, n_vocab_full);
    }

    const bool use_short = req_vocab < n_vocab_full;
    const bool quant_w2  = want_w2_type == GGML_TYPE_Q4_0 || want_w2_type == GGML_TYPE_Q8_0;
    if (!use_short && !quant_w2) {
        return nullptr;
    }
    if (!dmodel.dev_output() || !dmodel.dspark_markov_w2 || !dmodel.dspark_markov_w2_b ||
            dmodel.dspark_markov_w2->type != GGML_TYPE_BF16 || !target.output || !target.output->data ||
            target.output->ne[0] != n_embd || target.output->ne[1] < n_vocab_full || !ggml_is_contiguous(target.output) ||
            rank <= 0 || rank > 512 || (rank % 32) != 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(dmodel.inkling_head_mtx);
    for (const auto & entry : dmodel.inkling_head_cache) {
        if (entry->target_output_data == target.output->data &&
                entry->target_ne0 == target.output->ne[0] &&
                entry->target_ne1 == target.output->ne[1] &&
                entry->req_vocab == req_vocab &&
                entry->w2_type == want_w2_type) {
            return entry;
        }
    }

    // Bound total cached variants per draft model without evicting live entries.
    constexpr size_t MAX_CACHED_HEADS = 8;
    if (dmodel.inkling_head_cache.size() >= MAX_CACHED_HEADS) {
        return nullptr;
    }

    const int64_t t0 = ggml_time_us();

    std::vector<int32_t> selected_ids;
    std::vector<float> bias_full;
    int64_t eff_vocab = req_vocab;
    if (use_short) {
        bias_full.resize((size_t) n_vocab_full);
        ggml_backend_tensor_get(dmodel.dspark_markov_w2_b, bias_full.data(), 0, bias_full.size() * sizeof(float));

        selected_ids.reserve((size_t) req_vocab);
        std::vector<int32_t> candidates;
        candidates.reserve((size_t) n_vocab_full);

        for (int32_t id = 0; id < (int32_t) n_vocab_full; ++id) {
            if (id < 256 || dmodel.vocab.is_control(id) || dmodel.vocab.is_eog(id)) {
                selected_ids.push_back(id);
            } else {
                candidates.push_back(id);
            }
        }
        if ((int64_t) selected_ids.size() < req_vocab && !candidates.empty()) {
            const size_t need = std::min(candidates.size(), (size_t) req_vocab - selected_ids.size());
            std::nth_element(candidates.begin(), candidates.begin() + need, candidates.end(),
                    [&](int32_t a, int32_t b) {
                        const float ba = bias_full[(size_t) a];
                        const float bb = bias_full[(size_t) b];
                        return ba != bb ? ba > bb : a < b;
                    });
            selected_ids.insert(selected_ids.end(), candidates.begin(), candidates.begin() + need);
        }
        std::sort(selected_ids.begin(), selected_ids.end());
        eff_vocab = (int64_t) selected_ids.size();
    }

    auto entry = std::make_shared<llama_model_dflash::inkling_dspark_fast_head_entry>();
    entry->target_output_data = target.output->data;
    entry->target_ne0         = target.output->ne[0];
    entry->target_ne1         = target.output->ne[1];
    entry->req_vocab          = req_vocab;
    entry->eff_vocab          = eff_vocab;
    entry->w2_type            = want_w2_type;

    const int n_tensors = use_short ? 4 : 1;
    ggml_init_params init_p = {
        /* .mem_size   = */ (size_t) n_tensors * ggml_tensor_overhead() + 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    entry->ctx.reset(ggml_init(init_p));
    if (!entry->ctx) { return nullptr; }

    const ggml_type w2_type = quant_w2 ? want_w2_type : GGML_TYPE_BF16;
    entry->w2_fast = ggml_new_tensor_2d(entry->ctx.get(), w2_type, rank, eff_vocab);
    ggml_set_name(entry->w2_fast, "dspark_markov_w2_fast");

    if (use_short) {
        entry->output_short  = ggml_new_tensor_2d(entry->ctx.get(), target.output->type, n_embd, eff_vocab);
        entry->w2_b_fast     = ggml_new_tensor_1d(entry->ctx.get(), GGML_TYPE_F32, eff_vocab);
        entry->shortlist_ids = ggml_new_tensor_1d(entry->ctx.get(), GGML_TYPE_I32, eff_vocab);
        ggml_set_name(entry->output_short,  "dspark_output_short");
        ggml_set_name(entry->w2_b_fast,     "dspark_markov_w2_b_short");
        ggml_set_name(entry->shortlist_ids, "dspark_shortlist_ids");
    }

    auto * buft = ggml_backend_dev_buffer_type(dmodel.dev_output());
    entry->buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(entry->ctx.get(), buft));
    if (!entry->buf) {
        return nullptr;
    }

    // Gather/quantize markov_w2 in a single bulk read
    {
        std::vector<uint16_t> w2_bf16((size_t) rank * (size_t) n_vocab_full);
        ggml_backend_tensor_get(dmodel.dspark_markov_w2, w2_bf16.data(), 0, w2_bf16.size() * sizeof(uint16_t));

        const size_t dst_row_bytes = ggml_row_size(w2_type, rank);
        std::vector<uint8_t> w2_dst((size_t) eff_vocab * dst_row_bytes);
        std::vector<float> f32_row(quant_w2 ? (size_t) rank : 0);

        for (int64_t i = 0; i < eff_vocab; ++i) {
            const int64_t src_id = use_short ? (int64_t) selected_ids[(size_t) i] : i;
            const uint16_t * src_row = w2_bf16.data() + (size_t) src_id * (size_t) rank;
            uint8_t * dst_row = w2_dst.data() + (size_t) i * dst_row_bytes;
            if (quant_w2) {
                for (int64_t k = 0; k < rank; ++k) {
                    const uint32_t bits = uint32_t(src_row[k]) << 16;
                    std::memcpy(&f32_row[(size_t) k], &bits, sizeof(float));
                }
                ggml_quantize_chunk(w2_type, f32_row.data(), dst_row, 0, 1, rank, nullptr);
            } else {
                std::memcpy(dst_row, src_row, dst_row_bytes);
            }
        }
        ggml_backend_tensor_set(entry->w2_fast, w2_dst.data(), 0, w2_dst.size());
    }

    if (use_short) {
        // Bulk-copy target.output once instead of one synced device read per vocab row.
        const size_t out_row_bytes = ggml_row_size(target.output->type, n_embd);
        const size_t out_stride    = target.output->nb[1];
        std::vector<uint8_t> out_all(ggml_nbytes(target.output));
        ggml_backend_tensor_get(target.output, out_all.data(), 0, out_all.size());
        std::vector<uint8_t> out_dst((size_t) eff_vocab * out_row_bytes);
        for (int64_t i = 0; i < eff_vocab; ++i) {
            const size_t src_off = (size_t) selected_ids[(size_t) i] * out_stride;
            std::memcpy(out_dst.data() + (size_t) i * out_row_bytes, out_all.data() + src_off, out_row_bytes);
        }
        ggml_backend_tensor_set(entry->output_short, out_dst.data(), 0, out_dst.size());

        std::vector<float> b_dst((size_t) eff_vocab);
        for (int64_t i = 0; i < eff_vocab; ++i) {
            b_dst[(size_t) i] = bias_full[(size_t) selected_ids[(size_t) i]];
        }
        ggml_backend_tensor_set(entry->w2_b_fast, b_dst.data(), 0, b_dst.size() * sizeof(float));
        ggml_backend_tensor_set(entry->shortlist_ids, selected_ids.data(), 0, selected_ids.size() * sizeof(int32_t));
    }

    LLAMA_LOG_INFO("%s: built immutable Inkling DSpark fast head (vocab=%lld/%lld, w2=%s, size=%.2f MiB, %.1f ms)\n",
            __func__, (long long) eff_vocab, (long long) n_vocab_full,
            ggml_type_name(w2_type),
            ggml_backend_buffer_get_size(entry->buf.get()) / (1024.0 * 1024.0),
            (ggml_time_us() - t0) / 1000.0);
    dmodel.inkling_head_cache.push_back(entry);
    return entry;
}

static ggml_tensor * build_inkling2_head(llm_graph_context & g, const llama_model & model, const llama_model & target,
        ggml_tensor * prev, ggml_tensor * & state, ggml_tensor * hidden, ggml_tensor * base, ggml_tensor * context, inkling_mark_add mark_add) {
    auto * ctx0 = g.ctx0;
    const int64_t rank = model.dspark_markov_w1->ne[0];
    auto * joint = ggml_concat(ctx0, state, ggml_get_rows(ctx0, model.dspark_markov_w1, prev), 0);
    joint = ggml_concat(ctx0, joint, ggml_concat(ctx0, hidden, context, 0), 0);
    joint = ggml_add(ctx0, g.build_lora_mm(model.dspark_joint_proj, joint), model.dspark_joint_proj_b);
    const auto chunk = [&](int64_t j) {
        return ggml_cont(ctx0, ggml_view_2d(ctx0, joint, rank, joint->ne[1], joint->nb[1], j*rank*joint->nb[0]));
    };
    auto * gate = ggml_sigmoid(ctx0, chunk(0));
    auto * candidate = ggml_tanh(ctx0, chunk(1));
    state = ggml_add(ctx0, candidate, ggml_mul(ctx0, gate, ggml_sub(ctx0, state, candidate)));
    auto * latent = ggml_tanh(ctx0, chunk(2));
    ggml_tensor * bias;
    if (model.dspark_markov_w2) {
        bias = g.build_lora_mm(model.dspark_markov_w2, latent);
    } else if (model.dspark_folded_output && g.loras->empty() && !target.output_s) {
        // This optional sidecar tensor folds the target output projection
        // through the rank-sized DSpark correction, including its bias.
        auto * one = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, latent->ne[1]), 1.0f);
        auto * augmented = ggml_concat(ctx0, latent, one, 0);
        augmented = ggml_pad(ctx0, augmented, model.dspark_folded_output->ne[0] - augmented->ne[0], 0, 0, 0);
        bias = ggml_mul_mat(ctx0, model.dspark_folded_output, augmented);
    } else {
        auto * correction = ggml_add(ctx0, g.build_lora_mm(model.dspark_hidden_out, latent), model.dspark_hidden_out_b);
        bias = g.build_lora_mm(target.output, correction, target.output_s);
    }
    ggml_prec_set_acc(bias, GGML_PREC_F32);
    if (model.dspark_markov_w2_b) {
        bias = ggml_add(ctx0, bias, model.dspark_markov_w2_b);
        if (mark_add) { mark_add(bias); }
    }
    if (bias->ne[0] != base->ne[0]) {
        bias = ggml_cont(ctx0, ggml_view_2d(ctx0, bias, base->ne[0], bias->ne[1], bias->nb[1], 0));
    }
    auto * logits = ggml_add(ctx0, base, bias);
    if (mark_add) { mark_add(logits); }
    const float cap = model.hparams.f_final_logit_softcapping;
    return ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, logits, 1.0f/cap)), cap);
}

namespace {
// A prefix request can stop the recurrent head early. Another mask of the same size
// can need later recurrent steps, so it must not reuse the prefix graph.
class inkling_draft_input final : public llm_graph_input_embd_h {
public:
    inkling_draft_input(int64_t width, const llm_graph_params & params)
        : llm_graph_input_embd_h(width, width), prefix_rows(output_prefix(params)) {}

    bool can_reuse(const llm_graph_params & params) override {
        return prefix_rows == output_prefix(params) && llm_graph_input_embd_h::can_reuse(params);
    }

    const uint32_t prefix_rows;

private:
    static uint32_t output_prefix(const llm_graph_params & params) {
        const auto & batch = params.ubatch;
        if (batch.n_seqs_unq == 1 && params.n_outputs < batch.n_tokens && batch.output &&
                std::all_of(batch.output, batch.output + params.n_outputs, [](int8_t flag) { return flag != 0; })) {
            return params.n_outputs;
        }
        return 0;
    }
};
} // namespace

// dspark_fuse has one row of target-layer mixing weights per draft layer and a final row for the recurrent head.
llama_model_dflash::graph_inkling2::graph_inkling2(const llm_graph_params & params) : llm_graph_context(params) {
    auto * positions = build_inp_pos();
    auto * attn = build_attn_inp_kv_iswa();
    const int64_t rank = model.dspark_markov_w1->ne[0];
    const int64_t ntarget = model.target_layer_ids.size();
    const auto mix = [&](ggml_tensor * features, int row, ggml_tensor * norm) {
        if (model.dspark_markov_w2) {
            auto * weights = row == n_layer ? model.dspark_fuse : model.layers[row].dflash_ctx_mix;
            auto * taps = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, features, n_embd, ntarget, n_tokens), 1, 0, 2, 3));
            auto * mixed = ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, weights, taps), n_embd, n_tokens);
            return build_norm(mixed, norm, nullptr, LLM_NORM_RMS, row == n_layer ? -1 : row);
        }
        auto * weights = ggml_soft_max(ctx0, ggml_view_1d(ctx0, model.dspark_fuse, ntarget, row*model.dspark_fuse->nb[1]));
        auto * x = ggml_reshape_3d(ctx0, features, n_embd, ntarget, n_tokens);
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
        x = ggml_sum_rows(ctx0, ggml_mul(ctx0, x, weights));
        return build_norm(ggml_reshape_2d(ctx0, x, n_embd, n_tokens), norm, nullptr, LLM_NORM_RMS, row == n_layer ? -1 : row);
    };

    if (ubatch.embd || ubatch.embd_row) {
        auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());
        auto * features = build_inp_target_features(*this, *inp, hparams.n_embd_inp_enc());
        res->add_input(std::move(inp));
        for (int il = 0; il < n_layer; ++il) {
            const auto & l = model.layers[il];
            const int64_t hd = hparams.n_embd_head_k(il);
            const int64_t nk = hparams.n_head_kv(il);
            auto * context = mix(features, il, l.dflare_norm);
            cb(context, "dspark_context", il);
            auto * k = ggml_reshape_3d(ctx0, build_lora_mm(l.attn_k_ctx, context), hd, nk, n_tokens);
            auto * v = ggml_reshape_3d(ctx0, build_lora_mm(l.attn_v_ctx, context), hd, nk, n_tokens);
            k = build_norm(k, l.attn_k_ctx_norm, nullptr, LLM_NORM_RMS, il);
            k = build_rope(k, positions, il);
            const bool swa = hparams.is_swa(il);
            const auto * kv = swa ? attn->mctx->get_swa() : attn->mctx->get_base();
            auto * k_rot = swa ? attn->self_k_rot_swa[il] : attn->self_k_rot[il];
            auto * v_rot = swa ? attn->self_v_rot_swa[il] : attn->self_v_rot[il];
            if (k_rot) {
                k = llama_mul_mat_hadamard(ctx0, k, k_rot);
            }
            if (v_rot) {
                v = llama_mul_mat_hadamard(ctx0, v, v_rot);
            }
            ggml_build_forward_expand(gf, kv->cpy_k(ctx0, k, swa ? attn->get_k_idxs_swa() : attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, kv->cpy_v(ctx0, v, swa ? attn->get_v_idxs_swa() : attn->get_v_idxs(), il));
        }
        res->t_embd = res->t_h_nextn = mix(features, n_layer, model.dspark_norm);
        cb(res->t_h_nextn, "dspark_f_rnn", -1);
        ggml_build_forward_expand(gf, res->t_h_nextn);
        return;
    }

    GGML_ASSERT(cparams.ctx_other);
    const auto * target = llama_get_model(cparams.ctx_other);
    GGML_ASSERT(target->tok_embd && target->tok_norm && target->output);
    auto inp = std::make_unique<inkling_draft_input>(n_embd, params);
    const auto prefix_rows = inp->prefix_rows;
    auto * tokens = inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    auto * f_rnn = inp->h = build_inp_embd_state(inp.get(), n_embd);
    res->add_input(std::move(inp));

    const int64_t nb = ubatch.n_seqs_unq;
    const int64_t block = n_tokens/nb;
    auto * ids = ggml_cont_1d(ctx0, ggml_view_2d(ctx0, tokens, 1, nb, block*tokens->nb[0], 0), nb);
    auto * anchors = ggml_get_rows(ctx0, target->tok_embd, ids);
    anchors = ggml_mul(ctx0, ggml_rms_norm(ctx0, anchors, model.tok_norm ? hparams.f_norm_rms_eps : target->hparams.f_norm_rms_eps),
            model.tok_norm ? model.tok_norm : target->tok_norm);
    auto * cur = ggml_reshape_3d(ctx0, anchors, n_embd, 1, nb);
    if (model.dspark_markov_w2 && block > hparams.dflash_block_size) {
        cur = ggml_get_rows(ctx0, target->tok_embd, tokens);
    } else if (block > 1) {
        auto * mask = model.dspark_mask_embd;
        if (model.dspark_markov_w2) {
            mask = ggml_view_2d(ctx0, mask, n_embd, block - 1, mask->nb[1], 0);
        }
        auto * masks = ggml_repeat_4d(ctx0, ggml_cast(ctx0, mask, GGML_TYPE_F32), n_embd, block - 1, nb, 1);
        cur = ggml_concat(ctx0, cur, masks, 1);
    }
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    attn->self_attn_mask_swa.block_uniform = true;
    for (int il = 0; il < n_layer; ++il) {
        const auto & l = model.layers[il];
        const int64_t hd = hparams.n_embd_head_k(il);
        auto * residual = cur;
        cur = build_norm(cur, l.attn_norm, nullptr, LLM_NORM_RMS, il);
        auto * q = ggml_reshape_3d(ctx0, build_lora_mm(l.wq, cur), hd, hparams.n_head(il), n_tokens);
        auto * k = ggml_reshape_3d(ctx0, build_lora_mm(l.wk, cur), hd, hparams.n_head_kv(il), n_tokens);
        auto * v = ggml_reshape_3d(ctx0, build_lora_mm(l.wv, cur), hd, hparams.n_head_kv(il), n_tokens);
        q = build_rope(build_norm(q, l.attn_q_norm, nullptr, LLM_NORM_RMS, il), positions, il);
        k = build_rope(build_norm(k, l.attn_k_norm, nullptr, LLM_NORM_RMS, il), positions, il);
        cur = build_attn(attn, l.wo, nullptr, nullptr, q, k, v, nullptr, nullptr, nullptr, 1.0f/sqrtf(hd), il);
        residual = ggml_add(ctx0, residual, cur);
        cur = build_norm(residual, l.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_ffn(cur, l.ffn_up, nullptr, nullptr, l.ffn_gate, nullptr, nullptr,
                l.ffn_down, nullptr, nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur = ggml_add(ctx0, residual, cur);
        cb(cur, "dspark_layer", il);
    }
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd = cur;

    // Fast GPU-argmax path for opt-in single-slot greedy Inkling2 DSpark drafting:
    // projects only the active mask rows (1..head_rows-1) through an optional compact
    // vocab shortlist + Q8_0 markov_w2, fuses w2*latent + w2_b + base + softcap + first-max argmax
    // into a single Metal op, and exposes exact int32 token IDs via res->t_draft_ids.
    const int64_t head_rows = prefix_rows ? prefix_rows : block;
    auto has_lora = [&](ggml_tensor * w) -> bool {
        if (!loras || !w) { return false; }
        for (const auto & lora : *loras) {
            if (lora.second != 0.0f && lora.first->get_weight(w) != nullptr) { return true; }
        }
        return false;
    };
    if (cparams.dspark_fast_argmax && cparams.n_seq_max == 1 && nb == 1 &&
            block <= hparams.dflash_block_size && head_rows > 1 && head_rows <= block &&
            rank > 0 && rank <= 512 && (rank % 32) == 0 &&
            model.dspark_markov_w2 && model.dspark_markov_w2_b &&
            !has_lora(model.dspark_markov_w2) && !has_lora(target->output) &&
            !target->output_s && model.dev_output() &&
            std::strcmp(ggml_backend_dev_name(model.dev_output()), "MTL0") == 0) {
        const char * fast_env = std::getenv("LLAMA_INKLING_DSPARK_FAST");
        if (!fast_env || std::strcmp(fast_env, "0") != 0) {
            auto reg = ggml_backend_dev_backend_reg(model.dev_output());
            auto ready_fn = reinterpret_cast<inkling_dspark_argmax_ready_fn>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_dspark_argmax_ready_v2"));
            auto argmax_fn = reinterpret_cast<inkling_dspark_argmax_fn>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_dspark_argmax_v2"));
            ggml_backend_t dspark_backend = nullptr;
            for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
                auto b = ggml_backend_sched_get_backend(sched, i);
                if (ggml_backend_get_device(b) == model.dev_output()) { dspark_backend = b; break; }
            }
            if (ready_fn && argmax_fn && dspark_backend) {
                const char * sl_env = std::getenv("LLAMA_INKLING_DSPARK_SHORTLIST");
                const int64_t want_vocab = sl_env ? std::atoll(sl_env) : 0;
                const char * q4_env = std::getenv("LLAMA_INKLING_DSPARK_W2_Q4");
                const char * q8_env = std::getenv("LLAMA_INKLING_DSPARK_W2_Q8");
                const bool want_q4 = q4_env && std::strcmp(q4_env, "1") == 0;
                const bool want_q8 = !want_q4 && (!q8_env || std::strcmp(q8_env, "0") != 0);
                const ggml_type planned_w2 = want_q4 ? GGML_TYPE_Q4_0
                                           : want_q8 ? GGML_TYPE_Q8_0
                                                     : model.dspark_markov_w2->type;

                if (ready_fn(dspark_backend, rank, planned_w2)) {
                    const auto & dmodel = static_cast<const llama_model_dflash &>(model);
                    const auto fh = ensure_inkling_fast_head(dmodel, *target, want_vocab, planned_w2);

                    ggml_tensor * output_w      = (fh && fh->output_short)  ? fh->output_short  : target->output;
                    ggml_tensor * w2_w          = (fh && fh->w2_fast)       ? fh->w2_fast       : model.dspark_markov_w2;
                    ggml_tensor * w2_b_w        = (fh && fh->w2_b_fast)     ? fh->w2_b_fast     : model.dspark_markov_w2_b;
                    ggml_tensor * shortlist_ids = fh ? fh->shortlist_ids : nullptr;

                    if (ready_fn(dspark_backend, w2_w->ne[0], w2_w->type)) {
                        const int64_t v_eff   = w2_w->ne[1];
                        const int64_t n_draft = head_rows - 1;
                        const float   cap     = model.hparams.f_final_logit_softcapping;

                        auto * cur_active = ggml_view_2d(ctx0, cur, n_embd, n_draft, cur->nb[1], cur->nb[1]);
                        auto * base_fast  = build_lora_mm(output_w, cur_active, target->output_s);
                        ggml_prec_set_acc(base_fast, GGML_PREC_F32);

                        auto * state   = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rank, 1), 0.0f);
                        auto * context = ggml_view_2d(ctx0, f_rnn, n_embd, 1, f_rnn->nb[1], 0);
                        auto * prev    = ids;

                        ggml_tensor * ids_vec     = nullptr;
                        ggml_tensor * margins_vec = nullptr;
                        bool fast_ok = true;
                        for (int64_t d = 0; d < n_draft; ++d) {
                            auto * hidden = ggml_view_2d(ctx0, cur, n_embd, 1, cur->nb[1], (d + 1) * cur->nb[1]);
                            auto * joint  = ggml_concat(ctx0, state, ggml_get_rows(ctx0, model.dspark_markov_w1, prev), 0);
                            joint = ggml_concat(ctx0, joint, ggml_concat(ctx0, hidden, context, 0), 0);
                            joint = ggml_add(ctx0, build_lora_mm(model.dspark_joint_proj, joint), model.dspark_joint_proj_b);
                            const auto chunk = [&](int64_t j) {
                                return ggml_cont(ctx0, ggml_view_2d(ctx0, joint, rank, 1, joint->nb[1], j * rank * joint->nb[0]));
                            };
                            auto * gate      = ggml_sigmoid(ctx0, chunk(0));
                            auto * candidate = ggml_tanh(ctx0, chunk(1));
                            state = ggml_add(ctx0, candidate, ggml_mul(ctx0, gate, ggml_sub(ctx0, state, candidate)));
                            auto * latent = ggml_tanh(ctx0, chunk(2));

                            auto * base_row   = ggml_view_1d(ctx0, base_fast, v_eff, d * base_fast->nb[1]);
                            auto * argmax_out = argmax_fn(ctx0, w2_w, w2_b_w, base_row, latent, shortlist_ids, cap);
                            if (!argmax_out || argmax_out->type != GGML_TYPE_I32 || ggml_nelements(argmax_out) < 3 ||
                                    !ggml_backend_supports_op(dspark_backend, argmax_out)) {
                                fast_ok = false;
                                break;
                            }
                            ggml_backend_sched_set_tensor_backend(sched, argmax_out, dspark_backend);
                            // Elem 0 = raw token ID (-1 on all-NaN); elem 1 = safe clamped index (>= 0); elem 2 = float bits of top1-top2 margin.
                            auto * raw_id  = ggml_view_1d(ctx0, argmax_out, 1, 0);
                            prev           = ggml_view_1d(ctx0, argmax_out, 1, sizeof(int32_t));
                            auto * margin  = ggml_view_1d(ctx0, argmax_out, 1, 2 * sizeof(int32_t));
                            ids_vec        = ids_vec     ? ggml_concat(ctx0, ids_vec, raw_id, 0)         : raw_id;
                            margins_vec    = margins_vec ? ggml_concat(ctx0, margins_vec, margin, 0)     : margin;
                        }

                        if (fast_ok && ids_vec && margins_vec) {
                            auto * packed = ggml_concat(ctx0,
                                    ggml_reshape_2d(ctx0, ids_vec, n_draft, 1),
                                    ggml_reshape_2d(ctx0, margins_vec, n_draft, 1), 1);
                            res->t_draft_ids = packed;
                            res->t_logits    = nullptr;
                            ggml_build_forward_expand(gf, packed);
                            return;
                        }
                    }
                }
            }
        }
    }

    auto * base = build_lora_mm(target->output, cur, target->output_s);
    ggml_prec_set_acc(base, GGML_PREC_F32);
    if (model.dspark_markov_w2 && base->ne[0] > hparams.n_vocab_unpadded) {
        base = ggml_view_2d(ctx0, base, hparams.n_vocab_unpadded, n_tokens, base->nb[1], 0);
    }
    if (block > hparams.dflash_block_size) {
        res->t_logits = ggml_cont(ctx0, ggml_view_2d(ctx0, base, hparams.n_vocab_unpadded, n_tokens, base->nb[1], 0));
        return;
    }
    auto * state = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rank, nb), 0.0f);
    auto * logits = ggml_cont(ctx0, ggml_view_2d(ctx0, base, base->ne[0], nb, block*base->nb[1], 0));
    const auto column = [&](ggml_tensor * t, int64_t i) {
        return ggml_cont(ctx0, ggml_view_2d(ctx0, t, t->ne[0], nb, block*t->nb[1], i*t->nb[1]));
    };
    auto * context = column(f_rnn, 0);
    auto * tie_rank = ggml_arange(ctx0, 0, hparams.n_vocab_unpadded, 1);
    tie_rank = ggml_scale_bias(ctx0, tie_rank, -1.0f, hparams.n_vocab_unpadded);
    auto * prev = ids;
    inkling_mark_add mark_add = nullptr;
    if (model.dspark_markov_w2 && nb == 1 && model.dev_output()) {
        const auto * opt = std::getenv("LLAMA_EXPERIMENTAL_INKLING_W2_PARALLEL_ADD");
        if (!opt || std::strcmp(opt, "1") == 0) {
            auto reg = ggml_backend_dev_backend_reg(model.dev_output());
            mark_add = reinterpret_cast<inkling_mark_add>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_mark_inkling_vocab_add"));
        }
    }
    for (int64_t i = 1; i < head_rows; ++i) {
        auto * col = build_inkling2_head(*this, model, *target, prev, state, column(cur, i), column(base, i), context, mark_add);
        logits = ggml_concat(ctx0, logits, col, 1);
        // Select the first maximum, independent of backend argmax tie order.
        auto * valid = ggml_cont(ctx0, ggml_view_2d(ctx0, col, hparams.n_vocab_unpadded, nb, col->nb[1], 0));
        auto * maximum = ggml_pool_1d(ctx0, valid, GGML_OP_POOL_MAX, valid->ne[0], 1, 0);
        auto * is_max = ggml_scale_bias(ctx0, ggml_step(ctx0, ggml_sub(ctx0, ggml_repeat(ctx0, maximum, valid), valid)), -1.0f, 1.0f);
        prev = ggml_argmax(ctx0, ggml_mul(ctx0, is_max, tie_rank));
    }
    logits = ggml_reshape_3d(ctx0, logits, logits->ne[0], nb, head_rows);
    logits = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, logits, 0, 2, 1, 3)), logits->ne[0], nb*head_rows);
    res->t_logits = ggml_cont(ctx0, ggml_view_2d(ctx0, logits, hparams.n_vocab_unpadded, nb*head_rows, logits->nb[1], 0));
}
