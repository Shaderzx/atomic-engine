#include "models.h"
#include "../llama-expert-executor.h"
#include "../llama-memory-hybrid-iswa.h"
#include <cstdlib>
#include <cstring>

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_head_kv   = hparams.n_head_kv(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff_exp    = hparams.n_ff_exp(i);
        const int64_t n_extent    = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(i)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale
ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    if (inkling_dense_backend && clip_kernel) {
        if (auto * fused = clip_kernel(ctx0, cur, scale, threshold)) {
            ggml_backend_sched_set_tensor_backend(sched, fused, inkling_dense_backend);
            return fused;
        }
    }
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    ggml_tensor * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
    norm = ggml_scale_bias(ctx0, norm, 1.0f, 1e-6f);

    ggml_tensor * factor = ggml_div(ctx0, ggml_fill(ctx0, norm, threshold), norm);

    return ggml_mul(ctx0, cur, ggml_clamp(ctx0, factor, 0.0f, 1.0f));
}

// attention over the depth: mix the earlier block outputs and cur with learned queries
ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query,
        bool * out_fused) {
    if (out_fused) { *out_fused = false; }
    // GGML_MAX_SRC is 10 and ggml_custom_4d requires n_args < GGML_MAX_SRC, so cap at 8 states + 1 query = 9.
    constexpr size_t MAX_FUSED_MIX_STATES = 8;
    if (inkling_dense_backend && mix_kernel && blocks.size() + 1 <= MAX_FUSED_MIX_STATES) {
        ggml_tensor * st[MAX_FUSED_MIX_STATES];
        int n_st = 0;
        for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
            st[n_st++] = *it;
        }
        st[n_st++] = cur;
        if (auto * fused = mix_kernel(ctx0, st, n_st, query)) {
            if (ggml_backend_supports_op(inkling_dense_backend, fused)) {
                ggml_backend_sched_set_tensor_backend(sched, fused, inkling_dense_backend);
                if (out_fused) { *out_fused = true; }
                return fused;
            }
        }
    }
    ggml_tensor * states = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    for (ggml_tensor * block : blocks) {
        states = ggml_concat(ctx0, ggml_reshape_3d(ctx0, block, n_embd, 1, n_tokens), states, 1);
    }
    states = ggml_rms_norm(ctx0, states, 1e-6f);

    ggml_tensor * queries = ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));

    ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);

    return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);

    build_conv_state_store(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

    const float scale = 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1);
    if (inkling_dense_backend && conv_residual_kernel) {
        if (auto * fused = conv_residual_kernel(ctx0, cur, weight, x, dynamic, scale)) {
            ggml_backend_sched_set_tensor_backend(sched, fused, inkling_dense_backend);
            return fused;
        }
    }

    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);
    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        dynamic = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);

        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_3d(ctx0, x, n_seq_tokens, n_channels, n_seqs, x->nb[1], x->nb[2], (n_conv - 1 - lag)*x->nb[0]);

            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, n_channels/n_heads, n_heads, n_tokens);

            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);

            out = ggml_add(ctx0, out, term);
        }
    }

    return ggml_scale(ctx0, ggml_add(ctx0, cur, out), scale);
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * message, const loop_layer & loop, int ctrl) {
    if (inkling_dense_backend && ctrl_down_kernel && ctrl_apply_kernel && n_tokens <= 8 && (!loras || loras->empty())) {
        if (auto * gate = ctrl_down_kernel(ctx0, cur, message, loop.ctrl_down[ctrl])) {
            ggml_backend_sched_set_tensor_backend(sched, gate, inkling_dense_backend);
            if (auto * out = ctrl_apply_kernel(ctx0, cur, message, gate, loop.ctrl_gain[ctrl], loop.ctrl_forget[ctrl])) {
                ggml_backend_sched_set_tensor_backend(sched, out, inkling_dense_backend);
                return out;
            }
        }
    }
    ggml_tensor * gate = ggml_concat(ctx0, ggml_rms_norm(ctx0, cur, 1e-6f), ggml_rms_norm(ctx0, message, 1e-6f), 0);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);

    ggml_tensor * gain   = build_lora_mm(loop.ctrl_gain[ctrl],   gate);
    ggml_tensor * forget = build_lora_mm(loop.ctrl_forget[ctrl], gate);
    if (control_kernel && inkling_dense_backend) {
        if (auto * fused = control_kernel(ctx0, cur, message, gain, forget)) {
            ggml_backend_sched_set_tensor_backend(sched, fused, inkling_dense_backend);
            return fused;
        }
    }
    gain   = ggml_tanh(ctx0, gain);
    forget = ggml_tanh(ctx0, forget);

    return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
}

ggml_tensor * llama_model_inkling2::graph::build_inputs() {
    ggml_backend_t flash_backend = nullptr;
    decltype(inp->flash_attention.create) flash_create = nullptr;
    {
        const char * env = std::getenv("LLAMA_INKLING_FUSED_ATTN");
        const bool enabled = !env || std::strcmp(env, "0") != 0;
        const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);
        const auto * attn_ctx = mctx_cur ? mctx_cur->get_attn() : nullptr;
        const auto device = model.dev_output();
        bool head_dims_match = hparams.n_relative_extent > 0 && hparams.n_layer() > 0;
        for (uint32_t il = 0; head_dims_match && il < hparams.n_layer(); ++il) {
            head_dims_match &= hparams.n_head(il) == 24 && hparams.n_head_kv(il) == 8 &&
                               hparams.n_embd_head_k(il) == 128 && hparams.n_embd_head_v(il) == 128;
        }
        bool seq_ids_ok = (cparams.kv_unified || cparams.n_seq_max == 1) &&
                          cparams.n_seq_max <= 64 && ubatch.n_seqs <= 64;
        if (seq_ids_ok && ubatch.n_seq_id && ubatch.seq_id) {
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i][0] < 0 || ubatch.seq_id[i][0] >= 64) {
                    seq_ids_ok = false;
                    break;
                }
            }
        }
        if (enabled && !expert_executor && !cparams.training && seq_ids_ok &&
                cparams.causal_attn && cparams.flash_attn && head_dims_match &&
                attn_ctx && device && std::strcmp(ggml_backend_dev_name(device), "MTL0") == 0 &&
                model.devices.size() == 1 && model.n_gpu_layers() > hparams.n_layer_all &&
                attn_ctx->get_base()->get_type_k() == attn_ctx->get_base()->get_type_v() &&
                attn_ctx->get_swa()->get_type_k()  == attn_ctx->get_swa()->get_type_v() &&
                attn_ctx->get_base()->get_n_kv() > 0 && attn_ctx->get_base()->get_n_kv() % 64 == 0 &&
                attn_ctx->get_swa()->get_n_kv()  > 0 && attn_ctx->get_swa()->get_n_kv()  % 64 == 0) {
            const auto reg = ggml_backend_dev_backend_reg(device);
            const auto ready = reinterpret_cast<bool (*)(ggml_backend_t, int64_t, ggml_type)>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_flash_attn_ready_v1"));
            const auto create = reinterpret_cast<decltype(flash_create)>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_inkling_flash_attn_v1"));
            if (ready && create) {
                for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
                    auto backend = ggml_backend_sched_get_backend(sched, i);
                    if (ggml_backend_get_device(backend) != device) { continue; }
                    if (ready(backend, n_tokens, attn_ctx->get_base()->get_type_k()) &&
                            ready(backend, n_tokens, attn_ctx->get_swa()->get_type_k())) {
                        flash_backend = backend;
                        flash_create  = create;
                    }
                    break;
                }
            }
        }

        // On Metal, fail fast instead of hanging in sched_reserve when the dense kq_b fallback
        // exceeds free GPU memory (e.g. non-Metal-4 Macs or --no-kv-unified at 256K).
        const char * allow_dense = std::getenv("LLAMA_INKLING_ALLOW_DENSE_KQ_B");
        const bool bypass_dense_guard = allow_dense && std::strcmp(allow_dense, "1") == 0;
        if (!flash_create && !bypass_dense_guard && attn_ctx && device &&
                std::strcmp(ggml_backend_dev_name(device), "MTL0") == 0) {
            size_t mem_free = 0, mem_total = 0;
            ggml_backend_dev_memory(device, &mem_free, &mem_total);
            const int64_t n_kv_base = attn_ctx->get_base()->get_n_kv();
            const int64_t kq_b_elem_bytes = (inkling_dense_backend && rel_mask_kernel) ? 2 : 4;
            const int64_t est_dense_bytes = n_kv_base * n_tokens * (hparams.n_head(0) * kq_b_elem_bytes + 4);
            if (mem_free > 0 && est_dense_bytes > (int64_t) mem_free) {
                LLAMA_LOG_ERROR("%s: Metal dense Inkling kq_b fallback requires %.2f GiB > %.2f GiB free GPU memory "
                        "(n_kv=%lld, n_tokens=%lld); long-context Inkling on Metal requires fused FlashAttention "
                        "(Metal 4 tensor ops, -fa on, unified KV or -np 1, -np <= 64, matching K/V dtypes in {f16, q8_0, q4_0}; "
                        "set LLAMA_INKLING_ALLOW_DENSE_KQ_B=1 to override)\n",
                        __func__, est_dense_bytes / double(1LL << 30), mem_free / double(1LL << 30),
                        (long long) n_kv_base, (long long) n_tokens);
                throw std::runtime_error("Metal dense Inkling kq_b fallback exceeds free GPU memory");
            }
        }
    }

    inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa, flash_create != nullptr);
    if (flash_create) {
        inp->flash_attention = { flash_backend, flash_create };
    }
    attn = inp->get_attn();
    recr = inp->get_recr();

    ggml_tensor * inp_pos = build_inp_pos();
    inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
        ggml_tensor * messages, const loop_layer * loop, bool messages_fused) {
    GGML_ASSERT(!messages == !loop);

    const auto & layer = model.layers[il];

    const bool    is_swa       = hparams.is_swa(il);
    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_head  = hparams.n_embd_head_k(il);
    const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

    // the message of controller ctrl, [n_embd, n_tokens]
    auto message = [&](int ctrl) {
        if (messages_fused) {
            return ggml_view_2d(ctx0, messages, n_embd, n_tokens, messages->nb[1], ctrl*messages->nb[2]);
        }
        return ggml_cont(ctx0, ggml_view_2d(ctx0, messages, n_embd, n_tokens, messages->nb[2], ctrl*messages->nb[1]));
    };

    ggml_tensor * state   = recr->mctx->get_r_l(il);
    ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

    ggml_tensor * inpSA = cur;

    // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
    ggml_tensor * inputs[3] = {cur, cur, cur};
    if (layer.attn_res_query) {
        bool attn_mix_fused = false;
        ggml_tensor * mixed = mix(blocks, cur, layer.attn_res_query, &attn_mix_fused);
        if (attn_mix_fused) {
            if (is_swa) {
                ggml_tensor * s = ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[1], 0);
                inputs[0] = inputs[1] = inputs[2] = s;
            } else {
                for (int j = 0; j < 3; ++j) {
                    inputs[j] = ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[1], j*mixed->nb[2]);
                }
            }
        } else {
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_cont(ctx0, ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], (is_swa ? 0 : j)*mixed->nb[1]));
            }
        }
    }

    if (messages) {
        // the messages make the Q, K and V inputs of a SWA layer differ
        for (int j = 0; j < 3; ++j) {
            inputs[j] = control(inputs[j], message(LOOP_CTRL_Q + j), *loop, LOOP_CTRL_Q + j);
        }
    }

    if (is_swa && messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
    } else if (is_swa) {
        inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }

    // self-attention
    {
        ggml_tensor * Qcur = build_lora_mm(layer.wq, inputs[0]);
        ggml_tensor * Kcur = convolve(build_lora_mm(layer.wk, inputs[1]), layer.attn_k_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
        ggml_tensor * Vcur = convolve(build_lora_mm(layer.wv, inputs[2]), layer.attn_v_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa);
        ggml_tensor * Rcur = build_lora_mm(layer.attn_rel, inputs[0]);

        Qcur = clip(ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), layer.attn_q_clip, hparams.f_qk_clip);
        Kcur = clip(ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), nullptr,           hparams.f_qk_clip);
        Vcur = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens), hparams.f_norm_rms_eps);
        Rcur = clip(ggml_reshape_3d(ctx0, Rcur, hparams.n_relative, n_head, n_tokens), layer.attn_rel_clip, hparams.f_relative_clip);

        if (!is_swa) {
            Qcur = ggml_mul(ctx0, Qcur, tau);
            Rcur = ggml_mul(ctx0, Rcur, tau);
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cb(Rcur, "Rcur", il);

        ggml_tensor * sinks = build_lora_mm(layer.attn_sink_proj, inputs[0]);

        if (inp->flash_attention.create && inp->self_k_pos) {
            // Fused Inkling FlashAttention handles distances >= extent as zero relative bias (-sink) directly,
            // avoiding the per-layer ggml_pad on attn_rel_bias.
            ggml_tensor * table = ggml_mul_mat(ctx0, layer.attn_rel_bias, Rcur);
            cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, table, sinks, 1.0f/sqrtf(n_embd_head), il, false);
        } else {
            // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x n_head
            ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);
            ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;

            const int64_t n_stream = (is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw)->ne[3];
            const int64_t n_kv     = rel_pos->ne[0];

            ggml_tensor * kq_b = nullptr;
            if (inkling_dense_backend && rel_mask_kernel) {
                const auto & mask = is_swa ? attn->self_attn_mask_swa : attn->self_attn_mask;
                if (!mask.rows) { kq_b = rel_mask_kernel(ctx0, table, sinks, rel_pos, mask.raw); }
            }
            const bool prebuilt_mask = kq_b != nullptr;
            if (prebuilt_mask) {
                ggml_backend_sched_set_tensor_backend(sched, kq_b, inkling_dense_backend);
            } else {
                // Shift the logits by the per-query sink logit, using a zero sink.
                table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens));
                table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));
                kq_b = ggml_get_rows(ctx0, table, rel_pos);
                kq_b = ggml_reshape_4d(ctx0, kq_b, n_head, n_kv, n_tokens/n_stream, n_stream);
                kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));
                cb(kq_b, "kq_b", il);
            }

            sinks = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);

            cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, kq_b, sinks, 1.0f/sqrtf(n_embd_head), il, prebuilt_mask);
        }

        ggml_tensor * attn_gate = build_lora_mm(layer.wqkv_gate, inputs[0]);
        if (inkling_dense_backend && mul_sigmoid_kernel) {
            if (auto * gated = mul_sigmoid_kernel(ctx0, cur, attn_gate)) {
                ggml_backend_sched_set_tensor_backend(sched, gated, inkling_dense_backend);
                cur = gated;
            } else {
                cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, attn_gate));
            }
        } else {
            cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, attn_gate));
        }
        cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa);
        cb(cur, "attn_out", il);
    }

    // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
    if (ip % hparams.attn_res_block_size == 0) {
        blocks.push_back(inpSA);
    } else {
        cur = ggml_add(ctx0, inpSA, cur);
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query), n_embd, n_tokens);
    if (messages) {
        cur = control(cur, message(LOOP_CTRL_FFN), *loop, LOOP_CTRL_FFN);
    }
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

    const auto * bank = expert_executor && loras->empty() ? expert_executor->inkling_core_layer(ip) : nullptr;
    cur = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            nullptr,
            nullptr,
            bank ? (*bank)[1] : layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            hparams.n_expert,
            hparams.n_expert_used(il),
            LLM_FFN_SITU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
            il,
            nullptr,
            bank ? (*bank)[0] : layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);

    cur = convolve(cur, layer.ffn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd);
    cb(cur, "ffn_out", il);

    cur = ggml_add(ctx0, ffn_inp, cur);
    cb(cur, "l_out", il);

    return cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks) {
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    if (cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = cur;
    }

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;
        if (inkling_dense_backend && softcap_trim_kernel) {
            if (auto * fused = softcap_trim_kernel(ctx0, cur, hparams.n_vocab_unpadded, cap)) {
                ggml_backend_sched_set_tensor_backend(sched, fused, inkling_dense_backend);
                cur = fused;
            } else {
                cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
                cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));
            }
        } else {
            cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
            cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));
        }
    } else {
        // drop the padded vocabulary
        cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

llama_model_inkling2::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inputs();

    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < n_layer; ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = cur;
        }
        cur = build_layer(il, il, cur, blocks);
    }

    build_output(cur, blocks);
}
