#pragma once

#include "ggml-backend.h"

// Private contract: CPU-prepared map in a bound output's existing extra space.
// Valid only for an exclusively owned single-expert 32-row op with immutable IDs.
enum { GGML_METAL_PREPARED_SINGLE_EXPERT = 0x41544531 };

// Set only on an immutable, private Inkling bank leaf (GGML_OP_NONE).
enum { GGML_METAL_INKLING_GROUP_BANK = 0x41544742 };

// The tag lives past the ordinary custom callback and its application data.
enum { GGML_METAL_HOST_EXPERT_BRIDGE = 0x41544532 };
enum { GGML_METAL_INKLING_REL_MASK = 0x41544534 };
enum { GGML_METAL_INKLING_CONV_RESIDUAL = 0x41544535 };
enum { GGML_METAL_INKLING_CLIP = 0x41544536 };
enum { GGML_METAL_INKLING_SITU = 0x41544537 };
enum { GGML_METAL_INKLING_CONTROL = 0x41544538 };
// Hint on a standard ADD node in the projected Inkling DSpark head.
enum { GGML_METAL_INKLING_VOCAB_ADD = 0x41545732 };
enum { GGML_METAL_INKLING_PREFILL_ATTN = 0x41544539 };
enum { GGML_METAL_INKLING_FLASH_ATTN = 0x4154453A };
enum { GGML_METAL_INKLING_MIX = 0x4154453B };
enum { GGML_METAL_INKLING_CTRL_DOWN = 0x4154453C };
enum { GGML_METAL_INKLING_CTRL_APPLY = 0x4154453D };
enum { GGML_METAL_INKLING_DSPARK_ARGMAX = 0x4154453E };
enum { GGML_METAL_INKLING_MUL_SIGMOID = 0x4154453F };
enum { GGML_METAL_INKLING_SOFTCAP_TRIM = 0x41544540 };
enum { GGML_METAL_INKLING_CONV_FUSED = 0x41544541 };
enum { GGML_METAL_INKLING_MIX_NORM = 0x41544542 };
enum { GGML_METAL_INKLING_MOE_FFN = 0x41544543 };

enum { GGML_METAL_INKLING_MOE_MAX_TOKENS = 11 };

// Shared layout constants between host Metal allocator/encoders and kernels.
// Split-K decode partition layout: 128 output floats + m + s + 2 pad floats = 132 floats (528 B, 16-byte aligned for float4).
enum { GGML_METAL_INKLING_FA_SPLIT_STRIDE  = 132 };
enum { GGML_METAL_INKLING_FA_SPLIT_MAX_NWG = 128 };
// Depth-attention mix max input states: GGML_MAX_SRC (10) - 2 = 8 states (+ 1 query = 9 < GGML_MAX_SRC).
enum { GGML_METAL_INKLING_MIX_MAX_STATES   = 8   };

static inline size_t ggml_metal_inkling_fa_tensor_smem(uint32_t q_tile, bool has_skv) {
    // Matches fa_tensor.metal layout: sq[Q*128 half] + ss[Q*64 float] + sp[Q*64 half] +
    // sr[Q float] + sf[4 int] + kpos_tg[64 int] + kseq_tg[64 uint64] + optional skv[64*128 half].
    const size_t bytes =
        (size_t) q_tile * 128u * 2u +
        (size_t) q_tile *  64u * 4u +
        (size_t) q_tile *  64u * 2u +
        (size_t) q_tile *        4u +
        4u  * 4u +
        64u * 4u +
        64u * 8u +
        (has_skv ? (size_t) 64u * 128u * 2u : 0u);
    return (bytes + 255u) & ~(size_t) 255u;
}

static inline bool ggml_metal_is_host_expert(const struct ggml_tensor * node) {
    return node->op == GGML_OP_CUSTOM &&
        node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] == GGML_METAL_HOST_EXPERT_BRIDGE;
}

// Internal proc-address implementations; callers must obey exclusive ownership.
void * ggml_backend_metal_buffer_get_host_base(ggml_backend_buffer_t buffer);
bool ggml_backend_metal_prepare_single_expert(struct ggml_tensor * op);
