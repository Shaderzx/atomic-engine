#include "ggml-metal-expert.h"
#include "ggml-metal-route-publication.h"
#include "ggml-metal.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-device.h"
#include "ggml-metal-expert-buffer.h"
#include "ggml-metal-fusion.h"
#include "ggml-metal-context.h"
#include "ggml-metal-ops.h"
#include "ggml-metal-tuning.h"

#include <cmath>
#include <cstring>
#include <mutex>
#include <string>

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
static int g_devices = 1;

// forward declaration
static bool ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);
static void ggml_backend_metal_expert_free_buffer(ggml_backend_buffer_t buffer);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

static void ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

static void * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    return ggml_metal_buffer_get_base(ctx);
}

static void ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_shared_set_tensor_n(ggml_backend_buffer_t buffer, struct ggml_tensor * const * tensors, const void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor_n(ctx, tensors, data, offsets, sizes, n_tensors);
}

static void ggml_backend_metal_buffer_shared_get_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * tensors, void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor_n(ctx, tensors, data, offsets, sizes, n_tensors);
}

static bool ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    if (!ggml_backend_buffer_is_metal(src->buffer)) {
        return false;
    }

    return ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

static bool ggml_backend_metal_buffer_shared_cpy_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * srcs, struct ggml_tensor * const * dsts, size_t n_tensors) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    for (size_t i = 0; i < n_tensors; ++i) {
        if (!ggml_backend_buffer_is_metal(srcs[i]->buffer)) {
            return false;
        }
    }

    return ggml_metal_buffer_cpy_tensor_n(ctx, srcs, dsts, n_tensors);
}

static void ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

static ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset         = */ NULL,
    /* .set_tensor_n  = */ ggml_backend_metal_buffer_shared_set_tensor_n,
    /* .get_tensor_n  = */ ggml_backend_metal_buffer_shared_get_tensor_n,
    /* .cpy_tensor_n  = */ ggml_backend_metal_buffer_shared_cpy_tensor_n,
};

// private buffer

static void ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

static void * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    return ggml_metal_buffer_get_base(ctx);
}

static void ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_private_set_tensor_n(ggml_backend_buffer_t buffer, struct ggml_tensor * const * tensors, const void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor_n(ctx, tensors, data, offsets, sizes, n_tensors);
}

static void ggml_backend_metal_buffer_private_get_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * tensors, void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor_n(ctx, tensors, data, offsets, sizes, n_tensors);
}

static bool ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    if (!ggml_backend_buffer_is_metal(src->buffer)) {
        return false;
    }

    return ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

static bool ggml_backend_metal_buffer_private_cpy_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * srcs, struct ggml_tensor * const * dsts, size_t n_tensors) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    for (size_t i = 0; i < n_tensors; ++i) {
        if (!ggml_backend_buffer_is_metal(srcs[i]->buffer)) {
            return false;
        }
    }

    return ggml_metal_buffer_cpy_tensor_n(ctx, srcs, dsts, n_tensors);
}

static void ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

static ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_private_clear,
    /* .reset         = */ NULL,
    /* .set_tensor_n  = */ ggml_backend_metal_buffer_private_set_tensor_n,
    /* .get_tensor_n  = */ ggml_backend_metal_buffer_private_get_tensor_n,
    /* .cpy_tensor_n  = */ ggml_backend_metal_buffer_private_cpy_tensor_n,
};

static bool ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_buffer_private_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_expert_free_buffer;
}

// Synchronous writes are permitted only while the caller exclusively owns the
// buffer and no GPU work references it. Private storage never exposes a pointer.
void * ggml_backend_metal_buffer_get_host_base(ggml_backend_buffer_t buffer) {
    if (!buffer || !ggml_backend_buffer_is_metal(buffer)) {
        return nullptr;
    }
    auto ctx = (ggml_metal_buffer_t) buffer->context;
    return ggml_metal_buffer_is_shared(ctx) ? ggml_metal_buffer_get_base(ctx) : nullptr;
}

//
// buffer types
//

struct ggml_backend_metal_buffer_type {
    int device;
    std::string name;
};

struct ggml_backend_metal_buffer_type_deleter {
    void operator()(ggml_backend_metal_buffer_type * ctx) const {
        delete ctx;
    }
};

typedef std::unique_ptr<ggml_backend_metal_buffer_type, ggml_backend_metal_buffer_type_deleter> ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
static ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, bool shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);

    if (res == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        return NULL;
    }

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    return ggml_backend_buffer_init(buft, buf_i, res, size);
}

static size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    switch (tensor->op) {
        case GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_rows(tensor);
            } break;
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
            {
                res *= 2;
            } break;
        case GGML_OP_TOP_K:
            {
                res = 2*sizeof(int32_t)*ggml_nelements(tensor->src[0]);
            } break;
        case GGML_OP_CONV_2D:
            {
                res += ggml_metal_op_conv_2d_extra_tmp(tensor);
            } break;
        case GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } break;
        case GGML_OP_CUSTOM:
            {
                const int32_t tag = tensor->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1];
                if (tag == GGML_METAL_INKLING_FLASH_ATTN &&
                        tensor->src[0] && tensor->src[0]->ne[2] < 16) {
                    res = GGML_PAD(res, 64) +
                        (size_t) GGML_METAL_INKLING_FA_SPLIT_MAX_NWG * 24u *
                        (size_t) tensor->src[0]->ne[2] * GGML_METAL_INKLING_FA_SPLIT_STRIDE * sizeof(float);
                } else if (tag == GGML_METAL_INKLING_DSPARK_ARGMAX) {
                    res = GGML_PAD(res, 16) + 256u * 4u * sizeof(uint32_t);
                } else if (tag == GGML_METAL_INKLING_MOE_FFN && tensor->src[2]) {
                    const size_t n_ff = (size_t)(tensor->src[2]->ne[1] / 2);
                    res = GGML_PAD(res, 64) +
                        n_ff * 8u * (size_t)tensor->ne[2] * sizeof(float);
                }
            } break;
        default:
            break;
    }

    return res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

static const char * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx =
                new ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .alloc_buffer_n      = */ NULL,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .get_alloc_size_n    = */ NULL,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// default (private) buffer type

static const char * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, false);
}

static size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = new ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .alloc_buffer_n      = */ NULL,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .get_alloc_size_n    = */ NULL,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// mapped buffer type

static const char * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = new ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_mapped_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_mapped_alloc_buffer,
                    /* .alloc_buffer_n      = */ NULL,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .get_alloc_size_n    = */ NULL,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// backend

static const char * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_get_name(ctx);
}

static void ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    free(backend);
}

static void ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

static void ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

static bool ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_metal(src->buffer) || !ggml_backend_buffer_is_metal(dst->buffer)) {
        return false;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    return ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

static enum ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_graph_compute(ctx, cgraph);
}

static void ggml_backend_metal_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_record(ctx, ev);
}

static void ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

static void ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(params && params->add_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_add_alloc_deps(params->user_data, params->add_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

static void ggml_backend_metal_set_n_cb(ggml_backend_t backend, int n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

static ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

static ggml_guid_t ggml_backend_metal_guid(void) {
    static ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    return &guid;
}

ggml_backend_t ggml_backend_metal_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;
}

bool ggml_backend_is_metal(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_supports_family(ctx, family);
}

void ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

static const char * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return props_dev->name;
}

static const char * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_get_props(ctx_dev)->desc;
}

static void ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, free, total);
}

static enum ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ true,
        /* .mmap_support         = */ true,
    };
}

static ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return props_dev->use_shared_buffers ? ggml_backend_metal_buffer_type_shared(props_dev->device) : ggml_backend_metal_buffer_type_private(props_dev->device);
}

static ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

static bool ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_supports_op(ctx_dev, op);
}

#include "ggml-metal-expert-buffer.cpp.inc"

static bool ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name ||
        buft->iface.get_name == ggml_backend_metal_expert_buffer_name);

    GGML_UNUSED(dev);
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    return ev;
}

static void ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

static void ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

static ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

struct ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

typedef struct ggml_backend_metal_reg * ggml_backend_metal_reg_t;

static ggml_backend_metal_reg_t ggml_backend_metal_reg_init(void) {
    ggml_backend_metal_reg_t ctx = new struct ggml_backend_metal_reg;

    return ctx;
}

static void ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

struct ggml_backend_metal_reg_deleter {
    void operator()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

typedef std::unique_ptr<struct ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

static const char * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_metal_reg_device_count(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature g_ggml_backend_metal_features[] = {
#if defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { NULL, NULL },
};

static ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    return g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
static void ggml_backend_metal_tuning_set_fa_vec_override(int Q, int NE) {
    ggml_metal_tuning::fa_vec_set_override({ (int8_t) Q, (int8_t) NE });
}

static void ggml_backend_metal_tuning_clear_fa_vec_override(void) {
    ggml_metal_tuning::fa_vec_clear_override();
}

static int ggml_backend_metal_tuning_fa_vec_ne11_bucket(int64_t ne11) {
    return ggml_metal_tuning::fa_vec_ne11_bucket(ne11);
}

static int ggml_backend_metal_tuning_fa_vec_ne01_bucket(int64_t ne01) {
    return ggml_metal_tuning::fa_vec_ne01_bucket(ne01);
}

static int ggml_backend_metal_tuning_fa_vec_baseline_ne(int dk, int dv) {
    return ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

static const char * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
typedef void * ggml_backend_fusion_t;

static ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    return ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

static void ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((struct ggml_metal_fusion_info *) finfo);
}

static void ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((struct ggml_metal_fusion_info *) finfo);
}

static int ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, const char ** labels, uint64_t * counts, int n) {
    return ggml_metal_fusion_info_stats_get((struct ggml_metal_fusion_info *) finfo, labels, counts, n);
}

static void ggml_backend_metal_fusion_set_enabled(ggml_backend_fusion_t finfo, bool enabled) {
    ggml_metal_fusion_info_set_enabled((struct ggml_metal_fusion_info *) finfo, enabled);
}

static bool ggml_backend_metal_last_graph_timing(ggml_backend_t backend, uint64_t * gpu_ns, uint64_t * buffers) {
    GGML_ASSERT(ggml_backend_is_metal(backend));
    return ggml_metal_last_graph_timing((ggml_metal_t) backend->context, gpu_ns, buffers);
}

static enum ggml_status ggml_backend_metal_compute_graph_prefix(ggml_backend_t backend, ggml_cgraph * graph, int nodes) {
    GGML_ASSERT(ggml_backend_is_metal(backend));
    return ggml_metal_graph_compute_prefix((ggml_metal_t) backend->context, graph, nodes);
}

static bool ggml_backend_metal_wait_graph_prefix(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));
    return ggml_metal_wait_graph_prefix((ggml_metal_t) backend->context);
}

static bool ggml_backend_metal_host_expert_call(ggml_tensor * op) {
    const auto shared = [](const ggml_tensor * t) {
        return t && t->data && ggml_backend_metal_buffer_get_host_base(t->buffer);
    };
    if (!shared(op) || !shared(op->src[0]) || !shared(op->src[1])) {
        GGML_LOG_ERROR("%s: expected shared Metal expert tensors\n", __func__);
        return false;
    }
    ggml_custom_op_params params;
    std::memcpy(&params, op->op_params, sizeof(params));
    params.fun(op, 0, 1, params.userdata);
    return true;
}

static bool ggml_backend_metal_host_expert_bridge_v1(ggml_backend_t backend, ggml_tensor * op) {
    if (!ggml_backend_is_metal(backend) || !op || op->op != GGML_OP_CUSTOM) { return false; }
    const auto dev = (ggml_metal_device_t) ggml_backend_get_device(backend)->context;
    if (!ggml_metal_device_get_props(dev)->use_shared_buffers) { return false; }
    ggml_custom_op_params params;
    std::memcpy(&params, op->op_params, sizeof(params));
    if (params.n_tasks != 1 || !params.fun || !op->src[0] || !op->src[1] || op->src[2]) { return false; }
    op->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_HOST_EXPERT_BRIDGE;
    ggml_metal_set_host_expert_bridge((ggml_metal_t) backend->context, ggml_backend_metal_host_expert_call);
    return true;
}

static void ggml_backend_metal_mark_inkling_group_bank(ggml_tensor * weight) {
    if (weight->op == GGML_OP_NONE && weight->ne[2] == 128 && ggml_is_contiguous(weight)) {
        weight->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_GROUP_BANK;
    }
}

static ggml_tensor * ggml_backend_metal_inkling_control_v1(ggml_context * ctx, ggml_tensor * cur,
        ggml_tensor * message, ggml_tensor * gain, ggml_tensor * forget) {
    ggml_tensor * args[] = {cur, message, gain, forget};
    if (cur->ne[0] <= 0 || cur->ne[1] <= 0 || !ggml_is_matrix(cur)) { return nullptr; }
    for (auto * arg : args) {
        if (arg->type != GGML_TYPE_F32 || !ggml_are_same_shape(arg, cur) || !ggml_is_contiguous(arg)) {
            return nullptr;
        }
    }
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], cur->ne[1], 1, 1, args, 4,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling controller requires Metal"); }, 1, nullptr);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_CONTROL;
    return node;
}

static ggml_tensor * ggml_backend_metal_inkling_mix_v1(ggml_context * ctx,
        ggml_tensor * const * states, int n_states, ggml_tensor * query) {
    static_assert(GGML_METAL_INKLING_MIX_MAX_STATES + 1 < GGML_MAX_SRC,
            "Inkling mix inputs (states + query) must stay strictly below GGML_MAX_SRC");
    if (!states || !query || n_states < 1 || n_states > GGML_METAL_INKLING_MIX_MAX_STATES ||
            query->type != GGML_TYPE_F32 || !ggml_is_contiguous(query) ||
            query->ne[0] <= 0 || query->ne[0] % 4 != 0 ||
            query->ne[1] < 1 || query->ne[1] > 4 || query->ne[2] != 1 || query->ne[3] != 1) {
        return nullptr;
    }
    const int64_t n_embd   = query->ne[0];
    const int64_t n_q      = query->ne[1];
    const int64_t n_tokens = states[0] ? states[0]->ne[1] : 0;
    if (n_tokens <= 0) { return nullptr; }
    ggml_tensor * args[GGML_METAL_INKLING_MIX_MAX_STATES + 1] = {};
    for (int i = 0; i < n_states; ++i) {
        auto * s = states[i];
        if (!s || s->type != GGML_TYPE_F32 || !ggml_is_contiguous(s) ||
                s->ne[0] != n_embd || s->ne[1] != n_tokens || s->ne[2] != 1 || s->ne[3] != 1) {
            return nullptr;
        }
        args[i] = s;
    }
    args[n_states] = query;
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_tokens, n_q, 1, args, n_states + 1,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling depth mix requires Metal"); }, 1, nullptr);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 3] = n_states;
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 2] = int32_t(n_q);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_MIX;
    return node;
}

static ggml_tensor * ggml_backend_metal_inkling_mix_norm_v1(ggml_context * ctx,
        ggml_tensor * const * states, int n_states, ggml_tensor * query,
        ggml_tensor * const * norm_weights, float norm_eps) {
    if (!states || !query || !norm_weights || !std::isfinite(norm_eps) || norm_eps < 0.0f ||
            n_states < 1 || n_states > GGML_METAL_INKLING_MIX_MAX_STATES ||
            query->type != GGML_TYPE_F32 || !ggml_is_contiguous(query) ||
            query->ne[0] <= 0 || query->ne[0] > 4096 || query->ne[0] % 4 != 0 ||
            query->ne[1] < 1 || query->ne[1] > 3 || query->ne[2] != 1 || query->ne[3] != 1) {
        return nullptr;
    }
    const int64_t n_embd   = query->ne[0];
    const int64_t n_q      = query->ne[1];
    const int64_t n_tokens = states[0] ? states[0]->ne[1] : 0;
    if (n_tokens <= 0 || n_states + 1 + n_q >= GGML_MAX_SRC) { return nullptr; }
    ggml_tensor * args[GGML_MAX_SRC] = {};
    for (int i = 0; i < n_states; ++i) {
        auto * s = states[i];
        if (!s || s->type != GGML_TYPE_F32 || !ggml_is_contiguous(s) ||
                s->ne[0] != n_embd || s->ne[1] != n_tokens || s->ne[2] != 1 || s->ne[3] != 1) {
            return nullptr;
        }
        args[i] = s;
    }
    args[n_states] = query;
    for (int64_t q = 0; q < n_q; ++q) {
        auto * nw = norm_weights[q];
        if (!nw || nw->type != GGML_TYPE_F32 || !ggml_is_contiguous(nw) ||
                nw->ne[0] != n_embd || ggml_nelements(nw) != n_embd) {
            return nullptr;
        }
        args[n_states + 1 + q] = nw;
    }
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_tokens, n_q, 1, args, n_states + 1 + (int)n_q,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling depth mix+norm requires Metal"); }, 1, nullptr);
    ggml_set_op_params_f32(node, GGML_MAX_OP_PARAMS / sizeof(int32_t) - 4, norm_eps);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 3] = n_states;
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 2] = int32_t(n_q);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_MIX_NORM;
    return node;
}

static ggml_tensor * ggml_backend_metal_inkling_ctrl_down_v1(ggml_context * ctx,
        ggml_tensor * cur, ggml_tensor * message, ggml_tensor * w_down) {
    if (!cur || !message || !w_down ||
            cur->type != GGML_TYPE_F32 || message->type != GGML_TYPE_F32 || w_down->type != GGML_TYPE_Q8_0 ||
            !ggml_is_contiguous(cur) || !ggml_is_contiguous(message) || !ggml_is_contiguous(w_down) ||
            !ggml_are_same_shape(cur, message) || !ggml_is_matrix(cur) ||
            cur->ne[0] <= 0 || cur->ne[0] % 32 != 0 || cur->ne[1] <= 0 ||
            w_down->ne[0] != 2 * cur->ne[0] || w_down->ne[1] <= 0 || w_down->ne[1] > 256 ||
            w_down->ne[2] != 1 || w_down->ne[3] != 1 ||
            w_down->nb[1] != ggml_row_size(GGML_TYPE_Q8_0, w_down->ne[0])) {
        return nullptr;
    }
    ggml_tensor * args[] = {cur, message, w_down};
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, w_down->ne[1], cur->ne[1], 1, 1, args, 3,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling ctrl down requires Metal"); }, 1, nullptr);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_CTRL_DOWN;
    return node;
}

static ggml_tensor * ggml_backend_metal_inkling_ctrl_apply_v1(ggml_context * ctx,
        ggml_tensor * cur, ggml_tensor * message, ggml_tensor * gate,
        ggml_tensor * w_gain, ggml_tensor * w_forget) {
    if (!cur || !message || !gate || !w_gain || !w_forget ||
            cur->type != GGML_TYPE_F32 || message->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 ||
            w_gain->type != GGML_TYPE_Q8_0 || w_forget->type != GGML_TYPE_Q8_0 ||
            !ggml_is_contiguous(cur) || !ggml_is_contiguous(message) || !ggml_is_contiguous(gate) ||
            !ggml_is_contiguous(w_gain) || !ggml_is_contiguous(w_forget) ||
            !ggml_are_same_shape(cur, message) || !ggml_is_matrix(cur) || !ggml_is_matrix(gate) ||
            cur->ne[0] <= 0 || cur->ne[1] <= 0 || gate->ne[1] != cur->ne[1] ||
            gate->ne[0] <= 0 || gate->ne[0] > 256 || gate->ne[0] % 32 != 0 ||
            w_gain->ne[0] != gate->ne[0] || w_gain->ne[1] != cur->ne[0] || w_gain->ne[2] != 1 || w_gain->ne[3] != 1 ||
            !ggml_are_same_shape(w_gain, w_forget) ||
            w_gain->nb[1] != ggml_row_size(GGML_TYPE_Q8_0, w_gain->ne[0]) ||
            w_forget->nb[1] != ggml_row_size(GGML_TYPE_Q8_0, w_forget->ne[0])) {
        return nullptr;
    }
    ggml_tensor * args[] = {cur, message, gate, w_gain, w_forget};
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], cur->ne[1], 1, 1, args, 5,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling ctrl apply requires Metal"); }, 1, nullptr);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_CTRL_APPLY;
    return node;
}

static bool ggml_backend_metal_inkling_dspark_argmax_ready_v2(ggml_backend_t backend,
        int64_t rank, ggml_type w2_type) {
    if (!ggml_backend_is_metal(backend) || rank <= 0 || rank > 512 || (rank % 32) != 0) {
        return false;
    }
    if (w2_type == GGML_TYPE_Q4_0 || w2_type == GGML_TYPE_Q8_0) {
        return true;
    }
    if (w2_type == GGML_TYPE_BF16) {
        const auto dev = (ggml_metal_device_t) ggml_backend_get_device(backend)->context;
        const auto * props = ggml_metal_device_get_props(dev);
        return props && props->has_bfloat;
    }
    return false;
}

static ggml_tensor * ggml_backend_metal_inkling_dspark_argmax_v2(ggml_context * ctx,
        ggml_tensor * w2, ggml_tensor * w2_b, ggml_tensor * base_row,
        ggml_tensor * latent, ggml_tensor * shortlist_ids, float cap) {
    if (!w2 || !w2_b || !base_row || !latent ||
            !std::isfinite(cap) || cap < 0.0f ||
            (w2->type != GGML_TYPE_Q4_0 && w2->type != GGML_TYPE_Q8_0 && w2->type != GGML_TYPE_BF16) ||
            !ggml_is_contiguous(w2) || w2->ne[2] != 1 || w2->ne[3] != 1 ||
            w2->ne[0] <= 0 || w2->ne[0] > 512 || w2->ne[0] % 32 != 0 || w2->ne[1] <= 0 ||
            w2->nb[1] != ggml_row_size(w2->type, w2->ne[0]) ||
            w2_b->type != GGML_TYPE_F32 || !ggml_is_contiguous(w2_b) || ggml_nelements(w2_b) != w2->ne[1] ||
            base_row->type != GGML_TYPE_F32 || !ggml_is_contiguous(base_row) || ggml_nelements(base_row) != w2->ne[1] ||
            latent->type != GGML_TYPE_F32 || !ggml_is_contiguous(latent) || ggml_nelements(latent) != w2->ne[0]) {
        return nullptr;
    }
    if (shortlist_ids && (shortlist_ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(shortlist_ids) ||
            ggml_nelements(shortlist_ids) != w2->ne[1])) {
        return nullptr;
    }
    ggml_tensor * args[] = {w2, w2_b, base_row, latent, shortlist_ids};
    // Output [3]: elem 0 = raw token ID (or -1 on all-NaN), elem 1 = safe in-bounds index (0 on all-NaN),
    // elem 2 = float bits of top1 - top2 softcapped logit margin (0 on tie / all-NaN).
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_I32, 3, 1, 1, 1, args, shortlist_ids ? 5 : 4,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling DSpark argmax requires Metal"); }, 1, nullptr);
    ggml_set_op_params_f32(node, GGML_MAX_OP_PARAMS / sizeof(int32_t) - 2, cap);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_DSPARK_ARGMAX;
    return node;
}

static ggml_tensor * ggml_backend_metal_inkling_mul_sigmoid_v1(ggml_context * ctx,
        ggml_tensor * x, ggml_tensor * gate) {
    if (!x || !gate || x->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(x) || !ggml_is_contiguous(gate) || !ggml_are_same_shape(x, gate) ||
            ggml_nelements(x) <= 0 || ggml_nelements(x) % 4 != 0 ||
            ggml_nelements(x) / 4 > (int64_t) UINT32_MAX) {
        return nullptr;
    }
    ggml_tensor * args[] = {x, gate};
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2], x->ne[3], args, 2,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling mul_sigmoid requires Metal"); }, 1, nullptr);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_MUL_SIGMOID;
    return node;
}

static ggml_tensor * ggml_backend_metal_inkling_softcap_trim_v1(ggml_context * ctx,
        ggml_tensor * logits, int64_t n_out, float cap) {
    if (!logits || logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits) ||
            logits->ne[2] != 1 || logits->ne[3] != 1 || logits->ne[1] <= 0 ||
            n_out <= 0 || n_out > logits->ne[0] || !std::isfinite(cap) || cap <= 0.0f) {
        return nullptr;
    }
    ggml_tensor * args[] = {logits};
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, n_out, logits->ne[1], 1, 1, args, 1,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling softcap_trim requires Metal"); }, 1, nullptr);
    ggml_set_op_params_f32(node, GGML_MAX_OP_PARAMS / sizeof(int32_t) - 2, cap);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_SOFTCAP_TRIM;
    return node;
}

static ggml_tensor * ggml_backend_metal_inkling_situ_v1(ggml_context * ctx, ggml_tensor * gate_up,
        float beta, float linear_beta) {
    if (gate_up->type != GGML_TYPE_F32 || !ggml_is_contiguous(gate_up) ||
            gate_up->ne[0] <= 0 || gate_up->ne[0] > 16384 || gate_up->ne[0] % 8 != 0 ||
            gate_up->ne[1] <= 0 || gate_up->ne[3] != 1 || ggml_nrows(gate_up) <= 0 ||
            beta <= 0.0f || linear_beta < 0.0f || !std::isfinite(beta) || !std::isfinite(linear_beta)) {
        return nullptr;
    }
    ggml_tensor * args[] = {gate_up};
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, gate_up->ne[0] / 2, gate_up->ne[1], gate_up->ne[2], 1,
            args, 1, +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling SiTU requires Metal"); }, 1, nullptr);
    ggml_set_op_params_f32(node, GGML_MAX_OP_PARAMS / sizeof(int32_t) - 3, beta);
    ggml_set_op_params_f32(node, GGML_MAX_OP_PARAMS / sizeof(int32_t) - 2, linear_beta);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_SITU;
    return node;
}

static void ggml_backend_metal_mark_inkling_vocab_add(ggml_tensor * node) {
    if (node->op == GGML_OP_ADD && node->type == GGML_TYPE_F32 &&
            ggml_is_vector(node) && ggml_are_same_shape(node->src[0], node->src[1])) {
        node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_VOCAB_ADD;
    }
}

static ggml_tensor * ggml_backend_metal_inkling_clip_v1(ggml_context * ctx, ggml_tensor * cur,
        ggml_tensor * scale, float threshold) {
    if (cur->type != GGML_TYPE_F32 || !ggml_is_contiguous(cur) ||
            (cur->ne[0] != 16 && cur->ne[0] != 128) || cur->ne[3] != 1) {
        return nullptr;
    }
    if (scale && (scale->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(scale) || scale->ne[0] != cur->ne[0] || scale->ne[1] != cur->ne[1] ||
            scale->ne[2] != 1 || scale->ne[3] != 1)) {
        return nullptr;
    }
    ggml_tensor * args[] = {cur, scale};
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], cur->ne[1], cur->ne[2], 1,
            args, scale ? 2 : 1,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling attention clipping requires Metal"); }, 1, nullptr);
    ggml_set_op_params_f32(node, GGML_MAX_OP_PARAMS / sizeof(int32_t) - 2, threshold);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_CLIP;
    return node;
}

static ggml_tensor * ggml_backend_metal_inkling_rel_mask_v1(ggml_context * ctx, ggml_tensor * table,
        ggml_tensor * sinks, ggml_tensor * distances, ggml_tensor * causal) {
    if (table->type != GGML_TYPE_F32 || sinks->type != GGML_TYPE_F32 || distances->type != GGML_TYPE_I32 ||
            causal->type != GGML_TYPE_F16 || !ggml_are_same_shape(distances, causal) ||
            !ggml_is_contiguous(table) || !ggml_is_contiguous(sinks) ||
            !ggml_is_contiguous(distances) || !ggml_is_contiguous(causal) ||
            causal->ne[2] != 1 || causal->ne[3] != 1 || table->ne[3] != 1 ||
            table->ne[2] != causal->ne[1] || sinks->ne[0] != table->ne[1] ||
            sinks->ne[1] != table->ne[2] || ggml_nrows(sinks) != sinks->ne[1]) {
        return nullptr;
    }
    ggml_tensor * args[] = {table, sinks, distances, causal};
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F16, causal->ne[0], causal->ne[1], table->ne[1], 1,
            args, 4, +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling relative mask requires Metal"); }, 1, nullptr);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_REL_MASK;
    return node;
}

static ggml_tensor * ggml_backend_metal_inkling_prefill_attention_v1(ggml_context * ctx,
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask, float scale) {
    if (mask->op != GGML_OP_CUSTOM ||
            mask->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] != GGML_METAL_INKLING_REL_MASK ||
            q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_Q4_0 || v->type != k->type ||
            q->ne[0] != 128 || q->ne[1] != 24 || q->ne[3] != 1 || !ggml_is_contiguous(q) ||
            k->ne[0] != 128 || k->ne[1] != 8 || k->ne[3] != 1 || !ggml_are_same_shape(k, v) ||
            !ggml_is_contiguous_1(k) || !ggml_is_contiguous_1(v) || k->nb[0] != ggml_type_size(GGML_TYPE_Q4_0) ||
            v->nb[0] != ggml_type_size(GGML_TYPE_Q4_0) || k->nb[1] != 4*k->nb[0] || v->nb[1] != 4*v->nb[0] ||
            k->ne[2] != mask->ne[0] || q->ne[2] != mask->ne[1] || mask->ne[2] != q->ne[1] ||
            k->ne[2] > 4096 || q->ne[2] > 1024 || k->ne[2] % 64 || q->ne[2] % 16) {
        return nullptr;
    }
    ggml_tensor * args[] = {q, k, v, mask->src[0], mask->src[1], mask->src[2], mask->src[3]};
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, q->ne[0]*q->ne[1], q->ne[2], 1, 1, args, 7,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling prompt attention requires Metal"); }, 1, nullptr);
    ggml_set_op_params_f32(node, GGML_MAX_OP_PARAMS / sizeof(int32_t) - 2, scale);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_PREFILL_ATTN;
    ggml_set_name(node, "inkling_prefill_attention");
    return node;
}

static bool ggml_backend_metal_inkling_flash_attn_ready_v1(ggml_backend_t backend, int64_t n_q, ggml_type kv_type) {
    if (!ggml_backend_is_metal(backend)) { return false; }
    const auto dev = (ggml_metal_device_t) ggml_backend_get_device(backend)->context;
    const auto * props = ggml_metal_device_get_props(dev);
    if (!props || props->max_theadgroup_memory_size < 32768) { return false; }
    if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_Q8_0 && kv_type != GGML_TYPE_Q4_0) { return false; }
    return n_q < 16 || props->has_tensor;
}

static ggml_tensor * ggml_backend_metal_inkling_flash_attn_v2(ggml_context * ctx,
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * table, ggml_tensor * sinks, ggml_tensor * q_pos, ggml_tensor * k_pos,
        ggml_tensor * gate, float scale, int32_t extent, int32_t n_swa) {
    if (!q || !k || !v || !table || !sinks || !q_pos || !k_pos ||
            extent <= 0 || n_swa < 0 || q->ne[2] <= 0 || k->ne[2] <= 0 ||
            q->type != GGML_TYPE_F32 || q->ne[0] != 128 || q->ne[1] != 24 || q->ne[3] != 1 || !ggml_is_contiguous(q) ||
            (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0 && k->type != GGML_TYPE_Q4_0) || v->type != k->type ||
            k->ne[0] != 128 || k->ne[1] != 8 || k->ne[3] != 1 || (k->ne[2] % 64) != 0 || !ggml_are_same_shape(k, v) ||
            !ggml_is_contiguous_1(k) || !ggml_is_contiguous_1(v) ||
            table->type != GGML_TYPE_F32 || table->ne[0] != extent || table->ne[1] != 24 || table->ne[2] != q->ne[2] || table->ne[3] != 1 || !ggml_is_contiguous(table) ||
            sinks->type != GGML_TYPE_F32 || sinks->ne[0] != 24 || sinks->ne[1] != q->ne[2] || ggml_nrows(sinks) != q->ne[2] || !ggml_is_contiguous(sinks) ||
            q_pos->type != GGML_TYPE_I32 || q_pos->ne[0] != 2 || q_pos->ne[1] != q->ne[2] || q_pos->ne[2] != 1 || q_pos->ne[3] != 1 || !ggml_is_contiguous(q_pos) ||
            k_pos->type != GGML_TYPE_I32 || k_pos->ne[0] != 3 || k_pos->ne[1] != k->ne[2] || k_pos->ne[2] != 1 || k_pos->ne[3] != 1 || !ggml_is_contiguous(k_pos)) {
        return nullptr;
    }
    if (gate && (q->ne[2] >= 16 || gate->type != GGML_TYPE_F32 || !ggml_is_contiguous(gate) ||
            gate->ne[0] != q->ne[0] * q->ne[1] || gate->ne[1] != q->ne[2] ||
            gate->ne[2] != 1 || gate->ne[3] != 1)) {
        return nullptr;
    }
    const size_t expected_nb1 = ggml_row_size(k->type, 128);
    if (k->nb[1] != expected_nb1 || v->nb[1] != expected_nb1) {
        return nullptr;
    }
    ggml_tensor * args[] = {q, k, v, table, sinks, q_pos, k_pos, gate};
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, q->ne[0]*q->ne[1], q->ne[2], 1, 1, args, gate ? 8 : 7,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling fused flash attention requires Metal"); }, 1, nullptr);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 4] = n_swa;
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 3] = extent;
    ggml_set_op_params_f32(node, GGML_MAX_OP_PARAMS / sizeof(int32_t) - 2, scale);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_FLASH_ATTN;
    ggml_set_name(node, gate ? "inkling_flash_attention_gated" : "inkling_flash_attention");
    return node;
}

static ggml_tensor * ggml_backend_metal_inkling_flash_attn_v1(ggml_context * ctx,
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * table, ggml_tensor * sinks, ggml_tensor * q_pos, ggml_tensor * k_pos,
        float scale, int32_t extent, int32_t n_swa) {
    return ggml_backend_metal_inkling_flash_attn_v2(ctx, q, k, v, table, sinks, q_pos, k_pos, nullptr, scale, extent, n_swa);
}

static ggml_tensor * ggml_backend_metal_inkling_moe_ffn_v1(ggml_context * ctx,
        ggml_tensor * cur, ggml_tensor * ids, ggml_tensor * gate_up, ggml_tensor * down,
        float beta, float linear_beta) {
    if (!cur || !ids || !gate_up || !down ||
            !std::isfinite(beta) || beta <= 0.0f || !std::isfinite(linear_beta) || linear_beta < 0.0f ||
            cur->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
            !ggml_is_contiguous(cur) || !ggml_is_contiguous(gate_up) || !ggml_is_contiguous(down) ||
            !ggml_is_matrix(ids) || ids->nb[0] != sizeof(int32_t) || (ids->nb[1] % sizeof(int32_t)) != 0 ||
            cur->ne[1] != 1 || cur->ne[3] != 1 ||
            ids->ne[0] != 8 || ids->ne[1] < 1 || ids->ne[1] > GGML_METAL_INKLING_MOE_MAX_TOKENS ||
            cur->ne[2] != ids->ne[1] ||
            gate_up->ne[0] != cur->ne[0] || (gate_up->ne[0] % 256) != 0 ||
            (gate_up->ne[1] % 8) != 0 || gate_up->ne[1] <= 0 ||
            gate_up->ne[2] < 8 || gate_up->ne[2] > 128 || gate_up->ne[3] != 1 ||
            down->ne[0] != gate_up->ne[1] / 2 || (down->ne[0] % 64) != 0 ||
            down->ne[1] != cur->ne[0] || (down->ne[1] % 4) != 0 ||
            down->ne[2] != gate_up->ne[2] || down->ne[3] != 1) {
        return nullptr;
    }
    const bool gu_ok = gate_up->type == GGML_TYPE_Q4_0 ||
                       gate_up->type == GGML_TYPE_Q4_K ||
                       gate_up->type == GGML_TYPE_IQ2_XXS;
    const bool dn_ok = down->type == GGML_TYPE_Q4_0 ||
                       down->type == GGML_TYPE_Q5_0 ||
                       down->type == GGML_TYPE_Q2_0 ||
                       down->type == GGML_TYPE_IQ4_NL;
    if (!gu_ok || !dn_ok) { return nullptr; }
    ggml_tensor * args[] = { cur, ids, gate_up, down };
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], 8, ids->ne[1], 1, args, 4,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling fused MoE FFN requires Metal"); }, 1, nullptr);
    ggml_set_op_params_f32(node, GGML_MAX_OP_PARAMS / sizeof(int32_t) - 3, beta);
    ggml_set_op_params_f32(node, GGML_MAX_OP_PARAMS / sizeof(int32_t) - 2, linear_beta);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_MOE_FFN;
    ggml_set_name(node, "inkling_moe_ffn");
    return node;
}

static ggml_tensor * ggml_backend_metal_inkling_conv_fused_v2(ggml_context * ctx,
        ggml_tensor * cur, ggml_tensor * weights, ggml_tensor * history, ggml_tensor * state,
        ggml_tensor * dynamic, ggml_tensor * residual, ggml_tensor * exp_weights,
        int32_t ch_offset, int32_t n_seq_tokens, int32_t n_seqs,
        int32_t kw, int32_t head, int32_t n_cells, float scale) {
    const int64_t n_tokens = (int64_t)n_seq_tokens * (int64_t)n_seqs;
    const bool cur_shape_ok = cur && (exp_weights
            ? (cur->ne[0] > 0 && cur->ne[1] > 0 && cur->ne[2] == n_tokens && cur->ne[3] == 1)
            : (cur->ne[0] > 0 && cur->ne[1] == n_tokens && cur->ne[2] == 1 && cur->ne[3] == 1));
    if (!cur || !weights || !history || !state || !std::isfinite(scale) ||
            ch_offset < 0 || n_seq_tokens <= 0 || n_seq_tokens > 32 || n_seqs <= 0 ||
            kw <= 0 || kw > n_seq_tokens || head < 0 || n_cells <= 0 ||
            cur->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 ||
            history->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(cur) || !ggml_is_contiguous(weights) ||
            !ggml_is_contiguous(history) || !ggml_is_contiguous(state) ||
            !cur_shape_ok ||
            weights->ne[0] != 4 || ggml_nrows(weights) != cur->ne[0] ||
            history->ne[0] != 3 || history->ne[2] != (int64_t)n_seqs || history->ne[3] != 1 ||
            (int64_t)ch_offset + cur->ne[0] > history->ne[1] ||
            (state->nb[1] % sizeof(float)) != 0 ||
            ((int64_t)ch_offset + cur->ne[0]) * 3 > (int64_t)(state->nb[1] / sizeof(float)) ||
            (int64_t)head + (int64_t)n_seqs > (int64_t)n_cells ||
            (int64_t)(kw - 1) * (int64_t)n_cells + (int64_t)head + (int64_t)n_seqs > state->ne[1]) {
        return nullptr;
    }
    if (dynamic && (dynamic->type != GGML_TYPE_F32 || !ggml_is_contiguous(dynamic) ||
            dynamic->ne[1] != n_tokens || dynamic->ne[2] != 1 || dynamic->ne[3] != 1 ||
            dynamic->ne[0] <= 0 || (dynamic->ne[0] % 4) != 0 || (cur->ne[0] % (dynamic->ne[0] / 4)) != 0)) {
        return nullptr;
    }
    if (residual && (residual->type != GGML_TYPE_F32 || !ggml_is_contiguous(residual) ||
            residual->ne[0] != cur->ne[0] || residual->ne[1] != n_tokens ||
            residual->ne[2] != 1 || residual->ne[3] != 1)) {
        return nullptr;
    }
    if (exp_weights && (exp_weights->type != GGML_TYPE_F32 || !ggml_is_contiguous(exp_weights) ||
            ggml_nelements(exp_weights) != cur->ne[1] * n_tokens)) {
        return nullptr;
    }
    ggml_tensor * args[7] = { cur, weights, history, state, nullptr, nullptr, nullptr };
    int n_args = 4;
    uint32_t flags = 0;
    if (dynamic)     { args[n_args++] = dynamic;     flags |= 1u; }
    if (residual)    { args[n_args++] = residual;    flags |= 2u; }
    if (exp_weights) { args[n_args++] = exp_weights; flags |= 4u; }
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], n_tokens, 1, 1, args, n_args,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling fused conv requires Metal"); }, 1, nullptr);
    node->op_params[0] = ch_offset;
    node->op_params[1] = n_seq_tokens;
    node->op_params[2] = n_seqs;
    node->op_params[3] = kw;
    node->op_params[4] = head;
    node->op_params[5] = n_cells;
    node->op_params[6] = (int32_t)flags;
    ggml_set_op_params_f32(node, GGML_MAX_OP_PARAMS / sizeof(int32_t) - 2, scale);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_CONV_FUSED;
    return node;
}

static ggml_tensor * ggml_backend_metal_inkling_conv_fused_v1(ggml_context * ctx,
        ggml_tensor * cur, ggml_tensor * weights, ggml_tensor * history, ggml_tensor * state,
        ggml_tensor * dynamic, ggml_tensor * residual,
        int32_t ch_offset, int32_t n_seq_tokens, int32_t n_seqs,
        int32_t kw, int32_t head, int32_t n_cells, float scale) {
    return ggml_backend_metal_inkling_conv_fused_v2(ctx, cur, weights, history, state,
            dynamic, residual, nullptr, ch_offset, n_seq_tokens, n_seqs, kw, head, n_cells, scale);
}

static ggml_tensor * ggml_backend_metal_inkling_conv_residual_v1(ggml_context * ctx, ggml_tensor * cur,
        ggml_tensor * weights, ggml_tensor * window, ggml_tensor * dynamic, float scale) {
    if (cur->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || window->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(cur) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(window) ||
            window->ne[2] != 1 || window->ne[3] != 1 || ggml_nrows(cur) != cur->ne[1] ||
            weights->ne[0] != 4 || ggml_nrows(weights) != cur->ne[0] ||
            window->ne[1] != cur->ne[0] || window->ne[0] != cur->ne[1] + 3) {
        return nullptr;
    }
    if (dynamic && (dynamic->type != GGML_TYPE_F32 || !ggml_is_contiguous(dynamic) ||
            ggml_nrows(dynamic) != dynamic->ne[1] || dynamic->ne[1] != cur->ne[1] ||
            dynamic->ne[0] <= 0 || dynamic->ne[0] % 4 != 0 || cur->ne[0] % (dynamic->ne[0] / 4) != 0)) {
        return nullptr;
    }
    ggml_tensor * args[] = {cur, weights, window, dynamic};
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], cur->ne[1], 1, 1,
            args, dynamic ? 4 : 3,
            +[](ggml_tensor *, int, int, void *) { GGML_ABORT("Inkling convolution residual requires Metal"); }, 1, nullptr);
    ggml_set_op_params_f32(node, GGML_MAX_OP_PARAMS / sizeof(int32_t) - 2, scale);
    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_METAL_INKLING_CONV_RESIDUAL;
    return node;
}

// Private v1 contract: a host-visible I64 table holds pointers to leased,
// separately allocated expert tensors. The table and every lease must outlive
// graph completion. Only the backend interprets the address-routing marker.
static bool ggml_backend_metal_expert_addresses_v1(ggml_tensor * op, ggml_tensor * table) {
    if (!op || op->op != GGML_OP_MUL_MAT_ID || !table || table->type != GGML_TYPE_I64 ||
        table->ne[0] < 1 || table->ne[0] > GGML_METAL_EXPERT_ADDRESS_LIMIT || table->ne[1] != 1 || table->ne[2] != 1 ||
        !table->data || !ggml_backend_metal_buffer_get_host_base(table->buffer) || op->src[1]->ne[2] != 1) { return false; }
    const auto * refs = static_cast<const uint64_t *>(table->data);
    for (int64_t i = 0; i < table->ne[0]; ++i) {
        const auto * tensor = reinterpret_cast<const ggml_tensor *>(uintptr_t(refs[i]));
        if (!tensor || !tensor->buffer || !ggml_backend_buffer_is_metal(tensor->buffer) || tensor->type != op->src[0]->type || tensor->ne[2] != 1 ||
            tensor->ne[0] != op->src[0]->ne[0] || tensor->ne[1] != op->src[0]->ne[1]) { return false; }
    }
    op->src[3] = table;
    op->op_params[0] = 0x41544144;
    return true;
}

// Mark a dedicated contiguous route copy. Scheduler copies preserve op_params.
static bool ggml_backend_metal_route_attach(ggml_tensor * node, void * channel,
        void (*arm)(void *, uint64_t, uint64_t, size_t), void * user, uint64_t generation) {
    if (!node || node->op != GGML_OP_CONT || node->type != GGML_TYPE_I32 || !ggml_is_contiguous(node) ||
        ggml_nelements(node) < 1 || ggml_nelements(node) > GGML_METAL_ROUTE_MAX_IDS || !channel || !arm) return false;
    struct params { int32_t marker, reserved; void * channel; decltype(arm) callback; void * user; uint64_t generation; };
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "route marker must fit op_params");
    const params p{GGML_METAL_ROUTE_MARKER, 0, channel, arm, user, generation};
    memcpy(node->op_params, &p, sizeof(p));
    return true;
}
static void * ggml_backend_metal_route_create(ggml_backend_t backend) {
    return ggml_backend_is_metal(backend) ? ggml_metal_route_create(ggml_backend_get_device(backend)->context) : nullptr;
}
static const ggml_metal_route_api route_api = {
    65536, ggml_backend_metal_route_create, ggml_metal_route_destroy,
    ggml_backend_metal_route_attach, ggml_metal_route_read,
};

static void * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_metal_window_append") == 0) {
        return (void *) +[](ggml_backend_t backend, ggml_cgraph * graph) {
            return ggml_metal_window_append((ggml_metal_t) backend->context, graph);
        };
    }
    if (strcmp(name, "ggml_backend_metal_window_flush") == 0) {
        return (void *) +[](ggml_backend_t backend, uint64_t * submissions) {
            return ggml_metal_window_flush((ggml_metal_t) backend->context, submissions);
        };
    }
    if (strcmp(name, "ggml_backend_metal_route_publication_v1") == 0) {
        return (void *) &route_api;
    }
    if (strcmp(name, "ggml_backend_metal_expert_addresses_v1") == 0) {
        return (void *) ggml_backend_metal_expert_addresses_v1;
    }
    if (strcmp(name, "ggml_backend_metal_host_expert_bridge_v1") == 0) {
        return (void *) ggml_backend_metal_host_expert_bridge_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_prefill_attention_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_prefill_attention_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_flash_attn_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_flash_attn_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_flash_attn_v2") == 0) {
        return (void *) ggml_backend_metal_inkling_flash_attn_v2;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_flash_attn_ready_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_flash_attn_ready_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_rel_mask_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_rel_mask_v1;
    }
    if (strcmp(name, "ggml_backend_metal_mark_inkling_vocab_add") == 0) {
        return (void *) ggml_backend_metal_mark_inkling_vocab_add;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_clip_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_clip_v1;
    }
    if (strcmp(name, "ggml_backend_metal_mark_inkling_group_bank") == 0) {
        return (void *) ggml_backend_metal_mark_inkling_group_bank;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_control_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_control_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_mix_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_mix_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_mix_norm_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_mix_norm_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_conv_fused_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_conv_fused_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_conv_fused_v2") == 0) {
        return (void *) ggml_backend_metal_inkling_conv_fused_v2;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_moe_ffn_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_moe_ffn_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_ctrl_down_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_ctrl_down_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_ctrl_apply_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_ctrl_apply_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_dspark_argmax_ready_v2") == 0) {
        return (void *) ggml_backend_metal_inkling_dspark_argmax_ready_v2;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_dspark_argmax_v2") == 0) {
        return (void *) ggml_backend_metal_inkling_dspark_argmax_v2;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_mul_sigmoid_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_mul_sigmoid_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_softcap_trim_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_softcap_trim_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_situ_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_situ_v1;
    }
    if (strcmp(name, "ggml_backend_metal_inkling_conv_residual_v1") == 0) {
        return (void *) ggml_backend_metal_inkling_conv_residual_v1;
    }
    if (strcmp(name, "ggml_backend_metal_host_expert_window_active_v1") == 0) {
        return (void *) +[](ggml_backend_t backend) {
            return ggml_metal_host_expert_window_active((ggml_metal_t) backend->context);
        };
    }
    if (strcmp(name, "ggml_backend_metal_host_expert_window_append_v1") == 0) {
        return (void *) +[](ggml_backend_t backend, ggml_cgraph * graph, void (*release)(void *), void * user) {
            return ggml_metal_host_expert_window_append((ggml_metal_t) backend->context, graph, release, user);
        };
    }
    if (strcmp(name, "ggml_backend_metal_compute_graph_prefix") == 0) {
        return (void *) ggml_backend_metal_compute_graph_prefix;
    }
    if (strcmp(name, "ggml_backend_metal_wait_graph_prefix") == 0) {
        return (void *) ggml_backend_metal_wait_graph_prefix;
    }
    if (strcmp(name, "ggml_backend_metal_prepare_single_expert") == 0) {
        return (void *) ggml_backend_metal_prepare_single_expert;
    }
    if (strcmp(name, "ggml_backend_metal_buffer_get_host_base") == 0) {
        return (void *) ggml_backend_metal_buffer_get_host_base;
    }
    if (strcmp(name, "ggml_backend_metal_last_graph_timing") == 0) {
        return (void *) ggml_backend_metal_last_graph_timing;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_metal_get_features;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        return (void *)ggml_backend_metal_tuning_set_fa_vec_override;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        return (void *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        return (void *)ggml_backend_metal_tuning_fa_vec_ne11_bucket;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        return (void *)ggml_backend_metal_tuning_fa_vec_ne01_bucket;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        return (void *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        return (void *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    if (strcmp(name, "ggml_backend_fusion_get") == 0) {
        return (void *)ggml_backend_metal_fusion_get;
    }
    if (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        return (void *)ggml_backend_metal_fusion_stats_init;
    }
    if (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        return (void *)ggml_backend_metal_fusion_stats_reset;
    }
    if (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        return (void *)ggml_backend_metal_fusion_stats_get;
    }
    if (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        return (void *)ggml_backend_metal_fusion_set_enabled;
    }
    if (strcmp(name, "ggml_backend_metal_create_expert_buffer_type") == 0) {
        return (void *) ggml_backend_metal_create_expert_buffer_type;
    }
    if (strcmp(name, "ggml_backend_metal_free_expert_buffer_type") == 0) {
        return (void *) ggml_backend_metal_free_expert_buffer_type;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_count,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

static ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, int device) {
    return new ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

static void ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

struct ggml_backend_device_deleter {
    void operator()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        const char * env = getenv("GGML_METAL_DEVICES");
        if (env) {
            g_devices = atoi(env);
        }

        static std::vector<ggml_backend_device_ptr> devs;

        if (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", true);

            static ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            for (int i = 0; i < g_devices; ++i) {
                auto * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
