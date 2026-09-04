#include "arch/sanotts/sano_graph.h"

#include "arch/sanotts/sano_weights.h"
#include "backend/backend.h"
#include "model/model.h"

#include <new>

#include <ggml.h>

namespace kokopop {

ggml_context * sano_graph_context(ScratchArena & arena, size_t bytes,
                                  const char * label, std::string & error) {
    uint8_t * mem = nullptr;
    try {
        mem = arena.data(bytes);
    } catch (const std::bad_alloc &) {
        error = std::string("failed to allocate the sanoTTS ") + label + " arena";
        return nullptr;
    }

    ggml_init_params params{};
    params.mem_size   = bytes;
    params.mem_buffer = mem;
    params.no_alloc   = true;

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        error = std::string("failed to create the sanoTTS ") + label + " context";
    }
    return ctx;
}

ggml_tensor * sano_conv1d(ggml_context * ctx, ggml_tensor * weight,
                          ggml_tensor * x, int64_t in_ch, int kernel,
                          int padding, int dilation) {
    const int64_t out_ch = weight->ne[1];

    // im2col only reads the shape and type of its first argument, so a proxy
    // describes the kernel geometry without the weight having to be reshaped —
    // which a quantized tensor could not be anyway.
    ggml_tensor * proxy = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kernel, in_ch, out_ch);

    ggml_tensor * im2col = ggml_im2col(ctx, proxy, x, 1, 0, padding, 0,
                                       dilation, 0, false, GGML_TYPE_F32);
    const int64_t out_len = im2col->ne[1];

    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, im2col, im2col->ne[0], out_len));
    ggml_tensor * out  = ggml_mul_mat(ctx, weight, flat);   // [out_ch, out_len]
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);

    return ggml_cont(ctx, ggml_transpose(ctx, out));        // [out_len, out_ch]
}

ggml_tensor * sano_conv1d_dw(ggml_context * ctx, ggml_tensor * weight,
                             ggml_tensor * x, int kernel, int padding) {
    const int64_t channels = weight->ne[1];

    ggml_tensor * kernel3d = ggml_reshape_3d(ctx, weight, kernel, 1, channels);
    ggml_tensor * data     = ggml_reshape_4d(ctx, x, x->ne[0], 1, channels, 1);

    ggml_tensor * im2col = ggml_im2col(ctx, kernel3d, data, 1, 0, padding, 0,
                                       1, 0, false, GGML_TYPE_F32);
    ggml_tensor * out = ggml_mul_mat(ctx, im2col, kernel3d);  // [out_len, 1, C]
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);

    return ggml_cont(ctx, ggml_reshape_2d(ctx, out, out->ne[0], channels));
}

ggml_tensor * sano_conv_transpose1d(ggml_context * ctx, ggml_tensor * weight,
                                    ggml_tensor * x, int stride, int padding) {
    ggml_tensor * full = ggml_conv_transpose_1d(ctx, weight, x, stride, 0, 1);
    if (padding == 0) {
        return full;
    }

    // PyTorch's `padding` crops the transposed output symmetrically. The view
    // stays a view: the next op is an add, which reads strided sources fine.
    const int64_t out_len = full->ne[0] - 2 * static_cast<int64_t>(padding);
    return ggml_view_2d(ctx, full, out_len, full->ne[1], full->nb[1],
                        static_cast<size_t>(padding) * full->nb[0]);
}

ggml_tensor * sano_add_channel_bias(ggml_context * ctx, ggml_tensor * x,
                                    ggml_tensor * bias) {
    // x is [T, C] and bias is [C]: it has to broadcast along ne[0], which is
    // what transposing it to [1, C] expresses.
    return ggml_add(ctx, x, ggml_transpose(ctx, bias));
}

ggml_tensor * sano_leaky_relu(ggml_context * ctx, const Backend * backend,
                              ggml_tensor * x, float slope) {
    if (backend == nullptr || backend->has_leaky_relu()) {
        return ggml_leaky_relu(ctx, x, slope, false);
    }
    ggml_tensor * pos = ggml_relu(ctx, x);
    ggml_tensor * neg = ggml_scale(ctx, ggml_relu(ctx, ggml_neg(ctx, x)), slope);
    return ggml_sub(ctx, pos, neg);
}

ggml_tensor * sano_layer_norm(ggml_context * ctx, ggml_tensor * x,
                              ggml_tensor * weight, ggml_tensor * bias) {
    ggml_tensor * normed = ggml_norm(ctx, x, SANO_LAYER_NORM_EPS);
    return ggml_add(ctx, ggml_mul(ctx, normed, weight), bias);
}

ggml_tensor * sano_res_block(ggml_context * ctx, ggml_tensor * x,
                             const SanoResBlock & block, int64_t channels) {
    const int padding = block.kernel / 2;

    ggml_tensor * t = sano_conv1d(ctx, block.net0_w, x, channels, block.kernel, padding, 1);
    t = sano_add_channel_bias(ctx, t, block.net0_b);
    t = ggml_silu(ctx, t);

    ggml_tensor * u = sano_conv1d(ctx, block.net2_w, t, channels, block.kernel, padding, 1);
    u = sano_add_channel_bias(ctx, u, block.net2_b);

    return ggml_add(ctx, x, ggml_scale(ctx, u, block.scale));
}

} // namespace kokopop
