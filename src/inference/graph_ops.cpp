#include "kokoro.h"
#include "lstm_fused.h"

#ifdef KOKOPOP_HAS_METAL
#include "backend/metal_vocoder.h"
#endif

#include "core/constants.h"

#include <cstdio>
#include <cstring>

namespace kokopop {

namespace {

#ifdef KOKOPOP_HAS_METAL
static float convt_weight_at(const ggml_tensor * weight, int64_t k, int64_t oc, int64_t ic) {
    const size_t index = static_cast<size_t>(k + weight->ne[0] * oc + weight->ne[0] * weight->ne[1] * ic);
    if (weight->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(weight->data)[index]);
    }
    return static_cast<const float *>(weight->data)[index];
}

void convt_crop_bias_cpu_fallback(
    ggml_tensor * dst,
    const ggml_tensor * input,
    const ggml_tensor * weight,
    const ggml_tensor * bias,
    int stride,
    int crop_left) {
    const float * x = static_cast<const float *>(input->data);
    const float * b = static_cast<const float *>(bias->data);
    float * y = static_cast<float *>(dst->data);
    const int64_t il = input->ne[0];
    const int64_t ic_count = input->ne[1];
    const int64_t k_count = weight->ne[0];
    const int64_t oc_count = weight->ne[1];
    const int64_t ol = dst->ne[0];
    for (int64_t oc = 0; oc < oc_count; ++oc) {
        for (int64_t t = 0; t < ol; ++t) {
            const int64_t full_t = t + crop_left;
            float acc = b[oc];
            for (int64_t ic = 0; ic < ic_count; ++ic) {
                for (int64_t k = 0; k < k_count; ++k) {
                    const int64_t src_num = full_t - k;
                    if (src_num < 0 || (src_num % stride) != 0) {
                        continue;
                    }
                    const int64_t ti = src_num / stride;
                    if (ti >= il) {
                        continue;
                    }
                    acc += convt_weight_at(weight, k, oc, ic) *
                           x[static_cast<size_t>(ti + il * ic)];
                }
            }
            y[static_cast<size_t>(t + ol * oc)] = acc;
        }
    }
}

void metal_vocoder_convt_callback(
    ggml_tensor * dst,
    const ggml_tensor * output_storage,
    const ggml_tensor * input,
    const ggml_tensor * weight,
    int /* ith */,
    int /* nth */,
    void * userdata) {
    (void)output_storage;
    const auto * params = static_cast<const MetalVocoderConvTransposeParams *>(userdata);
    if (params == nullptr || params->kernel == nullptr || params->bias == nullptr ||
        !metal_vocoder_conv_transpose1d_crop_bias(
            static_cast<MetalVocoderState *>(params->kernel),
            input,
            weight,
            params->bias,
            dst,
            params->stride,
            params->crop_left)) {
        std::fprintf(stderr, "[metal_vocoder] conv_transpose1d_crop_bias failed\n");
        convt_crop_bias_cpu_fallback(dst, input, weight, params->bias, params->stride, params->crop_left);
    }
}
#endif

} // namespace

// ---------------------------------------------------------------------------
// Basic graph operations
// ---------------------------------------------------------------------------

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias, float eps) {
    if (x->type != GGML_TYPE_F32) {
        x = ggml_cast(ctx, x, GGML_TYPE_F32);
    }
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), weight), bias);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias, ggml_tensor * x) {
    if (x->type != GGML_TYPE_F32) {
        x = ggml_cast(ctx, x, GGML_TYPE_F32);
    }
    return ggml_add(ctx, ggml_mul_mat(ctx, weight, x), bias);
}

ggml_tensor * add_channel_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    if (bias->ne[0] == x->ne[0]) {
        return ggml_add(ctx, x, bias);
    }
    if (bias->ne[0] == x->ne[1]) {
        return ggml_add(ctx, x, ggml_transpose(ctx, bias));
    }
    return ggml_add(ctx, x, bias);
}

// conv1d dispatch:
//
// - F16 weights → native ggml_conv_1d (reshape 2D→3D first)
// - Quantized weights → im2col + mul_mat (keeps weight quantized)
//
// Do NOT use ggml_cast(ctx, quantized_weight, GGML_TYPE_F16) here.
// Runtime graph dequantization hits unsupported dup/cast paths on Metal
// and can abort in ggml_compute_forward_dup. If you want F16 for small
// convs, export those weights as F16 in the GGUF (see Option A below).
//
// Future options for small-conv optimization:
//   A. Export small convs as F16 in GGUF (simplest, most robust).
//   B. Pre-dequantize at load time into a separate F16 tensor constant.
//   C. Stay quantized everywhere and optimize im2col later.

ggml_tensor * conv1d(
    ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input,
    int stride, int padding, int dilation, int kernel_size) {
    GGML_ASSERT(kernel_size > 0);
    const bool kernel_is_3d = (weight->ne[2] > 1) ||
                              (weight->ne[0] == kernel_size);

    // Path A: 3D (legacy) — route through ggml_conv_1d directly.
    if (kernel_is_3d) {
        return ggml_conv_1d(ctx, weight, input, stride, padding, dilation);
    }

    const int64_t ick = weight->ne[0];
    const int64_t oc  = weight->ne[1];
    GGML_ASSERT(ick % kernel_size == 0);
    const int64_t ic = ick / kernel_size;

    // Safe F16 path: only when the tensor is already F16.
    if (weight->type == GGML_TYPE_F16) {
        ggml_tensor * w3d = ggml_reshape_3d(ctx, weight, kernel_size, ic, oc);
        return ggml_conv_1d(ctx, w3d, input, stride, padding, dilation);
    }

    // Path B: 2D quantized — direct im2col + mul_mat with the quantized
    // weight as first operand of mul_mat (hits the quantized vec_dot kernel).
    ggml_tensor * shape_proxy = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F16, kernel_size, ic, oc);
    ggml_tensor * im2col = ggml_im2col(
        ctx, shape_proxy, input,
        stride, 0, padding, 0, dilation, 0,
        /*is_2D=*/false, GGML_TYPE_F32);

    const int64_t ol = im2col->ne[1];
    const int64_t n  = im2col->ne[2];
    GGML_ASSERT(ol > 0);

    ggml_tensor * im2col_2d = ggml_reshape_2d(ctx, im2col, ick, ol * n);

    ggml_tensor * out = ggml_mul_mat(ctx, weight, im2col_2d);
    out = ggml_reshape_3d(ctx, out, oc, ol, n);
    out = ggml_cont(ctx, ggml_permute(ctx, out, 1, 0, 2, 3));
    return out;
}

ggml_tensor * conv_transpose1d_crop(
    ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * input,
    int stride,
    int crop_left,
    int out_len) {
    ggml_tensor * conv = ggml_conv_transpose_1d(ctx, weight, input, stride, 0, 1);
    if (crop_left == 0 && out_len == conv->ne[0]) {
        return conv;
    }
    return ggml_view_2d(ctx, conv, out_len, conv->ne[1], conv->nb[1], crop_left * conv->nb[0]);
}

ggml_tensor * conv_transpose1d_crop_bias(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * weight,
    ggml_tensor * input,
    ggml_tensor * bias,
    int stride,
    int crop_left,
    int out_len) {
#ifdef KOKOPOP_HAS_METAL
    if (model.backend != nullptr &&
        model.backend->use_metal_vocoder_convt() &&
        input->type == GGML_TYPE_F32 &&
        (weight->type == GGML_TYPE_F32 || weight->type == GGML_TYPE_F16) &&
        bias->type == GGML_TYPE_F32 &&
        input->ne[1] == weight->ne[2] &&
        bias->ne[0] == weight->ne[1] &&
        out_len > 0) {
        model.metal_vocoder_convt_params.push_back({
            model.backend->metal_vocoder_kernel(),
            bias,
            stride,
            crop_left,
        });
        const MetalVocoderConvTransposeParams * params = &model.metal_vocoder_convt_params.back();
        ggml_tensor * out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out_len, weight->ne[1]);
        return ggml_map_custom3_inplace(
            ctx,
            out,
            input,
            weight,
            metal_vocoder_convt_callback,
            1,
            const_cast<MetalVocoderConvTransposeParams *>(params));
    }
#endif

    return add_channel_bias(ctx, conv_transpose1d_crop(ctx, weight, input, stride, crop_left, out_len), bias);
}

// ---------------------------------------------------------------------------
// AdaIN / normalization operations
// ---------------------------------------------------------------------------

ggml_tensor * repeat_style(ggml_context * ctx, ggml_tensor * style, int64_t n_steps) {
    return ggml_repeat(ctx, style, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, style->ne[0], n_steps));
}

ggml_tensor * ada_layer_norm(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    const std::string & prefix,
    std::string & error) {
    ggml_tensor * gw = require_tensor(model, (prefix + ".fc.gamma.weight").c_str(), error);
    ggml_tensor * gb = require_tensor(model, (prefix + ".fc.gamma.bias").c_str(), error);
    ggml_tensor * bw = require_tensor(model, (prefix + ".fc.beta.weight").c_str(), error);
    ggml_tensor * bb = require_tensor(model, (prefix + ".fc.beta.bias").c_str(), error);
    if (!error.empty()) {
        return nullptr;
    }
    ggml_tensor * gamma = linear(ctx, gw, gb, style);
    ggml_tensor * beta  = linear(ctx, bw, bb, style);
    ggml_tensor * normed = ggml_norm(ctx, x, 1e-5f);
    // norm(x)*(1+gamma)+beta == norm(x) + norm(x)*gamma + beta
    return ggml_add(ctx, ggml_add(ctx, normed, ggml_mul(ctx, normed, gamma)), beta);
}

ggml_tensor * adain_1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * style,
    const AdaIn1dWeights & weights) {
    ggml_tensor * gamma = linear(ctx, weights.gamma_w, weights.gamma_b, style);
    ggml_tensor * beta = linear(ctx, weights.beta_w, weights.beta_b, style);
    ggml_tensor * cur = ggml_norm(ctx, x, 1e-5f);

    ggml_tensor * nw_t = ggml_transpose(ctx, weights.norm_w);
    ggml_tensor * nb_t = ggml_transpose(ctx, weights.norm_b);
    cur = ggml_add(ctx, ggml_mul(ctx, cur, nw_t), nb_t);

    ggml_tensor * gamma_t = ggml_transpose(ctx, gamma);
    cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, gamma_t));
    ggml_tensor * beta_t = ggml_transpose(ctx, beta);
    return ggml_add(ctx, cur, beta_t);
}

ggml_tensor * maybe_upsample_nearest(ggml_context * ctx, ggml_tensor * x, bool upsample) {
    if (!upsample) {
        return x;
    }
    ggml_tensor * cur = ggml_cont(ctx, ggml_transpose(ctx, x));
    cur = ggml_interpolate(ctx, cur, cur->ne[0], cur->ne[1] * 2, cur->ne[2], cur->ne[3], GGML_SCALE_MODE_NEAREST);
    return ggml_cont(ctx, ggml_transpose(ctx, cur));
}

// ---------------------------------------------------------------------------
// Snake1D activation:  x + sin²(x · α) / α
//
// Implemented with native GGML ops (ggml_sin, ggml_mul, ggml_div, ggml_add)
// so the graph can run on any backend including Metal.  Trained α values are
// strictly positive, so division by zero does not occur in practice.
// ---------------------------------------------------------------------------

ggml_tensor * graph_snake1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * alpha,
    const std::string & alpha_name,
    std::string & error) {
    ggml_tensor * alpha_2d = nullptr;
    if (alpha->ne[0] == 1 && alpha->ne[1] == x->ne[1]) {
        alpha_2d = ggml_view_2d(ctx, alpha, 1, x->ne[1], alpha->nb[1], 0);
    } else if (alpha->ne[0] == x->ne[1]) {
        alpha_2d = ggml_view_2d(ctx, alpha, 1, x->ne[1], alpha->nb[0], 0);
    } else {
        error = "invalid Snake1D alpha shape: " + alpha_name;
        return nullptr;
    }

    ggml_tensor * a  = ggml_repeat(ctx, alpha_2d, x);   // broadcast α to x's shape
    ggml_tensor * xa = ggml_mul(ctx, x, a);              // x * α
    ggml_tensor * s  = ggml_sin(ctx, xa);                // sin(x * α)
    ggml_tensor * s2 = ggml_mul(ctx, s, s);              // sin²(x * α)
    return ggml_add(ctx, x, ggml_div(ctx, s2, a));       // x + sin²(x·α) / α
}

ggml_tensor * graph_snake1d(
    ggml_context * ctx, Model & model, ggml_tensor * x,
    const std::string & alpha_name, std::string & error) {
    ggml_tensor * alpha = require_tensor(model, alpha_name.c_str(), error);
    if (!error.empty()) {
        return nullptr;
    }
    return graph_snake1d(ctx, x, alpha, alpha_name, error);
}

// ---------------------------------------------------------------------------
// AdaIN ResBlock 1D
// ---------------------------------------------------------------------------

ggml_tensor * adain_1d(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    const std::string & prefix,
    std::string & error) {
    const auto cached = model.adain_1d_weights.find(prefix);
    if (cached != model.adain_1d_weights.end()) {
        return adain_1d(ctx, x, style, cached->second);
    }

    ggml_tensor * nw = require_tensor(model, (prefix + ".norm.weight").c_str(), error);
    ggml_tensor * nb = require_tensor(model, (prefix + ".norm.bias").c_str(), error);
    ggml_tensor * gw = require_tensor(model, (prefix + ".fc.gamma.weight").c_str(), error);
    ggml_tensor * gb = require_tensor(model, (prefix + ".fc.gamma.bias").c_str(), error);
    ggml_tensor * bw = require_tensor(model, (prefix + ".fc.beta.weight").c_str(), error);
    ggml_tensor * bb = require_tensor(model, (prefix + ".fc.beta.bias").c_str(), error);
    if (!error.empty()) {
        return nullptr;
    }

    return adain_1d(ctx, x, style, AdaIn1dWeights{nw, nb, gw, gb, bw, bb});
}

ggml_tensor * adain_resblk1d(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    const std::string & prefix,
    bool upsample,
    std::string & error) {
    const AdainResblk1dWeights * cached = nullptr;
    const auto cached_it = model.adain_resblk1d_weights.find(prefix);
    if (cached_it != model.adain_resblk1d_weights.end()) {
        cached = &cached_it->second;
    }

    ggml_tensor * conv1 = nullptr;
    ggml_tensor * conv1_b = nullptr;
    ggml_tensor * conv2 = nullptr;
    ggml_tensor * conv2_b = nullptr;
    if (cached != nullptr) {
        conv1 = cached->conv1_w;
        conv1_b = cached->conv1_b;
        conv2 = cached->conv2_w;
        conv2_b = cached->conv2_b;
    } else {
        conv1 = require_tensor(model, (prefix + ".conv1.weight").c_str(), error);
        conv1_b = require_tensor(model, (prefix + ".conv1.bias").c_str(), error);
        conv2 = require_tensor(model, (prefix + ".conv2.weight").c_str(), error);
        conv2_b = require_tensor(model, (prefix + ".conv2.bias").c_str(), error);
        if (!error.empty()) {
            return nullptr;
        }
    }

    ggml_tensor * cur = cached != nullptr
        ? adain_1d(ctx, x, style, cached->norm1)
        : adain_1d(ctx, model, x, style, prefix + ".norm1", error);
    if (cur == nullptr) {
        return nullptr;
    }

    cur = ggml_leaky_relu(ctx, cur, 0.2f, false);
    if (upsample) {
        cur = depthwise_pool_upsample(ctx, model, cur, prefix, error);
        if (cur == nullptr) {
            return nullptr;
        }
    }
    cur = add_channel_bias(ctx, conv1d(ctx, conv1, cur, 1, 1, 1, 3), conv1_b);

    cur = cached != nullptr
        ? adain_1d(ctx, cur, style, cached->norm2)
        : adain_1d(ctx, model, cur, style, prefix + ".norm2", error);
    if (cur == nullptr) {
        return nullptr;
    }
    cur = ggml_leaky_relu(ctx, cur, 0.2f, false);
    cur = add_channel_bias(ctx, conv1d(ctx, conv2, cur, 1, 1, 1, 3), conv2_b);

    ggml_tensor * residual = maybe_upsample_nearest(ctx, x, upsample);
    ggml_tensor * conv1x1 = cached != nullptr
        ? cached->conv1x1_w
        : model.cached_tensor(prefix + ".conv1x1.weight");
    if (conv1x1 != nullptr) {
        residual = conv1d(ctx, conv1x1, residual, 1, 0, 1, 1);
    }
    return ggml_scale(ctx, ggml_add(ctx, cur, residual), KOKOPOP_INV_SQRT2);
}

// ---------------------------------------------------------------------------
// Generator ResBlock
// ---------------------------------------------------------------------------

ggml_tensor * graph_generator_resblock(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * style,
    const std::string & prefix,
    int kernel_size,
    const GeneratorResblockWeights & weights,
    std::string & error) {
    for (int i = 0; i < 3; ++i) {
        const size_t idx = static_cast<size_t>(i);
        ggml_tensor * cur = adain_1d(ctx, x, style, weights.adain1[idx]);
        cur = graph_snake1d(ctx, cur, weights.alpha1[idx], prefix + ".alpha1." + std::to_string(i), error);
        if (cur == nullptr) {
            return nullptr;
        }
        cur = add_channel_bias(ctx,
            conv1d(ctx, weights.convs1_w[idx], cur, 1, weights.paddings[idx], KOKOPOP_RESBLOCK_DILATIONS[i], kernel_size),
            weights.convs1_b[idx]);
        cur = adain_1d(ctx, cur, style, weights.adain2[idx]);
        cur = graph_snake1d(ctx, cur, weights.alpha2[idx], prefix + ".alpha2." + std::to_string(i), error);
        if (cur == nullptr) {
            return nullptr;
        }
        cur = add_channel_bias(ctx,
            conv1d(ctx, weights.convs2_w[idx], cur, 1, kernel_size / 2, 1, kernel_size),
            weights.convs2_b[idx]);
        x = ggml_add(ctx, x, cur);
    }
    return x;
}

ggml_tensor * graph_generator_resblock(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    const std::string & prefix,
    int kernel_size,
    std::string & error) {
    const auto cached = model.generator_resblock_weights.find(prefix);
    if (cached != model.generator_resblock_weights.end()) {
        return graph_generator_resblock(ctx, x, style, prefix, kernel_size, cached->second, error);
    }

    int paddings[3];
    for (int i = 0; i < 3; ++i) {
        const int d = KOKOPOP_RESBLOCK_DILATIONS[i];
        paddings[i] = (kernel_size * d - d) / 2;
    }

    for (int i = 0; i < 3; ++i) {
        ggml_tensor * cur = adain_1d(ctx, model, x, style, prefix + ".adain1." + std::to_string(i), error);
        if (cur == nullptr) {
            return nullptr;
        }
        cur = graph_snake1d(ctx, model, cur, prefix + ".alpha1." + std::to_string(i), error);
        if (cur == nullptr) {
            return nullptr;
        }
        cur = add_channel_bias(ctx,
            conv1d(ctx,
                require_tensor(model, (prefix + ".convs1." + std::to_string(i) + ".weight").c_str(), error),
                cur,
                1,
                paddings[i],
                KOKOPOP_RESBLOCK_DILATIONS[i],
                kernel_size),
            require_tensor(model, (prefix + ".convs1." + std::to_string(i) + ".bias").c_str(), error));
        cur = adain_1d(ctx, model, cur, style, prefix + ".adain2." + std::to_string(i), error);
        if (cur == nullptr) {
            return nullptr;
        }
        cur = graph_snake1d(ctx, model, cur, prefix + ".alpha2." + std::to_string(i), error);
        if (cur == nullptr) {
            return nullptr;
        }
        cur = add_channel_bias(ctx,
            conv1d(ctx,
                require_tensor(model, (prefix + ".convs2." + std::to_string(i) + ".weight").c_str(), error),
                cur,
                1,
                kernel_size / 2,
                1,
                kernel_size),
            require_tensor(model, (prefix + ".convs2." + std::to_string(i) + ".bias").c_str(), error));
        if (!error.empty()) {
            return nullptr;
        }
        x = ggml_add(ctx, x, cur);
    }
    return x;
}

// ---------------------------------------------------------------------------
// LSTM
// ---------------------------------------------------------------------------

ggml_tensor * col_view(ggml_context * ctx, ggml_tensor * x, int64_t index) {
    return ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], index * x->nb[1]);
}

struct LstmWeights {
    ggml_tensor * w_ih_packed = nullptr;  // [input_size, 4*hidden]
    ggml_tensor * w_hh_packed = nullptr;  // [hidden, 4*hidden]
    ggml_tensor * b_ih_packed = nullptr;  // [4*hidden]
    ggml_tensor * b_hh_packed = nullptr;  // [4*hidden]

    int64_t hidden = 0;
};

static LstmWeights load_lstm_weights(
    Model & model,
    const std::string & prefix,
    bool reverse,
    std::string & error) {
    LstmWeights w;

    const std::string suffix = reverse ? "_reverse" : "";

    const std::string name_w_ih = prefix + ".weight_ih_l0" + suffix;
    const std::string name_w_hh = prefix + ".weight_hh_l0" + suffix;
    const std::string name_b_ih = prefix + ".bias_ih_l0" + suffix;
    const std::string name_b_hh = prefix + ".bias_hh_l0" + suffix;

    w.w_ih_packed = require_tensor(model, name_w_ih.c_str(), error);
    w.w_hh_packed = require_tensor(model, name_w_hh.c_str(), error);
    w.b_ih_packed = require_tensor(model, name_b_ih.c_str(), error);
    w.b_hh_packed = require_tensor(model, name_b_hh.c_str(), error);

    if (!error.empty()) {
        return w;
    }

    if (w.w_ih_packed == nullptr ||
        w.w_hh_packed == nullptr ||
        w.b_ih_packed == nullptr ||
        w.b_hh_packed == nullptr) {
        error = "missing packed LSTM tensors for: " + prefix;
        return w;
    }

    if (w.w_hh_packed->ne[1] % 4 != 0) {
        error = "invalid packed LSTM weight_hh shape: " + name_w_hh;
        return w;
    }

    if (w.w_ih_packed->ne[1] % 4 != 0) {
        error = "invalid packed LSTM weight_ih shape: " + name_w_ih;
        return w;
    }

    if (w.b_ih_packed->ne[0] % 4 != 0) {
        error = "invalid packed LSTM bias_ih shape: " + name_b_ih;
        return w;
    }

    if (w.b_hh_packed->ne[0] % 4 != 0) {
        error = "invalid packed LSTM bias_hh shape: " + name_b_hh;
        return w;
    }

    const int64_t hidden_from_w_hh = w.w_hh_packed->ne[1] / 4;
    const int64_t hidden_from_w_ih = w.w_ih_packed->ne[1] / 4;
    const int64_t hidden_from_b_ih = w.b_ih_packed->ne[0] / 4;
    const int64_t hidden_from_b_hh = w.b_hh_packed->ne[0] / 4;

    if (hidden_from_w_hh != hidden_from_w_ih ||
        hidden_from_w_hh != hidden_from_b_ih ||
        hidden_from_w_hh != hidden_from_b_hh) {
        error = "inconsistent packed LSTM hidden size for: " + prefix;
        return w;
    }

    w.hidden = hidden_from_w_hh;

    return w;
}

ggml_tensor * lstm_direction(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * input,
    const std::string & prefix,
    bool reverse,
    int64_t n_steps,
    std::string & error) {
    LstmWeights w = load_lstm_weights(model, prefix, reverse, error);
    if (!error.empty() || w.hidden == 0) {
        return nullptr;
    }

    const int64_t hidden = w.hidden;

    // --- Pre-compute W_ih * input + b_ih for the entire sequence ---
    //
    // input:       [input_size, n_steps]
    // w_ih_packed: [input_size, 4 * hidden]
    //
    // Result:
    // pre_input_gates_packed: [4 * hidden, n_steps]
    //
    // This replaces T small matmuls (W_ih * x_t) with a single large one.
    ggml_tensor * pre_input_gates_packed = ggml_add(ctx,
        ggml_mul_mat(ctx, w.w_ih_packed, input),
        w.b_ih_packed);

    // Fused LSTM: replace the per-timestep ggml graph loop (n_steps × ~18 nodes
    // per direction) with a single custom2 node that runs the full recurrence in
    // one C++ callback or Metal compute shader.
    //
    // LstmCustomParams must outlive graph execution.  We store it in
    // model.lstm_custom_params (pre-reserved before graph construction in
    // run_kokoro_generation_probe) so the pointer is stable after push_back.
    const std::string whh_key =
        prefix + ".weight_hh_l0" + (reverse ? "_reverse" : "");
    const auto it = model.lstm_w_hh_f32.find(whh_key);
    if (it == model.lstm_w_hh_f32.end()) {
        error = "fused LSTM: w_hh not preloaded for " + whh_key;
        return nullptr;
    }

    // b_hh_packed is a F32 tensor in the CPU weight buffer; ->data is a valid
    // CPU pointer for the lifetime of the model.
    const float * b_hh_ptr = static_cast<const float *>(w.b_hh_packed->data);

    model.lstm_custom_params.push_back({
        it->second.data(),                     // w_hh_f32
        b_hh_ptr,                              // b_hh
        model.backend->use_metal_lstm(n_steps)
            ? model.backend->metal_lstm_kernel()
            : nullptr,                         // metal_kernel (null on CPU/small LSTM)
        it->first.c_str(),                     // whh_key (key stable in lstm_w_hh_f32)
        hidden,                                // hidden
        n_steps,                               // n_steps
        reverse                                // reverse
    });
    const LstmCustomParams * params = &model.lstm_custom_params.back();

    ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_steps);
    model.backend->queue_zero_tensor(output);

    output = ggml_map_custom2_inplace(
        ctx,
        output,
        pre_input_gates_packed,
        lstm_fused_callback,
        1,
        const_cast<LstmCustomParams *>(params));

    return output;
}

ggml_tensor * bidirectional_lstm(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * input,
    const std::string & prefix,
    int64_t n_steps,
    std::string & error) {
    ggml_tensor * fw = lstm_direction(ctx, model, input, prefix, false, n_steps, error);
    ggml_tensor * bw = lstm_direction(ctx, model, input, prefix, true, n_steps, error);
    if (fw == nullptr || bw == nullptr) {
        return nullptr;
    }
    return ggml_concat(ctx, fw, bw, 0);
}

// ---------------------------------------------------------------------------
// Encoder layers
// ---------------------------------------------------------------------------

ggml_tensor * duration_encoder(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    int64_t n_steps,
    std::string & error) {
    ggml_tensor * style_repeated = repeat_style(ctx, style, n_steps);
    ggml_tensor * cur = ggml_concat(ctx, x, style_repeated, 0);
    for (int block = 0; block < 3; ++block) {
        const int lstm_index = block * 2;
        const int ada_index = lstm_index + 1;
        cur = bidirectional_lstm(
            ctx, model, cur,
            "kokopop.predictor.text_encoder.lstms." + std::to_string(lstm_index),
            n_steps, error);
        if (cur == nullptr) {
            return nullptr;
        }
        cur = ada_layer_norm(
            ctx, model, cur, style,
            "kokopop.predictor.text_encoder.lstms." + std::to_string(ada_index),
            error);
        if (cur == nullptr) {
            return nullptr;
        }
        cur = ggml_concat(ctx, cur, style_repeated, 0);
    }
    return cur;
}

ggml_tensor * text_encoder(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * token_ids,
    ggml_tensor * duration_mask,
    int64_t n_tokens,
    std::string & error) {
    ggml_tensor * emb = require_tensor(model, "kokopop.text_encoder.embedding.weight", error);
    if (!error.empty()) {
        return nullptr;
    }
    ggml_tensor * cur = ggml_get_rows(ctx, emb, token_ids);
    // Layout per CNN block: [ch, time] → T+cont → [time, ch] → conv1d → [time', ch'] → T+cont → [ch', time']
    // The outer transpose+cont is required because ggml_norm normalizes over axis 0
    // (per-token layer norm needs [channel, time]), while conv1d expects [time, channel].
    // On GPU each ggml_cont() materializes a copy → 6 extra copies per forward pass.
    // TODO: pick a stable layout ([ch, time] or [time, ch]) and either:
    //   - transpose the norm weights instead of cur, or
    //   - pre-transpose the conv weights at model load time,
    //   - or write a fused conv1d+layer_norm kernel that avoids layout switches.
    for (int i = 0; i < 3; ++i) {
        const std::string prefix = "kokopop.text_encoder.cnn." + std::to_string(i);
        ggml_tensor * conv_w = require_tensor(model, (prefix + ".0.weight").c_str(), error);
        ggml_tensor * conv_b = require_tensor(model, (prefix + ".0.bias").c_str(), error);
        ggml_tensor * norm_w = require_tensor(model, (prefix + ".1.gamma").c_str(), error);
        ggml_tensor * norm_b = require_tensor(model, (prefix + ".1.beta").c_str(), error);
        if (!error.empty()) {
            return nullptr;
        }
        cur = ggml_cont(ctx, ggml_transpose(ctx,
            add_channel_bias(ctx, conv1d(ctx, conv_w, ggml_cont(ctx, ggml_transpose(ctx, cur)), 1, 2, 1, 5), conv_b)));
        cur = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, cur, 1e-5f), norm_w), norm_b);
        cur = ggml_leaky_relu(ctx, cur, 0.2f, false);
    }
    cur = bidirectional_lstm(ctx, model, cur, "kokopop.text_encoder.lstm", n_tokens, error);
    if (cur == nullptr) {
        return nullptr;
    }
    return ggml_mul_mat(ctx,
        ggml_cont(ctx, ggml_transpose(ctx, cur)),
        ggml_cont(ctx, ggml_transpose(ctx, duration_mask)));
}

} // namespace kokopop
