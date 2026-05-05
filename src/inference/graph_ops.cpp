#include "kokoro.h"

#include "core/constants.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <new>

namespace kokopop {

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

ggml_tensor * conv1d(
    ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input,
    int stride, int padding, int dilation) {
    if (weight->type != GGML_TYPE_F16) {
        weight = ggml_cast(ctx, weight, GGML_TYPE_F16);
    }
    return ggml_conv_1d(ctx, weight, input, stride, padding, dilation);
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
    ggml_tensor * beta = linear(ctx, bw, bb, style);
    ggml_tensor * one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    model.backend->queue_f32_tensor(one, 1.0f);
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), ggml_add(ctx, gamma, one)), beta);
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
// 8.3 — Single-pass custom kernel (CPU).
//       Replaces three separate ggml ops (sin + div + sqr + add + mul)
//       with one element-wise kernel.  Falls back to the standard graph
//       when the custom op is not available (e.g. Metal backend).
// ---------------------------------------------------------------------------

namespace {

static void snake1d_kernel(
    struct ggml_tensor * dst,
    const struct ggml_tensor * x,
    const struct ggml_tensor * alpha,
    int ith, int nth, void *) {
    const int64_t ne0 = x->ne[0];
    const int64_t ne1 = x->ne[1];
    const int64_t total = ne0 * ne1;
    const size_t  row_stride_x     = x->nb[1]     / sizeof(float);
    const size_t  row_stride_alpha = alpha->nb[1] / sizeof(float);

    const float * __restrict x_data    = static_cast<const float *>(x->data);
    const float * __restrict alpha_data = static_cast<const float *>(alpha->data);
    float * __restrict dst_data    = static_cast<float *>(dst->data);

    // alpha may be [1, T] — one value per column.
    const bool alpha_per_col = (alpha->ne[0] == 1);

    // Manual work partitioning (no ggml_compute_task_init in stock ggml).
    const int64_t chunk = (total + nth - 1) / nth;
    const int64_t start = chunk * ith;
    const int64_t end   = std::min(start + chunk, total);
    for (int64_t idx = start; idx < end; ++idx) {
        const int64_t i0 = idx % ne0;
        const int64_t i1 = idx / ne0;

        const float xi = x_data[i0 + i1 * row_stride_x];
        const float alpha_val = alpha_per_col
            ? alpha_data[i1 * row_stride_alpha]   // alpha[0, i1]  (broadcast column)
            : alpha_data[i0 + i1 * row_stride_alpha];

        float result;
        if (std::abs(alpha_val) > 1e-6f) {
            const float s = std::sin(xi * alpha_val);
            result = xi + (s * s) / alpha_val;
        } else {
            // α ≈ 0:  sin²(x·α)/α → 0 by L'Hôpital (derivative of sin² at 0 is 0)
            result = xi;
        }
        dst_data[idx] = result;
    }
}

} // anonymous namespace

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

    // 8.3 — Custom single-pass kernel.
    //       ggml_repeat creates the broadcasted alpha matching x's shape,
    //       then the kernel computes x + sin²(x·α)/α in one element-wise pass.
    ggml_tensor * a = ggml_repeat(ctx, alpha_2d, x);
    return ggml_map_custom2(ctx, x, a, snake1d_kernel, GGML_N_TASKS_MAX, nullptr);
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
    cur = add_channel_bias(ctx, conv1d(ctx, conv1, cur, 1, 1, 1), conv1_b);

    cur = cached != nullptr
        ? adain_1d(ctx, cur, style, cached->norm2)
        : adain_1d(ctx, model, cur, style, prefix + ".norm2", error);
    if (cur == nullptr) {
        return nullptr;
    }
    cur = ggml_leaky_relu(ctx, cur, 0.2f, false);
    cur = add_channel_bias(ctx, conv1d(ctx, conv2, cur, 1, 1, 1), conv2_b);

    ggml_tensor * residual = maybe_upsample_nearest(ctx, x, upsample);
    ggml_tensor * conv1x1 = cached != nullptr
        ? cached->conv1x1_w
        : model.cached_tensor(prefix + ".conv1x1.weight");
    if (conv1x1 != nullptr) {
        residual = conv1d(ctx, conv1x1, residual, 1, 0, 1);
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
            conv1d(ctx, weights.convs1_w[idx], cur, 1, weights.paddings[idx], KOKOPOP_RESBLOCK_DILATIONS[i]),
            weights.convs1_b[idx]);
        cur = adain_1d(ctx, cur, style, weights.adain2[idx]);
        cur = graph_snake1d(ctx, cur, weights.alpha2[idx], prefix + ".alpha2." + std::to_string(i), error);
        if (cur == nullptr) {
            return nullptr;
        }
        cur = add_channel_bias(ctx,
            conv1d(ctx, weights.convs2_w[idx], cur, 1, kernel_size / 2, 1),
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
                KOKOPOP_RESBLOCK_DILATIONS[i]),
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
                1),
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
//
// 8.2 — Gate tensor splitting.
//        lstm_gate_tensor / lstm_gate_bias split 4-gate-packed weight
//        matrices into per-gate views.  Split views are cached in
//        model.tensor_cache (keyed by "name.gate_index") so that
//        repeated calls (forward + backward directions) reuse the
//        same views without re-splitting.
//        In lstm_direction, ALL gate lookups happen BEFORE the time-step
//        loop — the n_steps × 4 gate computations never hit the hash map.
//
// 8.1 — Output pre-allocation.
//        lstm_direction pre-allocates the full output tensor
//        [hidden × n_steps] and writes each step's hidden state to the
//        correct column via ggml_set_2d.  This replaces the O(n²)
//        ggml_concat chain (each concat copies ALL previous columns)
//        with O(n) direct column writes (each write touches only
//        `hidden` floats).
// ---------------------------------------------------------------------------

ggml_tensor * col_view(ggml_context * ctx, ggml_tensor * x, int64_t index) {
    return ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], index * x->nb[1]);
}

ggml_tensor * lstm_gate_tensor(
    ggml_context * ctx,
    Model & model,
    const std::string & name,
    int gate,
    std::string & error) {
    ggml_tensor * split = model.cached_tensor(name + "." + std::to_string(gate));
    if (split != nullptr) {
        return split;
    }
    ggml_tensor * raw = require_tensor(model, name.c_str(), error);
    if (raw == nullptr) {
        return nullptr;
    }
    if (raw->ne[1] % 4 != 0) {
        error = "invalid unsplit LSTM tensor shape: " + name;
        return nullptr;
    }
    const int64_t gate_rows = raw->ne[1] / 4;
    return ggml_view_2d(ctx, raw, raw->ne[0], gate_rows, raw->nb[1], gate * gate_rows * raw->nb[1]);
}

ggml_tensor * lstm_gate_bias(
    ggml_context * ctx,
    Model & model,
    const std::string & name,
    int gate,
    std::string & error) {
    ggml_tensor * split = model.cached_tensor(name + "." + std::to_string(gate));
    if (split != nullptr) {
        return split;
    }
    ggml_tensor * raw = require_tensor(model, name.c_str(), error);
    if (raw == nullptr) {
        return nullptr;
    }
    if (raw->ne[0] % 4 != 0) {
        error = "invalid unsplit LSTM bias shape: " + name;
        return nullptr;
    }
    const int64_t gate_rows = raw->ne[0] / 4;
    return ggml_view_1d(ctx, raw, gate_rows, gate * gate_rows * raw->nb[0]);
}

ggml_tensor * lstm_direction(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * input,
    const std::string & prefix,
    bool reverse,
    int64_t n_steps,
    std::string & error) {
    const char * suffix = reverse ? "_reverse" : "";

    // 8.2 — Gate lookups outside the loop.  Split views are cached
    //       so forward/backward directions reuse the same tensors.
    ggml_tensor * w_ih[4]{};
    ggml_tensor * w_hh[4]{};
    ggml_tensor * b_ih[4]{};
    ggml_tensor * b_hh[4]{};
    for (int g = 0; g < 4; ++g) {
        w_ih[g] = lstm_gate_tensor(ctx, model, prefix + ".weight_ih_l0" + suffix, g, error);
        w_hh[g] = lstm_gate_tensor(ctx, model, prefix + ".weight_hh_l0" + suffix, g, error);
        b_ih[g] = lstm_gate_bias(ctx, model, prefix + ".bias_ih_l0" + suffix, g, error);
        b_hh[g] = lstm_gate_bias(ctx, model, prefix + ".bias_ih_l0" + suffix, g, error);
    }
    if (!error.empty()) {
        return nullptr;
    }

    const int64_t hidden = w_hh[0]->ne[1];

    // 8.1 — Pre-allocate the full output tensor [hidden × n_steps]
    //       and write each hidden state to its column via ggml_set_2d.
    //       This avoids the O(n²) memory copies from ggml_concat.
    ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_steps);
    model.backend->queue_zero_tensor(output);

    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    model.backend->queue_zero_tensor(h);
    model.backend->queue_zero_tensor(c);
    for (int64_t step = 0; step < n_steps; ++step) {
        const int64_t t = reverse ? n_steps - 1 - step : step;
        ggml_tensor * x_t = col_view(ctx, input, t);

        // Compute LSTM gates — h and c carry over between iterations.
        ggml_tensor * gates[4]{};
        for (int g = 0; g < 4; ++g) {
            gates[g] = ggml_add(ctx,
                ggml_add(ctx, ggml_mul_mat(ctx, w_ih[g], x_t), b_ih[g]),
                ggml_add(ctx, ggml_mul_mat(ctx, w_hh[g], h), b_hh[g]));
        }
        ggml_tensor * i_gate = ggml_sigmoid(ctx, gates[0]);
        ggml_tensor * f_gate = ggml_sigmoid(ctx, gates[1]);
        ggml_tensor * g_gate = ggml_tanh(ctx, gates[2]);
        ggml_tensor * o_gate = ggml_sigmoid(ctx, gates[3]);
        c = ggml_add(ctx, ggml_mul(ctx, f_gate, c), ggml_mul(ctx, i_gate, g_gate));
        h = ggml_mul(ctx, o_gate, ggml_tanh(ctx, c));

        // 8.1 — Write h to column t of the pre-allocated output.
        //       ggml_set_2d(dst, src, nb1, offset_bytes):
        //         nb1    = row stride of the destination (bytes)
        //         offset = byte offset for column t
        output = ggml_set_2d(
            ctx, output, h,
            output->nb[1],
            static_cast<size_t>(t) * output->nb[1]);
    }
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
            add_channel_bias(ctx, conv1d(ctx, conv_w, ggml_cont(ctx, ggml_transpose(ctx, cur)), 1, 2, 1), conv_b)));
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
