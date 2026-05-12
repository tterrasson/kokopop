#include "kokoro.h"
#include "lstm_fused.h"

#ifdef KOKOPOP_HAS_METAL
#include "backend/metal_vocoder.h"
#endif

#include "core/constants.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace kokopop {
namespace {

// ---------------------------------------------------------------------------
// Small tensor helpers
// ---------------------------------------------------------------------------

static inline const char * tensor_data_c(const ggml_tensor * t) {
    return static_cast<const char *>(t->data);
}

static inline char * tensor_data(ggml_tensor * t) {
    return static_cast<char *>(t->data);
}

static inline float tensor_get_f32_2d(const ggml_tensor * t, int64_t i0, int64_t i1) {
    return *reinterpret_cast<const float *>(
        tensor_data_c(t)
        + static_cast<size_t>(i0) * t->nb[0]
        + static_cast<size_t>(i1) * t->nb[1]);
}

static inline void tensor_set_f32_2d(ggml_tensor * t, int64_t i0, int64_t i1, float v) {
    *reinterpret_cast<float *>(
        tensor_data(t)
        + static_cast<size_t>(i0) * t->nb[0]
        + static_cast<size_t>(i1) * t->nb[1]) = v;
}

static inline bool tensor_is_f32_2d_contiguous(const ggml_tensor * t) {
    return t != nullptr &&
           t->type == GGML_TYPE_F32 &&
           t->data != nullptr &&
           t->nb[0] == sizeof(float) &&
           t->nb[1] == static_cast<size_t>(t->ne[0]) * sizeof(float);
}

static inline bool backend_needs_contiguous_im2col(const Model & model) {
#ifdef __AVX2__
    if (model.backend_type == KOKOPOP_BACKEND_CPU) {
        return true;
    }
#endif
    return model.backend_type == KOKOPOP_BACKEND_CUDA;
}

static float tensor_get_weight_f32_3d(const ggml_tensor * weight, int64_t k, int64_t oc, int64_t ic) {
    const char * base = tensor_data_c(weight)
        + static_cast<size_t>(k)  * weight->nb[0]
        + static_cast<size_t>(oc) * weight->nb[1]
        + static_cast<size_t>(ic) * weight->nb[2];

    if (weight->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(base));
    }
    return *reinterpret_cast<const float *>(base);
}

void convt_crop_bias_cpu_fallback(
    ggml_tensor * dst,
    const ggml_tensor * input,
    const ggml_tensor * weight,
    const ggml_tensor * bias,
    int stride,
    int crop_left,
    int ith = 0,
    int nth = 1) {

    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(input != nullptr);
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(bias != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);

    const int64_t il       = input->ne[0];
    const int64_t ic_count = input->ne[1];
    const int64_t k_count  = weight->ne[0];
    const int64_t oc_count = weight->ne[1];
    const int64_t ol       = dst->ne[0];
    const int64_t oc_begin = (oc_count * ith) / nth;
    const int64_t oc_end   = (oc_count * (ith + 1)) / nth;

    for (int64_t oc = oc_begin; oc < oc_end; ++oc) {
        const float b = *reinterpret_cast<const float *>(
            tensor_data_c(bias) + static_cast<size_t>(oc) * bias->nb[0]);

        for (int64_t t = 0; t < ol; ++t) {
            tensor_set_f32_2d(dst, t, oc, b);
        }

        for (int64_t ic = 0; ic < ic_count; ++ic) {
            for (int64_t ti = 0; ti < il; ++ti) {
                const float x = tensor_get_f32_2d(input, ti, ic);
                for (int64_t k = 0; k < k_count; ++k) {
                    const int64_t t = ti * stride + k - crop_left;
                    if (t < 0 || t >= ol) {
                        continue;
                    }

                    float cur = tensor_get_f32_2d(dst, t, oc);
                    cur += tensor_get_weight_f32_3d(weight, k, oc, ic) * x;
                    tensor_set_f32_2d(dst, t, oc, cur);
                }
            }
        }
    }
}

#ifdef KOKOPOP_HAS_METAL

void metal_vocoder_convt_callback(
    ggml_tensor * dst,
    const ggml_tensor * output_storage,
    const ggml_tensor * input,
    const ggml_tensor * weight,
    int /*ith*/,
    int /*nth*/,
    void * userdata) {

    (void) output_storage;

    const auto * params = static_cast<const MetalVocoderConvTransposeParams *>(userdata);
    if (params == nullptr ||
        params->kernel == nullptr ||
        params->bias == nullptr ||
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

#endif // KOKOPOP_HAS_METAL

void cpu_vocoder_convt_callback(
    ggml_tensor * dst,
    const ggml_tensor * output_storage,
    const ggml_tensor * input,
    const ggml_tensor * weight,
    int ith,
    int nth,
    void * userdata) {

    (void) output_storage;
    const auto * params = static_cast<const MetalVocoderConvTransposeParams *>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(params->bias != nullptr);
    convt_crop_bias_cpu_fallback(dst, input, weight, params->bias, params->stride, params->crop_left, ith, nth);
}

// ---------------------------------------------------------------------------
// Snake1D fused callback
//
// Computes:
//   y = x + sin^2(x * alpha) / alpha
//
// a = x         [time, channels]
// b = alpha_2d [1, channels]
//
// The callback is stride-safe. It uses a SIMD fast path only when input and
// output are truly contiguous in the expected [time, channel] layout.
// ---------------------------------------------------------------------------

static void snake1d_fused_callback(
    ggml_tensor       * dst,
    const ggml_tensor * a,
    const ggml_tensor * b,
    int ith,
    int nth,
    void * /*userdata*/) {

    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(a != nullptr);
    GGML_ASSERT(b != nullptr);
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[0] == a->ne[0]);
    GGML_ASSERT(dst->ne[1] == a->ne[1]);
    GGML_ASSERT(b->ne[0] == 1);
    GGML_ASSERT(b->ne[1] == a->ne[1]);

    const int64_t n_time     = a->ne[0];
    const int64_t n_channels = a->ne[1];

    const int64_t c_begin = (n_channels * ith)       / nth;
    const int64_t c_end   = (n_channels * (ith + 1)) / nth;

    const bool fast_contiguous = tensor_is_f32_2d_contiguous(a) && tensor_is_f32_2d_contiguous(dst);

    if (fast_contiguous) {
        const float * xd = static_cast<const float *>(a->data);
        float * od       = static_cast<float *>(dst->data);

        for (int64_t c = c_begin; c < c_end; ++c) {
            const float alpha = tensor_get_f32_2d(b, 0, c);
            const float inv_a = 1.0f / alpha;

            const float * xc = xd + c * n_time;
            float * oc       = od + c * n_time;

            int64_t t = 0;

#ifdef __ARM_NEON
            {
                const float32x4_t va   = vdupq_n_f32(alpha);
                const float32x4_t viva = vdupq_n_f32(inv_a);

                for (; t + 3 < n_time; t += 4) {
                    float32x4_t vx = vld1q_f32(xc + t);
                    float xa[4];
                    vst1q_f32(xa, vmulq_f32(vx, va));

                    float s[4] = {
                        std::sinf(xa[0]),
                        std::sinf(xa[1]),
                        std::sinf(xa[2]),
                        std::sinf(xa[3]),
                    };

                    float32x4_t vs  = vld1q_f32(s);
                    float32x4_t vs2 = vmulq_f32(vs, vs);
                    vst1q_f32(oc + t, vaddq_f32(vx, vmulq_f32(vs2, viva)));
                }
            }
#elif defined(__AVX2__)
            {
                const __m256 va   = _mm256_set1_ps(alpha);
                const __m256 viva = _mm256_set1_ps(inv_a);

                for (; t + 7 < n_time; t += 8) {
                    __m256 vx = _mm256_loadu_ps(xc + t);

                    alignas(32) float xa[8];
                    alignas(32) float s[8];
                    _mm256_store_ps(xa, _mm256_mul_ps(vx, va));

                    for (int k = 0; k < 8; ++k) {
                        s[k] = std::sinf(xa[k]);
                    }

                    __m256 vs  = _mm256_load_ps(s);
                    __m256 vs2 = _mm256_mul_ps(vs, vs);
                    _mm256_storeu_ps(oc + t, _mm256_add_ps(vx, _mm256_mul_ps(vs2, viva)));
                }
            }
#endif

            for (; t < n_time; ++t) {
                const float v = xc[t];
                const float s = std::sinf(v * alpha);
                oc[t] = v + s * s * inv_a;
            }
        }

        return;
    }

    for (int64_t c = c_begin; c < c_end; ++c) {
        const float alpha = tensor_get_f32_2d(b, 0, c);
        const float inv_a = 1.0f / alpha;

        for (int64_t t = 0; t < n_time; ++t) {
            const float v = tensor_get_f32_2d(a, t, c);
            const float s = std::sinf(v * alpha);
            tensor_set_f32_2d(dst, t, c, v + s * s * inv_a);
        }
    }
}

static ggml_tensor * snake1d_impl(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * alpha,
    const std::string & alpha_name,
    std::string & error,
    bool allow_fused) {

    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(alpha != nullptr);

    ggml_tensor * alpha_2d = nullptr;
    if (alpha->ne[0] == 1 && alpha->ne[1] == x->ne[1]) {
        alpha_2d = ggml_view_2d(ctx, alpha, 1, x->ne[1], alpha->nb[1], 0);
    } else if (alpha->ne[0] == x->ne[1]) {
        alpha_2d = ggml_view_2d(ctx, alpha, 1, x->ne[1], alpha->nb[0], 0);
    } else {
        error = "invalid Snake1D alpha shape: " + alpha_name;
        return nullptr;
    }

    if (allow_fused && x->type == GGML_TYPE_F32) {
        // The callback itself is stride-safe, but materializing x keeps the
        // fast path active and makes allocator behaviour easier to reason about.
        x = ggml_cont(ctx, x);
        return ggml_map_custom2(ctx, x, alpha_2d, snake1d_fused_callback, GGML_N_TASKS_MAX, nullptr);
    }

    ggml_tensor * a  = ggml_repeat(ctx, alpha_2d, x);
    ggml_tensor * xa = ggml_mul(ctx, x, a);
    ggml_tensor * s  = ggml_sin(ctx, xa);
    ggml_tensor * s2 = ggml_mul(ctx, s, s);
    return ggml_add(ctx, x, ggml_div(ctx, s2, a));
}

// ---------------------------------------------------------------------------
// Quantized Conv1D helper
//
// For quantized 2D conv weights, do NOT reshape the quantized weight to
// [kernel, in_ch, out_ch]. Quantized tensors have block constraints along
// ne[0]. A shape-proxy tensor is used only to tell ggml_im2col the kernel
// geometry; the actual quantized 2D weight remains the MUL_MAT src0.
// ---------------------------------------------------------------------------

static ggml_tensor * conv1d_im2col_mulmat(
    ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * input,
    int stride,
    int padding,
    int dilation,
    int kernel_size,
    bool channel_first,
    bool force_contiguous_im2col) {

#ifdef __AVX2__
    force_contiguous_im2col = true;
#endif

    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(input != nullptr);
    GGML_ASSERT(kernel_size > 0);

    const int64_t ick = weight->ne[0];
    const int64_t oc  = weight->ne[1];

    GGML_ASSERT(ick % kernel_size == 0);
    const int64_t ic = ick / kernel_size;

    ggml_tensor * shape_proxy = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kernel_size, ic, oc);

    ggml_tensor * im2col = ggml_im2col(
        ctx,
        shape_proxy,
        input,
        stride, 0,
        padding, 0,
        dilation, 0,
        false,
        GGML_TYPE_F32);

    const int64_t ol = im2col->ne[1];
    const int64_t n  = im2col->ne[2];

    GGML_ASSERT(ol > 0);
    GGML_ASSERT(im2col->ne[0] == ick);

    ggml_tensor * im2col_2d = ggml_reshape_2d(ctx, im2col, ick, ol * n);

    if (force_contiguous_im2col) {
        im2col_2d = ggml_cont(ctx, im2col_2d);
    }

    ggml_tensor * out = ggml_mul_mat(ctx, weight, im2col_2d);

    if (channel_first) {
        return ggml_reshape_3d(ctx, out, oc, ol, n);
    }

    out = ggml_reshape_3d(ctx, out, oc, ol, n);
    out = ggml_cont(ctx, ggml_permute(ctx, out, 1, 0, 2, 3));
    return out;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Basic graph operations
// ---------------------------------------------------------------------------

ggml_tensor * layer_norm(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * weight,
    ggml_tensor * bias,
    float eps) {

    if (x->type != GGML_TYPE_F32) {
        x = ggml_cast(ctx, x, GGML_TYPE_F32);
    }

    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), weight), bias);
}

ggml_tensor * linear(
    ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * bias,
    ggml_tensor * x) {

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

// ---------------------------------------------------------------------------
// Conv1D dispatch
//
// - 3D or F16 weights: native ggml_conv_1d
// - 2D quantized weights: im2col + quantized mul_mat
// ---------------------------------------------------------------------------

ggml_tensor * conv1d(
    ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * input,
    int stride,
    int padding,
    int dilation,
    int kernel_size,
    bool force_contiguous_im2col) {

    GGML_ASSERT(kernel_size > 0);

    const bool kernel_is_3d =
        (weight->ne[2] > 1) ||
        (weight->ne[0] == kernel_size);

    if (kernel_is_3d) {
        return ggml_conv_1d(ctx, weight, input, stride, padding, dilation);
    }

    const int64_t ick = weight->ne[0];
    const int64_t oc  = weight->ne[1];

    GGML_ASSERT(ick % kernel_size == 0);
    const int64_t ic = ick / kernel_size;

    if (weight->type == GGML_TYPE_F16) {
        ggml_tensor * w3d = ggml_reshape_3d(ctx, weight, kernel_size, ic, oc);
        return ggml_conv_1d(ctx, w3d, input, stride, padding, dilation);
    }

    return conv1d_im2col_mulmat(ctx, weight, input, stride, padding, dilation, kernel_size, false, force_contiguous_im2col);
}

ggml_tensor * conv1d_chfirst(
    ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * input,
    int stride,
    int padding,
    int dilation,
    int kernel_size,
    bool force_contiguous_im2col = false) {

    GGML_ASSERT(kernel_size > 0);

    const bool kernel_is_3d =
        (weight->ne[2] > 1) ||
        (weight->ne[0] == kernel_size);

    if (kernel_is_3d || weight->type == GGML_TYPE_F16) {
        ggml_tensor * out = conv1d(ctx, weight, input, stride, padding, dilation, kernel_size);
        return ggml_cont(ctx, ggml_transpose(ctx, out));
    }

    return conv1d_im2col_mulmat(ctx, weight, input, stride, padding, dilation, kernel_size, true, force_contiguous_im2col);
}

// ---------------------------------------------------------------------------
// ConvTranspose1D helpers
// ---------------------------------------------------------------------------

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
    [[maybe_unused]] Model & model,
    ggml_tensor * weight,
    ggml_tensor * input,
    ggml_tensor * bias,
    int stride,
    int crop_left,
    int out_len) {

    if (std::getenv("KOKOPOP_CPU_VOCODER_CONVT") != nullptr &&
        model.backend_type == KOKOPOP_BACKEND_CPU &&
        input->type == GGML_TYPE_F32 &&
        (weight->type == GGML_TYPE_F32 || weight->type == GGML_TYPE_F16) &&
        bias->type == GGML_TYPE_F32 &&
        input->ne[1] == weight->ne[2] &&
        bias->ne[0] == weight->ne[1] &&
        out_len > 0) {

        model.metal_vocoder_convt_params.push_back({
            nullptr,
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
            cpu_vocoder_convt_callback,
            GGML_N_TASKS_MAX,
            const_cast<MetalVocoderConvTransposeParams *>(params));
    }

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

    return ggml_add(ctx, ggml_add(ctx, normed, ggml_mul(ctx, normed, gamma)), beta);
}

ggml_tensor * adain_1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * style,
    const AdaIn1dWeights & weights) {

    ggml_tensor * gamma = linear(ctx, weights.gamma_w, weights.gamma_b, style);
    ggml_tensor * beta  = linear(ctx, weights.beta_w,  weights.beta_b,  style);
    ggml_tensor * cur   = ggml_norm(ctx, x, 1e-5f);

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
// Snake1D public wrappers
// ---------------------------------------------------------------------------

ggml_tensor * graph_snake1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * alpha,
    const std::string & alpha_name,
    std::string & error) {

    return snake1d_impl(ctx, x, alpha, alpha_name, error, false);
}

ggml_tensor * graph_snake1d(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    const std::string & alpha_name,
    std::string & error) {

    ggml_tensor * alpha = require_tensor(model, alpha_name.c_str(), error);
    if (!error.empty()) {
        return nullptr;
    }

    return snake1d_impl(ctx, x, alpha, alpha_name, error, model.backend_type == KOKOPOP_BACKEND_CPU);
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

    ggml_tensor * conv1   = nullptr;
    ggml_tensor * conv1_b = nullptr;
    ggml_tensor * conv2   = nullptr;
    ggml_tensor * conv2_b = nullptr;

    if (cached != nullptr) {
        conv1   = cached->conv1_w;
        conv1_b = cached->conv1_b;
        conv2   = cached->conv2_w;
        conv2_b = cached->conv2_b;
    } else {
        conv1   = require_tensor(model, (prefix + ".conv1.weight").c_str(), error);
        conv1_b = require_tensor(model, (prefix + ".conv1.bias").c_str(), error);
        conv2   = require_tensor(model, (prefix + ".conv2.weight").c_str(), error);
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

    cur = add_channel_bias(ctx, conv1d(ctx, conv1, cur, 1, 1, 1, 3, backend_needs_contiguous_im2col(model)), conv1_b);

    cur = cached != nullptr
        ? adain_1d(ctx, cur, style, cached->norm2)
        : adain_1d(ctx, model, cur, style, prefix + ".norm2", error);
    if (cur == nullptr) {
        return nullptr;
    }

    cur = ggml_leaky_relu(ctx, cur, 0.2f, false);
    cur = add_channel_bias(ctx, conv1d(ctx, conv2, cur, 1, 1, 1, 3, backend_needs_contiguous_im2col(model)), conv2_b);

    ggml_tensor * residual = maybe_upsample_nearest(ctx, x, upsample);
    ggml_tensor * conv1x1 = cached != nullptr ? cached->conv1x1_w : model.cached_tensor(prefix + ".conv1x1.weight");
    if (conv1x1 != nullptr) {
        residual = conv1d(ctx, conv1x1, residual, 1, 0, 1, 1, backend_needs_contiguous_im2col(model));
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
    std::string & error,
    bool fused_snake,
    bool force_contiguous_im2col) {

    for (int i = 0; i < 3; ++i) {
        const size_t idx = static_cast<size_t>(i);

        ggml_tensor * cur = adain_1d(ctx, x, style, weights.adain1[idx]);
        cur = snake1d_impl(ctx, cur, weights.alpha1[idx], prefix + ".alpha1." + std::to_string(i), error, fused_snake);
        if (cur == nullptr) {
            return nullptr;
        }

        cur = add_channel_bias(
            ctx,
            conv1d(ctx, weights.convs1_w[idx], cur, 1, weights.paddings[idx], KOKOPOP_RESBLOCK_DILATIONS[i], kernel_size, force_contiguous_im2col),
            weights.convs1_b[idx]);

        cur = adain_1d(ctx, cur, style, weights.adain2[idx]);
        cur = snake1d_impl(ctx, cur, weights.alpha2[idx], prefix + ".alpha2." + std::to_string(i), error, fused_snake);
        if (cur == nullptr) {
            return nullptr;
        }

        cur = add_channel_bias(
            ctx,
            conv1d(ctx, weights.convs2_w[idx], cur, 1, kernel_size / 2, 1, kernel_size, force_contiguous_im2col),
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
        return graph_generator_resblock(
            ctx,
            x,
            style,
            prefix,
            kernel_size,
            cached->second,
            error,
            model.backend_type == KOKOPOP_BACKEND_CPU,
            backend_needs_contiguous_im2col(model));
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

        ggml_tensor * conv1_w = require_tensor(model, (prefix + ".convs1." + std::to_string(i) + ".weight").c_str(), error);
        ggml_tensor * conv1_b = require_tensor(model, (prefix + ".convs1." + std::to_string(i) + ".bias").c_str(), error);
        if (!error.empty()) {
            return nullptr;
        }

        cur = add_channel_bias(
            ctx,
            conv1d(ctx, conv1_w, cur, 1, paddings[i], KOKOPOP_RESBLOCK_DILATIONS[i], kernel_size, backend_needs_contiguous_im2col(model)),
            conv1_b);

        cur = adain_1d(ctx, model, cur, style, prefix + ".adain2." + std::to_string(i), error);
        if (cur == nullptr) {
            return nullptr;
        }

        cur = graph_snake1d(ctx, model, cur, prefix + ".alpha2." + std::to_string(i), error);
        if (cur == nullptr) {
            return nullptr;
        }

        ggml_tensor * conv2_w = require_tensor(model, (prefix + ".convs2." + std::to_string(i) + ".weight").c_str(), error);
        ggml_tensor * conv2_b = require_tensor(model, (prefix + ".convs2." + std::to_string(i) + ".bias").c_str(), error);
        if (!error.empty()) {
            return nullptr;
        }

        cur = add_channel_bias(
            ctx,
            conv1d(ctx, conv2_w, cur, 1, kernel_size / 2, 1, kernel_size, backend_needs_contiguous_im2col(model)),
            conv2_b);

        x = ggml_add(ctx, x, cur);
    }

    return x;
}

// ---------------------------------------------------------------------------
// LSTM graph construction
// ---------------------------------------------------------------------------

ggml_tensor * col_view(ggml_context * ctx, ggml_tensor * x, int64_t index) {
    return ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], index * x->nb[1]);
}

struct LstmWeights {
    ggml_tensor * w_ih_packed = nullptr;
    ggml_tensor * w_hh_packed = nullptr;
    ggml_tensor * b_ih_packed = nullptr;
    ggml_tensor * b_hh_packed = nullptr;
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

    ggml_tensor * pre_input_gates_packed = ggml_add(
        ctx,
        ggml_mul_mat(ctx, w.w_ih_packed, input),
        w.b_ih_packed);

    const std::string whh_key = prefix + ".weight_hh_l0" + (reverse ? "_reverse" : "");
    const auto it = model.lstm_w_hh_f32.find(whh_key);
    if (it == model.lstm_w_hh_f32.end()) {
        error = "fused LSTM: w_hh not preloaded for " + whh_key;
        return nullptr;
    }

    const std::string b_hh_key = prefix + ".bias_hh_l0" + (reverse ? "_reverse" : "");
    const auto b_it = model.lstm_b_hh_f32.find(b_hh_key);
    if (b_it == model.lstm_b_hh_f32.end()) {
        error = "fused LSTM: b_hh not preloaded for " + b_hh_key;
        return nullptr;
    }

    const auto rowwise_it = model.lstm_w_hh_rowwise.find(whh_key);
    const float * rowwise = rowwise_it != model.lstm_w_hh_rowwise.end()
        ? rowwise_it->second.data()
        : nullptr;

    model.lstm_custom_params.push_back({
        it->second.data(),
        b_it->second.data(),
        rowwise,
        model.backend->use_metal_lstm(n_steps) ? model.backend->metal_lstm_kernel() : nullptr,
        it->first.c_str(),
        hidden,
        n_steps,
        reverse,
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
    ggml_tensor * bw = lstm_direction(ctx, model, input, prefix, true,  n_steps, error);
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
        const int ada_index  = lstm_index + 1;

        cur = bidirectional_lstm(
            ctx,
            model,
            cur,
            "kokopop.predictor.text_encoder.lstms." + std::to_string(lstm_index),
            n_steps,
            error);
        if (cur == nullptr) {
            return nullptr;
        }

        cur = ada_layer_norm(
            ctx,
            model,
            cur,
            style,
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

        ggml_tensor * input_tw = ggml_cont(ctx, ggml_transpose(ctx, cur));
        cur = add_channel_bias(ctx, conv1d_chfirst(ctx, conv_w, input_tw, 1, 2, 1, 5, backend_needs_contiguous_im2col(model)), conv_b);
        cur = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, cur, 1e-5f), norm_w), norm_b);
        cur = ggml_leaky_relu(ctx, cur, 0.2f, false);
    }

    cur = bidirectional_lstm(ctx, model, cur, "kokopop.text_encoder.lstm", n_tokens, error);
    if (cur == nullptr) {
        return nullptr;
    }

    return ggml_mul_mat(
        ctx,
        ggml_cont(ctx, ggml_transpose(ctx, cur)),
        ggml_cont(ctx, ggml_transpose(ctx, duration_mask)));
}

} // namespace kokopop
