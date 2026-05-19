#include "kokoro.h"
#include "lstm_fused.h"

#ifdef KOKOPOP_HAS_METAL
#include "backend/metal_vocoder.h"
#include "backend/metal_lstm.h"
#endif

#include "core/constants.h"

#include <cmath>
#include <cstdio>
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

#ifdef KOKOPOP_HAS_METAL

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
    int crop_left) {

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

    for (int64_t oc = 0; oc < oc_count; ++oc) {
        const float b = *reinterpret_cast<const float *>(
            tensor_data_c(bias) + static_cast<size_t>(oc) * bias->nb[0]);

        for (int64_t t = 0; t < ol; ++t) {
            const int64_t full_t = t + crop_left;
            float acc = b;

            for (int64_t ic = 0; ic < ic_count; ++ic) {
                for (int64_t k = 0; k < k_count; ++k) {
                    const int64_t src_num = full_t - k;
                    if (src_num < 0 || (src_num % stride) != 0) {
                        continue;
                    }

                    const int64_t ti = src_num / stride;
                    if (ti < 0 || ti >= il) {
                        continue;
                    }

                    const float x = tensor_get_f32_2d(input, ti, ic);
                    const float w = tensor_get_weight_f32_3d(weight, k, oc, ic);
                    acc += w * x;
                }
            }

            tensor_set_f32_2d(dst, t, oc, acc);
        }
    }
}

void metal_vocoder_convt_callback(
    ggml_tensor * dst,
    const ggml_tensor * output_storage,
    const ggml_tensor * input,
    const ggml_tensor * weight,
    int ith,
    int /*nth*/,
    void * userdata) {

    if (ith != 0) return;
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

#ifdef KOKOPOP_HAS_METAL
// ---------------------------------------------------------------------------
// Generator ResBlock Metal callback
// ---------------------------------------------------------------------------
// CPU-side linear projection: out[oc] = sum_d(w[d + ne[0]*oc] * style[d]) + b[oc]
// w layout follows ggml_mul_mat: ne[0]=style_dim, ne[1]=IC → row-major per output channel.
static void metal_resblock_project_style(
    const ggml_tensor * w,
    const ggml_tensor * b,
    const float       * style,
    float             * out
) {
    const int64_t OC  = w->ne[1];
    const int64_t dim = w->ne[0];
    const float * bias = static_cast<const float *>(b->data);
    for (int64_t oc = 0; oc < OC; ++oc) {
        float sum = bias[oc];
        if (w->type == GGML_TYPE_F16) {
            const ggml_fp16_t * row = static_cast<const ggml_fp16_t *>(w->data) + oc * dim;
            for (int64_t d = 0; d < dim; ++d) sum += ggml_fp16_to_fp32(row[d]) * style[d];
        } else {
            const float * row = static_cast<const float *>(w->data) + oc * dim;
            for (int64_t d = 0; d < dim; ++d) sum += row[d] * style[d];
        }
        out[oc] = sum;
    }
}

// ggml_map_custom3_inplace callback: dst = resblock(x=b, style=c)
// Mirrors the convt pattern: a = dst (fresh tensor, first arg = inplace target).
// b = x (input, read-only), c = style (read-only).
static void metal_generator_resblock_callback(
    ggml_tensor       * dst,
    const ggml_tensor * /*out_ignored*/,  // a == dst, not used
    const ggml_tensor * x_in,             // b = x
    const ggml_tensor * style,            // c = style
    int ith, int /*nth*/,
    void * userdata
) {
    if (ith != 0) return;

    const auto * p = static_cast<const MetalGeneratorResblockParams *>(userdata);
    if (p == nullptr || p->kernel == nullptr || p->weights == nullptr) {
        return;
    }

    const auto * w      = static_cast<const GeneratorResblockWeights *>(p->weights);
    auto       * kernel = static_cast<MetalVocoderState *>(p->kernel);

    const int64_t T  = dst->ne[0];
    const int64_t IC = dst->ne[1];

    // Project all 6 adain gamma/beta pairs on CPU (12 * IC floats total).
    // Thread-local scratch — avoids a heap allocation per ResBlock call on the
    // hot generator path. Safe because the callback is serialized (ith != 0
    // returns early) and the projection is fully consumed before the GPU
    // call below releases it.
    static thread_local std::vector<float> projs;
    projs.assign(static_cast<size_t>(12 * IC), 0.0f);
    const float * style_f32 = static_cast<const float *>(style->data);

    for (int iter = 0; iter < 3; ++iter) {
        float * base = projs.data() + static_cast<size_t>(iter * 4 * IC);
        metal_resblock_project_style(w->adain1[iter].gamma_w, w->adain1[iter].gamma_b, style_f32, base + 0 * IC);
        metal_resblock_project_style(w->adain1[iter].beta_w,  w->adain1[iter].beta_b,  style_f32, base + 1 * IC);
        metal_resblock_project_style(w->adain2[iter].gamma_w, w->adain2[iter].gamma_b, style_f32, base + 2 * IC);
        metal_resblock_project_style(w->adain2[iter].beta_w,  w->adain2[iter].beta_b,  style_f32, base + 3 * IC);
    }

    // Build per-iteration weight descriptors.
    MetalResblockIterWeights iters[3];
    for (int i = 0; i < 3; ++i) {
        const int d    = KOKOPOP_RESBLOCK_DILATIONS[i];
        const int pad2 = p->kernel_size / 2;

        iters[i] = {
            w->adain1[i].norm_w, w->adain1[i].norm_b, w->alpha1[i],
            w->convs1_w[i], w->convs1_b[i], p->kernel_size, d, w->paddings[i],
            w->adain2[i].norm_w, w->adain2[i].norm_b, w->alpha2[i],
            w->convs2_w[i], w->convs2_b[i], p->kernel_size, 1, pad2,
        };
    }

    const float * x_data  = static_cast<const float *>(x_in->data);
    float       * out_data = static_cast<float *>(dst->data);

    if (!metal_vocoder_run_generator_resblocks(kernel, x_data, out_data, T, IC, projs.data(), iters)) {
        std::fprintf(stderr, "[metal_vocoder] generator_resblock GPU call failed — output undefined\n");
    }
}

// Helper: build MetalResblockIterWeights[3] and pack 12*IC style projections
// for a given GeneratorResblockWeights + kernel_size + style.
static void build_resblock_iters_and_projs(
    const GeneratorResblockWeights & w,
    int kernel_size,
    const float * style, int64_t style_dim, int64_t IC,
    MetalResblockIterWeights iters[3],
    float * projs_out  // [12 * IC]
) {
    (void)style_dim;
    for (int it = 0; it < 3; ++it) {
        float * base = projs_out + static_cast<size_t>(it * 4 * IC);
        metal_resblock_project_style(w.adain1[it].gamma_w, w.adain1[it].gamma_b, style, base + 0 * IC);
        metal_resblock_project_style(w.adain1[it].beta_w,  w.adain1[it].beta_b,  style, base + 1 * IC);
        metal_resblock_project_style(w.adain2[it].gamma_w, w.adain2[it].gamma_b, style, base + 2 * IC);
        metal_resblock_project_style(w.adain2[it].beta_w,  w.adain2[it].beta_b,  style, base + 3 * IC);

        const int d    = KOKOPOP_RESBLOCK_DILATIONS[it];
        const int pad2 = kernel_size / 2;
        iters[it] = {
            w.adain1[it].norm_w, w.adain1[it].norm_b, w.alpha1[it],
            w.convs1_w[it], w.convs1_b[it], kernel_size, d, w.paddings[it],
            w.adain2[it].norm_w, w.adain2[it].norm_b, w.alpha2[it],
            w.convs2_w[it], w.convs2_b[it], kernel_size, 1, pad2,
        };
    }
}

// ggml_map_custom3_inplace callback for the fused per-stage generator op.
//   a = dst   (in-place target, holds post-stage x — shape [T_post_pad, IC_x_out])
//   b = x_in  (read-only, post-decoder or post-previous-stage)
//   c = style (read-only)
// har_t is accessed via params->har_t->data.
static void metal_generator_stage_callback(
    ggml_tensor       * dst,
    const ggml_tensor * /*out_ignored*/,
    const ggml_tensor * x_in,
    const ggml_tensor * style,
    int ith, int /*nth*/,
    void * userdata
) {
    if (ith != 0) return;

    const auto * p = static_cast<const MetalGeneratorStageParams *>(userdata);
    if (p == nullptr || p->kernel == nullptr || p->har_t == nullptr ||
        p->noise_resblock == nullptr) {
        std::fprintf(stderr, "[metal_vocoder] stage callback: missing params\n");
        return;
    }

    auto * kernel = static_cast<MetalVocoderState *>(p->kernel);

    const auto * w_noise = static_cast<const GeneratorResblockWeights *>(p->noise_resblock);
    const GeneratorResblockWeights * w_main[3] = {
        static_cast<const GeneratorResblockWeights *>(p->main_resblocks[0]),
        static_cast<const GeneratorResblockWeights *>(p->main_resblocks[1]),
        static_cast<const GeneratorResblockWeights *>(p->main_resblocks[2]),
    };
    if (w_main[0] == nullptr || w_main[1] == nullptr || w_main[2] == nullptr) {
        std::fprintf(stderr, "[metal_vocoder] stage callback: missing main resblock weights\n");
        return;
    }

    const int64_t T_x_in     = x_in->ne[0];
    const int64_t T_post_pad = dst->ne[0];
    const int64_t har_len    = p->har_t->ne[0];
    const int64_t har_C      = p->har_t->ne[1];

    const int64_t IC_noise = w_noise->adain1[0].gamma_w->ne[1];
    const int64_t IC_main  = w_main[0]->adain1[0].gamma_w->ne[1];
    const int64_t style_dim = w_noise->adain1[0].gamma_w->ne[0];

    // Pack style projections for noise + 3 main resblocks.
    const size_t proj_n = static_cast<size_t>(12 * IC_noise + 3 * 12 * IC_main);
    static thread_local std::vector<float> projs;
    projs.assign(proj_n, 0.0f);

    const float * style_f32 = static_cast<const float *>(style->data);

    MetalResblockIterWeights noise_iters[3];
    build_resblock_iters_and_projs(*w_noise, p->noise_kernel_size,
                                    style_f32, style_dim, IC_noise,
                                    noise_iters, projs.data());

    MetalResblockIterWeights main_iters[3][3];
    for (int br = 0; br < 3; ++br) {
        float * base = projs.data() + static_cast<size_t>(12 * IC_noise + br * 12 * IC_main);
        build_resblock_iters_and_projs(*w_main[br], p->main_kernel_sizes[br],
                                        style_f32, style_dim, IC_main,
                                        main_iters[br], base);
    }

    const float * x_in_data = static_cast<const float *>(x_in->data);
    const float * har_data  = static_cast<const float *>(p->har_t->data);
    float       * out_data  = static_cast<float *>(dst->data);

    if (!metal_vocoder_run_stage(
            kernel,
            x_in_data, T_x_in,
            har_data,  har_len, har_C,
            projs.data(),
            IC_noise, IC_main,
            out_data, T_post_pad,
            p->up_stride, p->up_padding,
            p->pad_reflect_left1,
            p->noise_kernel, p->noise_stride, p->noise_padding,
            p->noise_conv_w, p->noise_conv_b,
            p->up_w, p->up_b,
            noise_iters, main_iters)) {
        std::fprintf(stderr, "[metal_vocoder] stage GPU call failed — output undefined\n");
    }
}
#endif // KOKOPOP_HAS_METAL

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
    bool channel_first) {

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

    // Keep this for the CUDA path until the corruption is fully isolated.
    // If removing it does not change results, remove it later for performance.
    im2col_2d = ggml_cont(ctx, im2col_2d);

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
    int kernel_size) {

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

    return conv1d_im2col_mulmat(ctx, weight, input, stride, padding, dilation, kernel_size, false);
}

ggml_tensor * conv1d_chfirst(
    ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * input,
    int stride,
    int padding,
    int dilation,
    int kernel_size) {

    GGML_ASSERT(kernel_size > 0);

    const bool kernel_is_3d =
        (weight->ne[2] > 1) ||
        (weight->ne[0] == kernel_size);

    if (kernel_is_3d || weight->type == GGML_TYPE_F16) {
        ggml_tensor * out = conv1d(ctx, weight, input, stride, padding, dilation, kernel_size);
        return ggml_cont(ctx, ggml_transpose(ctx, out));
    }

    return conv1d_im2col_mulmat(ctx, weight, input, stride, padding, dilation, kernel_size, true);
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

#ifdef KOKOPOP_HAS_METAL
    if (model.backend != nullptr &&
        model.backend->metal_vocoder_kernel() != nullptr &&
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
    ggml_tensor * conv1x1 = cached != nullptr ? cached->conv1x1_w : model.cached_tensor(prefix + ".conv1x1.weight");
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
    std::string & error,
    bool fused_snake) {

    for (int i = 0; i < 3; ++i) {
        const size_t idx = static_cast<size_t>(i);

        ggml_tensor * cur = adain_1d(ctx, x, style, weights.adain1[idx]);
        cur = snake1d_impl(ctx, cur, weights.alpha1[idx], prefix + ".alpha1." + std::to_string(i), error, fused_snake);
        if (cur == nullptr) {
            return nullptr;
        }

        cur = add_channel_bias(
            ctx,
            conv1d(ctx, weights.convs1_w[idx], cur, 1, weights.paddings[idx], KOKOPOP_RESBLOCK_DILATIONS[i], kernel_size),
            weights.convs1_b[idx]);

        cur = adain_1d(ctx, cur, style, weights.adain2[idx]);
        cur = snake1d_impl(ctx, cur, weights.alpha2[idx], prefix + ".alpha2." + std::to_string(i), error, fused_snake);
        if (cur == nullptr) {
            return nullptr;
        }

        cur = add_channel_bias(
            ctx,
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
#ifdef KOKOPOP_HAS_METAL
        if (model.backend != nullptr && model.backend->metal_vocoder_kernel() != nullptr &&
            cached->second.valid()) {
            model.metal_vocoder_resblock_params.push_back({
                model.backend->metal_vocoder_kernel(),
                &cached->second,
                kernel_size,
                cached->second.adain1[0].gamma_w->ne[0],
            });
            const MetalGeneratorResblockParams * params = &model.metal_vocoder_resblock_params.back();

            ggml_tensor * out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1]);
            return ggml_map_custom3_inplace(
                ctx, out, x, style,
                metal_generator_resblock_callback, 1,
                const_cast<MetalGeneratorResblockParams *>(params));
        }
#endif
        return graph_generator_resblock(
            ctx,
            x,
            style,
            prefix,
            kernel_size,
            cached->second,
            error,
            model.backend_type == KOKOPOP_BACKEND_CPU);
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
            conv1d(ctx, conv1_w, cur, 1, paddings[i], KOKOPOP_RESBLOCK_DILATIONS[i], kernel_size),
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
            conv1d(ctx, conv2_w, cur, 1, kernel_size / 2, 1, kernel_size),
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

// ---------------------------------------------------------------------------
// 3-branch main resblock + sum/3
// ---------------------------------------------------------------------------
// Explicit per-branch construction. On Metal, this path is only reached as
// the fallback when graph_generator_stage_fused() returns nullptr (e.g.
// some resblock weight missing from the cache); the normal Metal path
// fuses everything in graph_generator_stage_fused().
ggml_tensor * graph_3branch_main_sum(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    int stage,
    std::string & error) {

    ggml_tensor * accum = nullptr;
    for (int k = 0; k < 3; ++k) {
        const std::string prefix = "kokopop.decoder.generator.resblocks." +
                                    std::to_string(stage * 3 + k);
        ggml_tensor * branch = graph_generator_resblock(
            ctx, model, x, style, prefix, KOKOPOP_RESBLOCK_KERNELS[k], error);
        if (branch == nullptr) {
            return nullptr;
        }
        accum = (accum == nullptr) ? branch : ggml_add(ctx, accum, branch);
    }
    return ggml_scale(ctx, accum, 1.0f / 3.0f);
}

// ---------------------------------------------------------------------------
// Fused per-stage generator graph helper
// ---------------------------------------------------------------------------
ggml_tensor * graph_generator_stage_fused(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    ggml_tensor * har_t,
    int stage,
    std::string & error) {

    (void)error;

#ifndef KOKOPOP_HAS_METAL
    (void)ctx; (void)model; (void)x; (void)style; (void)har_t; (void)stage;
    return nullptr;
#else
    if (model.backend == nullptr || model.backend->metal_vocoder_kernel() == nullptr) {
        return nullptr;
    }
    if (stage < 0 || stage > 1) return nullptr;

    // Per-stage constants (mirror the table in audio_utils.cpp).
    struct StageConst {
        int kernel_size_noise;
        int up_stride;
        int up_padding;
        int noise_stride;
        int noise_padding;
        int noise_kernel;
        bool pad_left1;
        const char * noise_prefix;
        const char * up_weight;
        const char * up_bias;
        const char * noise_conv_w;
        const char * noise_conv_b;
    };
    static const StageConst kStages[2] = {
        {7, 10, 5, 6, 3, 12, false,
         "kokopop.decoder.generator.noise_res.0",
         "kokopop.decoder.generator.ups.0.weight",
         "kokopop.decoder.generator.ups.0.bias",
         "kokopop.decoder.generator.noise_convs.0.weight",
         "kokopop.decoder.generator.noise_convs.0.bias"},
        {11, 6, 3, 1, 0, 1, true,
         "kokopop.decoder.generator.noise_res.1",
         "kokopop.decoder.generator.ups.1.weight",
         "kokopop.decoder.generator.ups.1.bias",
         "kokopop.decoder.generator.noise_convs.1.weight",
         "kokopop.decoder.generator.noise_convs.1.bias"},
    };
    const StageConst & sc = kStages[stage];

    // Look up cached resblock weights (noise + 3 main).
    const auto it_noise = model.generator_resblock_weights.find(sc.noise_prefix);
    if (it_noise == model.generator_resblock_weights.end() || !it_noise->second.valid()) {
        return nullptr;
    }
    const GeneratorResblockWeights * w_main[3] = {nullptr, nullptr, nullptr};
    for (int br = 0; br < 3; ++br) {
        const std::string pfx = "kokopop.decoder.generator.resblocks." +
                                 std::to_string(stage * 3 + br);
        const auto it = model.generator_resblock_weights.find(pfx);
        if (it == model.generator_resblock_weights.end() || !it->second.valid()) {
            return nullptr;
        }
        w_main[br] = &it->second;
    }

    // Look up weight tensors.
    std::string local_err;
    ggml_tensor * noise_conv_w = require_tensor(model, sc.noise_conv_w, local_err);
    ggml_tensor * noise_conv_b = require_tensor(model, sc.noise_conv_b, local_err);
    ggml_tensor * up_w = require_tensor(model, sc.up_weight, local_err);
    ggml_tensor * up_b = require_tensor(model, sc.up_bias,   local_err);
    if (!local_err.empty()) return nullptr;

    // Derive output shape (matches audio_utils.cpp formula).
    const int64_t T_x_in    = x->ne[0];
    const int64_t IC_x_out  = up_w->ne[1];
    const int64_t T_convt   = (T_x_in - 1) * sc.up_stride - 2 * sc.up_padding + up_w->ne[0];
    const int64_t T_post_pad = sc.pad_left1 ? T_convt + 1 : T_convt;

    // Build params (pushed into stable storage in the model).
    MetalGeneratorStageParams params{};
    params.kernel          = model.backend->metal_vocoder_kernel();
    params.har_t           = har_t;
    params.noise_conv_w    = noise_conv_w;
    params.noise_conv_b    = noise_conv_b;
    params.noise_kernel    = sc.noise_kernel;
    params.noise_stride    = sc.noise_stride;
    params.noise_padding   = sc.noise_padding;
    params.up_w            = up_w;
    params.up_b            = up_b;
    params.up_stride       = sc.up_stride;
    params.up_padding      = sc.up_padding;
    params.pad_reflect_left1 = sc.pad_left1;
    params.noise_resblock    = &it_noise->second;
    params.noise_kernel_size = sc.kernel_size_noise;
    for (int br = 0; br < 3; ++br) {
        params.main_resblocks[br]     = w_main[br];
        params.main_kernel_sizes[br]  = KOKOPOP_RESBLOCK_KERNELS[br];
    }
    params.style_dim = w_main[0]->adain1[0].gamma_w->ne[0];

    model.metal_vocoder_stage_params.push_back(params);
    const MetalGeneratorStageParams * stable = &model.metal_vocoder_stage_params.back();

    ggml_tensor * out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_post_pad, IC_x_out);
    return ggml_map_custom3_inplace(
        ctx, out, x, style,
        metal_generator_stage_callback, 1,
        const_cast<MetalGeneratorStageParams *>(stable));
#endif
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

    // Pre-gates matmul. Runs `pre_gates = w_ih @ input` with an explicit
    // F32 kernel for CPU/Metal, matching the layout used by the Metal kernel:
    //   w_ih[i + I*g] * input[i + I*t] -> pre_gates[g + 4H*t].
    ggml_tensor * mul_result = nullptr;
#ifdef KOKOPOP_HAS_METAL
    const bool use_pregates_metal =
        model.backend != nullptr &&
        model.backend->metal_lstm_kernel() != nullptr;
#else
    const bool use_pregates_metal = false;
#endif

    const std::string wih_key = prefix + ".weight_ih_l0" + (reverse ? "_reverse" : "");
    const auto wih_it = model.lstm_w_ih_f32.find(wih_key);

    // CUDA/Vulkan keep using backend mul_mat; CPU and Metal use pre-dequantized
    // F32 weights for numerically stable LSTM pre-gates.
    ggml_tensor * w_ih = w.w_ih_packed;
    const bool use_backend_pregates =
        model.backend_type == KOKOPOP_BACKEND_CUDA ||
        model.backend_type == KOKOPOP_BACKEND_VULKAN;
    if (!use_backend_pregates) {
        if (wih_it == model.lstm_w_ih_f32.end()) {
            error = "fused LSTM: w_ih not preloaded for " + wih_key;
            return nullptr;
        }
    }

    if (!use_backend_pregates) {
        const int64_t I      = w.w_ih_packed->ne[0];
        const int64_t four_H = w.w_ih_packed->ne[1];

        model.lstm_pregates_params.push_back(LstmPregatesParams{
            use_pregates_metal ? model.backend->metal_lstm_kernel() : nullptr,
            wih_it->first.c_str(),
            wih_it->second.data(),
            static_cast<int>(I),
            static_cast<int>(four_H),
            static_cast<int>(n_steps),
        });
        const LstmPregatesParams * pg = &model.lstm_pregates_params.back();

        ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, four_H, n_steps);
        mul_result = ggml_map_custom2_inplace(
            ctx, dst, input,
            lstm_pregates_callback,
            use_pregates_metal ? 1 : GGML_N_TASKS_MAX,
            const_cast<LstmPregatesParams *>(pg));
    }

    if (mul_result == nullptr) {
        mul_result = ggml_mul_mat(ctx, w_ih, input);
    }
    ggml_tensor * pre_input_gates_packed = ggml_add(ctx, mul_result, w.b_ih_packed);

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
        model.backend->metal_lstm_kernel(),
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
        cur = add_channel_bias(ctx, conv1d_chfirst(ctx, conv_w, input_tw, 1, 2, 1, 5), conv_b);
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
