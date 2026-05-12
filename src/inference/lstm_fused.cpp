#include "lstm_fused.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef KOKOPOP_HAS_METAL
#include "backend/metal_lstm.h"
#endif

namespace kokopop {
namespace {

// ---------------------------------------------------------------------------
// Tensor helpers
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

static inline bool tensor_is_f32_2d_contiguous(const ggml_tensor * t, int64_t ne0, int64_t ne1) {
    return t != nullptr &&
           t->type == GGML_TYPE_F32 &&
           t->data != nullptr &&
           t->ne[0] == ne0 &&
           t->ne[1] == ne1 &&
           t->nb[0] == sizeof(float) &&
           t->nb[1] == static_cast<size_t>(ne0) * sizeof(float);
}

static inline float sigmoidf_stable(float x) {
    // Good enough for LSTM gate ranges and avoids expf overflow noise.
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

static inline float clamp_cell(float x) {
    if (x >  50.0f) return  50.0f;
    if (x < -50.0f) return -50.0f;
    return x;
}

// ---------------------------------------------------------------------------
// Dot-product helpers
//
// The AVX2 path intentionally preserves scalar accumulation order after each
// 8-lane product batch. This avoids larger numerical drift in duration models.
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static float dot_product_avx2_ordered(
    const float * __restrict__ a,
    const float * __restrict__ b,
    int64_t len) {

    alignas(32) float products[8];
    float dot = 0.0f;

    int64_t i = 0;
    for (; i + 7 < len; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 vp = _mm256_mul_ps(va, vb);
        _mm256_store_ps(products, vp);

        dot += products[0];
        dot += products[1];
        dot += products[2];
        dot += products[3];
        dot += products[4];
        dot += products[5];
        dot += products[6];
        dot += products[7];
    }

    for (; i < len; ++i) {
        dot += a[i] * b[i];
    }

    return dot;
}

#endif

#ifdef __ARM_NEON

static float dot_product_neon(
    const float * __restrict__ a,
    const float * __restrict__ b,
    int64_t len) {

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int64_t i = 0;
    for (; i + 15 < len; i += 16) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a + i +  4), vld1q_f32(b + i +  4));
        acc2 = vmlaq_f32(acc2, vld1q_f32(a + i +  8), vld1q_f32(b + i +  8));
        acc3 = vmlaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    float dot = vgetq_lane_f32(acc0, 0)
              + vgetq_lane_f32(acc0, 1)
              + vgetq_lane_f32(acc0, 2)
              + vgetq_lane_f32(acc0, 3);

    for (; i < len; ++i) {
        dot += a[i] * b[i];
    }

    return dot;
}

#endif

static float dot_product_scalar(
    const float * __restrict__ a,
    const float * __restrict__ b,
    int64_t len) {

    float dot = 0.0f;
    for (int64_t i = 0; i < len; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

static inline float dot_product(
    const float * __restrict__ a,
    const float * __restrict__ b,
    int64_t len) {

#ifdef __AVX2__
    return dot_product_avx2_ordered(a, b, len);
#elif defined(__ARM_NEON)
    return dot_product_neon(a, b, len);
#else
    return dot_product_scalar(a, b, len);
#endif
}

// ---------------------------------------------------------------------------
// LSTM recurrence, contiguous fast path
//
// Layout:
//   pre_gates[g, t] = pre_gates[g + 4*H*t]
//   output[h, t]    = output[h + H*t]
//   w_rowwise[j,k]  = w_rowwise[j*H + k]
// ---------------------------------------------------------------------------

static void cpu_lstm_contiguous(
    const LstmCustomParams & p,
    const float * __restrict__ pre_gates,
    float * __restrict__ output) {

    const int64_t H  = p.hidden;
    const int64_t N  = p.n_steps;
    const int64_t H4 = 4 * H;

    std::vector<float> h(static_cast<size_t>(H), 0.0f);
    std::vector<float> c(static_cast<size_t>(H), 0.0f);
    std::vector<float> gates(static_cast<size_t>(H4));

    const float * w_base = p.w_hh_rowwise ? p.w_hh_rowwise : p.w_hh_f32;

    for (int64_t step = 0; step < N; ++step) {
        const int64_t t = p.reverse ? (N - 1 - step) : step;
        const float * pg_t = pre_gates + H4 * t;

        for (int64_t j = 0; j < H4; ++j) {
            gates[static_cast<size_t>(j)] =
                p.b_hh[j] + pg_t[j] + dot_product(w_base + j * H, h.data(), H);
        }

        float * out_t = output + H * t;
        for (int64_t i = 0; i < H; ++i) {
            const float i_gate = sigmoidf_stable(gates[static_cast<size_t>(i)]);
            const float f_gate = sigmoidf_stable(gates[static_cast<size_t>(H + i)]);
            const float g_gate = std::tanh(gates[static_cast<size_t>(2 * H + i)]);
            const float o_gate = sigmoidf_stable(gates[static_cast<size_t>(3 * H + i)]);

            const float c_new = clamp_cell(f_gate * c[static_cast<size_t>(i)] + i_gate * g_gate);
            c[static_cast<size_t>(i)] = c_new;
            h[static_cast<size_t>(i)] = o_gate * std::tanh(c_new);
            out_t[i] = h[static_cast<size_t>(i)];
        }
    }
}

// ---------------------------------------------------------------------------
// LSTM recurrence, stride-safe path
// ---------------------------------------------------------------------------

static void cpu_lstm_strided(
    const LstmCustomParams & p,
    const ggml_tensor * pre_gates,
    ggml_tensor * output) {

    const int64_t H  = p.hidden;
    const int64_t N  = p.n_steps;
    const int64_t H4 = 4 * H;

    std::vector<float> h(static_cast<size_t>(H), 0.0f);
    std::vector<float> c(static_cast<size_t>(H), 0.0f);
    std::vector<float> gates(static_cast<size_t>(H4));

    const float * w_base = p.w_hh_rowwise ? p.w_hh_rowwise : p.w_hh_f32;

    for (int64_t step = 0; step < N; ++step) {
        const int64_t t = p.reverse ? (N - 1 - step) : step;

        for (int64_t j = 0; j < H4; ++j) {
            const float pg_jt = tensor_get_f32_2d(pre_gates, j, t);
            gates[static_cast<size_t>(j)] =
                p.b_hh[j] + pg_jt + dot_product(w_base + j * H, h.data(), H);
        }

        for (int64_t i = 0; i < H; ++i) {
            const float i_gate = sigmoidf_stable(gates[static_cast<size_t>(i)]);
            const float f_gate = sigmoidf_stable(gates[static_cast<size_t>(H + i)]);
            const float g_gate = std::tanh(gates[static_cast<size_t>(2 * H + i)]);
            const float o_gate = sigmoidf_stable(gates[static_cast<size_t>(3 * H + i)]);

            const float c_new = clamp_cell(f_gate * c[static_cast<size_t>(i)] + i_gate * g_gate);
            c[static_cast<size_t>(i)] = c_new;
            h[static_cast<size_t>(i)] = o_gate * std::tanh(c_new);
            tensor_set_f32_2d(output, i, t, h[static_cast<size_t>(i)]);
        }
    }
}

static void cpu_lstm(
    const LstmCustomParams & p,
    const ggml_tensor * pre_gates,
    ggml_tensor * output) {

    const int64_t H  = p.hidden;
    const int64_t N  = p.n_steps;
    const int64_t H4 = 4 * H;

    const bool contiguous =
        tensor_is_f32_2d_contiguous(pre_gates, H4, N) &&
        tensor_is_f32_2d_contiguous(output, H, N);

    if (contiguous) {
        cpu_lstm_contiguous(
            p,
            static_cast<const float *>(pre_gates->data),
            static_cast<float *>(output->data));
    } else {
        cpu_lstm_strided(p, pre_gates, output);
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// ggml custom op callback
//
// Called from:
//   ggml_map_custom2_inplace(ctx, output, pre_input_gates_packed, ...)
//
// Convention for ggml_map_custom2_inplace:
//   dst        = output storage
//   a_unused   = same logical output/input storage
//   pre_gates  = W_ih*x + b_ih, shape [4*H, N]
// ---------------------------------------------------------------------------

void lstm_fused_callback(
    ggml_tensor       * dst,
    const ggml_tensor * /*a_unused*/,
    const ggml_tensor * pre_gates,
    int ith,
    int /*nth*/,
    void * userdata) {

    if (ith != 0) {
        return;
    }

    const LstmCustomParams * p = static_cast<const LstmCustomParams *>(userdata);
    GGML_ASSERT(p != nullptr);
    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(pre_gates != nullptr);

#ifdef KOKOPOP_HAS_METAL
    if (p->metal_kernel) {
        // Metal path currently expects contiguous host-visible buffers.
        // If this trips, either force ggml_cont() before the custom op or add
        // a staging copy for the Metal kernel.
        GGML_ASSERT(tensor_is_f32_2d_contiguous(pre_gates, 4 * p->hidden, p->n_steps));
        GGML_ASSERT(tensor_is_f32_2d_contiguous(dst, p->hidden, p->n_steps));

        metal_lstm_run(
            static_cast<MetalLstmKernelState *>(p->metal_kernel),
            p->whh_key,
            static_cast<const float *>(pre_gates->data),
            p->b_hh,
            static_cast<float *>(dst->data),
            static_cast<int>(p->hidden),
            static_cast<int>(p->n_steps),
            p->reverse);
        return;
    }
#endif

    cpu_lstm(*p, pre_gates, dst);
}

} // namespace kokopop
