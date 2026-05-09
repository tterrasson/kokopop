#include "lstm_fused.h"
#include <cmath>
#include <cstring>

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

// ---------------------------------------------------------------------------
// CPU tight-loop LSTM recurrence — SIMD-optimised.
//
// Replaces the per-timestep ggml graph loop: instead of ~18 ggml nodes per
// step (view, add, sigmoid, tanh, set_2d …), we compute the full recurrence
// in a single C++ function call.
//
// Layout (ggml column-major — ne[0] is the fast / contiguous dimension):
//   pre_gates       : float[4*H * N]   element (g, t) = pre_gates[g + 4*H*t]
//   w_hh_f32        : float[H * 4*H]   element (k, j) = w_hh_f32[k + H*j]
//   w_hh_rowwise    : float[4*H * H]   element (j, k) = w_hh_rowwise[j*H + k]
//   b_hh            : float[4*H]
//   output          : float[H * N]     element (h, t) = output[h + H*t]
//
// Optimisations applied:
//   1. Transposed w_hh layout (w_hh_rowwise) for contiguous access per gate
//   2. NEON / AVX2 vectorised dot product (FMA)
//   3. Loop unrolling of activation update over 4 gates for ILP
//   4. Scalar fallback for architectures without NEON/AVX2
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Dot-product helpers — one per ISA
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static float dot_product_avx2(const float * __restrict__ a,
                               const float * __restrict__ b,
                               int64_t len)
{
    // Keep the accumulation order identical to the scalar recurrence.
    //
    // The duration LSTMs are sensitive enough that a wide AVX2 reduction
    // tree/FMA contraction can audibly change predicted frame counts over
    // long-form chunks.  We still use AVX2 for the 8 products at a time, but
    // fold those products back into the scalar accumulator in k order.
    alignas(32) float products[8];
    float dot = 0.0f;
    int64_t i = 0;
    for (; i + 7 < len; i += 8) {
        const __m256 prod = _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        _mm256_store_ps(products, prod);
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

#elif defined(__ARM_NEON)

static float dot_product_neon(const float * __restrict__ a,
                               const float * __restrict__ b,
                               int64_t len)
{
    // NEON: 4-wide F32 vectors.
    // 4 accumulators → 16 elements per loop iteration.
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int64_t i = 0;
    for (; i + 15 < len; i += 16) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a + i +  4), vld1q_f32(b + i +  4));
        acc2 = vmlaq_f32(acc2, vld1q_f32(a + i +  8), vld1q_f32(b + i +  8));
        acc3 = vmlaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }

    // Merge 4 accumulators
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    // Horizontal sum
    float dot = vgetq_lane_f32(acc0, 0)
              + vgetq_lane_f32(acc0, 1)
              + vgetq_lane_f32(acc0, 2)
              + vgetq_lane_f32(acc0, 3);

    // Scalar tail
    for (; i < len; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

#else

// Scalar fallback (any architecture without NEON or AVX2)
static float dot_product_scalar(const float * a, const float * b, int64_t len)
{
    float dot = 0.0f;
    for (int64_t i = 0; i < len; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

#endif // __AVX2__ / __ARM_NEON

// ---------------------------------------------------------------------------
// Shared cpu_lstm body — delegates to the ISA-specific dot_product_* above.
// ---------------------------------------------------------------------------

static void cpu_lstm(
    const LstmCustomParams & p,
    const float * pre_gates,
    float       * output)
{
    const int64_t H  = p.hidden;
    const int64_t N  = p.n_steps;
    const int64_t H4 = 4 * H;

    GGML_ASSERT(H <= 256 && "cpu_lstm: hidden > 256 overflows stack buffers — increase array sizes");
    float h[256] = {};
    float c[256] = {};

    // Temporary gate accumulator for one timestep
    float gates[1024];

    // Prefer transposed (rowwise) layout: w_rowwise[j*H + k] is contiguous over k
    // for each gate j. Falls back to original column-major layout if not provided.
    const float * w_base = p.w_hh_rowwise ? p.w_hh_rowwise : p.w_hh_f32;

    for (int64_t step = 0; step < N; ++step) {
        const int64_t t = p.reverse ? (N - 1 - step) : step;

        // ---- Compute gates[j] = b_hh[j] + pre_gates[j,t] + dot(w[:,j], h) ----
        const float * pg_t = pre_gates + H4 * t;
        for (int64_t j = 0; j < H4; ++j) {
#ifdef __AVX2__
            gates[j] = p.b_hh[j] + pg_t[j] + dot_product_avx2(w_base + j * H, h, H);
#elif defined(__ARM_NEON)
            gates[j] = p.b_hh[j] + pg_t[j] + dot_product_neon(w_base + j * H, h, H);
#else
            gates[j] = p.b_hh[j] + pg_t[j] + dot_product_scalar(w_base + j * H, h, H);
#endif
        }

        // ---- Apply activations and update h, c ----
        float * out_t = output + H * t;
        for (int64_t i = 0; i < H; ++i) {
            const float i_gate = 1.0f / (1.0f + expf(-gates[i]));
            const float f_gate = 1.0f / (1.0f + expf(-gates[H   + i]));
            const float g_gate = tanhf(gates[2*H + i]);
            const float o_gate = 1.0f / (1.0f + expf(-gates[3*H + i]));

            float c_new = f_gate * c[i] + i_gate * g_gate;
            if (c_new >  50.0f) c_new =  50.0f;
            if (c_new < -50.0f) c_new = -50.0f;
            c[i] = c_new;
            h[i] = o_gate * tanhf(c[i]);
            out_t[i] = h[i];
        }
    }
}

// ---------------------------------------------------------------------------
// ggml custom op callback
// ---------------------------------------------------------------------------
void lstm_fused_callback(
    ggml_tensor       * dst,
    const ggml_tensor * /*a_unused*/,
    const ggml_tensor * pre_gates,
    int   ith,
    int   /*nth*/,
    void * userdata)
{
    if (ith != 0) {
        return;
    }

    const LstmCustomParams * p = static_cast<const LstmCustomParams *>(userdata);
    const float * pg  = static_cast<const float *>(pre_gates->data);
    float       * out = static_cast<float *>(dst->data);
#ifdef KOKOPOP_HAS_METAL
    if (p->metal_kernel) {
        metal_lstm_run(
            static_cast<MetalLstmKernelState *>(p->metal_kernel),
            p->whh_key,
            pg, p->b_hh, out,
            static_cast<int>(p->hidden),
            static_cast<int>(p->n_steps),
            p->reverse);
        return;
    }
#endif

    cpu_lstm(*p, pg, out);
}

} // namespace kokopop
