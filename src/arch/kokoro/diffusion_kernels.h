#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace kokopop {
namespace diffusion_kernel {

inline float dot_scalar(const float * a, const float * b, int64_t len) {
    float acc = 0.0f;
    for (int64_t i = 0; i < len; ++i) acc += a[i] * b[i];
    return acc;
}

inline float sum_scalar(const float * x, int64_t len) {
    float acc = 0.0f;
    for (int64_t i = 0; i < len; ++i) acc += x[i];
    return acc;
}

inline float squared_diff_sum_scalar(const float * x, int64_t len, float mean) {
    float acc = 0.0f;
    for (int64_t i = 0; i < len; ++i) {
        const float d = x[i] - mean;
        acc += d * d;
    }
    return acc;
}

inline void add_inplace_scalar(float * dst, const float * src, size_t len) {
    for (size_t i = 0; i < len; ++i) dst[i] += src[i];
}

inline void scale_scalar(float * dst, const float * src, float scale, size_t len) {
    for (size_t i = 0; i < len; ++i) dst[i] = src[i] * scale;
}

inline void layer_norm_affine_scalar(
    float * y,
    const float * x,
    const float * gamma,
    const float * beta,
    int64_t dim,
    float mean,
    float inv_std) {
    for (int64_t i = 0; i < dim; ++i) {
        y[i] = ((x[i] - mean) * inv_std) * gamma[i] + beta[i];
    }
}

inline void ada_layer_norm_scalar(
    float * y,
    const float * x,
    const float * gamma,
    const float * beta,
    int64_t dim,
    float mean,
    float inv_std) {
    for (int64_t i = 0; i < dim; ++i) {
        const float xn = (x[i] - mean) * inv_std;
        y[i] = (1.0f + gamma[i]) * xn + beta[i];
    }
}

inline void attention_weighted_sum_scalar(
    float * out,
    const float * values,
    const float * probs,
    float inv_denom,
    int64_t rows,
    int64_t row_stride,
    int64_t head_dim) {
    for (int64_t d = 0; d < head_dim; ++d) out[d] = 0.0f;
    for (int64_t j = 0; j < rows; ++j) {
        const float p = probs[j] * inv_denom;
        const float * vj = values + j * row_stride;
        for (int64_t d = 0; d < head_dim; ++d) {
            out[d] += p * vj[d];
        }
    }
}

inline void denoise_combine_scalar(
    float * out,
    const float * x_noisy,
    const float * pred,
    float c_skip,
    float c_out,
    size_t len) {
    for (size_t i = 0; i < len; ++i) {
        out[i] = c_skip * x_noisy[i] + c_out * pred[i];
    }
}

inline void classifier_free_guidance_scalar(
    float * pred,
    const float * masked,
    float embedding_scale,
    size_t len) {
    for (size_t i = 0; i < len; ++i) {
        pred[i] = masked[i] + (pred[i] - masked[i]) * embedding_scale;
    }
}

inline void euler_midpoint_scalar(
    float * x_mid,
    const float * x,
    const float * denoised,
    float sigma,
    float sigma_mid,
    size_t len) {
    const float scale = (sigma_mid - sigma) / sigma;
    for (size_t i = 0; i < len; ++i) {
        x_mid[i] = x[i] + (x[i] - denoised[i]) * scale;
    }
}

inline void euler_update_with_noise_scalar(
    float * x,
    const float * x_mid,
    const float * denoised_mid,
    const float * noise,
    float sigma_mid,
    float sigma_down,
    float sigma,
    float sigma_up,
    size_t len) {
    const float scale = (sigma_down - sigma) / sigma_mid;
    for (size_t i = 0; i < len; ++i) {
        x[i] = x[i] + (x_mid[i] - denoised_mid[i]) * scale + noise[i] * sigma_up;
    }
}

inline void blend_style_scalar(
    float * style,
    const float * sampled,
    float alpha,
    float beta,
    size_t half_dim) {
    for (size_t i = 0; i < half_dim; ++i) {
        style[i] = (1.0f - alpha) * style[i] + alpha * sampled[i];
        style[i + half_dim] = (1.0f - beta) * style[i + half_dim] + beta * sampled[i + half_dim];
    }
}

#ifdef __AVX2__

inline float horizontal_sum(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

inline float dot_avx2(const float * a, const float * b, int64_t len) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 15 < len; i += 16) {
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
    }
    float acc = horizontal_sum(_mm256_add_ps(acc0, acc1));
    for (; i < len; ++i) acc += a[i] * b[i];
    return acc;
}

inline float sum_avx2(const float * x, int64_t len) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 15 < len; i += 16) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(x + i));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(x + i + 8));
    }
    float acc = horizontal_sum(_mm256_add_ps(acc0, acc1));
    for (; i < len; ++i) acc += x[i];
    return acc;
}

inline float squared_diff_sum_avx2(const float * x, int64_t len, float mean) {
    const __m256 vm = _mm256_set1_ps(mean);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 15 < len; i += 16) {
        const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(x + i), vm);
        const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(x + i + 8), vm);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
    }
    float acc = horizontal_sum(_mm256_add_ps(acc0, acc1));
    for (; i < len; ++i) {
        const float d = x[i] - mean;
        acc += d * d;
    }
    return acc;
}

#endif

#ifdef __ARM_NEON

inline float horizontal_sum(float32x4_t v) {
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
}

inline float dot_neon(const float * a, const float * b, int64_t len) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 15 < len; i += 16) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc2 = vmlaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        acc3 = vmlaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float acc = horizontal_sum(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < len; ++i) acc += a[i] * b[i];
    return acc;
}

inline float sum_neon(const float * x, int64_t len) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 15 < len; i += 16) {
        acc0 = vaddq_f32(acc0, vld1q_f32(x + i));
        acc1 = vaddq_f32(acc1, vld1q_f32(x + i + 4));
        acc2 = vaddq_f32(acc2, vld1q_f32(x + i + 8));
        acc3 = vaddq_f32(acc3, vld1q_f32(x + i + 12));
    }
    float acc = horizontal_sum(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < len; ++i) acc += x[i];
    return acc;
}

inline float squared_diff_sum_neon(const float * x, int64_t len, float mean) {
    const float32x4_t vm = vdupq_n_f32(mean);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 15 < len; i += 16) {
        const float32x4_t d0 = vsubq_f32(vld1q_f32(x + i), vm);
        const float32x4_t d1 = vsubq_f32(vld1q_f32(x + i + 4), vm);
        const float32x4_t d2 = vsubq_f32(vld1q_f32(x + i + 8), vm);
        const float32x4_t d3 = vsubq_f32(vld1q_f32(x + i + 12), vm);
        acc0 = vmlaq_f32(acc0, d0, d0);
        acc1 = vmlaq_f32(acc1, d1, d1);
        acc2 = vmlaq_f32(acc2, d2, d2);
        acc3 = vmlaq_f32(acc3, d3, d3);
    }
    float acc = horizontal_sum(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < len; ++i) {
        const float d = x[i] - mean;
        acc += d * d;
    }
    return acc;
}

#endif

inline float dot(const float * a, const float * b, int64_t len) {
#ifdef __AVX2__
    return dot_avx2(a, b, len);
#elif defined(__ARM_NEON)
    return dot_neon(a, b, len);
#else
    return dot_scalar(a, b, len);
#endif
}

inline float sum(const float * x, int64_t len) {
#ifdef __AVX2__
    return sum_avx2(x, len);
#elif defined(__ARM_NEON)
    return sum_neon(x, len);
#else
    return sum_scalar(x, len);
#endif
}

inline float squared_diff_sum(const float * x, int64_t len, float mean) {
#ifdef __AVX2__
    return squared_diff_sum_avx2(x, len, mean);
#elif defined(__ARM_NEON)
    return squared_diff_sum_neon(x, len, mean);
#else
    return squared_diff_sum_scalar(x, len, mean);
#endif
}

inline void add_inplace(float * dst, const float * src, size_t len) {
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 7 < len; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
    }
    for (; i < len; ++i) dst[i] += src[i];
#elif defined(__ARM_NEON)
    size_t i = 0;
    for (; i + 3 < len; i += 4) {
        vst1q_f32(dst + i, vaddq_f32(vld1q_f32(dst + i), vld1q_f32(src + i)));
    }
    for (; i < len; ++i) dst[i] += src[i];
#else
    add_inplace_scalar(dst, src, len);
#endif
}

inline void scale(float * dst, const float * src, float scale_value, size_t len) {
#if defined(__AVX2__)
    const __m256 vs = _mm256_set1_ps(scale_value);
    size_t i = 0;
    for (; i + 7 < len; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), vs));
    }
    for (; i < len; ++i) dst[i] = src[i] * scale_value;
#elif defined(__ARM_NEON)
    const float32x4_t vs = vdupq_n_f32(scale_value);
    size_t i = 0;
    for (; i + 3 < len; i += 4) {
        vst1q_f32(dst + i, vmulq_f32(vld1q_f32(src + i), vs));
    }
    for (; i < len; ++i) dst[i] = src[i] * scale_value;
#else
    scale_scalar(dst, src, scale_value, len);
#endif
}

inline void layer_norm_affine(
    float * y,
    const float * x,
    const float * gamma,
    const float * beta,
    int64_t dim,
    float mean,
    float inv_std) {
#if defined(__AVX2__)
    const __m256 vm = _mm256_set1_ps(mean);
    const __m256 vis = _mm256_set1_ps(inv_std);
    int64_t i = 0;
    for (; i + 7 < dim; i += 8) {
        const __m256 xn = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), vm), vis);
        const __m256 yg = _mm256_mul_ps(xn, _mm256_loadu_ps(gamma + i));
        _mm256_storeu_ps(y + i, _mm256_add_ps(yg, _mm256_loadu_ps(beta + i)));
    }
    for (; i < dim; ++i) y[i] = ((x[i] - mean) * inv_std) * gamma[i] + beta[i];
#elif defined(__ARM_NEON)
    const float32x4_t vm = vdupq_n_f32(mean);
    const float32x4_t vis = vdupq_n_f32(inv_std);
    int64_t i = 0;
    for (; i + 3 < dim; i += 4) {
        const float32x4_t xn = vmulq_f32(vsubq_f32(vld1q_f32(x + i), vm), vis);
        const float32x4_t yg = vmulq_f32(xn, vld1q_f32(gamma + i));
        vst1q_f32(y + i, vaddq_f32(yg, vld1q_f32(beta + i)));
    }
    for (; i < dim; ++i) y[i] = ((x[i] - mean) * inv_std) * gamma[i] + beta[i];
#else
    layer_norm_affine_scalar(y, x, gamma, beta, dim, mean, inv_std);
#endif
}

inline void ada_layer_norm(
    float * y,
    const float * x,
    const float * gamma,
    const float * beta,
    int64_t dim,
    float mean,
    float inv_std) {
#if defined(__AVX2__)
    const __m256 vm = _mm256_set1_ps(mean);
    const __m256 vis = _mm256_set1_ps(inv_std);
    const __m256 one = _mm256_set1_ps(1.0f);
    int64_t i = 0;
    for (; i + 7 < dim; i += 8) {
        const __m256 xn = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), vm), vis);
        const __m256 g = _mm256_add_ps(one, _mm256_loadu_ps(gamma + i));
        _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_mul_ps(g, xn), _mm256_loadu_ps(beta + i)));
    }
    for (; i < dim; ++i) {
        const float xn = (x[i] - mean) * inv_std;
        y[i] = (1.0f + gamma[i]) * xn + beta[i];
    }
#elif defined(__ARM_NEON)
    const float32x4_t vm = vdupq_n_f32(mean);
    const float32x4_t vis = vdupq_n_f32(inv_std);
    const float32x4_t one = vdupq_n_f32(1.0f);
    int64_t i = 0;
    for (; i + 3 < dim; i += 4) {
        const float32x4_t xn = vmulq_f32(vsubq_f32(vld1q_f32(x + i), vm), vis);
        const float32x4_t g = vaddq_f32(one, vld1q_f32(gamma + i));
        vst1q_f32(y + i, vaddq_f32(vmulq_f32(g, xn), vld1q_f32(beta + i)));
    }
    for (; i < dim; ++i) {
        const float xn = (x[i] - mean) * inv_std;
        y[i] = (1.0f + gamma[i]) * xn + beta[i];
    }
#else
    ada_layer_norm_scalar(y, x, gamma, beta, dim, mean, inv_std);
#endif
}

inline void attention_weighted_sum(
    float * out,
    const float * values,
    const float * probs,
    float inv_denom,
    int64_t rows,
    int64_t row_stride,
    int64_t head_dim) {
    std::memset(out, 0, static_cast<size_t>(head_dim) * sizeof(float));
    for (int64_t j = 0; j < rows; ++j) {
        const float p = probs[j] * inv_denom;
        const float * vj = values + j * row_stride;
#if defined(__AVX2__)
        const __m256 vp = _mm256_set1_ps(p);
        int64_t d = 0;
        for (; d + 7 < head_dim; d += 8) {
            const __m256 cur = _mm256_loadu_ps(out + d);
            const __m256 add = _mm256_mul_ps(vp, _mm256_loadu_ps(vj + d));
            _mm256_storeu_ps(out + d, _mm256_add_ps(cur, add));
        }
        for (; d < head_dim; ++d) out[d] += p * vj[d];
#elif defined(__ARM_NEON)
        const float32x4_t vp = vdupq_n_f32(p);
        int64_t d = 0;
        for (; d + 3 < head_dim; d += 4) {
            const float32x4_t cur = vld1q_f32(out + d);
            const float32x4_t add = vmulq_f32(vp, vld1q_f32(vj + d));
            vst1q_f32(out + d, vaddq_f32(cur, add));
        }
        for (; d < head_dim; ++d) out[d] += p * vj[d];
#else
        for (int64_t d = 0; d < head_dim; ++d) out[d] += p * vj[d];
#endif
    }
}

inline void denoise_combine(
    float * out,
    const float * x_noisy,
    const float * pred,
    float c_skip,
    float c_out,
    size_t len) {
#if defined(__AVX2__)
    const __m256 vcs = _mm256_set1_ps(c_skip);
    const __m256 vco = _mm256_set1_ps(c_out);
    size_t i = 0;
    for (; i + 7 < len; i += 8) {
        const __m256 a = _mm256_mul_ps(vcs, _mm256_loadu_ps(x_noisy + i));
        const __m256 b = _mm256_mul_ps(vco, _mm256_loadu_ps(pred + i));
        _mm256_storeu_ps(out + i, _mm256_add_ps(a, b));
    }
    for (; i < len; ++i) out[i] = c_skip * x_noisy[i] + c_out * pred[i];
#elif defined(__ARM_NEON)
    const float32x4_t vcs = vdupq_n_f32(c_skip);
    const float32x4_t vco = vdupq_n_f32(c_out);
    size_t i = 0;
    for (; i + 3 < len; i += 4) {
        const float32x4_t a = vmulq_f32(vcs, vld1q_f32(x_noisy + i));
        const float32x4_t b = vmulq_f32(vco, vld1q_f32(pred + i));
        vst1q_f32(out + i, vaddq_f32(a, b));
    }
    for (; i < len; ++i) out[i] = c_skip * x_noisy[i] + c_out * pred[i];
#else
    denoise_combine_scalar(out, x_noisy, pred, c_skip, c_out, len);
#endif
}

inline void classifier_free_guidance(
    float * pred,
    const float * masked,
    float embedding_scale,
    size_t len) {
#if defined(__AVX2__)
    const __m256 ves = _mm256_set1_ps(embedding_scale);
    size_t i = 0;
    for (; i + 7 < len; i += 8) {
        const __m256 m = _mm256_loadu_ps(masked + i);
        const __m256 p = _mm256_loadu_ps(pred + i);
        _mm256_storeu_ps(pred + i, _mm256_add_ps(m, _mm256_mul_ps(_mm256_sub_ps(p, m), ves)));
    }
    for (; i < len; ++i) pred[i] = masked[i] + (pred[i] - masked[i]) * embedding_scale;
#elif defined(__ARM_NEON)
    const float32x4_t ves = vdupq_n_f32(embedding_scale);
    size_t i = 0;
    for (; i + 3 < len; i += 4) {
        const float32x4_t m = vld1q_f32(masked + i);
        const float32x4_t p = vld1q_f32(pred + i);
        vst1q_f32(pred + i, vaddq_f32(m, vmulq_f32(vsubq_f32(p, m), ves)));
    }
    for (; i < len; ++i) pred[i] = masked[i] + (pred[i] - masked[i]) * embedding_scale;
#else
    classifier_free_guidance_scalar(pred, masked, embedding_scale, len);
#endif
}

inline void euler_midpoint(
    float * x_mid,
    const float * x,
    const float * denoised,
    float sigma,
    float sigma_mid,
    size_t len) {
#if defined(__AVX2__)
    const __m256 scale = _mm256_set1_ps((sigma_mid - sigma) / sigma);
    size_t i = 0;
    for (; i + 7 < len; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        const __m256 dv = _mm256_loadu_ps(denoised + i);
        _mm256_storeu_ps(x_mid + i, _mm256_add_ps(xv, _mm256_mul_ps(_mm256_sub_ps(xv, dv), scale)));
    }
    for (; i < len; ++i) x_mid[i] = x[i] + (x[i] - denoised[i]) * ((sigma_mid - sigma) / sigma);
#elif defined(__ARM_NEON)
    const float32x4_t scale = vdupq_n_f32((sigma_mid - sigma) / sigma);
    size_t i = 0;
    for (; i + 3 < len; i += 4) {
        const float32x4_t xv = vld1q_f32(x + i);
        const float32x4_t dv = vld1q_f32(denoised + i);
        vst1q_f32(x_mid + i, vaddq_f32(xv, vmulq_f32(vsubq_f32(xv, dv), scale)));
    }
    for (; i < len; ++i) x_mid[i] = x[i] + (x[i] - denoised[i]) * ((sigma_mid - sigma) / sigma);
#else
    euler_midpoint_scalar(x_mid, x, denoised, sigma, sigma_mid, len);
#endif
}

inline void euler_update_with_noise(
    float * x,
    const float * x_mid,
    const float * denoised_mid,
    const float * noise,
    float sigma_mid,
    float sigma_down,
    float sigma,
    float sigma_up,
    size_t len) {
#if defined(__AVX2__)
    const __m256 scale = _mm256_set1_ps((sigma_down - sigma) / sigma_mid);
    const __m256 noise_scale = _mm256_set1_ps(sigma_up);
    size_t i = 0;
    for (; i + 7 < len; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        const __m256 xm = _mm256_loadu_ps(x_mid + i);
        const __m256 dm = _mm256_loadu_ps(denoised_mid + i);
        const __m256 n = _mm256_loadu_ps(noise + i);
        const __m256 step = _mm256_mul_ps(_mm256_sub_ps(xm, dm), scale);
        const __m256 add_noise = _mm256_mul_ps(n, noise_scale);
        _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_add_ps(xv, step), add_noise));
    }
    for (; i < len; ++i) {
        x[i] = x[i] + (x_mid[i] - denoised_mid[i]) * ((sigma_down - sigma) / sigma_mid) + noise[i] * sigma_up;
    }
#elif defined(__ARM_NEON)
    const float32x4_t scale = vdupq_n_f32((sigma_down - sigma) / sigma_mid);
    const float32x4_t noise_scale = vdupq_n_f32(sigma_up);
    size_t i = 0;
    for (; i + 3 < len; i += 4) {
        const float32x4_t xv = vld1q_f32(x + i);
        const float32x4_t xm = vld1q_f32(x_mid + i);
        const float32x4_t dm = vld1q_f32(denoised_mid + i);
        const float32x4_t n = vld1q_f32(noise + i);
        const float32x4_t step = vmulq_f32(vsubq_f32(xm, dm), scale);
        const float32x4_t add_noise = vmulq_f32(n, noise_scale);
        vst1q_f32(x + i, vaddq_f32(vaddq_f32(xv, step), add_noise));
    }
    for (; i < len; ++i) {
        x[i] = x[i] + (x_mid[i] - denoised_mid[i]) * ((sigma_down - sigma) / sigma_mid) + noise[i] * sigma_up;
    }
#else
    euler_update_with_noise_scalar(x, x_mid, denoised_mid, noise, sigma_mid, sigma_down, sigma, sigma_up, len);
#endif
}

inline void blend_style(
    float * style,
    const float * sampled,
    float alpha,
    float beta,
    size_t half_dim) {
#if defined(__AVX2__)
    const __m256 va = _mm256_set1_ps(alpha);
    const __m256 via = _mm256_set1_ps(1.0f - alpha);
    const __m256 vb = _mm256_set1_ps(beta);
    const __m256 vib = _mm256_set1_ps(1.0f - beta);
    size_t i = 0;
    for (; i + 7 < half_dim; i += 8) {
        const __m256 s0 = _mm256_loadu_ps(style + i);
        const __m256 x0 = _mm256_loadu_ps(sampled + i);
        const __m256 s1 = _mm256_loadu_ps(style + half_dim + i);
        const __m256 x1 = _mm256_loadu_ps(sampled + half_dim + i);
        _mm256_storeu_ps(style + i, _mm256_add_ps(_mm256_mul_ps(via, s0), _mm256_mul_ps(va, x0)));
        _mm256_storeu_ps(style + half_dim + i, _mm256_add_ps(_mm256_mul_ps(vib, s1), _mm256_mul_ps(vb, x1)));
    }
    for (; i < half_dim; ++i) {
        style[i] = (1.0f - alpha) * style[i] + alpha * sampled[i];
        style[i + half_dim] = (1.0f - beta) * style[i + half_dim] + beta * sampled[i + half_dim];
    }
#elif defined(__ARM_NEON)
    const float32x4_t va = vdupq_n_f32(alpha);
    const float32x4_t via = vdupq_n_f32(1.0f - alpha);
    const float32x4_t vb = vdupq_n_f32(beta);
    const float32x4_t vib = vdupq_n_f32(1.0f - beta);
    size_t i = 0;
    for (; i + 3 < half_dim; i += 4) {
        const float32x4_t s0 = vld1q_f32(style + i);
        const float32x4_t x0 = vld1q_f32(sampled + i);
        const float32x4_t s1 = vld1q_f32(style + half_dim + i);
        const float32x4_t x1 = vld1q_f32(sampled + half_dim + i);
        vst1q_f32(style + i, vaddq_f32(vmulq_f32(via, s0), vmulq_f32(va, x0)));
        vst1q_f32(style + half_dim + i, vaddq_f32(vmulq_f32(vib, s1), vmulq_f32(vb, x1)));
    }
    for (; i < half_dim; ++i) {
        style[i] = (1.0f - alpha) * style[i] + alpha * sampled[i];
        style[i + half_dim] = (1.0f - beta) * style[i + half_dim] + beta * sampled[i + half_dim];
    }
#else
    blend_style_scalar(style, sampled, alpha, beta, half_dim);
#endif
}

} // namespace diffusion_kernel
} // namespace kokopop
