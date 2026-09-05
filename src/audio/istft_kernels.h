#pragma once

// The inner loops of the iSTFT, in a scalar reference form and in AVX2 / NEON.
//
// Same shape as src/arch/kokoro/diffusion_kernels.h: a `_scalar` version that
// states what the kernel computes, a `_avx2` and a `_neon` version, and a
// dispatcher resolved at compile time — every target here is one the compiler
// was already told to emit, so there is no runtime probing. The scalar versions
// stay reachable so the tests can hold the vector ones against them.
//
// Every kernel works on contiguous float arrays. Making the data contiguous is
// the caller's job: `gather_strided()` is the one exception, and it exists
// precisely so the rest of this file never has to think about strides.

#include <cstddef>
#include <cstdint>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace kokopop {
namespace istft_kernel {

// ---------------------------------------------------------------------------
// Scalar reference
// ---------------------------------------------------------------------------

/// Copies `n` values that sit `stride` apart into a contiguous buffer.
///
/// No vector version: a gather is a gather, and the loads are the cost. What it
/// buys is that a bin read once per frame is no longer read once per sample.
inline void gather_strided(float * dst, const float * src, ptrdiff_t stride,
                           size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[stride * static_cast<ptrdiff_t>(i)];
    }
}

/// dst[i] += src[i]
inline void add_inplace_scalar(float * dst, const float * src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

/// acc[i] += cr * c[i] - ci * s[i] — one bin's contribution to a whole frame.
inline void accumulate_rotation_scalar(float * acc, float cr, const float * c,
                                       float ci, const float * s, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        acc[i] += cr * c[i] - ci * s[i];
    }
}

/// dst[i] = src[i] * scale * window[i]
inline void window_scaled_scalar(float * dst, const float * src,
                                 const float * window, float scale, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (src[i] * scale) * window[i];
    }
}

/// Interleaves the two sample parities the real IFFT produces, windowing each
/// with its own half of the window: frame[2m] = zr[m]*we[m],
/// frame[2m+1] = zi[m]*wo[m].
inline void interleave_windowed_scalar(float * frame, const float * zr,
                                       const float * zi, const float * we,
                                       const float * wo, size_t half) {
    for (size_t m = 0; m < half; ++m) {
        frame[2 * m]     = zr[m] * we[m];
        frame[2 * m + 1] = zi[m] * wo[m];
    }
}

/// out[i] = env[i] > min_env ? ola[i] / env[i] : 0
inline void normalise_scalar(float * out, const float * ola, const float * env,
                             size_t n, float min_env) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = env[i] > min_env ? ola[i] / env[i] : 0.0f;
    }
}

/// Splits a Hermitian half-spectrum into the half-length complex spectrum
/// Z[k] = E[k] + i*O[k] whose inverse transform carries both sample parities.
///
///   E[k] = (X[k] + conj(X[N/2-k])) / 2
///   O[k] = e^{+2*pi*i*k/N} * (X[k] - conj(X[N/2-k])) / 2
///
/// `scale` replaces that 1/2, so the 1/m the inverse FFT owes its output can be
/// folded in here instead of costing a second pass: m is a power of two, so the
/// factor is exact and the result is bit-for-bit the same either way.
///
/// `re`/`im` must hold N/2 + 1 bins; `zr`/`zi` receive N/2 values.
inline void hermitian_split_scalar(const float * re, const float * im,
                                   const float * split_cos,
                                   const float * split_sin, float scale,
                                   float * zr, float * zi, size_t half) {
    for (size_t k = 0; k < half; ++k) {
        const float ar = re[k];
        const float ai = im[k];
        const float mr = re[half - k];
        const float mi = im[half - k];

        const float er = scale * (ar + mr);
        const float ei = scale * (ai - mi);
        const float dr = scale * (ar - mr);
        const float di = scale * (ai + mi);

        const float sc = split_cos[k];
        const float ss = split_sin[k];
        zr[k] = er - (dr * ss + di * sc);
        zi[k] = ei + (dr * sc - di * ss);
    }
}

/// The first radix-2 stage, where every twiddle is 1: a sum and a difference
/// over each adjacent pair.
inline void fft_stage_pairs_scalar(float * re, float * im, size_t m) {
    for (size_t i = 0; i < m; i += 2) {
        const float ar = re[i];
        const float ai = im[i];
        const float br = re[i + 1];
        const float bi = im[i + 1];
        re[i]     = ar + br;
        im[i]     = ai + bi;
        re[i + 1] = ar - br;
        im[i + 1] = ai - bi;
    }
}

/// One radix-2 inverse-FFT stage over `m` points, with `h` = len/2 butterflies
/// per block and `h` contiguous twiddles.
inline void fft_stage_scalar(float * re, float * im, const float * twc,
                             const float * tws, size_t m, size_t h) {
    for (size_t base = 0; base < m; base += 2 * h) {
        for (size_t k = 0; k < h; ++k) {
            const size_t a = base + k;
            const size_t b = a + h;
            const float wc = twc[k];
            const float ws = tws[k];
            const float br = re[b] * wc - im[b] * ws;
            const float bi = re[b] * ws + im[b] * wc;
            re[b] = re[a] - br;
            im[b] = im[a] - bi;
            re[a] = re[a] + br;
            im[a] = im[a] + bi;
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2
// ---------------------------------------------------------------------------

#ifdef __AVX2__

// -mavx2 does not imply -mfma, and the three-operand forms below are the only
// place that distinction matters. Naming them once keeps every kernel readable
// on both.
#ifdef __FMA__
inline __m256 mul_add_avx2(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }
inline __m256 mul_sub_avx2(__m256 a, __m256 b, __m256 c) { return _mm256_fmsub_ps(a, b, c); }
inline __m256 nmul_add_avx2(__m256 a, __m256 b, __m256 c) { return _mm256_fnmadd_ps(a, b, c); }
#else
inline __m256 mul_add_avx2(__m256 a, __m256 b, __m256 c) { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
inline __m256 mul_sub_avx2(__m256 a, __m256 b, __m256 c) { return _mm256_sub_ps(_mm256_mul_ps(a, b), c); }
inline __m256 nmul_add_avx2(__m256 a, __m256 b, __m256 c) { return _mm256_sub_ps(c, _mm256_mul_ps(a, b)); }
#endif

/// [a0..a7] -> [a7..a0]
inline __m256 reverse_avx2(__m256 v) {
    return _mm256_permutevar8x32_ps(v, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
}

/// Splits 16 interleaved floats into their even and odd lanes.
inline void deinterleave_avx2(const float * p, __m256 & even, __m256 & odd) {
    const __m256i idx = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    const __m256 v0 = _mm256_loadu_ps(p);
    const __m256 v1 = _mm256_loadu_ps(p + 8);
    // _mm256_shuffle_ps works inside each 128-bit lane, so the halves come out
    // shuffled across lanes and the permute puts them back in order.
    even = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(2, 0, 2, 0)), idx);
    odd  = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(3, 1, 3, 1)), idx);
}

/// The inverse of deinterleave_avx2.
inline void interleave_store_avx2(float * p, __m256 even, __m256 odd) {
    const __m256 lo = _mm256_unpacklo_ps(even, odd);
    const __m256 hi = _mm256_unpackhi_ps(even, odd);
    _mm256_storeu_ps(p,     _mm256_permute2f128_ps(lo, hi, 0x20));
    _mm256_storeu_ps(p + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
}

inline void add_inplace_avx2(float * dst, const float * src, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i),
                                                _mm256_loadu_ps(src + i)));
    }
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}

inline void accumulate_rotation_avx2(float * acc, float cr, const float * c,
                                     float ci, const float * s, size_t n) {
    const __m256 vcr = _mm256_set1_ps(cr);
    const __m256 vci = _mm256_set1_ps(ci);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 t = nmul_add_avx2(vci, _mm256_loadu_ps(s + i),
                                       _mm256_mul_ps(vcr, _mm256_loadu_ps(c + i)));
        _mm256_storeu_ps(acc + i, _mm256_add_ps(_mm256_loadu_ps(acc + i), t));
    }
    for (; i < n; ++i) {
        acc[i] += cr * c[i] - ci * s[i];
    }
}

inline void window_scaled_avx2(float * dst, const float * src,
                               const float * window, float scale, size_t n) {
    const __m256 vs = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(src + i), vs);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(v, _mm256_loadu_ps(window + i)));
    }
    for (; i < n; ++i) {
        dst[i] = (src[i] * scale) * window[i];
    }
}

inline void interleave_windowed_avx2(float * frame, const float * zr,
                                     const float * zi, const float * we,
                                     const float * wo, size_t half) {
    size_t m = 0;
    for (; m + 8 <= half; m += 8) {
        const __m256 even = _mm256_mul_ps(_mm256_loadu_ps(zr + m), _mm256_loadu_ps(we + m));
        const __m256 odd  = _mm256_mul_ps(_mm256_loadu_ps(zi + m), _mm256_loadu_ps(wo + m));
        interleave_store_avx2(frame + 2 * m, even, odd);
    }
    for (; m < half; ++m) {
        frame[2 * m]     = zr[m] * we[m];
        frame[2 * m + 1] = zi[m] * wo[m];
    }
}

inline void normalise_avx2(float * out, const float * ola, const float * env,
                           size_t n, float min_env) {
    const __m256 vmin = _mm256_set1_ps(min_env);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 e = _mm256_loadu_ps(env + i);
        // The divide runs on every lane, including the ones below the
        // threshold; the blend then throws their result away.
        const __m256 q = _mm256_div_ps(_mm256_loadu_ps(ola + i), e);
        const __m256 keep = _mm256_cmp_ps(e, vmin, _CMP_GT_OQ);
        _mm256_storeu_ps(out + i, _mm256_and_ps(q, keep));
    }
    for (; i < n; ++i) {
        out[i] = env[i] > min_env ? ola[i] / env[i] : 0.0f;
    }
}

inline void hermitian_split_avx2(const float * re, const float * im,
                                 const float * split_cos,
                                 const float * split_sin, float scale,
                                 float * zr, float * zi, size_t half) {
    const __m256 vs = _mm256_set1_ps(scale);
    size_t k = 0;
    // The mirror index runs backwards, so its load is reversed. It stays in
    // bounds as long as a full vector of mirrors fits below `half`.
    for (; k + 8 <= half; k += 8) {
        const __m256 ar = _mm256_loadu_ps(re + k);
        const __m256 ai = _mm256_loadu_ps(im + k);
        const __m256 mr = reverse_avx2(_mm256_loadu_ps(re + half - k - 7));
        const __m256 mi = reverse_avx2(_mm256_loadu_ps(im + half - k - 7));

        const __m256 er = _mm256_mul_ps(vs, _mm256_add_ps(ar, mr));
        const __m256 ei = _mm256_mul_ps(vs, _mm256_sub_ps(ai, mi));
        const __m256 dr = _mm256_mul_ps(vs, _mm256_sub_ps(ar, mr));
        const __m256 di = _mm256_mul_ps(vs, _mm256_add_ps(ai, mi));

        const __m256 sc = _mm256_loadu_ps(split_cos + k);
        const __m256 ss = _mm256_loadu_ps(split_sin + k);
        const __m256 oi = mul_add_avx2(dr, ss, _mm256_mul_ps(di, sc));
        const __m256 orr = mul_sub_avx2(dr, sc, _mm256_mul_ps(di, ss));
        _mm256_storeu_ps(zr + k, _mm256_sub_ps(er, oi));
        _mm256_storeu_ps(zi + k, _mm256_add_ps(ei, orr));
    }
    for (; k < half; ++k) {
        const float ar = re[k];
        const float ai = im[k];
        const float mr = re[half - k];
        const float mi = im[half - k];
        const float er = scale * (ar + mr);
        const float ei = scale * (ai - mi);
        const float dr = scale * (ar - mr);
        const float di = scale * (ai + mi);
        zr[k] = er - (dr * split_sin[k] + di * split_cos[k]);
        zi[k] = ei + (dr * split_cos[k] - di * split_sin[k]);
    }
}

inline void fft_stage_pairs_avx2(float * re, float * im, size_t m) {
    size_t i = 0;
    for (; i + 16 <= m; i += 16) {
        __m256 ar, br, ai, bi;
        deinterleave_avx2(re + i, ar, br);
        deinterleave_avx2(im + i, ai, bi);
        interleave_store_avx2(re + i, _mm256_add_ps(ar, br), _mm256_sub_ps(ar, br));
        interleave_store_avx2(im + i, _mm256_add_ps(ai, bi), _mm256_sub_ps(ai, bi));
    }
    for (; i < m; i += 2) {
        const float ar = re[i];
        const float ai = im[i];
        const float br = re[i + 1];
        const float bi = im[i + 1];
        re[i]     = ar + br;
        im[i]     = ai + bi;
        re[i + 1] = ar - br;
        im[i + 1] = ai - bi;
    }
}

inline void fft_stage_avx2(float * re, float * im, const float * twc,
                           const float * tws, size_t m, size_t h) {
    if (h < 8) {
        fft_stage_scalar(re, im, twc, tws, m, h);
        return;
    }
    for (size_t base = 0; base < m; base += 2 * h) {
        for (size_t k = 0; k < h; k += 8) {
            const size_t a = base + k;
            const size_t b = a + h;
            const __m256 wc = _mm256_loadu_ps(twc + k);
            const __m256 ws = _mm256_loadu_ps(tws + k);
            const __m256 rb = _mm256_loadu_ps(re + b);
            const __m256 ib = _mm256_loadu_ps(im + b);
            const __m256 ra = _mm256_loadu_ps(re + a);
            const __m256 ia = _mm256_loadu_ps(im + a);

            const __m256 br = mul_sub_avx2(rb, wc, _mm256_mul_ps(ib, ws));
            const __m256 bi = mul_add_avx2(rb, ws, _mm256_mul_ps(ib, wc));

            _mm256_storeu_ps(re + b, _mm256_sub_ps(ra, br));
            _mm256_storeu_ps(im + b, _mm256_sub_ps(ia, bi));
            _mm256_storeu_ps(re + a, _mm256_add_ps(ra, br));
            _mm256_storeu_ps(im + a, _mm256_add_ps(ia, bi));
        }
    }
}

#endif // __AVX2__

// ---------------------------------------------------------------------------
// NEON
// ---------------------------------------------------------------------------

#ifdef __ARM_NEON

/// [a0 a1 a2 a3] -> [a3 a2 a1 a0]
inline float32x4_t reverse_neon(float32x4_t v) {
    const float32x4_t swapped = vrev64q_f32(v);
    return vextq_f32(swapped, swapped, 2);
}

inline void add_inplace_neon(float * dst, const float * src, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(dst + i, vaddq_f32(vld1q_f32(dst + i), vld1q_f32(src + i)));
    }
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}

inline void accumulate_rotation_neon(float * acc, float cr, const float * c,
                                     float ci, const float * s, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t t = vmulq_n_f32(vld1q_f32(c + i), cr);
        t = vmlsq_n_f32(t, vld1q_f32(s + i), ci);
        vst1q_f32(acc + i, vaddq_f32(vld1q_f32(acc + i), t));
    }
    for (; i < n; ++i) {
        acc[i] += cr * c[i] - ci * s[i];
    }
}

inline void window_scaled_neon(float * dst, const float * src,
                               const float * window, float scale, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t v = vmulq_n_f32(vld1q_f32(src + i), scale);
        vst1q_f32(dst + i, vmulq_f32(v, vld1q_f32(window + i)));
    }
    for (; i < n; ++i) {
        dst[i] = (src[i] * scale) * window[i];
    }
}

inline void interleave_windowed_neon(float * frame, const float * zr,
                                     const float * zi, const float * we,
                                     const float * wo, size_t half) {
    size_t m = 0;
    for (; m + 4 <= half; m += 4) {
        float32x4x2_t pair;
        pair.val[0] = vmulq_f32(vld1q_f32(zr + m), vld1q_f32(we + m));
        pair.val[1] = vmulq_f32(vld1q_f32(zi + m), vld1q_f32(wo + m));
        vst2q_f32(frame + 2 * m, pair);
    }
    for (; m < half; ++m) {
        frame[2 * m]     = zr[m] * we[m];
        frame[2 * m + 1] = zi[m] * wo[m];
    }
}

inline void normalise_neon(float * out, const float * ola, const float * env,
                           size_t n, float min_env) {
    const float32x4_t vmin = vdupq_n_f32(min_env);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t e = vld1q_f32(env + i);
        // The divide runs on every lane, including the ones below the
        // threshold; the select then throws their result away.
        const float32x4_t q = vdivq_f32(vld1q_f32(ola + i), e);
        vst1q_f32(out + i, vbslq_f32(vcgtq_f32(e, vmin), q, zero));
    }
    for (; i < n; ++i) {
        out[i] = env[i] > min_env ? ola[i] / env[i] : 0.0f;
    }
}

inline void hermitian_split_neon(const float * re, const float * im,
                                 const float * split_cos,
                                 const float * split_sin, float scale,
                                 float * zr, float * zi, size_t half) {
    size_t k = 0;
    // The mirror index runs backwards, so its load is reversed. It stays in
    // bounds as long as a full vector of mirrors fits below `half`.
    for (; k + 4 <= half; k += 4) {
        const float32x4_t ar = vld1q_f32(re + k);
        const float32x4_t ai = vld1q_f32(im + k);
        const float32x4_t mr = reverse_neon(vld1q_f32(re + half - k - 3));
        const float32x4_t mi = reverse_neon(vld1q_f32(im + half - k - 3));

        const float32x4_t er = vmulq_n_f32(vaddq_f32(ar, mr), scale);
        const float32x4_t ei = vmulq_n_f32(vsubq_f32(ai, mi), scale);
        const float32x4_t dr = vmulq_n_f32(vsubq_f32(ar, mr), scale);
        const float32x4_t di = vmulq_n_f32(vaddq_f32(ai, mi), scale);

        const float32x4_t sc = vld1q_f32(split_cos + k);
        const float32x4_t ss = vld1q_f32(split_sin + k);
        const float32x4_t oi  = vmlaq_f32(vmulq_f32(di, sc), dr, ss);
        const float32x4_t orr = vmlsq_f32(vmulq_f32(dr, sc), di, ss);
        vst1q_f32(zr + k, vsubq_f32(er, oi));
        vst1q_f32(zi + k, vaddq_f32(ei, orr));
    }
    for (; k < half; ++k) {
        const float ar = re[k];
        const float ai = im[k];
        const float mr = re[half - k];
        const float mi = im[half - k];
        const float er = scale * (ar + mr);
        const float ei = scale * (ai - mi);
        const float dr = scale * (ar - mr);
        const float di = scale * (ai + mi);
        zr[k] = er - (dr * split_sin[k] + di * split_cos[k]);
        zi[k] = ei + (dr * split_cos[k] - di * split_sin[k]);
    }
}

inline void fft_stage_pairs_neon(float * re, float * im, size_t m) {
    size_t i = 0;
    for (; i + 8 <= m; i += 8) {
        const float32x4x2_t a = vld2q_f32(re + i);
        const float32x4x2_t b = vld2q_f32(im + i);
        float32x4x2_t out_re;
        float32x4x2_t out_im;
        out_re.val[0] = vaddq_f32(a.val[0], a.val[1]);
        out_re.val[1] = vsubq_f32(a.val[0], a.val[1]);
        out_im.val[0] = vaddq_f32(b.val[0], b.val[1]);
        out_im.val[1] = vsubq_f32(b.val[0], b.val[1]);
        vst2q_f32(re + i, out_re);
        vst2q_f32(im + i, out_im);
    }
    for (; i < m; i += 2) {
        const float ar = re[i];
        const float ai = im[i];
        const float br = re[i + 1];
        const float bi = im[i + 1];
        re[i]     = ar + br;
        im[i]     = ai + bi;
        re[i + 1] = ar - br;
        im[i + 1] = ai - bi;
    }
}

inline void fft_stage_neon(float * re, float * im, const float * twc,
                           const float * tws, size_t m, size_t h) {
    if (h < 4) {
        fft_stage_scalar(re, im, twc, tws, m, h);
        return;
    }
    for (size_t base = 0; base < m; base += 2 * h) {
        for (size_t k = 0; k < h; k += 4) {
            const size_t a = base + k;
            const size_t b = a + h;
            const float32x4_t wc = vld1q_f32(twc + k);
            const float32x4_t ws = vld1q_f32(tws + k);
            const float32x4_t rb = vld1q_f32(re + b);
            const float32x4_t ib = vld1q_f32(im + b);
            const float32x4_t ra = vld1q_f32(re + a);
            const float32x4_t ia = vld1q_f32(im + a);

            const float32x4_t br = vmlsq_f32(vmulq_f32(rb, wc), ib, ws);
            const float32x4_t bi = vmlaq_f32(vmulq_f32(rb, ws), ib, wc);

            vst1q_f32(re + b, vsubq_f32(ra, br));
            vst1q_f32(im + b, vsubq_f32(ia, bi));
            vst1q_f32(re + a, vaddq_f32(ra, br));
            vst1q_f32(im + a, vaddq_f32(ia, bi));
        }
    }
}

#endif // __ARM_NEON

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

inline void add_inplace(float * dst, const float * src, size_t n) {
#if defined(__AVX2__)
    add_inplace_avx2(dst, src, n);
#elif defined(__ARM_NEON)
    add_inplace_neon(dst, src, n);
#else
    add_inplace_scalar(dst, src, n);
#endif
}

inline void accumulate_rotation(float * acc, float cr, const float * c,
                                float ci, const float * s, size_t n) {
#if defined(__AVX2__)
    accumulate_rotation_avx2(acc, cr, c, ci, s, n);
#elif defined(__ARM_NEON)
    accumulate_rotation_neon(acc, cr, c, ci, s, n);
#else
    accumulate_rotation_scalar(acc, cr, c, ci, s, n);
#endif
}

inline void window_scaled(float * dst, const float * src, const float * window,
                          float scale, size_t n) {
#if defined(__AVX2__)
    window_scaled_avx2(dst, src, window, scale, n);
#elif defined(__ARM_NEON)
    window_scaled_neon(dst, src, window, scale, n);
#else
    window_scaled_scalar(dst, src, window, scale, n);
#endif
}

inline void interleave_windowed(float * frame, const float * zr, const float * zi,
                                const float * we, const float * wo, size_t half) {
#if defined(__AVX2__)
    interleave_windowed_avx2(frame, zr, zi, we, wo, half);
#elif defined(__ARM_NEON)
    interleave_windowed_neon(frame, zr, zi, we, wo, half);
#else
    interleave_windowed_scalar(frame, zr, zi, we, wo, half);
#endif
}

inline void normalise(float * out, const float * ola, const float * env,
                      size_t n, float min_env) {
#if defined(__AVX2__)
    normalise_avx2(out, ola, env, n, min_env);
#elif defined(__ARM_NEON)
    normalise_neon(out, ola, env, n, min_env);
#else
    normalise_scalar(out, ola, env, n, min_env);
#endif
}

inline void hermitian_split(const float * re, const float * im,
                            const float * split_cos, const float * split_sin,
                            float scale, float * zr, float * zi, size_t half) {
#if defined(__AVX2__)
    hermitian_split_avx2(re, im, split_cos, split_sin, scale, zr, zi, half);
#elif defined(__ARM_NEON)
    hermitian_split_neon(re, im, split_cos, split_sin, scale, zr, zi, half);
#else
    hermitian_split_scalar(re, im, split_cos, split_sin, scale, zr, zi, half);
#endif
}

inline void fft_stage_pairs(float * re, float * im, size_t m) {
#if defined(__AVX2__)
    fft_stage_pairs_avx2(re, im, m);
#elif defined(__ARM_NEON)
    fft_stage_pairs_neon(re, im, m);
#else
    fft_stage_pairs_scalar(re, im, m);
#endif
}

inline void fft_stage(float * re, float * im, const float * twc,
                      const float * tws, size_t m, size_t h) {
#if defined(__AVX2__)
    fft_stage_avx2(re, im, twc, tws, m, h);
#elif defined(__ARM_NEON)
    fft_stage_neon(re, im, twc, tws, m, h);
#else
    fft_stage_scalar(re, im, twc, tws, m, h);
#endif
}

} // namespace istft_kernel
} // namespace kokopop
