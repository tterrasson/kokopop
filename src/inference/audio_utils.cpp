#include "kokoro.h"

#include "core/constants.h"

#ifdef KOKOPOP_HAS_METAL
#include "backend/metal_stft.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <ggml.h>
#include <new>
#include <numeric>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace kokopop {

// ---------------------------------------------------------------------------
// Precomputed audio constants
// ---------------------------------------------------------------------------

const float * hann_window_20() {
    constexpr int N = KOKOPOP_STFT_N;
    // Lambda-based magic static: C++11 guarantees thread-safe one-time init.
    static const auto win = []() {
        std::array<float, N> w;
        for (int n = 0; n < N; ++n)
            w[n] = 0.5f - 0.5f * std::cos(2.0f * M_PI * n / static_cast<float>(N));
        return w;
    }();
    return win.data();
}

struct StftTwiddles {
    // c[n][k]: for a fixed n, all k bins are contiguous — enables auto-vectorisation
    // of the inner-k loop in both the forward DFT and the IDFT.
    static constexpr int K = KOKOPOP_STFT_N / 2 + 1;
    float c[KOKOPOP_STFT_N][K];
    float s[KOKOPOP_STFT_N][K];
    StftTwiddles() {
        for (int n = 0; n < KOKOPOP_STFT_N; ++n) {
            for (int k = 0; k < K; ++k) {
                const float a = 2.0f * M_PI * static_cast<float>(k * n) / static_cast<float>(KOKOPOP_STFT_N);
                c[n][k] = std::cos(a);
                s[n][k] = std::sin(a);
            }
        }
    }
};

static const StftTwiddles & stft_twiddles() {
    static StftTwiddles tw;
    return tw;
}

// ---------------------------------------------------------------------------
// Depthwise pool upsample (used by adain_resblk1d in graph_ops.cpp)
// ---------------------------------------------------------------------------

ggml_tensor * depthwise_pool_upsample(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    const std::string & prefix,
    std::string & error) {
    ggml_tensor * pool_w = require_tensor(model, (prefix + ".pool.weight").c_str(), error);
    ggml_tensor * pool_b = require_tensor(model, (prefix + ".pool.bias").c_str(), error);
    if (!error.empty()) {
        return nullptr;
    }

    const int64_t kernel = pool_w->ne[0];
    const int64_t channels = x->ne[1];
    if (pool_w->ne[1] != 1 || pool_w->ne[2] != channels || pool_b->ne[0] != channels) {
        error = "invalid depthwise upsample pool shape: " + prefix;
        return nullptr;
    }

    ggml_tensor * full_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel, channels, channels);
    std::vector<float> & full_w_data = model.depthwise_pool_kernels[prefix];
    if (full_w_data.empty()) {
        full_w_data.assign(static_cast<size_t>(ggml_nelements(full_w)), 0.0f);

        std::vector<float> src_data;
        if (!tensor_to_f32(*model.backend, pool_w, src_data)) {
            error = "failed to read depthwise upsample pool weights: " + prefix;
            return nullptr;
        }
        const float * src = src_data.data();
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t k = 0; k < kernel; ++k) {
                full_w_data[static_cast<size_t>(k + kernel * (c + channels * c))] =
                    src[static_cast<size_t>(k + kernel * c)];
            }
        }
    }
    model.backend->queue_tensor_data(full_w, full_w_data.data(), full_w_data.size() * sizeof(float));

    ggml_tensor * y = conv_transpose1d_crop(ctx, full_w, x, 2, 1, static_cast<int>(x->ne[0] * 2));
    return add_channel_bias(ctx, y, pool_b);
}

// ---------------------------------------------------------------------------
// CPU Harmonic STFT
// ---------------------------------------------------------------------------

static bool fill_harmonic_stft(
    Model & model,
    const std::vector<float> & f0,
    int64_t target_frames,
    std::vector<float> & har_data,
    std::string & error) {
    constexpr int harmonic = KOKOPOP_HARMONIC_COUNT;
    constexpr int upsample = KOKOPOP_UPSAMPLE;
    constexpr int sample_rate = KOKOPOP_SAMPLE_RATE;
    constexpr int n_fft = KOKOPOP_STFT_N;
    constexpr int hop = KOKOPOP_STFT_HOP;
    constexpr float sine_amp = KOKOPOP_SINE_AMP;

    if (target_frames < 0) {
        error = "invalid negative target_frames";
        return false;
    }

    const int64_t n_samples = static_cast<int64_t>(f0.size()) * upsample;

    // 7.1 — Reuse pre-allocated source buffer instead of allocating every call.
    // The buffer lives in Model and grows to accommodate the largest chunk.
    std::vector<float> & source = model.tmp_stft_source_f32;
    if (source.size() < static_cast<size_t>(n_samples)) {
        source.resize(static_cast<size_t>(n_samples));
    }

    float phase_sin[harmonic]{};
    float phase_cos[harmonic];
    std::fill(phase_cos, phase_cos + harmonic, 1.0f);
    if (!model.harmonic_source_loaded) {
        ggml_tensor * merge_w = require_tensor(model, "kokopop.decoder.generator.m_source.l_linear.weight", error);
        ggml_tensor * merge_b = require_tensor(model, "kokopop.decoder.generator.m_source.l_linear.bias", error);
        if (!error.empty()) {
            return false;
        }
        std::vector<float> merge_b_data;
        if (!tensor_to_f32(*model.backend, merge_w, model.harmonic_merge_w) ||
            !tensor_to_f32(*model.backend, merge_b, merge_b_data) ||
            merge_b_data.empty()) {
            error = "failed to read harmonic source merge weights";
            return false;
        }
        model.harmonic_merge_b = merge_b_data[0];
        model.harmonic_source_loaded = true;
    }
    if (model.harmonic_merge_w.size() < static_cast<size_t>(harmonic)) {
        error = "invalid harmonic source merge weight size";
        return false;
    }
    const float mb = model.harmonic_merge_b;
    const float * mw = model.harmonic_merge_w.data();
    const int64_t n_segs = static_cast<int64_t>(f0.size());
    for (int64_t seg = 0; seg < n_segs; ++seg) {
        const int64_t seg_start = seg * upsample;
        const int64_t seg_end   = std::min(seg_start + upsample, n_samples);
        const int64_t seg_len   = seg_end - seg_start;
        const float base_f0 = f0[static_cast<size_t>(seg)];

        if (base_f0 > 10.0f) {
            // Per-segment trigonometric deltas — computed once outside the j-loop;
            // the j-loop only does a vectorisable phase rotation.
            // Pad to 16 lanes (AVX2 width × 2) so loads past h=8 are well-defined zeros
            // and don't contribute to the merged sum.
            alignas(32) float sin_delta[16] = {};
            alignas(32) float cos_delta[16] = {};
            alignas(32) float mw_scaled[16] = {};
            for (int h = 0; h < harmonic; ++h) {
                const float delta = 2.0f * M_PI * base_f0 * static_cast<float>(h + 1) / static_cast<float>(sample_rate);
                sin_delta[h] = std::sin(delta);
                cos_delta[h] = std::cos(delta);
                mw_scaled[h] = mw[h] * sine_amp;
            }

#if defined(__ARM_NEON)
            // NEON 4-wide × 2 covers h=0..7; h=8 handled scalar.
            static_assert(KOKOPOP_HARMONIC_COUNT == 9,
                          "vectorised harmonic update assumes HARMONIC_COUNT == 9");
            const float32x4_t cd0 = vld1q_f32(cos_delta);
            const float32x4_t cd1 = vld1q_f32(cos_delta + 4);
            const float32x4_t sd0 = vld1q_f32(sin_delta);
            const float32x4_t sd1 = vld1q_f32(sin_delta + 4);
            const float cd8 = cos_delta[8];
            const float sd8 = sin_delta[8];
            const float mw8 = mw_scaled[8];

            // Vectorise the phase rotation only; keep the merged accumulation
            // scalar to preserve the exact left-to-right reduction order of
            // the original code (which the compiler lowers to a chain of
            // sequential fmadd's). Mixing SIMD horizontal-sum into the
            // accumulator perturbs results enough to drop SNR below 60 dB on
            // long chunks, even when each lane is computed with FMA semantics
            // matching the scalar fallback.
            //
            // Use mul+sub/mul+add rather than vfma/vfms so that each NEON lane
            // does the same three-rounding sequence as the scalar fallback
            // would produce under -ffp-contract=off. Phase magnitude drift
            // therefore stays at the ULP level and resyncs to 1.0 every 16
            // segments via the existing renormalisation block.
            for (int64_t j = 0; j < seg_len; ++j) {
                float32x4_t ps0 = vld1q_f32(phase_sin);
                float32x4_t ps1 = vld1q_f32(phase_sin + 4);
                float32x4_t pc0 = vld1q_f32(phase_cos);
                float32x4_t pc1 = vld1q_f32(phase_cos + 4);

                // Match the scalar codegen exactly: clang lowers
                //   `s*cd + c*sd`  → `fmul c,sd; fmadd s,cd,t`  (c*sd rounded; s*cd fused)
                //   `c*cd - s*sd`  → `fnmul s,sd; fmadd c,cd,t` (-s*sd rounded; c*cd fused)
                // Mismatching this pattern accumulates ~1 ULP/step of phase
                // angle drift, which over a multi-thousand-step chunk degrades
                // audio SNR below 60 dB.
                float32x4_t ns0 = vfmaq_f32(vmulq_f32(pc0, sd0), ps0, cd0);
                float32x4_t ns1 = vfmaq_f32(vmulq_f32(pc1, sd1), ps1, cd1);
                float32x4_t nc0 = vfmaq_f32(vnegq_f32(vmulq_f32(ps0, sd0)), pc0, cd0);
                float32x4_t nc1 = vfmaq_f32(vnegq_f32(vmulq_f32(ps1, sd1)), pc1, cd1);

                vst1q_f32(phase_sin,     ns0);
                vst1q_f32(phase_sin + 4, ns1);
                vst1q_f32(phase_cos,     nc0);
                vst1q_f32(phase_cos + 4, nc1);

                // Scalar tail for h=8.
                const float ps8 = phase_sin[8], pc8 = phase_cos[8];
                const float ns8 = ps8 * cd8 + pc8 * sd8;
                const float nc8 = pc8 * cd8 - ps8 * sd8;
                phase_sin[8] = ns8;
                phase_cos[8] = nc8;

                // Sequential accumulation — same order as the scalar loop.
                float merged = mb;
                merged += mw_scaled[0] * vgetq_lane_f32(ns0, 0);
                merged += mw_scaled[1] * vgetq_lane_f32(ns0, 1);
                merged += mw_scaled[2] * vgetq_lane_f32(ns0, 2);
                merged += mw_scaled[3] * vgetq_lane_f32(ns0, 3);
                merged += mw_scaled[4] * vgetq_lane_f32(ns1, 0);
                merged += mw_scaled[5] * vgetq_lane_f32(ns1, 1);
                merged += mw_scaled[6] * vgetq_lane_f32(ns1, 2);
                merged += mw_scaled[7] * vgetq_lane_f32(ns1, 3);
                merged += mw8 * ns8;
                source[static_cast<size_t>(seg_start + j)] = std::tanh(merged);
            }
#elif defined(__AVX2__)
            static_assert(KOKOPOP_HARMONIC_COUNT == 9,
                          "vectorised harmonic update assumes HARMONIC_COUNT == 9");
            const __m256 cd_v = _mm256_load_ps(cos_delta);
            const __m256 sd_v = _mm256_load_ps(sin_delta);
            const __m256 mw_v = _mm256_load_ps(mw_scaled);
            const float cd8 = cos_delta[8];
            const float sd8 = sin_delta[8];
            const float mw8 = mw_scaled[8];

            alignas(32) float ps_buf[16] = {};
            alignas(32) float pc_buf[16] = {};
            for (int h = 0; h < harmonic; ++h) { ps_buf[h] = phase_sin[h]; pc_buf[h] = phase_cos[h]; }

            // Match the scalar FMA contraction pattern lane-for-lane (same as
            // NEON path):
            //   next_sin = (pc*sd rounded) + (ps*cd fused)
            //   next_cos = (-(ps*sd) rounded) + (pc*cd fused)
            // Mixing SIMD horizontal-sum into the merged accumulator breaks
            // bit-equivalence and degrades audio SNR over long chunks, so the
            // accumulation runs scalar over the SIMD lane outputs.
            const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
            for (int64_t j = 0; j < seg_len; ++j) {
                __m256 ps = _mm256_load_ps(ps_buf);
                __m256 pc = _mm256_load_ps(pc_buf);

                __m256 t_pcsd     = _mm256_mul_ps(pc, sd_v);
                __m256 ns         = _mm256_fmadd_ps(ps, cd_v, t_pcsd);
                __m256 neg_pssd   = _mm256_xor_ps(_mm256_mul_ps(ps, sd_v), sign_mask);
                __m256 nc         = _mm256_fmadd_ps(pc, cd_v, neg_pssd);

                _mm256_store_ps(ps_buf, ns);
                _mm256_store_ps(pc_buf, nc);

                // Scalar tail for h=8.
                const float ps8 = phase_sin[8], pc8 = phase_cos[8];
                const float ns8 = ps8 * cd8 + pc8 * sd8;
                const float nc8 = pc8 * cd8 - ps8 * sd8;
                phase_sin[8] = ns8;
                phase_cos[8] = nc8;

                // Sequential accumulation matches the scalar reduction order.
                float merged = mb;
                merged += mw_scaled[0] * ps_buf[0];
                merged += mw_scaled[1] * ps_buf[1];
                merged += mw_scaled[2] * ps_buf[2];
                merged += mw_scaled[3] * ps_buf[3];
                merged += mw_scaled[4] * ps_buf[4];
                merged += mw_scaled[5] * ps_buf[5];
                merged += mw_scaled[6] * ps_buf[6];
                merged += mw_scaled[7] * ps_buf[7];
                merged += mw8 * ns8;
                source[static_cast<size_t>(seg_start + j)] = std::tanh(merged);
            }
            // Copy back lanes 0..7 only; phase_sin[8] / phase_cos[8] are kept
            // up to date in place by the scalar tail above.
            for (int h = 0; h < 8; ++h) { phase_sin[h] = ps_buf[h]; phase_cos[h] = pc_buf[h]; }
#else
            for (int64_t j = 0; j < seg_len; ++j) {
                float merged = mb;
                for (int h = 0; h < harmonic; ++h) {
                    const float sin_val = phase_sin[h];
                    const float cos_val = phase_cos[h];
                    const float next_sin = sin_val * cos_delta[h] + cos_val * sin_delta[h];
                    const float next_cos = cos_val * cos_delta[h] - sin_val * sin_delta[h];
                    phase_sin[h] = next_sin;
                    phase_cos[h] = next_cos;
                    merged += mw_scaled[h] * next_sin;
                }
                source[static_cast<size_t>(seg_start + j)] = std::tanh(merged);
            }
#endif
            // 7.2 — Phase normalisation every K segments instead of every segment.
            // Full normalisation (sqrt) after every upsample-step is expensive.
            // Every 16 segments (~4800 samples at 300 upsamp.) keeps float drift
            // well below 1e-6 — inaudible for audio synthesis.
            constexpr int64_t phase_reset_interval = 16;
            if ((seg + 1) % phase_reset_interval == 0) {
                for (int h = 0; h < harmonic; ++h) {
                    const float norm =
                        std::sqrt(phase_sin[h] * phase_sin[h] + phase_cos[h] * phase_cos[h]);
                    if (norm > 0.0f) {
                        phase_sin[h] /= norm;
                        phase_cos[h] /= norm;
                    }
                }
            }
        } else {
            const float unvoiced = std::tanh(mb);
            for (int64_t j = 0; j < seg_len; ++j) {
                source[static_cast<size_t>(seg_start + j)] = unvoiced;
            }
        }
    }

    // 7.1 — Reuse pre-allocated harmonic STFT buffer.
    // The buffer lives in Model; the CPU path overwrites all useful elements; no zero-fill needed.
    const size_t har_size = static_cast<size_t>(22 * target_frames);
    if (har_data.size() < har_size) {
        har_data.resize(har_size);
    }

    // GPU path: dispatch the DFT kernel when the Metal STFT state is available.
#ifdef KOKOPOP_HAS_METAL
    if (model.backend != nullptr) {
        if (auto * stft = static_cast<MetalStftState *>(model.backend->metal_stft_kernel())) {
            metal_stft_compute(stft, source.data(), har_data.data(),
                            static_cast<int>(n_samples),
                            static_cast<int>(target_frames));
            return true;
        }
    }
#endif

    const float * window = hann_window_20();
    const StftTwiddles & tw = stft_twiddles();
    const int64_t center_pad = n_fft / 2;
    for (int64_t frame = 0; frame < target_frames; ++frame) {
        // Precompute windowed samples once per frame (shared across all k bins).
        float ws[n_fft]{};
        const int64_t first = frame * hop - center_pad;
        if (first >= 0 && first + n_fft <= n_samples) {
            for (int n = 0; n < n_fft; ++n)
                ws[n] = source[static_cast<size_t>(first + n)] * window[n];
        } else {
            for (int n = 0; n < n_fft; ++n) {
                int64_t src_i = first + n;

                // reflect padding: matches torch.stft(center=True) default behaviour
                if (src_i < 0) {
                    src_i = -src_i - 1;
                } else if (src_i >= n_samples) {
                    src_i = 2 * n_samples - src_i - 1;
                }

                if (src_i >= 0 && src_i < n_samples) {
                    ws[n] = source[static_cast<size_t>(src_i)] * window[n];
                }
            }
        }

        // Forward DFT: outer loop over n so tw.c[n][0..K] is contiguous per row,
        // letting the compiler vectorise the inner k-loop across all 11 bins at once.
        float real_acc[StftTwiddles::K]{};
        float imag_acc[StftTwiddles::K]{};
        for (int n = 0; n < n_fft; ++n) {
            const float w   = ws[n];
            const float * tc = tw.c[n];
            const float * ts = tw.s[n];
            for (int k = 0; k < StftTwiddles::K; ++k) {
                real_acc[k] += w * tc[k];
                imag_acc[k] -= w * ts[k];
            }
        }
        for (int k = 0; k < StftTwiddles::K; ++k) {
            har_data[static_cast<size_t>(k * target_frames + frame)] = std::sqrt(real_acc[k] * real_acc[k] + imag_acc[k] * imag_acc[k]);
            har_data[static_cast<size_t>((k + n_fft / 2 + 1) * target_frames + frame)] = std::atan2(imag_acc[k], real_acc[k]);
        }
    }

    return true;
}

CpuTensor cpu_harmonic_stft(Model & model, const std::vector<float> & f0, int64_t target_frames, std::string & error) {
    std::vector<float> & har_data = model.tmp_stft_har_f32;
    if (!fill_harmonic_stft(model, f0, target_frames, har_data, error)) {
        return {};
    }
    const size_t har_size = static_cast<size_t>(22 * target_frames);
    return CpuTensor{
        22,
        target_frames,
        std::vector<float>(har_data.begin(), har_data.begin() + static_cast<ptrdiff_t>(har_size)),
    };
}

// 7.3 — cpu_istft now accepts a Model& and a std::vector<float>& for the output buffer.
// The intermediate y[] and denom[] arrays are cached inside Model and reused across calls,
// eliminating two large allocations per chunk.
static bool cpu_istft_data(Model & model, const float * post_data, int64_t n_frames, std::vector<float> & out) {
    constexpr int n_fft = KOKOPOP_STFT_N;
    constexpr int hop = KOKOPOP_STFT_HOP;
    if (n_frames <= 0) {
        out.clear();
        return true;
    }
    const int64_t padded_len = n_fft + hop * (n_frames - 1);
    const int64_t center_pad = n_fft / 2;
    const int64_t out_len = std::max<int64_t>(0, padded_len - 2 * center_pad);
    if (out_len <= 0) {
        out.clear();
        return true;
    }

#ifdef KOKOPOP_HAS_METAL
    if (model.backend != nullptr) {
        if (auto * stft = static_cast<MetalStftState *>(model.backend->metal_stft_kernel())) {
            out.resize(static_cast<size_t>(out_len));
            metal_istft_compute(stft, post_data, out.data(),
                                static_cast<int>(n_frames),
                                static_cast<int>(out_len));
            return !out.empty();
        }
    }
#endif

    // 7.3 — Reuse pre-allocated accumulator and denominator buffers.
    std::vector<float> & y = model.tmp_istft_y_f32;
    std::vector<float> & denom = model.tmp_istft_denom_f32;
    const size_t pad_size = static_cast<size_t>(padded_len);
    if (y.size() < pad_size) {
        y.resize(pad_size);
    }
    if (denom.size() < pad_size) {
        denom.resize(pad_size);
    }
    std::fill(y.begin(), y.begin() + static_cast<ptrdiff_t>(pad_size), 0.0f);
    std::fill(denom.begin(), denom.begin() + static_cast<ptrdiff_t>(pad_size), 0.0f);
    const float * window = hann_window_20();
    const StftTwiddles & tw = stft_twiddles();
    const auto post_at = [post_data, n_frames](int64_t c, int64_t t) -> float {
        return post_data[static_cast<size_t>(c * n_frames + t)];
    };

    // Precompute window² once per thread — same for every frame.
    static thread_local float win_sq[n_fft];
    static thread_local bool  win_sq_init = false;
    if (!win_sq_init) {
        for (int n = 0; n < n_fft; ++n) win_sq[n] = window[n] * window[n];
        win_sq_init = true;
    }

    for (int64_t frame = 0; frame < n_frames; ++frame) {
        float real[n_fft / 2 + 1]{};
        float imag[n_fft / 2 + 1]{};
        for (int k = 0; k <= n_fft / 2; ++k) {
            const float mag = std::exp(std::clamp(post_at(k, frame), -20.0f, 8.0f));
            const float phase = std::sin(post_at(k + n_fft / 2 + 1, frame));
            const float sin_ph = std::sin(phase);
            const float cos_ph = std::cos(phase);
            real[k] = mag * cos_ph;
            imag[k] = mag * sin_ph;
        }

        for (int n = 0; n < n_fft; ++n) {
            // (n%2)==0 ? 1:-1  →  1 - 2*(n&1), no branch
            float sample = real[0] + real[n_fft / 2] * static_cast<float>(1 - 2 * (n & 1));
            const float * tc = tw.c[n];
            const float * ts = tw.s[n];
            for (int k = 1; k < n_fft / 2; ++k) {
                sample += 2.0f * (real[k] * tc[k] - imag[k] * ts[k]);
            }
            sample *= (1.0f / static_cast<float>(n_fft));
            const int64_t dst = frame * hop + n;
            y[static_cast<size_t>(dst)]     += sample * window[n];
            denom[static_cast<size_t>(dst)] += win_sq[n];
        }
    }
    out.resize(static_cast<size_t>(out_len));
    for (int64_t i = 0; i < out_len; ++i) {
        const int64_t src = i + center_pad;
        const float d = denom[static_cast<size_t>(src)];
        out[static_cast<size_t>(i)] = std::clamp(
            d > 1e-8f ? y[static_cast<size_t>(src)] / d : 0.0f,
            -1.0f, 1.0f);
    }
    return !out.empty();
}

bool cpu_istft(Model & model, const CpuTensor & post, std::vector<float> & out) {
    return cpu_istft_data(model, post.data.data(), post.length, out);
}

// ---------------------------------------------------------------------------
// Graph sizing helpers
// ---------------------------------------------------------------------------

size_t frontend_graph_size(int64_t n_tokens) {
    // ~248 nodes/token (3 bidir LSTMs in duration_encoder + 1 in predictor, ~62 nodes each)
    // plus ~540 fixed nodes (12 transformer layers + misc). 2x safety margin.
    const size_t tokens = static_cast<size_t>(std::max<int64_t>(1, n_tokens));
    return 1500 + tokens * 500;
}

size_t generation_graph_size(int64_t total_frames, int64_t n_tokens) {
    // LSTM is unrolled per step: ~62 nodes/frame (bidir predictor.shared)
    // + ~62 nodes/token (bidir text_encoder.lstm) + ~1000 fixed (resblocks, conv).
    // 2x safety margin.
    const size_t frames = static_cast<size_t>(std::max<int64_t>(1, total_frames));
    const size_t tokens = static_cast<size_t>(std::max<int64_t>(1, n_tokens));
    return 2000 + frames * 130 + tokens * 130;
}

size_t generator_graph_size(int64_t) {
    return 32768;
}

// ---------------------------------------------------------------------------
// Scratch context initialization
// ---------------------------------------------------------------------------

ggml_context * init_scratch_context(
    Model & model, ScratchArena & arena, size_t mem_size,
    bool no_alloc, const char * label, std::string & error) {
    (void)model;
    uint8_t * mem = nullptr;
    try {
        mem = arena.data(mem_size);
    } catch (const std::bad_alloc &) {
        error = std::string("failed to allocate ggml ") + label + " scratch memory";
        return nullptr;
    }

    ggml_init_params params{};
    params.mem_size   = mem_size;
    params.mem_buffer = mem;
    params.no_alloc   = no_alloc;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        error = std::string("failed to allocate ggml ") + label + " context";
        return nullptr;
    }
    return ctx;
}

// ---------------------------------------------------------------------------
// Generator graph
// ---------------------------------------------------------------------------

bool ggml_generator(
    Model & model,
    const CpuTensor & decoder,
    const std::vector<float> & f0,
    const std::vector<float> & style,
    std::vector<float> & audio,
    std::string & error) {

    const int64_t har_len = decoder.length * 60 + 1;

    std::vector<float> & har_data = model.tmp_stft_har_f32;
    if (!fill_harmonic_stft(model, f0, har_len, har_data, error)) {
        return false;
    }

    const size_t mem_size = model.backend->generator_context_bytes(decoder.length);
    model.backend->clear_pending_inits();

    ggml_context * ctx = init_scratch_context(model, model.generator_scratch, mem_size, true, "generator", error);
    if (ctx == nullptr) {
        return false;
    }

    // CRITICAL: keep a stable pointer to the graph input tensor.
    // Do not reuse this variable for intermediate nodes, otherwise tensor_set()
    // will upload decoder data into the wrong tensor.
    ggml_tensor * decoder_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, decoder.length, decoder.channels);
    ggml_tensor * style_t       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, static_cast<int64_t>(style.size()));
    ggml_tensor * har_t         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, har_len, 22);

    ggml_set_input(decoder_input);
    ggml_set_input(style_t);
    ggml_set_input(har_t);

    ggml_tensor * x = decoder_input;

    model.metal_vocoder_convt_params.clear();
    model.metal_vocoder_convt_params.reserve(4);

    struct StageParams {
        int kernel;
        int up_stride;
        int up_padding;
        int noise_stride;
        int noise_padding;
        int noise_kernel;
        const char * noise_prefix;
        const char * up_weight;
        const char * up_bias;
    };

    static constexpr StageParams stage_params[2] = {
        {
            7, 10, 5, 6, 3, 12,
            "kokopop.decoder.generator.noise_res.0",
            "kokopop.decoder.generator.ups.0.weight",
            "kokopop.decoder.generator.ups.0.bias",
        },
        {
            11, 6, 3, 1, 0, 1,
            "kokopop.decoder.generator.noise_res.1",
            "kokopop.decoder.generator.ups.1.weight",
            "kokopop.decoder.generator.ups.1.bias",
        },
    };

    static const char * noise_conv_weights[2][2] = {
        {"kokopop.decoder.generator.noise_convs.0.weight", "kokopop.decoder.generator.noise_convs.0.bias"},
        {"kokopop.decoder.generator.noise_convs.1.weight", "kokopop.decoder.generator.noise_convs.1.bias"},
    };

    ggml_tensor * noise_conv_w[2] = {nullptr, nullptr};
    ggml_tensor * noise_conv_b[2] = {nullptr, nullptr};

    ggml_tensor * stage_debug[2] = {nullptr, nullptr};
    ggml_tensor * dbg_noise_conv[2] = {nullptr, nullptr};
    ggml_tensor * dbg_noise_res[2]  = {nullptr, nullptr};
    ggml_tensor * dbg_ups[2]        = {nullptr, nullptr};
    ggml_tensor * dbg_pre_rb[2]     = {nullptr, nullptr};
    const bool post_stats = std::getenv("KOKOPOP_POST_STATS") != nullptr;
    for (int stage = 0; stage < 2; ++stage) {
        noise_conv_w[stage] = require_tensor(model, noise_conv_weights[stage][0], error);
        noise_conv_b[stage] = require_tensor(model, noise_conv_weights[stage][1], error);
        if (!error.empty()) {
            ggml_free(ctx);
            return false;
        }
    }

    for (int stage = 0; stage < 2; ++stage) {
        const StageParams & sp = stage_params[stage];

        x = ggml_leaky_relu(ctx, x, 0.1f, false);

        ggml_tensor * x_source = add_channel_bias(
            ctx,
            conv1d(ctx, noise_conv_w[stage], har_t, sp.noise_stride, sp.noise_padding, 1, sp.noise_kernel),
            noise_conv_b[stage]);
        if (post_stats) {
            x_source = ggml_cont(ctx, x_source);
            ggml_set_name(x_source, stage == 0 ? "dbg_noise_conv0" : "dbg_noise_conv1");
            ggml_set_output(x_source);
            dbg_noise_conv[stage] = x_source;
        }

        x_source = graph_generator_resblock(ctx, model, x_source, style_t, sp.noise_prefix, sp.kernel, error);
        if (x_source == nullptr) {
            ggml_free(ctx);
            return false;
        }
        if (post_stats) {
            x_source = ggml_cont(ctx, x_source);
            ggml_set_name(x_source, stage == 0 ? "dbg_noise_res0" : "dbg_noise_res1");
            ggml_set_output(x_source);
            dbg_noise_res[stage] = x_source;
        }

        ggml_tensor * up_w = require_tensor(model, sp.up_weight, error);
        ggml_tensor * up_b = require_tensor(model, sp.up_bias, error);
        if (!error.empty()) {
            ggml_free(ctx);
            return false;
        }

        const int64_t out_len = (x->ne[0] - 1) * sp.up_stride - 2 * sp.up_padding + up_w->ne[0];
        x = conv_transpose1d_crop_bias(ctx, model, up_w, x, up_b, sp.up_stride, sp.up_padding, static_cast<int>(out_len));
        if (post_stats) {
            x = ggml_cont(ctx, x);
            ggml_set_name(x, stage == 0 ? "dbg_ups0" : "dbg_ups1");
            ggml_set_output(x);
            dbg_ups[stage] = x;
        }

        if (stage == 1) {
            x = ggml_pad_reflect_1d(ctx, x, 1, 0);
        }

        x = ggml_add(ctx, x, x_source);
        if (post_stats) {
            x = ggml_cont(ctx, x);
            ggml_set_name(x, stage == 0 ? "dbg_pre_rb0" : "dbg_pre_rb1");
            ggml_set_output(x);
            dbg_pre_rb[stage] = x;
        }

        ggml_tensor * accum = nullptr;
        for (int k = 0; k < 3; ++k) {
            const std::string resblock_prefix = "kokopop.decoder.generator.resblocks." + std::to_string(stage * 3 + k);
            ggml_tensor * branch = graph_generator_resblock(ctx, model, x, style_t, resblock_prefix, KOKOPOP_RESBLOCK_KERNELS[k], error);
            if (branch == nullptr) {
                ggml_free(ctx);
                return false;
            }
            accum = accum == nullptr ? branch : ggml_add(ctx, accum, branch);
        }

        x = ggml_scale(ctx, accum, 1.0f / 3.0f);
        if (std::getenv("KOKOPOP_POST_STATS") != nullptr) {
            x = ggml_cont(ctx, x);
            ggml_set_name(x, stage == 0 ? "kokopop_generator_stage0" : "kokopop_generator_stage1");
            ggml_set_output(x);
            stage_debug[stage] = x;
        }
    }

    x = ggml_leaky_relu(ctx, x, 0.01f, false);
    ggml_tensor * pre_post = ggml_cont(ctx, x);

    ggml_tensor * post_w = require_tensor(model, "kokopop.decoder.generator.conv_post.weight", error);
    ggml_tensor * post_b = require_tensor(model, "kokopop.decoder.generator.conv_post.bias", error);
    if (!error.empty()) {
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * post = add_channel_bias(ctx, conv1d(ctx, post_w, x, 1, 3, 1, 7), post_b);
    post = ggml_cont(ctx, post);
    ggml_set_name(post, "kokopop_generator_post");
    ggml_set_output(post);
    if (std::getenv("KOKOPOP_POST_STATS") != nullptr) {
        ggml_set_name(pre_post, "kokopop_generator_pre_post");
        ggml_set_output(pre_post);
    }

    const bool gen_profile = std::getenv("KOKOPOP_GEN_PROFILE") != nullptr;

    int64_t t0g = gen_profile ? ggml_time_us() : 0;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, generator_graph_size(decoder.length), false);
    ggml_build_forward_expand(gf, post);
    if (post_stats) {
        for (int stage = 0; stage < 2; ++stage) {
            if (stage_debug[stage] != nullptr) ggml_build_forward_expand(gf, stage_debug[stage]);
            if (dbg_noise_conv[stage]) ggml_build_forward_expand(gf, dbg_noise_conv[stage]);
            if (dbg_noise_res[stage])  ggml_build_forward_expand(gf, dbg_noise_res[stage]);
            if (dbg_ups[stage])        ggml_build_forward_expand(gf, dbg_ups[stage]);
            if (dbg_pre_rb[stage])     ggml_build_forward_expand(gf, dbg_pre_rb[stage]);
        }
        ggml_build_forward_expand(gf, pre_post);
    }

    if (gen_profile) {
        std::fprintf(stderr, "[gen-profile] generator n_nodes=%d build=%.1fms\n", ggml_graph_n_nodes(gf), (ggml_time_us() - t0g) / 1000.0);
    }

    model.backend->set_active_label("generator");
    model.backend->sched_reset();

    t0g = gen_profile ? ggml_time_us() : 0;
    if (!model.backend->sched_alloc_graph(gf)) {
        ggml_free(ctx);
        error = "ggml generator backend allocation failed";
        return false;
    }

    if (gen_profile) {
        std::fprintf(stderr, "[gen-profile] generator sched_alloc=%.1fms\n", (ggml_time_us() - t0g) / 1000.0);
    }

    if (!model.backend->apply_pending_inits()) {
        ggml_free(ctx);
        error = "ggml generator backend tensor initialization failed";
        return false;
    }

    const size_t decoder_bytes = static_cast<size_t>(decoder.length * decoder.channels) * sizeof(float);
    model.backend->tensor_set(decoder_input, decoder.data.data(), 0, decoder_bytes);
    model.backend->tensor_set(style_t,       style.data(),        0, style.size() * sizeof(float));
    model.backend->tensor_set(har_t,         har_data.data(),     0, ggml_nbytes(har_t));

    t0g = gen_profile ? ggml_time_us() : 0;
    const ggml_status status = model.backend->compute(ctx, gf);

    if (gen_profile) {
        std::fprintf(stderr, "[gen-profile] generator compute=%.1fms\n", (ggml_time_us() - t0g) / 1000.0);
    }

    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        error = "ggml generator graph compute failed";
        return false;
    }

    const size_t post_size = static_cast<size_t>(ggml_nelements(post));
    std::vector<float> & post_data = model.tmp_post_f32;
    if (post_data.size() < post_size) {
        post_data.resize(post_size);
    }

    model.backend->tensor_get(post, post_data.data(), 0, post_size * sizeof(float));

    if (std::getenv("KOKOPOP_POST_STATS") != nullptr) {
        const char * dump_dir = std::getenv("KOKOPOP_DUMP_DIR");
        auto print_tensor_stats = [&](const char * label, ggml_tensor * t) {
            std::vector<float> data(static_cast<size_t>(ggml_nelements(t)));
            model.backend->tensor_get(t, data.data(), 0, data.size() * sizeof(float));
            if (dump_dir) {
                char path[512];
                std::snprintf(path, sizeof(path), "%s/cpp_%s.bin", dump_dir, label);
                if (FILE * f = std::fopen(path, "wb")) {
                    int32_t hdr[4] = {static_cast<int32_t>(t->ne[0]), static_cast<int32_t>(t->ne[1]),
                                      static_cast<int32_t>(t->ne[2]), static_cast<int32_t>(t->ne[3])};
                    std::fwrite(hdr, sizeof(hdr), 1, f);
                    std::fwrite(data.data(), sizeof(float), data.size(), f);
                    std::fclose(f);
                }
            }
            double sum = 0.0;
            double sum_sq = 0.0;
            float min_v = std::numeric_limits<float>::infinity();
            float max_v = -std::numeric_limits<float>::infinity();
            for (float v : data) {
                sum += v;
                sum_sq += static_cast<double>(v) * static_cast<double>(v);
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
            }
            std::fprintf(stderr,
                         "[kokopop][%s] frames=%lld channels=%lld mean=%.6f rms=%.6f min=%.6f max=%.6f\n",
                         label,
                         static_cast<long long>(t->ne[0]),
                         static_cast<long long>(t->ne[1]),
                         sum / static_cast<double>(std::max<size_t>(1, data.size())),
                         std::sqrt(sum_sq / static_cast<double>(std::max<size_t>(1, data.size()))),
                         min_v,
                         max_v);
        };
        for (int stage = 0; stage < 2; ++stage) {
            char buf[32];
            if (dbg_noise_conv[stage]) { std::snprintf(buf, sizeof(buf), "noise_conv%d", stage); print_tensor_stats(buf, dbg_noise_conv[stage]); }
            if (dbg_noise_res[stage])  { std::snprintf(buf, sizeof(buf), "noise_res%d",  stage); print_tensor_stats(buf, dbg_noise_res[stage]);  }
            if (dbg_ups[stage])        { std::snprintf(buf, sizeof(buf), "ups%d",        stage); print_tensor_stats(buf, dbg_ups[stage]);        }
            if (dbg_pre_rb[stage])     { std::snprintf(buf, sizeof(buf), "pre_rb%d",     stage); print_tensor_stats(buf, dbg_pre_rb[stage]);     }
            if (stage_debug[stage] != nullptr) {
                print_tensor_stats(stage == 0 ? "stage0" : "stage1", stage_debug[stage]);
            }
        }
        std::vector<float> pre_data(static_cast<size_t>(ggml_nelements(pre_post)));
        model.backend->tensor_get(pre_post, pre_data.data(), 0, pre_data.size() * sizeof(float));
        double pre_sum = 0.0;
        double pre_sum_sq = 0.0;
        float pre_min = std::numeric_limits<float>::infinity();
        float pre_max = -std::numeric_limits<float>::infinity();
        for (float v : pre_data) {
            pre_sum += v;
            pre_sum_sq += static_cast<double>(v) * static_cast<double>(v);
            pre_min = std::min(pre_min, v);
            pre_max = std::max(pre_max, v);
        }
        std::fprintf(stderr,
                     "[kokopop][pre_post] frames=%lld channels=%lld mean=%.6f rms=%.6f min=%.6f max=%.6f\n",
                     static_cast<long long>(pre_post->ne[0]),
                     static_cast<long long>(pre_post->ne[1]),
                     pre_sum / static_cast<double>(std::max<size_t>(1, pre_data.size())),
                     std::sqrt(pre_sum_sq / static_cast<double>(std::max<size_t>(1, pre_data.size()))),
                     pre_min,
                     pre_max);

        auto channel_stats = [&](int64_t c0, int64_t c1, double & mean, double & rms, float & min_v, float & max_v) {
            double sum = 0.0;
            double sum_sq = 0.0;
            size_t count = 0;
            min_v = std::numeric_limits<float>::infinity();
            max_v = -std::numeric_limits<float>::infinity();
            for (int64_t c = c0; c < c1; ++c) {
                for (int64_t t = 0; t < post->ne[0]; ++t) {
                    const float v = post_data[static_cast<size_t>(c * post->ne[0] + t)];
                    sum += v;
                    sum_sq += static_cast<double>(v) * static_cast<double>(v);
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                    ++count;
                }
            }
            mean = count > 0 ? sum / static_cast<double>(count) : 0.0;
            rms = count > 0 ? std::sqrt(sum_sq / static_cast<double>(count)) : 0.0;
        };
        double mean = 0.0, rms = 0.0, spec_mean = 0.0, spec_rms = 0.0, phase_mean = 0.0, phase_rms = 0.0;
        float min_v = 0.0f, max_v = 0.0f, spec_min = 0.0f, spec_max = 0.0f, phase_min = 0.0f, phase_max = 0.0f;
        channel_stats(0, 22, mean, rms, min_v, max_v);
        channel_stats(0, 11, spec_mean, spec_rms, spec_min, spec_max);
        channel_stats(11, 22, phase_mean, phase_rms, phase_min, phase_max);
        std::fprintf(stderr,
                     "[kokopop][post] frames=%lld values=%zu mean=%.6f rms=%.6f min=%.6f max=%.6f spec_mean=%.6f spec_rms=%.6f spec_min=%.6f spec_max=%.6f phase_mean=%.6f phase_rms=%.6f phase_min=%.6f phase_max=%.6f\n",
                     static_cast<long long>(post->ne[0]), post_size, mean, rms, min_v, max_v,
                     spec_mean, spec_rms, spec_min, spec_max,
                     phase_mean, phase_rms, phase_min, phase_max);
    }

    if (!cpu_istft_data(model, post_data.data(), post->ne[0], audio)) {
        ggml_free(ctx);
        error = "cpu_istft failed";
        return false;
    }

    ggml_free(ctx);
    return !audio.empty();
}

} // namespace kokopop
