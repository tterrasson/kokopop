#include "kokoro.h"

#include "core/constants.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>

namespace kokopop {

// ---------------------------------------------------------------------------
// Precomputed audio constants
// ---------------------------------------------------------------------------

const float * hann_window_20() {
    constexpr int N = KOKOPOP_STFT_N;
    static float win[N];
    static bool  init = false;
    if (!init) {
        for (int n = 0; n < N; ++n) {
            win[n] = 0.5f - 0.5f * std::cos(2.0f * M_PI * n / static_cast<float>(N));
        }
        init = true;
    }
    return win;
}

struct StftTwiddles {
    float c[11][KOKOPOP_STFT_N];
    float s[11][KOKOPOP_STFT_N];
    StftTwiddles() {
        for (int k = 0; k <= 10; ++k) {
            for (int n = 0; n < KOKOPOP_STFT_N; ++n) {
                const float a = 2.0f * M_PI * static_cast<float>(k * n) / static_cast<float>(KOKOPOP_STFT_N);
                c[k][n] = std::cos(a);
                s[k][n] = std::sin(a);
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

CpuTensor cpu_harmonic_stft(Model & model, const std::vector<float> & f0, int64_t target_frames, std::string & error) {
    constexpr int harmonic = KOKOPOP_HARMONIC_COUNT;
    constexpr int upsample = KOKOPOP_UPSAMPLE;
    constexpr int sample_rate = KOKOPOP_SAMPLE_RATE;
    constexpr int n_fft = KOKOPOP_STFT_N;
    constexpr int hop = KOKOPOP_STFT_HOP;
    constexpr float sine_amp = KOKOPOP_SINE_AMP;

    const int64_t n_samples = static_cast<int64_t>(f0.size()) * upsample;

    // 7.1 — Reuse pre-allocated source buffer instead of allocating every call.
    // The buffer lives in Model and grows to accommodate the largest chunk.
    std::vector<float> & source = model.tmp_stft_source_f32;
    if (source.size() < static_cast<size_t>(n_samples)) {
        source.resize(static_cast<size_t>(n_samples));
    }
    std::fill(source.begin(), source.begin() + static_cast<ptrdiff_t>(n_samples), 0.0f);

    float phase_sin[harmonic]{};
    float phase_cos[harmonic];
    std::fill(phase_cos, phase_cos + harmonic, 1.0f);
    if (!model.harmonic_source_loaded) {
        ggml_tensor * merge_w = require_tensor(model, "kokopop.decoder.generator.m_source.l_linear.weight", error);
        ggml_tensor * merge_b = require_tensor(model, "kokopop.decoder.generator.m_source.l_linear.bias", error);
        if (!error.empty()) {
            return {};
        }
        std::vector<float> merge_b_data;
        if (!tensor_to_f32(*model.backend, merge_w, model.harmonic_merge_w) ||
            !tensor_to_f32(*model.backend, merge_b, merge_b_data) ||
            merge_b_data.empty()) {
            error = "failed to read harmonic source merge weights";
            return {};
        }
        model.harmonic_merge_b = merge_b_data[0];
        model.harmonic_source_loaded = true;
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
            float sin_delta[harmonic];
            float cos_delta[harmonic];
            for (int h = 0; h < harmonic; ++h) {
                const float delta = 2.0f * M_PI * base_f0 * static_cast<float>(h + 1) / static_cast<float>(sample_rate);
                sin_delta[h] = std::sin(delta);
                cos_delta[h] = std::cos(delta);
            }
            for (int64_t j = 0; j < seg_len; ++j) {
                float merged = mb;
                for (int h = 0; h < harmonic; ++h) {
                    const float sin_val = phase_sin[h];
                    const float cos_val = phase_cos[h];
                    const float next_sin = sin_val * cos_delta[h] + cos_val * sin_delta[h];
                    const float next_cos = cos_val * cos_delta[h] - sin_val * sin_delta[h];
                    phase_sin[h] = next_sin;
                    phase_cos[h] = next_cos;
                    merged += mw[h] * sine_amp * next_sin;
                }
                source[static_cast<size_t>(seg_start + j)] = std::tanh(merged);
            }
            // 7.2 — Phase normalisation every K segments instead of every segment.
            // Full normalisation (sqrt) after every upsample-step is expensive.
            // Every 16 segments (~4800 samples at 300 upsamp.) keeps float drift
            // well below 1e-6 — inaudible for audio synthesis.
            constexpr int64_t phase_reset_interval = 16;
            if (seg % phase_reset_interval == 0) {
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
    // The buffer lives in Model; we resize + zero it, fill via STFT,
    // then move a copy into the returned CpuTensor.
    std::vector<float> & har_data = model.tmp_stft_har_f32;
    const size_t har_size = static_cast<size_t>(22 * target_frames);
    if (har_data.size() < har_size) {
        har_data.resize(har_size);
    }
    std::fill(har_data.begin(), har_data.begin() + static_cast<ptrdiff_t>(har_size), 0.0f);

    const float * window = hann_window_20();
    const StftTwiddles & tw = stft_twiddles();
    const int64_t center_pad = n_fft / 2;
    for (int64_t frame = 0; frame < target_frames; ++frame) {
        for (int k = 0; k <= n_fft / 2; ++k) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int n = 0; n < n_fft; ++n) {
                const int64_t src_i = frame * hop + n - center_pad;
                const float sample = (src_i >= 0 && src_i < n_samples)
                    ? source[static_cast<size_t>(src_i)] * window[n]
                    : 0.0f;
                real += sample * tw.c[k][n];
                imag -= sample * tw.s[k][n];
            }
            har_data[static_cast<size_t>(k * target_frames + frame)] = std::sqrt(real * real + imag * imag);
            har_data[static_cast<size_t>((k + n_fft / 2 + 1) * target_frames + frame)] = std::atan2(imag, real);
        }
    }
    return CpuTensor{22, target_frames, std::move(har_data)};
}

// 7.3 — cpu_istft now accepts a Model& and a std::vector<float>& for the output buffer.
// The intermediate y[] and denom[] arrays are cached inside Model and reused across calls,
// eliminating two large allocations per chunk.
bool cpu_istft(Model & model, const CpuTensor & post, std::vector<float> & out) {
    constexpr int n_fft = KOKOPOP_STFT_N;
    constexpr int hop = KOKOPOP_STFT_HOP;
    const int64_t n_frames = post.length;
    if (n_frames <= 0) {
        out.clear();
        return true;
    }
    const int64_t padded_len = n_fft + hop * (n_frames - 1);

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
    for (int64_t frame = 0; frame < n_frames; ++frame) {
        float real[11]{};
        float imag[11]{};
        for (int k = 0; k <= n_fft / 2; ++k) {
            const float mag = std::exp(std::clamp(post.at(k, frame), -20.0f, 8.0f));
            const float ph = std::sin(post.at(k + n_fft / 2 + 1, frame));
            real[k] = mag * std::cos(ph);
            imag[k] = mag * std::sin(ph);
        }
        for (int n = 0; n < n_fft; ++n) {
            float sample = real[0] + real[n_fft / 2] * ((n % 2) == 0 ? 1.0f : -1.0f);
            for (int k = 1; k < n_fft / 2; ++k) {
                sample += 2.0f * (real[k] * tw.c[k][n] - imag[k] * tw.s[k][n]);
            }
            sample /= static_cast<float>(n_fft);
            const int64_t dst = frame * hop + n;
            y[static_cast<size_t>(dst)] += sample * window[n];
            denom[static_cast<size_t>(dst)] += window[n] * window[n];
        }
    }
    const int64_t center_pad = n_fft / 2;
    const int64_t out_len = std::max<int64_t>(0, padded_len - 2 * center_pad);
    if (out_len <= 0) {
        out.clear();
        return true;
    }
    if (static_cast<size_t>(out_len) > out.size()) {
        out.resize(static_cast<size_t>(out_len));
    } else {
        out.resize(static_cast<size_t>(out_len));
    }
    for (int64_t i = 0; i < out_len; ++i) {
        const int64_t src = i + center_pad;
        float v = y[static_cast<size_t>(src)];
        if (denom[static_cast<size_t>(src)] > 1e-8f) {
            v /= denom[static_cast<size_t>(src)];
        }
        out[static_cast<size_t>(i)] = std::clamp(v, -1.0f, 1.0f);
    }
    return !out.empty();
}

// ---------------------------------------------------------------------------
// Generator graph (GGML-based audio synthesis)
// ---------------------------------------------------------------------------

bool ggml_generator(
    Model & model, const CpuTensor & decoder,
    const std::vector<float> & f0,
    const std::vector<float> & style,
    std::vector<float> & audio,
    std::string & error) {
    CpuTensor har = cpu_harmonic_stft(model, f0, decoder.length * 60 + 1, error);
    if (!error.empty()) {
        return false;
    }

    const size_t mem_size = model.backend->generator_context_bytes(decoder.length);
    model.backend->clear_pending_inits();
    ggml_context * ctx = init_scratch_context(model, model.generator_scratch, mem_size, true, "generator", error);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * x       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, decoder.length, decoder.channels);
    ggml_tensor * style_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, static_cast<int64_t>(style.size()));
    ggml_tensor * har_t   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, har.length, har.channels);
    ggml_set_input(x);
    ggml_set_input(style_t);
    ggml_set_input(har_t);

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
            "kokopop.decoder.generator.ups.0.bias"
        },
        {
            11, 6, 3, 1, 0, 1,
            "kokopop.decoder.generator.noise_res.1",
            "kokopop.decoder.generator.ups.1.weight",
            "kokopop.decoder.generator.ups.1.bias"
        }
    };

    static const char * noise_conv_weights[2][2] = {
        {"kokopop.decoder.generator.noise_convs.0.weight", "kokopop.decoder.generator.noise_convs.0.bias"},
        {"kokopop.decoder.generator.noise_convs.1.weight", "kokopop.decoder.generator.noise_convs.1.bias"}
    };
    ggml_tensor * noise_conv_w[2] = {nullptr, nullptr};
    ggml_tensor * noise_conv_b[2] = {nullptr, nullptr};
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
        ggml_tensor * x_source = add_channel_bias(ctx,
            conv1d(ctx, noise_conv_w[stage], har_t,
                sp.noise_stride,
                sp.noise_padding,
                1,
                sp.noise_kernel),
            noise_conv_b[stage]);
        x_source = graph_generator_resblock(ctx, model, x_source, style_t, sp.noise_prefix, sp.kernel, error);
        if (x_source == nullptr) {
            ggml_free(ctx);
            return false;
        }

        ggml_tensor * up_w = require_tensor(model, sp.up_weight, error);
        ggml_tensor * up_b = require_tensor(model, sp.up_bias, error);
        if (!error.empty()) {
            ggml_free(ctx);
            return false;
        }
        const int64_t out_len = (x->ne[0] - 1) * sp.up_stride - 2 * sp.up_padding + up_w->ne[0];
        x = add_channel_bias(ctx,
            conv_transpose1d_crop(ctx, up_w, x, sp.up_stride,
                sp.up_padding, static_cast<int>(out_len)),
            up_b);
        if (stage == 1) {
            x = ggml_pad_reflect_1d(ctx, x, 1, 0);
        }
        x = ggml_add(ctx, x, x_source);

        ggml_tensor * accum = nullptr;
        for (int k = 0; k < 3; ++k) {
            const std::string resblock_prefix = "kokopop.decoder.generator.resblocks." + std::to_string(stage * 3 + k);
            ggml_tensor * branch = graph_generator_resblock(
                ctx, model, x, style_t, resblock_prefix,
                KOKOPOP_RESBLOCK_KERNELS[k], error);
            if (branch == nullptr) {
                ggml_free(ctx);
                return false;
            }
            accum = accum == nullptr ? branch : ggml_add(ctx, accum, branch);
        }
        x = ggml_scale(ctx, accum, 1.0f / 3.0f);
    }

    x = ggml_leaky_relu(ctx, x, 0.01f, false);
    ggml_tensor * post = add_channel_bias(ctx,
        conv1d(ctx,
            require_tensor(model, "kokopop.decoder.generator.conv_post.weight", error),
            x,
            1,
            3,
            1,
            7),
        require_tensor(model, "kokopop.decoder.generator.conv_post.bias", error));
    if (!error.empty()) {
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, generator_graph_size(decoder.length), false);
    ggml_set_output(post);
    ggml_build_forward_expand(gf, post);

    model.backend->set_active_label("generator");
    model.backend->sched_reset();
    if (!model.backend->sched_alloc_graph(gf)) {
        ggml_free(ctx);
        error = "ggml generator backend allocation failed";
        return false;
    }
    if (!model.backend->apply_pending_inits()) {
        ggml_free(ctx);
        error = "ggml generator backend tensor initialization failed";
        return false;
    }
    model.backend->tensor_set(x,       decoder.data.data(), 0, decoder.data.size() * sizeof(float));
    model.backend->tensor_set(style_t, style.data(),        0, style.size()        * sizeof(float));
    model.backend->tensor_set(har_t,   har.data.data(),     0, har.data.size()     * sizeof(float));
    ggml_status status = model.backend->compute(ctx, gf);

    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        error = "ggml generator graph compute failed";
        return false;
    }

    CpuTensor post_cpu{
        post->ne[1],
        post->ne[0],
        {},
    };
    const size_t post_size = static_cast<size_t>(post->ne[0] * post->ne[1]);
    post_cpu.data.resize(post_size);
    model.backend->tensor_get(post, post_cpu.data.data(), 0, post_cpu.data.size() * sizeof(float));
    // 7.3 — Pass the output buffer; cpu_istft writes directly into it.
    if (!cpu_istft(model, post_cpu, audio)) {
        ggml_free(ctx);
        error = "cpu_istft failed";
        return false;
    }
    ggml_free(ctx);
    return !audio.empty();
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

} // namespace kokopop
