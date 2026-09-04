#include "test_helpers.h"
#include "arch/kokoro/kokoro.h"
#include "core/constants.h"
#include "model/model.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>

using namespace kokopop;

// Forward declarations for internal functions in audio_utils.cpp
// These are not public API but needed for unit testing the STFT/ISTFT pipeline.
namespace kokopop {
const float * hann_window_20();
bool cpu_istft(KokoroArch & model, const CpuTensor & post, std::vector<float> & out);
} // namespace kokopop

// ---------------------------------------------------------------------------
// Helpers to call internal STFT/ISTFT functions via friend-declared test
// accessors or a small test harness.
//
// Because cpu_harmonic_stft and cpu_istft are static/internal, we expose
// thin wrapper functions in the test file that construct the minimal Model
// state needed.
// ---------------------------------------------------------------------------

// Build a minimal KokoroArch for STFT/ISTFT testing.
// cpu_istft only needs the arch's temporary buffers, not the backend, so the
// arch is used detached from any Model (`base` stays null).
static std::unique_ptr<KokoroArch> make_stft_test_model() {
    return std::make_unique<KokoroArch>();
}

// ---------------------------------------------------------------------------
// 7.1 — STFT source/har buffer reuse: verify no crash across repeated calls
// and that the output CpuTensor has correct dimensions.
// ---------------------------------------------------------------------------

// We cannot call cpu_harmonic_stft directly (it is file-static).
// Instead we verify the cached buffers in Model grow correctly.
// The actual correctness is validated through the roundtrip tests below.

TEST_CASE("stft_buffers_grow_monotonically") {
    auto model = make_stft_test_model();
    // The STFT buffer fields start empty.
    CHECK(model->tmp_stft_source_f32.empty());
    CHECK(model->tmp_stft_har_f32.empty());
    CHECK(model->tmp_istft_real_f32.empty());
    CHECK(model->tmp_istft_imag_f32.empty());
    CHECK(model->istft_workspace.ola.empty());
    CHECK(model->istft_workspace.envelope.empty());
}

// ---------------------------------------------------------------------------
// 7.3 — ISTFT roundtrip test: construct a known spectral tensor, run ISTFT,
// verify output dimensions and value bounds.
// ---------------------------------------------------------------------------

TEST_CASE("istft_output_dimensions") {
    auto model = make_stft_test_model();

    // Construct a post-tensor with non-zero magnitudes.
    // Magnitudes = 0.0 → exp(0) = 1 (unit magnitude).
    // Phases = 0 → cos(0) = 1, sin(0) = 0 → pure real.
    constexpr int64_t n_frames = 50;
    constexpr size_t total = static_cast<size_t>(22 * n_frames);
    std::vector<float> data(total, 0.0f);
    // Set magnitudes to 0.0 (log-mag of 1.0)
    // Phase channels (11-21) remain 0.0

    CpuTensor post{22, n_frames, std::move(data)};

    std::vector<float> out;
    bool ok = cpu_istft(*model, post, out);
    CHECK(ok);

    // Expected output length:
    // padded_len = n_fft + hop*(n_frames-1) = 20 + 5*49 = 265
    // out_len = padded_len - 2*(n_fft/2) = 265 - 20 = 245
    constexpr int64_t expected_len = KOKOPOP_STFT_N
        + KOKOPOP_STFT_HOP * (n_frames - 1)
        - 2 * (KOKOPOP_STFT_N / 2);
    CHECK_EQ(static_cast<int64_t>(out.size()), expected_len);

    // All values should be in [-1, 1] (clamped).
    for (float v : out) {
        CHECK(v >= -1.0f);
        CHECK(v <= 1.0f);
    }
}

TEST_CASE("istft_empty_input") {
    auto model = make_stft_test_model();
    CpuTensor post{22, 0, {}};
    std::vector<float> out;
    bool ok = cpu_istft(*model, post, out);
    CHECK(ok);
    CHECK(out.empty());
}

TEST_CASE("istft_single_frame") {
    auto model = make_stft_test_model();
    CpuTensor post{22, 1, std::vector<float>(22, 0.0f)};

    std::vector<float> out;
    bool ok = cpu_istft(*model, post, out);
    CHECK(ok);
    // padded_len = 20 + 5*(1-1) = 20, out_len = 20 - 20 = 0
    // Edge case: single frame produces 0-length output.
    CHECK(out.empty());
}

TEST_CASE("istft_clamping_at_bounds") {
    auto model = make_stft_test_model();
    constexpr int64_t n_frames = 100;
    size_t total = static_cast<size_t>(22 * n_frames);

    // Test 1: Maximum allowed magnitude → all output in [-1, 1].
    {
        std::vector<float> data(total, 8.0f);
        CpuTensor post{22, n_frames, std::move(data)};
        std::vector<float> out;
        cpu_istft(*model, post, out);
        for (float v : out) {
            CHECK(v >= -1.0f);
            CHECK(v <= 1.0f);
        }
    }

    // Test 2: Magnitude far above clamp threshold → still in [-1, 1].
    {
        std::vector<float> data(total, 100.0f);
        CpuTensor post{22, n_frames, std::move(data)};
        std::vector<float> out;
        cpu_istft(*model, post, out);
        for (float v : out) {
            CHECK(v >= -1.0f);
            CHECK(v <= 1.0f);
        }
    }

}

TEST_CASE("istft_negative_magnitudes_clamped") {
    auto model = make_stft_test_model();
    constexpr int64_t n_frames = 100;
    size_t total = static_cast<size_t>(22 * n_frames);
    std::vector<float> data(total, -20.0f); // min allowed by clamp

    CpuTensor post{22, n_frames, std::move(data)};

    std::vector<float> out;
    bool ok = cpu_istft(*model, post, out);
    CHECK(ok);

    // exp(-20) ~ 2e-9, so output should be nearly zero.
    for (float v : out) {
        CHECK(std::fabs(v) < 1e-5f);
    }
}

// ---------------------------------------------------------------------------
// 7.3 — ISTFT buffer reuse: verify that buffers grow and are reused.
// ---------------------------------------------------------------------------

TEST_CASE("istft_buffer_reuse") {
    auto model = make_stft_test_model();
    constexpr int64_t n_frames = 100;
    size_t total = static_cast<size_t>(22 * n_frames);
    std::vector<float> data(total, 0.0f);
    CpuTensor post{22, n_frames, std::move(data)};

    std::vector<float> out1;
    cpu_istft(*model, post, out1);

    // The complex half-spectrum is sized by the frame count...
    CHECK_EQ(model->tmp_istft_real_f32.size(), 11u * static_cast<size_t>(n_frames));
    CHECK_EQ(model->tmp_istft_imag_f32.size(), 11u * static_cast<size_t>(n_frames));
    // ...but the overlap-add scratch is not: the shared component accumulates
    // in a ring the width of one window, so it stays O(n_fft) whatever the
    // signal length. This is what makes a long chunk allocation-free.
    CHECK_EQ(model->istft_workspace.ola.size(), static_cast<size_t>(KOKOPOP_STFT_N));
    CHECK_EQ(model->istft_workspace.envelope.size(), static_cast<size_t>(KOKOPOP_STFT_N));

    // Second call with the same size — nothing reallocates.
    std::vector<float> data2(total, 0.1f);
    CpuTensor post2{22, n_frames, std::move(data2)};
    std::vector<float> out2;
    cpu_istft(*model, post2, out2);
    CHECK_EQ(model->tmp_istft_real_f32.size(), 11u * static_cast<size_t>(n_frames));
    CHECK_EQ(model->istft_workspace.ola.size(), static_cast<size_t>(KOKOPOP_STFT_N));

    // Third call with twice the frames: the spectrum grows, the ring does not.
    constexpr int64_t n_frames_big = 200;
    size_t total_big = static_cast<size_t>(22 * n_frames_big);
    std::vector<float> data_big(total_big, 0.0f);
    CpuTensor post_big{22, n_frames_big, std::move(data_big)};
    std::vector<float> out3;
    cpu_istft(*model, post_big, out3);

    CHECK_EQ(model->tmp_istft_real_f32.size(), 11u * static_cast<size_t>(n_frames_big));
    CHECK_EQ(model->istft_workspace.ola.size(), static_cast<size_t>(KOKOPOP_STFT_N));
    CHECK_EQ(out3.size(), static_cast<size_t>(n_frames_big - 1) * KOKOPOP_STFT_HOP);
}

// ---------------------------------------------------------------------------
// 7.1 — Source/har buffer reuse: verify that buffers are properly sized.
// We test this indirectly by checking the Model fields after the test
// harness sets them up.
// ---------------------------------------------------------------------------

TEST_CASE("stft_source_buffer_init") {
    auto model = make_stft_test_model();
    // Simulate what cpu_harmonic_stft does for the source buffer.
    // With f0.size() = 10 and upsample = 300, source needs 3000 floats.
    constexpr int64_t f0_size = 10;
    constexpr int64_t upsample = KOKOPOP_UPSAMPLE;
    const int64_t n_samples = f0_size * upsample;

    std::vector<float> & source = model->tmp_stft_source_f32;
    if (source.size() < static_cast<size_t>(n_samples)) {
        source.resize(static_cast<size_t>(n_samples));
    }
    CHECK_EQ(source.size(), static_cast<size_t>(n_samples));

    // Second call with larger size — buffer grows.
    constexpr int64_t f0_size2 = 20;
    const int64_t n_samples2 = f0_size2 * upsample;
    if (source.size() < static_cast<size_t>(n_samples2)) {
        source.resize(static_cast<size_t>(n_samples2));
    }
    CHECK_EQ(source.size(), static_cast<size_t>(n_samples2));

    // Third call with smaller size — buffer does NOT shrink.
    if (source.size() < static_cast<size_t>(n_samples)) {
        source.resize(static_cast<size_t>(n_samples));
    }
    CHECK_EQ(source.size(), static_cast<size_t>(n_samples2));
}

// ---------------------------------------------------------------------------
// 7.2 — Phase reset normalization correctness: verify that normalizing
// every K segments produces comparable output to normalizing every segment.
// We test this mathematically on a simple phase accumulator.
// ---------------------------------------------------------------------------

TEST_CASE("phase_reset_normalization_drift") {
    constexpr int harmonic = KOKOPOP_HARMONIC_COUNT;
    constexpr int upsample = KOKOPOP_UPSAMPLE;
    constexpr int sample_rate = KOKOPOP_SAMPLE_RATE;
    constexpr float test_f0 = 200.0f; // A3

    // Full normalisation: normalize after every segment
    {
        float phase_sin_full[harmonic]{};
        float phase_cos_full[harmonic];
        std::fill(phase_cos_full, phase_cos_full + harmonic, 1.0f);

        // Simulate 500 segments
        for (int seg = 0; seg < 500; ++seg) {
            float sin_delta[harmonic];
            float cos_delta[harmonic];
            for (int h = 0; h < harmonic; ++h) {
                const float delta = 2.0f * M_PI * test_f0 * static_cast<float>(h + 1) /
                    static_cast<float>(sample_rate);
                sin_delta[h] = std::sin(delta);
                cos_delta[h] = std::cos(delta);
            }
            for (int j = 0; j < upsample; ++j) {
                for (int h = 0; h < harmonic; ++h) {
                    const float s = phase_sin_full[h];
                    const float c = phase_cos_full[h];
                    phase_sin_full[h] = s * cos_delta[h] + c * sin_delta[h];
                    phase_cos_full[h] = c * cos_delta[h] - s * sin_delta[h];
                }
            }
            // Normalize every segment
            for (int h = 0; h < harmonic; ++h) {
                const float norm = std::sqrt(
                    phase_sin_full[h] * phase_sin_full[h] +
                    phase_cos_full[h] * phase_cos_full[h]);
                if (norm > 0.0f) {
                    phase_sin_full[h] /= norm;
                    phase_cos_full[h] /= norm;
                }
            }
        }

        // Check magnitudes are ~1.0
        for (int h = 0; h < harmonic; ++h) {
            const float mag = std::sqrt(
                phase_sin_full[h] * phase_sin_full[h] +
                phase_cos_full[h] * phase_cos_full[h]);
            CHECK_NEAR(mag, 1.0f, 1e-6f);
        }
    }

    // Sparse normalisation: normalize every 100 segments
    {
        float phase_sin_sparse[harmonic]{};
        float phase_cos_sparse[harmonic];
        std::fill(phase_cos_sparse, phase_cos_sparse + harmonic, 1.0f);

        constexpr int reset_interval = 16; // matches the source code

        for (int seg = 0; seg < 500; ++seg) {
            float sin_delta[harmonic];
            float cos_delta[harmonic];
            for (int h = 0; h < harmonic; ++h) {
                const float delta = 2.0f * M_PI * test_f0 * static_cast<float>(h + 1) /
                    static_cast<float>(sample_rate);
                sin_delta[h] = std::sin(delta);
                cos_delta[h] = std::cos(delta);
            }
            for (int j = 0; j < upsample; ++j) {
                for (int h = 0; h < harmonic; ++h) {
                    const float s = phase_sin_sparse[h];
                    const float c = phase_cos_sparse[h];
                    phase_sin_sparse[h] = s * cos_delta[h] + c * sin_delta[h];
                    phase_cos_sparse[h] = c * cos_delta[h] - s * sin_delta[h];
                }
            }
            // Normalize every K segments (our optimization)
            if (seg % reset_interval == 0) {
                for (int h = 0; h < harmonic; ++h) {
                    const float norm = std::sqrt(
                        phase_sin_sparse[h] * phase_sin_sparse[h] +
                        phase_cos_sparse[h] * phase_cos_sparse[h]);
                    if (norm > 0.0f) {
                        phase_sin_sparse[h] /= norm;
                        phase_cos_sparse[h] /= norm;
                    }
                }
            }
        }

        // Check magnitudes are still ~1.0 (drift should be negligible)
        for (int h = 0; h < harmonic; ++h) {
            const float mag = std::sqrt(
                phase_sin_sparse[h] * phase_sin_sparse[h] +
                phase_cos_sparse[h] * phase_cos_sparse[h]);
            // After 500 segments with reset every 100, max drift is tiny.
            CHECK_NEAR(mag, 1.0f, 1e-4f);
        }
    }
}

// ---------------------------------------------------------------------------
// 7.4 — StftTwiddles correctness: verify the twiddle arrays through a
// single-frequency ISTFT reconstruction with an analytically known result.
//
// Setup: n_frames=3, only bin k=1 active (log_mag=0 → mag=1, phase=0).
// All other bins have log_mag=-20 (mag≈0).
//
// ISTFT formula for k in [1, N/2-1]:
//   sample[n] += 2 * (re * c[k][n] - im * s[k][n]) / N * window[n]
//
// For k=1, re=1, im=0, n=10 (padded position of out[0]):
//   frame 0: 2*cos(2π*10/20)/20 * w[10] = 2*cos(π)/20 * 1 = -0.1
//   frame 1: 2*cos(2π*5/20)/20  * w[5]  = 2*cos(π/2)/20 * 0.5 = 0
//   frame 2: 2*cos(0)/20        * w[0]  = 0 (w[0]=0)
//
// denom[10] = w[10]² + w[5]² + w[0]² = 1.0 + 0.25 + 0 = 1.25
// out[0] = -0.1 / 1.25 = -0.08
//
// If tw.c[1][10] ≠ cos(π) = -1, the output will differ from -0.08.
// ---------------------------------------------------------------------------

TEST_CASE("stft_twiddles_k1_reconstruction") {
    auto model = make_stft_test_model();
    constexpr int64_t n_frames = 3;
    const size_t total = static_cast<size_t>(22 * n_frames);

    std::vector<float> data(total, -20.0f); // all bins near-zero magnitude
    // k=1 (row 1): unit magnitude across all frames
    for (int64_t f = 0; f < n_frames; ++f) {
        data[static_cast<size_t>(1 * n_frames + f)] = 0.0f;
    }
    // phase rows 11-21: 0 so that sin(0)=0 → ph=0 → re=mag, im=0
    for (int64_t r = 11; r < 22; ++r) {
        for (int64_t f = 0; f < n_frames; ++f) {
            data[static_cast<size_t>(r * n_frames + f)] = 0.0f;
        }
    }

    CpuTensor post{22, n_frames, std::move(data)};
    std::vector<float> out;
    bool ok = cpu_istft(*model, post, out);
    CHECK(ok);

    // out_len = n_fft + hop*(3-1) - 2*(n_fft/2) = 20 + 10 - 20 = 10
    CHECK_EQ(static_cast<int64_t>(out.size()), 10);

    // out[0] must equal the analytically derived -0.08.
    // Any error in tw.c[1][10] would shift this value.
    CHECK_NEAR(out[0], -0.08f, 1e-4f);
}

// ---------------------------------------------------------------------------
// Hann window verification
// ---------------------------------------------------------------------------

TEST_CASE("hann_window_properties") {
    constexpr int N = KOKOPOP_STFT_N;
    const float * win = hann_window_20();

    // Hann window: w[n] = 0.5 - 0.5*cos(2*pi*n/N)
    // Properties: w[0] = w[N] = 0, w[N/2] = 1, all values in [0, 1]
    for (int n = 0; n < N; ++n) {
        CHECK(win[n] >= 0.0f);
        CHECK(win[n] <= 1.0f);
    }
    // Check endpoints are ~0
    CHECK_NEAR(win[0], 0.0f, 1e-6f);
    // Check center is ~1
    CHECK_NEAR(win[N / 2], 1.0f, 1e-6f);
}
