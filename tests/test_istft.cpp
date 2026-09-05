#include "audio/istft.h"
#include "core/constants.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <string>
#include <vector>

using kokopop::ComplexSpectrumView;
using kokopop::IstftConfig;
using kokopop::IstftPlan;
using kokopop::IstftWorkspace;

namespace {

constexpr double PI = 3.14159265358979323846;

struct Spectrum {
    size_t bins = 0;
    size_t frames = 0;
    std::vector<float> real;  // [bins][frames]
    std::vector<float> imag;

    ComplexSpectrumView view() const {
        ComplexSpectrumView v;
        v.real = real.data();
        v.imag = imag.data();
        v.bins = bins;
        v.frames = frames;
        v.bin_stride = static_cast<ptrdiff_t>(frames);
        v.frame_stride = 1;
        return v;
    }
};

Spectrum random_spectrum(size_t bins, size_t frames, uint32_t seed,
                         float scale = 1.0f) {
    Spectrum s;
    s.bins = bins;
    s.frames = frames;
    s.real.resize(bins * frames);
    s.imag.resize(bins * frames);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < s.real.size(); ++i) {
        s.real[i] = dist(rng);
        s.imag[i] = dist(rng);
    }
    // A real signal has no imaginary part at DC or Nyquist.
    for (size_t f = 0; f < frames; ++f) {
        s.imag[f] = 0.0f;
        s.imag[(bins - 1) * frames + f] = 0.0f;
    }
    return s;
}

/// Reference inverse DFT of one frame of a Hermitian half-spectrum, in double.
/// Deliberately the textbook O(N^2) form: it is the oracle, so it must be the
/// version that is obviously right rather than the fast one.
std::vector<double> reference_irfft(const float * real, const float * imag,
                                    size_t bins, ptrdiff_t bin_stride,
                                    size_t n_fft) {
    std::vector<double> out(n_fft, 0.0);
    for (size_t i = 0; i < n_fft; ++i) {
        std::complex<double> acc(0.0, 0.0);
        for (size_t k = 0; k < n_fft; ++k) {
            // Hermitian extension: X[N-k] = conj(X[k]).
            std::complex<double> x;
            if (k < bins) {
                x = {static_cast<double>(real[bin_stride * static_cast<ptrdiff_t>(k)]),
                     static_cast<double>(imag[bin_stride * static_cast<ptrdiff_t>(k)])};
            } else {
                const size_t m = n_fft - k;
                x = {static_cast<double>(real[bin_stride * static_cast<ptrdiff_t>(m)]),
                     -static_cast<double>(imag[bin_stride * static_cast<ptrdiff_t>(m)])};
            }
            const double angle = 2.0 * PI * static_cast<double>(k * i) /
                                 static_cast<double>(n_fft);
            acc += x * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        out[i] = acc.real() / static_cast<double>(n_fft);
    }
    return out;
}

/// Reference iSTFT: reference_irfft per frame, then a flat overlap-add and a
/// division by the accumulated window energy.
std::vector<double> reference_istft(const Spectrum & s, const IstftConfig & cfg,
                                    const std::vector<float> & window) {
    const size_t n = cfg.n_fft;
    const size_t padded = n + cfg.hop * (s.frames - 1);
    std::vector<double> y(padded, 0.0);
    std::vector<double> env(padded, 0.0);
    for (size_t f = 0; f < s.frames; ++f) {
        const std::vector<double> frame =
            reference_irfft(s.real.data() + f, s.imag.data() + f, s.bins,
                            static_cast<ptrdiff_t>(s.frames), n);
        for (size_t i = 0; i < n; ++i) {
            y[cfg.hop * f + i] += frame[i] * static_cast<double>(window[i]);
            env[cfg.hop * f + i] += static_cast<double>(window[i]) *
                                    static_cast<double>(window[i]);
        }
    }
    const size_t trim = cfg.center ? n / 2 : 0;
    const size_t out_len = padded > 2 * trim ? padded - 2 * trim : 0;
    std::vector<double> out(out_len, 0.0);
    for (size_t i = 0; i < out_len; ++i) {
        const double d = env[i + trim];
        out[i] = d > 1e-8 ? y[i + trim] / d : 0.0;
    }
    return out;
}

/// Forward STFT of a real signal, in double: reflect-pad by n_fft/2, window,
/// naive DFT, keep bins 0..n_fft/2. The inverse under test must undo this.
Spectrum reference_stft(const std::vector<double> & signal, const IstftConfig & cfg,
                        const std::vector<float> & window, size_t frames) {
    const size_t n = cfg.n_fft;
    const size_t bins = n / 2 + 1;
    const ptrdiff_t pad = cfg.center ? static_cast<ptrdiff_t>(n / 2) : 0;

    Spectrum s;
    s.bins = bins;
    s.frames = frames;
    s.real.assign(bins * frames, 0.0f);
    s.imag.assign(bins * frames, 0.0f);

    const ptrdiff_t len = static_cast<ptrdiff_t>(signal.size());
    for (size_t f = 0; f < frames; ++f) {
        std::vector<double> frame(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            ptrdiff_t idx = static_cast<ptrdiff_t>(cfg.hop * f + i) - pad;
            // torch.stft(center=True)'s reflect padding.
            if (idx < 0) {
                idx = -idx;
            } else if (idx >= len) {
                idx = 2 * len - idx - 2;
            }
            const double sample = (idx >= 0 && idx < len)
                                ? signal[static_cast<size_t>(idx)] : 0.0;
            frame[i] = sample * static_cast<double>(window[i]);
        }
        for (size_t k = 0; k < bins; ++k) {
            double re = 0.0;
            double im = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double angle = 2.0 * PI * static_cast<double>(k * i) /
                                     static_cast<double>(n);
                re += frame[i] * std::cos(angle);
                im -= frame[i] * std::sin(angle);
            }
            s.real[k * frames + f] = static_cast<float>(re);
            s.imag[k * frames + f] = static_cast<float>(im);
        }
    }
    return s;
}

double worst_abs_diff(const std::vector<float> & got,
                      const std::vector<double> & want) {
    double worst = 0.0;
    const size_t n = std::min(got.size(), want.size());
    for (size_t i = 0; i < n; ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[i]));
    }
    return worst;
}

IstftPlan make_plan(uint32_t n_fft, uint32_t hop, bool center = true) {
    IstftPlan plan;
    std::string error;
    REQUIRE(IstftPlan::create(IstftConfig{n_fft, hop, center}, plan, error));
    REQUIRE(error.empty());
    return plan;
}

} // namespace

// ---------------------------------------------------------------------------
// Configuration validation — the plan constructor is the only gate
// ---------------------------------------------------------------------------

TEST_CASE("istft_plan_rejects_invalid_configurations") {
    IstftPlan plan;
    std::string error;

    CHECK_FALSE(IstftPlan::create(IstftConfig{0, 1, true}, plan, error));
    CHECK(error.find("n_fft") != std::string::npos);

    // Odd n_fft has no n_fft/2 Nyquist bin.
    CHECK_FALSE(IstftPlan::create(IstftConfig{21, 5, true}, plan, error));

    // hop must be in (0, n_fft].
    CHECK_FALSE(IstftPlan::create(IstftConfig{20, 0, true}, plan, error));
    CHECK_FALSE(IstftPlan::create(IstftConfig{20, 21, true}, plan, error));

    // Even and large, but not a power of two: neither kernel can take it.
    CHECK_FALSE(IstftPlan::create(IstftConfig{1000, 250, true}, plan, error));
    CHECK(error.find("power of two") != std::string::npos);
}

TEST_CASE("istft_plan_picks_the_kernel_from_the_size") {
    CHECK(make_plan(KOKOPOP_STFT_N, KOKOPOP_STFT_HOP).kernel() ==
          IstftPlan::Kernel::DirectDft);
    CHECK(make_plan(1024, 256).kernel() == IstftPlan::Kernel::RealRadix2);
    CHECK(make_plan(64, 16).kernel() == IstftPlan::Kernel::RealRadix2);
    CHECK(make_plan(32, 8).kernel() == IstftPlan::Kernel::DirectDft);
}

TEST_CASE("istft_plan_reports_bins_and_output_length") {
    const IstftPlan small = make_plan(20, 5);
    CHECK_EQ(small.bins(), 11u);

    size_t samples = 0;
    REQUIRE(small.output_samples(0, samples));
    CHECK_EQ(samples, 0u);
    REQUIRE(small.output_samples(1, samples));
    CHECK_EQ(samples, 0u);  // one centred frame reconstructs nothing
    REQUIRE(small.output_samples(100, samples));
    CHECK_EQ(samples, 99u * 5u);

    const IstftPlan big = make_plan(1024, 256);
    CHECK_EQ(big.bins(), 513u);
    REQUIRE(big.output_samples(420, samples));
    CHECK_EQ(samples, 419u * 256u);

    // center=false keeps the padded length.
    const IstftPlan uncentred = make_plan(1024, 256, false);
    REQUIRE(uncentred.output_samples(420, samples));
    CHECK_EQ(samples, 1024u + 419u * 256u);

    // A frame count that would overflow the length arithmetic is rejected
    // rather than wrapping into a small allocation.
    CHECK_FALSE(big.output_samples(~size_t{0}, samples));
}

// ---------------------------------------------------------------------------
// The window
// ---------------------------------------------------------------------------

TEST_CASE("istft_window_is_the_periodic_hann") {
    for (uint32_t n : {20u, 64u, 1024u}) {
        const IstftPlan plan = make_plan(n, n / 4 == 0 ? 1 : n / 4);
        const std::vector<float> & w = plan.window();
        REQUIRE_EQ(w.size(), n);
        // Periodic, not symmetric: w[0] == 0 and w[n-1] != 0.
        CHECK_EQ(w.front(), 0.0f);
        CHECK(w.back() > 0.0f);
        for (uint32_t i = 0; i < n; ++i) {
            // sin^2(pi*i/N) is the algebraically identical form.
            const double want = std::pow(std::sin(PI * i / static_cast<double>(n)), 2.0);
            CHECK_NEAR(w[i], want, 1e-6);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel correctness against the naive oracle
// ---------------------------------------------------------------------------

TEST_CASE("istft_direct_kernel_matches_the_reference_dft") {
    const IstftPlan plan = make_plan(KOKOPOP_STFT_N, KOKOPOP_STFT_HOP);
    const Spectrum s = random_spectrum(plan.bins(), 40, 0x51f7u);

    IstftWorkspace ws;
    std::vector<float> got;
    std::string error;
    REQUIRE(kokopop::istft(plan, ws, s.view(), got, error));

    const std::vector<double> want = reference_istft(s, plan.config(), plan.window());
    REQUIRE_EQ(got.size(), want.size());
    CHECK(worst_abs_diff(got, want) < 2e-5);
}

TEST_CASE("istft_radix2_kernel_matches_the_reference_dft") {
    const IstftPlan plan = make_plan(1024, 256);
    // 6 frames keeps the O(N^2) oracle affordable at N=1024.
    const Spectrum s = random_spectrum(plan.bins(), 6, 0xbeefu);

    IstftWorkspace ws;
    std::vector<float> got;
    std::string error;
    REQUIRE(kokopop::istft(plan, ws, s.view(), got, error));

    const std::vector<double> want = reference_istft(s, plan.config(), plan.window());
    REQUIRE_EQ(got.size(), want.size());
    CHECK(worst_abs_diff(got, want) < 1e-4);
}

TEST_CASE("istft_radix2_kernel_matches_at_other_power_of_two_sizes") {
    for (uint32_t n : {64u, 128u, 256u}) {
        const IstftPlan plan = make_plan(n, n / 4);
        const Spectrum s = random_spectrum(plan.bins(), 8, 0x1234u + n);

        IstftWorkspace ws;
        std::vector<float> got;
        std::string error;
        REQUIRE(kokopop::istft(plan, ws, s.view(), got, error));

        const std::vector<double> want =
            reference_istft(s, plan.config(), plan.window());
        REQUIRE_EQ(got.size(), want.size());
        CHECK(worst_abs_diff(got, want) < 5e-5);
    }
}

// A round trip is the test that would catch a sign, phase or normalisation
// error: forward STFT in double, inverse under test, and the interior of the
// signal must come back. A single sinusoid makes any phase error obvious.
TEST_CASE("istft_round_trips_a_pure_tone") {
    constexpr uint32_t n_fft = 1024;
    constexpr uint32_t hop = 256;
    constexpr size_t frames = 24;
    const IstftPlan plan = make_plan(n_fft, hop);

    const size_t n_samples = (frames - 1) * hop;
    std::vector<double> signal(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        signal[i] = 0.25 * std::cos(2.0 * PI * 40.0 * static_cast<double>(i) /
                                        static_cast<double>(n_fft) + 0.7);
    }

    const Spectrum s = reference_stft(signal, plan.config(), plan.window(), frames);
    IstftWorkspace ws;
    std::vector<float> got;
    std::string error;
    REQUIRE(kokopop::istft(plan, ws, s.view(), got, error));
    REQUIRE_EQ(got.size(), n_samples);

    // Skip one window at each end: reflect padding makes the edges a
    // reconstruction of the padded signal, not of the original.
    for (size_t i = n_fft; i + n_fft < got.size(); ++i) {
        CHECK_NEAR(got[i], signal[i], 3e-5);
    }
}

TEST_CASE("istft_round_trips_a_broadband_signal") {
    constexpr uint32_t n_fft = 256;
    constexpr uint32_t hop = 64;
    constexpr size_t frames = 40;
    const IstftPlan plan = make_plan(n_fft, hop);

    const size_t n_samples = (frames - 1) * hop;
    std::mt19937 rng(0x5eedu);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<double> signal(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        signal[i] = dist(rng);
    }

    const Spectrum s = reference_stft(signal, plan.config(), plan.window(), frames);
    IstftWorkspace ws;
    std::vector<float> got;
    std::string error;
    REQUIRE(kokopop::istft(plan, ws, s.view(), got, error));
    REQUIRE_EQ(got.size(), n_samples);

    for (size_t i = n_fft; i + n_fft < got.size(); ++i) {
        CHECK_NEAR(got[i], signal[i], 2e-5);
    }
}

// The same round trip at Kokoro's size, which uses the other kernel.
TEST_CASE("istft_round_trips_at_the_kokoro_size") {
    const IstftPlan plan = make_plan(KOKOPOP_STFT_N, KOKOPOP_STFT_HOP);
    constexpr size_t frames = 120;
    const size_t n_samples = (frames - 1) * KOKOPOP_STFT_HOP;

    std::vector<double> signal(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        signal[i] = 0.3 * std::sin(0.11 * static_cast<double>(i)) +
                    0.1 * std::sin(0.7 * static_cast<double>(i) + 1.0);
    }

    const Spectrum s = reference_stft(signal, plan.config(), plan.window(), frames);
    IstftWorkspace ws;
    std::vector<float> got;
    std::string error;
    REQUIRE(kokopop::istft(plan, ws, s.view(), got, error));
    REQUIRE_EQ(got.size(), n_samples);

    for (size_t i = KOKOPOP_STFT_N; i + KOKOPOP_STFT_N < got.size(); ++i) {
        CHECK_NEAR(got[i], signal[i], 1e-5);
    }
}

// ---------------------------------------------------------------------------
// Overlap-add, envelope and edges
// ---------------------------------------------------------------------------

// The ring buffer must give exactly what a flat accumulator would. The
// reference above uses a flat one, so agreeing with it over many frames — far
// more than n_fft/hop, so the ring wraps repeatedly — is the check.
TEST_CASE("istft_ring_overlap_add_equals_a_flat_accumulator") {
    const IstftPlan plan = make_plan(64, 16);
    const Spectrum s = random_spectrum(plan.bins(), 200, 0xa5a5u);

    IstftWorkspace ws;
    std::vector<float> got;
    std::string error;
    REQUIRE(kokopop::istft(plan, ws, s.view(), got, error));

    const std::vector<double> want = reference_istft(s, plan.config(), plan.window());
    REQUIRE_EQ(got.size(), want.size());
    CHECK(worst_abs_diff(got, want) < 5e-5);
}

// A hop that does not divide n_fft makes the ring wrap at a different offset
// on every frame, so the frame no longer lands on the ring as one aligned
// block. Every other size here divides evenly and would never notice.
TEST_CASE("istft_ring_handles_a_hop_that_does_not_divide_n_fft") {
    for (uint32_t hop : {7u, 25u, 63u}) {
        const IstftPlan plan = make_plan(64, hop);
        const Spectrum s = random_spectrum(plan.bins(), 37, 0x1234u);

        IstftWorkspace ws;
        std::vector<float> got;
        std::string error;
        REQUIRE(kokopop::istft(plan, ws, s.view(), got, error));

        const std::vector<double> want = reference_istft(s, plan.config(), plan.window());
        REQUIRE_EQ(got.size(), want.size());
        INFO("hop " << hop);
        CHECK(worst_abs_diff(got, want) < 5e-5);
    }
}

// A constant signal is the cleanest way to see the window: after the Sum(w^2)
// division the interior must be flat, with no scalloping at the frame seams.
TEST_CASE("istft_envelope_normalisation_removes_the_window") {
    constexpr uint32_t n_fft = 256;
    constexpr uint32_t hop = 64;
    constexpr size_t frames = 24;
    const IstftPlan plan = make_plan(n_fft, hop);

    const size_t n_samples = (frames - 1) * hop;
    const std::vector<double> signal(n_samples, 0.5);

    const Spectrum s = reference_stft(signal, plan.config(), plan.window(), frames);
    IstftWorkspace ws;
    std::vector<float> got;
    std::string error;
    REQUIRE(kokopop::istft(plan, ws, s.view(), got, error));
    REQUIRE_EQ(got.size(), n_samples);

    for (size_t i = n_fft; i + n_fft < got.size(); ++i) {
        CHECK_NEAR(got[i], 0.5f, 2e-5);
    }
}

// Edge samples use the envelope that actually accumulated over them, so they
// stay bounded rather than being divided by a precomputed interior sum.
TEST_CASE("istft_edges_use_their_own_accumulated_envelope") {
    const IstftPlan plan = make_plan(256, 64);
    const Spectrum s = random_spectrum(plan.bins(), 12, 0x77u, 20.0f);

    IstftWorkspace ws;
    std::vector<float> got;
    std::string error;
    REQUIRE(kokopop::istft(plan, ws, s.view(), got, error));

    const std::vector<double> want = reference_istft(s, plan.config(), plan.window());
    REQUIRE_EQ(got.size(), want.size());
    // The first and last hop are the interesting part; compare everywhere.
    CHECK(worst_abs_diff(got, want) < 1e-3);
    for (float v : got) {
        CHECK(std::isfinite(v));
    }
}

// ---------------------------------------------------------------------------
// Views, workspaces, degenerate input
// ---------------------------------------------------------------------------

// A [frames, bins] layout must give the same answer as [bins, frames] through
// nothing but different strides — no transposition copy anywhere.
TEST_CASE("istft_accepts_either_spectrum_layout_through_strides") {
    const IstftPlan plan = make_plan(128, 32);
    const Spectrum column_major = random_spectrum(plan.bins(), 30, 0x2020u);

    // Transpose into [frames][bins].
    Spectrum row_major;
    row_major.bins = column_major.bins;
    row_major.frames = column_major.frames;
    row_major.real.resize(column_major.real.size());
    row_major.imag.resize(column_major.imag.size());
    for (size_t k = 0; k < column_major.bins; ++k) {
        for (size_t f = 0; f < column_major.frames; ++f) {
            row_major.real[f * column_major.bins + k] =
                column_major.real[k * column_major.frames + f];
            row_major.imag[f * column_major.bins + k] =
                column_major.imag[k * column_major.frames + f];
        }
    }

    ComplexSpectrumView v;
    v.real = row_major.real.data();
    v.imag = row_major.imag.data();
    v.bins = row_major.bins;
    v.frames = row_major.frames;
    v.bin_stride = 1;
    v.frame_stride = static_cast<ptrdiff_t>(row_major.bins);

    IstftWorkspace ws_a;
    IstftWorkspace ws_b;
    std::vector<float> from_columns;
    std::vector<float> from_rows;
    std::string error;
    REQUIRE(kokopop::istft(plan, ws_a, column_major.view(), from_columns, error));
    REQUIRE(kokopop::istft(plan, ws_b, v, from_rows, error));
    CHECK_EQ(from_columns, from_rows);
}

TEST_CASE("istft_rejects_a_spectrum_with_the_wrong_bin_count") {
    const IstftPlan plan = make_plan(1024, 256);
    Spectrum s = random_spectrum(plan.bins(), 4, 1u);
    ComplexSpectrumView v = s.view();
    v.bins = 512;  // one short

    IstftWorkspace ws;
    std::vector<float> out;
    std::string error;
    CHECK_FALSE(kokopop::istft(plan, ws, v, out, error));
    CHECK(error.find("bins") != std::string::npos);
}

TEST_CASE("istft_handles_zero_and_one_frame") {
    const IstftPlan plan = make_plan(1024, 256);
    IstftWorkspace ws;
    std::vector<float> out{1.0f, 2.0f};
    std::string error;

    ComplexSpectrumView empty;
    empty.bins = plan.bins();
    empty.frames = 0;
    CHECK(kokopop::istft(plan, ws, empty, out, error));
    CHECK(out.empty());

    // A single centred frame reconstructs no samples at all, and must not read
    // out of bounds getting there.
    const Spectrum one = random_spectrum(plan.bins(), 1, 3u);
    CHECK(kokopop::istft(plan, ws, one.view(), out, error));
    CHECK(out.empty());
}

TEST_CASE("istft_rejects_null_spectrum_buffers") {
    const IstftPlan plan = make_plan(64, 16);
    ComplexSpectrumView v;
    v.bins = plan.bins();
    v.frames = 4;
    v.real = nullptr;
    v.imag = nullptr;

    IstftWorkspace ws;
    std::vector<float> out;
    std::string error;
    CHECK_FALSE(kokopop::istft(plan, ws, v, out, error));
}

// One plan, several workspaces: the plan holds no mutable state, so two
// concurrent sessions can share it. Reusing one workspace across calls must
// also not leak state from the previous call.
TEST_CASE("istft_plan_is_reusable_and_workspace_state_does_not_leak") {
    const IstftPlan plan = make_plan(256, 64);
    const Spectrum a = random_spectrum(plan.bins(), 15, 11u);
    const Spectrum b = random_spectrum(plan.bins(), 40, 12u);

    IstftWorkspace fresh_a;
    IstftWorkspace fresh_b;
    std::vector<float> want_a;
    std::vector<float> want_b;
    std::string error;
    REQUIRE(kokopop::istft(plan, fresh_a, a.view(), want_a, error));
    REQUIRE(kokopop::istft(plan, fresh_b, b.view(), want_b, error));

    // Same plan, one shared workspace, in the other order.
    IstftWorkspace shared;
    std::vector<float> got_b;
    std::vector<float> got_a;
    REQUIRE(kokopop::istft(plan, shared, b.view(), got_b, error));
    REQUIRE(kokopop::istft(plan, shared, a.view(), got_a, error));
    CHECK_EQ(got_a, want_a);
    CHECK_EQ(got_b, want_b);
}

// The workspace must stay O(n_fft): a per-frame allocation, or a buffer sized
// by the frame count, would defeat the point of it being reusable.
TEST_CASE("istft_workspace_stays_proportional_to_n_fft") {
    const IstftPlan plan = make_plan(1024, 256);
    const Spectrum s = random_spectrum(plan.bins(), 2000, 13u);

    IstftWorkspace ws;
    std::vector<float> out;
    std::string error;
    REQUIRE(kokopop::istft(plan, ws, s.view(), out, error));
    REQUIRE_EQ(out.size(), 1999u * 256u);

    const size_t n = plan.config().n_fft;
    CHECK(ws.frame.size() <= n);
    CHECK(ws.ola.size() <= n);
    CHECK(ws.envelope.size() <= n);
    CHECK(ws.spec_real.size() <= n);
    CHECK(ws.spec_imag.size() <= n);
}
