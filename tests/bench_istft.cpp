// Micro-benchmark for the iSTFT component.
//
// Standalone by design: no GGUF, no backend, no phonemizer. It runs the two
// kernels at the sizes the two architectures actually use, with the spectrum
// laid out the way each caller lays it out, so a measurement here maps onto
// real synthesis time.
//
//   kokopop_bench_istft [--seconds S] [--iters N] [--case NAME]
//
// Every case reports, next to the timing, the worst error against a double
// precision inverse DFT over the first few frames and a checksum of the whole
// output. The timing says whether a change is faster; those two say whether it
// is still the same transform.

#include "audio/istft.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using kokopop::ComplexSpectrumView;
using kokopop::IstftConfig;
using kokopop::IstftPlan;
using kokopop::IstftWorkspace;

namespace {

constexpr double PI = 3.14159265358979323846;

/// The two layouts the callers hand in. Neither is a copy of the other: the
/// Kokoro path reads a bin-major tensor with a frame-sized stride, the sanoTTS
/// path a frame-major one that is contiguous over bins.
enum class Layout {
    BinMajor,    ///< [bins][frames], bin_stride = frames — Kokoro
    FrameMajor,  ///< [frames][bins], bin_stride = 1     — sanoTTS vocos
};

/// xorshift64*, so the input is identical on every platform and every standard
/// library. std::mt19937 would be too, but its distributions are not.
class Rng {
public:
    explicit Rng(uint64_t seed) : _state(seed ? seed : 0x9e3779b97f4a7c15ull) {}

    /// Uniform in [-1, 1).
    float next() {
        _state ^= _state >> 12;
        _state ^= _state << 25;
        _state ^= _state >> 27;
        const uint64_t x = _state * 0x2545f4914f6cdd1dull;
        // 24 bits of mantissa, exactly representable.
        const uint32_t bits = static_cast<uint32_t>(x >> 40);
        return static_cast<float>(bits) * (2.0f / 16777216.0f) - 1.0f;
    }

private:
    uint64_t _state;
};

struct Spectrum {
    size_t bins = 0;
    size_t frames = 0;
    Layout layout = Layout::BinMajor;
    std::vector<float> real;
    std::vector<float> imag;

    size_t index(size_t bin, size_t frame) const {
        return layout == Layout::BinMajor ? bin * frames + frame
                                          : frame * bins + bin;
    }

    ComplexSpectrumView view() const {
        ComplexSpectrumView v;
        v.real = real.data();
        v.imag = imag.data();
        v.bins = bins;
        v.frames = frames;
        if (layout == Layout::BinMajor) {
            v.bin_stride = static_cast<ptrdiff_t>(frames);
            v.frame_stride = 1;
        } else {
            v.bin_stride = 1;
            v.frame_stride = static_cast<ptrdiff_t>(bins);
        }
        return v;
    }
};

/// A spectrum with a 1/k magnitude roll-off — closer to speech than white
/// noise, and it keeps the arithmetic away from the denormal range.
Spectrum make_spectrum(size_t bins, size_t frames, Layout layout, uint64_t seed) {
    Spectrum s;
    s.bins = bins;
    s.frames = frames;
    s.layout = layout;
    s.real.assign(bins * frames, 0.0f);
    s.imag.assign(bins * frames, 0.0f);

    Rng rng(seed);
    for (size_t f = 0; f < frames; ++f) {
        for (size_t k = 0; k < bins; ++k) {
            const float rolloff = 1.0f / (1.0f + static_cast<float>(k));
            const size_t i = s.index(k, f);
            s.real[i] = rng.next() * rolloff;
            // A real signal has no imaginary part at DC or Nyquist; a caller
            // that produced one would be feeding a spectrum with no inverse.
            s.imag[i] = (k == 0 || k + 1 == bins) ? 0.0f : rng.next() * rolloff;
        }
    }
    return s;
}

/// Textbook O(N^2) inverse DFT of one Hermitian half-spectrum, in double.
/// The oracle: obviously right rather than fast.
std::vector<double> reference_irfft(const Spectrum & s, size_t frame, size_t n_fft) {
    std::vector<double> out(n_fft, 0.0);
    for (size_t i = 0; i < n_fft; ++i) {
        std::complex<double> acc(0.0, 0.0);
        for (size_t k = 0; k < n_fft; ++k) {
            const size_t m = k < s.bins ? k : n_fft - k;
            const double re = s.real[s.index(m, frame)];
            const double im = s.imag[s.index(m, frame)];
            const std::complex<double> x(re, k < s.bins ? im : -im);
            const double angle =
                2.0 * PI * static_cast<double>(k * i) / static_cast<double>(n_fft);
            acc += x * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        out[i] = acc.real() / static_cast<double>(n_fft);
    }
    return out;
}

/// Reference iSTFT: reference_irfft per frame, flat overlap-add, division by
/// the window energy that actually accumulated.
std::vector<double> reference_istft(const Spectrum & s, const IstftConfig & cfg,
                                    const std::vector<float> & window) {
    const size_t n = cfg.n_fft;
    const size_t padded = n + static_cast<size_t>(cfg.hop) * (s.frames - 1);
    std::vector<double> y(padded, 0.0);
    std::vector<double> env(padded, 0.0);
    for (size_t f = 0; f < s.frames; ++f) {
        const std::vector<double> frame = reference_irfft(s, f, n);
        for (size_t i = 0; i < n; ++i) {
            const size_t p = static_cast<size_t>(cfg.hop) * f + i;
            const double w = window[i];
            y[p] += frame[i] * w;
            env[p] += w * w;
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

/// FNV-1a over the raw bits: any change in any sample changes it. Two builds
/// agreeing here did the exact same arithmetic in the exact same order.
uint64_t checksum(const std::vector<float> & audio) {
    uint64_t h = 1469598103934665603ull;
    for (const float sample : audio) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
            h ^= (bits >> (byte * 8)) & 0xffu;
            h *= 1099511628211ull;
        }
    }
    return h;
}

struct Case {
    const char * name;
    uint32_t n_fft;
    uint32_t hop;
    uint32_t sample_rate;
    Layout layout;
};

/// Worst absolute deviation from the oracle, measured on a prefix short enough
/// for an O(N^2) double transform: the kernel is per frame, so a handful of
/// frames exercises every line of it.
double accuracy(const Case & c, const IstftPlan & plan, size_t frames) {
    const Spectrum s = make_spectrum(plan.bins(), frames, c.layout, 12345);
    IstftWorkspace ws;
    std::vector<float> got;
    std::string error;
    if (!istft(plan, ws, s.view(), got, error)) {
        std::fprintf(stderr, "istft failed: %s\n", error.c_str());
        return -1.0;
    }
    const std::vector<double> want = reference_istft(s, plan.config(), plan.window());
    double worst = 0.0;
    const size_t len = std::min(got.size(), want.size());
    for (size_t i = 0; i < len; ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[i]));
    }
    return worst;
}

void run_case(const Case & c, double seconds, int iters) {
    IstftPlan plan;
    std::string error;
    if (!IstftPlan::create(IstftConfig{c.n_fft, c.hop, /*center=*/true}, plan, error)) {
        std::fprintf(stderr, "%s: %s\n", c.name, error.c_str());
        return;
    }

    const size_t frames = static_cast<size_t>(
        seconds * c.sample_rate / static_cast<double>(c.hop)) + 1;
    const Spectrum s = make_spectrum(plan.bins(), frames, c.layout, 0xa5a5a5a5u);

    IstftWorkspace ws;
    std::vector<float> audio;
    // One untimed run: it settles the workspace and output allocations, which
    // a caller pays once per session and not once per utterance.
    if (!istft(plan, ws, s.view(), audio, error)) {
        std::fprintf(stderr, "%s: %s\n", c.name, error.c_str());
        return;
    }

    std::vector<double> ms;
    ms.reserve(static_cast<size_t>(iters));
    for (int it = 0; it < iters; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!istft(plan, ws, s.view(), audio, error)) {
            std::fprintf(stderr, "%s: %s\n", c.name, error.c_str());
            return;
        }
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double best = ms.front();
    const double median = ms[ms.size() / 2];

    const double audio_seconds =
        static_cast<double>(audio.size()) / static_cast<double>(c.sample_rate);
    // Frames the accuracy check can afford: the oracle is O(n_fft^2) per frame.
    const size_t check_frames = c.n_fft >= 512 ? 8 : 64;

    std::printf("%-22s %7zu %9zu %9.3f %9.3f %11.1f %10.2g  %016llx\n",
                c.name, frames, audio.size(), best, median,
                audio_seconds / (best / 1000.0),
                accuracy(c, plan, check_frames),
                static_cast<unsigned long long>(checksum(audio)));
    std::fflush(stdout);
}

const Case CASES[] = {
    // Kokoro: n_fft=20, hop=5 at 24 kHz, spectrum bin-major with a
    // frame-sized bin stride (src/arch/kokoro/audio_utils.cpp).
    {"kokoro n=20 hop=5", 20, 5, 24000, Layout::BinMajor},
    // sanoTTS vocos: n_fft=1024, hop=256 at 24 kHz, spectrum frame-major and
    // contiguous over bins (src/arch/sanotts/sano_vocos.cpp).
    {"sanotts n=1024 hop=256", 1024, 256, 24000, Layout::FrameMajor},
    // Neither architecture uses these; they are here so a change that only
    // helps one size shows up as such.
    {"radix2 n=256 hop=64", 256, 64, 24000, Layout::FrameMajor},
    {"direct n=32 hop=8", 32, 8, 24000, Layout::BinMajor},
};

} // namespace

int main(int argc, char ** argv) {
    double seconds = 10.0;
    int iters = 20;
    std::string only;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--seconds" && has_value) {
            seconds = std::atof(argv[++i]);
        } else if (arg == "--iters" && has_value) {
            iters = std::atoi(argv[++i]);
        } else if (arg == "--case" && has_value) {
            only = argv[++i];
        } else {
            std::fprintf(stderr,
                         "usage: %s [--seconds S] [--iters N] [--case SUBSTRING]\n",
                         argv[0]);
            return 1;
        }
    }
    if (seconds <= 0.0 || iters <= 0) {
        std::fprintf(stderr, "--seconds and --iters must be positive\n");
        return 1;
    }

    std::printf("iSTFT micro-benchmark — %.1f s of audio per call, %d iterations\n",
                seconds, iters);
#if defined(__AVX2__)
    std::printf("SIMD: AVX2\n");
#elif defined(__ARM_NEON)
    std::printf("SIMD: NEON\n");
#else
    std::printf("SIMD: none (scalar)\n");
#endif
    std::printf("%-22s %7s %9s %9s %9s %11s %10s  %s\n",
                "case", "frames", "samples", "best ms", "med ms", "xRT",
                "max err", "checksum");

    for (const Case & c : CASES) {
        if (!only.empty() && std::string(c.name).find(only) == std::string::npos) {
            continue;
        }
        run_case(c, seconds, iters);
    }
    return 0;
}
