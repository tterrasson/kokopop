#include "audio/istft.h"

#include <algorithm>
#include <cmath>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace kokopop {
namespace {

/// Largest n_fft the direct O(n_fft * bins) kernel is allowed to take. Above
/// this, only the radix-2 kernel is offered — a 1024-point naive inverse DFT
/// would be ~500x the work of the FFT for the same result.
constexpr uint32_t MAX_DIRECT_DFT_N = 64;

/// Below this window energy a sample is considered uncovered and emitted as
/// zero. Only reachable at the very edges of a short signal.
constexpr float MIN_ENVELOPE = 1e-8f;

bool is_power_of_two(uint32_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}

/// Adds with overflow detection, so a hostile frame count fails loudly instead
/// of wrapping into a small allocation.
bool checked_mul_add(size_t a, size_t b, size_t c, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    const size_t product = a * b;
    if (product > std::numeric_limits<size_t>::max() - c) {
        return false;
    }
    out = product + c;
    return true;
}

} // namespace

bool IstftPlan::create(const IstftConfig & config, IstftPlan & out,
                       std::string & error) {
    const uint32_t n = config.n_fft;
    if (n == 0 || (n % 2) != 0) {
        error = "iSTFT n_fft must be a non-zero even number, got "
              + std::to_string(n);
        return false;
    }
    if (config.hop == 0 || config.hop > n) {
        error = "iSTFT hop must be in (0, n_fft], got " + std::to_string(config.hop)
              + " for n_fft=" + std::to_string(n);
        return false;
    }

    Kernel kernel;
    if (is_power_of_two(n) && n >= MAX_DIRECT_DFT_N) {
        kernel = Kernel::RealRadix2;
    } else if (n <= MAX_DIRECT_DFT_N) {
        kernel = Kernel::DirectDft;
    } else {
        error = "unsupported iSTFT n_fft=" + std::to_string(n)
              + ": sizes above " + std::to_string(MAX_DIRECT_DFT_N)
              + " must be a power of two";
        return false;
    }

    IstftPlan plan;
    plan._config = config;
    plan._kernel = kernel;

    // Periodic Hann, as 0.5 - 0.5*cos(2*pi*i/N) rather than the algebraically
    // identical sin^2(pi*i/N): the two differ in the last bits in float.
    //
    // The angle and the cosine are evaluated in double and only the result is
    // rounded to float. That is what Kokoro's hann_window_20() already did (its
    // `2.0f * M_PI` promotes the whole expression to double), and this table has
    // to stay bit-identical to it or Kokoro's PCM changes.
    plan._window.resize(n);
    plan._window_sq.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        const float w = static_cast<float>(
            0.5 - 0.5 * std::cos(2.0 * M_PI * static_cast<double>(i) /
                                 static_cast<double>(n)));
        plan._window[i] = w;
        plan._window_sq[i] = w * w;
    }

    const uint32_t bins = n / 2 + 1;
    const uint32_t half = n / 2;

    if (kernel == Kernel::DirectDft) {
        plan._dft_cos.resize(static_cast<size_t>(n) * bins);
        plan._dft_sin.resize(static_cast<size_t>(n) * bins);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t k = 0; k < bins; ++k) {
                // Angle in double, rounded to float, then a float cos/sin —
                // again matching Kokoro's existing StftTwiddles exactly.
                const float a = static_cast<float>(
                    2.0 * M_PI * static_cast<double>(k * i) / static_cast<double>(n));
                plan._dft_cos[static_cast<size_t>(i) * bins + k] = std::cos(a);
                plan._dft_sin[static_cast<size_t>(i) * bins + k] = std::sin(a);
            }
        }
    } else {
        // Bit-reversal permutation for the half-size complex IFFT.
        uint32_t log2_half = 0;
        while ((1u << log2_half) < half) {
            ++log2_half;
        }
        plan._bit_reverse.resize(half);
        for (uint32_t i = 0; i < half; ++i) {
            uint32_t r = 0;
            for (uint32_t b = 0; b < log2_half; ++b) {
                r = (r << 1) | ((i >> b) & 1u);
            }
            plan._bit_reverse[i] = r;
        }

        // Inverse-transform twiddles e^{+2*pi*i*j/half}, indexed by j.
        plan._half_tw_cos.resize(half);
        plan._half_tw_sin.resize(half);
        for (uint32_t j = 0; j < half; ++j) {
            const double a = 2.0 * M_PI * static_cast<double>(j) /
                             static_cast<double>(half);
            plan._half_tw_cos[j] = static_cast<float>(std::cos(a));
            plan._half_tw_sin[j] = static_cast<float>(std::sin(a));
        }

        // e^{+2*pi*i*k/n_fft}, the factor that undoes the odd-sample phase
        // shift when splitting the real spectrum into a half-length complex one.
        plan._split_cos.resize(half);
        plan._split_sin.resize(half);
        for (uint32_t k = 0; k < half; ++k) {
            const double a = 2.0 * M_PI * static_cast<double>(k) /
                             static_cast<double>(n);
            plan._split_cos[k] = static_cast<float>(std::cos(a));
            plan._split_sin[k] = static_cast<float>(std::sin(a));
        }
    }

    out = std::move(plan);
    return true;
}

bool IstftPlan::output_samples(size_t frames, size_t & out) const {
    if (frames == 0) {
        out = 0;
        return true;
    }
    // Padded length of the overlap-add signal.
    size_t padded = 0;
    if (!checked_mul_add(_config.hop, frames - 1, _config.n_fft, padded)) {
        return false;
    }
    if (!_config.center) {
        out = padded;
        return true;
    }
    const size_t trim = _config.n_fft;  // n_fft/2 at each end
    out = padded > trim ? padded - trim : 0;
    return true;
}

namespace {

/// In-place radix-2 inverse complex FFT of size `m`, normalised by 1/m.
///
/// `re`/`im` are read in natural order and left in natural order; the
/// bit-reversal is applied on the way in.
void inverse_fft_complex(const std::vector<uint32_t> & bit_reverse,
                         const std::vector<float> & tw_cos,
                         const std::vector<float> & tw_sin,
                         float * re, float * im, uint32_t m) {
    for (uint32_t i = 0; i < m; ++i) {
        const uint32_t j = bit_reverse[i];
        if (j > i) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    for (uint32_t len = 2; len <= m; len <<= 1) {
        const uint32_t half = len >> 1;
        const uint32_t step = m / len;
        for (uint32_t base = 0; base < m; base += len) {
            for (uint32_t k = 0; k < half; ++k) {
                const uint32_t t = k * step;
                const float wc = tw_cos[t];
                const float ws = tw_sin[t];
                const uint32_t a = base + k;
                const uint32_t b = a + half;
                const float br = re[b] * wc - im[b] * ws;
                const float bi = re[b] * ws + im[b] * wc;
                re[b] = re[a] - br;
                im[b] = im[a] - bi;
                re[a] += br;
                im[a] += bi;
            }
        }
    }

    const float scale = 1.0f / static_cast<float>(m);
    for (uint32_t i = 0; i < m; ++i) {
        re[i] *= scale;
        im[i] *= scale;
    }
}

} // namespace

bool istft(const IstftPlan & plan, IstftWorkspace & workspace,
           ComplexSpectrumView spectrum, std::vector<float> & audio,
           std::string & error) {
    const IstftConfig & cfg = plan.config();
    const uint32_t n = cfg.n_fft;
    const uint32_t hop = cfg.hop;
    const size_t bins = plan.bins();

    if (spectrum.bins != bins) {
        error = "iSTFT spectrum has " + std::to_string(spectrum.bins)
              + " bins, expected " + std::to_string(bins);
        return false;
    }
    if (spectrum.frames == 0) {
        audio.clear();
        return true;
    }
    if (spectrum.real == nullptr || spectrum.imag == nullptr) {
        error = "iSTFT spectrum has a null real or imaginary buffer";
        return false;
    }

    size_t out_len = 0;
    if (!plan.output_samples(spectrum.frames, out_len)) {
        error = "iSTFT output length overflows for "
              + std::to_string(spectrum.frames) + " frames";
        return false;
    }
    size_t padded_len = 0;
    if (!checked_mul_add(hop, spectrum.frames - 1, n, padded_len)) {
        error = "iSTFT padded length overflows";
        return false;
    }
    audio.assign(out_len, 0.0f);
    if (out_len == 0) {
        return true;
    }

    workspace.frame.assign(n, 0.0f);
    workspace.ola.assign(n, 0.0f);
    workspace.envelope.assign(n, 0.0f);
    const uint32_t half = n / 2;
    if (plan.kernel() == IstftPlan::Kernel::RealRadix2) {
        workspace.spec_real.assign(half, 0.0f);
        workspace.spec_imag.assign(half, 0.0f);
    }

    const size_t trim = cfg.center ? n / 2 : 0;

    // Emit every padded position strictly below `pending`, then free its ring
    // slot. Called before each frame is added, so the live range is always the
    // n_fft positions the current frame writes to.
    size_t emitted = 0;
    const auto flush_until = [&](size_t pending) {
        for (; emitted < pending; ++emitted) {
            const size_t slot = emitted % n;
            if (emitted >= trim) {
                const size_t index = emitted - trim;
                if (index < out_len) {
                    const float env = workspace.envelope[slot];
                    audio[index] = env > MIN_ENVELOPE
                                 ? workspace.ola[slot] / env
                                 : 0.0f;
                }
            }
            workspace.ola[slot] = 0.0f;
            workspace.envelope[slot] = 0.0f;
        }
    };

    for (size_t f = 0; f < spectrum.frames; ++f) {
        const size_t base = static_cast<size_t>(hop) * f;
        flush_until(base);

        const float * re_col = spectrum.real + spectrum.frame_stride * static_cast<ptrdiff_t>(f);
        const float * im_col = spectrum.imag + spectrum.frame_stride * static_cast<ptrdiff_t>(f);
        const auto bin_re = [&](size_t k) { return re_col[spectrum.bin_stride * static_cast<ptrdiff_t>(k)]; };
        const auto bin_im = [&](size_t k) { return im_col[spectrum.bin_stride * static_cast<ptrdiff_t>(k)]; };

        if (plan.kernel() == IstftPlan::Kernel::DirectDft) {
            // x[i] = (1/N) * (X[0] + (-1)^i X[N/2]
            //                 + 2 * sum_{k=1}^{N/2-1} (Re X[k] cos - Im X[k] sin))
            const float dc = bin_re(0);
            const float nyquist = bin_re(bins - 1);
            for (uint32_t i = 0; i < n; ++i) {
                // (-1)^i without a branch. The cast to int matters: the
                // subtraction would otherwise be unsigned and wrap.
                const int sign = 1 - 2 * static_cast<int>(i & 1u);
                float sample = dc + nyquist * static_cast<float>(sign);
                const float * tc = plan._dft_cos.data() + static_cast<size_t>(i) * bins;
                const float * ts = plan._dft_sin.data() + static_cast<size_t>(i) * bins;
                for (uint32_t k = 1; k < half; ++k) {
                    sample += 2.0f * (bin_re(k) * tc[k] - bin_im(k) * ts[k]);
                }
                workspace.frame[i] = sample * (1.0f / static_cast<float>(n));
            }
        } else {
            // Real inverse FFT through a half-length complex one.
            //
            // With E/O the transforms of the even/odd samples,
            //   X[k]         = E[k] + W^k O[k],  W = e^{-2*pi*i/N}
            //   X[k + N/2]   = E[k] - W^k O[k]  = conj(X[N/2 - k])
            // hence E[k] = (X[k] + conj(X[N/2-k])) / 2 and
            //       O[k] = e^{+2*pi*i*k/N} * (X[k] - conj(X[N/2-k])) / 2.
            // Z[k] = E[k] + i*O[k] is the transform of
            // z[m] = x[2m] + i*x[2m+1], so one IFFT of size N/2 yields both
            // sample parities.
            for (uint32_t k = 0; k < half; ++k) {
                const float ar = bin_re(k);
                const float ai = bin_im(k);
                const float br = bin_re(half - k);
                const float bi = -bin_im(half - k);  // conj

                const float er = 0.5f * (ar + br);
                const float ei = 0.5f * (ai + bi);
                const float dr = 0.5f * (ar - br);
                const float di = 0.5f * (ai - bi);

                const float sc = plan._split_cos[k];
                const float ss = plan._split_sin[k];
                const float orr = dr * sc - di * ss;
                const float oi  = dr * ss + di * sc;

                // Z[k] = E[k] + i*O[k]
                workspace.spec_real[k] = er - oi;
                workspace.spec_imag[k] = ei + orr;
            }

            inverse_fft_complex(plan._bit_reverse, plan._half_tw_cos,
                                plan._half_tw_sin, workspace.spec_real.data(),
                                workspace.spec_imag.data(), half);

            for (uint32_t m = 0; m < half; ++m) {
                workspace.frame[2 * m]     = workspace.spec_real[m];
                workspace.frame[2 * m + 1] = workspace.spec_imag[m];
            }
        }

        const float * window = plan._window.data();
        const float * window_sq = plan._window_sq.data();
        for (uint32_t i = 0; i < n; ++i) {
            const size_t slot = (base + i) % n;
            workspace.ola[slot] += workspace.frame[i] * window[i];
            workspace.envelope[slot] += window_sq[i];
        }
    }

    flush_until(padded_len);
    return true;
}

} // namespace kokopop
