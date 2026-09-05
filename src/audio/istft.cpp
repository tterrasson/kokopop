#include "audio/istft.h"

#include "audio/istft_kernels.h"

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
        plan._dft_cos.resize(static_cast<size_t>(bins) * n);
        plan._dft_sin.resize(static_cast<size_t>(bins) * n);
        for (uint32_t k = 0; k < bins; ++k) {
            for (uint32_t i = 0; i < n; ++i) {
                // Angle in double, rounded to float, then a float cos/sin —
                // again matching Kokoro's existing StftTwiddles exactly.
                const float a = static_cast<float>(
                    2.0 * M_PI * static_cast<double>(k * i) / static_cast<double>(n));
                plan._dft_cos[static_cast<size_t>(k) * n + i] = std::cos(a);
                plan._dft_sin[static_cast<size_t>(k) * n + i] = std::sin(a);
            }
        }
    } else {
        // Bit-reversal permutation for the half-size complex IFFT, kept as the
        // swaps it performs rather than as the permutation it comes from: the
        // fixed points and the mirror of each pair are what the loop skipped.
        uint32_t log2_half = 0;
        while ((1u << log2_half) < half) {
            ++log2_half;
        }
        for (uint32_t i = 0; i < half; ++i) {
            uint32_t r = 0;
            for (uint32_t b = 0; b < log2_half; ++b) {
                r = (r << 1) | ((i >> b) & 1u);
            }
            if (r > i) {
                plan._bit_reverse_pairs.push_back(i);
                plan._bit_reverse_pairs.push_back(r);
            }
        }

        // Inverse-transform twiddles e^{+2*pi*i*j/half}, one contiguous run per
        // stage. Stage `h` (h butterflies per block) reads entry k as
        // e^{+2*pi*i*k/(2h)}, which is the j = k * half/(2h) of the flat table
        // this replaces.
        plan._stage_tw_cos.resize(half - 1);
        plan._stage_tw_sin.resize(half - 1);
        for (uint32_t h = 1; h < half; h <<= 1) {
            const uint32_t step = half / (2 * h);
            for (uint32_t k = 0; k < h; ++k) {
                const double a = 2.0 * M_PI * static_cast<double>(k * step) /
                                 static_cast<double>(half);
                plan._stage_tw_cos[h - 1 + k] = static_cast<float>(std::cos(a));
                plan._stage_tw_sin[h - 1 + k] = static_cast<float>(std::sin(a));
            }
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

        // The kernel writes the even samples and the odd samples as two
        // separate arrays, so it wants the window in the same shape.
        plan._window_even.resize(half);
        plan._window_odd.resize(half);
        for (uint32_t m = 0; m < half; ++m) {
            plan._window_even[m] = plan._window[2 * m];
            plan._window_odd[m]  = plan._window[2 * m + 1];
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
    // A bin-major spectrum is read one bin per sample; gathering the frame's
    // bins first turns n_fft strided reads per bin into one.
    const bool gathered = spectrum.bin_stride != 1;
    if (gathered) {
        workspace.bin_real.assign(bins, 0.0f);
        workspace.bin_imag.assign(bins, 0.0f);
    }

    float * const frame = workspace.frame.data();
    float * const ola = workspace.ola.data();
    float * const envelope = workspace.envelope.data();
    const float * const window = plan._window.data();
    const float * const window_sq = plan._window_sq.data();

    const size_t trim = cfg.center ? n / 2 : 0;

    // Emit every padded position strictly below `pending`, then free its ring
    // slot. Called before each frame is added, so the live range is always the
    // n_fft positions the current frame writes to.
    //
    // One run per pass through the ring rather than one iteration per sample:
    // `emitted` advances over a contiguous slice of the ring, and the slice
    // that also lands inside the output is a contiguous slice of `audio`.
    size_t emitted = 0;
    const auto flush_until = [&](size_t pending) {
        while (emitted < pending) {
            const size_t slot = emitted % n;
            const size_t run = std::min(pending - emitted, static_cast<size_t>(n) - slot);

            const size_t begin = std::max(emitted, trim);
            const size_t end = std::min(emitted + run, trim + out_len);
            if (begin < end) {
                const size_t offset = begin - emitted;
                istft_kernel::normalise(audio.data() + (begin - trim),
                                        ola + slot + offset,
                                        envelope + slot + offset,
                                        end - begin, MIN_ENVELOPE);
            }
            std::fill_n(ola + slot, run, 0.0f);
            std::fill_n(envelope + slot, run, 0.0f);
            emitted += run;
        }
    };

    for (size_t f = 0; f < spectrum.frames; ++f) {
        const size_t base = static_cast<size_t>(hop) * f;
        flush_until(base);

        const float * re_col = spectrum.real + spectrum.frame_stride * static_cast<ptrdiff_t>(f);
        const float * im_col = spectrum.imag + spectrum.frame_stride * static_cast<ptrdiff_t>(f);
        if (gathered) {
            istft_kernel::gather_strided(workspace.bin_real.data(), re_col,
                                         spectrum.bin_stride, bins);
            istft_kernel::gather_strided(workspace.bin_imag.data(), im_col,
                                         spectrum.bin_stride, bins);
            re_col = workspace.bin_real.data();
            im_col = workspace.bin_imag.data();
        }

        if (plan.kernel() == IstftPlan::Kernel::DirectDft) {
            // x[i] = (1/N) * (X[0] + (-1)^i X[N/2]
            //                 + 2 * sum_{k=1}^{N/2-1} (Re X[k] cos - Im X[k] sin))
            //
            // Summed bin by bin over the whole frame instead of sample by
            // sample: the tables are [bin][sample], so each bin costs one
            // contiguous pass, and every sample still adds its bins in
            // ascending order — the same additions in the same order as the
            // sample-major form.
            const float dc = re_col[0];
            const float nyquist = re_col[bins - 1];
            for (uint32_t i = 0; i < n; i += 2) {
                frame[i]     = dc + nyquist;  // (-1)^i = +1
                frame[i + 1] = dc - nyquist;
            }
            for (uint32_t k = 1; k < half; ++k) {
                // Doubling the bin rather than the sum: a multiplication by two
                // is exact, so this is the same value the inner 2 * (...) gave.
                istft_kernel::accumulate_rotation(
                    frame, 2.0f * re_col[k], plan._dft_cos.data() + static_cast<size_t>(k) * n,
                    2.0f * im_col[k], plan._dft_sin.data() + static_cast<size_t>(k) * n, n);
            }
            istft_kernel::window_scaled(frame, frame, window,
                                        1.0f / static_cast<float>(n), n);
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
            float * const zr = workspace.spec_real.data();
            float * const zi = workspace.spec_imag.data();

            // The 1/half the inverse transform owes rides in on the 1/2 of the
            // split, which saves a pass over the spectrum: half is a power of
            // two, so the scaling is exact.
            istft_kernel::hermitian_split(re_col, im_col, plan._split_cos.data(),
                                          plan._split_sin.data(),
                                          0.5f / static_cast<float>(half),
                                          zr, zi, half);

            const uint32_t * pairs = plan._bit_reverse_pairs.data();
            const size_t swaps = plan._bit_reverse_pairs.size() / 2;
            for (size_t p = 0; p < swaps; ++p) {
                const uint32_t i = pairs[2 * p];
                const uint32_t j = pairs[2 * p + 1];
                std::swap(zr[i], zr[j]);
                std::swap(zi[i], zi[j]);
            }

            istft_kernel::fft_stage_pairs(zr, zi, half);
            for (uint32_t h = 2; h < half; h <<= 1) {
                istft_kernel::fft_stage(zr, zi, plan._stage_tw_cos.data() + (h - 1),
                                        plan._stage_tw_sin.data() + (h - 1), half, h);
            }

            // z[m] = x[2m] + i*x[2m+1]: the real part carries the even samples
            // and the imaginary part the odd ones, each windowed by its own
            // half of the window.
            istft_kernel::interleave_windowed(frame, zr, zi,
                                              plan._window_even.data(),
                                              plan._window_odd.data(), half);
        }

        // Overlap-add into the ring. The frame covers exactly n_fft positions
        // starting at `base`, so it wraps at most once: two contiguous adds,
        // no per-sample modulo.
        const size_t start = base % n;
        const size_t first = n - start;
        istft_kernel::add_inplace(ola + start, frame, first);
        istft_kernel::add_inplace(envelope + start, window_sq, first);
        if (start != 0) {
            istft_kernel::add_inplace(ola, frame + first, start);
            istft_kernel::add_inplace(envelope, window_sq + first, start);
        }
    }

    flush_until(padded_len);
    return true;
}

} // namespace kokopop
