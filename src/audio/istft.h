#pragma once

// Inverse short-time Fourier transform.
//
// The iSTFT belongs to neither architecture: Kokoro runs it at n_fft=20 on a
// log-magnitude/phase pair, sanoTTS at n_fft=1024 on a magnitude/phase pair.
// What differs is the *conversion to a complex spectrum*, which stays with the
// caller. What is shared is windowing, the inverse transform, the overlap-add
// and the envelope normalisation, which is what lives here.
//
// Two kernels sit behind one API:
//   * a direct inverse DFT with precomputed tables, for the small non-power-of-
//     two sizes (Kokoro's 20);
//   * a real inverse FFT built on a radix-2 complex IFFT of size n_fft/2, for
//     the power-of-two sizes (sanoTTS's 1024). A naive DFT at that size is
//     ~500x the work and is deliberately not reachable.
//
// A plan is immutable and shareable; a workspace is mutable and must belong to
// one execution context at a time. Nothing here is global mutable state, so
// two synthesis sessions can run concurrently.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kokopop {

struct IstftConfig {
    uint32_t n_fft = 0;
    uint32_t hop = 0;

    /// True when the forward transform was taken with `center=True`, i.e. the
    /// signal was padded by n_fft/2 on both sides. The reconstruction then
    /// drops n_fft/2 samples at each end and yields `(frames - 1) * hop`
    /// samples.
    bool center = true;
};

/// A validated configuration plus its precomputed window and twiddle tables.
///
/// Immutable after `create()`, so one plan can be shared by every session
/// using the same configuration.
class IstftPlan {
public:
    enum class Kernel {
        DirectDft,  ///< tables of size n_fft x (n_fft/2 + 1)
        RealRadix2, ///< complex radix-2 IFFT of size n_fft/2
    };

    /// Validates `config` and builds the tables. This is the only place a
    /// configuration is accepted or rejected.
    ///
    /// Rejects: a zero or odd n_fft, a hop outside (0, n_fft], and any n_fft
    /// that is neither a small size the direct kernel can afford nor a power of
    /// two the radix-2 kernel handles.
    static bool create(const IstftConfig & config, IstftPlan & out,
                       std::string & error);

    const IstftConfig & config() const { return _config; }
    Kernel kernel() const { return _kernel; }

    /// Number of bins a spectrum must carry: n_fft/2 + 1.
    size_t bins() const { return _config.n_fft / 2 + 1; }

    /// Reconstructed length for `frames` frames, 0 when `frames` is 0.
    /// Returns false on arithmetic overflow.
    bool output_samples(size_t frames, size_t & out) const;

    /// The periodic Hann window, [n_fft] values.
    const std::vector<float> & window() const { return _window; }

private:
    friend bool istft(const IstftPlan &, struct IstftWorkspace &,
                      struct ComplexSpectrumView, std::vector<float> &,
                      std::string &);

    IstftConfig _config;
    Kernel _kernel = Kernel::DirectDft;

    std::vector<float> _window;     // [n_fft]
    std::vector<float> _window_sq;  // [n_fft], w^2

    // DirectDft: row-major [n_fft][bins], so the inner k loop is contiguous.
    std::vector<float> _dft_cos;
    std::vector<float> _dft_sin;

    // RealRadix2: bit-reversal permutation and twiddles for the half-size
    // complex IFFT, plus the e^{+2*pi*i*k/n_fft} factors of the real-input
    // recombination.
    std::vector<uint32_t> _bit_reverse;   // [n_fft/2]
    std::vector<float> _half_tw_cos;      // [n_fft/2]
    std::vector<float> _half_tw_sin;      // [n_fft/2]
    std::vector<float> _split_cos;        // [n_fft/2]
    std::vector<float> _split_sin;        // [n_fft/2]
};

/// Reusable scratch, sized O(n_fft) and independent of the frame count.
///
/// Owned by the session or execution context that runs the transform, never
/// shared: `istft()` writes into it.
struct IstftWorkspace {
    std::vector<float> frame;      // [n_fft]  windowed time-domain frame
    std::vector<float> spec_real;  // [n_fft/2]
    std::vector<float> spec_imag;  // [n_fft/2]
    std::vector<float> ola;        // [n_fft]  overlap-add ring
    std::vector<float> envelope;   // [n_fft]  sum of w^2, same ring
};

/// A strided view over a complex half-spectrum.
///
/// The graphs produce `[bins, frames]` with real and imaginary parts in
/// separate buffers; taking strides means no transposition or interleaving copy
/// is needed to feed them in.
struct ComplexSpectrumView {
    const float * real = nullptr;
    const float * imag = nullptr;
    size_t bins = 0;
    size_t frames = 0;
    ptrdiff_t bin_stride = 0;
    ptrdiff_t frame_stride = 0;
};

/// Windowed overlap-add reconstruction.
///
/// `audio` is resized to `plan.output_samples(spectrum.frames)`. Each finished
/// sample is divided by the window energy that actually accumulated over it, so
/// the edges are handled by the same expression as the interior rather than by
/// a special case. No clamping: a caller that needs one applies its own.
bool istft(const IstftPlan & plan, IstftWorkspace & workspace,
           ComplexSpectrumView spectrum, std::vector<float> & audio,
           std::string & error);

} // namespace kokopop
