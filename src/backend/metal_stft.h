#pragma once

// Metal DFT kernel for the harmonic STFT (Part B of cpu_harmonic_stft).
// Part A (harmonic synthesis) remains on CPU due to its sequential phase
// recurrence. Part B — the DFT over (k, frame) pairs — is perfectly
// parallel and is dispatched here as a 2D Metal kernel.
//
// Only available when KOKOPOP_HAS_METAL is defined.
// Implemented in metal_stft.mm (Objective-C++).

#ifdef KOKOPOP_HAS_METAL

struct MetalStftState;

// Create / destroy.
// n_fft and hop must match KOKOPOP_STFT_N / KOKOPOP_STFT_HOP.
// Returns nullptr on failure.
MetalStftState * metal_stft_create(int n_fft, int hop);
void             metal_stft_destroy(MetalStftState *);

// Compute the magnitude+phase DFT for every (k, frame) pair.
//
//   source    : [n_samples]          float (windowed harmonic waveform,
//                                    output of Part A)
//   har_data  : [22 * target_frames] float — written by this call.
//               Layout mirrors cpu_harmonic_stft:
//                 magnitude : har_data[k * target_frames + frame]
//                             for k = 0 .. n_fft/2
//                 phase     : har_data[(k + n_fft/2 + 1) * target_frames + frame]
//                             for k = 0 .. n_fft/2
void metal_stft_compute(
    MetalStftState * state,
    const float    * source,
    float          * har_data,
    int              n_samples,
    int              target_frames);

#endif // KOKOPOP_HAS_METAL
