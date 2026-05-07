// Metal DFT kernel for the harmonic STFT.
//
// Replaces the triple loop in cpu_harmonic_stft Part B:
//   for frame in [0, target_frames):       ~5000 frames
//     for k in [0, n_fft/2]:              11 bins
//       for n in [0, n_fft):             20 samples
//         DFT contribution
//
// GPU mapping: one thread per (k, frame).  Each thread runs the 20-step
// inner loop.  Grid = [n_fft/2+1, target_frames] = [11, target_frames].
//
// Twiddle factors and the Hann window are precomputed on the CPU once at
// state creation and uploaded to persistent MTLBuffers.

#include "metal_stft.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// MSL shader source
// ---------------------------------------------------------------------------
static const char * kStftShaderSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void kokopop_stft_dft(
    device float       * har_data    [[ buffer(0) ]],
    device const float * source      [[ buffer(1) ]],
    device const float * window_buf  [[ buffer(2) ]],
    device const float * tw_c        [[ buffer(3) ]],
    device const float * tw_s        [[ buffer(4) ]],
    constant uint      & n_fft       [[ buffer(5) ]],
    constant uint      & hop         [[ buffer(6) ]],
    constant uint      & center_pad  [[ buffer(7) ]],
    constant uint      & n_samples   [[ buffer(8) ]],
    constant uint      & n_frames    [[ buffer(9) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    const uint k     = gid.x;   // 0 .. n_fft/2  (= 0..10 for n_fft=20)
    const uint frame = gid.y;   // 0 .. n_frames-1

    const uint n_bins = n_fft / 2u + 1u;
    if (k >= n_bins || frame >= n_frames) return;

    const device float * twc = tw_c + k * n_fft;
    const device float * tws = tw_s + k * n_fft;

    float real_v = 0.0f;
    float imag_v = 0.0f;
    for (uint n = 0; n < n_fft; ++n) {
        int src_i = (int)(frame * hop + n) - (int)center_pad;
        float s = (src_i >= 0 && (uint)src_i < n_samples)
            ? source[src_i] * window_buf[n]
            : 0.0f;
        real_v += s * twc[n];
        imag_v -= s * tws[n];
    }

    const uint out_mag   = k * n_frames + frame;
    const uint out_phase = (k + n_fft / 2u + 1u) * n_frames + frame;
    har_data[out_mag]   = sqrt(real_v * real_v + imag_v * imag_v);
    har_data[out_phase] = atan2(imag_v, real_v);
}
)MSL";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
struct MetalStftState {
    id<MTLDevice>               device      = nil;
    id<MTLCommandQueue>         queue       = nil;
    id<MTLComputePipelineState> pipeline    = nil;

    // Preloaded constant buffers (created once, never reallocated).
    id<MTLBuffer> window_buf    = nil;  // Hann window [n_fft]
    id<MTLBuffer> tw_c_buf      = nil;  // cos twiddles [n_bins * n_fft]
    id<MTLBuffer> tw_s_buf      = nil;  // sin twiddles [n_bins * n_fft]
    id<MTLBuffer> n_fft_buf     = nil;
    id<MTLBuffer> hop_buf       = nil;
    id<MTLBuffer> center_pad_buf = nil;

    // Per-call scalar buffers (values change, buffers persist).
    id<MTLBuffer> n_samples_buf = nil;
    id<MTLBuffer> n_frames_buf  = nil;

    // Scratch buffers for source and output (grown as needed).
    id<MTLBuffer> source_buf    = nil;
    id<MTLBuffer> out_buf       = nil;

    int n_fft  = 0;
    int hop    = 0;
    int n_bins = 0;  // n_fft/2 + 1

    bool valid = false;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static id<MTLBuffer> ensure_buf(id<MTLDevice> device, id<MTLBuffer> buf,
                                size_t need_bytes) {
    if (buf && [buf length] >= need_bytes) return buf;
    return [device newBufferWithLength:need_bytes
                               options:MTLResourceStorageModeShared];
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
MetalStftState * metal_stft_create(int n_fft, int hop) {
    auto * s = new MetalStftState{};
    s->n_fft  = n_fft;
    s->hop    = hop;
    s->n_bins = n_fft / 2 + 1;

    s->device = MTLCreateSystemDefaultDevice();
    if (!s->device) {
        fprintf(stderr, "[metal_stft] MTLCreateSystemDefaultDevice failed\n");
        delete s;
        return nullptr;
    }

    s->queue = [s->device newCommandQueue];
    if (!s->queue) {
        fprintf(stderr, "[metal_stft] newCommandQueue failed\n");
        delete s;
        return nullptr;
    }

    NSError * err = nil;
    NSString * src = [NSString stringWithUTF8String:kStftShaderSrc];
    id<MTLLibrary> lib = [s->device newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        fprintf(stderr, "[metal_stft] shader compile error: %s\n",
                [[err localizedDescription] UTF8String]);
        delete s;
        return nullptr;
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"kokopop_stft_dft"];
    if (!fn) {
        fprintf(stderr, "[metal_stft] function 'kokopop_stft_dft' not found\n");
        delete s;
        return nullptr;
    }

    s->pipeline = [s->device newComputePipelineStateWithFunction:fn error:&err];
    if (!s->pipeline) {
        fprintf(stderr, "[metal_stft] pipeline error: %s\n",
                [[err localizedDescription] UTF8String]);
        delete s;
        return nullptr;
    }

    // --- Precompute Hann window ---
    {
        const size_t win_bytes = static_cast<size_t>(n_fft) * sizeof(float);
        s->window_buf = [s->device newBufferWithLength:win_bytes
                                               options:MTLResourceStorageModeShared];
        if (!s->window_buf) { delete s; return nullptr; }
        float * wp = static_cast<float *>([s->window_buf contents]);
        const float inv_N = 1.0f / static_cast<float>(n_fft);
        for (int n = 0; n < n_fft; ++n) {
            wp[n] = 0.5f - 0.5f * std::cos(2.0f * M_PI * n * inv_N);
        }
    }

    // --- Precompute twiddle factors ---
    {
        const size_t tw_bytes = static_cast<size_t>(s->n_bins * n_fft) * sizeof(float);
        s->tw_c_buf = [s->device newBufferWithLength:tw_bytes
                                             options:MTLResourceStorageModeShared];
        s->tw_s_buf = [s->device newBufferWithLength:tw_bytes
                                             options:MTLResourceStorageModeShared];
        if (!s->tw_c_buf || !s->tw_s_buf) { delete s; return nullptr; }
        float * cp = static_cast<float *>([s->tw_c_buf contents]);
        float * sp = static_cast<float *>([s->tw_s_buf contents]);
        const float inv_N = 1.0f / static_cast<float>(n_fft);
        for (int k = 0; k < s->n_bins; ++k) {
            for (int n = 0; n < n_fft; ++n) {
                const float a = 2.0f * M_PI * static_cast<float>(k * n) * inv_N;
                cp[k * n_fft + n] = std::cos(a);
                sp[k * n_fft + n] = std::sin(a);
            }
        }
    }

    // --- Fixed scalar buffers ---
    const NSUInteger scalar_len = sizeof(uint32_t);
    s->n_fft_buf      = [s->device newBufferWithLength:scalar_len options:MTLResourceStorageModeShared];
    s->hop_buf        = [s->device newBufferWithLength:scalar_len options:MTLResourceStorageModeShared];
    s->center_pad_buf = [s->device newBufferWithLength:scalar_len options:MTLResourceStorageModeShared];
    s->n_samples_buf  = [s->device newBufferWithLength:scalar_len options:MTLResourceStorageModeShared];
    s->n_frames_buf   = [s->device newBufferWithLength:scalar_len options:MTLResourceStorageModeShared];
    if (!s->n_fft_buf || !s->hop_buf || !s->center_pad_buf ||
        !s->n_samples_buf || !s->n_frames_buf) {
        fprintf(stderr, "[metal_stft] failed to allocate scalar buffers\n");
        delete s;
        return nullptr;
    }
    const uint32_t v_nfft = static_cast<uint32_t>(n_fft);
    const uint32_t v_hop  = static_cast<uint32_t>(hop);
    const uint32_t v_cp   = static_cast<uint32_t>(n_fft / 2);
    memcpy([s->n_fft_buf      contents], &v_nfft, sizeof(uint32_t));
    memcpy([s->hop_buf        contents], &v_hop,  sizeof(uint32_t));
    memcpy([s->center_pad_buf contents], &v_cp,   sizeof(uint32_t));

    s->valid = true;
    return s;
}

void metal_stft_destroy(MetalStftState * s) {
    delete s;
}

void metal_stft_compute(
    MetalStftState * s,
    const float    * source,
    float          * har_data,
    int              n_samples,
    int              target_frames)
{
    if (!s || !s->valid) return;

    const size_t src_bytes = static_cast<size_t>(n_samples)      * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(22 * target_frames) * sizeof(float);

    s->source_buf = ensure_buf(s->device, s->source_buf, src_bytes);
    s->out_buf    = ensure_buf(s->device, s->out_buf,    out_bytes);
    if (!s->source_buf || !s->out_buf) {
        fprintf(stderr, "[metal_stft] scratch buffer allocation failed\n");
        return;
    }

    memcpy([s->source_buf contents], source, src_bytes);

    const uint32_t v_ns = static_cast<uint32_t>(n_samples);
    const uint32_t v_nf = static_cast<uint32_t>(target_frames);
    memcpy([s->n_samples_buf contents], &v_ns, sizeof(uint32_t));
    memcpy([s->n_frames_buf  contents], &v_nf, sizeof(uint32_t));

    id<MTLCommandBuffer>         cmd = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:s->pipeline];
    [enc setBuffer:s->out_buf        offset:0 atIndex:0];
    [enc setBuffer:s->source_buf     offset:0 atIndex:1];
    [enc setBuffer:s->window_buf     offset:0 atIndex:2];
    [enc setBuffer:s->tw_c_buf       offset:0 atIndex:3];
    [enc setBuffer:s->tw_s_buf       offset:0 atIndex:4];
    [enc setBuffer:s->n_fft_buf      offset:0 atIndex:5];
    [enc setBuffer:s->hop_buf        offset:0 atIndex:6];
    [enc setBuffer:s->center_pad_buf offset:0 atIndex:7];
    [enc setBuffer:s->n_samples_buf  offset:0 atIndex:8];
    [enc setBuffer:s->n_frames_buf   offset:0 atIndex:9];

    // One thread per (k, frame). Threadgroup = [n_bins, 32] so that each
    // threadgroup processes all k-bins for 32 consecutive frames.
    const NSUInteger tg_x = static_cast<NSUInteger>(s->n_bins);
    const NSUInteger tg_y = 32;
    const MTLSize threads_per_group = MTLSizeMake(tg_x, tg_y, 1);
    const MTLSize threads_per_grid  = MTLSizeMake(
        static_cast<NSUInteger>(s->n_bins),
        static_cast<NSUInteger>(target_frames),
        1);
    [enc dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(har_data, [s->out_buf contents], out_bytes);
}
