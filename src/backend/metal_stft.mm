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
// MSL shader source — STFT + ISTFT kernels
// ---------------------------------------------------------------------------
static const char * kStftShaderSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// ---- Forward STFT ----

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
    const uint k     = gid.x;   // 0 .. n_fft/2
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

// ---- Inverse STFT — Pass 1: IDFT per (frame, sample-index n) ----
//
// Each thread computes one weighted IDFT sample within its frame and stores
// it in tmp_y[frame * n_fft + n].  tmp_d stores the matching window^2 value
// for the denominator accumulation in the OLA pass.
//
// Grid: [n_frames, n_fft]  —  gid.x = frame, gid.y = n
//
// Buffer layout:
//   0: tmp_y   [n_frames * n_fft]  (write)
//   1: tmp_d   [n_frames * n_fft]  (write)
//   2: post    [22 * n_frames]     (read)
//   3: window  [n_fft]             (read)
//   4: tw_c    [n_bins * n_fft]    (read, shared with STFT state)
//   5: tw_s    [n_bins * n_fft]    (read)
//   6: n_fft   (constant uint)
//   7: n_frames (constant uint)
kernel void kokopop_istft_idft(
    device float        * tmp_y      [[ buffer(0) ]],
    device float        * tmp_d      [[ buffer(1) ]],
    device const float  * post       [[ buffer(2) ]],
    device const float  * window_buf [[ buffer(3) ]],
    device const float  * tw_c       [[ buffer(4) ]],
    device const float  * tw_s       [[ buffer(5) ]],
    constant uint       & n_fft      [[ buffer(6) ]],
    constant uint       & n_frames   [[ buffer(7) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    const uint frame = gid.x;
    const uint n     = gid.y;
    if (frame >= n_frames || n >= n_fft) return;

    const uint n_half = n_fft / 2u;

    // Decode k=0 (DC)
    float mag0 = exp(clamp(post[0u * n_frames + frame], -20.0f, 8.0f));
    float ph0  = sin(post[(n_half + 1u) * n_frames + frame]);
    float sample = mag0 * cos(ph0);

    // Decode k=n_fft/2 (Nyquist)
    float magN = exp(clamp(post[n_half * n_frames + frame], -20.0f, 8.0f));
    float phN  = sin(post[(n_half + n_half + 1u) * n_frames + frame]);
    sample += magN * cos(phN) * ((n % 2u == 0u) ? 1.0f : -1.0f);

    // Decode k=1..n_fft/2-1
    for (uint k = 1u; k < n_half; ++k) {
        float mag = exp(clamp(post[k * n_frames + frame], -20.0f, 8.0f));
        float ph  = sin(post[(k + n_half + 1u) * n_frames + frame]);
        float r   = mag * cos(ph);
        float im  = mag * sin(ph);
        sample += 2.0f * (r * tw_c[k * n_fft + n] - im * tw_s[k * n_fft + n]);
    }
    sample /= (float)n_fft;

    const float win = window_buf[n];
    const uint  idx = frame * n_fft + n;
    tmp_y[idx] = sample * win;
    tmp_d[idx] = win * win;
}

// ---- Inverse STFT — Pass 2: overlap-add ----
//
// Each thread owns one output position s in [0, padded_len).  It iterates
// over the (at most n_fft/hop ≈ 4) frames that overlap at s, accumulating
// the weighted samples from tmp_y.  No atomics — each output position has
// exactly one writer.
//
// Grid: [padded_len]
//
// Buffer layout:
//   0: y_buf      [padded_len]       (write)
//   1: denom_buf  [padded_len]       (write)
//   2: tmp_y      [n_frames * n_fft] (read)
//   3: tmp_d      [n_frames * n_fft] (read)
//   4: n_fft      (constant uint)
//   5: hop        (constant uint)
//   6: n_frames   (constant uint)
//   7: padded_len (constant uint)
kernel void kokopop_istft_ola(
    device float        * y_buf      [[ buffer(0) ]],
    device float        * denom_buf  [[ buffer(1) ]],
    device const float  * tmp_y      [[ buffer(2) ]],
    device const float  * tmp_d      [[ buffer(3) ]],
    constant uint       & n_fft      [[ buffer(4) ]],
    constant uint       & hop        [[ buffer(5) ]],
    constant uint       & n_frames   [[ buffer(6) ]],
    constant uint       & padded_len [[ buffer(7) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= padded_len) return;

    const int s = (int)gid;
    const int nf  = (int)n_fft;
    const int h   = (int)hop;
    const int nfr = (int)n_frames;

    // Frames whose window covers position s: f*hop <= s < f*hop + n_fft
    const int f_start = max(0, (s - nf + 1) / h);
    const int f_end   = min(nfr - 1, s / h);

    float y_acc = 0.0f;
    float d_acc = 0.0f;
    for (int f = f_start; f <= f_end; ++f) {
        const int n = s - f * h;
        if (n >= 0 && n < nf) {
            const uint idx = (uint)f * n_fft + (uint)n;
            y_acc += tmp_y[idx];
            d_acc += tmp_d[idx];
        }
    }
    y_buf[gid]     = y_acc;
    denom_buf[gid] = d_acc;
}

// ---- Inverse STFT — Pass 3: normalize and clamp ----
//
// Trims the center-padded margins, divides by the window denominator, and
// clamps to [-1, 1].
//
// Grid: [out_len]
//
// Buffer layout:
//   0: out        [out_len]   (write)
//   1: y_buf      [padded_len](read)
//   2: denom_buf  [padded_len](read)
//   3: center_pad (constant uint)
//   4: out_len    (constant uint)
kernel void kokopop_istft_norm(
    device float        * out        [[ buffer(0) ]],
    device const float  * y_buf      [[ buffer(1) ]],
    device const float  * denom_buf  [[ buffer(2) ]],
    constant uint       & center_pad [[ buffer(3) ]],
    constant uint       & out_len    [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= out_len) return;
    const uint src = gid + center_pad;
    float v = y_buf[src];
    if (denom_buf[src] > 1e-8f) v /= denom_buf[src];
    out[gid] = clamp(v, -1.0f, 1.0f);
}
)MSL";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
struct MetalStftState {
    id<MTLDevice>               device      = nil;
    id<MTLCommandQueue>         queue       = nil;

    // STFT pipeline
    id<MTLComputePipelineState> pipeline    = nil;

    // ISTFT pipelines (3 passes)
    id<MTLComputePipelineState> pipeline_istft_idft = nil;
    id<MTLComputePipelineState> pipeline_istft_ola  = nil;
    id<MTLComputePipelineState> pipeline_istft_norm = nil;

    // Preloaded constant buffers (created once, shared by STFT + ISTFT).
    id<MTLBuffer> window_buf     = nil;  // Hann window [n_fft]
    id<MTLBuffer> tw_c_buf       = nil;  // cos twiddles [n_bins * n_fft]
    id<MTLBuffer> tw_s_buf       = nil;  // sin twiddles [n_bins * n_fft]
    id<MTLBuffer> n_fft_buf      = nil;
    id<MTLBuffer> hop_buf        = nil;
    id<MTLBuffer> center_pad_buf = nil;

    // STFT per-call scalar buffers.
    id<MTLBuffer> n_samples_buf  = nil;
    id<MTLBuffer> n_frames_buf   = nil;

    // STFT scratch buffers (grown as needed).
    id<MTLBuffer> source_buf     = nil;
    id<MTLBuffer> out_buf        = nil;

    // ISTFT per-call scalar buffers.
    id<MTLBuffer> istft_nf_buf   = nil;  // n_frames
    id<MTLBuffer> istft_pad_buf  = nil;  // padded_len
    id<MTLBuffer> istft_ol_buf   = nil;  // out_len

    // ISTFT scratch buffers (grown as needed).
    id<MTLBuffer> istft_post_buf   = nil;  // [22 * n_frames]      input
    id<MTLBuffer> istft_tmp_y_buf  = nil;  // [n_frames * n_fft]   weighted IDFT samples
    id<MTLBuffer> istft_tmp_d_buf  = nil;  // [n_frames * n_fft]   window^2
    id<MTLBuffer> istft_y_buf      = nil;  // [padded_len]          OLA accumulator
    id<MTLBuffer> istft_denom_buf  = nil;  // [padded_len]          window denominator
    id<MTLBuffer> istft_out_buf    = nil;  // [out_len]             final output

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

    // ISTFT pipelines
    static const char * kIstftFunctions[3] = {
        "kokopop_istft_idft", "kokopop_istft_ola", "kokopop_istft_norm"
    };
    id<MTLComputePipelineState> * istft_pipelines[3] = {
        &s->pipeline_istft_idft, &s->pipeline_istft_ola, &s->pipeline_istft_norm
    };
    for (int i = 0; i < 3; ++i) {
        id<MTLFunction> f = [lib newFunctionWithName:
                                [NSString stringWithUTF8String:kIstftFunctions[i]]];
        if (!f) {
            fprintf(stderr, "[metal_stft] function '%s' not found\n", kIstftFunctions[i]);
            delete s;
            return nullptr;
        }
        *istft_pipelines[i] = [s->device newComputePipelineStateWithFunction:f error:&err];
        if (!*istft_pipelines[i]) {
            fprintf(stderr, "[metal_stft] ISTFT pipeline[%d] error: %s\n",
                    i, [[err localizedDescription] UTF8String]);
            delete s;
            return nullptr;
        }
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

    // ISTFT per-call scalar buffers (values written at dispatch time).
    s->istft_nf_buf  = [s->device newBufferWithLength:scalar_len options:MTLResourceStorageModeShared];
    s->istft_pad_buf = [s->device newBufferWithLength:scalar_len options:MTLResourceStorageModeShared];
    s->istft_ol_buf  = [s->device newBufferWithLength:scalar_len options:MTLResourceStorageModeShared];
    if (!s->istft_nf_buf || !s->istft_pad_buf || !s->istft_ol_buf) {
        fprintf(stderr, "[metal_stft] failed to allocate ISTFT scalar buffers\n");
        delete s;
        return nullptr;
    }

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

void metal_istft_compute(
    MetalStftState * s,
    const float    * post,
    float          * output,
    int              n_frames,
    int              out_len)
{
    if (!s || !s->valid || n_frames <= 0 || out_len <= 0) return;

    const int padded_len = s->n_fft + s->hop * (n_frames - 1);

    // Ensure scratch buffers are large enough.
    const size_t post_bytes = static_cast<size_t>(22 * n_frames) * sizeof(float);
    const size_t tmp_bytes  = static_cast<size_t>(n_frames * s->n_fft) * sizeof(float);
    const size_t pad_bytes  = static_cast<size_t>(padded_len) * sizeof(float);
    const size_t out_bytes  = static_cast<size_t>(out_len) * sizeof(float);

    s->istft_post_buf  = ensure_buf(s->device, s->istft_post_buf,  post_bytes);
    s->istft_tmp_y_buf = ensure_buf(s->device, s->istft_tmp_y_buf, tmp_bytes);
    s->istft_tmp_d_buf = ensure_buf(s->device, s->istft_tmp_d_buf, tmp_bytes);
    s->istft_y_buf     = ensure_buf(s->device, s->istft_y_buf,     pad_bytes);
    s->istft_denom_buf = ensure_buf(s->device, s->istft_denom_buf, pad_bytes);
    s->istft_out_buf   = ensure_buf(s->device, s->istft_out_buf,   out_bytes);
    if (!s->istft_post_buf || !s->istft_tmp_y_buf || !s->istft_tmp_d_buf ||
        !s->istft_y_buf    || !s->istft_denom_buf  || !s->istft_out_buf) {
        fprintf(stderr, "[metal_istft] scratch buffer allocation failed\n");
        return;
    }

    // Upload input and per-call scalars.
    memcpy([s->istft_post_buf contents], post, post_bytes);

    const uint32_t v_nf  = static_cast<uint32_t>(n_frames);
    const uint32_t v_pad = static_cast<uint32_t>(padded_len);
    const uint32_t v_ol  = static_cast<uint32_t>(out_len);
    memcpy([s->istft_nf_buf  contents], &v_nf,  sizeof(uint32_t));
    memcpy([s->istft_pad_buf contents], &v_pad, sizeof(uint32_t));
    memcpy([s->istft_ol_buf  contents], &v_ol,  sizeof(uint32_t));

    id<MTLCommandBuffer> cmd = [s->queue commandBuffer];

    // Pass 1 — IDFT: grid [n_frames, n_fft]
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s->pipeline_istft_idft];
        [enc setBuffer:s->istft_tmp_y_buf offset:0 atIndex:0];
        [enc setBuffer:s->istft_tmp_d_buf offset:0 atIndex:1];
        [enc setBuffer:s->istft_post_buf  offset:0 atIndex:2];
        [enc setBuffer:s->window_buf      offset:0 atIndex:3];
        [enc setBuffer:s->tw_c_buf        offset:0 atIndex:4];
        [enc setBuffer:s->tw_s_buf        offset:0 atIndex:5];
        [enc setBuffer:s->n_fft_buf       offset:0 atIndex:6];
        [enc setBuffer:s->istft_nf_buf    offset:0 atIndex:7];

        const NSUInteger tg_x = 16;
        const NSUInteger tg_y = static_cast<NSUInteger>(s->n_fft);
        const MTLSize threads_per_group = MTLSizeMake(tg_x, tg_y, 1);
        const MTLSize threads_per_grid  = MTLSizeMake(
            static_cast<NSUInteger>(n_frames),
            static_cast<NSUInteger>(s->n_fft),
            1);
        [enc dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [enc endEncoding];
    }

    // Pass 2 — Overlap-add: grid [padded_len]
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s->pipeline_istft_ola];
        [enc setBuffer:s->istft_y_buf     offset:0 atIndex:0];
        [enc setBuffer:s->istft_denom_buf offset:0 atIndex:1];
        [enc setBuffer:s->istft_tmp_y_buf offset:0 atIndex:2];
        [enc setBuffer:s->istft_tmp_d_buf offset:0 atIndex:3];
        [enc setBuffer:s->n_fft_buf       offset:0 atIndex:4];
        [enc setBuffer:s->hop_buf         offset:0 atIndex:5];
        [enc setBuffer:s->istft_nf_buf    offset:0 atIndex:6];
        [enc setBuffer:s->istft_pad_buf   offset:0 atIndex:7];

        const NSUInteger tg = 256;
        const MTLSize threads_per_group = MTLSizeMake(tg, 1, 1);
        const MTLSize threads_per_grid  = MTLSizeMake(
            static_cast<NSUInteger>(padded_len), 1, 1);
        [enc dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [enc endEncoding];
    }

    // Pass 3 — Normalize + clamp: grid [out_len]
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s->pipeline_istft_norm];
        [enc setBuffer:s->istft_out_buf    offset:0 atIndex:0];
        [enc setBuffer:s->istft_y_buf      offset:0 atIndex:1];
        [enc setBuffer:s->istft_denom_buf  offset:0 atIndex:2];
        [enc setBuffer:s->center_pad_buf   offset:0 atIndex:3];
        [enc setBuffer:s->istft_ol_buf     offset:0 atIndex:4];

        const NSUInteger tg = 256;
        const MTLSize threads_per_group = MTLSizeMake(tg, 1, 1);
        const MTLSize threads_per_grid  = MTLSizeMake(
            static_cast<NSUInteger>(out_len), 1, 1);
        [enc dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [enc endEncoding];
    }

    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(output, [s->istft_out_buf contents], out_bytes);
}
