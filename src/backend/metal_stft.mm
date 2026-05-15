#include "metal_stft.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

// -----------------------------------------------------------------------------
// MSL shader source
// -----------------------------------------------------------------------------
static const char * kStftShaderSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

static inline int kokopop_reflect_index(int idx, int n_samples) {
    if (idx < 0) return -idx - 1;
    if (idx >= n_samples) return 2 * n_samples - idx - 1;
    return idx;
}

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
    const uint k = gid.x;
    const uint frame = gid.y;
    const uint n_bins = n_fft / 2u + 1u;

    if (k >= n_bins || frame >= n_frames) {
        return;
    }

    const device float * twc = tw_c + k * n_fft;
    const device float * tws = tw_s + k * n_fft;

    float real_v = 0.0f;
    float imag_v = 0.0f;

    for (uint n = 0; n < n_fft; ++n) {
        const int src_i = kokopop_reflect_index(
            int(frame * hop + n) - int(center_pad),
            int(n_samples));
        const float s = (src_i >= 0 && src_i < int(n_samples))
            ? source[uint(src_i)] * window_buf[n]
            : 0.0f;

        real_v += s * twc[n];
        imag_v -= s * tws[n];
    }

    const uint out_mag = k * n_frames + frame;
    const uint out_phase = (k + n_bins) * n_frames + frame;

    har_data[out_mag] = sqrt(real_v * real_v + imag_v * imag_v);
    har_data[out_phase] = atan2(imag_v, real_v);
}

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
    const uint n = gid.y;

    if (frame >= n_frames || n >= n_fft) {
        return;
    }

    const uint n_half = n_fft / 2u;
    const uint n_bins = n_half + 1u;

    // Keep precise math here. This stage reconstructs waveform samples and is
    // more sensitive to small numerical changes than the forward DFT.
    const float mag0 = precise::exp(clamp(post[0u * n_frames + frame], -20.0f, 8.0f));
    const float ph0 = precise::sin(post[n_bins * n_frames + frame]);
    float sample = mag0 * precise::cos(ph0);

    const float magN = precise::exp(clamp(post[n_half * n_frames + frame], -20.0f, 8.0f));
    const float phN = precise::sin(post[(n_half + n_bins) * n_frames + frame]);
    sample += magN * precise::cos(phN) * ((n % 2u == 0u) ? 1.0f : -1.0f);

    for (uint k = 1u; k < n_half; ++k) {
        const float mag = precise::exp(clamp(post[k * n_frames + frame], -20.0f, 8.0f));
        const float ph = precise::sin(post[(k + n_bins) * n_frames + frame]);
        const float real_v = mag * precise::cos(ph);
        const float imag_v = mag * precise::sin(ph);

        sample += 2.0f * (real_v * tw_c[k * n_fft + n] - imag_v * tw_s[k * n_fft + n]);
    }

    sample /= float(n_fft);

    if (!isfinite(sample)) {
        sample = 0.0f;
    }

    const float win = window_buf[n];
    const uint idx = frame * n_fft + n;
    tmp_y[idx] = sample * win;
    tmp_d[idx] = win * win;
}

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
    if (gid >= padded_len) {
        return;
    }

    const int s = int(gid);
    const int nf = int(n_fft);
    const int h = int(hop);
    const int nfr = int(n_frames);

    // Frames whose window covers position s:
    //   f * hop <= s < f * hop + n_fft
    // Avoid negative integer division; truncation around zero would be fragile.
    int f_start = 0;
    if (s >= nf) {
        f_start = (s - nf + h) / h;
    }
    const int f_end = min(nfr - 1, s / h);

    float y_acc = 0.0f;
    float d_acc = 0.0f;

    for (int f = f_start; f <= f_end; ++f) {
        const int n = s - f * h;
        if (n >= 0 && n < nf) {
            const uint idx = uint(f) * n_fft + uint(n);
            y_acc += tmp_y[idx];
            d_acc += tmp_d[idx];
        }
    }

    y_buf[gid] = y_acc;
    denom_buf[gid] = d_acc;
}

kernel void kokopop_istft_norm(
    device float        * out        [[ buffer(0) ]],
    device const float  * y_buf      [[ buffer(1) ]],
    device const float  * denom_buf  [[ buffer(2) ]],
    constant uint       & center_pad [[ buffer(3) ]],
    constant uint       & out_len    [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= out_len) {
        return;
    }

    const uint src = gid + center_pad;
    float value = y_buf[src];

    if (denom_buf[src] > 1e-8f) {
        value /= denom_buf[src];
    }
    if (!isfinite(value)) {
        value = 0.0f;
    }

    out[gid] = clamp(value, -1.0f, 1.0f);
}
)MSL";

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------
struct MetalStftState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;

    id<MTLComputePipelineState> pipeline = nil;
    id<MTLComputePipelineState> pipeline_istft_idft = nil;
    id<MTLComputePipelineState> pipeline_istft_ola = nil;
    id<MTLComputePipelineState> pipeline_istft_norm = nil;

    id<MTLBuffer> window_buf = nil;
    id<MTLBuffer> tw_c_buf = nil;
    id<MTLBuffer> tw_s_buf = nil;

    id<MTLBuffer> source_buf = nil;
    id<MTLBuffer> out_buf = nil;

    id<MTLBuffer> istft_post_buf = nil;
    id<MTLBuffer> istft_tmp_y_buf = nil;
    id<MTLBuffer> istft_tmp_d_buf = nil;
    id<MTLBuffer> istft_y_buf = nil;
    id<MTLBuffer> istft_denom_buf = nil;
    id<MTLBuffer> istft_out_buf = nil;

    int n_fft = 0;
    int hop = 0;
    int n_bins = 0;

    bool valid = false;
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
static id<MTLBuffer> ensure_buf(id<MTLDevice> device, id<MTLBuffer> buffer, size_t need_bytes) {
    if (buffer != nil && [buffer length] >= need_bytes) {
        return buffer;
    }
    return [device newBufferWithLength:need_bytes options:MTLResourceStorageModeShared];
}

static id<MTLComputePipelineState> make_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> lib,
    const char * name
) {
    NSError * err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (fn == nil) {
        std::fprintf(stderr, "[metal_stft] function '%s' not found\n", name);
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
    if (pipeline == nil) {
        std::fprintf(stderr,
                     "[metal_stft] pipeline '%s' error: %s\n",
                     name,
                     err ? [[err localizedDescription] UTF8String] : "unknown");
    }
    return pipeline;
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------
MetalStftState * metal_stft_create(int n_fft, int hop) {
    @autoreleasepool {
        if (n_fft <= 0 || hop <= 0) {
            return nullptr;
        }

        auto * state = new MetalStftState{};
        state->n_fft = n_fft;
        state->hop = hop;
        state->n_bins = n_fft / 2 + 1;

        state->device = MTLCreateSystemDefaultDevice();
        if (state->device == nil) {
            std::fprintf(stderr, "[metal_stft] MTLCreateSystemDefaultDevice failed\n");
            delete state;
            return nullptr;
        }

        state->queue = [state->device newCommandQueue];
        if (state->queue == nil) {
            std::fprintf(stderr, "[metal_stft] newCommandQueue failed\n");
            delete state;
            return nullptr;
        }

        NSError * err = nil;
        NSString * src = [NSString stringWithUTF8String:kStftShaderSrc];
        id<MTLLibrary> lib = [state->device newLibraryWithSource:src options:nil error:&err];
        if (lib == nil) {
            std::fprintf(stderr,
                         "[metal_stft] shader compile error: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            delete state;
            return nullptr;
        }

        state->pipeline = make_pipeline(state->device, lib, "kokopop_stft_dft");
        state->pipeline_istft_idft = make_pipeline(state->device, lib, "kokopop_istft_idft");
        state->pipeline_istft_ola = make_pipeline(state->device, lib, "kokopop_istft_ola");
        state->pipeline_istft_norm = make_pipeline(state->device, lib, "kokopop_istft_norm");

        if (state->pipeline == nil ||
            state->pipeline_istft_idft == nil ||
            state->pipeline_istft_ola == nil ||
            state->pipeline_istft_norm == nil) {
            delete state;
            return nullptr;
        }

        const size_t win_bytes = static_cast<size_t>(n_fft) * sizeof(float);
        state->window_buf = [state->device newBufferWithLength:win_bytes options:MTLResourceStorageModeShared];
        if (state->window_buf == nil) {
            delete state;
            return nullptr;
        }

        float * window = static_cast<float *>([state->window_buf contents]);
        const float inv_N = 1.0f / static_cast<float>(n_fft);
        for (int n = 0; n < n_fft; ++n) {
            window[n] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * n * inv_N);
        }

        const size_t tw_bytes = static_cast<size_t>(state->n_bins) * static_cast<size_t>(n_fft) * sizeof(float);
        state->tw_c_buf = [state->device newBufferWithLength:tw_bytes options:MTLResourceStorageModeShared];
        state->tw_s_buf = [state->device newBufferWithLength:tw_bytes options:MTLResourceStorageModeShared];
        if (state->tw_c_buf == nil || state->tw_s_buf == nil) {
            delete state;
            return nullptr;
        }

        float * tw_c = static_cast<float *>([state->tw_c_buf contents]);
        float * tw_s = static_cast<float *>([state->tw_s_buf contents]);
        for (int k = 0; k < state->n_bins; ++k) {
            for (int n = 0; n < n_fft; ++n) {
                const float angle = 2.0f * static_cast<float>(M_PI) * static_cast<float>(k * n) * inv_N;
                tw_c[k * n_fft + n] = std::cos(angle);
                tw_s[k * n_fft + n] = std::sin(angle);
            }
        }

        state->valid = true;
        return state;
    }
}

void metal_stft_destroy(MetalStftState * state) {
    delete state;
}

void metal_stft_compute(
    MetalStftState * state,
    const float * source,
    float * har_data,
    int n_samples,
    int target_frames
) {
    @autoreleasepool {
        if (state == nullptr ||
            !state->valid ||
            source == nullptr ||
            har_data == nullptr ||
            n_samples <= 0 ||
            target_frames <= 0) {
            return;
        }

        const size_t src_bytes = static_cast<size_t>(n_samples) * sizeof(float);
        const size_t out_bytes = static_cast<size_t>(2 * state->n_bins) *
                                 static_cast<size_t>(target_frames) *
                                 sizeof(float);

        state->source_buf = ensure_buf(state->device, state->source_buf, src_bytes);
        state->out_buf = ensure_buf(state->device, state->out_buf, out_bytes);
        if (state->source_buf == nil || state->out_buf == nil) {
            std::fprintf(stderr, "[metal_stft] scratch buffer allocation failed\n");
            return;
        }

        std::memcpy([state->source_buf contents], source, src_bytes);

        const uint32_t v_nfft = static_cast<uint32_t>(state->n_fft);
        const uint32_t v_hop = static_cast<uint32_t>(state->hop);
        const uint32_t v_center_pad = static_cast<uint32_t>(state->n_fft / 2);
        const uint32_t v_n_samples = static_cast<uint32_t>(n_samples);
        const uint32_t v_n_frames = static_cast<uint32_t>(target_frames);

        id<MTLCommandBuffer> cmd = [state->queue commandBuffer];
        if (cmd == nil) {
            return;
        }

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil) {
            return;
        }

        [enc setComputePipelineState:state->pipeline];
        [enc setBuffer:state->out_buf    offset:0 atIndex:0];
        [enc setBuffer:state->source_buf offset:0 atIndex:1];
        [enc setBuffer:state->window_buf offset:0 atIndex:2];
        [enc setBuffer:state->tw_c_buf   offset:0 atIndex:3];
        [enc setBuffer:state->tw_s_buf   offset:0 atIndex:4];
        [enc setBytes:&v_nfft      length:sizeof(v_nfft)      atIndex:5];
        [enc setBytes:&v_hop       length:sizeof(v_hop)       atIndex:6];
        [enc setBytes:&v_center_pad length:sizeof(v_center_pad) atIndex:7];
        [enc setBytes:&v_n_samples length:sizeof(v_n_samples) atIndex:8];
        [enc setBytes:&v_n_frames  length:sizeof(v_n_frames)  atIndex:9];

        // Threadgroup sized within Metal's 1024-threads/group ceiling.
        // 32×16=512 leaves headroom for the simdgroup scheduler. Kernel does
        // explicit bounds checks so any threadgroup size is safe.
        const NSUInteger tg_x = state->n_bins < 32 ? static_cast<NSUInteger>(state->n_bins) : 32;
        const NSUInteger tg_y = target_frames < 16 ? static_cast<NSUInteger>(target_frames) : 16;
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(state->n_bins),
                                         static_cast<NSUInteger>(target_frames),
                                         1)
       threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr,
                         "[metal_stft] command failed with status=%lu\n",
                         static_cast<unsigned long>([cmd status]));
            return;
        }

        std::memcpy(har_data, [state->out_buf contents], out_bytes);
    }
}

void metal_istft_compute(
    MetalStftState * state,
    const float * post,
    float * output,
    int n_frames,
    int out_len
) {
    @autoreleasepool {
        if (state == nullptr ||
            !state->valid ||
            post == nullptr ||
            output == nullptr ||
            n_frames <= 0 ||
            out_len <= 0) {
            return;
        }

        const int padded_len = state->n_fft + state->hop * (n_frames - 1);
        if (padded_len <= 0) {
            return;
        }

        const size_t post_bytes = static_cast<size_t>(2 * state->n_bins) *
                                  static_cast<size_t>(n_frames) *
                                  sizeof(float);
        const size_t tmp_bytes = static_cast<size_t>(n_frames) *
                                 static_cast<size_t>(state->n_fft) *
                                 sizeof(float);
        const size_t pad_bytes = static_cast<size_t>(padded_len) * sizeof(float);
        const size_t out_bytes = static_cast<size_t>(out_len) * sizeof(float);

        state->istft_post_buf = ensure_buf(state->device, state->istft_post_buf, post_bytes);
        state->istft_tmp_y_buf = ensure_buf(state->device, state->istft_tmp_y_buf, tmp_bytes);
        state->istft_tmp_d_buf = ensure_buf(state->device, state->istft_tmp_d_buf, tmp_bytes);
        state->istft_y_buf = ensure_buf(state->device, state->istft_y_buf, pad_bytes);
        state->istft_denom_buf = ensure_buf(state->device, state->istft_denom_buf, pad_bytes);
        state->istft_out_buf = ensure_buf(state->device, state->istft_out_buf, out_bytes);

        if (state->istft_post_buf == nil ||
            state->istft_tmp_y_buf == nil ||
            state->istft_tmp_d_buf == nil ||
            state->istft_y_buf == nil ||
            state->istft_denom_buf == nil ||
            state->istft_out_buf == nil) {
            std::fprintf(stderr, "[metal_istft] scratch buffer allocation failed\n");
            return;
        }

        std::memcpy([state->istft_post_buf contents], post, post_bytes);

        const uint32_t v_nfft = static_cast<uint32_t>(state->n_fft);
        const uint32_t v_hop = static_cast<uint32_t>(state->hop);
        const uint32_t v_center_pad = static_cast<uint32_t>(state->n_fft / 2);
        const uint32_t v_n_frames = static_cast<uint32_t>(n_frames);
        const uint32_t v_padded_len = static_cast<uint32_t>(padded_len);
        const uint32_t v_out_len = static_cast<uint32_t>(out_len);

        id<MTLCommandBuffer> cmd = [state->queue commandBuffer];
        if (cmd == nil) {
            return;
        }

        // Pass 1: IDFT per frame/sample.
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil) {
                return;
            }

            [enc setComputePipelineState:state->pipeline_istft_idft];
            [enc setBuffer:state->istft_tmp_y_buf offset:0 atIndex:0];
            [enc setBuffer:state->istft_tmp_d_buf offset:0 atIndex:1];
            [enc setBuffer:state->istft_post_buf  offset:0 atIndex:2];
            [enc setBuffer:state->window_buf      offset:0 atIndex:3];
            [enc setBuffer:state->tw_c_buf        offset:0 atIndex:4];
            [enc setBuffer:state->tw_s_buf        offset:0 atIndex:5];
            [enc setBytes:&v_nfft     length:sizeof(v_nfft)     atIndex:6];
            [enc setBytes:&v_n_frames length:sizeof(v_n_frames) atIndex:7];

            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n_frames),
                                             static_cast<NSUInteger>(state->n_fft),
                                             1)
           threadsPerThreadgroup:MTLSizeMake(16, static_cast<NSUInteger>(state->n_fft), 1)];
            [enc endEncoding];
        }

        // Pass 2: overlap-add.
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil) {
                return;
            }

            [enc setComputePipelineState:state->pipeline_istft_ola];
            [enc setBuffer:state->istft_y_buf     offset:0 atIndex:0];
            [enc setBuffer:state->istft_denom_buf offset:0 atIndex:1];
            [enc setBuffer:state->istft_tmp_y_buf offset:0 atIndex:2];
            [enc setBuffer:state->istft_tmp_d_buf offset:0 atIndex:3];
            [enc setBytes:&v_nfft      length:sizeof(v_nfft)      atIndex:4];
            [enc setBytes:&v_hop       length:sizeof(v_hop)       atIndex:5];
            [enc setBytes:&v_n_frames  length:sizeof(v_n_frames)  atIndex:6];
            [enc setBytes:&v_padded_len length:sizeof(v_padded_len) atIndex:7];

            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(padded_len), 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // Pass 3: normalize, trim center padding, clamp.
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil) {
                return;
            }

            [enc setComputePipelineState:state->pipeline_istft_norm];
            [enc setBuffer:state->istft_out_buf   offset:0 atIndex:0];
            [enc setBuffer:state->istft_y_buf     offset:0 atIndex:1];
            [enc setBuffer:state->istft_denom_buf offset:0 atIndex:2];
            [enc setBytes:&v_center_pad length:sizeof(v_center_pad) atIndex:3];
            [enc setBytes:&v_out_len    length:sizeof(v_out_len)    atIndex:4];

            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(out_len), 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr,
                         "[metal_istft] command failed with status=%lu\n",
                         static_cast<unsigned long>([cmd status]));
            return;
        }

        std::memcpy(output, [state->istft_out_buf contents], out_bytes);
    }
}
