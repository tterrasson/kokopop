// Fused Metal LSTM kernel for kokopop.
//
// One threadgroup per LSTM direction, 4*H = 1024 threads.
// The kernel loops over N time-steps sequentially inside the shader,
// eliminating the per-step graph-node overhead (~18 ggml nodes × N steps).
//
// Tensor layout (ggml column-major, ne[0] is fast / contiguous):
//   pre_gates : float[4*H, N]   element (g,t) = pre_gates[g + 4*H*t]
//   w_hh      : float[H, 4*H]   element (k,j) = w_hh[k + H*j]     (j = gate index)
//   b_hh      : float[4*H]
//   output    : float[H, N]     element (h,t) = output[h + H*t]

#include "metal_lstm.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstring>
#include <cstdio>

// ---------------------------------------------------------------------------
// MSL shader source
// ---------------------------------------------------------------------------
static const char * kLstmShaderSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void kokopop_lstm_fused(
    device float       * output    [[ buffer(0) ]],
    device const float * pre_gates [[ buffer(1) ]],
    device const float * w_hh      [[ buffer(2) ]],
    device const float * b_hh      [[ buffer(3) ]],
    constant uint      & H         [[ buffer(4) ]],
    constant uint      & N         [[ buffer(5) ]],
    constant uint      & rev       [[ buffer(6) ]],
    uint tid [[ thread_position_in_threadgroup ]]
) {
    // threadgroup memory: h (H floats), c (H floats), all gates (4*H floats)
    threadgroup float h_shm[256];
    threadgroup float c_shm[256];
    threadgroup float g_shm[1024];

    if (tid < H) {
        h_shm[tid] = 0.0f;
        c_shm[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0; step < N; ++step) {
        const uint t = rev ? (N - 1u - step) : step;

        // Each thread 'tid' computes one element of the gate vector:
        //   g[tid] = b_hh[tid] + pre_gates[tid + 4*H*t] + dot(w_hh[:,tid], h)
        // w_hh column 'tid' starts at w_hh + tid*H.
        const device float * col = w_hh + tid * H;
        float dot = 0.0f;
        for (uint k = 0; k < H; ++k) {
            dot += col[k] * h_shm[k];
        }
        g_shm[tid] = b_hh[tid] + pre_gates[tid + 4u * H * t] + dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // First H threads update c and h for their hidden unit.
        if (tid < H) {
            float x_i = clamp(g_shm[tid],        -80.0f, 80.0f);
            float x_f = clamp(g_shm[H + tid],    -80.0f, 80.0f);
            float x_g = clamp(g_shm[2u*H + tid], -80.0f, 80.0f);
            float x_o = clamp(g_shm[3u*H + tid], -80.0f, 80.0f);

            float i_gate = 1.0f / (1.0f + precise::exp(-x_i));
            float f_gate = 1.0f / (1.0f + precise::exp(-x_f));
            float g_gate = precise::tanh(x_g);
            float o_gate = 1.0f / (1.0f + precise::exp(-x_o));

            float c_new = f_gate * c_shm[tid] + i_gate * g_gate;
            c_new = clamp(c_new, -50.0f, 50.0f);

            float h_new = o_gate * precise::tanh(c_new);

            c_shm[tid] = c_new;
            h_shm[tid] = h_new;
            output[tid + H * t] = h_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
)MSL";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
struct MetalLstmKernelState {
    id<MTLDevice>              device      = nil;
    id<MTLCommandQueue>        queue       = nil;
    id<MTLComputePipelineState> pipeline   = nil;

    // Pre-loaded W_hh buffers keyed by LSTM direction name.
    std::unordered_map<std::string, id<MTLBuffer>> whh_buffers;

    // Scratch buffers for per-inference copies (pre_gates, b_hh, output).
    // Re-used across calls if size fits; reallocated otherwise.
    id<MTLBuffer> pre_gates_buf = nil;
    id<MTLBuffer> b_hh_buf      = nil;
    id<MTLBuffer> output_buf    = nil;

    // Persistent scalar parameter buffers (4 bytes each, never reallocated).
    id<MTLBuffer> h_buf   = nil;
    id<MTLBuffer> n_buf   = nil;
    id<MTLBuffer> rev_buf = nil;

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
MetalLstmKernelState * metal_lstm_create() {
    auto * s = new MetalLstmKernelState{};

    s->device = MTLCreateSystemDefaultDevice();
    if (!s->device) {
        fprintf(stderr, "[metal_lstm] MTLCreateSystemDefaultDevice failed\n");
        delete s;
        return nullptr;
    }

    s->queue = [s->device newCommandQueue];
    if (!s->queue) {
        fprintf(stderr, "[metal_lstm] newCommandQueue failed\n");
        delete s;
        return nullptr;
    }

    NSError * err = nil;
    NSString * src = [NSString stringWithUTF8String:kLstmShaderSrc];
    id<MTLLibrary> lib = [
        s->device newLibraryWithSource:src options:nil error:&err
    ];

    if (!lib) {
        fprintf(stderr, "[metal_lstm] shader compile error: %s\n",
                [[err localizedDescription] UTF8String]);
        delete s;
        return nullptr;
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"kokopop_lstm_fused"];
    if (!fn) {
        fprintf(stderr, "[metal_lstm] function 'kokopop_lstm_fused' not found\n");
        delete s;
        return nullptr;
    }

    s->pipeline = [s->device newComputePipelineStateWithFunction:fn error:&err];
    if (!s->pipeline) {
        fprintf(stderr, "[metal_lstm] pipeline error: %s\n",
                [[err localizedDescription] UTF8String]);
        delete s;
        return nullptr;
    }

    // Allocate persistent scalar buffers once (H, N, rev — 4 bytes each).
    const NSUInteger scalar_len = sizeof(uint);
    s->h_buf   = [s->device newBufferWithLength:scalar_len options:MTLResourceStorageModeShared];
    s->n_buf   = [s->device newBufferWithLength:scalar_len options:MTLResourceStorageModeShared];
    s->rev_buf = [s->device newBufferWithLength:scalar_len options:MTLResourceStorageModeShared];
    if (!s->h_buf || !s->n_buf || !s->rev_buf) {
        fprintf(stderr, "[metal_lstm] failed to allocate scalar parameter buffers\n");
        delete s;
        return nullptr;
    }

    s->valid = true;
    return s;
}

void metal_lstm_destroy(MetalLstmKernelState * s) {
    delete s;
}

void metal_lstm_preload_whh(
    MetalLstmKernelState * s,
    const char * key,
    const float * w_hh_f32,
    int H, int four_H)
{
    if (!s || !s->valid) return;
    const size_t bytes = static_cast<size_t>(H) * static_cast<size_t>(four_H) * sizeof(float);
    id<MTLBuffer> buf = [s->device newBufferWithBytes:w_hh_f32
                                               length:bytes
                                              options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "[metal_lstm] failed to allocate W_hh buffer for %s\n", key);
        return;
    }
    s->whh_buffers[key] = buf;
}

void metal_lstm_run(
    MetalLstmKernelState * s,
    const char * whh_key,
    const float * pre_gates,
    const float * b_hh,
    float       * output,
    int H, int N, bool reverse)
{
    if (!s || !s->valid) return;

    auto it = s->whh_buffers.find(whh_key);
    if (it == s->whh_buffers.end()) {
        fprintf(stderr, "[metal_lstm] W_hh not preloaded for key: %s\n", whh_key);
        return;
    }
    id<MTLBuffer> whh_buf = it->second;

    const size_t pg_bytes  = static_cast<size_t>(4 * H * N) * sizeof(float);
    const size_t bhh_bytes = static_cast<size_t>(4 * H)     * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(H * N)      * sizeof(float);

    // Get or reallocate scratch buffers
    s->pre_gates_buf = ensure_buf(s->device, s->pre_gates_buf, pg_bytes);
    s->b_hh_buf      = ensure_buf(s->device, s->b_hh_buf,      bhh_bytes);
    s->output_buf    = ensure_buf(s->device, s->output_buf,    out_bytes);

    if (!s->pre_gates_buf || !s->b_hh_buf || !s->output_buf) {
        fprintf(stderr, "[metal_lstm] scratch buffer allocation failed\n");
        return;
    }

    // Copy inputs into Metal buffers
    memcpy([s->pre_gates_buf contents], pre_gates, pg_bytes);
    memcpy([s->b_hh_buf      contents], b_hh,      bhh_bytes);

    // Write scalar params into persistent buffers (avoids per-call allocation).
    const uint params_H   = static_cast<uint>(H);
    const uint params_N   = static_cast<uint>(N);
    const uint params_rev = reverse ? 1u : 0u;
    memcpy([s->h_buf   contents], &params_H,   sizeof(uint));
    memcpy([s->n_buf   contents], &params_N,   sizeof(uint));
    memcpy([s->rev_buf contents], &params_rev, sizeof(uint));

    id<MTLCommandBuffer>        cmd  = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:s->pipeline];
    [enc setBuffer:s->output_buf    offset:0 atIndex:0];
    [enc setBuffer:s->pre_gates_buf offset:0 atIndex:1];
    [enc setBuffer:whh_buf          offset:0 atIndex:2];
    [enc setBuffer:s->b_hh_buf      offset:0 atIndex:3];
    [enc setBuffer:s->h_buf         offset:0 atIndex:4];
    [enc setBuffer:s->n_buf         offset:0 atIndex:5];
    [enc setBuffer:s->rev_buf       offset:0 atIndex:6];

    // One threadgroup, 4*H threads (= 1024 for H=256)
    const NSUInteger threads_per_group = static_cast<NSUInteger>(4 * H);
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    // Copy output back to CPU buffer
    memcpy(output, [s->output_buf contents], out_bytes);
}
