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

    // Scratch buffers for the synchronous metal_lstm_run path.
    // Re-used across calls if size fits; reallocated otherwise.
    id<MTLBuffer> pre_gates_buf = nil;
    id<MTLBuffer> b_hh_buf      = nil;
    id<MTLBuffer> output_buf    = nil;

    bool valid = false;
};

// Per in-flight async dispatch (returned as MetalLstmHandle).
struct MetalLstmInflight {
    id<MTLCommandBuffer> cmd;
    id<MTLBuffer>        out_staging; // GPU-shared output buffer
    float*               dst;         // caller's destination pointer
    size_t               out_bytes;
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

// ---------------------------------------------------------------------------
// Internal: encode one LSTM dispatch into a command buffer.
// Does NOT commit. Caller owns cmd lifecycle.
// ---------------------------------------------------------------------------
static bool encode_lstm(
    MetalLstmKernelState   * s,
    id<MTLCommandBuffer>     cmd,
    id<MTLBuffer>            pre_gates_buf,
    id<MTLBuffer>            b_hh_buf,
    id<MTLBuffer>            out_buf,
    id<MTLBuffer>            whh_buf,
    int H, int N, bool reverse)
{
    const uint params_H   = static_cast<uint>(H);
    const uint params_N   = static_cast<uint>(N);
    const uint params_rev = reverse ? 1u : 0u;

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:s->pipeline];
    [enc setBuffer:out_buf       offset:0 atIndex:0];
    [enc setBuffer:pre_gates_buf offset:0 atIndex:1];
    [enc setBuffer:whh_buf       offset:0 atIndex:2];
    [enc setBuffer:b_hh_buf      offset:0 atIndex:3];
    // Scalars < 4 KB go inline into the command buffer via setBytes —
    // no MTLBuffer allocation or memcpy into a persistent buffer needed.
    [enc setBytes:&params_H   length:sizeof(uint) atIndex:4];
    [enc setBytes:&params_N   length:sizeof(uint) atIndex:5];
    [enc setBytes:&params_rev length:sizeof(uint) atIndex:6];

    const NSUInteger tpg = static_cast<NSUInteger>(4 * H);
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    return true;
}

// ---------------------------------------------------------------------------
// Async API: submit without blocking.
// Returns an opaque handle; caller must pass it to metal_lstm_collect().
// Allows forward + backward of the same layer to run concurrently:
//   MetalLstmHandle hf = metal_lstm_submit(s, "fwd", ...);
//   MetalLstmHandle hb = metal_lstm_submit(s, "bwd", ...);
//   metal_lstm_collect(hf);  // waits once for both — Metal overlaps them
//   metal_lstm_collect(hb);
// ---------------------------------------------------------------------------
MetalLstmHandle metal_lstm_submit(
    MetalLstmKernelState * s,
    const char * whh_key,
    const float * pre_gates,
    const float * b_hh,
    float       * output,
    int H, int N, bool reverse)
{
    if (!s || !s->valid) return nullptr;

    auto it = s->whh_buffers.find(whh_key);
    if (it == s->whh_buffers.end()) {
        fprintf(stderr, "[metal_lstm] W_hh not preloaded for key: %s\n", whh_key);
        return nullptr;
    }

    const size_t pg_bytes  = static_cast<size_t>(4 * H * N) * sizeof(float);
    const size_t bhh_bytes = static_cast<size_t>(4 * H)     * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(H * N)      * sizeof(float);

    // Each in-flight call owns its buffers so parallel calls don't alias.
    id<MTLBuffer> pg_buf  = [s->device newBufferWithLength:pg_bytes
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> bhh_buf = [s->device newBufferWithLength:bhh_bytes
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf = [s->device newBufferWithLength:out_bytes
                                                    options:MTLResourceStorageModeShared];
    if (!pg_buf || !bhh_buf || !out_buf) {
        fprintf(stderr, "[metal_lstm] buffer alloc failed in submit\n");
        return nullptr;
    }

    memcpy([pg_buf  contents], pre_gates, pg_bytes);
    memcpy([bhh_buf contents], b_hh,      bhh_bytes);

    // retainedReferences:NO — Metal won't hold the encoder/library beyond endEncoding,
    // reducing memory pressure when many command buffers are in flight.
    MTLCommandBufferDescriptor * desc = [MTLCommandBufferDescriptor new];
    desc.retainedReferences = NO;
    id<MTLCommandBuffer> cmd = [s->queue commandBufferWithDescriptor:desc];

    encode_lstm(s, cmd, pg_buf, bhh_buf, out_buf, it->second, H, N, reverse);

    auto * inflight     = new MetalLstmInflight{};
    inflight->cmd        = cmd;
    inflight->out_staging = out_buf;
    inflight->dst        = output;
    inflight->out_bytes  = out_bytes;

    [cmd commit];
    return inflight;
}

void metal_lstm_collect(MetalLstmHandle handle)
{
    if (!handle) return;
    auto * inflight = static_cast<MetalLstmInflight *>(handle);
    [inflight->cmd waitUntilCompleted];
    memcpy(inflight->dst, [inflight->out_staging contents], inflight->out_bytes);
    delete inflight;
}

// ---------------------------------------------------------------------------
// Synchronous convenience wrapper (unchanged call-site).
// Uses persistent scratch buffers to avoid per-call MTLBuffer allocation.
// ---------------------------------------------------------------------------
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

    const size_t pg_bytes  = static_cast<size_t>(4 * H * N) * sizeof(float);
    const size_t bhh_bytes = static_cast<size_t>(4 * H)     * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(H * N)      * sizeof(float);

    s->pre_gates_buf = ensure_buf(s->device, s->pre_gates_buf, pg_bytes);
    s->b_hh_buf      = ensure_buf(s->device, s->b_hh_buf,      bhh_bytes);
    s->output_buf    = ensure_buf(s->device, s->output_buf,    out_bytes);

    if (!s->pre_gates_buf || !s->b_hh_buf || !s->output_buf) {
        fprintf(stderr, "[metal_lstm] scratch buffer allocation failed\n");
        return;
    }

    memcpy([s->pre_gates_buf contents], pre_gates, pg_bytes);
    memcpy([s->b_hh_buf      contents], b_hh,      bhh_bytes);

    MTLCommandBufferDescriptor * desc = [MTLCommandBufferDescriptor new];
    desc.retainedReferences = NO;
    id<MTLCommandBuffer> cmd = [s->queue commandBufferWithDescriptor:desc];

    encode_lstm(s, cmd, s->pre_gates_buf, s->b_hh_buf, s->output_buf,
                it->second, H, N, reverse);

    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(output, [s->output_buf contents], out_bytes);
}
