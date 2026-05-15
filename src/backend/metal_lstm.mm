#include "metal_lstm.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <string>
#include <unordered_map>

// -----------------------------------------------------------------------------
// MSL shader source
// -----------------------------------------------------------------------------
static const char * kLstmShaderSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

#define KOKOPOP_LSTM_H 256u
#define KOKOPOP_LSTM_4H 1024u

static inline float kokopop_sanitize_gate(float x) {
    if (!isfinite(x)) return 0.0f;
    return clamp(x, -80.0f, 80.0f);
}

static inline float kokopop_sanitize_state(float x) {
    if (!isfinite(x)) return 0.0f;
    return clamp(x, -50.0f, 50.0f);
}

kernel void kokopop_lstm_fused_h256(
    device float       * output    [[ buffer(0) ]],
    device const float * pre_gates [[ buffer(1) ]],
    device const float * w_hh      [[ buffer(2) ]],
    device const float * b_hh      [[ buffer(3) ]],
    constant uint      & N         [[ buffer(4) ]],
    constant uint      & rev       [[ buffer(5) ]],
    uint tid [[ thread_position_in_threadgroup ]]
) {
    threadgroup float h_shm[KOKOPOP_LSTM_H];
    threadgroup float c_shm[KOKOPOP_LSTM_H];
    threadgroup float g_shm[KOKOPOP_LSTM_4H];

    if (tid < KOKOPOP_LSTM_H) {
        h_shm[tid] = 0.0f;
        c_shm[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0; step < N; ++step) {
        const uint t = rev ? (N - 1u - step) : step;

        const device float * col = w_hh + tid * KOKOPOP_LSTM_H;
        float dot = 0.0f;

        // Fixed trip count lets the compiler optimize better while preserving
        // the original sequential accumulation order for each gate output.
        for (uint k = 0u; k < KOKOPOP_LSTM_H; ++k) {
            dot += col[k] * h_shm[k];
        }

        float gv = b_hh[tid] + pre_gates[tid + KOKOPOP_LSTM_4H * t] + dot;
        if (!isfinite(gv)) gv = 0.0f;
        g_shm[tid] = gv;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < KOKOPOP_LSTM_H) {
            const float x_i = kokopop_sanitize_gate(g_shm[tid]);
            const float x_f = kokopop_sanitize_gate(g_shm[KOKOPOP_LSTM_H + tid]);
            const float x_g = kokopop_sanitize_gate(g_shm[2u * KOKOPOP_LSTM_H + tid]);
            const float x_o = kokopop_sanitize_gate(g_shm[3u * KOKOPOP_LSTM_H + tid]);

            // Keep precise math. fast:: variants caused non-finite downstream f0 values.
            const float i_gate = 1.0f / (1.0f + precise::exp(-x_i));
            const float f_gate = 1.0f / (1.0f + precise::exp(-x_f));
            const float g_gate = precise::tanh(x_g);
            const float o_gate = 1.0f / (1.0f + precise::exp(-x_o));

            float c_new = f_gate * c_shm[tid] + i_gate * g_gate;
            c_new = kokopop_sanitize_state(c_new);

            float h_new = o_gate * precise::tanh(c_new);
            if (!isfinite(h_new)) h_new = 0.0f;

            c_shm[tid] = c_new;
            h_shm[tid] = h_new;
            output[tid + KOKOPOP_LSTM_H * t] = h_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Pre-gates matmul. Computes pre_gates = w_ih @ input  (no bias).
//   w_ih   : [I, 4H]  fast dim = I  (ggml column-major)  → w_ih[i + I*g]
//   input  : [I, N]   fast dim = I                       → input[i + I*t]
//   output : [4H, N]  fast dim = 4H                      → output[g + 4H*t]
//
// One simdgroup (32 lanes) cooperates on each (g, t) output; the I-axis
// reduction is split across lanes and merged via simd_sum. Launch shape:
//   dispatchThreads(32, 4H, N), threadsPerThreadgroup(32, 1, 1).
struct PregatesArgs {
    uint I;
    uint four_H;
    uint N;
};

kernel void kokopop_lstm_pregates_matmul(
    constant PregatesArgs & args [[ buffer(0) ]],
    device const float * w_ih    [[ buffer(1) ]],
    device const float * input   [[ buffer(2) ]],
    device float       * output  [[ buffer(3) ]],
    uint3 gid  [[ thread_position_in_grid ]],
    uint  lane [[ thread_index_in_simdgroup ]]
) {
    const uint g = gid.y;
    const uint t = gid.z;
    if (g >= args.four_H || t >= args.N) return;

    float acc = 0.0f;
    for (uint i = lane; i < args.I; i += 32u) {
        acc += w_ih[i + args.I * g] * input[i + args.I * t];
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        output[g + args.four_H * t] = acc;
    }
}
)MSL";

// -----------------------------------------------------------------------------
struct MetalLstmProfileStats {
    uint64_t sync_calls = 0;
    double sync_upload_ms = 0.0;
    double sync_encode_ms = 0.0;
    double sync_gpu_wait_ms = 0.0;
    double sync_download_ms = 0.0;
};

static inline bool metal_lstm_profile_enabled() {
    static int enabled = []() -> int {
        const char * e = std::getenv("KOKOPOP_METAL_PROFILE");
        return (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static inline double metal_lstm_now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// State
// -----------------------------------------------------------------------------
struct MetalLstmKernelState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;

    std::unordered_map<std::string, id<MTLBuffer>> whh_buffers;

    id<MTLBuffer> pre_gates_buf = nil;
    id<MTLBuffer> b_hh_buf = nil;
    id<MTLBuffer> output_buf = nil;

    // Pre-gates matmul pipeline + buffers.
    id<MTLComputePipelineState> pregates_pipeline = nil;
    std::unordered_map<std::string, id<MTLBuffer>> wih_buffers;
    id<MTLBuffer> pregates_input_buf  = nil;  // Shared, [I, N]
    id<MTLBuffer> pregates_output_buf = nil;  // Shared, [4H, N]

    bool valid = false;

    MetalLstmProfileStats profile;
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

static bool metal_lstm_validate_shape(MetalLstmKernelState * state, int H, int N) {
    if (state == nullptr || !state->valid || H <= 0 || N <= 0) {
        return false;
    }

    if (H != 256) {
        std::fprintf(stderr,
                     "[metal_lstm] unsupported H=%d; this shader supports H=256 only\n",
                     H);
        return false;
    }

    const NSUInteger threads_per_group = static_cast<NSUInteger>(4 * H);
    const NSUInteger max_threads = [state->pipeline maxTotalThreadsPerThreadgroup];
    if (threads_per_group > max_threads) {
        std::fprintf(stderr,
                     "[metal_lstm] threads-per-group=%lu exceeds pipeline limit=%lu\n",
                     static_cast<unsigned long>(threads_per_group),
                     static_cast<unsigned long>(max_threads));
        return false;
    }

    return true;
}

static bool encode_lstm(
    MetalLstmKernelState * state,
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> pre_gates_buf,
    id<MTLBuffer> b_hh_buf,
    id<MTLBuffer> out_buf,
    id<MTLBuffer> whh_buf,
    int H,
    int N,
    bool reverse
) {
    if (!metal_lstm_validate_shape(state, H, N) ||
        cmd == nil ||
        pre_gates_buf == nil ||
        b_hh_buf == nil ||
        out_buf == nil ||
        whh_buf == nil) {
        return false;
    }

    (void) H;
    const uint32_t params_N = static_cast<uint32_t>(N);
    const uint32_t params_rev = reverse ? 1u : 0u;

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (enc == nil) {
        return false;
    }

    [enc setComputePipelineState:state->pipeline];
    [enc setBuffer:out_buf       offset:0 atIndex:0];
    [enc setBuffer:pre_gates_buf offset:0 atIndex:1];
    [enc setBuffer:whh_buf       offset:0 atIndex:2];
    [enc setBuffer:b_hh_buf      offset:0 atIndex:3];
    [enc setBytes:&params_N      length:sizeof(params_N)   atIndex:4];
    [enc setBytes:&params_rev    length:sizeof(params_rev) atIndex:5];

    const NSUInteger threads_per_group = static_cast<NSUInteger>(4 * H);
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
    [enc endEncoding];

    return true;
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------
MetalLstmKernelState * metal_lstm_create() {
    @autoreleasepool {
        auto * state = new MetalLstmKernelState{};

        state->device = MTLCreateSystemDefaultDevice();
        if (state->device == nil) {
            std::fprintf(stderr, "[metal_lstm] MTLCreateSystemDefaultDevice failed\n");
            delete state;
            return nullptr;
        }

        state->queue = [state->device newCommandQueue];
        if (state->queue == nil) {
            std::fprintf(stderr, "[metal_lstm] newCommandQueue failed\n");
            delete state;
            return nullptr;
        }

        NSError * err = nil;
        NSString * src = [NSString stringWithUTF8String:kLstmShaderSrc];
        id<MTLLibrary> lib = [state->device newLibraryWithSource:src options:nil error:&err];
        if (lib == nil) {
            std::fprintf(stderr,
                         "[metal_lstm] shader compile error: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            delete state;
            return nullptr;
        }

        id<MTLFunction> fn = [lib newFunctionWithName:@"kokopop_lstm_fused_h256"];
        if (fn == nil) {
            std::fprintf(stderr, "[metal_lstm] function 'kokopop_lstm_fused_h256' not found\n");
            delete state;
            return nullptr;
        }

        state->pipeline = [state->device newComputePipelineStateWithFunction:fn error:&err];
        if (state->pipeline == nil) {
            std::fprintf(stderr,
                         "[metal_lstm] pipeline error: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            delete state;
            return nullptr;
        }

        // Pre-gates matmul pipeline (best-effort).
        id<MTLFunction> pregates_fn = [lib newFunctionWithName:@"kokopop_lstm_pregates_matmul"];
        if (pregates_fn != nil) {
            NSError * pgerr = nil;
            state->pregates_pipeline = [state->device newComputePipelineStateWithFunction:pregates_fn error:&pgerr];
            if (state->pregates_pipeline == nil) {
                std::fprintf(stderr, "[metal_lstm] pregates pipeline error: %s\n",
                             pgerr ? [[pgerr localizedDescription] UTF8String] : "unknown");
            }
        }

        state->valid = true;
        return state;
    }
}

void metal_lstm_destroy(MetalLstmKernelState * state) {
    if (state != nullptr && metal_lstm_profile_enabled()) {
        const auto & p = state->profile;
        if (p.sync_calls) {
            std::fprintf(stderr,
                         "[metal_lstm_profile] sync_calls=%llu upload=%.3fms encode=%.3fms gpu_wait=%.3fms download=%.3fms total=%.3fms\n",
                         static_cast<unsigned long long>(p.sync_calls),
                         p.sync_upload_ms,
                         p.sync_encode_ms,
                         p.sync_gpu_wait_ms,
                         p.sync_download_ms,
                         p.sync_upload_ms + p.sync_encode_ms + p.sync_gpu_wait_ms + p.sync_download_ms);
        }
    }
    delete state;
}

void metal_lstm_preload_whh(
    MetalLstmKernelState * state,
    const char * key,
    const float * w_hh_f32,
    int H,
    int four_H
) {
    if (state == nullptr || !state->valid || key == nullptr || w_hh_f32 == nullptr) {
        return;
    }

    if (H != 256 || four_H != 4 * H) {
        std::fprintf(stderr,
                     "[metal_lstm] invalid W_hh shape H=%d four_H=%d; expected H=256 four_H=1024\n",
                     H,
                     four_H);
        return;
    }

    const size_t bytes = static_cast<size_t>(H) * static_cast<size_t>(four_H) * sizeof(float);
    id<MTLBuffer> buf = [state->device newBufferWithBytes:w_hh_f32
                                                    length:bytes
                                                   options:MTLResourceStorageModeShared];
    if (buf == nil) {
        std::fprintf(stderr, "[metal_lstm] failed to allocate W_hh buffer for %s\n", key);
        return;
    }

    state->whh_buffers[key] = buf;
}

// Uploads I*4H f32 of w_ih once per (key, shape), caches the MTLBuffer,
// and reuses it across calls.
void metal_lstm_preload_wih(
    MetalLstmKernelState * state,
    const char * key,
    const float * w_ih_f32,
    int I,
    int four_H
) {
    if (state == nullptr || !state->valid || key == nullptr || w_ih_f32 == nullptr ||
        state->pregates_pipeline == nil || I <= 0 || four_H <= 0) {
        return;
    }
    const size_t bytes = static_cast<size_t>(I) * static_cast<size_t>(four_H) * sizeof(float);
    id<MTLBuffer> buf = [state->device newBufferWithBytes:w_ih_f32
                                                    length:bytes
                                                   options:MTLResourceStorageModeShared];
    if (buf == nil) {
        std::fprintf(stderr, "[metal_lstm] failed to allocate w_ih buffer for %s\n", key);
        return;
    }
    state->wih_buffers[key] = buf;
}

bool metal_lstm_pregates_matmul(
    MetalLstmKernelState * state,
    const char * key,
    const float * input,
    float       * pre_gates,
    int I, int four_H, int N
) {
    @autoreleasepool {
        if (state == nullptr || !state->valid || state->pregates_pipeline == nil ||
            key == nullptr || input == nullptr || pre_gates == nullptr ||
            I <= 0 || four_H <= 0 || N <= 0) {
            return false;
        }
        auto it = state->wih_buffers.find(key);
        if (it == state->wih_buffers.end()) return false;
        id<MTLBuffer> wih_buf = it->second;

        const size_t input_bytes  = static_cast<size_t>(I) * static_cast<size_t>(N) * sizeof(float);
        const size_t output_bytes = static_cast<size_t>(four_H) * static_cast<size_t>(N) * sizeof(float);
        state->pregates_input_buf  = ensure_buf(state->device, state->pregates_input_buf,  input_bytes);
        state->pregates_output_buf = ensure_buf(state->device, state->pregates_output_buf, output_bytes);
        if (state->pregates_input_buf == nil || state->pregates_output_buf == nil) return false;

        std::memcpy([state->pregates_input_buf contents], input, input_bytes);

        id<MTLCommandBuffer> cmd = [state->queue commandBuffer];
        if (cmd == nil) return false;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil) return false;

        const uint32_t args[3] = {
            static_cast<uint32_t>(I),
            static_cast<uint32_t>(four_H),
            static_cast<uint32_t>(N),
        };
        [enc setComputePipelineState:state->pregates_pipeline];
        [enc setBytes:args  length:sizeof(args) atIndex:0];
        [enc setBuffer:wih_buf                    offset:0 atIndex:1];
        [enc setBuffer:state->pregates_input_buf  offset:0 atIndex:2];
        [enc setBuffer:state->pregates_output_buf offset:0 atIndex:3];

        [enc dispatchThreads:MTLSizeMake(32,
                                          static_cast<NSUInteger>(four_H),
                                          static_cast<NSUInteger>(N))
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal_lstm] pregates command failed status=%lu\n",
                         static_cast<unsigned long>([cmd status]));
            return false;
        }

        std::memcpy(pre_gates, [state->pregates_output_buf contents], output_bytes);
        return true;
    }
}

void metal_lstm_run(
    MetalLstmKernelState * state,
    const char * whh_key,
    const float * pre_gates,
    const float * b_hh,
    float * output,
    int H,
    int N,
    bool reverse
) {
    @autoreleasepool {
        if (!metal_lstm_validate_shape(state, H, N) ||
            whh_key == nullptr ||
            pre_gates == nullptr ||
            b_hh == nullptr ||
            output == nullptr) {
            return;
        }

        auto it = state->whh_buffers.find(whh_key);
        if (it == state->whh_buffers.end()) {
            std::fprintf(stderr, "[metal_lstm] W_hh not preloaded for key: %s\n", whh_key);
            return;
        }

        const size_t pg_bytes = static_cast<size_t>(4 * H) * static_cast<size_t>(N) * sizeof(float);
        const size_t bhh_bytes = static_cast<size_t>(4 * H) * sizeof(float);
        const size_t out_bytes = static_cast<size_t>(H) * static_cast<size_t>(N) * sizeof(float);

        state->pre_gates_buf = ensure_buf(state->device, state->pre_gates_buf, pg_bytes);
        state->b_hh_buf = ensure_buf(state->device, state->b_hh_buf, bhh_bytes);
        state->output_buf = ensure_buf(state->device, state->output_buf, out_bytes);
        if (state->pre_gates_buf == nil || state->b_hh_buf == nil || state->output_buf == nil) {
            std::fprintf(stderr, "[metal_lstm] scratch buffer allocation failed\n");
            return;
        }

        double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;
        if (metal_lstm_profile_enabled()) t0 = metal_lstm_now_ms();
        std::memcpy([state->pre_gates_buf contents], pre_gates, pg_bytes);
        std::memcpy([state->b_hh_buf contents], b_hh, bhh_bytes);
        if (metal_lstm_profile_enabled()) t1 = metal_lstm_now_ms();

        id<MTLCommandBuffer> cmd = [state->queue commandBuffer];
        if (cmd == nil) {
            return;
        }

        if (!encode_lstm(state,
                         cmd,
                         state->pre_gates_buf,
                         state->b_hh_buf,
                         state->output_buf,
                         it->second,
                         H,
                         N,
                         reverse)) {
            return;
        }
        if (metal_lstm_profile_enabled()) t2 = metal_lstm_now_ms();

        [cmd commit];
        [cmd waitUntilCompleted];
        if (metal_lstm_profile_enabled()) t3 = metal_lstm_now_ms();

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr,
                         "[metal_lstm] sync command failed with status=%lu\n",
                         static_cast<unsigned long>([cmd status]));
            return;
        }

        std::memcpy(output, [state->output_buf contents], out_bytes);
        if (metal_lstm_profile_enabled()) {
            t4 = metal_lstm_now_ms();
            state->profile.sync_calls += 1;
            state->profile.sync_upload_ms += (t1 - t0);
            state->profile.sync_encode_ms += (t2 - t1);
            state->profile.sync_gpu_wait_ms += (t3 - t2);
            state->profile.sync_download_ms += (t4 - t3);
        }
    }
}
