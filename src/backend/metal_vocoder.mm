#include "metal_vocoder.h"

#ifdef KOKOPOP_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <unordered_map>

namespace kokopop {
namespace {

// -----------------------------------------------------------------------------
// MSL shader source
//
// Safe performance changes versus the naive version:
//   - Keep float accumulation.
//   - Keep exact convolution indexing.
//   - Skip k values that cannot satisfy the stride relation.
//   - No fast:: math and no half accumulation.
// -----------------------------------------------------------------------------
static constexpr const char * kVocoderShader = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct ConvtArgs {
    uint IL;
    uint IC;
    uint K;
    uint OC;
    uint OL;
    uint stride;
    uint crop_left;
};

template <typename W>
kernel void kokopop_convt1d_crop_bias_generic(
    constant ConvtArgs & args [[buffer(0)]],
    device const float * input [[buffer(1)]],
    device const W * weight [[buffer(2)]],
    device const float * bias [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint t = gid.x;
    const uint oc = gid.y;
    if (t >= args.OL || oc >= args.OC) return;

    const uint full_t = t + args.crop_left;
    float acc = bias[oc];

    // Original condition: (full_t - k) % stride == 0.
    // Iterate only k values satisfying k == full_t mod stride.
    const uint k0 = full_t % args.stride;

    for (uint ic = 0; ic < args.IC; ++ic) {
        for (uint k = k0; k < args.K; k += args.stride) {
            if (full_t < k) continue;

            const uint ti = (full_t - k) / args.stride;
            if (ti >= args.IL) continue;

            const uint w_idx = k + args.K * oc + args.K * args.OC * ic;
            const uint x_idx = ti + args.IL * ic;
            acc += float(weight[w_idx]) * input[x_idx];
        }
    }

#if KOKOPOP_PARANOID_NUMERIC_GUARDS
    if (!isfinite(acc)) acc = 0.0f;
#endif
    output[t + args.OL * oc] = acc;
}

template [[host_name("kokopop_convt1d_crop_bias_generic_f32")]]
kernel void kokopop_convt1d_crop_bias_generic<float>(
    constant ConvtArgs & args [[buffer(0)]],
    device const float * input [[buffer(1)]],
    device const float * weight [[buffer(2)]],
    device const float * bias [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
);

template [[host_name("kokopop_convt1d_crop_bias_generic_f16")]]
kernel void kokopop_convt1d_crop_bias_generic<half>(
    constant ConvtArgs & args [[buffer(0)]],
    device const float * input [[buffer(1)]],
    device const half * weight [[buffer(2)]],
    device const float * bias [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
);
)METAL";

struct ConvtArgs {
    uint32_t IL;
    uint32_t IC;
    uint32_t K;
    uint32_t OC;
    uint32_t OL;
    uint32_t stride;
    uint32_t crop_left;
};

struct CachedBuffer {
    id<MTLBuffer> buffer = nil;
    size_t bytes = 0;
};

struct MetalProfileStats {
    uint64_t calls = 0;
    double upload_ms = 0.0;
    double encode_ms = 0.0;
    double gpu_wait_ms = 0.0;
    double download_ms = 0.0;
};

static inline bool metal_profile_enabled() {
    static int enabled = []() -> int {
        const char * e = std::getenv("KOKOPOP_METAL_PROFILE");
        return (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static inline double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static inline void add_ms(double & dst, double start, double end) {
    if (metal_profile_enabled()) dst += (end - start);
}

static bool tensor_is_contiguous_f32(const ggml_tensor * tensor) {
    return tensor != nullptr &&
           tensor->type == GGML_TYPE_F32 &&
           tensor->data != nullptr &&
           ggml_is_contiguous(tensor);
}

static bool tensor_is_contiguous_weight(const ggml_tensor * tensor) {
    return tensor != nullptr &&
           (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16) &&
           tensor->data != nullptr &&
           ggml_is_contiguous(tensor);
}

static bool fits_u32(int64_t value) {
    return value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
}

} // namespace

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------
struct MetalVocoderState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;

    id<MTLComputePipelineState> convt_generic_f32 = nil;
    id<MTLComputePipelineState> convt_generic_f16 = nil;

    id<MTLBuffer> input_buf = nil;
    id<MTLBuffer> output_buf = nil;

    // Key by tensor pointer rather than tensor->data. This avoids stale cache hits
    // when allocator memory is reused for a different tensor of the same byte size.
    std::unordered_map<const ggml_tensor *, CachedBuffer> constants;

    MetalProfileStats convt_profile;
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
static id<MTLBuffer> ensure_scratch_buffer(id<MTLDevice> device, id<MTLBuffer> & buffer, size_t bytes) {
    if (buffer != nil && [buffer length] >= bytes) return buffer;
    buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return buffer;
}

static id<MTLBuffer> cached_constant_buffer(MetalVocoderState * state, const ggml_tensor * tensor) {
    const size_t bytes = ggml_nbytes(tensor);
    CachedBuffer & cached = state->constants[tensor];

    if (cached.buffer != nil && cached.bytes == bytes) return cached.buffer;

    cached.buffer = [state->device newBufferWithBytes:tensor->data
                                               length:bytes
                                              options:MTLResourceStorageModeShared];
    cached.bytes = bytes;
    return cached.buffer;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, const char * name) {
    NSString * ns_name = [NSString stringWithUTF8String:name];
    id<MTLFunction> fn = [library newFunctionWithName:ns_name];
    if (fn == nil) {
        std::fprintf(stderr, "[metal_vocoder] function '%s' not found\n", name);
        return nil;
    }

    NSError * error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
    if (pipeline == nil) {
        std::fprintf(stderr, "[metal_vocoder] pipeline '%s' error: %s\n",
                     name,
                     error ? [[error localizedDescription] UTF8String] : "unknown");
    }
    return pipeline;
}

static id<MTLComputePipelineState> select_convt_pipeline(MetalVocoderState * state, int weight_type) {
    return weight_type == GGML_TYPE_F16 ? state->convt_generic_f16 : state->convt_generic_f32;
}

static bool load_pipeline_pair(
    MetalVocoderState * state,
    id<MTLComputePipelineState> & dst,
    const char * name
) {
    dst = make_pipeline(state->device, state->library, name);
    return dst != nil;
}


// -----------------------------------------------------------------------------
// GPU-chainable encode API
// -----------------------------------------------------------------------------
// This function only encodes the transposed convolution into an existing command
// buffer. It does not upload input, does not commit, does not wait, and does not
// download output. It is intended for chaining several vocoder layers on GPU.
//
// input/output offsets are byte offsets inside their MTLBuffers.
// The existing CPU-facing wrapper below uses this function internally.
bool metal_vocoder_conv_transpose1d_crop_bias_encode(
    MetalVocoderState * state,
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> input_buf,
    NSUInteger input_offset,
    const ggml_tensor * weight,
    const ggml_tensor * bias,
    id<MTLBuffer> output_buf,
    NSUInteger output_offset,
    int64_t IL,
    int64_t IC,
    int64_t OL,
    int64_t OC,
    int stride,
    int crop_left
) {
    if (state == nullptr || cmd == nil || input_buf == nil || output_buf == nil ||
        !tensor_is_contiguous_weight(weight) || !tensor_is_contiguous_f32(bias) ||
        stride <= 0 || crop_left < 0) {
        return false;
    }

    const int64_t K = weight->ne[0];
    const int64_t W_OC = weight->ne[1];
    const int64_t W_IC = weight->ne[2];

    if (K <= 0 || OC <= 0 || IC <= 0 || IL <= 0 || OL <= 0 ||
        W_OC != OC || W_IC != IC || bias->ne[0] != OC || weight->ne[3] != 1 ||
        !fits_u32(K) || !fits_u32(OC) || !fits_u32(IC) ||
        !fits_u32(IL) || !fits_u32(OL) ||
        !fits_u32(stride) || !fits_u32(crop_left)) {
        return false;
    }

    id<MTLComputePipelineState> pipeline = select_convt_pipeline(state, weight->type);
    id<MTLBuffer> weight_buf = cached_constant_buffer(state, weight);
    id<MTLBuffer> bias_buf = cached_constant_buffer(state, bias);
    if (pipeline == nil || weight_buf == nil || bias_buf == nil) return false;

    ConvtArgs args {
        static_cast<uint32_t>(IL),
        static_cast<uint32_t>(IC),
        static_cast<uint32_t>(K),
        static_cast<uint32_t>(OC),
        static_cast<uint32_t>(OL),
        static_cast<uint32_t>(stride),
        static_cast<uint32_t>(crop_left),
    };

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (enc == nil) return false;

    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:input_buf  offset:input_offset  atIndex:1];
    [enc setBuffer:weight_buf offset:0             atIndex:2];
    [enc setBuffer:bias_buf   offset:0             atIndex:3];
    [enc setBuffer:output_buf offset:output_offset atIndex:4];

    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(OL), static_cast<NSUInteger>(OC), 1)
   threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
    [enc endEncoding];
    return true;
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------
MetalVocoderState * metal_vocoder_create() {
    @autoreleasepool {
        auto * state = new MetalVocoderState();

        state->device = MTLCreateSystemDefaultDevice();
        if (state->device == nil) {
            std::fprintf(stderr, "[metal_vocoder] MTLCreateSystemDefaultDevice failed\n");
            delete state;
            return nullptr;
        }

        state->queue = [state->device newCommandQueue];
        if (state->queue == nil) {
            std::fprintf(stderr, "[metal_vocoder] newCommandQueue failed\n");
            delete state;
            return nullptr;
        }

        NSError * error = nil;
        NSString * source = [NSString stringWithUTF8String:kVocoderShader];
        MTLCompileOptions * options = [MTLCompileOptions new];
        options.preprocessorMacros = @{ @"KOKOPOP_PARANOID_NUMERIC_GUARDS" : @0 };
        state->library = [state->device newLibraryWithSource:source options:options error:&error];
        if (state->library == nil) {
            std::fprintf(stderr, "[metal_vocoder] shader compile error: %s\n",
                         error ? [[error localizedDescription] UTF8String] : "unknown");
            delete state;
            return nullptr;
        }

        bool ok = true;
        ok &= load_pipeline_pair(state, state->convt_generic_f32, "kokopop_convt1d_crop_bias_generic_f32");
        ok &= load_pipeline_pair(state, state->convt_generic_f16, "kokopop_convt1d_crop_bias_generic_f16");
        if (!ok) {
            delete state;
            return nullptr;
        }

        return state;
    }
}

void metal_vocoder_destroy(MetalVocoderState * state) {
    if (state != nullptr && metal_profile_enabled() && state->convt_profile.calls > 0) {
        const auto & p = state->convt_profile;
        std::fprintf(stderr,
                     "[metal_vocoder_profile] calls=%llu upload=%.3fms encode=%.3fms gpu_wait=%.3fms download=%.3fms total=%.3fms\n",
                     static_cast<unsigned long long>(p.calls),
                     p.upload_ms,
                     p.encode_ms,
                     p.gpu_wait_ms,
                     p.download_ms,
                     p.upload_ms + p.encode_ms + p.gpu_wait_ms + p.download_ms);
    }
    delete state;
}

bool metal_vocoder_conv_transpose1d_crop_bias(
    MetalVocoderState * state,
    const ggml_tensor * input,
    const ggml_tensor * weight,
    const ggml_tensor * bias,
    ggml_tensor * output,
    int stride,
    int crop_left
) {
    @autoreleasepool {
        if (state == nullptr ||
            !tensor_is_contiguous_f32(input) ||
            !tensor_is_contiguous_weight(weight) ||
            !tensor_is_contiguous_f32(bias) ||
            output == nullptr ||
            output->type != GGML_TYPE_F32 ||
            output->data == nullptr ||
            !ggml_is_contiguous(output) ||
            stride <= 0 ||
            crop_left < 0) {
            return false;
        }

        const int64_t K = weight->ne[0];
        const int64_t OC = weight->ne[1];
        const int64_t IC = weight->ne[2];
        const int64_t IL = input->ne[0];
        const int64_t OL = output->ne[0];

        if (K <= 0 || OC <= 0 || IC <= 0 || IL <= 0 || OL <= 0 ||
            input->ne[1] != IC ||
            output->ne[1] != OC ||
            bias->ne[0] != OC ||
            weight->ne[3] != 1 ||
            !fits_u32(K) || !fits_u32(OC) || !fits_u32(IC) ||
            !fits_u32(IL) || !fits_u32(OL) ||
            !fits_u32(stride) || !fits_u32(crop_left)) {
            return false;
        }

        const size_t input_bytes = ggml_nbytes(input);
        const size_t output_bytes = ggml_nbytes(output);

        id<MTLBuffer> input_buf = ensure_scratch_buffer(state->device, state->input_buf, input_bytes);
        id<MTLBuffer> output_buf = ensure_scratch_buffer(state->device, state->output_buf, output_bytes);
        if (input_buf == nil || output_buf == nil) return false;

        double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;
        if (metal_profile_enabled()) t0 = now_ms();
        std::memcpy([input_buf contents], input->data, input_bytes);
        if (metal_profile_enabled()) t1 = now_ms();

        id<MTLCommandBuffer> cmd = [state->queue commandBuffer];
        if (cmd == nil) return false;

        if (!metal_vocoder_conv_transpose1d_crop_bias_encode(
                state,
                cmd,
                input_buf,
                0,
                weight,
                bias,
                output_buf,
                0,
                IL,
                IC,
                OL,
                OC,
                stride,
                crop_left)) {
            return false;
        }
        if (metal_profile_enabled()) t2 = now_ms();

        [cmd commit];
        [cmd waitUntilCompleted];
        if (metal_profile_enabled()) t3 = now_ms();

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal_vocoder] command failed with status=%lu\n",
                         static_cast<unsigned long>([cmd status]));
            return false;
        }

        std::memcpy(output->data, [output_buf contents], output_bytes);
        if (metal_profile_enabled()) {
            t4 = now_ms();
            state->convt_profile.calls += 1;
            add_ms(state->convt_profile.upload_ms, t0, t1);
            add_ms(state->convt_profile.encode_ms, t1, t2);
            add_ms(state->convt_profile.gpu_wait_ms, t2, t3);
            add_ms(state->convt_profile.download_ms, t3, t4);
        }
        return true;
    }
}

} // namespace kokopop

#endif // KOKOPOP_HAS_METAL
