#include "metal_vocoder.h"

#ifdef KOKOPOP_HAS_METAL

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace kokopop {

namespace {

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
kernel void kokopop_convt1d_crop_bias(
    constant ConvtArgs & args [[buffer(0)]],
    device const float * input [[buffer(1)]],
    device const W * weight [[buffer(2)]],
    device const float * bias [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint t = gid.x;
    const uint oc = gid.y;
    if (t >= args.OL || oc >= args.OC) {
        return;
    }

    const int full_t = int(t) + int(args.crop_left);
    float acc = bias[oc];

    for (uint ic = 0; ic < args.IC; ++ic) {
        for (uint k = 0; k < args.K; ++k) {
            const int src_num = full_t - int(k);
            if (src_num < 0 || (src_num % int(args.stride)) != 0) {
                continue;
            }
            const uint ti = uint(src_num / int(args.stride));
            if (ti >= args.IL) {
                continue;
            }
            const uint w_idx = k + args.K * oc + args.K * args.OC * ic;
            const uint x_idx = ti + args.IL * ic;
            acc += float(weight[w_idx]) * input[x_idx];
        }
    }

    output[t + args.OL * oc] = acc;
}

template [[host_name("kokopop_convt1d_crop_bias_f32")]]
kernel void kokopop_convt1d_crop_bias<float>(
    constant ConvtArgs & args [[buffer(0)]],
    device const float * input [[buffer(1)]],
    device const float * weight [[buffer(2)]],
    device const float * bias [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]);

template [[host_name("kokopop_convt1d_crop_bias_f16")]]
kernel void kokopop_convt1d_crop_bias<half>(
    constant ConvtArgs & args [[buffer(0)]],
    device const float * input [[buffer(1)]],
    device const half * weight [[buffer(2)]],
    device const float * bias [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]);
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

} // namespace

struct MetalVocoderState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> convt_f32 = nil;
    id<MTLComputePipelineState> convt_f16 = nil;
    id<MTLBuffer> input_buf = nil;
    id<MTLBuffer> output_buf = nil;
    id<MTLBuffer> args_buf = nil;
    std::unordered_map<const void *, CachedBuffer> constants;
};

static id<MTLBuffer> ensure_scratch_buffer(id<MTLDevice> device, id<MTLBuffer> & buffer, size_t bytes) {
    if (buffer != nil && [buffer length] >= bytes) {
        return buffer;
    }
    buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return buffer;
}

static id<MTLBuffer> cached_constant_buffer(MetalVocoderState * state, const ggml_tensor * tensor) {
    const size_t bytes = ggml_nbytes(tensor);
    CachedBuffer & cached = state->constants[tensor->data];
    if (cached.buffer != nil && cached.bytes == bytes) {
        return cached.buffer;
    }
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
                     name, error ? [[error localizedDescription] UTF8String] : "unknown");
    }
    return pipeline;
}

MetalVocoderState * metal_vocoder_create() {
    @autoreleasepool {
        MetalVocoderState * state = new MetalVocoderState();
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
        state->library = [state->device newLibraryWithSource:source options:nil error:&error];
        if (state->library == nil) {
            std::fprintf(stderr, "[metal_vocoder] shader compile error: %s\n",
                         error ? [[error localizedDescription] UTF8String] : "unknown");
            delete state;
            return nullptr;
        }

        state->convt_f32 = make_pipeline(state->device, state->library, "kokopop_convt1d_crop_bias_f32");
        state->convt_f16 = make_pipeline(state->device, state->library, "kokopop_convt1d_crop_bias_f16");
        if (state->convt_f32 == nil || state->convt_f16 == nil) {
            delete state;
            return nullptr;
        }
        return state;
    }
}

void metal_vocoder_destroy(MetalVocoderState * state) {
    delete state;
}

bool metal_vocoder_conv_transpose1d_crop_bias(
    MetalVocoderState * state,
    const ggml_tensor * input,
    const ggml_tensor * weight,
    const ggml_tensor * bias,
    ggml_tensor * output,
    int stride,
    int crop_left) {
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
            weight->ne[3] != 1) {
            return false;
        }

        const size_t input_bytes = ggml_nbytes(input);
        const size_t output_bytes = ggml_nbytes(output);
        id<MTLBuffer> input_buf = ensure_scratch_buffer(state->device, state->input_buf, input_bytes);
        id<MTLBuffer> output_buf = ensure_scratch_buffer(state->device, state->output_buf, output_bytes);
        id<MTLBuffer> args_buf = ensure_scratch_buffer(state->device, state->args_buf, sizeof(ConvtArgs));
        id<MTLBuffer> weight_buf = cached_constant_buffer(state, weight);
        id<MTLBuffer> bias_buf = cached_constant_buffer(state, bias);
        if (input_buf == nil || output_buf == nil || args_buf == nil || weight_buf == nil || bias_buf == nil) {
            return false;
        }

        ConvtArgs args{
            static_cast<uint32_t>(IL),
            static_cast<uint32_t>(IC),
            static_cast<uint32_t>(K),
            static_cast<uint32_t>(OC),
            static_cast<uint32_t>(OL),
            static_cast<uint32_t>(stride),
            static_cast<uint32_t>(crop_left),
        };
        std::memcpy([input_buf contents], input->data, input_bytes);
        std::memcpy([args_buf contents], &args, sizeof(args));

        id<MTLCommandBuffer> cmd = [state->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:(weight->type == GGML_TYPE_F16 ? state->convt_f16 : state->convt_f32)];
        [enc setBuffer:args_buf offset:0 atIndex:0];
        [enc setBuffer:input_buf offset:0 atIndex:1];
        [enc setBuffer:weight_buf offset:0 atIndex:2];
        [enc setBuffer:bias_buf offset:0 atIndex:3];
        [enc setBuffer:output_buf offset:0 atIndex:4];

        const NSUInteger tx = 16;
        const NSUInteger ty = 8;
        MTLSize threads_per_group = MTLSizeMake(tx, ty, 1);
        MTLSize threads_per_grid = MTLSizeMake(static_cast<NSUInteger>(OL), static_cast<NSUInteger>(OC), 1);
        [enc dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            return false;
        }

        std::memcpy(output->data, [output_buf contents], output_bytes);
        return true;
    }
}

} // namespace kokopop

#endif // KOKOPOP_HAS_METAL
