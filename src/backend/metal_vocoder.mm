#include "metal_vocoder.h"
#include "core/constants.h"

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

    // k0 = full_t % stride, so k = k0 + n*stride always satisfies k <= full_t
    // for n in [0, (full_t - k0)/stride]. Compute the exclusive upper bound directly
    // instead of guarding inside the loop.
    const uint k_max = min(args.K, full_t + 1);

    for (uint ic = 0; ic < args.IC; ++ic) {
        for (uint k = k0; k < k_max; k += args.stride) {
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

// -----------------------------------------------------------------------
// AdaIN + Snake fused kernel
//
// Fuses per-channel instance norm, learned affine (norm_w/norm_b),
// AdaIN style conditioning (gamma/beta pre-projected on CPU), and
// the Snake activation into a single GPU pass.
//
// x layout : [T, C]  (T = ne[0], C = ne[1] in ggml)
// norm_w/b : [C]
// gamma/beta: [C]  — pre-projected: gamma_w @ style + gamma_b
// alpha    : [C]  — snake parameter (ne[0]=1, ne[1]=C, contiguous)
//
// Dispatch: dispatchThreadgroups:(C,1,1)  threadsPerThreadgroup:(TG,1,1)
//   One threadgroup per channel; TG threads reduce across T.
// -----------------------------------------------------------------------
constant uint kAdainTgSize = 256;

struct AdainSnakeArgs {
    uint T;
    uint C;
    float eps;
};

kernel void kokopop_adain_snake(
    constant AdainSnakeArgs & args [[buffer(0)]],
    device const float * x         [[buffer(1)]],  // [T, C]
    device const float * norm_w    [[buffer(2)]],  // [C]
    device const float * norm_b    [[buffer(3)]],  // [C]
    device const float * gamma     [[buffer(4)]],  // [C]
    device const float * beta      [[buffer(5)]],  // [C]
    device const float * alpha     [[buffer(6)]],  // [C]
    device float       * out       [[buffer(7)]],  // [T, C]
    threadgroup float  * smem      [[threadgroup(0)]],
    uint  tid  [[thread_position_in_threadgroup]],
    uint  tgid [[threadgroup_position_in_grid]]
) {
    const uint c = tgid;
    if (c >= args.C || tid >= kAdainTgSize) return;

    const uint T = args.T;
    const device float * xc  = x   + c * T;
    device       float * outc = out + c * T;

    // Two-pass reduction for mean and E[x^2].
    float local_sum = 0.0f, local_sum2 = 0.0f;
    for (uint t = tid; t < T; t += kAdainTgSize) {
        float v    = xc[t];
        local_sum  += v;
        local_sum2 += v * v;
    }

    threadgroup float * s1 = smem;
    threadgroup float * s2 = smem + kAdainTgSize;
    s1[tid] = local_sum;
    s2[tid] = local_sum2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kAdainTgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s1[tid] += s1[tid + stride];
            s2[tid] += s2[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float ft    = float(T);
    const float mean  = s1[0] / ft;
    const float var   = max(s2[0] / ft - mean * mean, 0.0f);
    const float istd  = rsqrt(var + args.eps);

    const float nw    = norm_w[c];
    const float nb    = norm_b[c];
    const float g     = gamma[c];
    const float b     = beta[c];
    const float a     = alpha[c];
    const float inv_a = 1.0f / a;

    for (uint t = tid; t < T; t += kAdainTgSize) {
        float v  = (xc[t] - mean) * istd;   // instance norm
        v = v * nw + nb;                     // learned affine
        v = v + v * g + b;                   // adain: v*(1+gamma)+beta
        float sv = sin(a * v);
        outc[t]  = v + sv * sv * inv_a;      // snake
    }
}

// -----------------------------------------------------------------------
// Regular conv1d with dilation — symmetric same-padding, stride=1
//
// input  : [IL, IC]
// weight : [K, IC, OC]  (ggml_conv_1d layout: ne[0]=K, ne[1]=IC, ne[2]=OC)
// bias   : [OC]
// output : [OL, OC]
//
// output[t][oc] = bias[oc] + sum_ic sum_k input[t - padding + k*dilation][ic]
//                                          * weight[k][ic][oc]
// -----------------------------------------------------------------------
struct Conv1dArgs {
    uint IL;
    uint IC;
    uint OC;
    uint OL;
    uint K;
    uint dilation;
    uint padding;
};

template <typename W>
kernel void kokopop_conv1d_bias_generic(
    constant Conv1dArgs & args [[buffer(0)]],
    device const float * input  [[buffer(1)]],  // [IL, IC]
    device const W     * weight [[buffer(2)]],  // [K, IC, OC]
    device const float * bias   [[buffer(3)]],  // [OC]
    device float       * output [[buffer(4)]],  // [OL, OC]
    uint2 gid [[thread_position_in_grid]]
) {
    const uint t  = gid.x;
    const uint oc = gid.y;
    if (t >= args.OL || oc >= args.OC) return;

    float acc = bias[oc];
    for (uint ic = 0; ic < args.IC; ++ic) {
        for (uint k = 0; k < args.K; ++k) {
            const int src = int(t) - int(args.padding) + int(k) * int(args.dilation);
            if (src < 0 || src >= int(args.IL)) continue;
            const uint w_idx = k + args.K * ic + args.K * args.IC * oc;
            const uint x_idx = uint(src) + args.IL * ic;
            acc += float(weight[w_idx]) * input[x_idx];
        }
    }
    output[t + args.OL * oc] = acc;
}

template [[host_name("kokopop_conv1d_bias_generic_f32")]]
kernel void kokopop_conv1d_bias_generic<float>(
    constant Conv1dArgs & args [[buffer(0)]],
    device const float * input  [[buffer(1)]],
    device const float * weight [[buffer(2)]],
    device const float * bias   [[buffer(3)]],
    device float       * output [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
);

template [[host_name("kokopop_conv1d_bias_generic_f16")]]
kernel void kokopop_conv1d_bias_generic<half>(
    constant Conv1dArgs & args [[buffer(0)]],
    device const float * input  [[buffer(1)]],
    device const half  * weight [[buffer(2)]],
    device const float * bias   [[buffer(3)]],
    device float       * output [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
);

// -----------------------------------------------------------------------
// Elementwise in-place add : a[i] += b[i]
// -----------------------------------------------------------------------
kernel void kokopop_add_inplace_f32(
    device float       * a   [[buffer(0)]],
    device const float * b   [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    a[gid] += b[gid];
}

// -----------------------------------------------------------------------
// In-place leaky_relu: x[i] = x[i] < 0 ? x[i] * slope : x[i]
// -----------------------------------------------------------------------
struct LeakyReluArgs {
    float slope;
    uint  n;
};

kernel void kokopop_leaky_relu_inplace_f32(
    constant LeakyReluArgs & args [[buffer(0)]],
    device   float         * x    [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= args.n) return;
    const float v = x[gid];
    x[gid] = (v < 0.0f) ? v * args.slope : v;
}

// -----------------------------------------------------------------------
// Strided conv1d (extends kokopop_conv1d_bias_generic with stride support).
//
// input  : [IL, IC]
// weight : ggml-flat layout, weight[k + K*ic + K*IC*oc]  (= [K*IC, OC] 2D)
// bias   : [OC]
// output : [OL, OC]
//
// output[t, oc] = bias[oc]
//               + sum_ic sum_k input[t*stride - padding + k*dilation, ic]
//                              * weight[k + K*ic + K*IC*oc]
// -----------------------------------------------------------------------
struct Conv1dStridedArgs {
    uint IL;
    uint IC;
    uint OC;
    uint OL;
    uint K;
    uint stride;
    uint dilation;
    uint padding;
};

template <typename W>
kernel void kokopop_conv1d_strided_bias_generic(
    constant Conv1dStridedArgs & args [[buffer(0)]],
    device const float * input  [[buffer(1)]],
    device const W     * weight [[buffer(2)]],
    device const float * bias   [[buffer(3)]],
    device float       * output [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint t  = gid.x;
    const uint oc = gid.y;
    if (t >= args.OL || oc >= args.OC) return;

    float acc = bias[oc];
    const int base = int(t) * int(args.stride) - int(args.padding);
    for (uint ic = 0; ic < args.IC; ++ic) {
        for (uint k = 0; k < args.K; ++k) {
            const int src = base + int(k) * int(args.dilation);
            if (src < 0 || src >= int(args.IL)) continue;
            const uint w_idx = k + args.K * ic + args.K * args.IC * oc;
            const uint x_idx = uint(src) + args.IL * ic;
            acc += float(weight[w_idx]) * input[x_idx];
        }
    }
    output[t + args.OL * oc] = acc;
}

template [[host_name("kokopop_conv1d_strided_bias_generic_f32")]]
kernel void kokopop_conv1d_strided_bias_generic<float>(
    constant Conv1dStridedArgs & args [[buffer(0)]],
    device const float * input  [[buffer(1)]],
    device const float * weight [[buffer(2)]],
    device const float * bias   [[buffer(3)]],
    device float       * output [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
);

template [[host_name("kokopop_conv1d_strided_bias_generic_f16")]]
kernel void kokopop_conv1d_strided_bias_generic<half>(
    constant Conv1dStridedArgs & args [[buffer(0)]],
    device const float * input  [[buffer(1)]],
    device const half  * weight [[buffer(2)]],
    device const float * bias   [[buffer(3)]],
    device float       * output [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
);

// -----------------------------------------------------------------------
// Reflect-pad on left by 1, no right pad. Layout [T, C] (T fast dim).
//   in  : [T_in,  C]
//   out : [T_out=T_in+1, C]
//   out[0, c] = in[1, c]              (reflection skipping the boundary)
//   out[t, c] = in[t-1, c]  for t>=1
// -----------------------------------------------------------------------
struct PadReflectArgs {
    uint T_in;
    uint T_out;
    uint C;
};

kernel void kokopop_pad_reflect_left1_f32(
    constant PadReflectArgs & args [[buffer(0)]],
    device const float * in  [[buffer(1)]],
    device float       * out [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint t = gid.x;
    const uint c = gid.y;
    if (t >= args.T_out || c >= args.C) return;
    const uint src = (t == 0u) ? 1u : (t - 1u);
    out[t + args.T_out * c] = in[src + args.T_in * c];
}

// -----------------------------------------------------------------------
// out[i] = (a[i] + b[i] + c[i]) * scale
// -----------------------------------------------------------------------
struct WeightedSum3Args {
    float scale;
    uint  n;
};

kernel void kokopop_weighted_sum3_scale_f32(
    constant WeightedSum3Args & args [[buffer(0)]],
    device const float * a   [[buffer(1)]],
    device const float * b   [[buffer(2)]],
    device const float * c   [[buffer(3)]],
    device float       * out [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= args.n) return;
    out[gid] = (a[gid] + b[gid] + c[gid]) * args.scale;
}

// -----------------------------------------------------------------------
// Buffer-to-buffer copy (used to re-seed x_acc from a stable x_orig
// between the 3 main-branch resblocks within a stage).
// -----------------------------------------------------------------------
kernel void kokopop_copy_f32(
    device float       * dst [[buffer(0)]],
    device const float * src [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = src[gid];
}

// -----------------------------------------------------------------------
// iSTFT kernels (mirror of metal_stft.mm) so the full post_conv + iSTFT
// chain can run on the vocoder's command queue, in a single command buffer.
// Keep precise:: math here for the same reasons as in metal_stft.mm.
// -----------------------------------------------------------------------
kernel void kokopop_istft_idft_v(
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
    if (frame >= n_frames || n >= n_fft) return;

    const uint n_half = n_fft / 2u;
    const uint n_bins = n_half + 1u;

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
    if (!isfinite(sample)) sample = 0.0f;

    const float win = window_buf[n];
    const uint idx = frame * n_fft + n;
    tmp_y[idx] = sample * win;
    tmp_d[idx] = win * win;
}

kernel void kokopop_istft_ola_v(
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

    const int s = int(gid);
    const int nf = int(n_fft);
    const int h = int(hop);
    const int nfr = int(n_frames);

    int f_start = 0;
    if (s >= nf) f_start = (s - nf + h) / h;
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

kernel void kokopop_istft_norm_v(
    device float        * out        [[ buffer(0) ]],
    device const float  * y_buf      [[ buffer(1) ]],
    device const float  * denom_buf  [[ buffer(2) ]],
    constant uint       & center_pad [[ buffer(3) ]],
    constant uint       & out_len    [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= out_len) return;
    const uint src = gid + center_pad;
    float value = y_buf[src];
    if (denom_buf[src] > 1e-8f) value /= denom_buf[src];
    if (!isfinite(value)) value = 0.0f;
    out[gid] = clamp(value, -1.0f, 1.0f);
}
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

static constexpr uint32_t kAdainTgSize = 256;  // must match kAdainTgSize in MSL

struct CppAdainSnakeArgs {
    uint32_t T;
    uint32_t C;
    float    eps;
};

struct CppConv1dArgs {
    uint32_t IL;
    uint32_t IC;
    uint32_t OC;
    uint32_t OL;
    uint32_t K;
    uint32_t dilation;
    uint32_t padding;
};

struct CppConv1dStridedArgs {
    uint32_t IL;
    uint32_t IC;
    uint32_t OC;
    uint32_t OL;
    uint32_t K;
    uint32_t stride;
    uint32_t dilation;
    uint32_t padding;
};

struct CppLeakyReluArgs {
    float    slope;
    uint32_t n;
};

struct CppPadReflectArgs {
    uint32_t T_in;
    uint32_t T_out;
    uint32_t C;
};

struct CppWeightedSum3Args {
    float    scale;
    uint32_t n;
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

    // Resblock scratch buffers.
    // x_buf  (Shared)  : current residual stream, uploaded/downloaded each call.
    // ping_buf (Private): first intermediate scratch [T, IC].
    // pong_buf (Private): second intermediate scratch [T, IC].
    // style_proj_buf (Shared): pre-projected gamma/beta for all 6 adain calls [12*IC].
    id<MTLBuffer> x_buf         = nil;
    id<MTLBuffer> ping_buf      = nil;
    id<MTLBuffer> pong_buf      = nil;
    id<MTLBuffer> style_proj_buf = nil;

    id<MTLComputePipelineState> adain_snake        = nil;
    id<MTLComputePipelineState> conv1d_generic_f32 = nil;
    id<MTLComputePipelineState> conv1d_generic_f16 = nil;
    id<MTLComputePipelineState> add_inplace        = nil;
    id<MTLComputePipelineState> leaky_relu_inplace = nil;
    id<MTLComputePipelineState> conv1d_strided_f32 = nil;
    id<MTLComputePipelineState> conv1d_strided_f16 = nil;
    id<MTLComputePipelineState> pad_reflect_left1  = nil;
    id<MTLComputePipelineState> weighted_sum3_scale = nil;
    id<MTLComputePipelineState> copy_f32           = nil;

    // Scratch buffers for the fused per-stage generator. Kept GPU-resident
    // across kernels within a single command buffer; reused chunk-to-chunk.
    id<MTLBuffer> stage_x_in_buf       = nil;  // Shared  : input x (uploaded each stage)
    id<MTLBuffer> stage_har_buf        = nil;  // Shared  : har_t (uploaded each stage)
    id<MTLBuffer> stage_style_proj_buf = nil;  // Shared  : packed style projections (noise + 3 main)
    id<MTLBuffer> stage_x_source_buf   = nil;  // Private : noise_conv → noise_resblock output
    id<MTLBuffer> stage_x_post_pad_buf = nil;  // Private : convt output
    id<MTLBuffer> stage_x_scratch_buf  = nil;  // Private : pad-reflect output (stage 1 only)
    id<MTLBuffer> stage_branch0_buf    = nil;  // Private : main-resblock branch 0 accumulator
    id<MTLBuffer> stage_branch1_buf    = nil;  // Private : main-resblock branch 1 accumulator
    id<MTLBuffer> stage_branch2_buf    = nil;  // Private : main-resblock branch 2 accumulator
    id<MTLBuffer> stage_x_out_buf      = nil;  // Shared  : weighted-sum output (downloaded)

    // Cross-stage cache. Keeps the previous stage's output GPU-resident so
    // the next stage can blit it in instead of a CPU memcpy upload.
    id<MTLBuffer>   stage_prev_out_buf  = nil;  // Private : holds last stage's out_buf contents
    const float *   prev_out_addr       = nullptr;
    size_t          prev_out_bytes      = 0;

    // Same-har cache. har_t (~1 MB) is uploaded by both stage callbacks
    // every chunk; the second upload is redundant when the pointer+sentinels
    // match. Sentinels are a cheap "data unchanged" check that survives
    // allocator reuse for an unrelated tensor of the same length.
    const float *   prev_har_addr       = nullptr;
    size_t          prev_har_bytes      = 0;
    uint64_t        prev_har_sentinel0  = 0;
    uint64_t        prev_har_sentinel1  = 0;

    // Fused post_conv + iSTFT chain. Pipelines + scratch buffers used by
    // metal_vocoder_run_post_istft, encoded in a single command buffer.
    id<MTLComputePipelineState> istft_idft = nil;
    id<MTLComputePipelineState> istft_ola  = nil;
    id<MTLComputePipelineState> istft_norm = nil;
    int istft_n_fft = 0;
    int istft_hop   = 0;
    int istft_n_bins = 0;
    id<MTLBuffer> istft_window_buf = nil;     // Shared, [n_fft]
    id<MTLBuffer> istft_tw_c_buf   = nil;     // Shared, [n_bins * n_fft]
    id<MTLBuffer> istft_tw_s_buf   = nil;     // Shared, [n_bins * n_fft]
    id<MTLBuffer> post_in_buf      = nil;     // Shared : x_in for post_conv (T_in * IC_in)
    id<MTLBuffer> post_out_buf     = nil;     // Private : conv_post output (T_in * 22)
    id<MTLBuffer> istft_tmp_y_buf  = nil;     // Private : [n_frames * n_fft]
    id<MTLBuffer> istft_tmp_d_buf  = nil;     // Private : [n_frames * n_fft]
    id<MTLBuffer> istft_y_buf      = nil;     // Private : [padded_len]
    id<MTLBuffer> istft_denom_buf  = nil;     // Private : [padded_len]
    id<MTLBuffer> istft_audio_buf  = nil;     // Shared  : [out_len], downloaded

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

    // Upload via a staging buffer into GPU-private memory. The blit is synchronous
    // but only happens once per weight tensor for the lifetime of the state.
    id<MTLBuffer> staging = [state->device newBufferWithBytes:tensor->data
                                                       length:bytes
                                                      options:MTLResourceStorageModeShared];
    if (staging == nil) return nil;

    id<MTLBuffer> gpu_buf = [state->device newBufferWithLength:bytes
                                                       options:MTLResourceStorageModePrivate];
    if (gpu_buf == nil) return nil;

    id<MTLCommandBuffer> cmd = [state->queue commandBuffer];
    if (cmd == nil) return nil;

    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:gpu_buf destinationOffset:0 size:bytes];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    cached.buffer = gpu_buf;
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

static id<MTLComputePipelineState> select_conv1d_pipeline(MetalVocoderState * state, int weight_type) {
    return weight_type == GGML_TYPE_F16 ? state->conv1d_generic_f16 : state->conv1d_generic_f32;
}

// Ensure a private (GPU-only) scratch buffer of at least `bytes`.
static id<MTLBuffer> ensure_private_buffer(id<MTLDevice> device, id<MTLBuffer> & buf, size_t bytes) {
    if (buf != nil && [buf length] >= bytes) return buf;
    buf = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    return buf;
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

    const NSUInteger max_tg = pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger tg_y   = std::min<NSUInteger>(static_cast<NSUInteger>(OC), 8);
    const NSUInteger tg_x   = std::min<NSUInteger>(max_tg / tg_y, 32);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(OL), static_cast<NSUInteger>(OC), 1)
   threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
    [enc endEncoding];
    return true;
}

// Encode adain+snake into an existing command buffer.
// norm_w, norm_b, alpha must be contiguous F32 tensors of shape [C].
// gamma_ptr / beta_ptr are CPU-side projected float arrays of length C,
// already uploaded into style_proj_buf at the specified byte offsets.
static bool encode_adain_snake(
    MetalVocoderState   * state,
    id<MTLCommandBuffer>  cmd,
    id<MTLBuffer>         x_buf,
    NSUInteger            x_off,
    id<MTLBuffer>         out_buf,
    NSUInteger            out_off,
    id<MTLBuffer>         proj_buf,   // holds all projected gamma/beta
    NSUInteger            gamma_off,  // byte offset of gamma[C] in proj_buf
    NSUInteger            beta_off,   // byte offset of beta[C]  in proj_buf
    const ggml_tensor   * norm_w,
    const ggml_tensor   * norm_b,
    const ggml_tensor   * alpha,
    int64_t               T,
    int64_t               C
) {
    id<MTLBuffer> nw_buf = cached_constant_buffer(state, norm_w);
    id<MTLBuffer> nb_buf = cached_constant_buffer(state, norm_b);
    id<MTLBuffer> al_buf = cached_constant_buffer(state, alpha);
    if (nw_buf == nil || nb_buf == nil || al_buf == nil) return false;

    CppAdainSnakeArgs args {
        static_cast<uint32_t>(T),
        static_cast<uint32_t>(C),
        1e-5f,
    };

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (enc == nil) return false;

    [enc setComputePipelineState:state->adain_snake];
    [enc setBytes:&args     length:sizeof(args) atIndex:0];
    [enc setBuffer:x_buf    offset:x_off        atIndex:1];
    [enc setBuffer:nw_buf   offset:0            atIndex:2];
    [enc setBuffer:nb_buf   offset:0            atIndex:3];
    [enc setBuffer:proj_buf offset:gamma_off    atIndex:4];
    [enc setBuffer:proj_buf offset:beta_off     atIndex:5];
    [enc setBuffer:al_buf   offset:0            atIndex:6];
    [enc setBuffer:out_buf  offset:out_off      atIndex:7];
    [enc setThreadgroupMemoryLength:2 * kAdainTgSize * sizeof(float) atIndex:0];

    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(C), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(kAdainTgSize, 1, 1)];
    [enc endEncoding];
    return true;
}

// Encode regular conv1d (stride=1, symmetric padding) into an existing command buffer.
static bool encode_conv1d(
    MetalVocoderState   * state,
    id<MTLCommandBuffer>  cmd,
    id<MTLBuffer>         in_buf,
    NSUInteger            in_off,
    const ggml_tensor   * weight,
    const ggml_tensor   * bias,
    id<MTLBuffer>         out_buf,
    NSUInteger            out_off,
    int64_t               IL,
    int64_t               IC,
    int64_t               OL,
    int64_t               OC,
    int                   kernel_size,
    int                   dilation,
    int                   padding
) {
    id<MTLBuffer> w_buf = cached_constant_buffer(state, weight);
    id<MTLBuffer> b_buf = cached_constant_buffer(state, bias);
    if (w_buf == nil || b_buf == nil) return false;

    CppConv1dArgs args {
        static_cast<uint32_t>(IL),
        static_cast<uint32_t>(IC),
        static_cast<uint32_t>(OC),
        static_cast<uint32_t>(OL),
        static_cast<uint32_t>(kernel_size),
        static_cast<uint32_t>(dilation),
        static_cast<uint32_t>(padding),
    };

    id<MTLComputePipelineState> pipeline = select_conv1d_pipeline(state, weight->type);
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (enc == nil) return false;

    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args   length:sizeof(args) atIndex:0];
    [enc setBuffer:in_buf  offset:in_off      atIndex:1];
    [enc setBuffer:w_buf   offset:0           atIndex:2];
    [enc setBuffer:b_buf   offset:0           atIndex:3];
    [enc setBuffer:out_buf offset:out_off     atIndex:4];

    const NSUInteger max_tg = pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger tg_y   = std::min<NSUInteger>(static_cast<NSUInteger>(OC), 8);
    const NSUInteger tg_x   = std::min<NSUInteger>(max_tg / tg_y, 32);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(OL), static_cast<NSUInteger>(OC), 1)
   threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
    [enc endEncoding];
    return true;
}

// Encode elementwise in-place add: a[i] += b[i] for n_elems elements.
static bool encode_add_inplace(
    MetalVocoderState   * state,
    id<MTLCommandBuffer>  cmd,
    id<MTLBuffer>         a_buf,
    NSUInteger            a_off,
    id<MTLBuffer>         b_buf,
    NSUInteger            b_off,
    size_t                n_elems
) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (enc == nil) return false;

    // Temporarily re-bind a_buf and b_buf at their offsets by passing them normally.
    // Metal offsets are byte-based; n_elems is float count.
    [enc setComputePipelineState:state->add_inplace];
    [enc setBuffer:a_buf offset:a_off atIndex:0];
    [enc setBuffer:b_buf offset:b_off atIndex:1];
    [enc dispatchThreads:MTLSizeMake(n_elems, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    return true;
}

// Encoders for the new fused-generator kernels.

static bool encode_leaky_relu_inplace(
    MetalVocoderState   * state,
    id<MTLCommandBuffer>  cmd,
    id<MTLBuffer>         buf,
    NSUInteger            off,
    float                 slope,
    size_t                n
) {
    if (state->leaky_relu_inplace == nil) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (enc == nil) return false;

    CppLeakyReluArgs args { slope, static_cast<uint32_t>(n) };
    [enc setComputePipelineState:state->leaky_relu_inplace];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:buf offset:off atIndex:1];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    return true;
}

static bool encode_conv1d_strided(
    MetalVocoderState   * state,
    id<MTLCommandBuffer>  cmd,
    id<MTLBuffer>         in_buf,
    NSUInteger            in_off,
    const ggml_tensor   * weight,
    const ggml_tensor   * bias,
    id<MTLBuffer>         out_buf,
    NSUInteger            out_off,
    int64_t IL, int64_t IC, int64_t OL, int64_t OC,
    int kernel_size, int stride, int dilation, int padding
) {
    id<MTLBuffer> w_buf = cached_constant_buffer(state, weight);
    id<MTLBuffer> b_buf = cached_constant_buffer(state, bias);
    if (w_buf == nil || b_buf == nil) return false;

    id<MTLComputePipelineState> pipeline = weight->type == GGML_TYPE_F16
        ? state->conv1d_strided_f16 : state->conv1d_strided_f32;
    if (pipeline == nil) return false;

    CppConv1dStridedArgs args {
        static_cast<uint32_t>(IL), static_cast<uint32_t>(IC),
        static_cast<uint32_t>(OC), static_cast<uint32_t>(OL),
        static_cast<uint32_t>(kernel_size),
        static_cast<uint32_t>(stride),
        static_cast<uint32_t>(dilation),
        static_cast<uint32_t>(padding),
    };

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (enc == nil) return false;

    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:in_buf  offset:in_off  atIndex:1];
    [enc setBuffer:w_buf   offset:0       atIndex:2];
    [enc setBuffer:b_buf   offset:0       atIndex:3];
    [enc setBuffer:out_buf offset:out_off atIndex:4];

    const NSUInteger max_tg = pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger tg_y   = std::min<NSUInteger>(static_cast<NSUInteger>(OC), 8);
    const NSUInteger tg_x   = std::min<NSUInteger>(max_tg / tg_y, 32);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(OL),
                                      static_cast<NSUInteger>(OC), 1)
   threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
    [enc endEncoding];
    return true;
}

static bool encode_pad_reflect_left1(
    MetalVocoderState   * state,
    id<MTLCommandBuffer>  cmd,
    id<MTLBuffer>         in_buf,  NSUInteger in_off,
    id<MTLBuffer>         out_buf, NSUInteger out_off,
    int64_t T_in, int64_t C
) {
    if (state->pad_reflect_left1 == nil) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (enc == nil) return false;

    CppPadReflectArgs args {
        static_cast<uint32_t>(T_in),
        static_cast<uint32_t>(T_in + 1),
        static_cast<uint32_t>(C),
    };

    [enc setComputePipelineState:state->pad_reflect_left1];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:in_buf  offset:in_off  atIndex:1];
    [enc setBuffer:out_buf offset:out_off atIndex:2];

    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(T_in + 1),
                                      static_cast<NSUInteger>(C), 1)
   threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [enc endEncoding];
    return true;
}

static bool encode_weighted_sum3_scale(
    MetalVocoderState   * state,
    id<MTLCommandBuffer>  cmd,
    id<MTLBuffer>         a_buf, NSUInteger a_off,
    id<MTLBuffer>         b_buf, NSUInteger b_off,
    id<MTLBuffer>         c_buf, NSUInteger c_off,
    id<MTLBuffer>         out_buf, NSUInteger out_off,
    float scale, size_t n
) {
    if (state->weighted_sum3_scale == nil) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (enc == nil) return false;

    CppWeightedSum3Args args { scale, static_cast<uint32_t>(n) };
    [enc setComputePipelineState:state->weighted_sum3_scale];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:a_buf   offset:a_off   atIndex:1];
    [enc setBuffer:b_buf   offset:b_off   atIndex:2];
    [enc setBuffer:c_buf   offset:c_off   atIndex:3];
    [enc setBuffer:out_buf offset:out_off atIndex:4];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    return true;
}

static bool encode_copy(
    MetalVocoderState   * state,
    id<MTLCommandBuffer>  cmd,
    id<MTLBuffer>         dst_buf, NSUInteger dst_off,
    id<MTLBuffer>         src_buf, NSUInteger src_off,
    size_t n
) {
    if (state->copy_f32 == nil) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (enc == nil) return false;

    [enc setComputePipelineState:state->copy_f32];
    [enc setBuffer:dst_buf offset:dst_off atIndex:0];
    [enc setBuffer:src_buf offset:src_off atIndex:1];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    return true;
}

// Encode the 3-iteration resblock body into an existing command buffer.
// The accumulator x_acc_buf is read AND mutated (residual adds at end of
// each iter). Caller must initialize x_acc_buf with the desired input
// before this call. ping/pong are scratch buffers (private storage OK).
// sp_buf holds 12*IC consecutive floats for this resblock's projections.
// All conv kernel_size / dilation / padding values come from iters[iter].
static bool encode_resblock_into(
    MetalVocoderState   * state,
    id<MTLCommandBuffer>  cmd,
    id<MTLBuffer> x_acc_buf, NSUInteger x_acc_off,
    id<MTLBuffer> ping,
    id<MTLBuffer> pong,
    id<MTLBuffer> sp_buf, NSUInteger sp_base_off,
    int64_t T, int64_t IC,
    const MetalResblockIterWeights iters[3]
) {
    const NSUInteger ic_bytes = static_cast<NSUInteger>(IC) * sizeof(float);
    for (int iter = 0; iter < 3; ++iter) {
        const MetalResblockIterWeights & w = iters[iter];
        const NSUInteger base   = sp_base_off + static_cast<NSUInteger>(iter * 4) * ic_bytes;
        const NSUInteger g1_off = base + 0 * ic_bytes;
        const NSUInteger b1_off = base + 1 * ic_bytes;
        const NSUInteger g2_off = base + 2 * ic_bytes;
        const NSUInteger b2_off = base + 3 * ic_bytes;

        if (!encode_adain_snake(state, cmd, x_acc_buf, x_acc_off, ping, 0,
                                sp_buf, g1_off, b1_off,
                                w.adain1_norm_w, w.adain1_norm_b, w.snake1_alpha,
                                T, IC)) return false;
        if (!encode_conv1d(state, cmd, ping, 0,
                           w.conv1_w, w.conv1_b, pong, 0,
                           T, IC, T, IC,
                           w.conv1_kernel, w.conv1_dilation, w.conv1_padding)) return false;
        if (!encode_adain_snake(state, cmd, pong, 0, ping, 0,
                                sp_buf, g2_off, b2_off,
                                w.adain2_norm_w, w.adain2_norm_b, w.snake2_alpha,
                                T, IC)) return false;
        if (!encode_conv1d(state, cmd, ping, 0,
                           w.conv2_w, w.conv2_b, pong, 0,
                           T, IC, T, IC,
                           w.conv2_kernel, w.conv2_dilation, w.conv2_padding)) return false;
        if (!encode_add_inplace(state, cmd, x_acc_buf, x_acc_off, pong, 0,
                                static_cast<size_t>(T * IC))) return false;
    }
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
        ok &= load_pipeline_pair(state, state->adain_snake,        "kokopop_adain_snake");
        ok &= load_pipeline_pair(state, state->conv1d_generic_f32, "kokopop_conv1d_bias_generic_f32");
        ok &= load_pipeline_pair(state, state->conv1d_generic_f16, "kokopop_conv1d_bias_generic_f16");
        ok &= load_pipeline_pair(state, state->add_inplace,        "kokopop_add_inplace_f32");
        // Pipelines used exclusively by metal_vocoder_run_stage. We load all
        // of them unconditionally (no short-circuit) so any failure is logged
        // once and a single warning summarises the situation. If any pipeline
        // ends up null, metal_vocoder_run_stage detects it at call time,
        // returns false, and audio_utils.cpp falls back to the per-op graph.
        bool fused_ok = true;
        fused_ok = load_pipeline_pair(state, state->leaky_relu_inplace,  "kokopop_leaky_relu_inplace_f32") && fused_ok;
        fused_ok = load_pipeline_pair(state, state->conv1d_strided_f32,  "kokopop_conv1d_strided_bias_generic_f32") && fused_ok;
        fused_ok = load_pipeline_pair(state, state->conv1d_strided_f16,  "kokopop_conv1d_strided_bias_generic_f16") && fused_ok;
        fused_ok = load_pipeline_pair(state, state->pad_reflect_left1,   "kokopop_pad_reflect_left1_f32") && fused_ok;
        fused_ok = load_pipeline_pair(state, state->weighted_sum3_scale, "kokopop_weighted_sum3_scale_f32") && fused_ok;
        fused_ok = load_pipeline_pair(state, state->copy_f32,            "kokopop_copy_f32") && fused_ok;
        if (!fused_ok) {
            std::fprintf(stderr,
                         "[metal_vocoder] fused per-stage pipelines unavailable — falling back to per-op path\n");
        }

        // iSTFT pipelines + window/twiddle buffers. Failure is non-fatal:
        // metal_vocoder_run_post_istft detects missing pipelines and returns false,
        // so audio_utils falls back to the CPU/Metal-stft path.
        bool istft_ok = true;
        istft_ok = load_pipeline_pair(state, state->istft_idft, "kokopop_istft_idft_v") && istft_ok;
        istft_ok = load_pipeline_pair(state, state->istft_ola,  "kokopop_istft_ola_v")  && istft_ok;
        istft_ok = load_pipeline_pair(state, state->istft_norm, "kokopop_istft_norm_v") && istft_ok;
        if (istft_ok) {
            const int n_fft = KOKOPOP_STFT_N;
            const int hop   = KOKOPOP_STFT_HOP;
            const int n_bins = n_fft / 2 + 1;
            state->istft_n_fft  = n_fft;
            state->istft_hop    = hop;
            state->istft_n_bins = n_bins;

            const size_t win_bytes = static_cast<size_t>(n_fft) * sizeof(float);
            state->istft_window_buf =
                [state->device newBufferWithLength:win_bytes options:MTLResourceStorageModeShared];
            const size_t tw_bytes = static_cast<size_t>(n_bins) * static_cast<size_t>(n_fft) * sizeof(float);
            state->istft_tw_c_buf =
                [state->device newBufferWithLength:tw_bytes options:MTLResourceStorageModeShared];
            state->istft_tw_s_buf =
                [state->device newBufferWithLength:tw_bytes options:MTLResourceStorageModeShared];

            if (state->istft_window_buf != nil &&
                state->istft_tw_c_buf != nil &&
                state->istft_tw_s_buf != nil) {
                float * win  = static_cast<float *>([state->istft_window_buf contents]);
                float * tw_c = static_cast<float *>([state->istft_tw_c_buf contents]);
                float * tw_s = static_cast<float *>([state->istft_tw_s_buf contents]);
                const float inv_N = 1.0f / static_cast<float>(n_fft);
                for (int n = 0; n < n_fft; ++n) {
                    win[n] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * n * inv_N);
                }
                for (int k = 0; k < n_bins; ++k) {
                    for (int n = 0; n < n_fft; ++n) {
                        const float angle = 2.0f * static_cast<float>(M_PI) *
                                            static_cast<float>(k * n) * inv_N;
                        tw_c[k * n_fft + n] = std::cos(angle);
                        tw_s[k * n_fft + n] = std::sin(angle);
                    }
                }
            } else {
                std::fprintf(stderr, "[metal_vocoder] iSTFT scratch alloc failed\n");
                state->istft_idft = nil;
                state->istft_ola = nil;
                state->istft_norm = nil;
            }
        } else {
            std::fprintf(stderr,
                         "[metal_vocoder] iSTFT pipelines unavailable — post_conv+iSTFT fused path disabled\n");
        }

        if (!ok) {
            delete state;
            return nullptr;
        }

        return state;
    }
}

// -----------------------------------------------------------------------------
// Generator ResBlock GPU execution
// -----------------------------------------------------------------------------
// Runs 3 ResBlock iterations entirely on GPU (one command buffer, no CPU sync
// between kernels). Intermediate tensors stay in GPU-private memory.
//
// style_projs layout (12 * IC floats):
//   [0*IC .. 1*IC) : iter0 adain1 gamma
//   [1*IC .. 2*IC) : iter0 adain1 beta
//   [2*IC .. 3*IC) : iter0 adain2 gamma
//   [3*IC .. 4*IC) : iter0 adain2 beta
//   [4*IC .. 5*IC) : iter1 adain1 gamma
//   ...
bool metal_vocoder_run_generator_resblocks(
    MetalVocoderState              * state,
    const float                    * x_data,
    float                          * out_data,
    int64_t                          T,
    int64_t                          IC,
    const float                    * style_projs,   // [12, IC]
    const MetalResblockIterWeights   iters[3]
) {
    @autoreleasepool {
        if (state == nullptr || x_data == nullptr || out_data == nullptr ||
            style_projs == nullptr || T <= 0 || IC <= 0) return false;

        if (state->adain_snake == nil || state->conv1d_generic_f32 == nil ||
            state->add_inplace == nil) return false;

        const size_t tensor_bytes      = static_cast<size_t>(T * IC) * sizeof(float);
        const size_t style_proj_bytes  = 12 * static_cast<size_t>(IC) * sizeof(float);

        id<MTLBuffer> x_buf   = ensure_scratch_buffer(state->device, state->x_buf,   tensor_bytes);
        id<MTLBuffer> ping    = ensure_private_buffer(state->device, state->ping_buf, tensor_bytes);
        id<MTLBuffer> pong    = ensure_private_buffer(state->device, state->pong_buf, tensor_bytes);
        id<MTLBuffer> sp_buf  = ensure_scratch_buffer(state->device, state->style_proj_buf, style_proj_bytes);
        if (x_buf == nil || ping == nil || pong == nil || sp_buf == nil) return false;

        // Upload input and style projections (both Shared → visible to GPU immediately).
        std::memcpy([x_buf  contents], x_data,      tensor_bytes);
        std::memcpy([sp_buf contents], style_projs, style_proj_bytes);

        id<MTLCommandBuffer> cmd = [state->queue commandBuffer];
        if (cmd == nil) return false;

        const NSUInteger float_sz = sizeof(float);
        const NSUInteger ic_bytes = static_cast<NSUInteger>(IC) * float_sz;
        bool ok = true;

        for (int iter = 0; iter < 3 && ok; ++iter) {
            const MetalResblockIterWeights & w = iters[iter];

            // Style projection offsets for this iteration (4 slots × IC floats each).
            const NSUInteger base    = static_cast<NSUInteger>(iter * 4) * ic_bytes;
            const NSUInteger g1_off  = base + 0 * ic_bytes;
            const NSUInteger b1_off  = base + 1 * ic_bytes;
            const NSUInteger g2_off  = base + 2 * ic_bytes;
            const NSUInteger b2_off  = base + 3 * ic_bytes;

            // Step 1: adain1 + snake1  (x_buf → ping)
            ok = encode_adain_snake(state, cmd,
                                    x_buf, 0, ping, 0,
                                    sp_buf, g1_off, b1_off,
                                    w.adain1_norm_w, w.adain1_norm_b, w.snake1_alpha,
                                    T, IC);

            // Step 2: conv1_dilated  (ping → pong)
            if (ok) ok = encode_conv1d(state, cmd,
                                       ping, 0, w.conv1_w, w.conv1_b, pong, 0,
                                       T, IC, T, IC,
                                       w.conv1_kernel, w.conv1_dilation, w.conv1_padding);

            // Step 3: adain2 + snake2  (pong → ping)
            if (ok) ok = encode_adain_snake(state, cmd,
                                            pong, 0, ping, 0,
                                            sp_buf, g2_off, b2_off,
                                            w.adain2_norm_w, w.adain2_norm_b, w.snake2_alpha,
                                            T, IC);

            // Step 4: conv2  (ping → pong)
            if (ok) ok = encode_conv1d(state, cmd,
                                       ping, 0, w.conv2_w, w.conv2_b, pong, 0,
                                       T, IC, T, IC,
                                       w.conv2_kernel, w.conv2_dilation, w.conv2_padding);

            // Step 5: residual add  (x_buf += pong)
            if (ok) ok = encode_add_inplace(state, cmd,
                                            x_buf, 0, pong, 0,
                                            static_cast<size_t>(T * IC));
        }

        if (!ok) return false;

        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal_vocoder] resblock command failed status=%lu\n",
                         static_cast<unsigned long>([cmd status]));
            return false;
        }

        std::memcpy(out_data, [x_buf contents], tensor_bytes);
        return true;
    }
}

// -----------------------------------------------------------------------------
// Fused per-stage generator
// -----------------------------------------------------------------------------
bool metal_vocoder_run_stage(
    MetalVocoderState * state,
    const float * x_in_data,    int64_t T_x_in,
    const float * har_data,     int64_t har_len,    int64_t har_C,
    const float * style_projs_packed,
    int64_t IC_noise, int64_t IC_main,
    float * x_out_data, int64_t T_post_pad,
    int up_stride, int up_padding,
    bool pad_reflect_left1,
    int noise_kernel, int noise_stride, int noise_padding,
    const ggml_tensor * noise_conv_w,
    const ggml_tensor * noise_conv_b,
    const ggml_tensor * up_w,
    const ggml_tensor * up_b,
    const MetalResblockIterWeights noise_iters[3],
    const MetalResblockIterWeights main_iters[3][3]
) {
    @autoreleasepool {
        if (state == nullptr || x_in_data == nullptr || har_data == nullptr ||
            x_out_data == nullptr || style_projs_packed == nullptr ||
            T_x_in <= 0 || har_len <= 0 || har_C <= 0 ||
            T_post_pad <= 0 || IC_noise <= 0 || IC_main <= 0) {
            return false;
        }
        if (state->adain_snake == nil || state->conv1d_generic_f32 == nil ||
            state->add_inplace == nil  || state->leaky_relu_inplace == nil ||
            state->conv1d_strided_f32 == nil || state->pad_reflect_left1 == nil ||
            state->weighted_sum3_scale == nil || state->copy_f32 == nil ||
            state->convt_generic_f32 == nil) {
            return false;
        }

        // Derive shapes from weight tensors.
        const int64_t IC_x_in    = up_w->ne[2];   // convt input channels
        const int64_t IC_x_out   = up_w->ne[1];   // convt output channels
        const int64_t up_K       = up_w->ne[0];
        const int64_t T_convt    = (T_x_in - 1) * up_stride - 2 * up_padding + up_K;
        const int64_t T_pad      = pad_reflect_left1 ? (T_convt + 1) : T_convt;
        const int64_t T_noise    = (har_len + 2 * noise_padding - noise_kernel) / noise_stride + 1;

        if (T_pad != T_post_pad) {
            std::fprintf(stderr,
                "[metal_vocoder_stage] T_post_pad mismatch: computed=%lld provided=%lld\n",
                (long long)T_pad, (long long)T_post_pad);
            return false;
        }
        if (T_noise != T_post_pad) {
            std::fprintf(stderr,
                "[metal_vocoder_stage] T_noise=%lld != T_post_pad=%lld — add would fail\n",
                (long long)T_noise, (long long)T_post_pad);
            return false;
        }
        if (IC_noise != IC_x_out || IC_main != IC_x_out) {
            std::fprintf(stderr,
                "[metal_vocoder_stage] channel mismatch: IC_noise=%lld IC_main=%lld IC_x_out=%lld\n",
                (long long)IC_noise, (long long)IC_main, (long long)IC_x_out);
            return false;
        }

        // Byte sizes.
        const size_t x_in_bytes    = static_cast<size_t>(T_x_in    * IC_x_in)  * sizeof(float);
        const size_t har_bytes     = static_cast<size_t>(har_len   * har_C)    * sizeof(float);
        const size_t source_bytes  = static_cast<size_t>(T_noise   * IC_noise) * sizeof(float);
        const size_t convt_bytes   = static_cast<size_t>(T_convt   * IC_x_out) * sizeof(float);
        const size_t pad_bytes     = static_cast<size_t>(T_post_pad * IC_x_out) * sizeof(float);
        const size_t out_bytes     = pad_bytes;
        const size_t proj_bytes    = static_cast<size_t>(12 * IC_noise + 3 * 12 * IC_main) * sizeof(float);
        const size_t scratch_bytes = std::max(source_bytes, pad_bytes);

        // Allocate / ensure all buffers.
        id<MTLBuffer> x_in_buf  = ensure_scratch_buffer(state->device, state->stage_x_in_buf,         x_in_bytes);
        id<MTLBuffer> har_buf   = ensure_scratch_buffer(state->device, state->stage_har_buf,          har_bytes);
        id<MTLBuffer> sp_buf    = ensure_scratch_buffer(state->device, state->stage_style_proj_buf,   proj_bytes);
        id<MTLBuffer> src_buf   = ensure_private_buffer(state->device, state->stage_x_source_buf,     source_bytes);
        id<MTLBuffer> convt_buf = ensure_private_buffer(state->device, state->stage_x_post_pad_buf,   convt_bytes);
        id<MTLBuffer> pad_buf   = ensure_private_buffer(state->device, state->stage_x_scratch_buf,    pad_bytes);
        id<MTLBuffer> b0_buf    = ensure_private_buffer(state->device, state->stage_branch0_buf,      out_bytes);
        id<MTLBuffer> b1_buf    = ensure_private_buffer(state->device, state->stage_branch1_buf,      out_bytes);
        id<MTLBuffer> b2_buf    = ensure_private_buffer(state->device, state->stage_branch2_buf,      out_bytes);
        id<MTLBuffer> out_buf   = ensure_scratch_buffer(state->device, state->stage_x_out_buf,        out_bytes);
        id<MTLBuffer> ping      = ensure_private_buffer(state->device, state->ping_buf,               scratch_bytes);
        id<MTLBuffer> pong      = ensure_private_buffer(state->device, state->pong_buf,               scratch_bytes);
        if (x_in_buf == nil || har_buf == nil || sp_buf == nil || src_buf == nil ||
            convt_buf == nil || pad_buf == nil || b0_buf == nil || b1_buf == nil ||
            b2_buf == nil || out_buf == nil || ping == nil || pong == nil) {
            return false;
        }

        // Cross-stage cache: when this stage's input is the previous stage's
        // output (same CPU pointer + size), the data is still in
        // stage_prev_out_buf on GPU. Skip the CPU memcpy upload and emit a
        // GPU blit into stage_x_in_buf inside the cmd buffer instead.
        const bool cross_stage_hit =
            (state->prev_out_addr != nullptr) &&
            (state->prev_out_addr == x_in_data) &&
            (state->prev_out_bytes == x_in_bytes) &&
            (state->stage_prev_out_buf != nil) &&
            ([state->stage_prev_out_buf length] >= x_in_bytes);

        if (!cross_stage_hit) {
            std::memcpy([x_in_buf contents], x_in_data, x_in_bytes);
        }

        // Skip the redundant har upload when both stages see the same
        // har_data. The 1 MB memcpy is meaningful at ~50 µs/chunk on M1.
        // Cache is keyed on (CPU pointer, byte size, sentinels) AND the
        // MTLBuffer identity: a freshly grown buffer has no data and must
        // be re-uploaded even if the CPU sentinels match.
        uint64_t har_sent0 = 0, har_sent1 = 0;
        if (har_bytes >= 16) {
            std::memcpy(&har_sent0, har_data, sizeof(uint64_t));
            std::memcpy(&har_sent1,
                        reinterpret_cast<const char *>(har_data) + har_bytes - sizeof(uint64_t),
                        sizeof(uint64_t));
        }
        const bool har_hit =
            (state->prev_har_addr != nullptr) &&
            (state->prev_har_addr == har_data) &&
            (state->prev_har_bytes == har_bytes) &&
            (state->prev_har_sentinel0 == har_sent0) &&
            (state->prev_har_sentinel1 == har_sent1) &&
            (state->stage_har_buf == har_buf);  // same MTLBuffer instance
        if (!har_hit) {
            std::memcpy([har_buf contents], har_data, har_bytes);
            state->prev_har_addr      = har_data;
            state->prev_har_bytes     = har_bytes;
            state->prev_har_sentinel0 = har_sent0;
            state->prev_har_sentinel1 = har_sent1;
        }

        std::memcpy([sp_buf   contents], style_projs_packed,  proj_bytes);

        id<MTLCommandBuffer> cmd = [state->queue commandBuffer];
        if (cmd == nil) return false;

        if (cross_stage_hit) {
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:state->stage_prev_out_buf sourceOffset:0
                        toBuffer:x_in_buf destinationOffset:0
                            size:x_in_bytes];
            [blit endEncoding];
        }
        // Invalidate cache before potentially refreshing it below — a failure
        // path past this point must not leave a stale entry.
        state->prev_out_addr = nullptr;
        state->prev_out_bytes = 0;

        bool ok = true;

        // 1) leaky_relu(x_in, 0.1) in place.
        if (ok) ok = encode_leaky_relu_inplace(state, cmd, x_in_buf, 0, 0.1f,
                                                static_cast<size_t>(T_x_in * IC_x_in));

        // 2) x_source = conv1d_strided(har, noise_conv_w, noise_conv_b)
        if (ok) ok = encode_conv1d_strided(state, cmd,
                                            har_buf, 0,
                                            noise_conv_w, noise_conv_b,
                                            src_buf, 0,
                                            har_len, har_C, T_noise, IC_noise,
                                            noise_kernel, noise_stride, /*dilation=*/1, noise_padding);

        // 3) x_source = resblock_noise(x_source) — in-place via residual adds
        const NSUInteger noise_proj_off = 0;
        if (ok) ok = encode_resblock_into(state, cmd, src_buf, 0, ping, pong,
                                           sp_buf, noise_proj_off,
                                           T_noise, IC_noise, noise_iters);

        // 4) x_post = conv_transpose1d(x_in, up_w, up_b) → convt_buf
        if (ok) {
            ok = metal_vocoder_conv_transpose1d_crop_bias_encode(
                state, cmd,
                x_in_buf, 0,
                up_w, up_b,
                convt_buf, 0,
                T_x_in, IC_x_in,
                T_convt, IC_x_out,
                up_stride, /*crop_left=*/up_padding);
        }

        // 5) Reflect-pad if needed → pad_buf, else point pad_buf := convt_buf
        id<MTLBuffer> sink_buf = convt_buf;  // where we'll accumulate
        if (pad_reflect_left1) {
            if (ok) ok = encode_pad_reflect_left1(state, cmd,
                                                   convt_buf, 0, pad_buf, 0,
                                                   T_convt, IC_x_out);
            sink_buf = pad_buf;
        }

        // 6) sink_buf += x_source
        if (ok) ok = encode_add_inplace(state, cmd,
                                         sink_buf, 0, src_buf, 0,
                                         static_cast<size_t>(T_post_pad * IC_x_out));

        // 7) 3 main resblocks — each starts from a fresh copy of sink_buf
        id<MTLBuffer> branch_bufs[3] = { b0_buf, b1_buf, b2_buf };
        const NSUInteger main_per_branch = static_cast<NSUInteger>(12 * IC_main) * sizeof(float);
        const NSUInteger main_proj_base  = static_cast<NSUInteger>(12 * IC_noise) * sizeof(float);
        for (int br = 0; br < 3 && ok; ++br) {
            ok = encode_copy(state, cmd,
                             branch_bufs[br], 0,
                             sink_buf, 0,
                             static_cast<size_t>(T_post_pad * IC_x_out));
            if (!ok) break;

            const NSUInteger off = main_proj_base + static_cast<NSUInteger>(br) * main_per_branch;
            ok = encode_resblock_into(state, cmd, branch_bufs[br], 0, ping, pong,
                                       sp_buf, off,
                                       T_post_pad, IC_main, main_iters[br]);
        }

        // 8) out = (b0 + b1 + b2) / 3
        if (ok) ok = encode_weighted_sum3_scale(state, cmd,
                                                 b0_buf, 0, b1_buf, 0, b2_buf, 0,
                                                 out_buf, 0,
                                                 1.0f / 3.0f,
                                                 static_cast<size_t>(T_post_pad * IC_x_out));

        if (!ok) return false;

        // Persist this stage's output for the next call so it can blit
        // from GPU instead of memcpy from CPU. Encoded inside the same command
        // buffer; Metal hazard tracking serialises the write→read sequence.
        id<MTLBuffer> prev_buf = ensure_private_buffer(state->device, state->stage_prev_out_buf, out_bytes);
        if (prev_buf != nil) {
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:out_buf sourceOffset:0
                        toBuffer:prev_buf destinationOffset:0
                            size:out_bytes];
            [blit endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal_vocoder] stage command failed status=%lu\n",
                         static_cast<unsigned long>([cmd status]));
            return false;
        }

        std::memcpy(x_out_data, [out_buf contents], out_bytes);
        if (prev_buf != nil) {
            state->prev_out_addr  = x_out_data;
            state->prev_out_bytes = out_bytes;
        }
        return true;
    }
}

// -----------------------------------------------------------------------------
// Fused post_conv + iSTFT (7.2)
// -----------------------------------------------------------------------------
bool metal_vocoder_post_istft_available(const MetalVocoderState * state) {
    return state != nullptr &&
           state->istft_idft != nil &&
           state->istft_ola != nil &&
           state->istft_norm != nil &&
           state->istft_window_buf != nil &&
           state->istft_tw_c_buf != nil &&
           state->istft_tw_s_buf != nil &&
           state->leaky_relu_inplace != nil &&
           state->conv1d_generic_f32 != nil;
}

bool metal_vocoder_run_post_istft(
    MetalVocoderState * state,
    const float * x_in_data, int64_t T_in, int64_t IC_in,
    const ggml_tensor * post_w,
    const ggml_tensor * post_b,
    int n_frames, int out_len,
    float * audio_out
) {
    @autoreleasepool {
        if (state == nullptr || x_in_data == nullptr || audio_out == nullptr ||
            post_w == nullptr || post_b == nullptr ||
            T_in <= 0 || IC_in <= 0 || n_frames <= 0 || out_len <= 0) {
            return false;
        }
        if (state->istft_idft == nil || state->istft_ola == nil || state->istft_norm == nil ||
            state->leaky_relu_inplace == nil || state->conv1d_generic_f32 == nil ||
            state->istft_window_buf == nil || state->istft_tw_c_buf == nil ||
            state->istft_tw_s_buf == nil) {
            return false;
        }

        // post_conv weight is laid out [K*IC, OC] in ggml (matches
        // kokopop_conv1d_bias_generic).
        const int64_t K  = 7;
        const int64_t OC = 22;
        if (post_w->ne[0] != K * IC_in || post_w->ne[1] != OC ||
            post_b->ne[0] != OC) {
            std::fprintf(stderr, "[metal_vocoder] post_istft: unexpected post_conv shape\n");
            return false;
        }

        const int n_fft = state->istft_n_fft;
        const int hop   = state->istft_hop;
        const int padded_len = n_fft + hop * (n_frames - 1);
        if (padded_len <= 0) return false;

        const size_t x_in_bytes   = static_cast<size_t>(T_in * IC_in) * sizeof(float);
        const size_t post_bytes   = static_cast<size_t>(T_in * OC)    * sizeof(float);
        const size_t tmp_bytes    = static_cast<size_t>(n_frames) * static_cast<size_t>(n_fft) * sizeof(float);
        const size_t pad_bytes    = static_cast<size_t>(padded_len) * sizeof(float);
        const size_t audio_bytes  = static_cast<size_t>(out_len) * sizeof(float);

        id<MTLBuffer> x_in_buf  = ensure_scratch_buffer(state->device, state->post_in_buf,     x_in_bytes);
        id<MTLBuffer> post_buf  = ensure_private_buffer(state->device, state->post_out_buf,    post_bytes);
        id<MTLBuffer> tmp_y     = ensure_private_buffer(state->device, state->istft_tmp_y_buf, tmp_bytes);
        id<MTLBuffer> tmp_d     = ensure_private_buffer(state->device, state->istft_tmp_d_buf, tmp_bytes);
        id<MTLBuffer> y_buf     = ensure_private_buffer(state->device, state->istft_y_buf,     pad_bytes);
        id<MTLBuffer> denom_buf = ensure_private_buffer(state->device, state->istft_denom_buf, pad_bytes);
        id<MTLBuffer> audio_buf = ensure_scratch_buffer(state->device, state->istft_audio_buf, audio_bytes);
        if (x_in_buf == nil || post_buf == nil || tmp_y == nil || tmp_d == nil ||
            y_buf == nil || denom_buf == nil || audio_buf == nil) {
            return false;
        }

        // Cross-stage cache: skip CPU upload if pointer matches.
        const bool cross_stage_hit =
            (state->prev_out_addr != nullptr) &&
            (state->prev_out_addr == x_in_data) &&
            (state->prev_out_bytes == x_in_bytes) &&
            (state->stage_prev_out_buf != nil) &&
            ([state->stage_prev_out_buf length] >= x_in_bytes);

        if (!cross_stage_hit) {
            std::memcpy([x_in_buf contents], x_in_data, x_in_bytes);
        }

        id<MTLCommandBuffer> cmd = [state->queue commandBuffer];
        if (cmd == nil) return false;

        if (cross_stage_hit) {
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:state->stage_prev_out_buf sourceOffset:0
                        toBuffer:x_in_buf destinationOffset:0
                            size:x_in_bytes];
            [blit endEncoding];
        }
        state->prev_out_addr = nullptr;
        state->prev_out_bytes = 0;

        // 1) leaky_relu(x_in, 0.01) in place.
        if (!encode_leaky_relu_inplace(state, cmd, x_in_buf, 0, 0.01f,
                                        static_cast<size_t>(T_in * IC_in))) return false;

        // 2) conv1d (post_conv): K=7, stride=1, padding=3, IC=IC_in, OC=22.
        if (!encode_conv1d(state, cmd, x_in_buf, 0,
                           post_w, post_b, post_buf, 0,
                           T_in, IC_in, T_in, OC,
                           /*kernel=*/7, /*dilation=*/1, /*padding=*/3)) return false;

        const uint32_t v_nfft       = static_cast<uint32_t>(n_fft);
        const uint32_t v_hop        = static_cast<uint32_t>(hop);
        const uint32_t v_n_frames   = static_cast<uint32_t>(n_frames);
        const uint32_t v_padded_len = static_cast<uint32_t>(padded_len);
        const uint32_t v_center_pad = static_cast<uint32_t>(n_fft / 2);
        const uint32_t v_out_len    = static_cast<uint32_t>(out_len);

        // 3a) IDFT.
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil) return false;
            [enc setComputePipelineState:state->istft_idft];
            [enc setBuffer:tmp_y                offset:0 atIndex:0];
            [enc setBuffer:tmp_d                offset:0 atIndex:1];
            [enc setBuffer:post_buf             offset:0 atIndex:2];
            [enc setBuffer:state->istft_window_buf offset:0 atIndex:3];
            [enc setBuffer:state->istft_tw_c_buf   offset:0 atIndex:4];
            [enc setBuffer:state->istft_tw_s_buf   offset:0 atIndex:5];
            [enc setBytes:&v_nfft     length:sizeof(v_nfft)     atIndex:6];
            [enc setBytes:&v_n_frames length:sizeof(v_n_frames) atIndex:7];
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n_frames),
                                              static_cast<NSUInteger>(n_fft), 1)
           threadsPerThreadgroup:MTLSizeMake(16, static_cast<NSUInteger>(n_fft), 1)];
            [enc endEncoding];
        }

        // 3b) Overlap-add.
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil) return false;
            [enc setComputePipelineState:state->istft_ola];
            [enc setBuffer:y_buf      offset:0 atIndex:0];
            [enc setBuffer:denom_buf  offset:0 atIndex:1];
            [enc setBuffer:tmp_y      offset:0 atIndex:2];
            [enc setBuffer:tmp_d      offset:0 atIndex:3];
            [enc setBytes:&v_nfft       length:sizeof(v_nfft)       atIndex:4];
            [enc setBytes:&v_hop        length:sizeof(v_hop)        atIndex:5];
            [enc setBytes:&v_n_frames   length:sizeof(v_n_frames)   atIndex:6];
            [enc setBytes:&v_padded_len length:sizeof(v_padded_len) atIndex:7];
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(padded_len), 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // 3c) Normalize / trim / clamp.
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (enc == nil) return false;
            [enc setComputePipelineState:state->istft_norm];
            [enc setBuffer:audio_buf offset:0 atIndex:0];
            [enc setBuffer:y_buf     offset:0 atIndex:1];
            [enc setBuffer:denom_buf offset:0 atIndex:2];
            [enc setBytes:&v_center_pad length:sizeof(v_center_pad) atIndex:3];
            [enc setBytes:&v_out_len    length:sizeof(v_out_len)    atIndex:4];
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(out_len), 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal_vocoder] post_istft command failed status=%lu\n",
                         static_cast<unsigned long>([cmd status]));
            return false;
        }

        std::memcpy(audio_out, [audio_buf contents], audio_bytes);
        return true;
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

        if (metal_profile_enabled())
            std::fprintf(stderr, "[convt] IL=%lld IC=%lld K=%lld OC=%lld OL=%lld stride=%d\n",
                         (long long)IL, (long long)IC, (long long)K, (long long)OC, (long long)OL, stride);

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
