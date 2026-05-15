#pragma once

#include <ggml.h>

#ifdef KOKOPOP_HAS_METAL

namespace kokopop {

struct MetalVocoderState;

// Per-iteration weights for one generator ResBlock iteration.
// Passed as an array of 3 to metal_vocoder_run_generator_resblocks.
struct MetalResblockIterWeights {
    // adain1 + snake1
    const ggml_tensor * adain1_norm_w = nullptr;  // [IC]
    const ggml_tensor * adain1_norm_b = nullptr;  // [IC]
    const ggml_tensor * snake1_alpha  = nullptr;  // [1, IC]
    // conv1_dilated
    const ggml_tensor * conv1_w       = nullptr;  // [K, IC, IC]
    const ggml_tensor * conv1_b       = nullptr;  // [IC]
    int                 conv1_kernel   = 3;
    int                 conv1_dilation = 1;
    int                 conv1_padding  = 1;
    // adain2 + snake2
    const ggml_tensor * adain2_norm_w = nullptr;
    const ggml_tensor * adain2_norm_b = nullptr;
    const ggml_tensor * snake2_alpha  = nullptr;
    // conv2
    const ggml_tensor * conv2_w       = nullptr;
    const ggml_tensor * conv2_b       = nullptr;
    int                 conv2_kernel   = 3;
    int                 conv2_dilation = 1;
    int                 conv2_padding  = 1;
};

MetalVocoderState * metal_vocoder_create();
void                metal_vocoder_destroy(MetalVocoderState *);

bool metal_vocoder_conv_transpose1d_crop_bias(
    MetalVocoderState * state,
    const ggml_tensor * input,
    const ggml_tensor * weight,
    const ggml_tensor * bias,
    ggml_tensor       * output,
    int stride,
    int crop_left);

// Run 3 generator ResBlock iterations on GPU in a single command buffer.
//
// x_data / out_data : CPU pointers, layout [T, IC] (T=ne[0], IC=ne[1] in ggml).
// style_projs       : 12 * IC floats — pre-projected gamma/beta for all 6 adain
//                     calls, in order: iter0_adain1_gamma, iter0_adain1_beta,
//                     iter0_adain2_gamma, iter0_adain2_beta, iter1_*, iter2_*.
// iters             : array of 3 MetalResblockIterWeights.
bool metal_vocoder_run_generator_resblocks(
    MetalVocoderState              * state,
    const float                    * x_data,
    float                          * out_data,
    int64_t                          T,
    int64_t                          IC,
    const float                    * style_projs,
    const MetalResblockIterWeights   iters[3]);

// Fused per-stage generator. Executes one full generator stage on GPU as
// a single command buffer:
//   1. leaky_relu(x_in, 0.1)                                 [inplace]
//   2. x_source = conv1d_strided(har, noise_conv_w, noise_conv_b)
//   3. x_source = resblock(x_source, noise_iters)            [inplace]
//   4. x_post = conv_transpose1d(x_in, up_w, up_b)
//   5. if pad_left1: x_post = pad_reflect_left1(x_post)
//   6. x_post += x_source                                    [inplace]
//   7. for k in 0..3: branch_k = resblock_k(x_post, main_iters[k])
//   8. out = (b0 + b1 + b2) / 3
//
// All intermediate tensors stay in GPU-private memory; only x_in/har/style
// projections cross the upload boundary and only `out_data` is downloaded.
//
// Shape consistency: IC of noise_conv output MUST equal up_w->ne[1] (the
// convt output channels), since step 6 is an elementwise add.
//
// style_projs_packed layout:
//   [0          .. 12*IC_noise          ) — noise resblock projections
//   [12*IC_noise .. 12*IC_noise + 12*IC_main) — main resblock 0
//   [next 12*IC_main slots) — main resblock 1
//   [next 12*IC_main slots) — main resblock 2
bool metal_vocoder_run_stage(
    MetalVocoderState * state,

    // Inputs (CPU)
    const float * x_in_data,    int64_t T_x_in,
    const float * har_data,     int64_t har_len,    int64_t har_C,
    const float * style_projs_packed,

    int64_t IC_noise, int64_t IC_main,

    // Output (CPU)
    float * x_out_data, int64_t T_post_pad,

    // Convt + pad
    int up_stride, int up_padding,
    bool pad_reflect_left1,

    // Noise conv (har → x_source)
    int noise_kernel, int noise_stride, int noise_padding,
    const ggml_tensor * noise_conv_w,
    const ggml_tensor * noise_conv_b,
    const ggml_tensor * up_w,
    const ggml_tensor * up_b,

    // Resblock iter weights (caller-built; kernel/dilation/padding already
    // encoded per-iter inside each MetalResblockIterWeights).
    const MetalResblockIterWeights noise_iters[3],
    const MetalResblockIterWeights main_iters[3][3]);

// Fused post_conv + iSTFT path. Runs everything in a single Metal
// command buffer on the vocoder's queue:
//   1. leaky_relu(x_in, 0.01)                            x_in : [T_in, IC_in]
//   2. post = conv1d(post_w, x_in) + post_b              post : [T_in, 22]
//   3. iSTFT 3-pass (idft → overlap-add → normalize)     audio: [out_len]
//
// Cross-stage cache (set by metal_vocoder_run_stage) is honoured: if
// x_in_data matches the previous stage's downloaded output pointer, the
// CPU upload is skipped and the data is blitted in from GPU memory.
//
// Returns false if iSTFT pipelines are unavailable (caller must fall back
// to the CPU/separate-Metal-iSTFT path).
// Returns true if the fused post_conv+iSTFT kernels are loaded. Used by
// audio_utils.cpp to decide whether to emit the graph's CPU post_conv ops.
bool metal_vocoder_post_istft_available(const MetalVocoderState * state);

bool metal_vocoder_run_post_istft(
    MetalVocoderState * state,

    // x_in : post-stage-1 generator output (CPU pointer).
    const float * x_in_data, int64_t T_in, int64_t IC_in,

    // Post-conv weights (K=7, stride=1, padding=3, IC_in -> 22).
    const ggml_tensor * post_w,
    const ggml_tensor * post_b,

    // iSTFT shape.
    int n_frames, int out_len,

    // Output (CPU buffer of out_len floats).
    float * audio_out);

} // namespace kokopop

#endif // KOKOPOP_HAS_METAL
