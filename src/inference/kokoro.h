#pragma once

#include "model/model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kokopop {

struct KokoroFrontendProbe {
    std::vector<float> hidden;
    std::vector<float> durations;
    std::vector<float> style;
    std::vector<float> bert_embedding;
    int64_t n_tokens = 0;
    int64_t hidden_dim = 0;
    int64_t embedding_dim = 0;
};

struct KokoroGenerationProbe {
    std::vector<float> f0;
    std::vector<float> noise;
    std::vector<float> asr;
    std::vector<float> decoder;
    std::vector<float> audio;
    int64_t n_frames = 0;
    int64_t asr_dim = 0;
    int64_t decoder_dim = 0;
    int64_t decoder_len = 0;
};

// ---------------------------------------------------------------------------
// Internal helpers (defined in kokoro.cpp)
// ---------------------------------------------------------------------------

ggml_tensor * require_tensor(Model & model, const char * name, std::string & error);

// ---------------------------------------------------------------------------
// Graph operations (defined in graph_ops.cpp)
// ---------------------------------------------------------------------------

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias, float eps);
ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias, ggml_tensor * x);
ggml_tensor * add_channel_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias);
// ggml_leaky_relu, unless the active backend has no kernel for it, in which
// case the mathematically identical relu(x) - slope*relu(-x) is emitted so the
// tensor stays on the device (see Backend::has_leaky_relu).
ggml_tensor * graph_leaky_relu(ggml_context * ctx, const Model & model, ggml_tensor * x, float slope);
// direct=true asks for a single CONV_2D node instead of im2col + mul_mat; see
// Backend::prefers_direct_conv. Silently falls back when the shapes do not fit.
ggml_tensor * conv1d(
    ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input,
    int stride, int padding, int dilation, int kernel_size, bool direct = false);
ggml_tensor * conv_transpose1d_crop(
    ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * input,
    int stride,
    int crop_left,
    int out_len);
ggml_tensor * conv_transpose1d_crop_bias(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * weight,
    ggml_tensor * input,
    ggml_tensor * bias,
    int stride,
    int crop_left,
    int out_len);
ggml_tensor * depthwise_pool_upsample(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    const std::string & prefix,
    std::string & error);
ggml_context * init_scratch_context(
    Model & model, ScratchArena & arena, size_t mem_size,
    bool no_alloc, const char * label, std::string & error);
ggml_tensor * adain_1d(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    const std::string & prefix,
    std::string & error);
ggml_tensor * adain_1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * style,
    const AdaIn1dWeights & weights);
ggml_tensor * adain_resblk1d(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    const std::string & prefix,
    bool upsample,
    std::string & error);
ggml_tensor * ada_layer_norm(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    const std::string & prefix,
    std::string & error);
ggml_tensor * graph_snake1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * alpha,
    const std::string & alpha_name,
    std::string & error);
ggml_tensor * graph_snake1d(
    ggml_context * ctx, Model & model, ggml_tensor * x,
    const std::string & alpha_name, std::string & error);
ggml_tensor * graph_generator_resblock(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * style,
    const std::string & prefix,
    int kernel_size,
    const GeneratorResblockWeights & weights,
    std::string & error,
    bool fused_snake = false,
    bool direct = false);
ggml_tensor * graph_generator_resblock(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    const std::string & prefix,
    int kernel_size,
    std::string & error);

// 3-branch main resblock + sum/3. `stage` selects resblocks.<stage*3+k>
// for k in 0..3. Returns a tensor with the same shape as x, equal to
// (resblock_0(x) + resblock_1(x) + resblock_2(x)) / 3.
//
// On Metal this is only reached as a fallback when graph_generator_stage_fused
// is unavailable (missing cached resblock). The normal Metal path fuses the
// entire stage including this sum.
ggml_tensor * graph_3branch_main_sum(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    int stage,
    std::string & error);

// Fused per-stage generator op (replaces the entire body of one stage in
// audio_utils.cpp::ggml_generator). Performs leaky_relu | noise_conv |
// noise_resblock | conv_transpose1d | [pad_reflect_left1] | add |
// 3 main resblocks | sum/3 in a single Metal command buffer.
//
// Returns the post-stage x tensor (shape [T_post_pad, IC_x_out=64]).
// Returns nullptr when the fused path is not applicable (non-Metal backend,
// vocoder kernel unavailable, or any of the 4 stage resblocks not cached);
// caller falls back to the explicit per-op graph in that case.
//
// `stage` is 0 or 1. `har_t` is the harmonic STFT tensor (graph input).
ggml_tensor * graph_generator_stage_fused(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    ggml_tensor * har_t,
    int stage,
    std::string & error);
ggml_tensor * duration_encoder(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    int64_t n_steps,
    std::string & error);
ggml_tensor * text_encoder(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * token_ids,
    ggml_tensor * duration_mask,
    int64_t n_tokens,
    std::string & error);
ggml_tensor * bidirectional_lstm(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * input,
    const std::string & prefix,
    int64_t n_steps,
    std::string & error);

// ---------------------------------------------------------------------------
// Audio utilities (defined in audio_utils.cpp)
// ---------------------------------------------------------------------------

size_t generator_graph_size(int64_t decoder_len);

size_t frontend_graph_size(int64_t n_tokens);
size_t generation_graph_size(int64_t total_frames, int64_t n_tokens);

struct CpuTensor {
    int64_t channels = 0;
    int64_t length = 0;
    std::vector<float> data;

    float & at(int64_t c, int64_t t) {
        return data[static_cast<size_t>(c * length + t)];
    }

    float at(int64_t c, int64_t t) const {
        return data[static_cast<size_t>(c * length + t)];
    }
};

struct KokoroDiffusionOptions {
    bool enabled = false;
    uint32_t seed = 0;
    int steps = 5;
    float alpha = 0.1f;
    float beta = 0.5f;
    float embedding_scale = 1.0f;
};

bool ggml_generator(
    Model & model, const CpuTensor & decoder,
    const std::vector<float> & f0,
    const std::vector<float> & style,
    std::vector<float> & audio,
    std::string & error);

// ---------------------------------------------------------------------------
// Main pipeline (defined in kokoro.cpp)
// ---------------------------------------------------------------------------

bool run_kokoro_frontend_probe(
    Model & model,
    const std::vector<uint32_t> & ids,
    const std::string & voice,
    KokoroFrontendProbe & probe,
    std::string & error,
    int64_t style_len = -1,
    const KokoroDiffusionOptions * diffusion = nullptr);

bool run_kokoro_generation_probe(
    Model & model,
    const std::vector<uint32_t> & ids,
    const std::string & voice,
    float speed,
    const KokoroFrontendProbe & frontend,
    KokoroGenerationProbe & probe,
    std::string & error,
    int64_t style_len = -1);

} // namespace kokopop
