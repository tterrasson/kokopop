#pragma once

#include "model/model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kokopop {

struct KokoroFrontendProbe {
    std::vector<float> hidden;
    std::vector<float> durations;
    int64_t n_tokens = 0;
    int64_t hidden_dim = 0;
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
ggml_tensor * conv1d(
    ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input,
    int stride, int padding, int dilation, int kernel_size);
ggml_tensor * conv_transpose1d_crop(
    ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * input,
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
    std::string & error);
ggml_tensor * graph_generator_resblock(
    ggml_context * ctx,
    Model & model,
    ggml_tensor * x,
    ggml_tensor * style,
    const std::string & prefix,
    int kernel_size,
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
    std::string & error);

bool run_kokoro_generation_probe(
    Model & model,
    const std::vector<uint32_t> & ids,
    const std::string & voice,
    float speed,
    const KokoroFrontendProbe & frontend,
    KokoroGenerationProbe & probe,
    std::string & error);

} // namespace kokopop
