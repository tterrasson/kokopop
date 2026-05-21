#pragma once

#include "backend/backend.h"
#include "inference/lstm_fused.h"
#include "kokopop.h"

#include <cstdint>
#include <array>
#include <memory>
#include <string>
#include <functional>
#include <string_view>
#include <unordered_map>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct gguf_context;

namespace kokopop {

struct VoiceInfo {
    std::string name;
    std::vector<float> data;
    int64_t rows = 0;
    int64_t cols = 0;
};

struct ScratchArena {
    std::vector<uint8_t> bytes;
    size_t high_water = 0;

    uint8_t * data(size_t required);
    size_t capacity() const;
};

struct AdaIn1dWeights {
    ggml_tensor * norm_w = nullptr;
    ggml_tensor * norm_b = nullptr;
    ggml_tensor * gamma_w = nullptr;
    ggml_tensor * gamma_b = nullptr;
    ggml_tensor * beta_w = nullptr;
    ggml_tensor * beta_b = nullptr;

    bool valid() const {
        return norm_w != nullptr && norm_b != nullptr &&
               gamma_w != nullptr && gamma_b != nullptr &&
               beta_w != nullptr && beta_b != nullptr;
    }
};

struct GeneratorResblockWeights {
    std::array<AdaIn1dWeights, 3> adain1;
    std::array<AdaIn1dWeights, 3> adain2;
    std::array<ggml_tensor *, 3> alpha1{};
    std::array<ggml_tensor *, 3> alpha2{};
    std::array<ggml_tensor *, 3> convs1_w{};
    std::array<ggml_tensor *, 3> convs1_b{};
    std::array<ggml_tensor *, 3> convs2_w{};
    std::array<ggml_tensor *, 3> convs2_b{};
    std::array<int, 3> paddings{};

    bool valid() const;
};

struct AdainResblk1dWeights {
    AdaIn1dWeights norm1;
    AdaIn1dWeights norm2;
    ggml_tensor * conv1_w = nullptr;
    ggml_tensor * conv1_b = nullptr;
    ggml_tensor * conv2_w = nullptr;
    ggml_tensor * conv2_b = nullptr;
    ggml_tensor * conv1x1_w = nullptr;

    bool valid() const {
        return norm1.valid() && norm2.valid() &&
               conv1_w != nullptr && conv1_b != nullptr &&
               conv2_w != nullptr && conv2_b != nullptr;
    }
};

struct Model {
    uint32_t version = 1;
    int32_t n_threads = 1;
    int32_t sample_rate = 24000;
    int32_t backend_type = KOKOPOP_BACKEND_CPU;  // actual backend used
    bool is_mock = false;
    std::vector<std::string> vocab;

    // Hash functor that accepts both std::string and std::string_view.
    // Required for heterogeneous (string_view) lookups on maps keyed by std::string.
    struct transparent_string_hash {
        size_t operator()(std::string_view sv) const noexcept {
            return std::hash<std::string_view>{}(sv);
        }
    };

    // Token → ID map keyed by string_view so that lookups in tokenize_phonemes
    // avoid per-character allocations.  The views are anchored in vocab[] which
    // is populated once during load and never mutated afterwards, so the
    // underlying data always outlives the map entries.
    using token_map = std::unordered_map<
        std::string_view, uint32_t,
        transparent_string_hash,
        std::equal_to<> /* heterogeneous: std::string keys still work */
    >;
    token_map token_to_id;
    std::unordered_map<std::string, VoiceInfo> voices;

    // Tensor lookup map with transparent hash + equality so that
    // tensor(string_view) and cached_tensor(string_view) avoid constructing
    // temporary std::string keys on the hot inference path.
    using tensor_map = std::unordered_map<
        std::string, ggml_tensor *,
        transparent_string_hash,
        std::equal_to<>
    >;
    tensor_map tensors;
    std::unordered_map<std::string, std::string> tensor_physical_names;
    ScratchArena frontend_scratch;
    ScratchArena generation_scratch;
    ScratchArena generator_scratch;

    // Temporary buffers used by the inference pipeline (CPU-side only).
    std::vector<int32_t> tmp_ids_i32;
    std::vector<int32_t> tmp_pos_i32;
    std::vector<float> tmp_mask_f32;
    std::vector<float> tmp_post_f32;
    std::vector<float> tmp_decoder_style_f32;
    std::vector<float> tmp_decoder_cpu_f32;
    std::vector<float> tmp_audio_f32;

    // STFT / ISTFT reusable buffers (opt. 7.1, 7.3)
    // Caches intermediate audio-domain and spectral-domain data so that
    // cpu_harmonic_stft() and cpu_istft() avoid per-chunk allocations.
    std::vector<float> tmp_stft_source_f32;  // harmonic source waveform
    std::vector<float> tmp_stft_har_f32;     // harmonic STFT magnitude + phase
    std::vector<float> tmp_istft_y_f32;      // overlap-add accumulator
    std::vector<float> tmp_istft_denom_f32;  // window energy denominator

    // Pre-dequantized LSTM recurrent weight matrices (w_hh Q5_K → F32).
    // Key = tensor logical name (e.g. "kokopop.text_encoder.lstm.weight_hh_l0").
    // Populated once by preload_tensor_cache(); stable thereafter (no reallocation
    // after insertion).
    std::unordered_map<std::string, std::vector<float>> lstm_w_hh_f32;

    // Transposed w_hh layout for SIMD-friendly access in the fused LSTM kernel.
    // w_rowwise[j*H + k] provides contiguous access over hidden dimension k
    // for each gate j, enabling NEON/AVX2 vectorisation of the dot product.
    std::unordered_map<std::string, std::vector<float>> lstm_w_hh_rowwise;

    // CPU-side cache of LSTM b_hh tensors. The fused LSTM CPU callback reads
    // b_hh through a raw pointer captured at graph build time, so the data
    // must live in host memory regardless of where the GGUF weight tensor
    // is allocated (CPU buffer on CPU/Metal backends, VRAM on CUDA).
    // Key = tensor logical name (e.g. "kokopop.text_encoder.lstm.bias_hh_l0").
    std::unordered_map<std::string, std::vector<float>> lstm_b_hh_f32;

    // Host-side dequantized copies of the duration projection weights. The
    // final dur_logits = duration_w * dur_hidden + duration_b matmul is run on
    // CPU so the rounding behaviour of duration_to_frames() is identical
    // across backends — Vulkan/MoltenVK fp16 drift was flipping token
    // durations near k+0.5 boundaries and desynchronising the prosody.
    // Lazily populated on first frontend probe call.
    std::vector<float> duration_proj_w_f32;
    std::vector<float> duration_proj_b_f32;
    int64_t            duration_proj_in  = 0;  // hidden dim (= duration_w->ne[0])
    int64_t            duration_proj_out = 0;  // n_buckets (= duration_w->ne[1])

    // Scratch storage for LstmCustomParams instances built during graph
    // construction.  Reserve 24 slots before building the generation graph
    // (12 LSTM directions × safety margin) so no reallocation occurs and
    // stored pointers remain valid until after graph execution.
    // See graph_ops.cpp::lstm_direction for usage.
    std::vector<struct LstmCustomParams> lstm_custom_params;

    // Scratch storage for experimental Metal vocoder custom-op parameters.
    // Reserved before building the generator graph so userdata pointers remain
    // stable until graph execution completes.
    std::vector<MetalVocoderConvTransposeParams>  metal_vocoder_convt_params;
    std::vector<MetalGeneratorResblockParams>     metal_vocoder_resblock_params;
    std::vector<MetalGeneratorStageParams>        metal_vocoder_stage_params;

    // Lazily cached inference constants derived from immutable model weights.
    std::unordered_map<std::string, std::vector<float>> depthwise_pool_kernels;
    std::vector<float> harmonic_merge_w;
    float harmonic_merge_b = 0.0f;
    bool harmonic_source_loaded = false;

    // Cache for frequently accessed tensors to avoid hash lookups.
    // Uses the same transparent hash/equality as tensor_map so that
    // string_view lookups hit the cache without temporary allocations.
    using tensor_cache_map = std::unordered_map<
        std::string, ggml_tensor *,
        transparent_string_hash,
        std::equal_to<>
    >;
    tensor_cache_map tensor_cache;
    std::unordered_map<std::string, AdaIn1dWeights> adain_1d_weights;
    std::unordered_map<std::string, GeneratorResblockWeights> generator_resblock_weights;
    std::unordered_map<std::string, AdainResblk1dWeights> adain_resblk1d_weights;

    // Pre-populate the tensor cache with frequently used tensors.
    void preload_tensor_cache();

    // Pre-reserve scratch buffers used by the inference pipeline so that the
    // first chunk does not pay an allocate-from-zero cost.  Sizes are tuned
    // for typical chunks (~target_max_tokens=180); larger chunks still grow
    // these vectors lazily via resize().
    void prereserve_scratch_buffers();

    // Get tensor with caching (faster than tensor() for repeated accesses).
    ggml_tensor * cached_tensor(const std::string & logical_name) const;

    ggml_context * weight_ctx = nullptr;
    gguf_context * gguf_ctx = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;

    // Polymorphic backend (CPU or Metal).
    // Manages graph execution, tensor I/O, and deferred initialization.
    std::unique_ptr<Backend> backend;

    ~Model();
    bool tokenize_phonemes(const std::string & phonemes, std::vector<uint32_t> & ids, std::string & error) const;
    ggml_tensor * tensor(const std::string & logical_name) const;

    // Clear the tensor cache (useful when model is reloaded).
    void clear_tensor_cache() {
        tensor_cache.clear();
        adain_1d_weights.clear();
        generator_resblock_weights.clear();
        adain_resblk1d_weights.clear();
    }
};

bool load_model_from_gguf(
    const std::string & path,
    const kokopop_model_options * options,
    std::unique_ptr<Model> & model,
    std::string & error);

// Read tensor data into a float vector, handling F32, F16, and quantized types.
bool tensor_to_f32(Backend & backend, ggml_tensor * tensor, std::vector<float> & out);

// Resolve the effective voice name: use requested_voice, or fall back to the
// first available voice in the model if the request is empty.
inline std::string resolve_voice_name(
    const std::string & requested_voice,
    const std::unordered_map<std::string, VoiceInfo> & voices) {
    return requested_voice.empty() && !voices.empty()
        ? voices.begin()->first
        : requested_voice;
}

} // namespace kokopop
