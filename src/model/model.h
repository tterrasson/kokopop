#pragma once

#include "backend/backend.h"
#include "model/arch.h"
#include "kokopop.h"

#include <cstdint>
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

/// A grow-only byte arena backing one ggml graph context.
///
/// `high_water` is the largest size ever requested; the memory-sizing tests
/// compare it against the architecture's reservation formulas.
struct ScratchArena {
    std::vector<uint8_t> bytes;
    size_t high_water = 0;

    uint8_t * data(size_t required);
    size_t capacity() const;
};

/// A loaded GGUF file: the parts that do not depend on the architecture.
///
/// Everything model-shaped — weights, voices, tokenizer, graphs — belongs to
/// `arch`. What is left here is the file, the backend and the tensor map.
struct Model {
    uint32_t version = 1;
    int32_t n_threads = 1;
    int32_t backend_type = KOKOPOP_BACKEND_CPU;  // actual backend used
    bool is_mock = false;

    // Hash functor that accepts both std::string and std::string_view.
    // Required for heterogeneous (string_view) lookups on maps keyed by std::string.
    struct transparent_string_hash {
        size_t operator()(std::string_view sv) const noexcept {
            return std::hash<std::string_view>{}(sv);
        }
    };

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

    // Cache for frequently accessed tensors to avoid hash lookups. Populated
    // by the architecture at load time; the map itself is a neutral mechanism.
    using tensor_cache_map = std::unordered_map<
        std::string, ggml_tensor *,
        transparent_string_hash,
        std::equal_to<>
    >;
    tensor_cache_map tensor_cache;

    /// Voice metadata, name -> descriptor. Lookup index over `arch->voices()`,
    /// which stays authoritative for file order; built by
    /// `load_model_from_gguf` once the architecture has loaded.
    std::unordered_map<std::string, VoiceDesc> voices;

    /// Sample rate declared by `kokopop.sample_rate`. Only a fallback:
    /// `sample_rate()` prefers the loaded voice table.
    int32_t default_sample_rate = 24000;

    ggml_context * weight_ctx = nullptr;
    gguf_context * gguf_ctx = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;

    // Polymorphic backend (CPU or one of the GPU backends).
    // Manages graph execution, tensor I/O, and deferred initialization.
    std::unique_ptr<Backend> backend;

    // Architecture implementation: weights, voices, frontend, inference.
    std::unique_ptr<ModelArch> arch;

    ~Model();

    /// Sample rate of the default voice, in Hz.
    ///
    /// A method rather than a field because a model may carry several voices
    /// with different rates (sanoTTS packs mix 22050 and 24000 Hz); the
    /// per-voice rate is the one that matters for a session. Callers that only
    /// have a Model, and no voice, get the default voice's rate here.
    int32_t sample_rate() const;

    /// Sample rate of `voice`, falling back to the default voice's rate when
    /// the name is unknown.
    int32_t sample_rate(std::string_view voice) const;

    /// Phonemes -> token ids for `voice`, using that voice's tokenizer.
    bool tokenize_phonemes(const std::string & phonemes, std::string_view voice,
                           std::vector<uint32_t> & ids, std::string & error) const;

    /// Same, for the default voice.
    bool tokenize_phonemes(const std::string & phonemes,
                           std::vector<uint32_t> & ids, std::string & error) const;

    ggml_tensor * tensor(const std::string & logical_name) const;
    ggml_tensor * cached_tensor(const std::string & logical_name) const;
};

/// The two closures the chunker needs, bound to one resolved voice.
///
/// Both go through `ModelArch`, which is the point: the chunker used to call a
/// global `phonemize_text()` with a voice *name*, which silently applied
/// Kokoro's rules to any voice.
struct VoiceFrontend {
    VoiceDesc   voice;
    PhonemizeFn phonemize;
    TokenizeFn  tokenize;
};

/// Resolve `requested_voice` (empty = default) and bind its frontend.
///
/// The returned closures capture `model` by reference, so they must not
/// outlive it.
bool make_voice_frontend(Model & model, const std::string & requested_voice,
                         VoiceFrontend & out, std::string & error);

bool load_model_from_gguf(
    const std::string & path,
    const kokopop_model_options * options,
    std::unique_ptr<Model> & model,
    std::string & error);

// Read tensor data into a float vector, handling F32, F16, and quantized types.
bool tensor_to_f32(Backend & backend, ggml_tensor * tensor, std::vector<float> & out);

/// Resolve the effective voice name: `requested_voice`, or the model's default
/// voice when the request is empty.
///
/// The fallback is the architecture's declared default (`kokopop.default_voice`
/// when present, otherwise the first entry of `kokopop.voices`) rather than an
/// arbitrary hash-map entry, so it does not vary between runs.
std::string resolve_voice_name(const std::string & requested_voice,
                               const Model & model);

} // namespace kokopop
