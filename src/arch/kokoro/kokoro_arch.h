#pragma once

// Kokoro-82M architecture.
//
// `KokoroArch` owns everything that used to sit directly on `Model`: the
// AdaIN/resblock weight views, the LSTM and duration-projection host caches,
// the harmonic STFT buffers, the diffusion cache, the graph scratch arenas and
// the character-level tokenizer.
//
// The graph builders in this directory take a `KokoroArch &` (still named
// `model` at their call sites) and reach the neutral part of the file — the
// backend and the tensor map — through the forwarding accessors below.

#include "model/arch.h"
#include "model/model.h"
#include "arch/kokoro/lstm_fused.h"
#include "audio/istft.h"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace kokopop {

/// A Kokoro voice: a [rows, cols] style matrix, read to F32 at load time.
struct VoiceInfo {
    std::string name;
    std::vector<float> data;
    int64_t rows = 0;
    int64_t cols = 0;
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

struct KokoroArch final : ModelArch {
    // ---- ModelArch ----

    Arch arch() const override { return Arch::Kokoro; }
    const char * name() const override { return "kokoro-82m"; }

    bool load(Model & base, std::string & error) override;

    const std::vector<VoiceDesc> & voices() const override { return voice_descs; }
    const VoiceDesc * find_voice(std::string_view name) const override;
    const VoiceDesc * default_voice() const override;

    bool phonemize(const std::string & text, const VoiceDesc & voice,
                   std::string & phonemes, std::string & error) const override;
    bool tokenize(const std::string & phonemes, const VoiceDesc & voice,
                  std::vector<uint32_t> & ids, std::string & error) const override;

    ChunkConfig adjust_chunk_config(ChunkConfig cfg,
                                    const VoiceDesc & voice) const override;

    bool synthesize(Model & base,
                    const std::vector<uint32_t> & ids,
                    const VoiceDesc & voice,
                    float speed,
                    const SynthesisExtras & extras,
                    std::vector<float> & audio,
                    std::string & error) override;

    // ---- neutral state, forwarded ----
    //
    // `base` is null only for the standalone unit tests that exercise the CPU
    // audio helpers, which touch the scratch buffers below and nothing else.

    Model * base = nullptr;

    /// Cached `base->backend.get()`. The graph builders dereference this on
    /// nearly every line, hence the direct pointer rather than a call.
    Backend * backend = nullptr;

    int32_t backend_type = KOKOPOP_BACKEND_CPU;

    ggml_tensor * tensor(const std::string & logical_name) const;
    ggml_tensor * cached_tensor(const std::string & logical_name) const;

    // ---- voices ----

    std::vector<VoiceDesc> voice_descs;

    /// Index of the default voice inside `voice_descs`.
    ///
    /// Stored separately rather than by moving that voice to the front:
    /// `voices()` promises file order, which multi-voice enumeration relies on.
    size_t default_voice_index = 0;

    /// Style matrices, keyed by voice name.
    std::unordered_map<std::string, VoiceInfo> voice_styles;

    /// `requested` when non-empty, otherwise the default voice's canonical
    /// name. Returns an empty string only for a model with no voice at all.
    std::string resolve_voice(const std::string & requested) const;

    // ---- tokenizer ----

    std::vector<std::string> vocab;

    // Token → ID map keyed by string_view so that tokenize() avoids
    // per-character allocations. The views are anchored in vocab[], which is
    // populated once during load and never mutated afterwards, so the
    // underlying data always outlives the map entries.
    using token_map = std::unordered_map<
        std::string_view, uint32_t,
        Model::transparent_string_hash,
        std::equal_to<>
    >;
    token_map token_to_id;

    // ---- graph scratch ----

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

    // STFT / ISTFT reusable buffers
    // Caches intermediate audio-domain and spectral-domain data so that
    // cpu_harmonic_stft() and cpu_istft() avoid per-chunk allocations.
    std::vector<float> tmp_stft_source_f32;  // harmonic source waveform
    std::vector<float> tmp_stft_har_f32;     // harmonic STFT magnitude + phase

    // Complex half-spectrum handed to the shared iSTFT, [bins][frames]. The
    // log-magnitude/phase to complex conversion is Kokoro's, so it happens
    // here rather than inside the component.
    std::vector<float> tmp_istft_real_f32;
    std::vector<float> tmp_istft_imag_f32;

    // Overlap-add scratch for the shared iSTFT. O(n_fft), owned per arch
    // instance so that two synthesis contexts never share it.
    IstftWorkspace istft_workspace;

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

    // Host-side dequantized diffusion transformer weights. The sampler is CPU
    // only for now and reads these immutable tensors repeatedly across ADPM2
    // steps, so cache the F32 copies after first use.
    std::unordered_map<std::string, std::vector<float>> diffusion_f32;
    float diffusion_sigma_data = 0.2f;

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

    std::unordered_map<std::string, AdaIn1dWeights> adain_1d_weights;
    std::unordered_map<std::string, GeneratorResblockWeights> generator_resblock_weights;
    std::unordered_map<std::string, AdainResblk1dWeights> adain_resblk1d_weights;

    /// Populate `base->tensor_cache` and the AdaIN/resblock weight views.
    void preload_tensor_cache();

    /// Pre-reserve the scratch buffers used by the inference pipeline so that
    /// the first chunk does not pay an allocate-from-zero cost. Sizes are tuned
    /// for typical chunks (~target_max_tokens=180); larger chunks still grow
    /// these vectors lazily via resize().
    void prereserve_scratch_buffers();

    void clear_tensor_cache();
};

// ---------------------------------------------------------------------------
// Kokoro voice-name conventions
//
// Kokoro encodes a voice's language in the first letter of its name
// ("af_heart" -> American English). No other architecture does, which is why
// these two live here and not in src/synthesis/phonemizer.h.
// ---------------------------------------------------------------------------

/// espeak-ng voice for a Kokoro voice name. Unknown prefixes fall back to
/// American English, matching the behaviour the models were converted with.
std::string espeak_voice_for_kokoro_voice(const std::string & voice);

/// Phonemize with a Kokoro voice *name*, deriving the espeak voice and the
/// normalisation language from it.
bool phonemize_text(const std::string & text, const std::string & kokoro_voice,
                    std::string & phonemes, std::string & error);

/// The `KokoroArch` behind a loaded model, or null when the file is not Kokoro.
KokoroArch * kokoro_arch(Model & model);

// ---------------------------------------------------------------------------
// Graph context sizing.
//
// The formulas are Kokoro's, so they live with Kokoro; the backend only
// contributes its own metadata/margin term via Backend::graph_context_bytes.
// ---------------------------------------------------------------------------

size_t kokoro_frontend_context_bytes(const Backend & backend);
size_t kokoro_generation_context_bytes(const Backend & backend,
                                       int64_t total_frames, int64_t n_tokens);
size_t kokoro_generator_context_bytes(const Backend & backend,
                                      int64_t decoder_len);

} // namespace kokopop
