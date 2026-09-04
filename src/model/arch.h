#pragma once

// Architecture abstraction.
//
// A GGUF file names the architecture it was converted for (`kokopop.arch`).
// `Model` owns the neutral part of a loaded file — the gguf context, the
// backend, the tensor map — and delegates everything model-shaped to a
// `ModelArch` implementation: weight resolution, the voice table, the text
// frontend, chunk budgeting and inference.
//
// Kokoro is one implementation (src/arch/kokoro), sanoTTS another
// (src/arch/sanotts).

#include "synthesis/chunker/chunker.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

struct gguf_context;

namespace kokopop {

struct Model;

enum class Arch {
    Unknown = 0,
    Kokoro  = 1,
    SanoTTS = 2,
};

/// Which grapheme-to-phoneme contract a voice follows. The two differ in more
/// than the espeak flags: misaki normalises espeak's output and frames the ids
/// as [BOS] ids [EOS], Piper decomposes to NFD and interleaves a PAD id.
enum class FrontendKind {
    Misaki = 0,
    Piper  = 1,
};

enum class DecoderKind {
    Kokoro    = 0,
    PiperLite = 1,
    Vocos     = 2,
};

/// Everything the neutral layers (session, chunker, streaming, HTTP) need to
/// know about a voice. Architecture-specific weights stay in the arch.
struct VoiceDesc {
    std::string name;
    std::vector<std::string> aliases;

    /// espeak-ng voice identifier, e.g. "gmw/en-US".
    std::string espeak_voice;

    /// Language selector for misaki-style phoneme normalisation ('a', 'b',
    /// 'z', …). Unused by the Piper frontend.
    char normalization_lang = 'a';

    int32_t sample_rate  = 24000;
    float   length_scale = 1.0f;

    /// Chunker budget for this voice, counted in final token ids (framing and
    /// interleaved PAD included).
    int32_t max_tokens = 510;

    FrontendKind frontend = FrontendKind::Misaki;
    DecoderKind  decoder  = DecoderKind::Kokoro;
};

/// Kokoro's style-diffusion sampler options.
///
/// Lives here rather than in src/arch/kokoro because it is part of
/// `SynthesisExtras`, which the neutral synthesis path passes through.
struct KokoroDiffusionOptions {
    bool enabled = false;
    uint32_t seed = 0;
    int steps = 5;
    float alpha = 0.1f;
    float beta = 0.5f;
    float embedding_scale = 1.0f;
};

/// Optional, architecture-specific synthesis inputs.
///
/// Deliberately a flat struct and not a `void *`: there are two architectures,
/// each reads the fields it cares about and ignores the rest, and a caller that
/// forgets to set one gets a documented default instead of undefined memory.
struct SynthesisExtras {
    /// Kokoro only.
    KokoroDiffusionOptions diffusion;

    /// Kokoro only — number of phoneme code points in this chunk, which picks
    /// the row of the voice's style matrix.
    ///
    /// It is not `ids.size()`: the tokenizer drops symbols outside the vocab
    /// and adds two sentinels, so the two counts diverge on real text. A
    /// negative value means "derive it from the id count".
    int64_t kokoro_style_len = -1;

    /// sanoTTS only — deterministic noise feeding the vocos decoder.
    /// `has_noise_seed` is a separate flag so that the value 0 remains a valid
    /// explicit seed rather than meaning "unset".
    bool     has_noise_seed = false;
    uint64_t noise_seed     = 0;

    /// Stable index of this chunk within the utterance. Folded into the
    /// per-chunk seed so that consecutive chunks do not reuse the same noise.
    uint32_t chunk_index = 0;
};

/// One architecture's implementation of the model contract.
struct ModelArch {
    virtual ~ModelArch() = default;

    // ---- identity ----

    virtual Arch arch() const = 0;

    /// Stable identifier, matching what the converter writes into
    /// `kokopop.arch`: "kokoro-82m", "sanotts".
    virtual const char * name() const = 0;

    // ---- loading ----

    /// Called once, after `Model` has populated tensors/backend/gguf_ctx.
    /// Resolves weights, fills caches, pre-reserves scratch and validates the
    /// file. Must leave nothing half-initialised on failure.
    virtual bool load(Model & base, std::string & error) = 0;

    // ---- voices ----

    /// Voices in file order. The first entry is not necessarily the default.
    virtual const std::vector<VoiceDesc> & voices() const = 0;

    /// Looks up by canonical name first, then by alias. Null when unknown.
    virtual const VoiceDesc * find_voice(std::string_view name) const = 0;

    /// Never null once `load()` has succeeded on a model with a voice.
    virtual const VoiceDesc * default_voice() const = 0;

    // ---- text frontend ----

    /// Text → phonemes. The architecture owns this because misaki and Piper
    /// disagree on espeak flags, punctuation and normalisation.
    virtual bool phonemize(const std::string & text, const VoiceDesc & voice,
                           std::string & phonemes, std::string & error) const = 0;

    /// Phonemes → token ids, framing included.
    virtual bool tokenize(const std::string & phonemes, const VoiceDesc & voice,
                          std::vector<uint32_t> & ids, std::string & error) const = 0;

    // ---- chunking ----

    /// Clamp a preset to what this voice's duration model was trained for.
    virtual ChunkConfig adjust_chunk_config(ChunkConfig cfg,
                                            const VoiceDesc & voice) const = 0;

    // ---- inference ----

    /// One full chunk: token ids → mono float32 PCM at `voice.sample_rate`.
    virtual bool synthesize(Model & base,
                            const std::vector<uint32_t> & ids,
                            const VoiceDesc & voice,
                            float speed,
                            const SynthesisExtras & extras,
                            std::vector<float> & audio,
                            std::string & error) = 0;
};

/// Resolve the architecture named by a GGUF's metadata and instantiate it.
///
/// Resolution order:
///   1. `kokopop.arch` (string).
///   2. No `kokopop.arch` but a `kokopop.kokoro.version` key → Kokoro. This is
///      what keeps the already-distributed Kokoro GGUFs loading without a
///      reconversion.
///   3. Otherwise an error naming the architectures this build knows.
///
/// The returned arch is not loaded yet; call `ModelArch::load()`.
std::unique_ptr<ModelArch> create_arch(gguf_context * meta, std::string & error);

/// Architecture named by a GGUF's metadata, without instantiating anything.
///
/// `create_backend` runs before the tensors are read, so backend selection
/// cannot ask the arch object what it is; this reads the key directly from the
/// already-open gguf context. Returns Arch::Unknown when it cannot tell.
Arch peek_arch(gguf_context * meta);

/// Human-readable name for an `Arch`, for diagnostics and the C API.
const char * arch_name(Arch arch);

} // namespace kokopop
