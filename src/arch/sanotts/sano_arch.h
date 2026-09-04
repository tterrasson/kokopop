#pragma once

// sanoTTS architecture.
//
// One GGUF holds N independent voices: each carries its own duration model,
// acoustic model, decoder, symbol table and sample rate, and two voices in the
// same file need share nothing but the code-point decomposition table. So
// `SanoArch` is mostly a vector of `SanoVoice`, plus the pieces that really are
// model-wide: the NFD table, the iSTFT plans and the graph arena.
//
// Inference is four sequential ggml graphs (duration, acoustic/token,
// acoustic/frame, decoder) with host work between them. The split is forced:
// the frame count depends on the *values* the duration model produces, and the
// token-to-frame expansion is a gather ggml has no operator for.

#include "arch/sanotts/sano_tokenizer.h"
#include "arch/sanotts/sano_weights.h"
#include "audio/istft.h"
#include "model/arch.h"
#include "model/model.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace kokopop {

/// One voice: its public description, its symbol table and its weights.
struct SanoVoice {
    VoiceDesc desc;

    /// Index in `kokopop.voices`, which is also the tensor-name prefix.
    size_t index = 0;

    sano::TokenTable tokens;

    SanoDurationWeights dur;
    SanoAcousticWeights ac;

    /// Exactly one of these is populated, per `desc.decoder`.
    SanoPiperliteWeights piperlite;
    SanoVocosWeights vocos;
};

/// Intermediate results of one synthesis, kept addressable so the tests can
/// gate each stage against its golden fixture instead of only the final PCM.
struct SanoProbe {
    std::vector<int32_t> durations;   ///< [n_tokens]
    /// ggml `[hidden, n_tokens]`: one token's channels contiguous.
    std::vector<float>   token_ctx;
    /// ggml `[frames, out_channels]`: one channel's time series contiguous.
    std::vector<float>   latent;
    std::vector<float>   audio;
    int64_t n_tokens = 0;
    int64_t frames = 0;
};

struct SanoArch final : ModelArch {
    // ---- ModelArch ----

    Arch arch() const override { return Arch::SanoTTS; }
    const char * name() const override { return "sanotts"; }

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

    /// The full pipeline with every stage kept. `synthesize()` is this plus a
    /// move of `probe.audio`; the tests call this one.
    bool run(const std::vector<uint32_t> & ids, const VoiceDesc & voice,
             float speed, const SynthesisExtras & extras,
             SanoProbe & probe, std::string & error);

    // ---- neutral state, forwarded ----

    Model * base = nullptr;
    Backend * backend = nullptr;

    // ---- voices ----

    std::vector<SanoVoice> voice_weights;
    std::vector<VoiceDesc> voice_descs;
    size_t default_voice_index = 0;

    /// Voice for a description resolved through `find_voice()`. Null when the
    /// description did not come from this arch.
    const SanoVoice * voice_for(const VoiceDesc & desc) const;

    // ---- shared tables ----

    sano::NfdTable nfd;

    /// Storage the `nfd` views point into. `NfdTable` is deliberately a set of
    /// raw pointers so it can be built over GGUF metadata without copying; the
    /// gguf accessors return owned vectors, so the owner is here.
    std::vector<uint32_t> nfd_codepoints;
    std::vector<uint32_t> nfd_offsets;
    std::vector<uint32_t> nfd_values;
    std::vector<uint32_t> nfd_ccc_codepoints;
    std::vector<uint32_t> nfd_ccc_classes;

    /// `kokopop.sanotts.source`, folded into the default noise seed so that a
    /// caller who supplies none still gets a reproducible, model-specific
    /// stream.
    std::string provenance;

    // ---- execution state ----
    //
    // The four graphs are strictly sequential — each is computed and read back
    // before the next is built — so one arena serves them all, sized to the
    // largest. Two concurrent synthesis sessions on one model would share it,
    // which is why nothing here is touched outside `run()`.

    ScratchArena graph_scratch;

    /// One plan per distinct (n_fft, hop) in the file. Plans are immutable and
    /// shareable; the workspace below is not.
    std::unordered_map<uint64_t, std::unique_ptr<IstftPlan>> istft_plans;
    IstftWorkspace istft_workspace;

    const IstftPlan * istft_plan(uint32_t n_fft, uint32_t hop) const;

    ggml_tensor * tensor(const std::string & logical_name) const;
};

/// The `SanoArch` behind a loaded model, or null when the file is not sanoTTS.
SanoArch * sano_arch(Model & model);

// ---------------------------------------------------------------------------
// Graph budgets
//
// The graph contexts are `no_alloc`, so an arena holds ggml object and graph
// metadata and nothing else — its size follows the *shape* of the graph, not
// the length of the chunk. Deriving the bounds from the voice's own depths
// rather than from a constant is what keeps a file declaring 64 residual
// blocks from overflowing a fixed-size graph, which ggml answers with an
// abort rather than an error.
//
// The measured counts on the shipped voices are 67 / 65 / 105 / 262 / 153
// nodes; the formulas below sit roughly 50% above that.
// ---------------------------------------------------------------------------

struct SanoGraphBudget {
    size_t tensors = 0;
    size_t nodes = 0;
};

SanoGraphBudget sano_duration_budget(const SanoVoice & voice);
SanoGraphBudget sano_acoustic_token_budget(const SanoVoice & voice);
SanoGraphBudget sano_acoustic_frame_budget(const SanoVoice & voice);
SanoGraphBudget sano_piperlite_budget(const SanoVoice & voice);
SanoGraphBudget sano_vocos_budget(const SanoVoice & voice);

// ---------------------------------------------------------------------------
// Stages (defined in sano_frontend.cpp, sano_piperlite.cpp, sano_vocos.cpp)
// ---------------------------------------------------------------------------

/// Graph 1: ids -> integer durations, `length_scale` and the voice's ceiling
/// applied. The final projection runs on the CPU sub-backend on every backend:
/// `round(exp(x))` is a step function, and a backend that drifts by 1e-3 near
/// a tie produces a different frame count and a different audio length.
bool sano_run_duration(SanoArch & arch, const SanoVoice & voice,
                       const std::vector<uint32_t> & ids, float length_scale,
                       std::vector<int32_t> & durations, std::string & error);

/// Graph 2: ids + durations -> token context `[n_tokens, hidden]`.
bool sano_run_acoustic_token(SanoArch & arch, const SanoVoice & voice,
                             const std::vector<uint32_t> & ids,
                             const std::vector<int32_t> & durations,
                             std::vector<float> & token_ctx, std::string & error);

/// Graph 3: expanded token context + frame features -> `[frames, out_channels]`.
///
/// `frame_input` is the host-side expansion, already `[frames, hidden + 3]`.
bool sano_run_acoustic_frame(SanoArch & arch, const SanoVoice & voice,
                             const std::vector<float> & frame_input,
                             int64_t frames, std::vector<float> & latent,
                             std::string & error);

/// Host-side token-to-frame expansion plus the three frame features, laid out
/// as the `[frames, hidden + 3]` graph-3 input.
void sano_expand_to_frames(const std::vector<float> & token_ctx, int64_t n_tokens,
                           int64_t hidden, const std::vector<int32_t> & durations,
                           int64_t frames, std::vector<float> & frame_input);

/// Graph 4a: latent `[frames, 192]` -> PCM at the voice's rate.
bool sano_run_piperlite(SanoArch & arch, const SanoVoice & voice,
                        const std::vector<float> & latent, int64_t frames,
                        std::vector<float> & audio, std::string & error);

/// Graph 4b: mel `[frames, mels]` + deterministic noise -> PCM.
bool sano_run_vocos(SanoArch & arch, const SanoVoice & voice,
                    const std::vector<float> & mel, int64_t frames,
                    uint64_t noise_seed, std::vector<float> & audio,
                    std::string & error);

} // namespace kokopop
