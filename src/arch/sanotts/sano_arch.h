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
// token-to-frame expansion is a gather ggml has no operator for. A styled
// request runs one more — the emotion conditioner — before the duration model
// and again before the decoder, each time feeding a head that stays on the
// host.

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

    /// Empty (`dim == 0`) unless the pack declared an emotion capability.
    SanoEmotionWeights emo;

    /// Exactly one of these is populated, per `desc.decoder`.
    SanoPiperliteWeights piperlite;
    SanoVocosWeights vocos;
};

/// The style of one synthesis, resolved from a name once per chunk.
///
/// Resolution is where the style vector is split into the two things the heads
/// actually consume: a *direction*, which is all the nonlinear style encoder
/// ever sees, and a scalar *intensity*, which multiplies everything the encoder
/// produces. That separation is what makes the displacement radial — half a
/// vector is exactly half the effect — and it is why calibration can scale an
/// axis without rotating it into another style.
struct SanoStyle {
    /// `[dim]`, or empty for a voice without an emotion pack. Always filled
    /// when the voice has one, so the neutral style is the zero vector rather
    /// than the absence of one.
    std::vector<float> vector;

    /// `[style_code]`: the style encoder run on the L1-normalized direction.
    /// Empty for a neutral style, which needs neither head.
    std::vector<float> code;

    /// The effective intensity every term of both heads is multiplied by:
    /// the vector's own L1 norm, weighted by the pack's calibrated axis gains.
    float amount = 0.0f;

    /// True when every component is zero — the exact neutral bypass the pack
    /// declares, which skips both heads entirely rather than computing terms
    /// that are provably zero. Also true for a plain voice.
    bool neutral = true;

    /// The canonical style name this was resolved from, for diagnostics.
    std::string name;
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

    bool resolve_style_tag(const VoiceDesc & voice, std::string_view tag,
                           std::string & style) const override;

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

    /// Resolves a style name — or an alias, or the empty string for the pack's
    /// own default — into its vectors. Fails on an unknown name, and on any
    /// name at all for a voice that carries no emotion pack: a style the model
    /// cannot render is a silently wrong rendering, not a detail to drop.
    bool resolve_style(const SanoVoice & voice, const std::string & name,
                       SanoStyle & style, std::string & error) const;

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
    // The graphs are strictly sequential — each is computed and read back
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
SanoGraphBudget sano_emotion_conditioner_budget(const SanoVoice & voice);
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
                       const SanoStyle & style,
                       std::vector<int32_t> & durations, std::string & error);

/// Graph 1b: the shared conditioner, `[cond_hidden, n_tokens]`.
///
/// The one graph the emotion path runs, and both heads read it: a dilated
/// stack over the phoneme sequence with the phrase's mean, first and last
/// token added back to every position. `ids` are the *branch's* ids, clamped
/// to the vocabulary of whichever native model this conditioning accompanies.
///
/// Everything downstream of it — the style code, the FiLM pair, the bounds,
/// the phone means and the boundary crossfade — stays on the host: a handful
/// of matrix-vector products per token and one pass over the frames, which
/// costs less than the dispatches would.
bool sano_run_emotion_conditioner(SanoArch & arch, const SanoVoice & voice,
                                  const std::vector<uint32_t> & ids,
                                  uint32_t vocab, const char * label,
                                  std::vector<float> & tokens, std::string & error);

/// The style's log-duration residual, `[n_tokens]`, added to the neutral log
/// durations before they are exponentiated and rounded.
///
/// `neutral` is the frozen model's own log predictions: they are not only what
/// the residual corrects, they are the attention the per-token redistribution
/// is centered under, so that a style moves where the time goes without also
/// moving how much of it there is.
bool sano_emotion_duration_residual(const SanoVoice & voice,
                                    const std::vector<uint32_t> & ids,
                                    const SanoStyle & style,
                                    const std::vector<float> & conditioned,
                                    const std::vector<float> & neutral,
                                    std::vector<float> & residual,
                                    std::string & error);

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

/// The style's bounded per-channel correction, applied to `latent` in place.
///
/// `latent` is graph 3's output, `[frames, channels]` with frames fastest.
///
/// The affine pair is predicted once per phone — two takes of one sentence
/// share no frame grid, so the phone is the finest timeline the paired
/// supervision could ever have used — but it is *evaluated* on each frame's
/// own content, `phone_mean + frame_gain * (frame - phone_mean)`, so the gain
/// reaches the variation inside a phone instead of a single number standing
/// for all of it. The raw residual is then crossfaded across phone boundaries
/// with a raised cosine, bounded, and added. The native latent itself is never
/// smoothed, and a neutral style leaves it untouched.
bool sano_apply_emotion_latent(const SanoVoice & voice,
                               const std::vector<int32_t> & durations,
                               const SanoStyle & style,
                               const std::vector<float> & conditioned,
                               int64_t frames, std::vector<float> & latent,
                               std::string & error);

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
