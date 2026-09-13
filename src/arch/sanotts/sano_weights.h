#pragma once

// sanoTTS weights, resolved once at load time.
//
// A voice is four components: a duration model, an acoustic model and one of
// two decoders. The first two are structurally identical across both families
// and differ only in their dimensions; the decoders share nothing.
//
// Every field here is either a `ggml_tensor *` owned by the GGUF weight
// context or a scalar read from the metadata. Nothing is resolved lazily: an
// incomplete or mis-shaped file must fail in `SanoArch::load()`, not halfway
// through a graph build.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;

namespace kokopop {

/// `x + scale * conv2(silu(conv1(x)))`, the block both frontend stages use.
///
/// `kernel` is read from the weight tensor rather than the metadata, then
/// cross-checked against it: the tensor is what the graph will actually be
/// built from. `dilation` is 1 for the native students and doubles per level
/// in the emotion conditioner, where reach — not capacity — is the point.
struct SanoResBlock {
    ggml_tensor * net0_w = nullptr;
    ggml_tensor * net0_b = nullptr;
    ggml_tensor * net2_w = nullptr;
    ggml_tensor * net2_b = nullptr;

    /// Read to host at load time: it multiplies a tensor as a plain float, so
    /// keeping it as a one-element tensor would cost a graph node per block.
    float scale = 0.0f;

    int32_t kernel = 0;
    int32_t dilation = 1;
};

struct SanoDurationWeights {
    uint32_t vocab = 0;
    uint32_t hidden = 0;
    uint32_t depth = 0;
    uint32_t kernel = 0;
    uint32_t max_tokens = 0;
    uint32_t max_duration = 0;

    ggml_tensor * embedding = nullptr;
    ggml_tensor * input_proj_w = nullptr;
    ggml_tensor * input_proj_b = nullptr;
    std::vector<SanoResBlock> blocks;
    ggml_tensor * output_w = nullptr;
    ggml_tensor * output_b = nullptr;
};

/// The utterance style space, the shared conditioner, and the two FiLM heads.
///
/// One capability, `sanofr.utterance-emotion.v1`, in its FiLM shape: the
/// emotion never enters either native student. Both stay exactly the neutral
/// voice; the style is a correction applied *after* them.
///
///   * Acoustically, a feature-wise affine transform — a per-channel log gain
///     and a shift — predicted once per phone but evaluated on each frame's
///     own content, `phone_mean + frame_gain * (frame - phone_mean)`, with a
///     short raised-cosine crossfade joining the raw residual across phone
///     boundaries before it is bounded. Only the residual is ever crossfaded;
///     the native latent is never smoothed.
///   * On timing, one global log tempo plus a per-token redistribution that is
///     centered to zero under the neutral model's own attention, so a style
///     may move where the time goes without moving how much there is of it
///     beyond the tempo it asked for.
///
/// Both are driven by one shared conditioner (`sanofr.shared-phrase-film.v2`),
/// a small dilated stack over the phoneme sequence with a phrase summary added
/// back to every token, and by one style code — the direction of the style
/// vector through a biasless, 0-preserving encoder — scaled by one scalar
/// intensity. Splitting direction from intensity is what makes `0.5 * v`
/// exactly half the displacement of `v`.
///
/// `dim == 0` means the voice carries no emotion pack. A style vector of all
/// zeros is the neutral one and bypasses both heads exactly: the intensity is
/// zero, and every term of both heads is multiplied by it.
///
/// Everything below except the conditioner's own stack is kept on the host.
/// These are hundreds to a few thousand floats and the arithmetic over them is
/// a handful of matrix-vector products per token; a graph node each would cost
/// more to dispatch than to compute.
struct SanoEmotionWeights {
    uint32_t dim = 0;

    // -- the shared conditioner --------------------------------------------
    uint32_t cond_vocab = 0;
    uint32_t cond_hidden = 0;
    uint32_t cond_depth = 0;
    uint32_t cond_kernel = 0;
    uint32_t style_code = 0;

    ggml_tensor * cond_embedding = nullptr;

    /// `[cond_hidden + 1, cond_hidden]`: the embedding with the token's
    /// phrase-relative position appended, pointwise.
    ggml_tensor * cond_input_proj_w = nullptr;
    ggml_tensor * cond_input_proj_b = nullptr;

    /// Dilation doubles per level — block `i` runs at `1 << i` — which is an
    /// architecture constant and not carried in the file.
    std::vector<SanoResBlock> cond_blocks;

    /// `[3 * cond_hidden, cond_hidden]`: mean, first and last token of the
    /// phrase, added back to every token before the layer norm.
    ggml_tensor * cond_phrase_w = nullptr;
    ggml_tensor * cond_phrase_b = nullptr;
    ggml_tensor * cond_norm_w = nullptr;
    ggml_tensor * cond_norm_b = nullptr;

    /// The style encoder, on the host: `[dim, cond_hidden]` then
    /// `[cond_hidden, style_code]`, no biases, SiLU between them.
    std::vector<float> style_w0;
    std::vector<float> style_w2;

    /// `[dim]`. Calibration scales an axis's intensity, never its direction.
    std::vector<float> axis_gain;

    // -- the acoustic head -------------------------------------------------
    /// The latent channel count the heads produce a gain and a shift for;
    /// equal to the acoustic model's `out_channels`.
    uint32_t channels = 0;
    uint32_t context_rank = 0;

    /// Ceilings the head closes on, all in the units of the quantity they
    /// bound: the rendered per-channel displacement, the log gain before it,
    /// and the contextual departure from the phrase-global pair.
    float max_delta = 0.0f;
    float max_log_gain = 0.0f;
    float context_bound = 0.0f;

    /// How much of a frame's departure from its phone mean the gain sees, in
    /// `[0, 1]`. Zero is the retired v1 behaviour — one constant residual per
    /// phone — and is why a v1 pack is refused rather than rendered with a
    /// default: the same weights mean a different voice under each.
    float frame_gain = 0.0f;

    /// Half-width, in frames, of the raised-cosine crossfade applied to the
    /// raw residual at each phone boundary. Further limited per boundary to
    /// half of the shorter of the two phones it joins, so a short phone is
    /// never smeared over its neighbours. Zero disables the crossfade.
    uint32_t transition_frames = 0;

    /// How much of the donor-versus-teacher direction is removed from the
    /// rendered delta, in `[0, 1]`. Zero for every shipped pack; the axis is
    /// carried regardless so the two always travel together.
    float donor_projection = 0.0f;

    /// `[style_code, 2 * channels]` and `[context_rank, 2 * channels]`, both
    /// producing the log gain and the shift stacked, gain first.
    std::vector<float> global_film;
    std::vector<float> context_basis;

    /// `[cond_hidden, context_rank]` + `[context_rank]`, and the style's own
    /// `[style_code, context_rank]`: the two factors whose product is the
    /// style x context interaction.
    std::vector<float> context_coeff_w;
    std::vector<float> context_coeff_b;
    std::vector<float> style_rank;

    /// `[channels]`, zero unless the run fitted one.
    std::vector<float> donor_axis;

    // -- the duration head -------------------------------------------------
    /// Bound on the log-duration ratio the head may apply, in either
    /// direction: it closes on `tanh(...) * max_log_ratio`.
    float max_log_ratio = 0.0f;

    /// Bound on the per-token redistribution alone, before it is centered and
    /// added to the tempo. Halved in use, as the reference does.
    float local_bound = 0.0f;

    /// `[style_code]`, `[style_code, cond_hidden]`, `[cond_hidden]`.
    std::vector<float> tempo_w;
    std::vector<float> style_local;
    std::vector<float> dur_output_w;

    /// `[dur.vocab]`, 1 for a phoneme id the emotional corpus supervised.
    /// An id outside it receives no redistribution and does not take part in
    /// the centering, so a rare phone is never moved by a term nothing
    /// measured.
    std::vector<float> observed_ids;

    /// Style names and their vectors, `dim` values each, style-major.
    std::vector<std::string> styles;
    std::vector<float>       vectors;

    /// Extra spellings a caller may use, parallel arrays into `styles`.
    std::vector<std::string> alias_names;
    std::vector<std::string> alias_styles;

    std::string default_style;

    bool enabled() const { return dim > 0; }
};

struct SanoAcousticWeights {
    uint32_t vocab = 0;
    uint32_t hidden = 0;
    uint32_t token_depth = 0;
    uint32_t depth = 0;
    uint32_t kernel = 0;
    uint32_t out_channels = 0;

    ggml_tensor * embedding = nullptr;
    ggml_tensor * token_proj_w = nullptr;
    ggml_tensor * token_proj_b = nullptr;
    std::vector<SanoResBlock> token_blocks;
    ggml_tensor * frame_proj_w = nullptr;
    ggml_tensor * frame_proj_b = nullptr;
    std::vector<SanoResBlock> frame_blocks;
    ggml_tensor * output_w = nullptr;
    ggml_tensor * output_b = nullptr;
};

// ---------------------------------------------------------------------------
// piperlite
// ---------------------------------------------------------------------------

/// One branch of a residual bank. The three branches of a bank differ in
/// kernel size and dilation, which are architecture constants, not metadata.
struct SanoPiperBranch {
    ggml_tensor * conv1_w = nullptr;
    ggml_tensor * conv1_b = nullptr;
    ggml_tensor * conv2_w = nullptr;
    ggml_tensor * conv2_b = nullptr;
    int32_t kernel = 0;
    int32_t dilation1 = 0;
    int32_t dilation2 = 0;
};

struct SanoPiperStage {
    ggml_tensor * up_w = nullptr;
    ggml_tensor * up_b = nullptr;
    int32_t up_kernel = 0;
    int32_t up_stride = 0;
    int32_t up_padding = 0;

    /// Indices into `branches_all`; a stage may run a subset.
    std::vector<uint32_t> branches;
    std::array<SanoPiperBranch, 3> bank{};
};

struct SanoPostFilterUnit {
    float scale = 0.0f;
    ggml_tensor * conv1_w = nullptr;
    ggml_tensor * conv1_b = nullptr;
    ggml_tensor * conv2_w = nullptr;
    ggml_tensor * conv2_b = nullptr;
    int32_t kernel = 0;
    int32_t dilation1 = 1;
};

struct SanoPiperliteWeights {
    std::array<uint32_t, 4> channels{};
    uint32_t pre_kernel = 0;
    uint32_t post_kernel = 0;

    ggml_tensor * pre_w = nullptr;
    ggml_tensor * pre_b = nullptr;
    std::array<SanoPiperStage, 3> stages{};
    ggml_tensor * post_w = nullptr;
    ggml_tensor * post_b = nullptr;

    /// 0 disables the post filter entirely; the whole block below is then
    /// absent from the file and must stay absent.
    uint32_t post_filter_channels = 0;
    uint32_t post_filter_layers = 0;
    uint32_t post_filter_kernel = 9;
    uint32_t post_filter_unit_kernel = 3;
    float    post_filter_scale = 0.0f;

    ggml_tensor * post_filter_in_w = nullptr;
    ggml_tensor * post_filter_in_b = nullptr;
    ggml_tensor * post_filter_out_w = nullptr;
    ggml_tensor * post_filter_out_b = nullptr;
    std::vector<SanoPostFilterUnit> post_filter_units;

    /// Total upsampling of the three stages, which must equal the frame hop.
    int32_t upsampling() const {
        int32_t total = 1;
        for (const auto & stage : stages) {
            total *= stage.up_stride;
        }
        return total;
    }
};

// ---------------------------------------------------------------------------
// vocos / TinyVocos
// ---------------------------------------------------------------------------

struct SanoVocosBlock {
    ggml_tensor * dw_w = nullptr;
    ggml_tensor * dw_b = nullptr;
    ggml_tensor * norm_w = nullptr;
    ggml_tensor * norm_b = nullptr;
    ggml_tensor * pw0_w = nullptr;
    ggml_tensor * pw0_b = nullptr;
    ggml_tensor * pw1_w = nullptr;
    ggml_tensor * pw1_b = nullptr;
    ggml_tensor * gamma = nullptr;
};

struct SanoVocosWeights {
    uint32_t dim = 0;
    uint32_t blocks = 0;
    uint32_t pw_hidden = 0;
    uint32_t dw_kernel = 0;
    uint32_t embed_kernel = 0;
    uint32_t noise_ch = 0;
    uint32_t mels = 0;
    uint32_t n_fft = 0;
    uint32_t hop = 0;
    uint32_t bins = 0;
    float dc_pole = 0.0f;
    float mag_clip = 0.0f;

    ggml_tensor * embed_w = nullptr;
    ggml_tensor * embed_b = nullptr;
    ggml_tensor * noise_w = nullptr;
    ggml_tensor * noise_b = nullptr;
    ggml_tensor * norm_w = nullptr;
    ggml_tensor * norm_b = nullptr;
    std::vector<SanoVocosBlock> block_weights;
    ggml_tensor * final_norm_w = nullptr;
    ggml_tensor * final_norm_b = nullptr;
    ggml_tensor * head_w = nullptr;
    ggml_tensor * head_b = nullptr;
};

// ---------------------------------------------------------------------------
// Architecture constants
// ---------------------------------------------------------------------------

/// The residual bank's three branches. Not metadata: they are baked into the
/// upstream module and the weight shapes only make sense with these values.
inline constexpr std::array<int32_t, 3> SANO_BANK_KERNELS{3, 5, 7};
inline constexpr std::array<int32_t, 3> SANO_BANK_DILATION1{1, 2, 3};
inline constexpr std::array<int32_t, 3> SANO_BANK_DILATION2{2, 6, 12};

/// Upsampling geometry of the three piperlite stages: kernel, stride, padding.
inline constexpr std::array<std::array<int32_t, 3>, 3> SANO_PIPER_STAGES{{
    {16, 8, 4},
    {16, 8, 4},
    {8, 4, 2},
}};

/// LayerNorm epsilon. Explicit because it is not ggml's default and the
/// reference uses torch's.
inline constexpr float SANO_LAYER_NORM_EPS = 1e-6f;

/// The emotion conditioner's own LayerNorm epsilon: PyTorch's `nn.LayerNorm`
/// default, which is not the vocos trunk's.
inline constexpr float SANO_EMOTION_NORM_EPS = 1e-5f;

/// The widest crossfade a pack may ask for at a phone boundary, matching the
/// reference's own ceiling. A bound, not a modelling choice: the window is
/// still limited to half of each of the two phones it joins.
inline constexpr uint32_t SANO_MAX_TRANSITION_FRAMES = 8;

/// Upper bound on the log-magnitude before `exp`, an overflow guard rather
/// than a modelling choice.
inline constexpr float SANO_LOG_MAG_CLAMP = 60.0f;

} // namespace kokopop
