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
/// built from.
struct SanoResBlock {
    ggml_tensor * net0_w = nullptr;
    ggml_tensor * net0_b = nullptr;
    ggml_tensor * net2_w = nullptr;
    ggml_tensor * net2_b = nullptr;

    /// Read to host at load time: it multiplies a tensor as a plain float, so
    /// keeping it as a one-element tensor would cost a graph node per block.
    float scale = 0.0f;

    int32_t kernel = 0;
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

/// The utterance style space and the duration model's style residual.
///
/// One capability, `sanofr.utterance-emotion.v1`, in its `concat` shape: the
/// style vector is a constant per utterance, concatenated to the acoustic
/// model's two input projections — which is why nothing here touches the
/// acoustic weights, they are simply `dim` channels wider — and the duration
/// model keeps a separate residual branch on top of its frozen neutral self.
///
/// `dim == 0` means the voice carries no emotion pack. A style vector of all
/// zeros is the neutral one and bypasses the duration residual exactly: the
/// branch is zero-init in the style vector and every term is multiplied by it.
struct SanoEmotionWeights {
    uint32_t dim = 0;
    uint32_t hidden = 0;

    /// Bound on the log-duration ratio the residual may apply, in either
    /// direction: the branch closes on `tanh(...) * max_log_ratio`.
    float max_log_ratio = 0.0f;

    /// `[dur.hidden + dim, hidden]`: the frozen model's token embedding, with
    /// the utterance's style vector concatenated to every token. Both
    /// projections are pointwise, so a token's residual reads that token only.
    ggml_tensor * input_proj_w = nullptr;
    ggml_tensor * input_proj_b = nullptr;

    /// `[hidden, 1]`: one log-duration ratio per token.
    ggml_tensor * output_w = nullptr;
    ggml_tensor * output_b = nullptr;

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

/// Upper bound on the log-magnitude before `exp`, an overflow guard rather
/// than a modelling choice.
inline constexpr float SANO_LOG_MAG_CLAMP = 60.0f;

} // namespace kokopop
