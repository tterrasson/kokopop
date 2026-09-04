// sanoTTS architecture: loading, validation, voice routing and the pipeline.
//
// A GGUF is untrusted input. Everything below validates before it allocates:
// each tensor must exist with exactly the rank and shape the metadata implies,
// every dimension must be non-zero and within a sane ceiling, kernels must be
// odd, and the redundancy between metadata and tensor shapes is checked rather
// than trusted on one side. A file that disagrees with itself is rejected with
// the disagreement named, not silently reinterpreted.

#include "arch/sanotts/sano_arch.h"

#include "arch/sanotts/sano_frontend_text.h"
#include "arch/sanotts/sano_noise.h"
#include "backend/backend.h"
#include "model/gguf_util.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_set>

#include <ggml.h>
#include <gguf.h>

namespace kokopop {

namespace {

/// Ceilings that separate "a big voice" from "a corrupt or hostile header".
/// They are far above every shipped voice and far below anything that could
/// overflow the size arithmetic below.
constexpr uint32_t SANO_MAX_DIM = 1u << 16;
constexpr uint32_t SANO_MAX_DEPTH = 64;
constexpr uint32_t SANO_MAX_KERNEL = 63;
constexpr uint32_t SANO_MAX_VOICES = 256;
constexpr uint32_t SANO_SUPPORTED_VERSION = 1;

/// Frames one chunk may reach. At hop 256 this is a little over three minutes
/// of audio; a chunk that long is a bug upstream, not a request to allocate
/// gigabytes.
constexpr int64_t SANO_MAX_FRAMES = 1 << 19;

std::string voice_prefix(size_t index) {
    return "kokopop.sanotts.voice." + std::to_string(index) + ".";
}

/// Reader for one voice's metadata block, so that every failure names the
/// voice and the key rather than "missing metadata".
struct VoiceMeta {
    gguf_context * ctx = nullptr;
    std::string prefix;
    std::string name;

    std::string where(const std::string & suffix) const {
        return "voice " + (name.empty() ? prefix : name) + ": " + prefix + suffix;
    }

    bool u32(const std::string & suffix, uint32_t & out, std::string & error) const {
        if (!gguf_get_u32(ctx, (prefix + suffix).c_str(), out)) {
            error = "missing " + where(suffix);
            return false;
        }
        return true;
    }

    bool f32(const std::string & suffix, float & out, std::string & error) const {
        if (!gguf_get_f32(ctx, (prefix + suffix).c_str(), out)) {
            error = "missing " + where(suffix);
            return false;
        }
        return true;
    }

    bool str(const std::string & suffix, std::string & out, std::string & error) const {
        if (!gguf_get_str(ctx, (prefix + suffix).c_str(), out)) {
            error = "missing " + where(suffix);
            return false;
        }
        return true;
    }
};

bool in_range(uint32_t value, uint32_t max, const char * what,
              const VoiceMeta & meta, std::string & error) {
    if (value == 0 || value > max) {
        error = meta.where(what) + " = " + std::to_string(value)
              + " is outside the supported range [1, " + std::to_string(max) + "]";
        return false;
    }
    return true;
}

bool odd_kernel(uint32_t kernel, const char * what, const VoiceMeta & meta,
                std::string & error) {
    if (!in_range(kernel, SANO_MAX_KERNEL, what, meta, error)) {
        return false;
    }
    if ((kernel % 2) == 0) {
        error = meta.where(what) + " = " + std::to_string(kernel)
              + " must be odd: the reference pads convolutions by kernel/2 on "
                "both sides, which only reproduces \"same\" padding for odd kernels";
        return false;
    }
    return true;
}

std::string shape_of(const ggml_tensor * tensor) {
    char buffer[96];
    std::snprintf(buffer, sizeof buffer, "[%lld, %lld, %lld, %lld]",
                  static_cast<long long>(tensor->ne[0]), static_cast<long long>(tensor->ne[1]),
                  static_cast<long long>(tensor->ne[2]), static_cast<long long>(tensor->ne[3]));
    return buffer;
}

} // namespace

// ---------------------------------------------------------------------------
// Tensor resolution
// ---------------------------------------------------------------------------

namespace {

/// Resolves a tensor and checks its shape. `ne2` of 0 means "must be 1D/2D".
class TensorLoader {
public:
    TensorLoader(const Model & model, std::string prefix, std::string voice,
                 std::string & error)
        : _model(model), _prefix(std::move(prefix)), _voice(std::move(voice)),
          _error(error) {}

    bool has_prefix(const std::string & suffix) const {
        const std::string prefix = _prefix + suffix;
        for (const auto & entry : _model.tensors) {
            if (entry.first.compare(0, prefix.size(), prefix) == 0) return true;
        }
        return false;
    }

    ggml_tensor * get(const std::string & suffix, int64_t ne0, int64_t ne1 = 1,
                      int64_t ne2 = 1) {
        if (!_error.empty()) {
            return nullptr;
        }
        const std::string name = _prefix + suffix;
        ggml_tensor * tensor = _model.tensor(name);
        if (tensor == nullptr) {
            _error = "voice " + _voice + ": missing tensor " + name;
            return nullptr;
        }
        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
            _error = "voice " + _voice + ": tensor " + name
                   + " must be stored as F32 or F16";
            return nullptr;
        }
        if (tensor->ne[0] != ne0 || tensor->ne[1] != ne1 ||
            tensor->ne[2] != ne2 || tensor->ne[3] != 1) {
            _error = "voice " + _voice + ": tensor " + name + " has shape "
                   + shape_of(tensor) + ", expected ["
                   + std::to_string(ne0) + ", " + std::to_string(ne1) + ", "
                   + std::to_string(ne2) + ", 1]";
            return nullptr;
        }
        return tensor;
    }

    /// Same, but the tensor must additionally be F32: near-1e-6 values in the
    /// normalisation and LayerScale weights are subnormal in F16 and a backend
    /// with flush-to-zero would quietly turn them into zeros.
    ggml_tensor * get_f32(const std::string & suffix, int64_t ne0, int64_t ne1 = 1) {
        ggml_tensor * tensor = get(suffix, ne0, ne1);
        if (tensor != nullptr && tensor->type != GGML_TYPE_F32) {
            _error = "voice " + _voice + ": tensor " + _prefix + suffix
                   + " must be stored as F32";
            return nullptr;
        }
        return tensor;
    }

private:
    const Model & _model;
    std::string _prefix;
    std::string _voice;
    std::string & _error;
};

/// Reads a one-element F32 tensor to host. Residual scales multiply a whole
/// tensor by a constant; keeping them as tensors would cost a graph node each.
bool read_scalar(Backend & backend, ggml_tensor * tensor, float & out,
                 const std::string & name, std::string & error) {
    std::vector<float> values;
    if (!tensor_to_f32(backend, tensor, values) || values.size() != 1) {
        error = "failed to read the scalar tensor " + name;
        return false;
    }
    if (!std::isfinite(values[0])) {
        error = "the scalar tensor " + name + " is not finite";
        return false;
    }
    out = values[0];
    return true;
}

/// Resolves `count` residual blocks named `<stem>.<i>.` with `channels`
/// channels and the declared kernel.
bool load_res_blocks(TensorLoader & loader, Backend & backend,
                     const std::string & prefix, const std::string & stem,
                     uint32_t count, uint32_t channels, uint32_t kernel,
                     std::vector<SanoResBlock> & blocks, std::string & error) {
    const int64_t ch = static_cast<int64_t>(channels);
    const int64_t flat = ch * static_cast<int64_t>(kernel);

    blocks.clear();
    blocks.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const std::string base = stem + "." + std::to_string(i) + ".";
        SanoResBlock block;
        block.kernel = static_cast<int32_t>(kernel);
        block.net0_w = loader.get(base + "net0.weight", flat, ch);
        block.net0_b = loader.get_f32(base + "net0.bias", ch);
        block.net2_w = loader.get(base + "net2.weight", flat, ch);
        block.net2_b = loader.get_f32(base + "net2.bias", ch);
        ggml_tensor * scale = loader.get_f32(base + "scale", 1);
        if (!error.empty()) {
            return false;
        }
        if (!read_scalar(backend, scale, block.scale, prefix + base + "scale", error)) {
            return false;
        }
        blocks.push_back(block);
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// SanoArch
// ---------------------------------------------------------------------------

SanoArch * sano_arch(Model & model) {
    if (model.arch == nullptr || model.arch->arch() != Arch::SanoTTS) {
        return nullptr;
    }
    return static_cast<SanoArch *>(model.arch.get());
}

ggml_tensor * SanoArch::tensor(const std::string & logical_name) const {
    return base != nullptr ? base->tensor(logical_name) : nullptr;
}

const VoiceDesc * SanoArch::find_voice(std::string_view name) const {
    for (const auto & voice : voice_descs) {
        if (voice.name == name) {
            return &voice;
        }
    }
    for (const auto & voice : voice_descs) {
        for (const auto & alias : voice.aliases) {
            if (alias == name) {
                return &voice;
            }
        }
    }
    return nullptr;
}

const VoiceDesc * SanoArch::default_voice() const {
    if (voice_descs.empty()) {
        return nullptr;
    }
    const size_t index = default_voice_index < voice_descs.size() ? default_voice_index : 0;
    return &voice_descs[index];
}

const SanoVoice * SanoArch::voice_for(const VoiceDesc & desc) const {
    for (const auto & voice : voice_weights) {
        if (voice.desc.name == desc.name) {
            return &voice;
        }
    }
    return nullptr;
}

const IstftPlan * SanoArch::istft_plan(uint32_t n_fft, uint32_t hop) const {
    const uint64_t key = (static_cast<uint64_t>(n_fft) << 32) | hop;
    const auto it = istft_plans.find(key);
    return it != istft_plans.end() ? it->second.get() : nullptr;
}

bool SanoArch::phonemize(const std::string & text, const VoiceDesc & voice,
                         std::string & phonemes, std::string & error) const {
    if (voice.frontend == FrontendKind::Piper) {
        return sano::phonemize_piper_sanotts(text, voice.espeak_voice, phonemes, error);
    }
    return sano::phonemize_misaki_sanotts(text, voice.espeak_voice, phonemes, error);
}

bool SanoArch::tokenize(const std::string & phonemes, const VoiceDesc & voice,
                        std::vector<uint32_t> & ids, std::string & error) const {
    const SanoVoice * resolved = voice_for(voice);
    if (resolved == nullptr) {
        error = "unknown sanoTTS voice: " + voice.name;
        return false;
    }
    if (voice.frontend == FrontendKind::Piper) {
        return sano::tokenize_piper(phonemes, resolved->tokens, nfd, ids, error);
    }
    return sano::tokenize_misaki(phonemes, resolved->tokens, ids, error);
}

ChunkConfig SanoArch::adjust_chunk_config(ChunkConfig cfg,
                                          const VoiceDesc & voice) const {
    // The duration model was trained at a fixed maximum token count, and its
    // `length_hint` feature is a function of that ceiling: exceeding it does
    // not degrade gracefully, it moves the model off its training manifold.
    const int limit = voice.max_tokens;
    if (limit <= 0) {
        return cfg;
    }

    cfg.hard_max_tokens = std::min(cfg.hard_max_tokens, limit);
    cfg.soft_max_tokens = std::min(cfg.soft_max_tokens, cfg.hard_max_tokens);
    cfg.target_max_tokens = std::min(cfg.target_max_tokens, cfg.soft_max_tokens);
    cfg.target_min_tokens = std::min(cfg.target_min_tokens, cfg.target_max_tokens);
    cfg.first_chunk_target_max_tokens =
        std::min(cfg.first_chunk_target_max_tokens, cfg.target_max_tokens);
    return cfg;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

namespace {

bool load_nfd(gguf_context * meta, SanoArch & arch, std::string & error) {
    if (!gguf_get_u32_array(meta, "kokopop.sanotts.nfd_codepoints", arch.nfd_codepoints) ||
        !gguf_get_u32_array(meta, "kokopop.sanotts.nfd_offsets", arch.nfd_offsets) ||
        !gguf_get_u32_array(meta, "kokopop.sanotts.nfd_values", arch.nfd_values) ||
        !gguf_get_u32_array(meta, "kokopop.sanotts.nfd_ccc_codepoints", arch.nfd_ccc_codepoints) ||
        !gguf_get_u32_array(meta, "kokopop.sanotts.nfd_ccc_classes", arch.nfd_ccc_classes)) {
        error = "kokopop.sanotts.nfd_*: missing or incorrectly typed array";
        return false;
    }

    arch.nfd.codepoints = arch.nfd_codepoints.data();
    arch.nfd.offsets = arch.nfd_offsets.data();
    arch.nfd.values = arch.nfd_values.data();
    arch.nfd.count = arch.nfd_codepoints.size();
    arch.nfd.n_offsets = arch.nfd_offsets.size();
    arch.nfd.n_values = arch.nfd_values.size();
    arch.nfd.ccc_codepoints = arch.nfd_ccc_codepoints.data();
    arch.nfd.ccc_classes = arch.nfd_ccc_classes.data();
    arch.nfd.ccc_count = arch.nfd_ccc_codepoints.size();
    arch.nfd.n_ccc_classes = arch.nfd_ccc_classes.size();

    if (!arch.nfd.validate(error)) {
        error = "kokopop.sanotts.nfd_*: " + error;
        return false;
    }
    return true;
}

bool load_token_table(const VoiceMeta & meta, sano::TokenTable & table,
                      std::string & error) {
    std::vector<std::string> symbols;
    std::vector<uint32_t> ids;
    if (!gguf_get_str_array(meta.ctx, (meta.prefix + "token_symbols").c_str(), symbols)) {
        error = "missing " + meta.where("token_symbols");
        return false;
    }
    if (!gguf_get_u32_array(meta.ctx, (meta.prefix + "token_ids").c_str(), ids)) {
        error = "missing " + meta.where("token_ids");
        return false;
    }
    if (symbols.size() != ids.size() || symbols.empty()) {
        error = meta.where("token_symbols") + " and token_ids have different lengths";
        return false;
    }

    for (size_t i = 0; i < symbols.size(); ++i) {
        if (!table.to_id.emplace(symbols[i], ids[i]).second) {
            error = meta.where("token_symbols") + " lists " + symbols[i] + " twice";
            return false;
        }
    }

    if (!meta.u32("bos_id", table.bos_id, error) ||
        !meta.u32("eos_id", table.eos_id, error)) {
        return false;
    }

    int32_t pad = -1;
    int32_t fallback = -1;
    if (!gguf_get_i32(meta.ctx, (meta.prefix + "pad_id").c_str(), pad)) {
        error = "missing " + meta.where("pad_id");
        return false;
    }
    if (!gguf_get_i32(meta.ctx, (meta.prefix + "fallback_id").c_str(), fallback)) {
        error = "missing " + meta.where("fallback_id");
        return false;
    }
    table.pad_id = pad;
    table.fallback_id = fallback;

    // Whatever the input text contains, a framing symbol must never be emitted
    // from it. Deriving the set from the ids rather than hard-coding names
    // keeps it correct for a vocabulary that spells them differently.
    for (const auto & entry : table.to_id) {
        if (entry.second == table.bos_id || entry.second == table.eos_id ||
            (pad >= 0 && entry.second == static_cast<uint32_t>(pad))) {
            table.special_symbols.insert(entry.first);
        }
    }

    if (!table.validate(error)) {
        error = "voice " + meta.name + ": " + error;
        return false;
    }
    return true;
}

bool load_duration(const VoiceMeta & meta, TensorLoader & loader, Backend & backend,
                   SanoDurationWeights & w, std::string & error) {
    if (!meta.u32("dur.vocab", w.vocab, error) ||
        !meta.u32("dur.hidden", w.hidden, error) ||
        !meta.u32("dur.depth", w.depth, error) ||
        !meta.u32("dur.kernel", w.kernel, error) ||
        !meta.u32("dur.max_tokens", w.max_tokens, error) ||
        !meta.u32("dur.max_duration", w.max_duration, error)) {
        return false;
    }
    if (!in_range(w.vocab, SANO_MAX_DIM, "dur.vocab", meta, error) ||
        !in_range(w.hidden, SANO_MAX_DIM, "dur.hidden", meta, error) ||
        !in_range(w.depth, SANO_MAX_DEPTH, "dur.depth", meta, error) ||
        !odd_kernel(w.kernel, "dur.kernel", meta, error) ||
        !in_range(w.max_tokens, SANO_MAX_DIM, "dur.max_tokens", meta, error) ||
        !in_range(w.max_duration, SANO_MAX_DIM, "dur.max_duration", meta, error)) {
        return false;
    }

    const int64_t hidden = static_cast<int64_t>(w.hidden);
    w.embedding    = loader.get("dur.embedding.weight", hidden, w.vocab);
    w.input_proj_w = loader.get("dur.input_proj.weight", hidden + 3, hidden);
    w.input_proj_b = loader.get_f32("dur.input_proj.bias", hidden);
    if (!error.empty()) {
        return false;
    }
    if (!load_res_blocks(loader, backend, meta.prefix, "dur.blocks", w.depth,
                         w.hidden, w.kernel, w.blocks, error)) {
        return false;
    }
    // The duration projection is computed on the CPU sub-backend on every
    // backend (see sano_run_duration), which is only meaningful if it is not
    // already lossy in the file.
    w.output_w = loader.get_f32("dur.output.weight", hidden, 1);
    w.output_b = loader.get_f32("dur.output.bias", 1);
    return error.empty();
}

bool load_acoustic(const VoiceMeta & meta, TensorLoader & loader, Backend & backend,
                   SanoAcousticWeights & w, std::string & error) {
    if (!meta.u32("ac.vocab", w.vocab, error) ||
        !meta.u32("ac.hidden", w.hidden, error) ||
        !meta.u32("ac.token_depth", w.token_depth, error) ||
        !meta.u32("ac.depth", w.depth, error) ||
        !meta.u32("ac.kernel", w.kernel, error) ||
        !meta.u32("ac.out_channels", w.out_channels, error)) {
        return false;
    }
    if (!in_range(w.vocab, SANO_MAX_DIM, "ac.vocab", meta, error) ||
        !in_range(w.hidden, SANO_MAX_DIM, "ac.hidden", meta, error) ||
        !in_range(w.token_depth, SANO_MAX_DEPTH, "ac.token_depth", meta, error) ||
        !in_range(w.depth, SANO_MAX_DEPTH, "ac.depth", meta, error) ||
        !odd_kernel(w.kernel, "ac.kernel", meta, error) ||
        !in_range(w.out_channels, SANO_MAX_DIM, "ac.out_channels", meta, error)) {
        return false;
    }

    const int64_t hidden = static_cast<int64_t>(w.hidden);
    w.embedding     = loader.get("ac.embedding.weight", hidden, w.vocab);
    w.token_proj_w  = loader.get("ac.token_input_proj.weight", hidden + 2, hidden);
    w.token_proj_b  = loader.get_f32("ac.token_input_proj.bias", hidden);
    w.frame_proj_w  = loader.get("ac.frame_input_proj.weight", hidden + 3, hidden);
    w.frame_proj_b  = loader.get_f32("ac.frame_input_proj.bias", hidden);
    if (!error.empty()) {
        return false;
    }
    if (!load_res_blocks(loader, backend, meta.prefix, "ac.token_blocks", w.token_depth,
                         w.hidden, w.kernel, w.token_blocks, error) ||
        !load_res_blocks(loader, backend, meta.prefix, "ac.frame_blocks", w.depth,
                         w.hidden, w.kernel, w.frame_blocks, error)) {
        return false;
    }
    w.output_w = loader.get("ac.output.weight", hidden, w.out_channels);
    w.output_b = loader.get_f32("ac.output.bias", w.out_channels);
    return error.empty();
}

bool load_piperlite(const VoiceMeta & meta, TensorLoader & loader, Backend & backend,
                    uint32_t latent_channels, SanoPiperliteWeights & w,
                    std::string & error) {
    std::vector<uint32_t> channels;
    if (!gguf_get_u32_array(meta.ctx, (meta.prefix + "dec.channels").c_str(), channels) ||
        channels.size() != 4) {
        error = meta.where("dec.channels") + " must list exactly four channel counts";
        return false;
    }
    for (size_t i = 0; i < 4; ++i) {
        if (!in_range(channels[i], SANO_MAX_DIM, "dec.channels", meta, error)) {
            return false;
        }
        w.channels[i] = channels[i];
    }
    if (!meta.u32("dec.pre_kernel", w.pre_kernel, error) ||
        !meta.u32("dec.post_kernel", w.post_kernel, error) ||
        !odd_kernel(w.pre_kernel, "dec.pre_kernel", meta, error) ||
        !odd_kernel(w.post_kernel, "dec.post_kernel", meta, error)) {
        return false;
    }

    const int64_t latent = static_cast<int64_t>(latent_channels);
    w.pre_w = loader.get("dec.pre.weight", latent * w.pre_kernel, w.channels[0]);
    w.pre_b = loader.get_f32("dec.pre.bias", w.channels[0]);

    for (size_t s = 0; s < 3; ++s) {
        SanoPiperStage & stage = w.stages[s];
        stage.up_kernel  = SANO_PIPER_STAGES[s][0];
        stage.up_stride  = SANO_PIPER_STAGES[s][1];
        stage.up_padding = SANO_PIPER_STAGES[s][2];

        const int64_t in_ch = static_cast<int64_t>(w.channels[s]);
        const int64_t out_ch = static_cast<int64_t>(w.channels[s + 1]);
        const std::string up = "dec.up" + std::to_string(s) + ".";
        stage.up_w = loader.get(up + "weight", stage.up_kernel, out_ch, in_ch);
        stage.up_b = loader.get_f32(up + "bias", out_ch);

        std::vector<uint32_t> branches;
        if (!gguf_get_u32_array(meta.ctx,
                                (meta.prefix + "dec.branches." + std::to_string(s)).c_str(),
                                branches)) {
            branches = {0, 1, 2};
        }
        if (branches.empty()) {
            error = meta.where("dec.branches." + std::to_string(s)) + " is empty";
            return false;
        }
        std::unordered_set<uint32_t> seen;
        for (const uint32_t branch : branches) {
            if (branch > 2 || !seen.insert(branch).second) {
                error = meta.where("dec.branches." + std::to_string(s))
                      + " must be a subset of [0, 1, 2] without repeats";
                return false;
            }
        }
        stage.branches = branches;

        const std::string bank = "dec.res" + std::to_string(s) + ".blocks.";
        for (const uint32_t branch : branches) {
            SanoPiperBranch & entry = stage.bank[branch];
            entry.kernel = SANO_BANK_KERNELS[branch];
            entry.dilation1 = SANO_BANK_DILATION1[branch];
            entry.dilation2 = SANO_BANK_DILATION2[branch];
            const std::string base = bank + std::to_string(branch) + ".";
            const int64_t flat = out_ch * entry.kernel;
            entry.conv1_w = loader.get(base + "conv1.weight", flat, out_ch);
            entry.conv1_b = loader.get_f32(base + "conv1.bias", out_ch);
            entry.conv2_w = loader.get(base + "conv2.weight", flat, out_ch);
            entry.conv2_b = loader.get_f32(base + "conv2.bias", out_ch);
        }
    }

    const int64_t last = static_cast<int64_t>(w.channels[3]);
    w.post_w = loader.get("dec.post.weight", last * w.post_kernel, 1);
    w.post_b = loader.get_f32("dec.post.bias", 1);
    if (!error.empty()) {
        return false;
    }

    if (!meta.u32("dec.post_filter_channels", w.post_filter_channels, error)) return false;
    if (w.post_filter_channels == 0) {
        if (loader.has_prefix("dec.post_filter.")) {
            error = meta.where("dec.post_filter_channels")
                  + " disables a post-filter whose tensors are present";
            return false;
        }
        return true;
    }

    if (!meta.u32("dec.post_filter_layers", w.post_filter_layers, error) ||
        !meta.u32("dec.post_filter_kernel", w.post_filter_kernel, error) ||
        !meta.u32("dec.post_filter_unit_kernel", w.post_filter_unit_kernel, error)) return false;
    if (!meta.f32("dec.post_filter_scale", w.post_filter_scale, error)) {
        return false;
    }
    if (!in_range(w.post_filter_channels, SANO_MAX_DIM, "dec.post_filter_channels", meta, error) ||
        !in_range(w.post_filter_layers, SANO_MAX_DEPTH, "dec.post_filter_layers", meta, error) ||
        !odd_kernel(w.post_filter_kernel, "dec.post_filter_kernel", meta, error) ||
        !odd_kernel(w.post_filter_unit_kernel, "dec.post_filter_unit_kernel", meta, error)) {
        return false;
    }
    if (!std::isfinite(w.post_filter_scale)) {
        error = meta.where("dec.post_filter_scale") + " is not finite";
        return false;
    }

    const int64_t pf = static_cast<int64_t>(w.post_filter_channels);
    const int64_t pfk = static_cast<int64_t>(w.post_filter_kernel);
    const int64_t unit_k = static_cast<int64_t>(w.post_filter_unit_kernel);
    w.post_filter_in_w  = loader.get("dec.post_filter.in_conv.weight", pfk, pf);
    w.post_filter_in_b  = loader.get_f32("dec.post_filter.in_conv.bias", pf);
    w.post_filter_out_w = loader.get("dec.post_filter.out_conv.weight", pf * pfk, 1);
    w.post_filter_out_b = loader.get_f32("dec.post_filter.out_conv.bias", 1);

    w.post_filter_units.clear();
    w.post_filter_units.reserve(w.post_filter_layers);
    for (uint32_t layer = 0; layer < w.post_filter_layers; ++layer) {
        const std::string base = "dec.post_filter.units." + std::to_string(layer) + ".";
        SanoPostFilterUnit unit;
        unit.kernel = static_cast<int32_t>(unit_k);
        // The reference dilates the first convolution of unit `l` by `1 + l`
        // and leaves the second at 1.
        unit.dilation1 = static_cast<int32_t>(1 + layer);
        unit.conv1_w = loader.get(base + "conv1.weight", pf * unit_k, pf);
        unit.conv1_b = loader.get_f32(base + "conv1.bias", pf);
        unit.conv2_w = loader.get(base + "conv2.weight", pf * unit_k, pf);
        unit.conv2_b = loader.get_f32(base + "conv2.bias", pf);
        ggml_tensor * scale = loader.get_f32(base + "scale", 1);
        if (!error.empty()) {
            return false;
        }
        if (!read_scalar(backend, scale, unit.scale, meta.prefix + base + "scale", error)) {
            return false;
        }
        w.post_filter_units.push_back(unit);
    }
    return error.empty();
}

bool load_vocos(const VoiceMeta & meta, TensorLoader & loader, uint32_t mels,
                SanoVocosWeights & w, std::string & error) {
    if (!meta.u32("dec.dim", w.dim, error) ||
        !meta.u32("dec.blocks", w.blocks, error) ||
        !meta.u32("dec.pw_hidden", w.pw_hidden, error) ||
        !meta.u32("dec.dw_kernel", w.dw_kernel, error) ||
        !meta.u32("dec.embed_kernel", w.embed_kernel, error) ||
        !meta.u32("dec.noise_ch", w.noise_ch, error) ||
        !meta.u32("dec.mels", w.mels, error) ||
        !meta.u32("dec.n_fft", w.n_fft, error) ||
        !meta.u32("dec.hop", w.hop, error) ||
        !meta.u32("dec.bins", w.bins, error) ||
        !meta.f32("dec.dc_pole", w.dc_pole, error) ||
        !meta.f32("dec.mag_clip", w.mag_clip, error)) {
        return false;
    }

    uint32_t norm_type = 0;
    uint32_t act_type = 0;
    if (!meta.u32("dec.norm_type", norm_type, error) ||
        !meta.u32("dec.act_type", act_type, error)) return false;
    if (norm_type != 0 || act_type != 0) {
        // DyT normalisation and ReLU are the E13 arm. No shipped voice uses
        // them, so running them as LayerNorm/GELU would be a silent
        // substitution rather than a limitation.
        error = meta.where("dec.norm_type/act_type") + " selects an operator arm "
                "this build does not implement (norm_type="
              + std::to_string(norm_type) + ", act_type=" + std::to_string(act_type)
              + "); only 0/0 (LayerNorm + exact GELU) is supported";
        return false;
    }

    if (!in_range(w.dim, SANO_MAX_DIM, "dec.dim", meta, error) ||
        !in_range(w.blocks, SANO_MAX_DEPTH, "dec.blocks", meta, error) ||
        !in_range(w.pw_hidden, SANO_MAX_DIM, "dec.pw_hidden", meta, error) ||
        !odd_kernel(w.dw_kernel, "dec.dw_kernel", meta, error) ||
        !odd_kernel(w.embed_kernel, "dec.embed_kernel", meta, error) ||
        !in_range(w.noise_ch, SANO_MAX_DIM, "dec.noise_ch", meta, error) ||
        !in_range(w.mels, SANO_MAX_DIM, "dec.mels", meta, error) ||
        !in_range(w.n_fft, SANO_MAX_DIM, "dec.n_fft", meta, error) ||
        !in_range(w.hop, SANO_MAX_DIM, "dec.hop", meta, error)) {
        return false;
    }
    if (w.mels != mels) {
        error = meta.where("dec.mels") + " = " + std::to_string(w.mels)
              + " disagrees with ac.out_channels = " + std::to_string(mels);
        return false;
    }
    if (w.bins != w.n_fft / 2 + 1) {
        error = meta.where("dec.bins") + " = " + std::to_string(w.bins)
              + " must equal n_fft/2 + 1 = " + std::to_string(w.n_fft / 2 + 1);
        return false;
    }
    if (w.hop == 0 || w.hop > w.n_fft) {
        error = meta.where("dec.hop") + " must lie in (0, n_fft]";
        return false;
    }
    if (!std::isfinite(w.dc_pole) || std::fabs(w.dc_pole) >= 1.0f) {
        error = meta.where("dec.dc_pole") + " must be a stable pole in (-1, 1)";
        return false;
    }
    if (!std::isfinite(w.mag_clip) || w.mag_clip <= 0.0f) {
        error = meta.where("dec.mag_clip") + " must be finite and positive";
        return false;
    }

    const int64_t dim = static_cast<int64_t>(w.dim);
    const int64_t hidden = static_cast<int64_t>(w.pw_hidden);
    w.embed_w = loader.get("dec.embed.weight",
                           static_cast<int64_t>(w.mels) * w.embed_kernel, dim);
    w.embed_b = loader.get_f32("dec.embed.bias", dim);
    w.noise_w = loader.get("dec.noise.weight",
                           static_cast<int64_t>(w.noise_ch) * w.embed_kernel, dim);
    w.noise_b = loader.get_f32("dec.noise.bias", dim);
    w.norm_w  = loader.get_f32("dec.norm.weight", dim);
    w.norm_b  = loader.get_f32("dec.norm.bias", dim);

    w.block_weights.clear();
    w.block_weights.reserve(w.blocks);
    for (uint32_t i = 0; i < w.blocks; ++i) {
        const std::string base = "dec.blocks." + std::to_string(i) + ".";
        SanoVocosBlock block;
        block.dw_w   = loader.get_f32(base + "dw.weight", w.dw_kernel, dim);
        block.dw_b   = loader.get_f32(base + "dw.bias", dim);
        block.norm_w = loader.get_f32(base + "norm.weight", dim);
        block.norm_b = loader.get_f32(base + "norm.bias", dim);
        block.pw0_w  = loader.get(base + "pw0.weight", dim, hidden);
        block.pw0_b  = loader.get_f32(base + "pw0.bias", hidden);
        block.pw1_w  = loader.get(base + "pw1.weight", hidden, dim);
        block.pw1_b  = loader.get_f32(base + "pw1.bias", dim);
        block.gamma  = loader.get_f32(base + "gamma", dim);
        if (!error.empty()) {
            return false;
        }
        w.block_weights.push_back(block);
    }

    w.final_norm_w = loader.get_f32("dec.final_norm.weight", dim);
    w.final_norm_b = loader.get_f32("dec.final_norm.bias", dim);
    w.head_w = loader.get("dec.head.weight", dim, 2 * static_cast<int64_t>(w.bins));
    w.head_b = loader.get_f32("dec.head.bias", 2 * static_cast<int64_t>(w.bins));
    return error.empty();
}

} // namespace

bool SanoArch::load(Model & base_model, std::string & error) {
    base = &base_model;
    backend = base_model.backend.get();
    if (backend == nullptr) {
        error = "sanoTTS model loaded without a backend";
        return false;
    }

    gguf_context * meta = base_model.gguf_ctx;

    uint32_t version = 0;
    if (!gguf_get_u32(meta, "kokopop.sanotts.version", version)) {
        error = "GGUF declares kokopop.arch=\"sanotts\" but no kokopop.sanotts.version";
        return false;
    }
    if (version != SANO_SUPPORTED_VERSION) {
        error = "unsupported sanoTTS GGUF version " + std::to_string(version)
              + "; this build implements version "
              + std::to_string(SANO_SUPPORTED_VERSION);
        return false;
    }
    base_model.version = version;

    gguf_get_str(meta, "kokopop.sanotts.source", provenance);

    if (!load_nfd(meta, *this, error)) {
        return false;
    }

    std::vector<std::string> names;
    if (!gguf_get_str_array(meta, "kokopop.voices", names) || names.empty()) {
        error = "sanoTTS GGUF has no kokopop.voices";
        return false;
    }
    if (names.size() > SANO_MAX_VOICES) {
        error = "sanoTTS GGUF declares " + std::to_string(names.size())
              + " voices, above the supported ceiling of "
              + std::to_string(SANO_MAX_VOICES);
        return false;
    }

    voice_weights.clear();
    voice_weights.reserve(names.size());
    voice_descs.clear();
    voice_descs.reserve(names.size());

    std::unordered_set<std::string> claimed;
    for (size_t index = 0; index < names.size(); ++index) {
        VoiceMeta info;
        info.ctx = meta;
        info.prefix = voice_prefix(index);
        info.name = names[index];

        std::string declared;
        if (!info.str("name", declared, error)) {
            return false;
        }
        if (declared != names[index]) {
            error = "kokopop.voices[" + std::to_string(index) + "] = " + names[index]
                  + " but " + info.prefix + "name = " + declared;
            return false;
        }

        SanoVoice voice;
        voice.index = index;
        voice.desc.name = declared;
        gguf_get_str_array(meta, (info.prefix + "aliases").c_str(), voice.desc.aliases);

        uint32_t sample_rate = 0;
        if (!info.u32("sample_rate", sample_rate, error) ||
            !info.f32("length_scale", voice.desc.length_scale, error) ||
            !info.str("espeak_voice", voice.desc.espeak_voice, error)) {
            return false;
        }
        if (sample_rate < 8000 || sample_rate > 192000) {
            error = info.where("sample_rate") + " = " + std::to_string(sample_rate)
                  + " is not a plausible audio sample rate";
            return false;
        }
        if (!std::isfinite(voice.desc.length_scale) || voice.desc.length_scale <= 0.0f) {
            error = info.where("length_scale") + " must be finite and positive";
            return false;
        }
        voice.desc.sample_rate = static_cast<int32_t>(sample_rate);

        std::string normalization;
        if (!info.str("normalization_lang", normalization, error)) {
            return false;
        }
        voice.desc.normalization_lang = normalization.empty() ? 'a' : normalization[0];

        std::string frontend;
        std::string decoder;
        if (!info.str("frontend", frontend, error) || !info.str("decoder", decoder, error)) {
            return false;
        }
        if (frontend == "piper") {
            voice.desc.frontend = FrontendKind::Piper;
        } else if (frontend == "misaki") {
            voice.desc.frontend = FrontendKind::Misaki;
        } else {
            error = info.where("frontend") + " = " + frontend + " is not \"piper\" or \"misaki\"";
            return false;
        }
        if (decoder == "piperlite") {
            voice.desc.decoder = DecoderKind::PiperLite;
        } else if (decoder == "vocos") {
            voice.desc.decoder = DecoderKind::Vocos;
        } else {
            error = info.where("decoder") + " = " + decoder + " is not \"piperlite\" or \"vocos\"";
            return false;
        }

        uint32_t max_tokens = 0;
        if (!info.u32("max_tokens", max_tokens, error) ||
            !in_range(max_tokens, SANO_MAX_DIM, "max_tokens", info, error)) {
            return false;
        }
        voice.desc.max_tokens = static_cast<int32_t>(max_tokens);

        if (!load_token_table(info, voice.tokens, error)) {
            return false;
        }
        voice.tokens.max_tokens = max_tokens;
        const uint32_t framing_min = voice.desc.frontend == FrontendKind::Piper ? 5 : 3;
        if (max_tokens < framing_min) {
            error = info.where("max_tokens") + " cannot fit framing and one phoneme";
            return false;
        }

        TensorLoader loader(base_model, info.prefix, declared, error);
        if (!load_duration(info, loader, *backend, voice.dur, error) ||
            !load_acoustic(info, loader, *backend, voice.ac, error)) {
            return false;
        }
        if (voice.dur.max_tokens != max_tokens) {
            error = info.where("dur.max_tokens") + " disagrees with max_tokens";
            return false;
        }
        const uint32_t vocab = std::min(voice.dur.vocab, voice.ac.vocab);
        if (voice.tokens.bos_id >= vocab || voice.tokens.eos_id >= vocab ||
            (voice.tokens.pad_id >= 0 && static_cast<uint32_t>(voice.tokens.pad_id) >= vocab) ||
            (voice.tokens.fallback_id >= 0 && static_cast<uint32_t>(voice.tokens.fallback_id) >= vocab)) {
            error = info.where("token_ids") + ": special ids must fit both embedding vocabularies";
            return false;
        }
        if (voice.desc.frontend == FrontendKind::Piper &&
            (voice.tokens.pad_id < 0 || !nfd.present())) {
            error = "voice " + declared + ": Piper requires a PAD id and an NFD table";
            return false;
        }

        if (voice.desc.decoder == DecoderKind::PiperLite) {
            if (!load_piperlite(info, loader, *backend, voice.ac.out_channels,
                                voice.piperlite, error)) {
                return false;
            }
        } else {
            if (!load_vocos(info, loader, voice.ac.out_channels, voice.vocos, error)) {
                return false;
            }
            const uint64_t key = (static_cast<uint64_t>(voice.vocos.n_fft) << 32)
                               | voice.vocos.hop;
            if (istft_plans.find(key) == istft_plans.end()) {
                auto plan = std::make_unique<IstftPlan>();
                IstftConfig config;
                config.n_fft = voice.vocos.n_fft;
                config.hop = voice.vocos.hop;
                config.center = true;
                if (!IstftPlan::create(config, *plan, error)) {
                    error = "voice " + declared + ": " + error;
                    return false;
                }
                istft_plans.emplace(key, std::move(plan));
            }
        }

        if (!claimed.insert(declared).second) {
            error = "voice name " + declared + " appears twice in the model";
            return false;
        }
        for (const std::string & alias : voice.desc.aliases) {
            if (!claimed.insert(alias).second) {
                error = "voice alias " + alias + " collides with another voice";
                return false;
            }
        }

        voice_descs.push_back(voice.desc);
        voice_weights.push_back(std::move(voice));
    }

    default_voice_index = 0;
    std::string default_name;
    if (gguf_get_str(meta, "kokopop.default_voice", default_name) && !default_name.empty()) {
        const auto it = std::find(names.begin(), names.end(), default_name);
        if (it == names.end()) {
            error = "kokopop.default_voice names a voice absent from the model: "
                  + default_name;
            return false;
        }
        default_voice_index = static_cast<size_t>(it - names.begin());
    }

    return true;
}

// ---------------------------------------------------------------------------
// Graph budgets
// ---------------------------------------------------------------------------

namespace {

/// Nodes one residual block emits: two "same" convolutions (proxy, im2col,
/// reshape, cont, matmul, transpose, cont), their bias adds, a SiLU, a scale
/// and the residual add. Measured at 16; rounded up.
constexpr size_t SANO_NODES_PER_RES_BLOCK = 24;

/// Embedding, feature concat, the input projection and the two transposes
/// around the convolutional stack.
constexpr size_t SANO_FRONTEND_BASE_NODES = 64;

SanoGraphBudget budget(size_t nodes) {
    // Leaves and view tensors outnumber compute nodes but not by much; the
    // arena is floored at the backend's own margin anyway, so a loose factor
    // here costs nothing.
    return SanoGraphBudget{nodes * 4, nodes};
}

} // namespace

SanoGraphBudget sano_duration_budget(const SanoVoice & voice) {
    return budget(SANO_FRONTEND_BASE_NODES +
                  SANO_NODES_PER_RES_BLOCK * voice.dur.depth);
}

SanoGraphBudget sano_acoustic_token_budget(const SanoVoice & voice) {
    return budget(SANO_FRONTEND_BASE_NODES +
                  SANO_NODES_PER_RES_BLOCK * voice.ac.token_depth);
}

SanoGraphBudget sano_acoustic_frame_budget(const SanoVoice & voice) {
    return budget(SANO_FRONTEND_BASE_NODES +
                  SANO_NODES_PER_RES_BLOCK * voice.ac.depth);
}

SanoGraphBudget sano_piperlite_budget(const SanoVoice & voice) {
    // pre + post + tanh, then per stage a leaky ReLU, a transposed
    // convolution and up to three branches of two dilated convolutions.
    size_t nodes = 128;
    for (const auto & stage : voice.piperlite.stages) {
        nodes += 16 + stage.branches.size() * SANO_NODES_PER_RES_BLOCK;
    }
    nodes += 32 + SANO_NODES_PER_RES_BLOCK * voice.piperlite.post_filter_units.size();
    return budget(nodes);
}

SanoGraphBudget sano_vocos_budget(const SanoVoice & voice) {
    // Two stem convolutions, three norms, the head and the spectrum, then per
    // block a depthwise convolution, a norm, two pointwise projections and the
    // LayerScale residual.
    return budget(128 + SANO_NODES_PER_RES_BLOCK * voice.vocos.blocks);
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

bool SanoArch::run(const std::vector<uint32_t> & ids, const VoiceDesc & desc,
                   float speed, const SynthesisExtras & extras,
                   SanoProbe & probe, std::string & error) {
    const SanoVoice * voice = voice_for(desc);
    if (voice == nullptr) {
        error = "unknown sanoTTS voice: " + desc.name;
        return false;
    }
    if (ids.empty()) {
        error = "sanoTTS synthesis: empty token sequence";
        return false;
    }
    if (ids.size() > voice->dur.max_tokens) {
        error = "sanoTTS synthesis: token sequence exceeds the voice's limit of "
              + std::to_string(voice->dur.max_tokens);
        return false;
    }
    if (!(speed > 0.0f) || !std::isfinite(speed)) {
        error = "sanoTTS synthesis: speed must be finite and positive";
        return false;
    }

    // `speed` scales the timeline, and the timeline is exactly what
    // `length_scale` controls: a faster rendering is a shorter duration for
    // every token.
    const float length_scale = desc.length_scale / speed;

    if (!extras.dur_override.empty()) {
        if (extras.dur_override.size() != ids.size()) {
            error = "sanoTTS synthesis: dur_override has "
                  + std::to_string(extras.dur_override.size()) + " entries for "
                  + std::to_string(ids.size()) + " tokens";
            return false;
        }
        for (const int32_t duration : extras.dur_override) {
            if (duration < 1 || duration > static_cast<int32_t>(voice->dur.max_duration)) {
                error = "sanoTTS synthesis: dur_override contains "
                      + std::to_string(duration) + ", outside [1, "
                      + std::to_string(voice->dur.max_duration) + "]";
                return false;
            }
        }
        probe.durations = extras.dur_override;
    } else if (!sano_run_duration(*this, *voice, ids, length_scale, probe.durations, error)) {
        return false;
    }

    int64_t frames = 0;
    for (const int32_t duration : probe.durations) {
        frames += duration;
        if (frames > SANO_MAX_FRAMES) {
            error = "sanoTTS synthesis: the chunk expands to more than "
                  + std::to_string(SANO_MAX_FRAMES) + " frames";
            return false;
        }
    }
    if (frames < 2) {
        error = "sanoTTS synthesis: the chunk expands to fewer than two frames";
        return false;
    }

    probe.n_tokens = static_cast<int64_t>(ids.size());
    probe.frames = frames;

    if (!sano_run_acoustic_token(*this, *voice, ids, probe.durations, probe.token_ctx, error)) {
        return false;
    }

    std::vector<float> frame_input;
    sano_expand_to_frames(probe.token_ctx, probe.n_tokens, voice->ac.hidden,
                          probe.durations, frames, frame_input);

    if (!sano_run_acoustic_frame(*this, *voice, frame_input, frames, probe.latent, error)) {
        return false;
    }
    frame_input.clear();
    frame_input.shrink_to_fit();

    if (desc.decoder == DecoderKind::PiperLite) {
        return sano_run_piperlite(*this, *voice, probe.latent, frames, probe.audio, error);
    }

    // Chunk 0 draws from the base seed itself, so a caller who pins a seed gets
    // exactly the stream that seed names — which is what lets a test inject the
    // seed a golden fixture was rendered with. Later chunks fold their index in:
    // reusing one draw across consecutive chunks is audible.
    const uint64_t base_seed = extras.has_noise_seed
        ? extras.noise_seed
        : sano::derive_base_seed(provenance, desc.name, ids);
    const uint64_t seed = extras.chunk_index == 0
        ? base_seed
        : sano::chunk_seed(base_seed, extras.chunk_index);

    return sano_run_vocos(*this, *voice, probe.latent, frames, seed, probe.audio, error);
}

bool SanoArch::synthesize(Model & base_model,
                          const std::vector<uint32_t> & ids,
                          const VoiceDesc & voice,
                          float speed,
                          const SynthesisExtras & extras,
                          std::vector<float> & audio,
                          std::string & error) {
    (void)base_model;

    SanoProbe probe;
    if (!run(ids, voice, speed, extras, probe, error)) {
        return false;
    }
    if (probe.audio.empty()) {
        error = "sanoTTS decoder produced no audio";
        return false;
    }
    audio = std::move(probe.audio);
    return true;
}

} // namespace kokopop
