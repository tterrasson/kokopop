// sanoTTS frontend: the duration model and the two acoustic stages.
//
// Three of the four graphs live here. They are separate because the pipeline
// has two hard barriers, not because it would be tidier:
//
//   1. the number of frames is a *value* the duration model produces, so the
//      acoustic frame stage cannot even be shaped before graph 1 has run;
//   2. `repeat_interleave` by per-token durations has no ggml operator, and
//      the alignment-matrix trick that would emulate it wastes n x frames
//      floats to express a gather.
//
// Layout note: `ne[0]` is the fast dimension. `[hidden, n_tokens]` therefore
// stores one token's channels contiguously, and `[n_tokens, hidden]` stores one
// channel's time series contiguously — which is what the conv operators want.
// The stages transpose between the two exactly where the operator changes.

#include "arch/sanotts/sano_arch.h"
#include "arch/sanotts/sano_graph.h"
#include "backend/backend.h"

#include <cmath>
#include <cstring>
#include <vector>

#include <ggml.h>

namespace kokopop {

namespace {

/// `numpy.round` / `torch.round`: halfway cases go to the even integer.
///
/// Not `std::round` (which rounds halves away from zero) and not
/// `std::nearbyint` (which follows the ambient FP rounding mode, so a caller
/// that changed it would silently change the prosody).
float round_half_to_even(float value) {
    const float floor_value = std::floor(value);
    const float fraction = value - floor_value;
    if (fraction > 0.5f) {
        return floor_value + 1.0f;
    }
    if (fraction < 0.5f) {
        return floor_value;
    }
    return std::fmod(floor_value, 2.0f) == 0.0f ? floor_value : floor_value + 1.0f;
}

/// `linspace(0, 1, n)`, with numpy's degenerate case: a single point is 0.
float linspace01(int64_t index, int64_t n) {
    return n > 1 ? static_cast<float>(index) / static_cast<float>(n - 1) : 0.0f;
}

/// Reserves backend memory for `graph`. Inputs are written after this — the
/// tensors have no buffer until the scheduler has allocated them — and the
/// caller computes once they are filled.
bool alloc_graph(SanoArch & arch, ggml_cgraph * graph,
                 const char * label, std::string & error) {
    arch.backend->sched_reset();
    if (!arch.backend->sched_alloc_graph(graph)) {
        error = std::string("sanoTTS ") + label + " graph allocation failed";
        return false;
    }
    if (!arch.backend->apply_pending_inits()) {
        error = std::string("sanoTTS ") + label + " tensor initialisation failed";
        return false;
    }
    return true;
}

/// `embedding[ids]` concatenated with `n_features` host-filled feature rows,
/// as `[hidden + n_features, n_tokens]`.
struct FrontendInput {
    ggml_tensor * ids = nullptr;       // I32 [n_tokens]
    ggml_tensor * features = nullptr;  // F32 [n_features, n_tokens]
    ggml_tensor * concat = nullptr;
};

FrontendInput embed_with_features(ggml_context * ctx, ggml_tensor * embedding,
                                  int64_t n_tokens, int64_t n_features) {
    FrontendInput in;
    in.ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    in.features = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_features, n_tokens);
    ggml_set_input(in.ids);
    ggml_set_input(in.features);

    ggml_tensor * embedded = ggml_get_rows(ctx, embedding, in.ids);
    in.concat = ggml_concat(ctx, embedded, in.features, 0);
    return in;
}

/// 1x1 conv on a `[C, T]` activation: a plain matmul, no im2col.
ggml_tensor * project(ggml_context * ctx, ggml_tensor * weight,
                      ggml_tensor * bias, ggml_tensor * x) {
    ggml_tensor * out = ggml_mul_mat(ctx, weight, x);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    return ggml_add(ctx, out, bias);
}

/// Runs `blocks` over a `[C, T]` activation, transposing into and out of the
/// `[T, C]` layout the convolutions need.
ggml_tensor * residual_stack(ggml_context * ctx, ggml_tensor * x_ct,
                             const std::vector<SanoResBlock> & blocks,
                             int64_t channels) {
    ggml_tensor * x = ggml_cont(ctx, ggml_transpose(ctx, x_ct));
    for (const auto & block : blocks) {
        x = sano_res_block(ctx, x, block, channels);
    }
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

} // namespace

// ---------------------------------------------------------------------------
// Graph 1 — duration
// ---------------------------------------------------------------------------

bool sano_run_duration(SanoArch & arch, const SanoVoice & voice,
                       const std::vector<uint32_t> & ids, float length_scale,
                       std::vector<int32_t> & durations, std::string & error) {
    const SanoDurationWeights & w = voice.dur;
    const int64_t n_tokens = static_cast<int64_t>(ids.size());
    if (n_tokens <= 0) {
        error = "sanoTTS duration model: empty token sequence";
        return false;
    }
    if (n_tokens > static_cast<int64_t>(w.max_tokens)) {
        error = "sanoTTS duration model: " + std::to_string(n_tokens)
              + " tokens exceeds the voice's limit of " + std::to_string(w.max_tokens);
        return false;
    }
    if (!(length_scale > 0.0f) || !std::isfinite(length_scale)) {
        error = "sanoTTS duration model: length_scale must be finite and positive";
        return false;
    }

    std::vector<uint32_t> clamped;
    if (!sano::clamp_ids_to_vocab(ids, w.vocab, voice.tokens, "duration", clamped, error)) {
        return false;
    }

    const SanoGraphBudget budget = sano_duration_budget(voice);
    const size_t bytes = arch.backend->graph_context_bytes(budget.tensors, budget.nodes);
    ggml_context * ctx = sano_graph_context(arch.graph_scratch, bytes, "duration", error);
    if (ctx == nullptr) {
        return false;
    }

    arch.backend->set_input_tokens(static_cast<int>(n_tokens));
    arch.backend->set_active_label("sanotts_duration");

    const int64_t hidden = static_cast<int64_t>(w.hidden);
    FrontendInput in = embed_with_features(ctx, w.embedding, n_tokens, 3);

    ggml_tensor * x = project(ctx, w.input_proj_w, w.input_proj_b, in.concat);
    x = residual_stack(ctx, x, w.blocks, hidden);

    // The last matmul, the bias and the exp/round that consume it are what
    // decide the frame count. Pin them to the CPU sub-backend: fp16 matmul
    // drift of 1e-3 around a tie flips a duration, which changes the audio
    // length and makes cross-backend parity impossible to even define.
    ggml_tensor * log_duration = project(ctx, w.output_w, w.output_b, x);
    arch.backend->defer_cpu_assignment(log_duration);
    arch.backend->defer_cpu_assignment(log_duration->src[0]);

    ggml_set_name(log_duration, "sanotts_log_duration");
    ggml_set_output(log_duration);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, budget.nodes, false);
    ggml_build_forward_expand(graph, log_duration);

    if (!alloc_graph(arch, graph, "duration", error)) {
        ggml_free(ctx);
        return false;
    }

    std::vector<float> features(static_cast<size_t>(n_tokens) * 3);
    const float length_hint = static_cast<float>(
        std::log1p(static_cast<double>(n_tokens)) /
        std::log1p(static_cast<double>(w.max_tokens)));
    for (int64_t t = 0; t < n_tokens; ++t) {
        features[static_cast<size_t>(t) * 3 + 0] = linspace01(t, n_tokens);
        features[static_cast<size_t>(t) * 3 + 1] = length_hint;
        features[static_cast<size_t>(t) * 3 + 2] = 1.0f;  // valid_hint
    }
    std::vector<int32_t> ids_i32(clamped.begin(), clamped.end());

    arch.backend->tensor_set(in.ids, ids_i32.data(), 0, ggml_nbytes(in.ids));
    arch.backend->tensor_set(in.features, features.data(), 0, ggml_nbytes(in.features));

    if (arch.backend->compute(ctx, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        error = "sanoTTS duration graph compute failed";
        return false;
    }

    std::vector<float> log_durations(static_cast<size_t>(n_tokens));
    arch.backend->tensor_get(log_duration, log_durations.data(), 0,
                             log_durations.size() * sizeof(float));
    ggml_free(ctx);

    durations.resize(static_cast<size_t>(n_tokens));
    for (int64_t t = 0; t < n_tokens; ++t) {
        const float raw = std::exp(log_durations[static_cast<size_t>(t)]);
        float value = round_half_to_even(std::fmax(raw, 1.0f) * length_scale);
        value = std::fmin(std::fmax(value, 1.0f), static_cast<float>(w.max_duration));
        durations[static_cast<size_t>(t)] = static_cast<int32_t>(value);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Graph 2 — acoustic, token stage
// ---------------------------------------------------------------------------

bool sano_run_acoustic_token(SanoArch & arch, const SanoVoice & voice,
                             const std::vector<uint32_t> & ids,
                             const std::vector<int32_t> & durations,
                             std::vector<float> & token_ctx, std::string & error) {
    const SanoAcousticWeights & w = voice.ac;
    const int64_t n_tokens = static_cast<int64_t>(ids.size());
    if (durations.size() != ids.size()) {
        error = "sanoTTS acoustic model: duration and id counts disagree";
        return false;
    }

    std::vector<uint32_t> clamped;
    if (!sano::clamp_ids_to_vocab(ids, w.vocab, voice.tokens, "acoustic", clamped, error)) {
        return false;
    }

    const SanoGraphBudget budget = sano_acoustic_token_budget(voice);
    const size_t bytes = arch.backend->graph_context_bytes(budget.tensors, budget.nodes);
    ggml_context * ctx = sano_graph_context(arch.graph_scratch, bytes, "acoustic token", error);
    if (ctx == nullptr) {
        return false;
    }

    arch.backend->set_input_tokens(static_cast<int>(n_tokens));
    arch.backend->set_active_label("sanotts_acoustic_token");

    const int64_t hidden = static_cast<int64_t>(w.hidden);
    FrontendInput in = embed_with_features(ctx, w.embedding, n_tokens, 2);

    ggml_tensor * x = project(ctx, w.token_proj_w, w.token_proj_b, in.concat);
    x = residual_stack(ctx, x, w.token_blocks, hidden);

    ggml_set_name(x, "sanotts_token_context");
    ggml_set_output(x);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, budget.nodes, false);
    ggml_build_forward_expand(graph, x);

    if (!alloc_graph(arch, graph, "acoustic token", error)) {
        ggml_free(ctx);
        return false;
    }

    double max_duration = 1.0;
    for (const int32_t d : durations) {
        max_duration = std::fmax(max_duration, static_cast<double>(d));
    }
    const double log_max = std::log1p(max_duration);

    std::vector<float> features(static_cast<size_t>(n_tokens) * 2);
    for (int64_t t = 0; t < n_tokens; ++t) {
        features[static_cast<size_t>(t) * 2 + 0] = linspace01(t, n_tokens);
        features[static_cast<size_t>(t) * 2 + 1] = static_cast<float>(
            std::log1p(static_cast<double>(durations[static_cast<size_t>(t)])) / log_max);
    }
    std::vector<int32_t> ids_i32(clamped.begin(), clamped.end());

    arch.backend->tensor_set(in.ids, ids_i32.data(), 0, ggml_nbytes(in.ids));
    arch.backend->tensor_set(in.features, features.data(), 0, ggml_nbytes(in.features));

    if (arch.backend->compute(ctx, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        error = "sanoTTS acoustic token graph compute failed";
        return false;
    }

    token_ctx.resize(static_cast<size_t>(n_tokens * hidden));
    arch.backend->tensor_get(x, token_ctx.data(), 0, token_ctx.size() * sizeof(float));
    ggml_free(ctx);
    return true;
}

// ---------------------------------------------------------------------------
// Token -> frame expansion (host)
// ---------------------------------------------------------------------------

void sano_expand_to_frames(const std::vector<float> & token_ctx, int64_t n_tokens,
                           int64_t hidden, const std::vector<int32_t> & durations,
                           int64_t frames, std::vector<float> & frame_input) {
    const int64_t rows = hidden + 3;
    frame_input.assign(static_cast<size_t>(rows * frames), 0.0f);

    const int64_t token_span = n_tokens > 1 ? n_tokens - 1 : 1;
    int64_t frame = 0;
    for (int64_t token = 0; token < n_tokens; ++token) {
        const int64_t duration = durations[static_cast<size_t>(token)];
        const float * source = token_ctx.data() + static_cast<size_t>(token * hidden);
        const float token_pos = static_cast<float>(token) / static_cast<float>(token_span);

        for (int64_t d = 0; d < duration; ++d, ++frame) {
            float * dest = frame_input.data() + static_cast<size_t>(frame * rows);
            std::memcpy(dest, source, static_cast<size_t>(hidden) * sizeof(float));
            dest[hidden + 0] = linspace01(frame, frames);
            dest[hidden + 1] = token_pos;
            dest[hidden + 2] = duration > 1
                ? static_cast<float>(d) / static_cast<float>(duration - 1)
                : 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Graph 3 — acoustic, frame stage
// ---------------------------------------------------------------------------

bool sano_run_acoustic_frame(SanoArch & arch, const SanoVoice & voice,
                             const std::vector<float> & frame_input,
                             int64_t frames, std::vector<float> & latent,
                             std::string & error) {
    const SanoAcousticWeights & w = voice.ac;
    const int64_t hidden = static_cast<int64_t>(w.hidden);
    const int64_t rows = hidden + 3;
    if (frame_input.size() != static_cast<size_t>(rows * frames)) {
        error = "sanoTTS acoustic model: frame input has the wrong size";
        return false;
    }

    const SanoGraphBudget budget = sano_acoustic_frame_budget(voice);
    const size_t bytes = arch.backend->graph_context_bytes(budget.tensors, budget.nodes);
    ggml_context * ctx = sano_graph_context(arch.graph_scratch, bytes, "acoustic frame", error);
    if (ctx == nullptr) {
        return false;
    }

    arch.backend->set_input_tokens(static_cast<int>(frames));
    arch.backend->set_active_label("sanotts_acoustic_frame");

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, frames);
    ggml_set_input(input);

    ggml_tensor * x = project(ctx, w.frame_proj_w, w.frame_proj_b, input);
    x = residual_stack(ctx, x, w.frame_blocks, hidden);

    ggml_tensor * out = project(ctx, w.output_w, w.output_b, x);
    // [out_channels, frames] -> [frames, out_channels], the layout the decoder
    // convolutions consume.
    out = ggml_cont(ctx, ggml_transpose(ctx, out));
    ggml_set_name(out, "sanotts_latent");
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, budget.nodes, false);
    ggml_build_forward_expand(graph, out);

    if (!alloc_graph(arch, graph, "acoustic frame", error)) {
        ggml_free(ctx);
        return false;
    }

    arch.backend->tensor_set(input, frame_input.data(), 0, ggml_nbytes(input));

    if (arch.backend->compute(ctx, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        error = "sanoTTS acoustic frame graph compute failed";
        return false;
    }

    latent.resize(static_cast<size_t>(frames) * w.out_channels);
    arch.backend->tensor_get(out, latent.data(), 0, latent.size() * sizeof(float));
    ggml_free(ctx);
    return true;
}

} // namespace kokopop
