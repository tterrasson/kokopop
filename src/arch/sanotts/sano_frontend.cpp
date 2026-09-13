// sanoTTS frontend: the duration model and the two acoustic stages.
//
// Three of the four graphs live here, plus the emotion conditioner and the two
// host heads it drives. They are separate because the pipeline has two hard
// barriers, not because it would be tidier:
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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <ggml.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

// ---------------------------------------------------------------------------
// Host arithmetic for the emotion heads
//
// Everything the style drives outside the shared conditioner is a matrix of a
// few thousand floats against a vector of a few dozen. Kept here rather than
// in a graph: one dispatch costs more than the whole product, and the duration
// half feeds `round(exp(x))`, where a backend's own summation order is exactly
// what must not decide a frame count.
// ---------------------------------------------------------------------------

/// `out[j] = sum_i m[j * n_in + i] * v[i]`.
///
/// Every host matrix here comes from a ggml tensor read verbatim, so `ne[0]`
/// — the input width — is the fast axis, which is the same layout a PyTorch
/// `Linear` stores its `[out, in]` weight in.
void matvec(const std::vector<float> & m, const float * v,
            size_t n_in, size_t n_out, std::vector<float> & out) {
    out.assign(n_out, 0.0f);
    for (size_t j = 0; j < n_out; ++j) {
        const float * row = m.data() + j * n_in;
        float acc = 0.0f;
        for (size_t i = 0; i < n_in; ++i) {
            acc += row[i] * v[i];
        }
        out[j] = acc;
    }
}

float dot(const std::vector<float> & a, const std::vector<float> & b) {
    float acc = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

/// `bound * tanh(x / bound)`, and the identity when the bound is zero.
///
/// A zero bound means the term is switched off entirely — which is what the
/// reference's `max_log_gain == 0` does — not that it is left unbounded.
float bounded(float x, float bound) {
    return bound > 0.0f ? bound * std::tanh(x / bound) : 0.0f;
}

} // namespace

// ---------------------------------------------------------------------------
// Graph 1 — duration
// ---------------------------------------------------------------------------

bool sano_run_duration(SanoArch & arch, const SanoVoice & voice,
                       const std::vector<uint32_t> & ids, float length_scale,
                       const SanoStyle & style,
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

    // The style acts in log space, as a bounded correction of the frozen
    // model's own prediction: one global tempo, plus a redistribution centered
    // under that prediction. A neutral style is skipped rather than computed —
    // every term of both is multiplied by an intensity that is exactly zero.
    if (!style.neutral) {
        std::vector<float> conditioned;
        if (!sano_run_emotion_conditioner(arch, voice, ids, w.vocab, "duration",
                                          conditioned, error)) {
            return false;
        }
        std::vector<float> residual;
        if (!sano_emotion_duration_residual(voice, clamped, style, conditioned,
                                            log_durations, residual, error)) {
            return false;
        }
        for (int64_t t = 0; t < n_tokens; ++t) {
            log_durations[static_cast<size_t>(t)] += residual[static_cast<size_t>(t)];
        }
    }

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
// Graph 1b — the shared conditioner
// ---------------------------------------------------------------------------

bool sano_run_emotion_conditioner(SanoArch & arch, const SanoVoice & voice,
                                  const std::vector<uint32_t> & ids,
                                  uint32_t vocab, const char * label,
                                  std::vector<float> & tokens, std::string & error) {
    const SanoEmotionWeights & w = voice.emo;
    const int64_t n_tokens = static_cast<int64_t>(ids.size());
    if (!w.enabled()) {
        error = "sanoTTS emotion conditioner: the voice carries no emotion pack";
        return false;
    }
    if (n_tokens <= 0) {
        error = "sanoTTS emotion conditioner: empty token sequence";
        return false;
    }

    // The conditioner's own vocabulary spans both native students, so clamping
    // to the branch's is what keeps a token meaning the same phone here as it
    // does in the model this conditioning corrects.
    std::vector<uint32_t> clamped;
    if (!sano::clamp_ids_to_vocab(ids, std::min(vocab, w.cond_vocab), voice.tokens,
                                  label, clamped, error)) {
        return false;
    }

    const SanoGraphBudget budget = sano_emotion_conditioner_budget(voice);
    const size_t bytes = arch.backend->graph_context_bytes(budget.tensors, budget.nodes);
    ggml_context * ctx = sano_graph_context(arch.graph_scratch, bytes, "emotion conditioner", error);
    if (ctx == nullptr) {
        return false;
    }

    arch.backend->set_input_tokens(static_cast<int>(n_tokens));
    arch.backend->set_active_label("sanotts_emotion_conditioner");

    const int64_t hidden = static_cast<int64_t>(w.cond_hidden);
    // One feature: where the token sits in the phrase, in `[0, 1]`. The
    // dilated stack below carries order *within* a neighbourhood; this is what
    // tells a token whether the neighbourhood is at the start or the end.
    FrontendInput in = embed_with_features(ctx, w.cond_embedding, n_tokens, 1);

    ggml_tensor * x = ggml_silu(
        ctx, project(ctx, w.cond_input_proj_w, w.cond_input_proj_b, in.concat));
    x = residual_stack(ctx, x, w.cond_blocks, hidden);

    // Mean, first and last: global content and both phrase boundaries. A mean
    // alone is order-blind and the convolutions alone cannot see past their
    // reach, so a phrase-final token has no way to know that it is one.
    ggml_tensor * time_major = ggml_cont(ctx, ggml_transpose(ctx, x));  // [n_tokens, hidden]
    ggml_tensor * mean = ggml_cont(ctx, ggml_transpose(ctx, ggml_mean(ctx, time_major)));
    ggml_tensor * first = ggml_cont(ctx, ggml_view_2d(ctx, x, hidden, 1, x->nb[1], 0));
    ggml_tensor * last = ggml_cont(
        ctx, ggml_view_2d(ctx, x, hidden, 1, x->nb[1],
                          static_cast<size_t>(n_tokens - 1) * x->nb[1]));

    ggml_tensor * summary = ggml_concat(ctx, ggml_concat(ctx, mean, first, 0), last, 0);
    ggml_tensor * phrase = ggml_silu(
        ctx, project(ctx, w.cond_phrase_w, w.cond_phrase_b, summary));  // [hidden, 1]

    // Broadcast back onto every token, then normalise per token over channels.
    ggml_tensor * out = ggml_add(ctx, x, phrase);
    out = ggml_add(ctx,
                   ggml_mul(ctx, ggml_norm(ctx, out, SANO_EMOTION_NORM_EPS), w.cond_norm_w),
                   w.cond_norm_b);

    // Pinned for the same reason the neutral duration projection is: this
    // feeds a `round(exp(x))` and a 1e-3 drift near a tie is a different frame
    // count, which would make cross-backend parity impossible to define.
    arch.backend->defer_cpu_assignment(out);
    arch.backend->defer_cpu_assignment(out->src[0]);

    ggml_set_name(out, "sanotts_emotion_conditioner");
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, budget.nodes, false);
    ggml_build_forward_expand(graph, out);

    if (!alloc_graph(arch, graph, "emotion conditioner", error)) {
        ggml_free(ctx);
        return false;
    }

    std::vector<float> position(static_cast<size_t>(n_tokens));
    for (int64_t t = 0; t < n_tokens; ++t) {
        position[static_cast<size_t>(t)] = linspace01(t, n_tokens);
    }
    std::vector<int32_t> ids_i32(clamped.begin(), clamped.end());

    arch.backend->tensor_set(in.ids, ids_i32.data(), 0, ggml_nbytes(in.ids));
    arch.backend->tensor_set(in.features, position.data(), 0, ggml_nbytes(in.features));

    if (arch.backend->compute(ctx, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        error = "sanoTTS emotion conditioner graph compute failed";
        return false;
    }

    tokens.resize(static_cast<size_t>(n_tokens * hidden));
    arch.backend->tensor_get(out, tokens.data(), 0, tokens.size() * sizeof(float));
    ggml_free(ctx);
    return true;
}

// ---------------------------------------------------------------------------
// The duration head (host)
// ---------------------------------------------------------------------------

bool sano_emotion_duration_residual(const SanoVoice & voice,
                                    const std::vector<uint32_t> & ids,
                                    const SanoStyle & style,
                                    const std::vector<float> & conditioned,
                                    const std::vector<float> & neutral,
                                    std::vector<float> & residual,
                                    std::string & error) {
    const SanoEmotionWeights & w = voice.emo;
    const size_t n_tokens = ids.size();
    const size_t hidden = w.cond_hidden;

    if (!w.enabled() || style.code.size() != w.style_code) {
        error = "sanoTTS duration head: the voice carries no style of this width";
        return false;
    }
    if (conditioned.size() != n_tokens * hidden || neutral.size() != n_tokens) {
        error = "sanoTTS duration head: the conditioning does not match the tokens";
        return false;
    }

    // `output(tokens * style_local(code))` is a product of two pointwise maps
    // onto one scalar, so the style collapses into a single `[cond_hidden]`
    // vector the whole utterance shares.
    std::vector<float> projection;
    matvec(w.style_local, style.code.data(), w.style_code, hidden, projection);
    for (size_t h = 0; h < hidden; ++h) {
        projection[h] *= w.dur_output_w[h];
    }

    std::vector<float> local(n_tokens, 0.0f);
    std::vector<float> observed(n_tokens, 0.0f);
    for (size_t t = 0; t < n_tokens; ++t) {
        const uint32_t id = ids[t];
        observed[t] = id < w.observed_ids.size() ? w.observed_ids[id] : 0.0f;
        const float * token = conditioned.data() + t * hidden;
        float acc = 0.0f;
        for (size_t h = 0; h < hidden; ++h) {
            acc += token[h] * projection[h];
        }
        // Half the bound: the redistribution is centered below, so a token may
        // move by `local_bound / 2` in each direction and no pair of tokens
        // may trade more than the bound itself.
        local[t] = std::tanh(acc) * (w.local_bound * 0.5f) * observed[t];
    }

    // Centered under the frozen model's own attention — a softmax over its log
    // durations — so what the head redistributes is time, not total length.
    // The tempo above is the only term allowed to change how much there is.
    float peak = 0.0f;
    bool any = false;
    for (size_t t = 0; t < n_tokens; ++t) {
        if (observed[t] != 0.0f && (!any || neutral[t] > peak)) {
            peak = neutral[t];
            any = true;
        }
    }
    float mean = 0.0f;
    if (any) {
        std::vector<float> weights(n_tokens, 0.0f);
        float total = 0.0f;
        for (size_t t = 0; t < n_tokens; ++t) {
            if (observed[t] != 0.0f) {
                weights[t] = std::exp(neutral[t] - peak);
                total += weights[t];
            }
        }
        if (total > 0.0f) {
            for (size_t t = 0; t < n_tokens; ++t) {
                mean += local[t] * (weights[t] / total);
            }
        }
    }

    const float tempo = dot(w.tempo_w, style.code);
    residual.resize(n_tokens);
    for (size_t t = 0; t < n_tokens; ++t) {
        const float centered = (local[t] - mean) * observed[t];
        // What closes the head: the declared bound, so that a style may at
        // most halve or double a token's duration however far the sum ran.
        residual[t] = bounded((tempo + centered) * style.amount, w.max_log_ratio);
    }
    return true;
}

// ---------------------------------------------------------------------------
// The acoustic head (host)
// ---------------------------------------------------------------------------

bool sano_apply_emotion_latent(const SanoVoice & voice,
                               const std::vector<int32_t> & durations,
                               const SanoStyle & style,
                               const std::vector<float> & conditioned,
                               int64_t frames, std::vector<float> & latent,
                               std::string & error) {
    const SanoEmotionWeights & w = voice.emo;
    const size_t n_tokens = durations.size();
    const size_t hidden = w.cond_hidden;
    const size_t channels = w.channels;

    if (!w.enabled() || style.code.size() != w.style_code) {
        error = "sanoTTS acoustic head: the voice carries no style of this width";
        return false;
    }
    if (channels != voice.ac.out_channels ||
        latent.size() != static_cast<size_t>(frames) * channels) {
        error = "sanoTTS acoustic head: the latent does not match the voice";
        return false;
    }
    if (conditioned.size() != n_tokens * hidden) {
        error = "sanoTTS acoustic head: the conditioning does not match the tokens";
        return false;
    }

    // The phrase-global half of the pair, shared by every token.
    std::vector<float> global;
    matvec(w.global_film, style.code.data(), w.style_code, 2 * channels, global);

    // The style's coordinates in the low-rank context basis. A zero bound is
    // the whole contextual term switched off — not a term left unbounded — so
    // nothing about it is evaluated.
    const bool contextual = w.context_bound > 0.0f;
    std::vector<float> rank;
    if (contextual) {
        matvec(w.style_rank, style.code.data(), w.style_code, w.context_rank, rank);
    }

    const bool project = w.donor_projection > 0.0f &&
        std::any_of(w.donor_axis.begin(), w.donor_axis.end(),
                    [](float v) { return v != 0.0f; });

    // Zero-duration positions are transparent here, exactly as they are in the
    // reference: they occupy no frame, pool to nothing, and — this is the part
    // that is easy to get wrong — do not separate the two phones around them,
    // which are neighbours for the crossfade below.
    std::vector<size_t>  voiced;      // token index of each sounded phone
    std::vector<int64_t> lengths;     // its duration, in frames
    std::vector<int64_t> starts;      // its first frame
    voiced.reserve(n_tokens);
    lengths.reserve(n_tokens);
    starts.reserve(n_tokens);
    int64_t frame = 0;
    for (size_t t = 0; t < n_tokens; ++t) {
        const int64_t duration = durations[t];
        if (duration < 0 || frame + duration > frames) {
            error = "sanoTTS acoustic head: the durations do not fill the latent";
            return false;
        }
        if (duration > 0) {
            voiced.push_back(t);
            lengths.push_back(duration);
            starts.push_back(frame);
        }
        frame += duration;
    }
    if (frame != frames) {
        error = "sanoTTS acoustic head: the durations do not fill the latent";
        return false;
    }
    if (voiced.empty()) {
        return true;
    }

    // -- the raw residual, one value per frame and channel -----------------
    //
    // Unbounded on purpose: the crossfade joins raw residuals and the tanh
    // closes on the blend, not on each side of it. Bounding first would let a
    // saturated neighbour drag a boundary somewhere neither phone asked for.
    std::vector<float> raw(latent.size());
    std::vector<float> coefficients, local;
    for (size_t i = 0; i < voiced.size(); ++i) {
        const size_t  t        = voiced[i];
        const int64_t duration = lengths[i];
        const int64_t first    = starts[i];

        if (contextual) {
            matvec(w.context_coeff_w, conditioned.data() + t * hidden, hidden,
                   w.context_rank, coefficients);
            for (size_t k = 0; k < coefficients.size(); ++k) {
                coefficients[k] = (coefficients[k] + w.context_coeff_b[k]) * rank[k];
            }
            matvec(w.context_basis, coefficients.data(), w.context_rank,
                   2 * channels, local);
        }

        for (size_t c = 0; c < channels; ++c) {
            float log_gain = global[c];
            float shift = global[channels + c];
            if (contextual) {
                log_gain += bounded(local[c], w.context_bound);
                shift += bounded(local[channels + c], w.context_bound);
            }
            log_gain = bounded(log_gain, w.max_log_gain);

            const float * column = latent.data() + c * static_cast<size_t>(frames);
            float sum = 0.0f;
            for (int64_t d = 0; d < duration; ++d) {
                sum += column[first + d];
            }
            const float mean = sum / static_cast<float>(duration);

            // What the pair is evaluated on: the phone's mean, plus the share
            // `frame_gain` of each frame's own departure from it. At 0 this is
            // one constant per phone — the retired v1 residual — and at 1 the
            // full native variation inside the phone.
            const float gain = std::expm1(log_gain);
            float * out = raw.data() + c * static_cast<size_t>(frames);
            for (int64_t d = 0; d < duration; ++d) {
                const float content =
                    mean + w.frame_gain * (column[first + d] - mean);
                // Multiplicative on what the base produced, plus an additive
                // term: energy and timbre are scalings, and a purely additive
                // style would be blind to the base's own magnitude.
                out[first + d] = style.amount * (gain * content + shift);
            }
        }
    }

    // -- the crossfade windows, one per frame ------------------------------
    //
    // Frame-wise and channel-independent, so they are resolved once here
    // rather than inside the channel loop below. A window spans at most
    // `transition_frames` frames on each side of a boundary and at most half
    // of either phone it joins, which is what keeps a short phone from being
    // smeared across its neighbours.
    std::vector<int64_t> low(frames, 0), high(frames, 0);
    std::vector<float>   weight(frames, 0.0f);
    std::vector<char>    blend(frames, 0);
    if (w.transition_frames > 0 && voiced.size() > 1) {
        const int64_t reach = static_cast<int64_t>(w.transition_frames);
        for (size_t i = 0; i < voiced.size(); ++i) {
            const int64_t first    = starts[i];
            const int64_t boundary = first + lengths[i];
            const int64_t before =
                i == 0 ? 0
                       : std::min(reach, std::min(lengths[i - 1], lengths[i]) / 2);
            const int64_t after =
                i + 1 == voiced.size()
                    ? 0
                    : std::min(reach, std::min(lengths[i], lengths[i + 1]) / 2);
            for (int64_t f = first; f < boundary; ++f) {
                // A phone shorter than both its windows can be near both
                // boundaries at once; its opening side wins, as in the
                // reference's `where(near_start, ...)`.
                const bool near_start = f - first < before;
                const bool near_end = boundary - f <= after;
                if (!near_start && !near_end) {
                    continue;
                }
                const int64_t width = near_start ? before : after;
                const int64_t edge = near_start ? first : boundary;
                low[f] = std::clamp(edge - width, int64_t{0}, frames - 1);
                high[f] = std::clamp(edge + width - 1, int64_t{0}, frames - 1);
                const float position = static_cast<float>(f - low[f] + 1) /
                                       static_cast<float>(2 * width + 1);
                weight[f] = 0.5f - 0.5f * std::cos(
                    static_cast<float>(M_PI) * position);
                blend[f] = 1;
            }
        }
    }

    // -- bound, project, and add -------------------------------------------
    std::vector<float> delta(channels);
    for (int64_t f = 0; f < frames; ++f) {
        for (size_t c = 0; c < channels; ++c) {
            const float * out = raw.data() + c * static_cast<size_t>(frames);
            const float value =
                blend[f] ? out[low[f]] + weight[f] * (out[high[f]] - out[low[f]])
                         : out[f];
            delta[c] = bounded(value, w.max_delta);
        }

        if (project) {
            // Optionally remove the donors' shared direction from the rendered
            // delta. Off for every pack shipped so far; the pack declares how
            // much of it to remove and carries the axis either way.
            float component = 0.0f;
            for (size_t c = 0; c < channels; ++c) {
                component += delta[c] * w.donor_axis[c];
            }
            component *= w.donor_projection;
            for (size_t c = 0; c < channels; ++c) {
                delta[c] -= component * w.donor_axis[c];
            }
        }

        for (size_t c = 0; c < channels; ++c) {
            latent[c * static_cast<size_t>(frames) + f] += delta[c];
        }
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
    const int64_t n_features = 2;
    FrontendInput in = embed_with_features(ctx, w.embedding, n_tokens, n_features);

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

    std::vector<float> features(static_cast<size_t>(n_tokens * n_features));
    for (int64_t t = 0; t < n_tokens; ++t) {
        float * row = features.data() + static_cast<size_t>(t * n_features);
        row[0] = linspace01(t, n_tokens);
        row[1] = static_cast<float>(
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
    // Taken from the projection rather than recomputed: it is the one place
    // the expansion's width is already known to match the weights.
    const int64_t rows = w.frame_proj_w->ne[0];
    if (rows != hidden + 3 ||
        frame_input.size() != static_cast<size_t>(rows * frames)) {
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
