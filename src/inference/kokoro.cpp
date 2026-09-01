#include "kokoro.h"

#include "constants.h"
#include "inference/diffusion_sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <thread>

#include <ggml.h>

namespace kokopop {

ggml_tensor * require_tensor(Model & model, const char * name, std::string & error) {
    ggml_tensor * t = model.cached_tensor(name);
    if (t == nullptr && error.empty()) {
        error = std::string("missing Kokoro tensor: ") + name;
    }
    return t;
}

namespace {

bool add_required(ggml_tensor * t, std::string & error) {
    return t != nullptr && error.empty();
}

struct AlbertLayerWeights {
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * k_w = nullptr;
    ggml_tensor * k_b = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;
    ggml_tensor * o_w = nullptr;
    ggml_tensor * o_b = nullptr;
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    ggml_tensor * ffn_w = nullptr;
    ggml_tensor * ffn_b = nullptr;
    ggml_tensor * ffn_out_w = nullptr;
    ggml_tensor * ffn_out_b = nullptr;
    ggml_tensor * out_norm_w = nullptr;
    ggml_tensor * out_norm_b = nullptr;
};

bool load_albert_layer(Model & model, AlbertLayerWeights & w, std::string & error) {
    constexpr const char * base = KOKOPOP_PREFIX_ALBERT_LAYER;

    // Single reusable key string — avoids 16 temporary std::string allocations.
    std::string key;
    key.reserve(96);
    const auto t = [&](const char * suffix) -> ggml_tensor * {
        key = base; key += suffix;
        return require_tensor(model, key.c_str(), error);
    };

    w.q_w         = t("attention.query.weight");
    w.q_b         = t("attention.query.bias");
    w.k_w         = t("attention.key.weight");
    w.k_b         = t("attention.key.bias");
    w.v_w         = t("attention.value.weight");
    w.v_b         = t("attention.value.bias");
    w.o_w         = t("attention.dense.weight");
    w.o_b         = t("attention.dense.bias");
    w.attn_norm_w = t("attention.LayerNorm.weight");
    w.attn_norm_b = t("attention.LayerNorm.bias");
    w.ffn_w       = t("ffn.weight");
    w.ffn_b       = t("ffn.bias");
    w.ffn_out_w   = t("ffn_output.weight");
    w.ffn_out_b   = t("ffn_output.bias");
    w.out_norm_w  = t("full_layer_layer_norm.weight");
    w.out_norm_b  = t("full_layer_layer_norm.bias");

    return error.empty();
}

int64_t voice_style_row(ggml_tensor * voice, int64_t n_tokens, int64_t style_len) {
    if (voice == nullptr || voice->ne[1] <= 0) {
        return 0;
    }
    const int64_t row_key = style_len > 0 ? style_len - 1 : n_tokens - 3;
    return std::clamp<int64_t>(row_key, 0, voice->ne[1] - 1);
}

bool all_finite(const std::vector<float> & values, const char * label, std::string & error) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            error = std::string("non-finite Kokoro ") + label + " value at index " + std::to_string(i);
            return false;
        }
    }
    return true;
}

void fill_i32_tokens(std::vector<int32_t> & out, const std::vector<uint32_t> & ids) {
    out.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        out[i] = static_cast<int32_t>(ids[i]);
    }
}

void fill_i32_positions(std::vector<int32_t> & out, int64_t n_tokens) {
    out.resize(static_cast<size_t>(n_tokens));
    for (int64_t i = 0; i < n_tokens; ++i) {
        out[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
}

bool build_duration_mask(
    const std::vector<int> & durations,
    int64_t total_frames,
    int64_t n_tokens,
    std::vector<float> & mask_data,
    std::string & error) {

    const size_t mask_size = static_cast<size_t>(total_frames * n_tokens);
    if (mask_data.size() < mask_size) {
        mask_data.resize(mask_size);
    }

    std::fill(mask_data.begin(), mask_data.begin() + static_cast<ptrdiff_t>(mask_size), 0.0f);

    int64_t running = 0;
    for (int64_t token = 0; token < n_tokens; ++token) {
        const int64_t token_dur = durations[static_cast<size_t>(token)];
        const size_t row_offset = static_cast<size_t>(token * total_frames);

        for (int64_t frame = 0; frame < token_dur; ++frame) {
            const size_t idx = static_cast<size_t>(row_offset + running + frame);
            if (idx >= mask_size) {
                error = "duration mask index out of bounds";
                return false;
            }
            mask_data[idx] = 1.0f;
        }

        running += token_dur;
    }

    if (running != total_frames) {
        error = "duration mask total frame mismatch";
        return false;
    }

    return true;
}

bool copy_voice_style_row(
    const Model & model,
    const std::string & voice_name,
    ggml_tensor * voice,
    int64_t n_tokens,
    int64_t style_len,
    std::vector<float> & out,
    std::string & error) {
    const auto it = model.voices.find(voice_name);
    if (it == model.voices.end()) {
        error = "missing Kokoro voice data for style sampling";
        return false;
    }
    const VoiceInfo & info = it->second;
    if (info.cols < 256 || info.rows <= 0 || info.data.size() < static_cast<size_t>(info.rows * info.cols)) {
        error = "invalid Kokoro voice tensor shape";
        return false;
    }
    const int64_t row = std::min<int64_t>(voice_style_row(voice, n_tokens, style_len), info.rows - 1);
    out.assign(
        info.data.begin() + static_cast<ptrdiff_t>(row * info.cols),
        info.data.begin() + static_cast<ptrdiff_t>(row * info.cols + 256));
    return true;
}

} // anonymous namespace

int duration_to_frames(float value) {
    if (!std::isfinite(value)) {
        return 1;
    }
    return std::clamp(static_cast<int>(std::lrint(value)), 1, 50);
}

int duration_to_frames(float value, float speed) {
    if (!std::isfinite(speed) || speed <= 0.0f) {
        speed = 1.0f;
    }
    return duration_to_frames(value / speed);
}

// ---------------------------------------------------------------------------
// Frontend probe
// ---------------------------------------------------------------------------

bool run_kokoro_frontend_probe(
    Model & model,
    const std::vector<uint32_t> & ids,
    const std::string & requested_voice,
    KokoroFrontendProbe & probe,
    std::string & error,
    int64_t style_len,
    const KokoroDiffusionOptions * diffusion) {

    if (ids.empty()) {
        error = "cannot run Kokoro frontend on empty token sequence";
        return false;
    }

    ggml_tensor * word       = require_tensor(model, "kokopop.albert.embeddings.word_embeddings.weight", error);
    ggml_tensor * pos        = require_tensor(model, "kokopop.albert.embeddings.position_embeddings.weight", error);
    ggml_tensor * type       = require_tensor(model, "kokopop.albert.embeddings.token_type_embeddings.weight", error);
    ggml_tensor * norm_w     = require_tensor(model, "kokopop.albert.embeddings.LayerNorm.weight", error);
    ggml_tensor * norm_b     = require_tensor(model, "kokopop.albert.embeddings.LayerNorm.bias", error);
    ggml_tensor * map_w      = require_tensor(model, "kokopop.albert.encoder.embedding_hidden_mapping_in.weight", error);
    ggml_tensor * map_b      = require_tensor(model, "kokopop.albert.encoder.embedding_hidden_mapping_in.bias", error);
    ggml_tensor * enc_w      = require_tensor(model, "kokopop.bert_encoder.weight", error);
    ggml_tensor * enc_b      = require_tensor(model, "kokopop.bert_encoder.bias", error);
    ggml_tensor * duration_w = require_tensor(model, "kokopop.predictor.duration_proj.linear_layer.weight", error);
    ggml_tensor * duration_b = require_tensor(model, "kokopop.predictor.duration_proj.linear_layer.bias", error);

    AlbertLayerWeights layer;
    load_albert_layer(model, layer, error);

    if (!add_required(word, error) ||
        !add_required(pos, error) ||
        !add_required(type, error) ||
        !add_required(norm_w, error) ||
        !add_required(norm_b, error) ||
        !add_required(map_w, error) ||
        !add_required(map_b, error) ||
        !add_required(enc_w, error) ||
        !add_required(enc_b, error) ||
        !add_required(duration_w, error) ||
        !add_required(duration_b, error)) {
        return false;
    }

    const std::string voice_name = resolve_voice_name(requested_voice, model.voices);
    ggml_tensor * voice = !voice_name.empty() ? model.cached_tensor("kokopop.voice." + voice_name) : nullptr;
    if (voice == nullptr) {
        error = "missing Kokoro voice tensor for duration graph";
        return false;
    }
    if (!copy_voice_style_row(model, voice_name, voice, static_cast<int64_t>(ids.size()), style_len, probe.style, error)) {
        return false;
    }

    const size_t mem_size = model.backend->frontend_context_bytes();
    model.backend->clear_pending_inits();

    ggml_context * ctx = init_scratch_context(model, model.frontend_scratch, mem_size, true, "frontend", error);
    if (ctx == nullptr) {
        return false;
    }

    const int64_t n_tokens = static_cast<int64_t>(ids.size());
    model.backend->set_input_tokens(static_cast<int>(n_tokens));
    model.backend->set_active_label("frontend");

    ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * pos_ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * style_in  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_set_input(token_ids);
    ggml_set_input(pos_ids);
    ggml_set_input(style_in);

    // Snapshot the last leaf tensor before any frontend compute op. After
    // building the graph, every op created since this snapshot will be pinned
    // to the CPU sub-backend on GPU backends with fp16 drift (MoltenVK). The
    // entire frontend (Albert encoder + bert_encoder + duration_encoder +
    // predictor.lstm) runs on CPU, making the output (probe.hidden,
    // probe.durations) bit-identical with the pure-CPU path. The
    // generation/generator graphs remain on GPU.
    ggml_tensor * pin_snapshot = nullptr;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        pin_snapshot = t;
    }

    ggml_tensor * cur = ggml_add(ctx, ggml_get_rows(ctx, word, token_ids), ggml_get_rows(ctx, pos, pos_ids));

    ggml_tensor * type_zero = ggml_view_2d(ctx, type, type->ne[0], 1, type->nb[1], 0);
    cur = ggml_add(ctx, cur, ggml_repeat(ctx, type_zero, cur));
    cur = layer_norm(ctx, cur, norm_w, norm_b, 1e-12f);
    cur = linear(ctx, map_w, map_b, cur);

    constexpr int64_t hidden_size = KOKOPOP_HIDDEN_SIZE;
    constexpr int64_t n_heads     = KOKOPOP_NUM_HEADS;
    constexpr int64_t head_size   = KOKOPOP_HEAD_SIZE;
    constexpr float attn_scale    = KOKOPOP_ATTN_SCALE;

    for (int r = 0; r < 12; ++r) {
        ggml_tensor * residual = cur;

        ggml_tensor * q = linear(ctx, layer.q_w, layer.q_b, cur);
        ggml_tensor * k = linear(ctx, layer.k_w, layer.k_b, cur);
        ggml_tensor * v = linear(ctx, layer.v_w, layer.v_b, cur);

        q = ggml_reshape_3d(ctx, q, head_size, n_heads, n_tokens);
        k = ggml_reshape_3d(ctx, k, head_size, n_heads, n_tokens);

        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

        ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
        // Force fp32 accumulation; see graph_ops::linear comment.
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_soft_max_ext(ctx, scores, nullptr, attn_scale, 0.0f);

        v = ggml_cont_3d(ctx, ggml_transpose(ctx, v), n_tokens, head_size, n_heads);

        ggml_tensor * attended = ggml_mul_mat(ctx, scores, v);
        ggml_mul_mat_set_prec(attended, GGML_PREC_F32);
        attended = ggml_permute(ctx, attended, 2, 0, 1, 3);
        attended = ggml_cont_2d(ctx, attended, hidden_size, n_tokens);
        attended = linear(ctx, layer.o_w, layer.o_b, attended);

        cur = layer_norm(ctx, ggml_add(ctx, attended, residual), layer.attn_norm_w, layer.attn_norm_b, 1e-12f);
        residual = cur;

        cur = ggml_gelu(ctx, linear(ctx, layer.ffn_w, layer.ffn_b, cur));
        cur = linear(ctx, layer.ffn_out_w, layer.ffn_out_b, cur);
        cur = layer_norm(ctx, ggml_add(ctx, cur, residual), layer.out_norm_w, layer.out_norm_b, 1e-12f);
    }

    ggml_tensor * bert_embedding = nullptr;
    const bool diffusion_requested =
        diffusion != nullptr && diffusion->enabled && diffusion->steps > 0 &&
        (diffusion->alpha != 0.0f || diffusion->beta != 0.0f);
    if (diffusion_requested) {
        bert_embedding = ggml_cont(ctx, cur);
    }

    cur = linear(ctx, enc_w, enc_b, cur);

    ggml_tensor * style = style_in;

    model.lstm_custom_params.clear();
    model.lstm_custom_params.reserve(24);

    ggml_tensor * d = duration_encoder(ctx, model, cur, style, n_tokens, error);
    if (d == nullptr) {
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * dur_hidden = bidirectional_lstm(ctx, model, d, "kokopop.predictor.lstm", n_tokens, error);
    if (dur_hidden == nullptr) {
        ggml_free(ctx);
        return false;
    }

    // Walk every op created since pin_snapshot (covers the whole frontend:
    // Albert encoder + bert_encoder + duration_encoder + predictor.lstm) and
    // pin each compute op to the CPU sub-backend. No-op on backends without
    // fp16 drift. Skip pure metadata ops (view/reshape/permute/transpose) —
    // they share storage with view_src and ggml-sched can't relocate them.
    {
        ggml_tensor * first_new = pin_snapshot != nullptr
            ? ggml_get_next_tensor(ctx, pin_snapshot)
            : ggml_get_first_tensor(ctx);
        // Pin "amplifier" ops to the CPU sub-backend on GPU backends with fp16
        // drift (MoltenVK, Adreno OpenCL). Layer norm divides by sqrt(variance)
        // and explodes for near-zero variance, magnifying tiny fp16 noise into
        // audible deformation in the decoder (observed for af_bella). Softmax
        // can drift independently but does not amplify, so it stays on GPU by
        // default — set KOKOPOP_PIN_STRICT=1 to pin it too for bit-identical
        // outputs across backends (slower, ~10x more ops on CPU).
        // KOKOPOP_VULKAN_PIN_STRICT is the former, Vulkan-only name; it is
        // still honoured now that the setting governs more than one backend.
        static const bool strict_pin =
            std::getenv("KOKOPOP_PIN_STRICT") != nullptr ||
            std::getenv("KOKOPOP_VULKAN_PIN_STRICT") != nullptr;
        for (ggml_tensor * t = first_new; t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            switch (t->op) {
                case GGML_OP_NORM:
                case GGML_OP_RMS_NORM:
                    model.backend->defer_cpu_assignment(t);
                    break;
                case GGML_OP_SOFT_MAX:
                    if (strict_pin) {
                        model.backend->defer_cpu_assignment(t);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    // Stop the graph at dur_hidden and run the final duration projection +
    // sigmoid + sum_rows on host. The GPU matmul + sum_rows(sigmoid(logits))
    // is off by ~1e-3 vs CPU on Vulkan/MoltenVK, and duration_to_frames()
    // rounds via std::lrint — when a token's predicted duration sits near
    // k+0.5, the backends round to different ints, shifting total_frames by
    // a few frames and desynchronising the entire downstream pipeline
    // (audible as "deformed voice" with timing drift). Doing the projection
    // on host (with duration_w/_b dequantized once and cached on Model)
    // makes the result bit-identical across backends.

    // Materialize both host-read outputs:
    //   d           — duration encoder output, used as conditioning input by
    //                  the generation graph (KokoroFrontendProbe::hidden)
    //   dur_hidden  — predictor.lstm output, multiplied by duration_w on host
    //                  to produce duration logits with bit-identical rounding
    // dur_hidden must be a graph output (not just an intermediate) so the
    // predictor.lstm stays anchored in the forward set and its result is
    // readable on host after graph compute.
    d          = ggml_cont(ctx, d);
    dur_hidden = ggml_cont(ctx, dur_hidden);

    ggml_set_name(d,          "kokopop_duration_hidden_probe");
    ggml_set_name(dur_hidden, "kokopop_predictor_lstm_out");
    ggml_set_output(d);
    ggml_set_output(dur_hidden);
    if (bert_embedding != nullptr) {
        ggml_set_name(bert_embedding, "kokopop_diffusion_bert_embedding");
        ggml_set_output(bert_embedding);
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, frontend_graph_size(n_tokens), false);
    if (bert_embedding != nullptr) {
        ggml_build_forward_expand(gf, bert_embedding);
    }
    ggml_build_forward_expand(gf, d);
    ggml_build_forward_expand(gf, dur_hidden);

    model.backend->sched_reset();
    if (!model.backend->sched_alloc_graph(gf)) {
        ggml_free(ctx);
        error = "ggml frontend backend allocation failed";
        return false;
    }

    if (!model.backend->apply_pending_inits()) {
        ggml_free(ctx);
        error = "ggml frontend backend tensor initialization failed";
        return false;
    }

    fill_i32_tokens(model.tmp_ids_i32, ids);
    fill_i32_positions(model.tmp_pos_i32, n_tokens);

    model.backend->tensor_set(token_ids, model.tmp_ids_i32.data(), 0, ggml_nbytes(token_ids));
    model.backend->tensor_set(pos_ids,   model.tmp_pos_i32.data(), 0, ggml_nbytes(pos_ids));
    model.backend->tensor_set(style_in,  probe.style.data() + 128, 0, ggml_nbytes(style_in));

    const ggml_status status = model.backend->compute(ctx, gf);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        error = "ggml frontend graph compute failed";
        return false;
    }

    const int64_t n_hidden    = ggml_nelements(d);
    const int64_t hidden_dim  = d->ne[0];
    const int64_t pred_dim    = dur_hidden->ne[0];
    const int64_t n_dur_hidden = ggml_nelements(dur_hidden);

    probe.hidden.resize(static_cast<size_t>(n_hidden));
    probe.durations.resize(static_cast<size_t>(n_tokens));
    probe.n_tokens   = n_tokens;
    probe.hidden_dim = hidden_dim;
    if (bert_embedding != nullptr) {
        probe.embedding_dim = bert_embedding->ne[0];
        probe.bert_embedding.resize(static_cast<size_t>(ggml_nelements(bert_embedding)));
        model.backend->tensor_get(
            bert_embedding, probe.bert_embedding.data(), 0,
            probe.bert_embedding.size() * sizeof(float));
    }

    model.backend->tensor_get(d, probe.hidden.data(), 0, static_cast<size_t>(n_hidden) * sizeof(float));

    std::vector<float> dur_hidden_host(static_cast<size_t>(n_dur_hidden));
    model.backend->tensor_get(dur_hidden, dur_hidden_host.data(), 0, dur_hidden_host.size() * sizeof(float));

    // Lazily dequantize duration_w / duration_b once. ggml mul_mat convention:
    // W shape [K, N], X shape [K, M] → output [N, M], with W stored col-major
    // such that W_f32[k + K*n] is the weight from input k to output n.
    if (model.duration_proj_w_f32.empty()) {
        if (!tensor_to_f32(*model.backend, duration_w, model.duration_proj_w_f32) ||
            !tensor_to_f32(*model.backend, duration_b, model.duration_proj_b_f32)) {
            ggml_free(ctx);
            error = "failed to dequantize duration projection weights";
            return false;
        }
        model.duration_proj_in  = duration_w->ne[0];
        model.duration_proj_out = duration_w->ne[1];
    }
    GGML_ASSERT(model.duration_proj_in == pred_dim &&
                "duration_proj input dim mismatch vs predictor.lstm output");

    const int64_t n_buckets = model.duration_proj_out;
    const float * W = model.duration_proj_w_f32.data();
    const float * B = model.duration_proj_b_f32.data();

    // pred_dur[t] = sum_n sigmoid(bias[n] + sum_k W[k + K*n] * dur_hidden[k + K*t])
    // Computed on host so duration rounding is bit-identical across backends.
    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * h = dur_hidden_host.data() + t * pred_dim;
        double sum = 0.0;
        for (int64_t n = 0; n < n_buckets; ++n) {
            const float * w_row = W + static_cast<size_t>(pred_dim) * static_cast<size_t>(n);
            double acc = static_cast<double>(B[n]);
            for (int64_t k = 0; k < pred_dim; ++k) {
                acc += static_cast<double>(w_row[k]) * static_cast<double>(h[k]);
            }
            const float x = static_cast<float>(acc);
            sum += (x >= 0.0f)
                ? 1.0 / (1.0 + std::exp(-static_cast<double>(x)))
                : std::exp(static_cast<double>(x)) / (1.0 + std::exp(static_cast<double>(x)));
        }
        probe.durations[static_cast<size_t>(t)] = static_cast<float>(sum);
    }

    if (!apply_diffusion_style_options(model, diffusion, probe.style,
                                       probe.bert_embedding, probe.n_tokens,
                                       probe.embedding_dim, error)) {
        ggml_free(ctx);
        return false;
    }

    if (const char * dump_dir = std::getenv("KOKOPOP_DUMP_DIR")) {
        auto dump = [&](const char * name, const float * data, size_t n) {
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/front_%s_n%lld.bin",
                          dump_dir, name, static_cast<long long>(n_tokens));
            if (FILE * f = std::fopen(path, "wb")) {
                std::fwrite(data, sizeof(float), n, f);
                std::fclose(f);
            }
        };
        dump("hidden",      probe.hidden.data(),    probe.hidden.size());
        dump("durations",   probe.durations.data(), probe.durations.size());
    }

    ggml_free(ctx);
    return true;
}

// ---------------------------------------------------------------------------
// Generation probe
// ---------------------------------------------------------------------------

bool run_kokoro_generation_probe(
    Model & model,
    const std::vector<uint32_t> & ids,
    const std::string & requested_voice,
    float speed,
    const KokoroFrontendProbe & frontend,
    KokoroGenerationProbe & probe,
    std::string & error,
    int64_t style_len) {

    if (ids.empty() || frontend.n_tokens != static_cast<int64_t>(ids.size())) {
        error = "invalid Kokoro generation inputs";
        return false;
    }

    if (frontend.hidden_dim <= 0 ||
        frontend.hidden.empty() ||
        frontend.hidden.size() != static_cast<size_t>(frontend.hidden_dim * frontend.n_tokens)) {
        error = "missing Kokoro duration hidden states";
        return false;
    }

    std::vector<int> durations;
    durations.reserve(frontend.durations.size());

    int64_t total_frames = 0;
    for (float value : frontend.durations) {
        const int d = duration_to_frames(value, speed);
        durations.push_back(d);
        total_frames += d;
    }

    if (durations.size() != static_cast<size_t>(frontend.n_tokens) || total_frames <= 0) {
        error = "invalid Kokoro predicted durations";
        return false;
    }

    if (total_frames > 8192) {
        std::fprintf(
            stderr,
            "[kokopop] WARNING: large chunk (%lld frames, ~%.1fs audio), may use significant RAM\n",
            static_cast<long long>(total_frames),
            static_cast<double>(total_frames) * 300.0 / static_cast<double>(KOKOPOP_SAMPLE_RATE));
    }

    ggml_tensor * shared_w = model.cached_tensor("kokopop.predictor.shared.weight_ih_l0.0");
    if (shared_w == nullptr) {
        shared_w = require_tensor(model, "kokopop.predictor.shared.weight_ih_l0", error);
    }

    ggml_tensor * f0_proj_w = require_tensor(model, "kokopop.predictor.F0_proj.weight", error);
    ggml_tensor * f0_proj_b = require_tensor(model, "kokopop.predictor.F0_proj.bias", error);
    ggml_tensor * n_proj_w  = require_tensor(model, "kokopop.predictor.N_proj.weight", error);
    ggml_tensor * n_proj_b  = require_tensor(model, "kokopop.predictor.N_proj.bias", error);

    if (!add_required(shared_w, error) ||
        !add_required(f0_proj_w, error) ||
        !add_required(f0_proj_b, error) ||
        !add_required(n_proj_w, error) ||
        !add_required(n_proj_b, error)) {
        return false;
    }

    const std::string voice_name = resolve_voice_name(requested_voice, model.voices);
    ggml_tensor * voice = !voice_name.empty() ? model.cached_tensor("kokopop.voice." + voice_name) : nullptr;
    if (voice == nullptr) {
        error = "missing Kokoro voice tensor for generation graph";
        return false;
    }
    std::vector<float> fallback_style;
    const std::vector<float> * style_data = &frontend.style;
    if (style_data->size() < 256) {
        if (!copy_voice_style_row(model, voice_name, voice, frontend.n_tokens, style_len, fallback_style, error)) {
            return false;
        }
        style_data = &fallback_style;
    }

    const size_t mem_size = model.backend->generation_context_bytes(total_frames, frontend.n_tokens);
    model.backend->clear_pending_inits();

    ggml_context * ctx = init_scratch_context(model, model.generation_scratch, mem_size, true, "generation", error);
    if (ctx == nullptr) {
        return false;
    }
    const bool direct_conv = model.backend != nullptr && model.backend->prefers_direct_conv();

    const int64_t n_tokens = frontend.n_tokens;
    model.backend->set_input_tokens(static_cast<int>(n_tokens));
    model.backend->set_active_label("generation");

    model.lstm_custom_params.clear();
    model.lstm_custom_params.reserve(24);

    ggml_tensor * token_ids     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * duration_pred = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frontend.hidden_dim, n_tokens);
    ggml_tensor * duration_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, total_frames, n_tokens);
    ggml_tensor * prosody_style_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_tensor * decoder_style_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);

    ggml_set_input(token_ids);
    ggml_set_input(duration_pred);
    ggml_set_input(duration_mask);
    ggml_set_input(prosody_style_in);
    ggml_set_input(decoder_style_in);

    // Snapshot the last leaf tensor before any generation compute op. After
    // building the graph, every op created since this snapshot will be pinned
    // to the CPU sub-backend on backends with fp16 drift (MoltenVK). This
    // covers the predictor.shared LSTM + F0/N predictors + text_encoder +
    // decoder chain. The generator (vocoder) graph remains on GPU.
    ggml_tensor * pin_snapshot = nullptr;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        pin_snapshot = t;
    }

    if (!build_duration_mask(durations, total_frames, n_tokens, model.tmp_mask_f32, error)) {
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * prosody_style = prosody_style_in;

    ggml_tensor * cur = ggml_mul_mat(
        ctx,
        ggml_cont(ctx, ggml_transpose(ctx, duration_mask)),
        ggml_cont(ctx, ggml_transpose(ctx, duration_pred)));
    ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

    cur = bidirectional_lstm(ctx, model, cur, "kokopop.predictor.shared", total_frames, error);
    if (cur == nullptr) {
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * f0_curve = ggml_cont(ctx, ggml_transpose(ctx, cur));
    for (int i = 0; i < 3; ++i) {
        f0_curve = adain_resblk1d(ctx, model, f0_curve, prosody_style, "kokopop.predictor.F0." + std::to_string(i), i == 1, error);
        if (f0_curve == nullptr) {
            ggml_free(ctx);
            return false;
        }
        // Same anchor as for the decoder AdaIN chain — prevents scheduler
        // aliasing of intermediate buffers on Vulkan/MoltenVK.
        ggml_set_output(f0_curve);
    }
    f0_curve = ggml_cont(ctx, ggml_transpose(ctx, f0_curve));
    f0_curve = add_channel_bias(ctx, conv1d(ctx, f0_proj_w, ggml_cont(ctx, ggml_transpose(ctx, f0_curve)), 1, 0, 1, 1), f0_proj_b);
    f0_curve = ggml_cont(ctx, ggml_view_1d(ctx, f0_curve, f0_curve->ne[0], 0));

    ggml_tensor * n_curve = ggml_cont(ctx, ggml_transpose(ctx, cur));
    for (int i = 0; i < 3; ++i) {
        n_curve = adain_resblk1d(ctx, model, n_curve, prosody_style, "kokopop.predictor.N." + std::to_string(i), i == 1, error);
        if (n_curve == nullptr) {
            ggml_free(ctx);
            return false;
        }
        ggml_set_output(n_curve);
    }
    n_curve = ggml_cont(ctx, ggml_transpose(ctx, n_curve));
    n_curve = add_channel_bias(ctx, conv1d(ctx, n_proj_w, ggml_cont(ctx, ggml_transpose(ctx, n_curve)), 1, 0, 1, 1), n_proj_b);
    n_curve = ggml_cont(ctx, ggml_view_1d(ctx, n_curve, n_curve->ne[0], 0));

    ggml_tensor * asr = text_encoder(ctx, model, token_ids, duration_mask, n_tokens, error);
    if (asr == nullptr) {
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * decoder_style = decoder_style_in;

    ggml_tensor * f0_conv_w = require_tensor(model, "kokopop.decoder.F0_conv.weight", error);
    ggml_tensor * f0_conv_b = require_tensor(model, "kokopop.decoder.F0_conv.bias", error);
    ggml_tensor * n_conv_w  = require_tensor(model, "kokopop.decoder.N_conv.weight", error);
    ggml_tensor * n_conv_b  = require_tensor(model, "kokopop.decoder.N_conv.bias", error);
    ggml_tensor * asr_res_w = require_tensor(model, "kokopop.decoder.asr_res.0.weight", error);
    ggml_tensor * asr_res_b = require_tensor(model, "kokopop.decoder.asr_res.0.bias", error);
    if (!error.empty()) {
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * f0_dec_in = ggml_reshape_2d(ctx, f0_curve, f0_curve->ne[0], 1);
    ggml_tensor * n_dec_in  = ggml_reshape_2d(ctx, n_curve,  n_curve->ne[0],  1);

    ggml_tensor * f0_dec = add_channel_bias(ctx, conv1d(ctx, f0_conv_w, f0_dec_in, 2, 1, 1, 3), f0_conv_b);
    ggml_tensor * n_dec  = add_channel_bias(ctx, conv1d(ctx, n_conv_w,  n_dec_in,  2, 1, 1, 3), n_conv_b);

    ggml_tensor * decoder_cur = ggml_concat(
        ctx,
        ggml_concat(ctx, ggml_cont(ctx, ggml_transpose(ctx, asr)), f0_dec, 1),
        n_dec,
        1);

    decoder_cur = adain_resblk1d(ctx, model, decoder_cur, decoder_style, "kokopop.decoder.encode", false, error);
    if (decoder_cur == nullptr) {
        ggml_free(ctx);
        return false;
    }
    // Anchor each AdaIN-block output as a graph output. Without this, the
    // Vulkan/MoltenVK scheduler aliases intermediate block-output buffers in a
    // way that produces audibly deformed audio on longer inputs (metallic /
    // phased — bug 3 in VULKAN.md). The set_output flag does not force a
    // materialised copy, only prevents reuse of the buffer; cost is negligible.
    ggml_set_output(decoder_cur);

    ggml_tensor * asr_res = add_channel_bias(ctx, conv1d(ctx, asr_res_w, ggml_cont(ctx, ggml_transpose(ctx, asr)), 1, 0, 1, 1, direct_conv), asr_res_b);

    for (int i = 0; i < 4; ++i) {
        decoder_cur = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, decoder_cur, asr_res, 1), f0_dec, 1), n_dec, 1);
        decoder_cur = adain_resblk1d(ctx, model, decoder_cur, decoder_style, "kokopop.decoder.decode." + std::to_string(i), i == 3, error);
        if (decoder_cur == nullptr) {
            ggml_free(ctx);
            return false;
        }
        ggml_set_output(decoder_cur);
    }

    // Make all host-read outputs explicit and materialized.
    f0_curve    = ggml_cont(ctx, f0_curve);
    n_curve     = ggml_cont(ctx, n_curve);
    asr         = ggml_cont(ctx, asr);
    decoder_cur = ggml_cont(ctx, decoder_cur);

    ggml_set_name(f0_curve, "kokopop_f0_probe");
    ggml_set_name(n_curve, "kokopop_noise_probe");
    ggml_set_name(asr, "kokopop_asr_probe");
    ggml_set_name(decoder_cur, "kokopop_decoder_probe");
    ggml_set_name(decoder_style, "kokopop_decoder_style_probe");

    ggml_set_output(f0_curve);
    ggml_set_output(n_curve);
    ggml_set_output(asr);
    ggml_set_output(decoder_cur);
    // decoder_style is read back via tensor_get after compute (feeds the CPU
    // vocoder). Without ggml_set_output, the scheduler allocator may alias its
    // buffer once the graph stops referencing it, and the read-back returns
    // overwritten bytes. Symptom on Vulkan: style values ~20× too large,
    // saturating the vocoder output to ±1.0 (white noise). CPU happened to
    // dodge the aliasing.
    ggml_set_output(decoder_style);

    const bool gen_profile = std::getenv("KOKOPOP_GEN_PROFILE") != nullptr;
    int64_t t_build = 0;
    int64_t t_alloc = 0;
    int64_t t_compute = 0;

    int64_t t0 = gen_profile ? ggml_time_us() : 0;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, generation_graph_size(total_frames, n_tokens), false);
    ggml_build_forward_expand(gf, f0_curve);
    ggml_build_forward_expand(gf, n_curve);
    ggml_build_forward_expand(gf, asr);
    ggml_build_forward_expand(gf, decoder_cur);

    if (gen_profile) {
        t_build = ggml_time_us() - t0;
        std::fprintf(stderr, "[gen-profile] n_nodes=%d build=%.1fms\n", ggml_graph_n_nodes(gf), t_build / 1000.0);
    }

    // Pin every compute op of the generation graph to the CPU sub-backend on
    // GPU backends with fp16 drift. Skip metadata ops (view/reshape/permute/
    // transpose) and leaves — sched can't relocate views and these have no
    // numerical impact. The vocoder/generator graph downstream remains on GPU.
    {
        ggml_tensor * first_new = pin_snapshot != nullptr
            ? ggml_get_next_tensor(ctx, pin_snapshot)
            : ggml_get_first_tensor(ctx);
        for (ggml_tensor * t = first_new; t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            // Only pin "amplifier" ops where small fp16 drift in the input
            // gets magnified — layer norm divides by sqrt(variance) which
            // explodes for near-zero variance, and softmax can saturate.
            // Linear ops (MUL_MAT/ADD/MUL) are stable; MUL_MAT already has
            // fp32 accumulation forced via mul_mat_set_prec. Keep them on GPU.
            switch (t->op) {
                case GGML_OP_NORM:
                case GGML_OP_RMS_NORM:
                    model.backend->defer_cpu_assignment(t);
                    break;
                default:
                    break;
            }
        }
    }

    model.backend->sched_reset();

    t0 = gen_profile ? ggml_time_us() : 0;
    if (!model.backend->sched_alloc_graph(gf)) {
        ggml_free(ctx);
        error = "ggml generation backend allocation failed";
        return false;
    }

    if (gen_profile) {
        t_alloc = ggml_time_us() - t0;
        std::fprintf(stderr, "[gen-profile] sched_alloc=%.1fms\n", t_alloc / 1000.0);
    }

    if (!model.backend->apply_pending_inits()) {
        ggml_free(ctx);
        error = "ggml generation backend tensor initialization failed";
        return false;
    }

    fill_i32_tokens(model.tmp_ids_i32, ids);

    model.backend->tensor_set(token_ids,     model.tmp_ids_i32.data(), 0, ggml_nbytes(token_ids));
    model.backend->tensor_set(duration_pred, frontend.hidden.data(),   0, ggml_nbytes(duration_pred));
    model.backend->tensor_set(duration_mask, model.tmp_mask_f32.data(), 0, ggml_nbytes(duration_mask));
    model.backend->tensor_set(decoder_style_in, style_data->data(),       0, ggml_nbytes(decoder_style_in));
    model.backend->tensor_set(prosody_style_in, style_data->data() + 128, 0, ggml_nbytes(prosody_style_in));

    t0 = gen_profile ? ggml_time_us() : 0;
    const ggml_status status = model.backend->compute(ctx, gf);

    if (gen_profile) {
        t_compute = ggml_time_us() - t0;
        std::fprintf(stderr, "[gen-profile] compute=%.1fms total=%.1fms\n", t_compute / 1000.0, (t_build + t_alloc + t_compute) / 1000.0);
    }

    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        error = "ggml generation graph compute failed";
        return false;
    }

    probe.n_frames = total_frames;

    probe.f0.resize(static_cast<size_t>(ggml_nelements(f0_curve)));
    probe.noise.resize(static_cast<size_t>(ggml_nelements(n_curve)));
    probe.asr.resize(static_cast<size_t>(ggml_nelements(asr)));
    probe.decoder.resize(static_cast<size_t>(ggml_nelements(decoder_cur)));

    probe.asr_dim     = asr->ne[0];
    probe.decoder_len = decoder_cur->ne[0];
    probe.decoder_dim = decoder_cur->ne[1];

    model.backend->tensor_get(f0_curve,    probe.f0.data(),      0, probe.f0.size()      * sizeof(float));
    model.backend->tensor_get(n_curve,     probe.noise.data(),   0, probe.noise.size()   * sizeof(float));
    model.backend->tensor_get(asr,         probe.asr.data(),     0, probe.asr.size()     * sizeof(float));
    model.backend->tensor_get(decoder_cur, probe.decoder.data(), 0, probe.decoder.size() * sizeof(float));

    model.tmp_decoder_style_f32.resize(128);
    model.backend->tensor_get(decoder_style, model.tmp_decoder_style_f32.data(), 0, model.tmp_decoder_style_f32.size() * sizeof(float));

    if (std::getenv("KOKOPOP_GEN_STATS") != nullptr) {
        auto dbg_stats = [&](const char * name, const std::vector<float> & v) {
            double s = 0, s2 = 0;
            float mn = std::numeric_limits<float>::infinity();
            float mx = -std::numeric_limits<float>::infinity();
            for (float x : v) {
                s += x; s2 += static_cast<double>(x) * x;
                if (x < mn) mn = x;
                if (x > mx) mx = x;
            }
            const double n = static_cast<double>(v.size());
            std::fprintf(stderr, "[gen-stats %s] n=%zu min=%.4f max=%.4f mean=%.4f rms=%.4f\n",
                         name, v.size(), mn, mx, s / n, std::sqrt(s2 / n));
        };
        std::fprintf(stderr, "[gen-stats tokens=%lld total_frames=%lld]\n",
                     static_cast<long long>(n_tokens), static_cast<long long>(total_frames));
        dbg_stats("f0",      probe.f0);
        dbg_stats("noise",   probe.noise);
        dbg_stats("asr",     probe.asr);
        dbg_stats("decoder", probe.decoder);
        dbg_stats("style",   model.tmp_decoder_style_f32);
        std::fflush(stderr);
    }

    if (const char * dump_dir = std::getenv("KOKOPOP_DUMP_DIR")) {
        auto dump = [&](const char * name, const std::vector<float> & v) {
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/gen_%s_n%lld.bin",
                          dump_dir, name, static_cast<long long>(n_tokens));
            if (FILE * f = std::fopen(path, "wb")) {
                std::fwrite(v.data(), sizeof(float), v.size(), f);
                std::fclose(f);
            }
        };
        dump("f0",      probe.f0);
        dump("noise",   probe.noise);
        dump("asr",     probe.asr);
        dump("decoder", probe.decoder);
        dump("style",   model.tmp_decoder_style_f32);
    }

    if (!all_finite(probe.f0, "f0", error) ||
        !all_finite(probe.noise, "noise", error) ||
        !all_finite(probe.asr, "asr", error) ||
        !all_finite(probe.decoder, "decoder", error) ||
        !all_finite(model.tmp_decoder_style_f32, "decoder style", error)) {
        ggml_free(ctx);
        return false;
    }

    const size_t decoder_total = static_cast<size_t>(probe.decoder_dim * probe.decoder_len);
    if (decoder_total == 0) {
        error = "zero-size decoder tensor";
        ggml_free(ctx);
        return false;
    }

    std::vector<float> & decoder_buf = model.tmp_decoder_cpu_f32;
    if (decoder_buf.size() < decoder_total) {
        decoder_buf.resize(decoder_total);
    }

    // decoder_cur is [time, channels] in ggml layout, so contiguous memory is:
    //   data[t + decoder_len * c]
    // CpuTensor::at(c, t) is:
    //   data[c * length + t]
    // These are identical when length == decoder_len. A memcpy is correct.
    std::memcpy(decoder_buf.data(), probe.decoder.data(), decoder_total * sizeof(float));

    CpuTensor decoder_cpu{
        probe.decoder_dim,
        probe.decoder_len,
        decoder_buf,
    };

    if (!ggml_generator(model, decoder_cpu, probe.f0, model.tmp_decoder_style_f32, probe.audio, error)) {
        if (error.empty()) {
            error = "Kokoro generator GGML inference failed";
        }
    }

    if (probe.audio.empty()) {
        if (error.empty()) {
            error = "Kokoro generator CPU inference failed";
        }
        ggml_free(ctx);
        return false;
    }

    ggml_free(ctx);
    return true;
}

} // namespace kokopop
