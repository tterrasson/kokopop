#include "kokoro.h"

#include "constants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <ggml.h>
#include <numeric>
#include <thread>

namespace kokopop {

ggml_tensor * require_tensor(Model & model, const char * name, std::string & error) {
    ggml_tensor * t = model.cached_tensor(name);
    if (t == nullptr) {
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
    w.q_w = require_tensor(model, (std::string(base) + "attention.query.weight").c_str(), error);
    w.q_b = require_tensor(model, (std::string(base) + "attention.query.bias").c_str(), error);
    w.k_w = require_tensor(model, (std::string(base) + "attention.key.weight").c_str(), error);
    w.k_b = require_tensor(model, (std::string(base) + "attention.key.bias").c_str(), error);
    w.v_w = require_tensor(model, (std::string(base) + "attention.value.weight").c_str(), error);
    w.v_b = require_tensor(model, (std::string(base) + "attention.value.bias").c_str(), error);
    w.o_w = require_tensor(model, (std::string(base) + "attention.dense.weight").c_str(), error);
    w.o_b = require_tensor(model, (std::string(base) + "attention.dense.bias").c_str(), error);
    w.attn_norm_w = require_tensor(model, (std::string(base) + "attention.LayerNorm.weight").c_str(), error);
    w.attn_norm_b = require_tensor(model, (std::string(base) + "attention.LayerNorm.bias").c_str(), error);
    w.ffn_w = require_tensor(model, (std::string(base) + "ffn.weight").c_str(), error);
    w.ffn_b = require_tensor(model, (std::string(base) + "ffn.bias").c_str(), error);
    w.ffn_out_w = require_tensor(model, (std::string(base) + "ffn_output.weight").c_str(), error);
    w.ffn_out_b = require_tensor(model, (std::string(base) + "ffn_output.bias").c_str(), error);
    w.out_norm_w = require_tensor(model, (std::string(base) + "full_layer_layer_norm.weight").c_str(), error);
    w.out_norm_b = require_tensor(model, (std::string(base) + "full_layer_layer_norm.bias").c_str(), error);
    return error.empty();
}

int64_t voice_style_row(ggml_tensor * voice, int64_t n_tokens) {
    if (voice == nullptr || voice->ne[1] <= 0) {
        return 0;
    }
    return std::clamp<int64_t>(n_tokens - 3, 0, voice->ne[1] - 1);
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

} // namespace

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

bool run_kokoro_frontend_probe(
    Model & model,
    const std::vector<uint32_t> & ids,
    const std::string & requested_voice,
    KokoroFrontendProbe & probe,
    std::string & error) {
    if (ids.empty()) {
        error = "cannot run Kokoro frontend on empty token sequence";
        return false;
    }

    ggml_tensor * word = require_tensor(model, "kokopop.albert.embeddings.word_embeddings.weight", error);
    ggml_tensor * pos = require_tensor(model, "kokopop.albert.embeddings.position_embeddings.weight", error);
    ggml_tensor * type = require_tensor(model, "kokopop.albert.embeddings.token_type_embeddings.weight", error);
    ggml_tensor * norm_w = require_tensor(model, "kokopop.albert.embeddings.LayerNorm.weight", error);
    ggml_tensor * norm_b = require_tensor(model, "kokopop.albert.embeddings.LayerNorm.bias", error);
    ggml_tensor * map_w = require_tensor(model, "kokopop.albert.encoder.embedding_hidden_mapping_in.weight", error);
    ggml_tensor * map_b = require_tensor(model, "kokopop.albert.encoder.embedding_hidden_mapping_in.bias", error);
    ggml_tensor * enc_w = require_tensor(model, "kokopop.bert_encoder.weight", error);
    ggml_tensor * enc_b = require_tensor(model, "kokopop.bert_encoder.bias", error);
    ggml_tensor * duration_w = require_tensor(model, "kokopop.predictor.duration_proj.linear_layer.weight", error);
    ggml_tensor * duration_b = require_tensor(model, "kokopop.predictor.duration_proj.linear_layer.bias", error);
    AlbertLayerWeights layer;
    load_albert_layer(model, layer, error);
    if (!add_required(word, error) || !add_required(pos, error) || !add_required(type, error) ||
        !add_required(norm_w, error) || !add_required(norm_b, error) || !add_required(map_w, error) ||
        !add_required(map_b, error) || !add_required(enc_w, error) || !add_required(enc_b, error)) {
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
    ggml_set_input(token_ids);
    ggml_set_input(pos_ids);

    ggml_tensor * cur = ggml_add(ctx, ggml_get_rows(ctx, word, token_ids), ggml_get_rows(ctx, pos, pos_ids));
    ggml_tensor * type_zero = ggml_view_2d(ctx, type, type->ne[0], 1, type->nb[1], 0);
    cur = ggml_add(ctx, cur, ggml_repeat(ctx, type_zero, cur));
    cur = layer_norm(ctx, cur, norm_w, norm_b, 1e-12f);
    cur = linear(ctx, map_w, map_b, cur);

    constexpr int64_t hidden_size = KOKOPOP_HIDDEN_SIZE;
    constexpr int64_t n_heads = KOKOPOP_NUM_HEADS;
    constexpr int64_t head_size = KOKOPOP_HEAD_SIZE;
    constexpr float attn_scale = KOKOPOP_ATTN_SCALE;

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
        scores = ggml_soft_max_ext(ctx, scores, nullptr, attn_scale, 0.0f);

        v = ggml_cont_3d(ctx, ggml_transpose(ctx, v), n_tokens, head_size, n_heads);
        ggml_tensor * attended = ggml_mul_mat(ctx, scores, v);
        attended = ggml_permute(ctx, attended, 2, 0, 1, 3);
        attended = ggml_cont_2d(ctx, attended, hidden_size, n_tokens);
        attended = linear(ctx, layer.o_w, layer.o_b, attended);

        cur = layer_norm(ctx, ggml_add(ctx, attended, residual), layer.attn_norm_w, layer.attn_norm_b, 1e-12f);
        residual = cur;
        cur = ggml_gelu(ctx, linear(ctx, layer.ffn_w, layer.ffn_b, cur));
        cur = linear(ctx, layer.ffn_out_w, layer.ffn_out_b, cur);
        cur = layer_norm(ctx, ggml_add(ctx, cur, residual), layer.out_norm_w, layer.out_norm_b, 1e-12f);
    }

    cur = linear(ctx, enc_w, enc_b, cur);

    const std::string voice_name = resolve_voice_name(requested_voice, model.voices);
    ggml_tensor * voice = nullptr;
    if (!voice_name.empty()) {
        voice = model.cached_tensor("kokopop.voice." + voice_name);
    }
    if (voice == nullptr) {
        error = "missing Kokoro voice tensor for duration graph";
        ggml_free(ctx);
        return false;
    }
    const int64_t voice_row = voice_style_row(voice, n_tokens);
    ggml_tensor * style = ggml_cast(
        ctx, ggml_view_1d(ctx, voice, 128,
            128 * voice->nb[0] + voice_row * voice->nb[1]),
        GGML_TYPE_F32);

    // Pre-reserve LstmCustomParams storage before graph construction so that
    // pointers into the vector remain stable while lstm_direction() pushes.
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
    ggml_tensor * dur_logits = linear(ctx, duration_w, duration_b, dur_hidden);
    ggml_tensor * pred_dur = ggml_sum_rows(ctx, ggml_sigmoid(ctx, dur_logits));
    ggml_set_name(pred_dur, "kokopop_duration_probe");
    ggml_set_output(d);
    ggml_set_output(pred_dur);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, frontend_graph_size(n_tokens), false);
    ggml_build_forward_expand(gf, pred_dur);

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
    {
        model.tmp_ids_i32.resize(static_cast<size_t>(n_tokens));
        model.tmp_pos_i32.resize(static_cast<size_t>(n_tokens));
        for (int64_t i = 0; i < n_tokens; ++i) {
            model.tmp_ids_i32[static_cast<size_t>(i)] = static_cast<int32_t>(ids[static_cast<size_t>(i)]);
            model.tmp_pos_i32[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        model.backend->tensor_set(token_ids, model.tmp_ids_i32.data(),  0, ggml_nbytes(token_ids));
        model.backend->tensor_set(pos_ids,   model.tmp_pos_i32.data(),  0, ggml_nbytes(pos_ids));
    }
    ggml_status status = model.backend->compute(ctx, gf);

    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        error = "ggml frontend graph compute failed";
        return false;
    }

    const int64_t n = ggml_nelements(d);
    probe.hidden.resize(static_cast<size_t>(n));
    probe.n_tokens = n_tokens;
    probe.hidden_dim = d->ne[0];
    const int64_t nd = ggml_nelements(pred_dur);
    probe.durations.resize(static_cast<size_t>(nd));
    model.backend->tensor_get(d,        probe.hidden.data(),    0, static_cast<size_t>(n)  * sizeof(float));
    model.backend->tensor_get(pred_dur, probe.durations.data(), 0, static_cast<size_t>(nd) * sizeof(float));
    ggml_free(ctx);
    return true;
}

bool run_kokoro_generation_probe(
    Model & model,
    const std::vector<uint32_t> & ids,
    const std::string & requested_voice,
    float speed,
    const KokoroFrontendProbe & frontend,
    KokoroGenerationProbe & probe,
    std::string & error) {
    if (ids.empty() || frontend.n_tokens != static_cast<int64_t>(ids.size())) {
        error = "invalid Kokoro generation inputs";
        return false;
    }
    if (frontend.hidden_dim <= 0 || frontend.hidden.empty() ||
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
        std::fprintf(stderr,
            "[kokopop] WARNING: large chunk (%lld frames, ~%.1fs audio), may use significant RAM\n",
            (long long)total_frames,
            (double)total_frames * 300 / static_cast<double>(KOKOPOP_SAMPLE_RATE));
    }

    ggml_tensor * shared_w = model.cached_tensor("kokopop.predictor.shared.weight_ih_l0.0");
    if (shared_w == nullptr) {
        shared_w = require_tensor(model, "kokopop.predictor.shared.weight_ih_l0", error);
    }
    ggml_tensor * f0_proj_w = require_tensor(model, "kokopop.predictor.F0_proj.weight", error);
    ggml_tensor * f0_proj_b = require_tensor(model, "kokopop.predictor.F0_proj.bias", error);
    ggml_tensor * n_proj_w = require_tensor(model, "kokopop.predictor.N_proj.weight", error);
    ggml_tensor * n_proj_b = require_tensor(model, "kokopop.predictor.N_proj.bias", error);
    if (!add_required(shared_w, error) || !add_required(f0_proj_w, error) || !add_required(f0_proj_b, error) ||
        !add_required(n_proj_w, error) || !add_required(n_proj_b, error)) {
        return false;
    }

    const std::string voice_name = resolve_voice_name(requested_voice, model.voices);
    ggml_tensor * voice = nullptr;
    if (!voice_name.empty()) {
        voice = model.cached_tensor("kokopop.voice." + voice_name);
    }
    if (voice == nullptr) {
        error = "missing Kokoro voice tensor for generation graph";
        return false;
    }

    const size_t mem_size = model.backend->generation_context_bytes(total_frames, frontend.n_tokens);
    model.backend->clear_pending_inits();
    ggml_context * ctx = init_scratch_context(model, model.generation_scratch, mem_size, true, "generation", error);
    if (ctx == nullptr) {
        return false;
    }

    const int64_t n_tokens = frontend.n_tokens;
    model.backend->set_input_tokens(static_cast<int>(n_tokens));
    model.backend->set_active_label("generation");

    // Pre-reserve LstmCustomParams storage before any LSTM node is created so
    // userdata pointers captured by ggml_map_custom2 stay valid until compute().
    model.lstm_custom_params.clear();
    model.lstm_custom_params.reserve(24);

    ggml_tensor * token_ids     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * duration_pred = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frontend.hidden_dim, n_tokens);
    ggml_tensor * duration_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, total_frames, n_tokens);
    ggml_set_input(token_ids);
    ggml_set_input(duration_pred);
    ggml_set_input(duration_mask);

    // 6.1 — Reusable mask buffer: grow the buffer once and reuse it across calls.
    //       Avoids repeated allocations/resizes when chunk sizes vary.
    std::vector<float> & mask_data = model.tmp_mask_f32;
    const size_t mask_size = static_cast<size_t>(total_frames * n_tokens);
    if (mask_data.size() < mask_size) {
        mask_data.resize(mask_size);
    }
    // Zero only the active region (the buffer may be larger than needed).
    std::fill(mask_data.begin(), mask_data.begin() + static_cast<ptrdiff_t>(mask_size), 0.0f);
    {
        int64_t running = 0;
        for (int64_t token = 0; token < n_tokens; ++token) {
            const int64_t token_dur = durations[static_cast<size_t>(token)];
            const size_t  row_offset = static_cast<size_t>(token * total_frames);
            for (int frame = 0; frame < token_dur; ++frame) {
                const size_t idx = static_cast<size_t>(row_offset + running + frame);
                // Bounds check: idx must be within [0, total_frames * n_tokens).
                if (idx >= mask_size) {
                    error = "duration mask index out of bounds";
                    ggml_free(ctx);
                    return false;
                }
                mask_data[idx] = 1.0f;
            }
            running += token_dur;
        }
    }

    const int64_t voice_row = voice_style_row(voice, n_tokens);
    ggml_tensor * prosody_style = ggml_cast(
        ctx, ggml_view_1d(ctx, voice, 128,
            128 * voice->nb[0] + voice_row * voice->nb[1]),
        GGML_TYPE_F32);

    ggml_tensor * cur = ggml_mul_mat(
        ctx, ggml_cont(ctx, ggml_transpose(ctx, duration_mask)),
        ggml_cont(ctx, ggml_transpose(ctx, duration_pred)));
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
    cur = bidirectional_lstm(ctx, model, cur, "kokopop.predictor.shared", total_frames, error);
    if (cur == nullptr) {
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * f0_curve = ggml_cont(ctx, ggml_transpose(ctx, cur));
    for (int i = 0; i < 3; ++i) {
        f0_curve = adain_resblk1d(
            ctx, model, f0_curve, prosody_style,
            "kokopop.predictor.F0." + std::to_string(i), i == 1, error);
        if (f0_curve == nullptr) {
            ggml_free(ctx);
            return false;
        }
    }
    f0_curve = ggml_cont(ctx, ggml_transpose(ctx, f0_curve));
    f0_curve = add_channel_bias(ctx,
        conv1d(ctx, f0_proj_w,
            ggml_cont(ctx, ggml_transpose(ctx, f0_curve)), 1, 0, 1, 1),
        f0_proj_b);
    f0_curve = ggml_cont(ctx, ggml_view_1d(ctx, f0_curve, f0_curve->ne[0], 0));
    ggml_set_name(f0_curve, "kokopop_f0_probe");
    ggml_set_output(f0_curve);

    ggml_tensor * n_curve = ggml_cont(ctx, ggml_transpose(ctx, cur));
    for (int i = 0; i < 3; ++i) {
        n_curve = adain_resblk1d(
            ctx, model, n_curve, prosody_style,
            "kokopop.predictor.N." + std::to_string(i), i == 1, error);
        if (n_curve == nullptr) {
            ggml_free(ctx);
            return false;
        }
    }
    n_curve = ggml_cont(ctx, ggml_transpose(ctx, n_curve));
    n_curve = add_channel_bias(ctx,
        conv1d(ctx, n_proj_w,
            ggml_cont(ctx, ggml_transpose(ctx, n_curve)), 1, 0, 1, 1),
        n_proj_b);
    n_curve = ggml_cont(ctx, ggml_view_1d(ctx, n_curve, n_curve->ne[0], 0));
    ggml_set_name(n_curve, "kokopop_noise_probe");
    ggml_set_output(n_curve);

    ggml_tensor * asr = text_encoder(ctx, model, token_ids, duration_mask, n_tokens, error);
    if (asr == nullptr) {
        ggml_free(ctx);
        return false;
    }
    ggml_set_name(asr, "kokopop_asr_probe");
    ggml_set_output(asr);

    ggml_tensor * decoder_style = ggml_cast(
        ctx, ggml_view_1d(ctx, voice, 128, voice_row * voice->nb[1]),
        GGML_TYPE_F32);
    ggml_tensor * f0_conv_w = require_tensor(model, "kokopop.decoder.F0_conv.weight", error);
    ggml_tensor * f0_conv_b = require_tensor(model, "kokopop.decoder.F0_conv.bias", error);
    ggml_tensor * n_conv_w = require_tensor(model, "kokopop.decoder.N_conv.weight", error);
    ggml_tensor * n_conv_b = require_tensor(model, "kokopop.decoder.N_conv.bias", error);
    ggml_tensor * asr_res_w = require_tensor(model, "kokopop.decoder.asr_res.0.weight", error);
    ggml_tensor * asr_res_b = require_tensor(model, "kokopop.decoder.asr_res.0.bias", error);
    if (!error.empty()) {
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * f0_dec_in = ggml_reshape_2d(ctx, f0_curve, f0_curve->ne[0], 1);
    ggml_tensor * n_dec_in = ggml_reshape_2d(ctx, n_curve, n_curve->ne[0], 1);
    ggml_tensor * f0_dec = add_channel_bias(ctx, conv1d(ctx, f0_conv_w, f0_dec_in, 2, 1, 1, 3), f0_conv_b);
    ggml_tensor * n_dec = add_channel_bias(ctx, conv1d(ctx, n_conv_w, n_dec_in, 2, 1, 1, 3), n_conv_b);

    ggml_tensor * decoder_cur = ggml_concat(
        ctx, ggml_concat(
            ctx, ggml_cont(ctx, ggml_transpose(ctx, asr)),
            f0_dec, 1),
        n_dec, 1);
    decoder_cur = adain_resblk1d(ctx, model, decoder_cur, decoder_style, "kokopop.decoder.encode", false, error);
    if (decoder_cur == nullptr) {
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * asr_res = add_channel_bias(ctx,
        conv1d(ctx, asr_res_w,
            ggml_cont(ctx, ggml_transpose(ctx, asr)), 1, 0, 1, 1),
        asr_res_b);
    for (int i = 0; i < 4; ++i) {
        decoder_cur = ggml_concat(
            ctx, ggml_concat(
                ctx, ggml_concat(ctx, decoder_cur, asr_res, 1),
                f0_dec, 1),
            n_dec, 1);
        decoder_cur = adain_resblk1d(
            ctx, model, decoder_cur, decoder_style,
            "kokopop.decoder.decode." + std::to_string(i), i == 3,
            error);
        if (decoder_cur == nullptr) {
            ggml_free(ctx);
            return false;
        }
    }
    ggml_set_name(decoder_cur, "kokopop_decoder_probe");
    ggml_set_output(decoder_cur);

    const bool gen_profile = std::getenv("KOKOPOP_GEN_PROFILE") != nullptr;
    int64_t t_build = 0, t_alloc = 0, t_compute = 0;

    int64_t t0 = gen_profile ? ggml_time_us() : 0;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, generation_graph_size(total_frames, n_tokens), false);
    ggml_build_forward_expand(gf, f0_curve);
    ggml_build_forward_expand(gf, n_curve);
    ggml_build_forward_expand(gf, asr);
    ggml_build_forward_expand(gf, decoder_cur);
    if (gen_profile) {
        t_build = ggml_time_us() - t0;
        std::fprintf(stderr, "[gen-profile] n_nodes=%d  build=%.1fms\n",
            ggml_graph_n_nodes(gf), t_build / 1000.0);
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
    {
        model.tmp_ids_i32.resize(static_cast<size_t>(n_tokens));
        for (int64_t i = 0; i < n_tokens; ++i) {
            model.tmp_ids_i32[static_cast<size_t>(i)] = static_cast<int32_t>(ids[static_cast<size_t>(i)]);
        }
        model.backend->tensor_set(token_ids,     model.tmp_ids_i32.data(), 0, ggml_nbytes(token_ids));
        model.backend->tensor_set(duration_pred, frontend.hidden.data(),   0, ggml_nbytes(duration_pred));
        model.backend->tensor_set(duration_mask, mask_data.data(),         0, ggml_nbytes(duration_mask));
    }
    t0 = gen_profile ? ggml_time_us() : 0;
    ggml_status status = model.backend->compute(ctx, gf);
    if (gen_profile) {
        t_compute = ggml_time_us() - t0;
        std::fprintf(stderr, "[gen-profile] compute=%.1fms  total=%.1fms\n",
            t_compute / 1000.0, (t_build + t_alloc + t_compute) / 1000.0);
    }

    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        error = "ggml generation graph compute failed";
        return false;
    }

    probe.n_frames = total_frames;
    probe.f0.resize(static_cast<size_t>(ggml_nelements(f0_curve)));
    probe.noise.resize(static_cast<size_t>(ggml_nelements(n_curve)));
    model.backend->tensor_get(f0_curve, probe.f0.data(),    0, probe.f0.size()    * sizeof(float));
    model.backend->tensor_get(n_curve,  probe.noise.data(), 0, probe.noise.size() * sizeof(float));

    probe.asr_dim = asr->ne[0];
    probe.asr.resize(static_cast<size_t>(ggml_nelements(asr)));
    probe.decoder_dim = decoder_cur->ne[1];
    probe.decoder_len = decoder_cur->ne[0];
    probe.decoder.resize(static_cast<size_t>(ggml_nelements(decoder_cur)));
    model.tmp_decoder_style_f32.resize(128);
    model.backend->tensor_get(asr,          probe.asr.data(),          0, probe.asr.size()          * sizeof(float));
    model.backend->tensor_get(decoder_cur,  probe.decoder.data(),      0, probe.decoder.size()      * sizeof(float));
    model.backend->tensor_get(
        decoder_style, model.tmp_decoder_style_f32.data(), 0,
        model.tmp_decoder_style_f32.size() * sizeof(float));

    if (!all_finite(probe.f0, "f0", error) ||
        !all_finite(probe.noise, "noise", error) ||
        !all_finite(probe.asr, "asr", error) ||
        !all_finite(probe.decoder, "decoder", error) ||
        !all_finite(model.tmp_decoder_style_f32, "decoder style", error)) {
        ggml_free(ctx);
        return false;
    }

    // 6.2 — Reuse model.tmp_decoder_cpu_f32 for the transposed decoder buffer.
    //       probe.decoder is row-major [decoder_dim][decoder_len] (raw flat).
    //       CpuTensor stores column-major [decoder_len][decoder_dim] via at(c,t) = data[c*len+t].
    //       The transpose maps raw[c*len+t] -> cpu[c*len+t], which is the same flat layout,
    //       so a single memcpy suffices — no element-wise loop needed.
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
    // probe.decoder is already in the flat layout that CpuTensor expects:
    //   raw[c * decoder_len + t] corresponds to decoder_cpu.at(c, t) = buf[c * decoder_len + t].
    // No element-wise transpose is needed — a single memcpy does the job.
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
