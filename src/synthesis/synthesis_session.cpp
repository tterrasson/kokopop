#include "synthesis/synthesis_session.h"

#include <algorithm>
#include <chrono>
#include <utility>

namespace kokopop {

SynthesisSession::SynthesisSession(Model & model, SynthesisSessionOptions options)
    : _model(model), _options(std::move(options)) {
}

bool SynthesisSession::push_text(const std::string & text, std::string & error) {
    if (_started || _input_finished) {
        error = "cannot push text after synthesis has started";
        return false;
    }
    if (text.empty()) {
        error = "text is empty";
        return false;
    }
    _text.append(text);
    return true;
}

bool SynthesisSession::finish_input(std::string & error) {
    if (_input_finished) return true;
    if (_text.empty()) {
        error = "text is empty";
        return false;
    }
    _input_finished = true;
    return true;
}

ChunkConfig SynthesisSession::selected_config() const {
    if (_options.use_exact_chunk_config) {
        return _options.chunk_config;
    }

    ChunkConfig cfg = _options.mode == StreamMode::LongForm
        ? make_long_form_config()
        : make_adaptative_config();
    if (_options.has_chunk_config) {
        cfg = merge_chunk_config(cfg, _options.chunk_config);
    }
    return cfg;
}

bool SynthesisSession::prepare(std::string & error) {
    if (_prepared) return true;
    if (!_input_finished) {
        error = "input is not finished";
        return false;
    }
    if (_text.empty()) {
        error = "text is empty";
        return false;
    }

    _started = true;
    VoiceFrontend frontend;
    if (!make_voice_frontend(_model, _options.voice, frontend, error)) {
        return false;
    }
    const ChunkConfig cfg =
        _model.arch->adjust_chunk_config(selected_config(), frontend.voice);
    // Kept for the adaptative path, which assembles a chunk per `next()` call
    // and re-tokenizes it into its canonical sequence.
    _tokenize = frontend.tokenize;

    if (_options.mode == StreamMode::LongForm) {
        _plan.chunks = chunk_text(_text, cfg,
                                  frontend.phonemize, frontend.tokenize, error);
        if (_plan.chunks.empty()) return false;
        _plan.voice = _options.voice;
        _plan.speed = _options.speed;
        _plan.mode = _options.mode;
        _plan.config = cfg;
        _plan.diffusion = _options.diffusion;
        _chunks_total = static_cast<int>(_plan.chunks.size());
    } else {
        _adaptative_units = prepare_chunk_units(_text, cfg,
                                                frontend.phonemize,
                                                frontend.tokenize, error);
        if (_adaptative_units.empty()) return false;

        _plan.voice = _options.voice;
        _plan.speed = _options.speed;
        _plan.mode = _options.mode;
        _plan.config = cfg;
        _plan.diffusion = _options.diffusion;
        _adaptative_controller.min_tokens = cfg.target_min_tokens;
        _adaptative_controller.max_tokens = cfg.soft_max_tokens;
        _adaptative_controller.growth_max_tokens = cfg.target_max_tokens;
        _chunks_total = 0;
    }

    _prepared = true;
    return true;
}

bool SynthesisSession::next(size_t max_chunks, size_t queued_requests,
                            std::vector<SynthesisAudioChunk> & out,
                            std::string & error) {
    out.clear();
    if (max_chunks == 0) return true;
    if (_done) return true;
    if (!prepare(error)) return false;

    while (!_done && out.size() < max_chunks) {
        std::vector<float> out_tail;
        std::vector<float> audio;
        bool is_last_chunk = false;
        int chunk_index = _chunks_completed;

        if (_options.mode == StreamMode::LongForm) {
            if (chunk_index >= _chunks_total) {
                _done = true;
                break;
            }
            audio = infer_chunk(_model, _plan, chunk_index, _prev_tail, out_tail, error);
            is_last_chunk = chunk_index >= _chunks_total - 1;
        } else {
            const int target_tokens = _adaptative_controller.target_tokens(queued_requests);
            std::string chunk_error;
            Chunk chunk = build_adaptative_chunk(
                _adaptative_units, _adaptative_next_unit, _plan.config,
                target_tokens, chunk_index == 0, _tokenize, chunk_error);
            if (chunk.tokens.empty()) {
                if (!chunk_error.empty()) {
                    error = chunk_error;
                    return false;
                }
                _done = true;
                break;
            }

            is_last_chunk = chunk.is_last;
            SynthesisPlan one_chunk_plan;
            one_chunk_plan.chunks.push_back(std::move(chunk));
            one_chunk_plan.voice = _options.voice;
            one_chunk_plan.speed = _options.speed;
            one_chunk_plan.mode = _options.mode;
            one_chunk_plan.config = _plan.config;
            one_chunk_plan.diffusion = _options.diffusion;

            const auto start = std::chrono::steady_clock::now();
            audio = infer_chunk(_model, one_chunk_plan, 0, _prev_tail, out_tail, error,
                                chunk_index);
            const auto end = std::chrono::steady_clock::now();

            if (!audio.empty()) {
                const double generation_ms =
                    std::chrono::duration<double, std::milli>(end - start).count();
                const double audio_ms = static_cast<double>(audio.size()) /
                    static_cast<double>(sample_rate()) * 1000.0;
                const bool first_chunk_oversized =
                    (chunk_index == 0 && _plan.config.first_chunk_target_max_tokens > 0);
                if (!first_chunk_oversized) {
                    _adaptative_controller.observe(
                        one_chunk_plan.chunks.front().n_tokens,
                        generation_ms, audio_ms);
                }
            }

            if (is_last_chunk) {
                _chunks_total = chunk_index + 1;
            }
        }

        if (audio.empty()) {
            if (error.empty()) error = "inference failed";
            return false;
        }

        _prev_tail = std::move(out_tail);
        ++_chunks_completed;
        if (is_last_chunk) _done = true;
        out.push_back({std::move(audio), chunk_index, is_last_chunk});
    }

    return true;
}

} // namespace kokopop
