#include "kokopop.h"

#include "core/error.h"
#include "model/model.h"
#include "audio/audio_postprocess.h"
#include "synthesis/chunker/chunker.h"
#include "synthesis/phonemizer.h"
#include "synthesis/synth.h"
#include "core/utf8.h"
#include "core/wav.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

struct kokopop_model {
    std::unique_ptr<kokopop::Model> impl;
};

namespace {

int fail(int code, const std::string & message) {
    kokopop::set_error(message);
    return code;
}

bool allocate_audio_from_vector(const std::vector<float> & samples, int sample_rate, kokopop_audio & out) {
    out.samples = static_cast<float *>(std::calloc(samples.size(), sizeof(float)));
    if (out.samples == nullptr) {
        return false;
    }
    out.n_samples = samples.size();
    out.sample_rate = sample_rate;
    if (!samples.empty()) {
        std::memcpy(out.samples, samples.data(), samples.size() * sizeof(float));
    }
    return true;
}

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.substr(text.size() - suffix.size()) == suffix;
}

bool zh_text_trace_enabled(const std::string & voice) {
    const char * enabled = std::getenv("KOKOPOP_ZH_TEXT_TRACE");
    return enabled != nullptr && enabled[0] != '\0' &&
           !voice.empty() && voice[0] == 'z';
}

const char * boundary_name(kokopop::Boundary boundary) {
    switch (boundary) {
        case kokopop::Boundary::None: return "none";
        case kokopop::Boundary::ClauseWeak: return "clause_weak";
        case kokopop::Boundary::ClauseStrong: return "clause_strong";
        case kokopop::Boundary::Sentence: return "sentence";
        case kokopop::Boundary::Newline: return "newline";
        case kokopop::Boundary::Paragraph: return "paragraph";
    }
    return "unknown";
}

std::string trailing_punctuation(std::string_view text) {
    static constexpr std::string_view kCandidates[] = {
        "…", "—", ",", ".", ";", ":", "!", "?"
    };
    for (std::string_view candidate : kCandidates) {
        if (ends_with(text, candidate)) {
            return std::string(candidate);
        }
    }
    return {};
}

void log_zh_chunk_trace(const kokopop::Chunk & chunk,
                        std::string_view synthesized_phonemes,
                        size_t index,
                        size_t total) {
    const std::string raw_trailing = trailing_punctuation(chunk.phonemes);
    const std::string synth_trailing = trailing_punctuation(synthesized_phonemes);
    std::fprintf(stderr,
                 "[kokopop][zh-trace] chunk[%zu/%zu] tokens=%d boundary=%s first=%d last=%d\n",
                 index + 1, total, chunk.n_tokens, boundary_name(chunk.boundary_after),
                 chunk.is_first ? 1 : 0, chunk.is_last ? 1 : 0);
    std::fprintf(stderr, "[kokopop][zh-trace]   text: %s\n", chunk.text.c_str());
    std::fprintf(stderr, "[kokopop][zh-trace]   phonemes(raw): %s\n", chunk.phonemes.c_str());
    std::fprintf(stderr, "[kokopop][zh-trace]   phonemes(run): %.*s\n",
                 static_cast<int>(synthesized_phonemes.size()), synthesized_phonemes.data());
    std::fprintf(stderr,
                 "[kokopop][zh-trace]   trailing(raw)=%s trailing(run)=%s\n",
                 raw_trailing.empty() ? "<none>" : raw_trailing.c_str(),
                 synth_trailing.empty() ? "<none>" : synth_trailing.c_str());
}

void trim_trailing_chunk_punctuation(std::string & phonemes) {
    auto trim_spaces = [&]() {
        while (!phonemes.empty() && phonemes.back() == ' ') {
            phonemes.pop_back();
        }
    };

    trim_spaces();
    for (;;) {
        bool trimmed = false;
        if (!phonemes.empty()) {
            const char c = phonemes.back();
            if (c == ',' || c == '.' || c == ';' || c == ':' || c == '!' || c == '?') {
                phonemes.pop_back();
                trimmed = true;
            }
        }
        if (!trimmed && ends_with(phonemes, "…")) {
            phonemes.erase(phonemes.size() - std::strlen("…"));
            trimmed = true;
        }
        if (!trimmed && ends_with(phonemes, "—")) {
            phonemes.erase(phonemes.size() - std::strlen("—"));
            trimmed = true;
        }
        if (!trimmed) {
            break;
        }
        trim_spaces();
    }
}

kokopop::ChunkConfig text_synthesis_config_for_voice(const std::string & voice) {
    kokopop::ChunkConfig config = kokopop::make_long_form_config();
    const char lang = voice.empty() ? 'a' : voice[0];
    if (lang == 'z') {
        // Mandarin is more stable when medium-length sentences stay in a
        // single inference pass instead of being split at clause boundaries.
        config.target_min_tokens = 80;
        config.target_max_tokens = 180;
        config.soft_max_tokens = 260;
        config.first_chunk_target_max_tokens = 180;
        config.allow_short_first_chunk = true;
        config.comma_pause_ms = 120;
        config.sentence_pause_ms = 260;
        config.paragraph_pause_ms = 500;
    }
    return config;
}

} // namespace

extern "C" {

int kokopop_model_load(const char * path, const kokopop_model_options * options, kokopop_model ** out_model) {
    if (out_model == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "out_model is null");
    }
    *out_model = nullptr;
    if (path == nullptr || path[0] == '\0') {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "model path is empty");
    }

    try {
        std::unique_ptr<kokopop::Model> impl;
        std::string error;
        if (!kokopop::load_model_from_gguf(path, options, impl, error)) {
            return fail(KOKOPOP_ERROR_MODEL, error);
        }
        std::unique_ptr<kokopop_model> wrapper(new kokopop_model());
        wrapper->impl = std::move(impl);
        *out_model = wrapper.release();
        return KOKOPOP_OK;
    } catch (const std::exception & e) {
        return fail(KOKOPOP_ERROR_MODEL, e.what());
    }
}

int kokopop_synthesize_text(
    kokopop_model * model, const char * text,
    const char * voice, float speed,
    kokopop_audio * out_audio) {
    if (out_audio == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "out_audio is null");
    }
    out_audio->samples = nullptr;
    out_audio->n_samples = 0;
    out_audio->sample_rate = 0;
    if (model == nullptr || model->impl == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "model is null");
    }
    if (text == nullptr || kokopop::trim_ascii(text).empty()) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "text is empty");
    }

    std::string error;
    const std::string voice_s = voice ? voice : "";

    const kokopop::ChunkConfig config = text_synthesis_config_for_voice(voice_s);
    kokopop::TokenizeFn tokenize_fn =
        [&](const std::string & phonemes, std::vector<uint32_t> & ids, std::string & token_error) -> bool {
            return model->impl->tokenize_phonemes(phonemes, ids, token_error);
        };

    auto chunks = kokopop::chunk_text(text, voice_s, config, tokenize_fn, error);
    if (chunks.empty()) {
        return fail(KOKOPOP_ERROR_INFERENCE, error.empty() ? "text chunker produced no chunks" : error);
    }
    const bool trace_zh = zh_text_trace_enabled(voice_s);

    std::vector<float> combined;
    combined.reserve(4096);
    for (size_t i = 0; i < chunks.size(); ++i) {
        auto & chunk = chunks[i];
        std::string chunk_phonemes = chunk.phonemes;
        if (!chunk.is_last) {
            // Chunk pauses are added by postprocess_chunk_audio(), so keeping
            // punctuation tokens at an intermediate chunk boundary only makes
            // the model less stable without improving prosody.
            trim_trailing_chunk_punctuation(chunk_phonemes);
        }
        if (trace_zh) {
            log_zh_chunk_trace(chunk, chunk_phonemes, i, chunks.size());
        }
        kokopop_audio chunk_audio{};
        if (!kokopop::synthesize_phonemes(*model->impl, chunk_phonemes, voice_s, speed, chunk_audio, error)) {
            return fail(KOKOPOP_ERROR_INFERENCE, error);
        }

        std::vector<float> raw(chunk_audio.samples, chunk_audio.samples + chunk_audio.n_samples);
        kokopop_audio_free(&chunk_audio);

        auto processed = kokopop::postprocess_chunk_audio(
            raw, chunk, static_cast<int>(i), static_cast<int>(chunks.size()),
            config, model->impl->sample_rate);
        combined.insert(combined.end(), processed.begin(), processed.end());
    }

    if (combined.empty()) {
        return fail(KOKOPOP_ERROR_INFERENCE, "synthesis produced no audio");
    }
    if (!allocate_audio_from_vector(combined, model->impl->sample_rate, *out_audio)) {
        return fail(KOKOPOP_ERROR_INFERENCE, "failed to allocate output audio");
    }
    return KOKOPOP_OK;
}

int kokopop_synthesize_phonemes(
    kokopop_model * model, const char * phonemes,
    const char * voice, float speed,
    kokopop_audio * out_audio) {
    if (out_audio == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "out_audio is null");
    }
    out_audio->samples = nullptr;
    out_audio->n_samples = 0;
    out_audio->sample_rate = 0;
    if (model == nullptr || model->impl == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "model is null");
    }
    if (phonemes == nullptr || kokopop::trim_ascii(phonemes).empty()) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "phonemes are empty");
    }

    std::string error;
    const std::string voice_s = voice ? voice : "";
    if (!kokopop::synthesize_phonemes(*model->impl, phonemes, voice_s, speed, *out_audio, error)) {
        return fail(KOKOPOP_ERROR_INFERENCE, error);
    }
    return KOKOPOP_OK;
}

int kokopop_write_wav(const char * path, const kokopop_audio * audio) {
    if (path == nullptr || audio == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "invalid WAV write argument");
    }
    std::string error;
    if (!kokopop::write_wav_file(path, *audio, error)) {
        return fail(KOKOPOP_ERROR_IO, error);
    }
    return KOKOPOP_OK;
}

void kokopop_audio_free(kokopop_audio * audio) {
    if (audio == nullptr) {
        return;
    }
    std::free(audio->samples);
    audio->samples = nullptr;
    audio->n_samples = 0;
    audio->sample_rate = 0;
}

void kokopop_model_free(kokopop_model * model) {
    delete model;
}

int kokopop_model_sample_rate(kokopop_model * model) {
    if (!model || !model->impl) return 0;
    return model->impl->sample_rate;
}

kokopop::Model * kokopop_model_get_impl(kokopop_model * model) {
    if (!model || !model->impl) return nullptr;
    return model->impl.get();
}

const char * kokopop_last_error(void) {
    return kokopop::last_error();
}

} // extern "C"
