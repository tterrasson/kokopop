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
#include <string>
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

kokopop::ChunkConfig text_synthesis_config_for_voice(const std::string & voice) {
    kokopop::ChunkConfig config = kokopop::make_long_form_config();
    const char lang = voice.empty() ? 'a' : voice[0];
    if (lang == 'z') {
        config.target_min_tokens = 40;
        config.target_max_tokens = 110;
        config.soft_max_tokens = 180;
        config.first_chunk_target_max_tokens = 110;
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

    std::vector<float> combined;
    combined.reserve(4096);
    for (size_t i = 0; i < chunks.size(); ++i) {
        auto & chunk = chunks[i];
        kokopop_audio chunk_audio{};
        if (!kokopop::synthesize_phonemes(*model->impl, chunk.phonemes, voice_s, speed, chunk_audio, error)) {
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
