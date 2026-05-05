#include "kokopop.h"

#include "core/error.h"
#include "model/model.h"
#include "synthesis/phonemizer.h"
#include "synthesis/synth.h"
#include "core/utf8.h"
#include "core/wav.h"

#include <cstdlib>
#include <exception>
#include <memory>
#include <string>

struct kokopop_model {
    std::unique_ptr<kokopop::Model> impl;
};

namespace {

int fail(int code, const std::string & message) {
    kokopop::set_error(message);
    return code;
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

    std::string phonemes;
    std::string error;
    const std::string voice_s = voice ? voice : "";
    if (!kokopop::phonemize_text(text, voice_s, phonemes, error)) {
        return fail(KOKOPOP_ERROR_PHONEMIZER, error);
    }
    if (phonemes.empty()) {
        return fail(KOKOPOP_ERROR_PHONEMIZER, "phonemizer produced no phonemes");
    }
    if (!kokopop::synthesize_phonemes(*model->impl, phonemes, voice_s, speed, *out_audio, error)) {
        return fail(KOKOPOP_ERROR_INFERENCE, error);
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
