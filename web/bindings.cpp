#include "kokopop.h"

#include <cstdlib>

#include <emscripten/emscripten.h>

// Keep C struct layout and ownership out of the JS API.
extern "C" {

EMSCRIPTEN_KEEPALIVE kokopop_model * kp_load(const char * path, int backend) {
    setenv("ESPEAK_DATA_PATH", "/", 1);
    kokopop_model_options options{1, backend};
    kokopop_model * model = nullptr;
    if (kokopop_model_load(path, &options, &model) != KOKOPOP_OK) {
        return nullptr;
    }
    return model;
}

EMSCRIPTEN_KEEPALIVE kokopop_audio * kp_synthesize(
    kokopop_model * model, const char * text, const char * voice, float speed, int phonemes) {
    auto * audio = new kokopop_audio{};
    const int result = phonemes
        ? kokopop_synthesize_phonemes(model, text, voice, speed, audio)
        : kokopop_synthesize_text(model, text, voice, speed, audio);
    if (result != KOKOPOP_OK) {
        kokopop_audio_free(audio);
        delete audio;
        return nullptr;
    }
    return audio;
}

EMSCRIPTEN_KEEPALIVE const float * kp_samples(kokopop_audio * audio) { return audio->samples; }
EMSCRIPTEN_KEEPALIVE size_t kp_length(kokopop_audio * audio) { return audio->n_samples; }
EMSCRIPTEN_KEEPALIVE int kp_rate(kokopop_audio * audio) { return audio->sample_rate; }

EMSCRIPTEN_KEEPALIVE void kp_audio_free(kokopop_audio * audio) {
    kokopop_audio_free(audio);
    delete audio;
}

EMSCRIPTEN_KEEPALIVE void kp_free(kokopop_model * model) { kokopop_model_free(model); }
EMSCRIPTEN_KEEPALIVE const char * kp_error() { return kokopop_last_error(); }

EMSCRIPTEN_KEEPALIVE size_t kp_voice_count(kokopop_model * model) { return kokopop_model_voice_count(model); }
EMSCRIPTEN_KEEPALIVE const char * kp_voice_name(kokopop_model * model, size_t i) { return kokopop_model_voice_name(model, i); }
EMSCRIPTEN_KEEPALIVE int kp_voice_rate(kokopop_model * model, const char * voice) { return kokopop_model_voice_sample_rate(model, voice); }

EMSCRIPTEN_KEEPALIVE int kp_backend(kokopop_model * model) { return kokopop_model_backend(model); }
EMSCRIPTEN_KEEPALIVE const char * kp_arch(kokopop_model * model) { return kokopop_model_arch_name(model); }

} // extern "C"
