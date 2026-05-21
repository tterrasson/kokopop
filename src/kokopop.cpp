#include "kokopop.h"

#include "core/error.h"
#include "model/model.h"
#include "audio/audio_postprocess.h"
#include "audio/audio_encoder.h"
#include "synthesis/chunker/chunker.h"
#include "synthesis/phonemizer.h"
#include "synthesis/synthesis_session.h"
#include "synthesis/synth.h"
#include "core/utf8.h"
#include "core/wav.h"

#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

struct kokopop_model {
    std::unique_ptr<kokopop::Model> impl;
};

struct kokopop_synthesis {
    std::unique_ptr<kokopop::SynthesisSession> impl;
};

struct kokopop_audio_encoder {
    std::unique_ptr<kokopop::AudioEncoder> impl;
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

bool allocate_bytes_from_vector(const std::vector<uint8_t> & bytes, kokopop_bytes & out) {
    out.data = nullptr;
    out.size = 0;
    if (bytes.empty()) return true;
    out.data = static_cast<uint8_t *>(std::malloc(bytes.size()));
    if (out.data == nullptr) return false;
    std::memcpy(out.data, bytes.data(), bytes.size());
    out.size = bytes.size();
    return true;
}

kokopop::StreamMode c_synthesis_mode(int32_t mode, bool & ok) {
    ok = true;
    if (mode == KOKOPOP_SYNTH_ADAPTATIVE) return kokopop::StreamMode::Adaptative;
    if (mode == KOKOPOP_SYNTH_LONG_FORM) return kokopop::StreamMode::LongForm;
    ok = false;
    return kokopop::StreamMode::Adaptative;
}

kokopop::EncodedAudioFormat c_audio_format(int32_t format, bool & ok) {
    ok = true;
    if (format == KOKOPOP_AUDIO_PCM_F32LE) return kokopop::EncodedAudioFormat::PcmF32Le;
    if (format == KOKOPOP_AUDIO_WAV_PCM16) return kokopop::EncodedAudioFormat::WavPcm16;
    if (format == KOKOPOP_AUDIO_OGG_OPUS) return kokopop::EncodedAudioFormat::OggOpus;
    ok = false;
    return kokopop::EncodedAudioFormat::PcmF32Le;
}

bool fill_chunk_config_override(const kokopop_synthesis_options * options,
                                kokopop::ChunkConfig & cfg) {
    if (options == nullptr) return false;
    bool has = false;
    auto set_pos = [&](int32_t value, int & field) {
        if (value > 0) {
            field = value;
            has = true;
        }
    };
    set_pos(options->target_min_tokens, cfg.target_min_tokens);
    set_pos(options->target_max_tokens, cfg.target_max_tokens);
    set_pos(options->soft_max_tokens, cfg.soft_max_tokens);
    set_pos(options->hard_max_tokens, cfg.hard_max_tokens);
    set_pos(options->first_chunk_target_tokens, cfg.first_chunk_target_max_tokens);
    set_pos(options->target_overshoot_tokens, cfg.target_overshoot_tokens);
    set_pos(options->comma_pause_ms, cfg.comma_pause_ms);
    set_pos(options->sentence_pause_ms, cfg.sentence_pause_ms);
    set_pos(options->paragraph_pause_ms, cfg.paragraph_pause_ms);
    set_pos(options->crossfade_ms, cfg.crossfade_ms);
    set_pos(options->max_silence_trim_ms, cfg.max_silence_trim_ms);
    if (options->trim_silence == 1 || options->trim_silence == -1) {
        cfg.trim_silence = options->trim_silence == 1;
        has = true;
    }
    return has;
}

kokopop::SynthesisSessionOptions make_session_options(
    const kokopop_synthesis_options * options,
    kokopop::StreamMode default_mode = kokopop::StreamMode::Adaptative) {
    kokopop::SynthesisSessionOptions out;
    out.voice = options && options->voice ? options->voice : "";
    out.speed = options && options->speed != 0.0f ? options->speed : 1.0f;
    bool mode_ok = true;
    out.mode = options ? c_synthesis_mode(options->mode, mode_ok) : default_mode;
    out.has_chunk_config = fill_chunk_config_override(options, out.chunk_config);
    return out;
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

    kokopop::SynthesisSessionOptions options;
    options.voice = voice_s;
    options.speed = speed;
    options.mode = kokopop::StreamMode::LongForm;
    options.chunk_config = text_synthesis_config_for_voice(voice_s);
    options.has_chunk_config = true;
    options.use_exact_chunk_config = true;
    kokopop::SynthesisSession session(*model->impl, options);
    if (!session.push_text(text, error) || !session.finish_input(error)) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, error);
    }
    std::vector<float> combined;
    combined.reserve(4096);
    while (!session.done()) {
        std::vector<kokopop::SynthesisAudioChunk> chunks;
        if (!session.next(1, 0, chunks, error)) {
            return fail(KOKOPOP_ERROR_INFERENCE, error);
        }
        for (auto & chunk : chunks) {
            combined.insert(combined.end(), chunk.samples.begin(), chunk.samples.end());
        }
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

int kokopop_synthesis_create(
    kokopop_model * model,
    const kokopop_synthesis_options * options,
    kokopop_synthesis ** out_synthesis) {
    if (out_synthesis == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "out_synthesis is null");
    }
    *out_synthesis = nullptr;
    if (model == nullptr || model->impl == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "model is null");
    }

    bool mode_ok = true;
    if (options != nullptr) {
        (void)c_synthesis_mode(options->mode, mode_ok);
        if (!mode_ok) {
            return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "unknown synthesis mode");
        }
    }

    auto session_options = make_session_options(options);
    auto wrapper = std::unique_ptr<kokopop_synthesis>(new kokopop_synthesis());
    wrapper->impl.reset(new kokopop::SynthesisSession(*model->impl, std::move(session_options)));
    *out_synthesis = wrapper.release();
    return KOKOPOP_OK;
}

int kokopop_synthesis_push_text(kokopop_synthesis * synthesis, const char * text) {
    if (synthesis == nullptr || synthesis->impl == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "synthesis is null");
    }
    if (text == nullptr || kokopop::trim_ascii(text).empty()) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "text is empty");
    }
    std::string error;
    if (!synthesis->impl->push_text(text, error)) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, error);
    }
    return KOKOPOP_OK;
}

int kokopop_synthesis_finish_input(kokopop_synthesis * synthesis) {
    if (synthesis == nullptr || synthesis->impl == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "synthesis is null");
    }
    std::string error;
    if (!synthesis->impl->finish_input(error)) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, error);
    }
    return KOKOPOP_OK;
}

int kokopop_synthesis_next(
    kokopop_synthesis * synthesis,
    size_t max_chunks,
    kokopop_audio_chunk ** out_chunks,
    size_t * out_n_chunks) {
    if (out_chunks == nullptr || out_n_chunks == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "chunk output is null");
    }
    *out_chunks = nullptr;
    *out_n_chunks = 0;
    if (synthesis == nullptr || synthesis->impl == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "synthesis is null");
    }

    std::vector<kokopop::SynthesisAudioChunk> chunks;
    std::string error;
    if (!synthesis->impl->next(max_chunks, 0, chunks, error)) {
        return fail(KOKOPOP_ERROR_INFERENCE, error);
    }
    if (chunks.empty()) return KOKOPOP_OK;

    auto * out = static_cast<kokopop_audio_chunk *>(
        std::calloc(chunks.size(), sizeof(kokopop_audio_chunk)));
    if (out == nullptr) {
        return fail(KOKOPOP_ERROR_INFERENCE, "failed to allocate output chunks");
    }

    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto & src = chunks[i];
        out[i].n_samples = src.samples.size();
        out[i].sample_rate = synthesis->impl->sample_rate();
        out[i].chunk_index = src.chunk_index;
        out[i].is_final = src.is_final ? 1 : 0;
        if (!src.samples.empty()) {
            out[i].samples = static_cast<float *>(
                std::malloc(src.samples.size() * sizeof(float)));
            if (out[i].samples == nullptr) {
                kokopop_audio_chunks_free(out, chunks.size());
                return fail(KOKOPOP_ERROR_INFERENCE, "failed to allocate output audio chunk");
            }
            std::memcpy(out[i].samples, src.samples.data(), src.samples.size() * sizeof(float));
        }
    }

    *out_chunks = out;
    *out_n_chunks = chunks.size();
    return KOKOPOP_OK;
}

void kokopop_audio_chunks_free(kokopop_audio_chunk * chunks, size_t n_chunks) {
    if (chunks == nullptr) return;
    for (size_t i = 0; i < n_chunks; ++i) {
        std::free(chunks[i].samples);
        chunks[i].samples = nullptr;
        chunks[i].n_samples = 0;
    }
    std::free(chunks);
}

void kokopop_synthesis_free(kokopop_synthesis * synthesis) {
    delete synthesis;
}

int kokopop_audio_encoder_create(
    const kokopop_encoder_options * options,
    kokopop_audio_encoder ** out_encoder) {
    if (out_encoder == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "out_encoder is null");
    }
    *out_encoder = nullptr;

    bool format_ok = true;
    const int32_t c_format = options ? options->format : KOKOPOP_AUDIO_PCM_F32LE;
    kokopop::EncodedAudioFormat format = c_audio_format(c_format, format_ok);
    if (!format_ok) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "unknown audio format");
    }
    if (!kokopop::AudioEncoder::available(format)) {
        return fail(KOKOPOP_ERROR_IO, "Ogg/Opus output is not available in this build");
    }

    kokopop::AudioEncoderOptions encoder_options;
    encoder_options.format = format;
    encoder_options.sample_rate = options && options->sample_rate > 0 ? options->sample_rate : 24000;
    encoder_options.ogg_prebuffer_chunks =
        options ? std::max<int32_t>(0, options->ogg_prebuffer_chunks) : 0;

    auto wrapper = std::unique_ptr<kokopop_audio_encoder>(new kokopop_audio_encoder());
    wrapper->impl.reset(new kokopop::AudioEncoder(encoder_options));
    *out_encoder = wrapper.release();
    return KOKOPOP_OK;
}

int kokopop_audio_encoder_start(kokopop_audio_encoder * encoder, kokopop_bytes * out_bytes) {
    if (out_bytes == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "out_bytes is null");
    }
    out_bytes->data = nullptr;
    out_bytes->size = 0;
    if (encoder == nullptr || encoder->impl == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "encoder is null");
    }
    std::vector<uint8_t> bytes;
    std::string error;
    if (!encoder->impl->start(bytes, error)) {
        return fail(KOKOPOP_ERROR_IO, error);
    }
    if (!allocate_bytes_from_vector(bytes, *out_bytes)) {
        return fail(KOKOPOP_ERROR_IO, "failed to allocate encoded bytes");
    }
    return KOKOPOP_OK;
}

int kokopop_audio_encoder_push(
    kokopop_audio_encoder * encoder,
    const float * samples,
    size_t n_samples,
    int32_t is_final,
    kokopop_bytes * out_bytes) {
    if (out_bytes == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "out_bytes is null");
    }
    out_bytes->data = nullptr;
    out_bytes->size = 0;
    if (encoder == nullptr || encoder->impl == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "encoder is null");
    }
    std::vector<uint8_t> bytes;
    std::string error;
    if (!encoder->impl->push(samples, n_samples, is_final != 0, bytes, error)) {
        return fail(KOKOPOP_ERROR_IO, error);
    }
    if (!allocate_bytes_from_vector(bytes, *out_bytes)) {
        return fail(KOKOPOP_ERROR_IO, "failed to allocate encoded bytes");
    }
    return KOKOPOP_OK;
}

int kokopop_audio_encoder_finish(
    kokopop_audio_encoder * encoder,
    int32_t success,
    kokopop_bytes * out_bytes) {
    if (out_bytes == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "out_bytes is null");
    }
    out_bytes->data = nullptr;
    out_bytes->size = 0;
    if (encoder == nullptr || encoder->impl == nullptr) {
        return fail(KOKOPOP_ERROR_INVALID_ARGUMENT, "encoder is null");
    }
    std::vector<uint8_t> bytes;
    std::string error;
    if (!encoder->impl->finish(success != 0, bytes, error)) {
        return fail(KOKOPOP_ERROR_IO, error);
    }
    if (!allocate_bytes_from_vector(bytes, *out_bytes)) {
        return fail(KOKOPOP_ERROR_IO, "failed to allocate encoded bytes");
    }
    return KOKOPOP_OK;
}

void kokopop_bytes_free(kokopop_bytes * bytes) {
    if (bytes == nullptr) return;
    std::free(bytes->data);
    bytes->data = nullptr;
    bytes->size = 0;
}

void kokopop_audio_encoder_free(kokopop_audio_encoder * encoder) {
    delete encoder;
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
