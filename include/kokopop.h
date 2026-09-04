#ifndef KOKOPOP_H
#define KOKOPOP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(KOKOPOP_BUILD_SHARED)
#  ifdef KOKOPOP_BUILDING_LIBRARY
#    define KOKOPOP_API __declspec(dllexport)
#  else
#    define KOKOPOP_API __declspec(dllimport)
#  endif
#else
#  define KOKOPOP_API
#endif

typedef struct kokopop_model kokopop_model;
typedef struct kokopop_synthesis kokopop_synthesis;
typedef struct kokopop_audio_encoder kokopop_audio_encoder;

enum {
    KOKOPOP_ARCH_UNKNOWN = 0,
    KOKOPOP_ARCH_KOKORO  = 1,
    KOKOPOP_ARCH_SANOTTS = 2
};

enum {
    KOKOPOP_BACKEND_AUTO = 0,
    KOKOPOP_BACKEND_CPU = 1,
    KOKOPOP_BACKEND_METAL = 2,
    KOKOPOP_BACKEND_CUDA = 3,
    KOKOPOP_BACKEND_VULKAN = 4,
    KOKOPOP_BACKEND_OPENCL = 5
};

typedef struct kokopop_model_options {
    int32_t n_threads;
    int32_t backend;
} kokopop_model_options;

typedef struct kokopop_audio {
    float * samples;
    size_t n_samples;
    int32_t sample_rate;
} kokopop_audio;

typedef struct kokopop_audio_chunk {
    float * samples;
    size_t n_samples;
    int32_t sample_rate;
    int32_t chunk_index;
    int32_t is_final;
} kokopop_audio_chunk;

typedef struct kokopop_bytes {
    uint8_t * data;
    size_t size;
} kokopop_bytes;

enum {
    KOKOPOP_SYNTH_ADAPTATIVE = 0,
    KOKOPOP_SYNTH_LONG_FORM = 1
};

enum {
    KOKOPOP_AUDIO_PCM_F32LE = 0,
    KOKOPOP_AUDIO_WAV_PCM16 = 1,
    KOKOPOP_AUDIO_OGG_OPUS = 2
};

typedef struct kokopop_synthesis_options {
    const char * voice;
    float speed;
    int32_t mode;

    // Optional chunk config overrides. Leave all fields at 0 to use the mode preset.
    int32_t target_min_tokens;
    int32_t target_max_tokens;
    int32_t soft_max_tokens;
    int32_t hard_max_tokens;
    int32_t first_chunk_target_tokens;
    int32_t target_overshoot_tokens;
    int32_t comma_pause_ms;
    int32_t sentence_pause_ms;
    int32_t paragraph_pause_ms;
    int32_t crossfade_ms;
    int32_t max_silence_trim_ms;
    int32_t trim_silence; // 0 = preset, 1 = true, -1 = false

    // Optional diffusion style sampling. Disabled by default.
    // Requires a GGUF converted with kokopop.diffusion.* tensors.
    int32_t enable_diffusion;
    uint32_t diffusion_seed;
    int32_t diffusion_steps;
    float diffusion_alpha;
    float diffusion_beta;
    float diffusion_embedding_scale;

    /// sanoTTS only. `has_sano_noise_seed` is a separate flag so that the
    /// value 0 stays a valid explicit seed rather than meaning "unset".
    /// Ignored by Kokoro voices.
    int32_t  has_sano_noise_seed;
    uint64_t sano_noise_seed;
} kokopop_synthesis_options;

typedef struct kokopop_encoder_options {
    int32_t format;
    int32_t sample_rate;
    int32_t ogg_prebuffer_chunks;
} kokopop_encoder_options;

enum {
    KOKOPOP_OK = 0,
    KOKOPOP_ERROR_INVALID_ARGUMENT = 1,
    KOKOPOP_ERROR_IO = 2,
    KOKOPOP_ERROR_MODEL = 3,
    KOKOPOP_ERROR_PHONEMIZER = 4,
    KOKOPOP_ERROR_INFERENCE = 5
};

KOKOPOP_API int kokopop_model_load(
    const char * path,
    const kokopop_model_options * options,
    kokopop_model ** out_model);

KOKOPOP_API int kokopop_synthesize_text(
    kokopop_model * model,
    const char * text,
    const char * voice,
    float speed,
    kokopop_audio * out_audio);

KOKOPOP_API int kokopop_synthesize_phonemes(
    kokopop_model * model,
    const char * phonemes,
    const char * voice,
    float speed,
    kokopop_audio * out_audio);

KOKOPOP_API int kokopop_write_wav(const char * path, const kokopop_audio * audio);

KOKOPOP_API int kokopop_synthesis_create(
    kokopop_model * model,
    const kokopop_synthesis_options * options,
    kokopop_synthesis ** out_synthesis);

KOKOPOP_API int kokopop_synthesis_push_text(kokopop_synthesis * synthesis, const char * text);
KOKOPOP_API int kokopop_synthesis_finish_input(kokopop_synthesis * synthesis);
KOKOPOP_API int kokopop_synthesis_next(
    kokopop_synthesis * synthesis,
    size_t max_chunks,
    kokopop_audio_chunk ** out_chunks,
    size_t * out_n_chunks);
KOKOPOP_API void kokopop_audio_chunks_free(kokopop_audio_chunk * chunks, size_t n_chunks);
KOKOPOP_API void kokopop_synthesis_free(kokopop_synthesis * synthesis);

KOKOPOP_API int kokopop_audio_encoder_create(
    const kokopop_encoder_options * options,
    kokopop_audio_encoder ** out_encoder);
KOKOPOP_API int kokopop_audio_encoder_start(kokopop_audio_encoder * encoder, kokopop_bytes * out_bytes);
KOKOPOP_API int kokopop_audio_encoder_push(
    kokopop_audio_encoder * encoder,
    const float * samples,
    size_t n_samples,
    int32_t is_final,
    kokopop_bytes * out_bytes);
KOKOPOP_API int kokopop_audio_encoder_finish(
    kokopop_audio_encoder * encoder,
    int32_t success,
    kokopop_bytes * out_bytes);
KOKOPOP_API void kokopop_bytes_free(kokopop_bytes * bytes);
KOKOPOP_API void kokopop_audio_encoder_free(kokopop_audio_encoder * encoder);

KOKOPOP_API void kokopop_audio_free(kokopop_audio * audio);
KOKOPOP_API void kokopop_model_free(kokopop_model * model);
KOKOPOP_API const char * kokopop_last_error(void);

/// Sample rate of the model's default voice, in Hz. Use
/// `kokopop_model_voice_sample_rate()` on a model that mixes rates.
KOKOPOP_API int kokopop_model_sample_rate(kokopop_model * model);

/// Architecture the model was converted for: one of KOKOPOP_ARCH_*.
KOKOPOP_API int32_t kokopop_model_arch(const kokopop_model * model);

/// Same, as the string the converter wrote into `kokopop.arch`
/// ("kokoro-82m", "sanotts"). Never null; "unknown" for a null model.
KOKOPOP_API const char * kokopop_model_arch_name(const kokopop_model * model);

/// Number of voices the model carries.
KOKOPOP_API size_t kokopop_model_voice_count(const kokopop_model * model);

/// Name of voice `i` in file order, or null when `i` is out of range.
/// The pointer stays valid until the model is freed.
KOKOPOP_API const char * kokopop_model_voice_name(const kokopop_model * model, size_t i);

/// Sample rate of one voice, in Hz. A model may mix rates: a sanoTTS pack
/// carries 22050 Hz Piperlite voices next to 24000 Hz Vocos ones. Returns 0
/// when the voice is unknown.
KOKOPOP_API int32_t kokopop_model_voice_sample_rate(const kokopop_model * model,
                                                    const char * voice);

/// Backend the model actually loaded on, once AUTO has been resolved.
/// Returns one of KOKOPOP_BACKEND_*, or KOKOPOP_BACKEND_CPU for a null model.
KOKOPOP_API int32_t kokopop_model_backend(const kokopop_model * model);

#ifdef __cplusplus

/// Get the internal C++ Model pointer from a kokopop_model handle
namespace kokopop {
struct Model;
}
KOKOPOP_API kokopop::Model * kokopop_model_get_impl(kokopop_model * model);

#endif

#ifdef __cplusplus
}
#endif

#endif
