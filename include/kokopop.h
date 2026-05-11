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

enum {
    KOKOPOP_BACKEND_AUTO = 0,
    KOKOPOP_BACKEND_CPU = 1,
    KOKOPOP_BACKEND_METAL = 2,
    KOKOPOP_BACKEND_CUDA = 3
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

KOKOPOP_API void kokopop_audio_free(kokopop_audio * audio);
KOKOPOP_API void kokopop_model_free(kokopop_model * model);
KOKOPOP_API const char * kokopop_last_error(void);

/// Get the sample rate of the loaded model
KOKOPOP_API int kokopop_model_sample_rate(kokopop_model * model);

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
