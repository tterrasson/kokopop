#include "kokopop.h"

#include "core/backend_names.h"
#include "core/parse_integer.h"
#include "playback/playback.h"
#include "playback/playback_dummy.h"

#if defined(__APPLE__)
#include "playback/playback_coreaudio.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model kokoro.gguf --voice ff_siwis (--text TEXT | --phonemes PHONEMES) "
        "[--out out.wav | --play] [--speed 1.0] [--threads N] [--backend %s] [--seed N]\n"
        "\n"
        "  --out out.wav   Write audio to a WAV file\n"
        "  --play          Play audio directly (mutually exclusive with --out)\n"
        "  --backend       Inference backend (default: auto)\n"
        "  --seed N        sanoTTS noise seed; default: derived from the voice.\n"
        "                  Ignored with --phonemes and on a Kokoro model.\n",
        argv0,
        kokopop::backend_name_list());
}

const char * arg_value(int & i, int argc, char ** argv) {
    if (i + 1 >= argc) {
        return nullptr;
    }
    ++i;
    return argv[i];
}

/// Long-form synthesis through the session API, concatenated into one buffer.
/// Only used when the caller pinned a noise seed.
int synthesize_text_with_seed(kokopop_model * model, const std::string & text,
                              const std::string & voice, float speed,
                              uint64_t seed, kokopop_audio & out) {
    kokopop_synthesis_options options{};
    options.voice = voice.c_str();
    options.speed = speed;
    options.mode = KOKOPOP_SYNTH_LONG_FORM;
    options.has_sano_noise_seed = 1;
    options.sano_noise_seed = seed;

    kokopop_synthesis * synthesis = nullptr;
    int rc = kokopop_synthesis_create(model, &options, &synthesis);
    if (rc != KOKOPOP_OK) return rc;

    std::vector<float> samples;
    int32_t sample_rate = 0;
    rc = kokopop_synthesis_push_text(synthesis, text.c_str());
    if (rc == KOKOPOP_OK) rc = kokopop_synthesis_finish_input(synthesis);
    while (rc == KOKOPOP_OK) {
        kokopop_audio_chunk * chunks = nullptr;
        size_t n_chunks = 0;
        rc = kokopop_synthesis_next(synthesis, 1, &chunks, &n_chunks);
        if (rc != KOKOPOP_OK || n_chunks == 0) {
            kokopop_audio_chunks_free(chunks, n_chunks);
            break;
        }
        bool final_chunk = false;
        for (size_t i = 0; i < n_chunks; ++i) {
            sample_rate = chunks[i].sample_rate;
            samples.insert(samples.end(), chunks[i].samples,
                           chunks[i].samples + chunks[i].n_samples);
            final_chunk = final_chunk || chunks[i].is_final != 0;
        }
        kokopop_audio_chunks_free(chunks, n_chunks);
        if (final_chunk) break;
    }
    kokopop_synthesis_free(synthesis);
    if (rc != KOKOPOP_OK) return rc;
    if (samples.empty()) {
        std::fprintf(stderr, "synthesis produced no audio\n");
        return KOKOPOP_ERROR_INFERENCE;
    }

    out.samples = static_cast<float *>(std::malloc(samples.size() * sizeof(float)));
    if (out.samples == nullptr) return KOKOPOP_ERROR_INFERENCE;
    std::memcpy(out.samples, samples.data(), samples.size() * sizeof(float));
    out.n_samples = samples.size();
    out.sample_rate = sample_rate;
    return KOKOPOP_OK;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string text;
    std::string phonemes;
    std::string voice;
    std::string out_path;
    bool play = false;
    float speed = 1.0f;
    int threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
    int32_t backend = KOKOPOP_BACKEND_AUTO;
    bool has_seed = false;
    uint64_t seed = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return 2; }
            model_path = v;
        } else if (std::strcmp(argv[i], "--text") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return 2; }
            text = v;
        } else if (std::strcmp(argv[i], "--phonemes") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return 2; }
            phonemes = v;
        } else if (std::strcmp(argv[i], "--voice") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return 2; }
            voice = v;
        } else if (std::strcmp(argv[i], "--out") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return 2; }
            if (play) {
                std::fprintf(stderr, "error: --out and --play are mutually exclusive\n");
                return 2;
            }
            out_path = v;
        } else if (std::strcmp(argv[i], "--play") == 0) {
            if (!out_path.empty()) {
                std::fprintf(stderr, "error: --out and --play are mutually exclusive\n");
                return 2;
            }
            play = true;
        } else if (std::strcmp(argv[i], "--speed") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return 2; }
            speed = std::stof(v);
        } else if (std::strcmp(argv[i], "--threads") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return 2; }
            threads = std::stoi(v);
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return 2; }
            if (!kokopop::parse_u64(v, seed)) {
                std::fprintf(stderr, "error: --seed must be an unsigned 64-bit integer\n");
                return 2;
            }
            has_seed = true;
        } else if (std::strcmp(argv[i], "--backend") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return 2; }
            if (!kokopop::backend_from_name(v, backend)) {
                std::fprintf(stderr, "error: invalid backend '%s' (use %s)\n",
                             v, kokopop::backend_name_list());
                return 2;
            }
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty() || (out_path.empty() && !play) || (text.empty() == phonemes.empty())) {
        usage(argv[0]);
        return 2;
    }

    kokopop_model_options options{};
    options.n_threads = threads;
    options.backend = backend;
    kokopop_model * model = nullptr;
    int rc = kokopop_model_load(model_path.c_str(), &options, &model);
    if (rc != KOKOPOP_OK) {
        std::fprintf(stderr, "kokopop_model_load: %s\n", kokopop_last_error());
        return rc;
    }

    // The seeded session path does not reproduce the per-language chunk config
    // `kokopop_synthesize_text()` applies (Mandarin Kokoro voices get wider
    // chunks and different pauses). Taking it for a model that ignores the
    // seed anyway would change the chunking, the pauses and the audio for
    // nothing, so it is reserved for the architecture the seed belongs to.
    const bool seeded = has_seed && !text.empty() &&
                        kokopop_model_arch(model) == KOKOPOP_ARCH_SANOTTS;
    if (has_seed && !seeded) {
        std::fprintf(stderr,
                     "warning: --seed applies to sanoTTS voices only, ignoring it\n");
    }

    kokopop_audio audio{};
    if (seeded) {
        rc = synthesize_text_with_seed(model, text, voice, speed, seed, audio);
    } else if (!text.empty()) {
        rc = kokopop_synthesize_text(model, text.c_str(), voice.c_str(), speed, &audio);
    } else {
        rc = kokopop_synthesize_phonemes(model, phonemes.c_str(), voice.c_str(), speed, &audio);
    }
    if (rc != KOKOPOP_OK) {
        std::fprintf(stderr, "synthesis: %s\n", kokopop_last_error());
        kokopop_model_free(model);
        return rc;
    }

    if (play) {
        // Play audio directly using the same playback mechanism as kokopop_play
        kokopop::AudioPlayback * playback = nullptr;

#if defined(__APPLE__)
        playback = new kokopop::CoreAudioPlayback();
#else
        playback = new kokopop::DummyPlayback();
#endif

        if (playback->start(audio.sample_rate)) {
            playback->write(audio.samples, audio.n_samples);
            playback->stop();
            playback->wait();
        } else {
            std::fprintf(stderr, "Failed to start playback\n");
            kokopop_audio_free(&audio);
            kokopop_model_free(model);
            return 3;
        }
        kokopop::free_playback(playback);
    } else {
        rc = kokopop_write_wav(out_path.c_str(), &audio);
        if (rc != KOKOPOP_OK) {
            std::fprintf(stderr, "kokopop_write_wav: %s\n", kokopop_last_error());
        }
    }

    kokopop_audio_free(&audio);
    kokopop_model_free(model);
    return rc;
}
