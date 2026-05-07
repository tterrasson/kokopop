#include "kokopop.h"

#if defined(__APPLE__)
#include "playback/playback_coreaudio.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model kokoro.gguf --voice ff_siwis (--text TEXT | --phonemes PHONEMES) "
        "[--out out.wav | --play] [--speed 1.0] [--threads N] [--backend cpu|metal]\n"
        "\n"
        "  --out out.wav   Write audio to a WAV file\n"
        "  --play          Play audio directly (mutually exclusive with --out)\n"
        "  --backend       Use CPU or Metal backend (default: auto)\n",
        argv0);
}

const char * arg_value(int & i, int argc, char ** argv) {
    if (i + 1 >= argc) {
        return nullptr;
    }
    ++i;
    return argv[i];
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
    int backend = KOKOPOP_BACKEND_AUTO;

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
        } else if (std::strcmp(argv[i], "--backend") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return 2; }
            if (std::strcmp(v, "cpu") == 0) {
                backend = KOKOPOP_BACKEND_CPU;
            } else if (std::strcmp(v, "metal") == 0) {
                backend = KOKOPOP_BACKEND_METAL;
            } else {
                std::fprintf(stderr, "error: invalid backend '%s' (use 'cpu' or 'metal')\n", v);
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

    kokopop_audio audio{};
    if (!text.empty()) {
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

