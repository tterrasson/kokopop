#include "kokopop.h"
#include "core/backend_names.h"
#include "model/model.h"
#include "arch/kokoro/kokoro.h"
#include "arch/kokoro/kokoro_arch.h"
#include "arch/sanotts/sano_arch.h"
#include "synthesis/phonemizer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>
#include <ggml.h>
#include <ggml-backend.h>

static void null_log(ggml_log_level, const char *, void *) {}

namespace {

int64_t now_us() {
    return ggml_time_us();
}

// Peak RSS in MB (getrusage returns bytes on macOS, kilobytes on Linux)
double peak_rss_mb() {
    struct rusage r{};
    if (getrusage(RUSAGE_SELF, &r) != 0) {
        return 0.0;
    }
#ifdef __APPLE__
    return static_cast<double>(r.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(r.ru_maxrss) / 1024.0;
#endif
}

std::string find_model(const char * hint) {
    if (hint && hint[0] != '\0') {
        std::ifstream f(hint, std::ios::binary);
        if (f) {
            return hint;
        }
    }
    const char * candidates[] = {
        "models/kokoro-md.gguf",
        "../models/kokoro-md.gguf",
    };
    for (const char * p : candidates) {
        std::ifstream f(p, std::ios::binary);
        if (f) {
            return p;
        }
    }
    return "";
}

// The two architectures do not benchmark on the same scale: a sanoTTS voice is
// a few megabytes against Kokoro's ~160, its chunks are shorter and its voice
// names come from a different table. Sharing one set of defaults would time
// Kokoro-sized work against a model that never sees it, so each gets its own.
struct Preset {
    /// Empty means "the file's default voice".
    const char * voice;
    /// Pinned phoneme string, kept so that historical numbers stay
    /// comparable. Only used when `voice` is the one it was written for;
    /// otherwise `text` goes through the model's frontend.
    const char * phonemes;
    const char * text;
    int repeat;
};

const Preset & preset_for(kokopop::Arch arch) {
    // ~30 tokens each, the length a streaming chunk actually reaches.
    static const Preset kokoro{
        "ff_siwis",
        "bɔ̃ʒˈuʁ kɔmɑ̃ alɛ vu, ʒɛspɛʁ kə vu alɛ bjɛ̃",
        "Bonjour, comment allez-vous, j'espère que vous allez bien.",
        1};
    static const Preset sanotts{
        "",       // a pack mixes decoders and rates; take the file's own default
        nullptr,  // no pinned string: the two frontends do not share a vocabulary
        "The acoustic student should speak clearly without sounding rushed.",
        1};
    return arch == kokopop::Arch::SanoTTS ? sanotts : kokoro;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s [--model PATH] [--backend %s] [--text TEXT | --phonemes IPA] "
        "[--repeat N] [--voice NAME] [--threads N] [--iters N]\n"
        "  --backend %s  force inference backend (default: auto)\n"
        "  --text TEXT   phonemized by the model\'s own frontend\n"
        "  --repeat N    repeat the phoneme string N times (multiplies token count)\n"
        "\n"
        "Voice, text and repeat default to a per-architecture preset read from\n"
        "the loaded file: a sanoTTS pack is ~50x smaller than Kokoro and its\n"
        "reference chunk sizes do not carry over.\n",
        argv0, kokopop::backend_name_list(), kokopop::backend_name_list());
}

struct BenchResult {
    double min_ms = 1e18;
    double max_ms = 0.0;
    double sum_ms = 0.0;
    int    count  = 0;

    void add(double ms) {
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
        ++count;
    }
    double mean_ms() const { return count > 0 ? sum_ms / count : 0.0; }
};

void print_result(const char * label, const BenchResult & r) {
    std::printf("[BENCH] %-28s min=%6.1fms  mean=%6.1fms  max=%6.1fms\n",
        label, r.min_ms, r.mean_ms(), r.max_ms);
}

const char * backend_name(const kokopop::Model & model) {
    return model.backend ? model.backend->label() : "CPU";
}

double bytes_to_mb(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

} // namespace

int main(int argc, char ** argv) {
    ggml_time_init();

    const char * model_hint  = nullptr;
    const char * phonemes    = nullptr;     // null = derive from --text
    const char * text        = nullptr;     // null = the preset\'s
    const char * voice       = nullptr;     // null = the preset\'s
    int          threads     = 0;           // 0 = auto
    int          iters       = 5;
    int          repeat      = 0;           // 0 = the preset\'s
    int32_t      backend     = KOKOPOP_BACKEND_AUTO; // 0 = auto

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_hint = argv[++i];
        } else if (std::strcmp(argv[i], "--phonemes") == 0 && i + 1 < argc) {
            phonemes = argv[++i];
        } else if (std::strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            text = argv[++i];
        } else if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
            if (repeat < 1) repeat = 1;
        } else if (std::strcmp(argv[i], "--voice") == 0 && i + 1 < argc) {
            voice = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            const char * b = argv[++i];
            if (!kokopop::backend_from_name(b, backend)) {
                std::fprintf(stderr, "[BENCH] unknown backend: %s (expected %s)\n",
                             b, kokopop::backend_name_list());
                usage(argv[0]);
                return 2;
            }
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    const std::string model_path = find_model(model_hint);
    if (model_path.empty()) {
        std::fprintf(stderr, "[BENCH] no model found — skipping (use --model PATH)\n");
        return 0;
    }

    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (threads <= 0) {
        threads = std::min(4, hw > 0 ? hw : 1);
    }

    ggml_log_set(null_log, nullptr);

    // --- Model load ---
    BenchResult r_load;
    std::unique_ptr<kokopop::Model> model;
    for (int it = 0; it < iters; ++it) {
        model.reset();
        kokopop_model_options opts{};
        opts.n_threads = threads;
        opts.backend   = backend;
        std::string err;
        const int64_t t0 = now_us();
        if (!kokopop::load_model_from_gguf(model_path, &opts, model, err)) {
            std::fprintf(stderr, "[BENCH] model load failed: %s\n", err.c_str());
            return 1;
        }
        r_load.add(static_cast<double>(now_us() - t0) / 1000.0);
    }

    // The file names the architecture, so anything the caller left unset is
    // filled in from that architecture's preset rather than from Kokoro's.
    const kokopop::Arch arch = model->arch->arch();
    const Preset & preset = preset_for(arch);
    const bool explicit_voice = voice != nullptr;
    if (voice == nullptr) voice = preset.voice;
    if (repeat <= 0)      repeat = preset.repeat;

    const kokopop::VoiceDesc * voice_desc =
        voice[0] != '\0' ? model->arch->find_voice(voice)
                         : model->arch->default_voice();
    bool on_preset_voice = true;
    if (voice_desc == nullptr) {
        if (explicit_voice) {
            std::fprintf(stderr, "[BENCH] voice not found: %s\n", voice);
            return 1;
        }
        // The preset names a voice this file does not carry. Fall back to its
        // own default rather than refusing to benchmark it.
        voice_desc = model->arch->default_voice();
        on_preset_voice = false;
    }
    if (voice_desc == nullptr) {
        std::fprintf(stderr, "[BENCH] model has no voice\n");
        return 1;
    }
    voice = voice_desc->name.c_str();

    // A phoneme string is only meaningful for the frontend and the language
    // that produced it, so the pinned one is only used on the voice it was
    // written for. Anything else goes through the model's own frontend.
    std::string phonemes_owned;
    if (phonemes == nullptr && text == nullptr && on_preset_voice &&
        preset.phonemes != nullptr) {
        phonemes = preset.phonemes;
    }
    if (phonemes == nullptr) {
        std::string err;
        if (!model->arch->phonemize(text != nullptr ? text : preset.text,
                                    *voice_desc, phonemes_owned, err)) {
            std::fprintf(stderr, "[BENCH] phonemize failed: %s\n", err.c_str());
            return 1;
        }
        phonemes = phonemes_owned.c_str();
    }

    std::string phonemes_str = phonemes;
    for (int r = 1; r < repeat; ++r) {
        phonemes_str += ' ';
        phonemes_str += phonemes;
    }
    const char * phonemes_run = phonemes_str.c_str();

    std::printf("[BENCH] model    : %s\n", model_path.c_str());
    std::printf("[BENCH] arch     : %s\n", model->arch->name());
    std::printf("[BENCH] phonemes : %s\n", phonemes_run);
    if (repeat > 1) {
        std::printf("[BENCH] repeat   : %d\n", repeat);
    }
    std::printf("[BENCH] voice    : %s (%d Hz)\n", voice, voice_desc->sample_rate);
    std::printf("[BENCH] threads  : %d (hw=%d)\n", threads, hw);
    std::printf("[BENCH] iters    : %d\n", iters);
    std::printf("[BENCH] backend  : %s\n\n", backend_name(*model));

    print_result("model_load", r_load);

    // --- Tokenize ---
    BenchResult r_tok;
    std::vector<uint32_t> ids;
    for (int it = 0; it < iters; ++it) {
        ids.clear();
        std::string err;
        const int64_t t0 = now_us();
        if (!model->arch->tokenize(phonemes_run, *voice_desc, ids, err)) {
            std::fprintf(stderr, "[BENCH] tokenize failed: %s\n", err.c_str());
            return 1;
        }
        r_tok.add(static_cast<double>(now_us() - t0) / 1000.0);
    }
    std::printf("[BENCH] %-28s n_tokens=%d\n", "tokenize", static_cast<int>(ids.size()));
    print_result("tokenize", r_tok);

    if (arch == kokopop::Arch::SanoTTS) {
        // The four sanoTTS graphs are chained inside one call, so they are
        // timed together rather than as separate probes.
        BenchResult r_pipeline;
        kokopop::SanoProbe probe;
        auto * sano = kokopop::sano_arch(*model);
        for (int it = 0; it < iters; ++it) {
            probe = {};
            std::string err;
            const int64_t t0 = now_us();
            if (!sano->run(ids, *voice_desc, 1.0f, kokopop::SynthesisExtras{}, probe, err)) {
                std::fprintf(stderr, "[BENCH] sanotts pipeline failed: %s\n", err.c_str());
                return 1;
            }
            r_pipeline.add(static_cast<double>(now_us() - t0) / 1000.0);
        }
        std::printf("[BENCH] %-28s n_frames=%lld n_samples=%d\n", "sanotts_pipeline",
            static_cast<long long>(probe.frames), static_cast<int>(probe.audio.size()));
        print_result("sanotts_pipeline", r_pipeline);
    } else {
        // --- Frontend probe ---
        BenchResult r_front;
        kokopop::KokoroFrontendProbe front;
        for (int it = 0; it < iters; ++it) {
            front = {};
            std::string err;
            const int64_t t0 = now_us();
            if (!kokopop::run_kokoro_frontend_probe(*model, ids, voice, front, err)) {
                std::fprintf(stderr, "[BENCH] frontend_probe failed: %s\n", err.c_str());
                return 1;
            }
            r_front.add(static_cast<double>(now_us() - t0) / 1000.0);
        }
        std::printf("[BENCH] %-28s n_frames=%.0f\n", "frontend_probe",
            static_cast<double>(std::accumulate(front.durations.begin(), front.durations.end(), 0.0f)));
        print_result("frontend_probe", r_front);

        // --- Generation probe ---
        BenchResult r_gen;
        kokopop::KokoroGenerationProbe gen;
        for (int it = 0; it < iters; ++it) {
            gen = {};
            std::string err;
            const int64_t t0 = now_us();
            if (!kokopop::run_kokoro_generation_probe(*model, ids, voice, 1.0f, front, gen, err)) {
                std::fprintf(stderr, "[BENCH] generation_probe failed: %s\n", err.c_str());
                return 1;
            }
            r_gen.add(static_cast<double>(now_us() - t0) / 1000.0);
        }
        std::printf("[BENCH] %-28s n_samples=%d\n", "generation_probe", static_cast<int>(gen.audio.size()));
        print_result("generation_probe", r_gen);
    }

    // --- Synthesis (public API) ---
    // cold_synthesis: model_load + first inference together (cold scheduler + galloc).
    // synthesis (steady-state): N inferences on the same loaded model (warm galloc).
    {
        kokopop_model * pub_model = nullptr;
        kokopop_model_options opts{};
        opts.n_threads = threads;
        opts.backend   = backend;

        // Cold start: load + first inference
        const int64_t t_cold0 = now_us();
        if (kokopop_model_load(model_path.c_str(), &opts, &pub_model) != KOKOPOP_OK) {
            std::fprintf(stderr, "[BENCH] kokopop_model_load failed\n");
            return 1;
        }
        {
            kokopop_audio cold_audio{};
            if (kokopop_synthesize_phonemes(pub_model, phonemes_run, voice, 1.0f, &cold_audio) != KOKOPOP_OK) {
                std::fprintf(stderr, "[BENCH] cold synthesis failed: %s\n", kokopop_last_error());
                kokopop_model_free(pub_model);
                return 1;
            }
            kokopop_audio_free(&cold_audio);
        }
        const double cold_ms = static_cast<double>(now_us() - t_cold0) / 1000.0;
        std::printf("[BENCH] %-28s %6.1fms\n", "cold_synthesis (load+infer)", cold_ms);

        // Steady-state: N inferences on the same model (warm scheduler + galloc)
        BenchResult r_synth;
        for (int it = 0; it < iters; ++it) {
            kokopop_audio audio{};
            const int64_t t0 = now_us();
            const int rc = kokopop_synthesize_phonemes(pub_model, phonemes_run, voice, 1.0f, &audio);
            const double ms = static_cast<double>(now_us() - t0) / 1000.0;
            kokopop_audio_free(&audio);
            if (rc != KOKOPOP_OK) {
                std::fprintf(stderr, "[BENCH] synthesis failed: %s\n", kokopop_last_error());
                kokopop_model_free(pub_model);
                return 1;
            }
            r_synth.add(ms);
        }
        kokopop_model_free(pub_model);
        print_result("synthesis (steady-state)", r_synth);
    }

    if (arch == kokopop::Arch::SanoTTS) {
        auto * sano = kokopop::sano_arch(*model);
        std::printf("\n[BENCH] scratch_graph               %.1f MB\n",
                    bytes_to_mb(sano->graph_scratch.capacity()));
    } else {
        auto * kokoro = kokopop::kokoro_arch(*model);
        std::printf("\n[BENCH] scratch_frontend            %.1f MB\n",
                    bytes_to_mb(kokoro->frontend_scratch.capacity()));
        std::printf("[BENCH] scratch_generation          %.1f MB\n",
                    bytes_to_mb(kokoro->generation_scratch.capacity()));
        std::printf("[BENCH] scratch_generator           %.1f MB\n",
                    bytes_to_mb(kokoro->generator_scratch.capacity()));
    }
    std::printf("\n[BENCH] %-28s %.1f MB\n", "peak_rss", peak_rss_mb());

    return 0;
}
