#include "kokopop.h"

#include "core/error.h"
#include "inference/kokoro.h"
#include "model/model.h"
#include "synthesis/phonemizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model = "models/kokoro-md.gguf";
    std::string voice = "zf_xiaoni";
    std::string text;
    std::string phonemes;
};

void print_usage() {
    std::fprintf(stderr,
        "usage: kokopop_probe --voice VOICE [--model PATH] (--text TEXT | --phonemes PS)\n");
}

bool parse_args(int argc, char ** argv, Options & options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char * name, std::string & out) -> bool {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return false;
            }
            out = argv[++i];
            return true;
        };

        if (arg == "--model") {
            if (!need_value("--model", options.model)) return false;
        } else if (arg == "--voice") {
            if (!need_value("--voice", options.voice)) return false;
        } else if (arg == "--text") {
            if (!need_value("--text", options.text)) return false;
        } else if (arg == "--phonemes") {
            if (!need_value("--phonemes", options.phonemes)) return false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    const bool has_text = !options.text.empty();
    const bool has_phonemes = !options.phonemes.empty();
    if (has_text == has_phonemes) {
        std::fprintf(stderr, "exactly one of --text or --phonemes is required\n");
        return false;
    }
    return true;
}

struct Stats {
    double mean = 0.0;
    double rms = 0.0;
    double peak = 0.0;
};

Stats stats(const std::vector<float> & values) {
    Stats s{};
    if (values.empty()) {
        return s;
    }
    double sum = 0.0;
    double sq = 0.0;
    double peak = 0.0;
    for (float value : values) {
        const double v = static_cast<double>(value);
        sum += v;
        sq += v * v;
        peak = std::max(peak, std::fabs(v));
    }
    s.mean = sum / static_cast<double>(values.size());
    s.rms = std::sqrt(sq / static_cast<double>(values.size()));
    s.peak = peak;
    return s;
}

void print_float_list(const char * key, const std::vector<float> & values) {
    std::printf("%s=", key);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::printf(",");
        }
        std::printf("%.6f", static_cast<double>(values[i]));
    }
    std::printf("\n");
}

void print_int_list(const char * key, const std::vector<int> & values) {
    std::printf("%s=", key);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::printf(",");
        }
        std::printf("%d", values[i]);
    }
    std::printf("\n");
}

} // namespace

int main(int argc, char ** argv) {
    Options options;
    if (!parse_args(argc, argv, options)) {
        print_usage();
        return 2;
    }

    kokopop_model_options model_options{};
    model_options.n_threads = 4;
    model_options.backend = KOKOPOP_BACKEND_CPU;

    kokopop_model * handle = nullptr;
    const int rc = kokopop_model_load(options.model.c_str(), &model_options, &handle);
    if (rc != KOKOPOP_OK || handle == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", kokopop_last_error());
        return 1;
    }

    std::unique_ptr<kokopop_model, void(*)(kokopop_model *)> guard(handle, kokopop_model_free);
    kokopop::Model * model = kokopop_model_get_impl(handle);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to access model implementation\n");
        return 1;
    }

    std::string phonemes = options.phonemes;
    std::string error;
    if (phonemes.empty()) {
        if (!kokopop::phonemize_text(options.text, options.voice, phonemes, error)) {
            std::fprintf(stderr, "phonemize_text failed: %s\n", error.c_str());
            return 1;
        }
    }

    std::vector<uint32_t> ids;
    if (!model->tokenize_phonemes(phonemes, ids, error)) {
        std::fprintf(stderr, "tokenize_phonemes failed: %s\n", error.c_str());
        return 1;
    }

    kokopop::KokoroFrontendProbe frontend;
    if (!kokopop::run_kokoro_frontend_probe(*model, ids, options.voice, frontend, error)) {
        std::fprintf(stderr, "run_kokoro_frontend_probe failed: %s\n", error.c_str());
        return 1;
    }

    kokopop::KokoroGenerationProbe gen;
    if (!kokopop::run_kokoro_generation_probe(*model, ids, options.voice, 1.0f, frontend, gen, error)) {
        std::fprintf(stderr, "run_kokoro_generation_probe failed: %s\n", error.c_str());
        return 1;
    }

    std::vector<int> rounded_durations;
    rounded_durations.reserve(frontend.durations.size());
    for (float duration : frontend.durations) {
        if (!std::isfinite(duration)) {
            rounded_durations.push_back(1);
        } else {
            rounded_durations.push_back(std::clamp(static_cast<int>(std::lrint(duration)), 1, 50));
        }
    }

    const Stats f0_stats = stats(gen.f0);
    const Stats noise_stats = stats(gen.noise);
    const Stats asr_stats = stats(gen.asr);
    const Stats decoder_stats = stats(gen.decoder);
    const Stats audio_stats = stats(gen.audio);
    const int rounded_frames = std::accumulate(rounded_durations.begin(), rounded_durations.end(), 0);

    std::printf("voice=%s\n", options.voice.c_str());
    std::printf("phonemes=%s\n", phonemes.c_str());
    std::printf("token_count=%zu\n", ids.size());
    std::printf("frontend_hidden_dim=%lld\n", static_cast<long long>(frontend.hidden_dim));
    std::printf("generation_frames=%lld\n", static_cast<long long>(gen.n_frames));
    std::printf("rounded_frames=%d\n", rounded_frames);
    std::printf("audio_samples=%zu\n", gen.audio.size());
    print_float_list("frontend_durations", frontend.durations);
    print_int_list("rounded_durations", rounded_durations);
    std::printf("f0_mean=%.6f\n", f0_stats.mean);
    std::printf("f0_rms=%.6f\n", f0_stats.rms);
    std::printf("noise_mean=%.6f\n", noise_stats.mean);
    std::printf("noise_rms=%.6f\n", noise_stats.rms);
    std::printf("asr_mean=%.6f\n", asr_stats.mean);
    std::printf("asr_rms=%.6f\n", asr_stats.rms);
    std::printf("decoder_mean=%.6f\n", decoder_stats.mean);
    std::printf("decoder_rms=%.6f\n", decoder_stats.rms);
    std::printf("audio_mean=%.6f\n", audio_stats.mean);
    std::printf("audio_rms=%.6f\n", audio_stats.rms);
    std::printf("audio_peak=%.6f\n", audio_stats.peak);
    return 0;
}
