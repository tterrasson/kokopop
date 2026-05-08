#include "kokopop.h"
#include "model/model.h"
#include "streaming/streaming.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Built-in test sentences — varied length, phoneme patterns, and punctuation
// ============================================================================
namespace {

constexpr std::array<const char *, 12> kSentences = {
    "The quick brown fox jumps over the lazy dog.",
    "In a world where technology evolves faster than we can adapt, we must learn to embrace change.",
    "She asked, 'Is this the real life? Is this just fantasy?'",
    "Hello world, this is a simple test of the text-to-speech synthesis engine.",
    "The temperature outside is currently twenty-three degrees Celsius with a slight chance of rain.",
    "Why did the chicken cross the road? To get to the other side, of course!",
    "Artificial intelligence, pronounced as A-I, has revolutionized many industries in recent years.",
    "Please note that the conference room has been moved to the third floor, room three-forty-two.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
    "Can you repeat that one more time, please?",
    "The project deadline is next Friday, but we might need an extension depending on the review feedback.",
    "Numbers like 42, 3.14159, and one hundred million should be handled correctly by the phonemizer.",
};

} // namespace

// ============================================================================
// Usage & argument parsing
// ============================================================================
namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model PATH [options]\n"
        "\n"
        "Measure real-time (RT) synthesis speed for Kokoro TTS.\n"
        "Generates audio from built-in test sentences and reports\n"
        "per-chunk and overall RT ratios (1.0x = real-time, >1 = faster).\n"
        "\n"
        "Options:\n"
        "  --model PATH        Path to Kokoro GGUF model (required)\n"
        "  --voice NAME        Voice name (default: auto-resolve to first available)\n"
        "  --backend cpu|metal Inference backend (default: auto)\n"
        "  --threads N         Thread count (default: min(4, hw_concurrency))\n"
        "  --n-sentences N     Number of sentences to use (default: 10, range: 3-12)\n"
        "  --speed F           Synthesis speed multiplier (default: 1.0)\n"
        "  --seed N            Random seed for sentence selection (default: 0 = time-based)\n"
        "  --help, -h          Show this help message\n"
        "\n"
        "Examples:\n"
        "  %s --model models/kokoro-md.gguf\n"
        "  %s --model models/kokoro-md.gguf --backend metal --threads 8\n"
        "  %s --model models/kokoro-md.gguf --n-sentences 5 --seed 42\n",
        argv0, argv0, argv0, argv0);
}

const char * arg_value(int & i, int argc, char ** argv) {
    if (i + 1 >= argc) return nullptr;
    ++i;
    return argv[i];
}

struct Options {
    std::string model_path;
    std::string voice = "";   // empty = auto-resolve to first available
    float speed = 1.0f;
    int threads = 0;          // 0 = auto
    int n_sentences = 10;
    int backend = KOKOPOP_BACKEND_AUTO;
    unsigned int seed = 0;
};

bool parse_args(int argc, char ** argv, Options & opts) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.model_path = v;
        } else if (std::strcmp(argv[i], "--voice") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.voice = v;
        } else if (std::strcmp(argv[i], "--speed") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.speed = std::stof(v);
        } else if (std::strcmp(argv[i], "--threads") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.threads = std::stoi(v);
        } else if (std::strcmp(argv[i], "--n-sentences") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.n_sentences = std::stoi(v);
        } else if (std::strcmp(argv[i], "--backend") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            if (std::strcmp(v, "cpu") == 0) {
                opts.backend = KOKOPOP_BACKEND_CPU;
            } else if (std::strcmp(v, "metal") == 0) {
                opts.backend = KOKOPOP_BACKEND_METAL;
            } else {
                std::fprintf(stderr, "error: invalid backend '%s' (use 'cpu' or 'metal')\n", v);
                return false;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.seed = static_cast<unsigned int>(std::stoi(v));
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
            usage(argv[0]);
            return false;
        }
    }

    if (opts.model_path.empty()) {
        std::fprintf(stderr, "error: --model is required\n");
        usage(argv[0]);
        return false;
    }

    return true;
}

} // namespace

// ============================================================================
// Timing helpers
// ============================================================================
namespace {

// ggml_time_us() wrapper — works on all platforms
int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

double ms(double us) { return us / 1000.0; }
double sec(double us) { return us / 1000000.0; }

} // namespace

// ============================================================================
// Chunk timing record
// ============================================================================
namespace {

struct ChunkResult {
    int chunk_index = 0;
    int n_tokens = 0;
    size_t n_samples = 0;
    double gen_time_ms = 0.0;
    double audio_duration_s = 0.0;
    double rt_ratio = 0.0;  // gen_time / audio_duration
};

} // namespace

// ============================================================================
// Sentence selection
// ============================================================================
namespace {

std::string select_sentences(int n_sentences, unsigned int seed) {
    // Clamp n_sentences to valid range
    n_sentences = std::clamp(n_sentences, 3, static_cast<int>(kSentences.size()));

    std::mt19937 rng(seed);

    // Build index shuffling
    std::vector<int> indices(kSentences.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Take first n_sentences and sort for deterministic display (not shuffle order)
    std::vector<int> selected(indices.begin(), indices.begin() + n_sentences);
    std::sort(selected.begin(), selected.end());

    std::ostringstream oss;
    for (int idx : selected) {
        if (oss.tellp() > 0) {
            oss << "\n";
        }
        oss << kSentences[idx];
    }
    return oss.str();
}

} // namespace

// ============================================================================
// Table printing helpers
// ============================================================================
namespace {

void print_header(const char * backend_label, const char * voice_name,
                  int threads, int sample_rate,
                  int n_sentences, int n_chunks, int total_tokens) {
    std::printf("\n");
    std::printf("  Backend:     %s\n", backend_label);
    std::printf("  Voice:       %s\n", voice_name);
    std::printf("  Threads:     %d\n", threads);
    std::printf("  Sample Rate: %d Hz\n", sample_rate);
    std::printf("  Sentences:   %d → %d chunk(s) (%d tokens)\n",
               n_sentences, n_chunks, total_tokens);
    std::printf("\n");
}

// Column widths: 6 | 8 | 12 | 10 | 10 | 10 = 56
void print_table_header() {
    std::printf("  ┌──────┬────────┬────────────┬──────────┬──────────┬──────────┐\n");
    std::printf("  │ Chunk│ Tokens │  Gen Time  │ Duration │    RT    │ Samples  │\n");
    std::printf("  ├──────┼────────┼────────────┼──────────┼──────────┼──────────┤\n");
}

void print_table_row(const ChunkResult & cr) {
    std::printf("  │%6d│%8d│%9.1fms │%9.2fs│%9.2fx│%10zu│\n",
               cr.chunk_index + 1,
               cr.n_tokens,
               cr.gen_time_ms,
               cr.audio_duration_s,
               cr.rt_ratio,
               cr.n_samples);
}

void print_table_footer() {
    std::printf("  └──────┴────────┴────────────┴──────────┴──────────┴──────────┘\n");
}

void print_summary(const std::vector<ChunkResult> & results,
                   double total_gen_ms, double total_audio_s,
                   double overall_rt, double ttfb_ms, int sample_rate) {
    std::printf("\n");
    std::printf("  Total Generation:  %6.1f ms\n", total_gen_ms);
    std::printf("  Total Audio:       %6.2f s  (%7zu samples @ %d Hz)\n",
               total_audio_s,
               std::accumulate(results.begin(), results.end(), size_t{0},
                   [](size_t acc, const ChunkResult & cr) { return acc + cr.n_samples; }),
               sample_rate);
    std::printf("  Overall RT:        %5.2fx\n", overall_rt);
    std::printf("  TTFB (1st chunk):  %6.1f ms\n", ttfb_ms);

    if (total_audio_s <= 0.0) {
        std::printf("  → no audio generated (check voice/model compatibility)\n");
    } else if (overall_rt > 1.0) {
        std::printf("  → %.1fx faster than real-time\n", overall_rt);
    } else if (overall_rt < 1.0) {
        std::printf("  → %.1fx slower than real-time\n", 1.0 / overall_rt);
    } else {
        std::printf("  → exactly real-time\n");
    }
    std::printf("\n");
}

// Backend label
const char * backend_label(int backend_type) {
    if (backend_type == KOKOPOP_BACKEND_METAL) return "Metal";
    return "CPU";
}

} // namespace

// ============================================================================
// Main
// ============================================================================
int main(int argc, char ** argv) {
    Options opts;
    if (!parse_args(argc, argv, opts)) {
        return 2;
    }

    // Validate n_sentences
    if (opts.n_sentences < 3 || opts.n_sentences > static_cast<int>(kSentences.size())) {
        std::fprintf(stderr, "error: --n-sentences must be between 3 and %d\n",
                    static_cast<int>(kSentences.size()));
        return 2;
    }

    // Auto-detect threads
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (opts.threads <= 0) {
        opts.threads = std::min(4, hw > 0 ? hw : 1);
    }

    // ---- Load model ----
    kokopop_model_options model_opts{};
    model_opts.n_threads = opts.threads;
    model_opts.backend = opts.backend;

    kokopop_model * model_handle = nullptr;
    {
        std::fprintf(stderr, "[kokopop_rt] Loading model: %s\n", opts.model_path.c_str());
        const int rc = kokopop_model_load(opts.model_path.c_str(), &model_opts, &model_handle);
        if (rc != KOKOPOP_OK) {
            std::fprintf(stderr, "[kokopop_rt] model load failed: %s\n", kokopop_last_error());
            return 1;
        }
    }

    auto model_guard = std::unique_ptr<kokopop_model, void(*)(kokopop_model *)>(
        model_handle, kokopop_model_free);

    kokopop::Model * model = kokopop_model_get_impl(model_handle);
    if (!model) {
        std::fprintf(stderr, "[kokopop_rt] failed to get model implementation\n");
        return 1;
    }

    const int sample_rate = model->sample_rate;
    const char * backend_name = backend_label(model->backend_type);

    // ---- Resolve voice (auto-select first available if not specified) ----
    std::string voice = kokopop::resolve_voice_name(opts.voice, model->voices);
    if (opts.voice.empty()) {
        std::fprintf(stderr, "[kokopop_rt] Auto-selected voice: %s\n", voice.c_str());
    }

    // ---- Select sentences ----
    const std::string text = select_sentences(opts.n_sentences, opts.seed);

    // ---- Prepare synthesis plan (Phase 1) ----
    std::string error;
    const auto t_prepare_start = now_us();
    auto plan = kokopop::prepare_synthesis(
        *model, text, voice, opts.speed,
        kokopop::StreamMode::Interactive, error);
    const double t_prepare_ms = ms(now_us() - t_prepare_start);

    if (plan.chunks.empty()) {
        std::fprintf(stderr, "[kokopop_rt] prepare_synthesis failed: %s\n", error.c_str());
        return 1;
    }

    const int n_chunks = static_cast<int>(plan.chunks.size());
    int total_tokens = 0;
    for (const auto & chunk : plan.chunks) {
        total_tokens += chunk.n_tokens;
    }

    // ---- Print header ----
    print_header(backend_name, voice.c_str(), opts.threads, sample_rate,
                opts.n_sentences, n_chunks, total_tokens);
    std::printf("  Prepare time:    %6.1f ms (chunking + phonemization)\n\n", t_prepare_ms);

    // ---- Infer each chunk (Phase 2) with timing ----
    print_table_header();

    std::vector<ChunkResult> results;
    results.reserve(n_chunks);

    double ttfb_ms = 0.0;  // Time to first byte (first chunk gen time)
    double total_gen_ms = 0.0;
    double total_audio_s = 0.0;

    std::vector<float> prev_tail;

    for (int i = 0; i < n_chunks; ++i) {
        const auto t_chunk_start = now_us();

        std::vector<float> out_tail;
        auto audio = kokopop::infer_chunk(
            *model, plan, i, prev_tail, out_tail, error);

        const double t_chunk_ms = ms(now_us() - t_chunk_start);

        if (audio.empty()) {
            std::fprintf(stderr, "\n  [WARN] chunk[%d] failed: %s — skipped\n", i + 1, error.c_str());
            continue;
        }

        const double audio_dur_s = static_cast<double>(audio.size()) / sample_rate;
        const double gen_sec = t_chunk_ms / 1000.0;
        const double rt_ratio = (gen_sec > 0.0) ? (audio_dur_s / gen_sec) : 0.0;

        ChunkResult cr;
        cr.chunk_index = i;
        cr.n_tokens = plan.chunks[i].n_tokens;
        cr.n_samples = audio.size();
        cr.gen_time_ms = t_chunk_ms;
        cr.audio_duration_s = audio_dur_s;
        cr.rt_ratio = rt_ratio;

        if (i == 0) {
            ttfb_ms = t_chunk_ms;
        }

        total_gen_ms += t_chunk_ms;
        total_audio_s += audio_dur_s;

        results.push_back(std::move(cr));
        prev_tail = std::move(out_tail);

        print_table_row(results.back());
    }

    print_table_footer();

    // ---- Summary ----
    const double overall_rt = (total_gen_ms > 0.0)
        ? (total_audio_s / (total_gen_ms / 1000.0))
        : 0.0;

    print_summary(results, total_gen_ms, total_audio_s, overall_rt, ttfb_ms, sample_rate);

    // ---- Suppress streaming stderr noise (already printed during infer) ----
    // The streaming module prints to stderr; we leave that visible for debugging.

    return 0;
}
