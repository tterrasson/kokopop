// kokopop_bench_adaptative — Validate et mesurer le mode adaptative (TTFB + inter-chunk)
//
// Réutilise les mêmes composants que le serveur async :
//   SynthesisScheduler :: submit() → worker loop → RequestContext::output_queue
// Sans réimplémenter rien à côté.
//
// Usage :
//   kokopop_bench_adaptative --model kokoro.gguf --voice af_heart --text "Hello world!"
//
// Output (stderr) :
//   - TTFB (time to first byte/chunk)
//   - Temps entre chaque chunk
//   - Token count par chunk
//   - RTF global

#include "kokopop.h"

#include "core/backend_names.h"
#include "streaming/streaming.h"
#include "http/synthesis_scheduler.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::fputs("usage: ", stderr);
    std::fputs(argv0, stderr);
    std::fputs(" --model kokoro.gguf --voice NAME --text \"Hello!\"\n"
        "\n"
        "Options:\n"
        "  --model PATH    Path to Kokoro GGUF model\n"
        "  --voice NAME    Voice name\n"
        "  --text STR      Text to synthesize (mutually exclusive with --file)\n"
        "  --file PATH     Read text from file (mutually exclusive with --text)\n"
        "  --threads N     Number of threads (default: min(4, hw_concurrency))\n"
        "  --backend       Inference backend, one of ", stderr);
    std::fputs(kokopop::backend_name_list(), stderr);
    std::fputs(" (default: auto)\n"
        "  --speed FLOAT   Synthesis speed (default: 1.0)\n"
        "  --runs N        Number of benchmark runs (default: 3)\n"
        "\n",
        stderr);
}

const char * arg_value(int & i, int argc, char ** argv) {
    if (i + 1 >= argc) return nullptr;
    ++i;
    return argv[i];
}

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

struct ChunkResult {
    int chunk_index = 0;
    int n_tokens    = 0;
    int n_samples   = 0;
    double audio_ms = 0.0;
};

// Poll the output queue until a chunk is available or timeout.
// Returns true if a chunk was popped.
bool poll_chunk(kokopop::RequestContext & ctx,
                kokopop::RequestContext::AudioChunk & out,
                int timeout_ms) {
    auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        if (ctx.try_pop(out)) return true;
        auto state = ctx.state.load();
        if (state == kokopop::RequestContext::State::DONE ||
            state == kokopop::RequestContext::State::ERROR ||
            state == kokopop::RequestContext::State::CANCELLED) {
            return false;
        }
        if (Clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string voice;
    std::string text;
    std::string file_path;
    int threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
    int32_t backend = KOKOPOP_BACKEND_AUTO;
    float speed = 1.0f;
    int n_runs = 3;

    try {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--model") == 0) {
                model_path = arg_value(i, argc, argv);
            } else if (std::strcmp(argv[i], "--voice") == 0) {
                voice = arg_value(i, argc, argv);
            } else if (std::strcmp(argv[i], "--text") == 0) {
                text = arg_value(i, argc, argv);
            } else if (std::strcmp(argv[i], "--file") == 0) {
                file_path = arg_value(i, argc, argv);
            } else if (std::strcmp(argv[i], "--threads") == 0) {
                threads = std::stoi(arg_value(i, argc, argv));
            } else if (std::strcmp(argv[i], "--backend") == 0) {
                const char * v = arg_value(i, argc, argv);
                if (!kokopop::backend_from_name(v, backend)) {
                    std::fprintf(stderr, "error: invalid backend '%s' (use %s)\n",
                                 v, kokopop::backend_name_list());
                    return 2;
                }
            } else if (std::strcmp(argv[i], "--speed") == 0) {
                speed = std::stof(arg_value(i, argc, argv));
            } else if (std::strcmp(argv[i], "--runs") == 0) {
                n_runs = std::stoi(arg_value(i, argc, argv));
            } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                usage(argv[0]);
                return 0;
            } else {
                usage(argv[0]);
                return 2;
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        usage(argv[0]);
        return 2;
    }

    if (model_path.empty() || voice.empty()) {
        usage(argv[0]);
        return 2;
    }
    if (text.empty() && file_path.empty()) {
        std::fprintf(stderr, "error: --text or --file is required\n");
        usage(argv[0]);
        return 2;
    }
    if (!text.empty() && !file_path.empty()) {
        std::fprintf(stderr, "error: --text and --file are mutually exclusive\n");
        usage(argv[0]);
        return 2;
    }

    // Read text from file if specified
    if (!file_path.empty()) {
        std::ifstream ifs(file_path);
        if (!ifs) {
            std::fprintf(stderr, "error: cannot open file '%s'\n", file_path.c_str());
            return 3;
        }
        std::ostringstream oss;
        oss << ifs.rdbuf();
        text = oss.str();
        if (text.empty()) {
            std::fprintf(stderr, "error: file '%s' is empty\n", file_path.c_str());
            return 3;
        }
        std::fprintf(stderr, "[bench] Loaded %zu chars from '%s'\n", text.size(), file_path.c_str());
    }

    // ---- Load model ----
    std::fprintf(stderr, "[bench] Loading model: %s\n", model_path.c_str());
    kokopop_model_options options{};
    options.n_threads = threads;
    options.backend = backend;
    kokopop_model * model_c = nullptr;
    int rc = kokopop_model_load(model_path.c_str(), &options, &model_c);
    if (rc != KOKOPOP_OK) {
        std::fprintf(stderr, "kokopop_model_load: %s\n", kokopop_last_error());
        return rc;
    }

    auto model_guard = std::unique_ptr<kokopop_model, void(*)(kokopop_model *)>(
        model_c, kokopop_model_free);
    auto * model = kokopop_model_get_impl(model_c);
    if (!model) {
        std::fprintf(stderr, "Failed to get model implementation\n");
        return 4;
    }
    std::fprintf(stderr, "[bench] Model loaded, sample_rate=%d, threads=%d\n\n",
                 model->sample_rate(), threads);

    // ---- Benchmark loop ----
    for (int run = 0; run < n_runs; ++run) {
        std::fprintf(stderr, "========== Run %d/%d ==========\n", run + 1, n_runs);

        // Create scheduler (same as HTTP server)
        kokopop::SynthesisScheduler scheduler(*model);

        // Submit request (same code path as AsyncHttpServer::_dispatch_request)
        auto ctx = scheduler.submit(
            text, voice, speed,
            kokopop::StreamMode::Adaptative,
            kokopop::RequestContext::AudioFormat::PCM,
            0,  // ogg_prebuffer_chunks
            kokopop::ChunkConfig{}, false);

        // Start timing immediately after submit
        auto t_submit = Clock::now();

        std::vector<ChunkResult> results;
        auto t_last = t_submit;

        while (true) {
            kokopop::RequestContext::AudioChunk chunk;
            auto t_poll_start = Clock::now();

            // Poll with generous timeout (prepare + first inference can take time)
            bool got = poll_chunk(*ctx, chunk, 60000);
            auto t_poll_end = Clock::now();

            if (!got) break;

            auto delay = Ms(t_poll_start - t_last);
            auto chunk_result = ChunkResult{
                chunk.chunk_index,
                0,  // tokens unknown from audio alone — scheduler logs them
                static_cast<int>(chunk.samples.size()),
                static_cast<double>(chunk.samples.size()) / model->sample_rate() * 1000.0
            };

            if (results.empty()) {
                double ttbf_ms = Ms(t_poll_end - t_submit).count();
                std::fprintf(stderr,
                    "  Chunk %2d (TTFB) : %6.1f ms  —  %6d samples (%5.1f ms audio)\n",
                    chunk_result.chunk_index, ttbf_ms,
                    chunk_result.n_samples, chunk_result.audio_ms);
            } else {
                double inter_ms = Ms(t_poll_end - t_last).count();
                std::fprintf(stderr,
                    "  Chunk %2d        : %6.1f ms (inter)  —  %6d samples (%5.1f ms audio)\n",
                    chunk_result.chunk_index, inter_ms,
                    chunk_result.n_samples, chunk_result.audio_ms);
            }

            t_last = t_poll_end;
            results.push_back(std::move(chunk_result));
        }

        auto t_end = Clock::now();
        double total_ms = Ms(t_end - t_submit).count();

        // Check final state
        auto state = ctx->state.load();
        if (state == kokopop::RequestContext::State::ERROR) {
            std::fprintf(stderr, "  ERROR: %s\n", ctx->error.c_str());
        } else if (state == kokopop::RequestContext::State::CANCELLED) {
            std::fprintf(stderr, "  CANCELLED\n");
        } else {
            std::fprintf(stderr, "  State: DONE\n");
        }

        // Compute summary
        double total_audio_ms = 0;
        for (auto & r : results) total_audio_ms += r.audio_ms;
        double total_samples = 0;
        for (auto & r : results) total_samples += r.n_samples;

        std::fprintf(stderr,
            "\n  Summary:\n"
            "    Chunks            : %d\n"
            "    Total samples     : %zd\n"
            "    Total audio       : %6.1f ms\n"
            "    Total wall time   : %6.1f ms\n"
            "    RTF               : %.3f\n",
            static_cast<int>(results.size()),
            static_cast<size_t>(total_samples),
            total_audio_ms,
            total_ms,
            total_ms > 0 ? total_ms / total_audio_ms : 0.0);

        // Stop scheduler gracefully
        scheduler.stop();
        scheduler.join();

        std::fprintf(stderr, "\n");
    }

    return 0;
}
