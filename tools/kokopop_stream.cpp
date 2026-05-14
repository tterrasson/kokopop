#include "kokopop.h"
#include "yyjson.h"

#include "streaming/streaming.h"
#include "stdio/stdio_stream.h"
#include "http/async_server.h"
#include "http/synthesis_scheduler.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>

#ifndef _WIN32
#include <unistd.h>
#include <fcntl.h>
#endif

// ============================================================================
// Usage & arg parsing
// ============================================================================

namespace {

void usage(const char * argv0) {
    std::fputs("usage: ", stderr);
    std::fputs(argv0, stderr);
    std::fputs(" --model kokoro.gguf "
        "[--voice NAME] [--speed 1.0] [--mode adaptative|long_form] [--threads N] [--backend cpu|metal|cuda]\n"
        "       [--out out.wav]\n"
        "       [--http] [--port N] [--bind ADDR]\n"
        "\n"
        "stdio mode (default):\n"
        "  --voice is required. Reads JSON commands from stdin, streams raw audio (float32) to stdout.\n"
        "  JSON protocol (one command per line):\n"
        "    {{\"text\": \"Hello world\"}}              >> accumulate text\n"
        "    {{\"text\": \"Hello\", \"flush\": true}}    >> add text and generate\n"
        "    {{\"flush\": true}}                        >> generate all accumulated text\n"
        "    {{\"stop\": true}}                         >> stop streaming\n"
        "\n"
        "HTTP server mode:\n"
        "  kokopop_stream --model m.gguf --http --port 8080\n"
        "    POST /tts  - Synthesize text to audio (voice is required in JSON payload)\n"
        "    GET  /health - Server health check\n"
        "    GET  /voices - List voices embedded in the GGUF model\n"
        "\n"
        "Options:\n"
        "  --model PATH    Path to Kokoro GGUF model\n"
        "  --voice NAME    Voice name for stdio mode (required without --http)\n"
        "  --speed FLOAT   Synthesis speed (default: 1.0)\n"
        "  --mode MODE     adaptative (default) or long_form\n"
        "  --out PATH      Save full audio to WAV file (stdio mode)\n"
        "  --threads N     Number of threads (default: min(4, hw_concurrency))\n"
        "  --backend       Use CPU, Metal, or CUDA backend (default: auto)\n"
        "  --http          Run in HTTP server mode (async, event-driven)\n"
        "  --port N        HTTP server port (default: 8080)\n"
        "  --bind ADDR     HTTP server bind address (default: 127.0.0.1)\n"
        "\n"
        "Examples:\n",
        stderr);
    std::fputs("  # stdio mode\n"
        "  echo '{{\"text\":\"Hello\", \"flush\":true}}' | ", stderr);
    std::fputs(argv0, stderr);
    std::fputs(" --model m.gguf --voice ff_siwis\n"
        "\n"
        "  # HTTP mode\n", stderr);
    std::fputs(argv0, stderr);
    std::fputs(" --model m.gguf --http --port 8080\n"
        "  curl -X POST http://localhost:8080/tts \\\n"
        "    -H 'Content-Type: application/json' \\\n"
        "    -d '{{\"text\":\"Hello world\", \"voice\":\"ff_siwis\"}}'\n", stderr);
}

const char * arg_value(int & i, int argc, char ** argv) {
    if (i + 1 >= argc) return nullptr;
    ++i;
    return argv[i];
}

// ---------------------------------------------------------------------------
// STDIO mode — delegated to stdio_stream module
// ---------------------------------------------------------------------------

static int run_stdio_mode(kokopop::Model * model, const std::string & voice,
                          float speed, kokopop::StreamMode stream_mode,
                          const std::string & out_path) {
    kokopop::StdioStreamer streamer(*model, voice, speed, stream_mode, out_path);
    streamer.run();
    streamer.join();
    return 0;
}

} // namespace

// ============================================================================
// HTTP mode (async, event-driven)
// ============================================================================

namespace {

using kokopop::json_error;

// /tts is handled directly in AsyncHttpServer::_dispatch_request
// (scheduler-based round-robin interleaving)

// ---------------------------------------------------------------------------
// handle_health
// ---------------------------------------------------------------------------

static bool handle_health(kokopop::Model * model, kokopop::HttpRequest & /*req*/, kokopop::HttpResponse & res) {
    std::string json = "{\"status\":\"ready\",\"sample_rate\":" + std::to_string(model->sample_rate) + "}";
    res.set_json_string(json);
    return true; // keep-alive
}

// ---------------------------------------------------------------------------
// handle_voices
// ---------------------------------------------------------------------------

static bool handle_voices(kokopop::Model * model, kokopop::HttpRequest & /*req*/, kokopop::HttpResponse & res) {
    yyjson_mut_doc * doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_val * arr = yyjson_mut_arr(doc);
    for (const auto & kv : model->voices) {
        yyjson_mut_val * obj = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_str(doc, obj, "name", kv.second.name.c_str());
        yyjson_mut_arr_append(arr, obj);
    }
    yyjson_mut_obj_add_val(doc, root, "voices", arr);

    size_t len = 0;
    char * json_str = yyjson_mut_write(doc, 0, &len);
    if (json_str) {
        res.set_json_string(std::string(json_str, len));
        free(json_str);
    }
    yyjson_mut_doc_free(doc);
    return true; // keep-alive
}

// ---------------------------------------------------------------------------
// handle_not_found
// ---------------------------------------------------------------------------

static bool handle_not_found(kokopop::HttpRequest & /*req*/, kokopop::HttpResponse & res) {
    res.status_code = 404;
    res.status_text = "Not Found";
    res.set_json_string(json_error("not_found"));
    return false; // close
}

} // namespace

static int run_http_mode(kokopop::Model * model, const std::string & default_voice,
                         float speed, kokopop::StreamMode stream_mode,
                         const std::string & bind_addr, int port) {
    kokopop::AsyncHttpServer server;
    kokopop::SynthesisScheduler scheduler(*model);

    server.set_scheduler(scheduler);
    server.set_model(model);
    server.set_default_voice(default_voice);
    server.set_default_speed(speed);
    server.set_stream_mode(stream_mode);

    // /tts is handled in AsyncHttpServer::_dispatch_request (scheduler-based)

    server.route("/health",
        [model](kokopop::HttpRequest & req, kokopop::HttpResponse & res) {
            return handle_health(model, req, res);
        });

    server.route("/voices",
        [model](kokopop::HttpRequest & req, kokopop::HttpResponse & res) {
            return handle_voices(model, req, res);
        });

    server.route_default(handle_not_found);

    server.start(bind_addr, port);
    server.join();
    scheduler.stop();
    scheduler.join();
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    std::string model_path;
    std::string voice;
    float speed = 1.0f;
    std::string mode_str = "adaptative";
    std::string out_path;
    int threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
    int backend = KOKOPOP_BACKEND_AUTO;

    bool http_mode = false;
    int http_port = 8080;
    std::string http_bind = "127.0.0.1";

    try {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--model") == 0) {
                const char * v = arg_value(i, argc, argv);
                if (!v) { usage(argv[0]); return 2; }
                model_path = v;
            } else if (std::strcmp(argv[i], "--voice") == 0) {
                const char * v = arg_value(i, argc, argv);
                if (!v) { usage(argv[0]); return 2; }
                voice = v;
            } else if (std::strcmp(argv[i], "--speed") == 0) {
                const char * v = arg_value(i, argc, argv);
                if (!v) { usage(argv[0]); return 2; }
                speed = std::stof(v);
            } else if (std::strcmp(argv[i], "--mode") == 0) {
                const char * v = arg_value(i, argc, argv);
                if (!v) { usage(argv[0]); return 2; }
                mode_str = v;
            } else if (std::strcmp(argv[i], "--out") == 0) {
                const char * v = arg_value(i, argc, argv);
                if (!v) { usage(argv[0]); return 2; }
                out_path = v;
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
                } else if (std::strcmp(v, "cuda") == 0) {
                    backend = KOKOPOP_BACKEND_CUDA;
                } else {
                    std::fprintf(stderr, "error: invalid backend '%s' (use 'cpu', 'metal', or 'cuda')\n", v);
                    return 2;
                }
            } else if (std::strcmp(argv[i], "--http") == 0) {
                http_mode = true;
            } else if (std::strcmp(argv[i], "--port") == 0) {
                const char * v = arg_value(i, argc, argv);
                if (!v) { usage(argv[0]); return 2; }
                http_port = std::stoi(v);
            } else if (std::strcmp(argv[i], "--bind") == 0) {
                const char * v = arg_value(i, argc, argv);
                if (!v) { usage(argv[0]); return 2; }
                http_bind = v;
            } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                usage(argv[0]);
                return 0;
            } else {
                usage(argv[0]);
                return 2;
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: invalid numeric argument: %s\n", e.what());
        usage(argv[0]);
        return 2;
    }

    if (model_path.empty()) {
        usage(argv[0]);
        return 2;
    }
    if (!http_mode && voice.empty()) {
        std::fprintf(stderr, "error: --voice is required for stdio mode\n");
        usage(argv[0]);
        return 2;
    }

    kokopop::StreamMode stream_mode = kokopop::StreamMode::Adaptative;
    if (mode_str == "long_form") {
        stream_mode = kokopop::StreamMode::LongForm;
    } else if (mode_str != "adaptative") {
        std::fprintf(stderr, "Unknown mode: %s (use 'adaptative' or 'long_form')\n", mode_str.c_str());
        return 2;
    }

    std::fprintf(stderr, "[kokopop] Loading model: %s\n", model_path.c_str());
    kokopop_model_options options{};
    options.n_threads = threads;
    options.backend = backend;
    kokopop_model * model_c = nullptr;
    int rc = kokopop_model_load(model_path.c_str(), &options, &model_c);
    if (rc != KOKOPOP_OK) {
        std::fprintf(stderr, "kokopop_model_load: %s\n", kokopop_last_error());
        return rc;
    }

    auto model_guard = std::unique_ptr<kokopop_model, void(*)(kokopop_model *)>(model_c, kokopop_model_free);
    auto * model = kokopop_model_get_impl(model_c);
    if (!model) {
        std::fprintf(stderr, "Failed to get model implementation\n");
        return 4;
    }

    std::fprintf(stderr, "[kokopop] Model loaded, sample_rate=%d, threads=%d\n",
                model->sample_rate, threads);

    if (http_mode) {
        std::fprintf(stderr, "[kokopop] Starting HTTP server (async) on %s:%d\n", http_bind.c_str(), http_port);
        return run_http_mode(model, voice, speed, stream_mode, http_bind, http_port);
    }

#ifndef _WIN32
    if (isatty(STDIN_FILENO)) {
        std::fprintf(stderr, "[kokopop] Warning: stdin is a terminal. Pipe JSON input or use --http mode.\n");
    }
#endif
    return run_stdio_mode(model, voice, speed, stream_mode, out_path);
}
