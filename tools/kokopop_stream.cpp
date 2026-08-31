#include "kokopop.h"
#include "yyjson.h"

#include "core/backend_names.h"

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
        "[--voice NAME] [--speed 1.0] [--mode adaptative|long_form] [--threads N] [--backend ", stderr);
    std::fputs(kokopop::backend_name_list(), stderr);
    std::fputs("]\n"
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
        "  --backend       Inference backend (default: auto)\n"
        "  --http          Run in HTTP server mode (async, event-driven)\n"
        "  --port N        HTTP server port (default: 8080)\n"
        "  --bind ADDR     HTTP server bind address (default: 127.0.0.1)\n"
        "  --idle-unload N Unload model after N minutes of inactivity (HTTP mode, default: disabled)\n"
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

static int run_http_mode(const std::string & model_path,
                         const kokopop_model_options & model_options,
                         kokopop_model * initial_model_c,
                         const std::string & default_voice,
                         float speed, kokopop::StreamMode stream_mode,
                         const std::string & bind_addr, int port,
                         int idle_unload_seconds) {
    // Lifecycle state shared between the server callbacks.
    struct LifecycleState {
        std::string              model_path;
        kokopop_model_options    model_options{};
        kokopop_model          * model_c  = nullptr;
        kokopop::Model         * model    = nullptr;
        std::unique_ptr<kokopop::SynthesisScheduler> scheduler;
    };

    auto ls = std::make_shared<LifecycleState>();
    ls->model_path    = model_path;
    ls->model_options = model_options;
    ls->model_c       = initial_model_c;
    ls->model         = kokopop_model_get_impl(initial_model_c);
    ls->scheduler     = std::make_unique<kokopop::SynthesisScheduler>(*ls->model);

    kokopop::AsyncHttpServer server;

    server.set_scheduler(*ls->scheduler);
    server.set_model(ls->model);
    server.set_default_voice(default_voice);
    server.set_default_speed(speed);
    server.set_stream_mode(stream_mode);

    if (idle_unload_seconds > 0) {
        auto reload_fn = [ls, &server]() -> bool {
            kokopop_model * new_model_c = nullptr;
            int rc = kokopop_model_load(ls->model_path.c_str(), &ls->model_options, &new_model_c);
            if (rc != KOKOPOP_OK) {
                std::fprintf(stderr, "[kokopop] Model reload failed: %s\n", kokopop_last_error());
                return false;
            }
            auto * new_model = kokopop_model_get_impl(new_model_c);
            if (!new_model) {
                kokopop_model_free(new_model_c);
                return false;
            }
            ls->model_c   = new_model_c;
            ls->model     = new_model;
            ls->scheduler = std::make_unique<kokopop::SynthesisScheduler>(*ls->model);
            server.set_model(ls->model);
            server.set_scheduler(*ls->scheduler);
            return true;
        };

        auto unload_fn = [ls]() {
            if (ls->scheduler) {
                ls->scheduler->stop();
                ls->scheduler->join();
                ls->scheduler.reset();
            }
            if (ls->model_c) {
                kokopop_model_free(ls->model_c);
                ls->model_c = nullptr;
                ls->model   = nullptr;
            }
            std::fprintf(stderr, "[kokopop] Model unloaded, memory freed\n");
        };

        server.set_idle_unload(idle_unload_seconds, std::move(reload_fn), std::move(unload_fn));
    }

    // /tts is handled in AsyncHttpServer::_dispatch_request (scheduler-based)

    server.route("/health",
        [&ls](kokopop::HttpRequest & req, kokopop::HttpResponse & res) {
            if (!ls->model) {
                res.status_code = 503;
                res.status_text = "Service Unavailable";
                res.set_json_string("{\"status\":\"unloaded\"}");
                return true;
            }
            return handle_health(ls->model, req, res);
        });

    server.route("/voices",
        [&ls](kokopop::HttpRequest & req, kokopop::HttpResponse & res) {
            if (!ls->model) {
                res.status_code = 503;
                res.status_text = "Service Unavailable";
                res.set_json_string("{\"status\":\"unloaded\"}");
                return true;
            }
            return handle_voices(ls->model, req, res);
        });

    server.route_default(handle_not_found);

    server.start(bind_addr, port);
    server.join();

    // Cleanup (if model wasn't already unloaded by idle timeout)
    if (ls->scheduler) {
        ls->scheduler->stop();
        ls->scheduler->join();
    }
    if (ls->model_c) {
        kokopop_model_free(ls->model_c);
    }
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
    int32_t backend = KOKOPOP_BACKEND_AUTO;

    bool http_mode = false;
    int http_port = 8080;
    std::string http_bind = "127.0.0.1";
    int idle_unload_minutes = 0;

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
                if (!kokopop::backend_from_name(v, backend)) {
                    std::fprintf(stderr, "error: invalid backend '%s' (use %s)\n",
                                 v, kokopop::backend_name_list());
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
            } else if (std::strcmp(argv[i], "--idle-unload") == 0) {
                const char * v = arg_value(i, argc, argv);
                if (!v) { usage(argv[0]); return 2; }
                idle_unload_minutes = std::stoi(v);
                if (idle_unload_minutes < 0) {
                    std::fprintf(stderr, "error: --idle-unload must be >= 0\n");
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
        // Transfer model ownership to run_http_mode which manages the lifecycle.
        model_guard.release();
        return run_http_mode(model_path, options, model_c, voice, speed, stream_mode,
                             http_bind, http_port, idle_unload_minutes * 60);
    }

#ifndef _WIN32
    if (isatty(STDIN_FILENO)) {
        std::fprintf(stderr, "[kokopop] Warning: stdin is a terminal. Pipe JSON input or use --http mode.\n");
    }
#endif
    return run_stdio_mode(model, voice, speed, stream_mode, out_path);
}
