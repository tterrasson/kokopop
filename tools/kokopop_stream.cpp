#include "kokopop.h"
#include "yyjson.h"

#include "synthesis/chunker/chunker.h"
#include "streaming/streaming.h"
#include "playback/playback.h"
#include "http/http_server.h"

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
#include <condition_variable>

#ifndef _WIN32
#include <unistd.h>
#include <fcntl.h>
#endif

namespace {

void usage(const char * argv0) {
    std::fputs("usage: ", stderr);
    std::fputs(argv0, stderr);
    std::fputs(" --model kokoro.gguf --voice ff_siwis "
        "[--speed 1.0] [--mode interactive|long_form] [--threads N]\n"
        "       [--out out.wav]\n"
        "\n"
        "stdio mode (default):\n"
        "  Reads JSON commands from stdin, streams raw audio (float32) to stdout.\n"
        "  JSON protocol (one command per line):\n"
        "    {{\"text\": \"Hello world\"}}              >> accumulate text\n"
        "    {{\"text\": \"Hello\", \"flush\": true}}    >> add text and generate\n"
        "    {{\"flush\": true}}                        >> generate all accumulated text\n"
        "    {{\"stop\": true}}                         >> stop streaming\n"
        "\n"
        "HTTP server mode:\n"
        "  kokopop_stream --model m.gguf --voice ff_siwis --http --port 8080\n"
        "    POST /tts  - Synthesize text to audio\n"
        "    GET  /health - Server health check\n"
        "    GET  /voices - List available voices (if model exposes them)\n"
        "\n"
        "Options:\n"
        "  --model PATH    Path to Kokoro GGUF model\n"
        "  --voice NAME    Default voice name (e.g., ff_siwis)\n"
        "  --speed FLOAT   Synthesis speed (default: 1.0)\n"
        "  --mode MODE     interactive (default) or long_form\n"
        "  --out PATH      Save full audio to WAV file (stdio mode)\n"
        "  --threads N     Number of threads (default: min(4, hw_concurrency))\n"
        "  --http          Run in HTTP server mode\n"
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
    std::fputs(" --model m.gguf --voice ff_siwis --http --port 8080\n"
        "  curl -X POST http://localhost:8080/tts \\\n"
        "    -H 'Content-Type: application/json' \\\n"
        "    -d '{{\"text\":\"Hello world\"}}'\n", stderr);
}

const char * arg_value(int & i, int argc, char ** argv) {
    if (i + 1 >= argc) return nullptr;
    ++i;
    return argv[i];
}

// ---------------------------------------------------------------------------
// WAV accumulator for --out option
// ---------------------------------------------------------------------------
struct WavAccumulator {
    std::mutex mutex;
    std::string path;
    std::vector<float> samples;
    int sample_rate = 0;

    void add(const float * data, size_t n) {
        std::lock_guard<std::mutex> lock(mutex);
        samples.insert(samples.end(), data, data + n);
    }

    bool save() {
        std::lock_guard<std::mutex> lock(mutex);
        if (samples.empty() || sample_rate <= 0) return true;

        kokopop_audio audio{};
        audio.samples = samples.data();
        audio.n_samples = samples.size();
        audio.sample_rate = sample_rate;

        int rc = kokopop_write_wav(path.c_str(), &audio);
        return rc == KOKOPOP_OK;
    }
};

// ---------------------------------------------------------------------------
// HTTP streaming context — holds per-request audio buffer + socket
// ---------------------------------------------------------------------------
struct HttpStreamContext {
    std::mutex mutex;
    std::vector<char> wav_buffer;
    int sample_rate = 24000;
    bool is_first_chunk = true;
    bool error = false;

    // For WAV output: accumulate raw samples then build WAV on flush
    std::vector<float> samples;
};

// Build a minimal WAV header
static std::vector<char> build_wav_header(int sample_rate, size_t num_samples) {
    size_t data_size = num_samples * sizeof(float); // float32
    size_t total_size = 44 + data_size;

    std::vector<char> header(44);
    // RIFF header
    header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
    // Chunk size (total - 8)
    uint32_t chunk_size = (uint32_t)(total_size - 8);
    std::memcpy(&header[4], &chunk_size, 4);
    // WAVE
    header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
    // fmt subchunk
    header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
    // Subchunk1Size (16 for PCM)
    uint32_t fmt_size = 16;
    std::memcpy(&header[16], &fmt_size, 4);
    // Audio format (3 = IEEE float)
    uint16_t audio_format = 3;
    std::memcpy(&header[20], &audio_format, 2);
    // Num channels
    uint16_t num_channels = 1;
    std::memcpy(&header[22], &num_channels, 2);
    // Sample rate
    uint32_t sr = (uint32_t)sample_rate;
    std::memcpy(&header[24], &sr, 4);
    // Byte rate
    uint32_t byte_rate = sr * sizeof(float);
    std::memcpy(&header[28], &byte_rate, 4);
    // Block align
    uint16_t block_align = sizeof(float);
    std::memcpy(&header[32], &block_align, 2);
    // Bits per sample
    uint16_t bits_per_sample = 32;
    std::memcpy(&header[34], &bits_per_sample, 2);
    // data subchunk
    header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
    uint32_t ds = (uint32_t)data_size;
    std::memcpy(&header[40], &ds, 4);

    return header;
}

static bool http_audio_callback(const float * data, size_t n_samples, int chunk_index, void * user_data) {
    auto * ctx = static_cast<HttpStreamContext *>(user_data);

    std::lock_guard<std::mutex> lock(ctx->mutex);

    // Accumulate samples for WAV
    ctx->samples.insert(ctx->samples.end(), data, data + n_samples);

    std::fprintf(stderr, "[http] audio_chunk[%d]: %zu samples (%.1fms)\n",
                chunk_index, n_samples, (double)n_samples / ctx->sample_rate * 1000.0);

    return true;
}

} // namespace

// ============================================================================
// STDIO mode
// ============================================================================

namespace {

// Global callback state for stdio mode
struct StdioCallbackState {
    bool use_stdout = true;
    WavAccumulator * wav = nullptr;
    int sample_rate = 24000;
    std::atomic<int> chunk_count{0};
};

static bool stdio_audio_callback(const float * data, size_t n_samples, int chunk_index, void * user_data) {
    auto * state = static_cast<StdioCallbackState *>(user_data);

    int count = state->chunk_count.fetch_add(1) + 1;
    std::fprintf(stderr, "[kokopop] audio_chunk[%d]: %zu samples (%.1fms)\n",
                chunk_index, n_samples, (double)n_samples / state->sample_rate * 1000.0);

    // Write raw float32 to stdout
    if (state->use_stdout) {
        fwrite(data, sizeof(float), n_samples, stdout);
        fflush(stdout);
    }

    // WAV accumulation
    if (state->wav) {
        state->wav->add(data, n_samples);
    }

    return true;
}

} // namespace

static int run_stdio_mode(kokopop::Model * model, const std::string & default_voice,
                          float speed, kokopop::StreamMode stream_mode,
                          const std::string & out_path) {
    // Setup WAV accumulator
    WavAccumulator wav_accum;
    wav_accum.path = out_path;
    wav_accum.sample_rate = model->sample_rate;

    // Callback state
    StdioCallbackState state;
    state.use_stdout = true;
    state.wav = out_path.empty() ? nullptr : &wav_accum;
    state.sample_rate = model->sample_rate;

    // Read stdin and process commands
    std::string line;
    std::string text_buffer;
    std::vector<kokopop::StreamHandle> handles;

    while (std::getline(std::cin, line)) {
        // Trim
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        size_t end = line.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) line = line.substr(0, end + 1);

        if (line.empty()) continue;

        // Parse JSON with yyjson
        yyjson_read_err err;
        yyjson_doc * doc = yyjson_read_opts(const_cast<char *>(line.c_str()), line.size(),
                                            YYJSON_READ_ALLOW_TRAILING_COMMAS, nullptr, &err);
        if (!doc) {
            std::fprintf(stderr, "[kokopop] JSON parse error: %s\n", err.msg);
            continue;
        }

        yyjson_val * root = yyjson_doc_get_root(doc);

        // Extract fields using yyjson
        // Note: strings from yyjson are owned by the doc, so copy them before freeing
        std::string text_str = yyjson_get_str(yyjson_obj_get(root, "text"));
        bool flush = yyjson_obj_get(root, "flush")
                     && !yyjson_is_null(yyjson_obj_get(root, "flush"))
                     && yyjson_get_bool(yyjson_obj_get(root, "flush"));
        bool stop = yyjson_obj_get(root, "stop")
                    && !yyjson_is_null(yyjson_obj_get(root, "stop"))
                    && yyjson_get_bool(yyjson_obj_get(root, "stop"));

        yyjson_doc_free(doc);

        // Handle stop
        if (stop) {
            if (!text_buffer.empty() && !text_str.empty()) {
                text_buffer += text_str;
                flush = true;
            } else {
                break;
            }
        }

        // Accumulate text
        if (!text_str.empty()) {
            text_buffer += text_str;
        }

        // Process if flush
        if (flush && !text_buffer.empty()) {
            if (!handles.empty()) {
                std::fprintf(stderr, "[kokopop]   joining previous stream...\n");
                handles.back().join();
                handles.pop_back();
            }

            std::string text_to_process = std::move(text_buffer);
            text_buffer.clear();

            std::fprintf(stderr, "[kokopop] Synthesizing: %zu chars, voice=%s, speed=%.1f, mode=%s\n",
                        text_to_process.size(), default_voice.c_str(), speed,
                        stream_mode == kokopop::StreamMode::LongForm ? "long_form" : "interactive");

            auto handle = kokopop::stream_synthesize(
                *model,
                text_to_process,
                default_voice,
                speed,
                stream_mode,
                stdio_audio_callback,
                &state);

            handles.push_back(std::move(handle));
        }
    }

    // Flush any remaining accumulated text on EOF
    if (!text_buffer.empty()) {
        if (!handles.empty()) {
            handles.back().join();
            handles.pop_back();
        }
        std::fprintf(stderr, "[kokopop] Synthesizing (EOF flush): %zu chars, voice=%s\n",
                    text_buffer.size(), default_voice.c_str());
        auto handle = kokopop::stream_synthesize(
            *model,
            text_buffer,
            default_voice,
            speed,
            stream_mode,
            stdio_audio_callback,
            &state);
        handles.push_back(std::move(handle));
        text_buffer.clear();
    }

    // Wait for all streams to complete
    std::fprintf(stderr, "[kokopop] Waiting for %zu stream(s) to complete...\n", handles.size());
    for (size_t i = 0; i < handles.size(); ++i) {
        std::fprintf(stderr, "[kokopop]   joining stream %zu...\n", i);
        handles[i].join();
        std::fprintf(stderr, "[kokopop]   stream %zu done (total chunks: %d)\n", i, state.chunk_count.load());
    }

    // Save WAV if requested
    if (!out_path.empty()) {
        if (!wav_accum.save()) {
            std::fprintf(stderr, "Failed to save WAV file\n");
        } else {
            std::fprintf(stderr, "WAV saved: %s (%zu samples)\n",
                        out_path.c_str(), wav_accum.samples.size());
        }
    }

    return 0;
}

// ============================================================================
// HTTP mode
// ============================================================================

namespace {

using kokopop::json_error;

bool handle_tts(kokopop::Model * model, const std::string & default_voice,
                float default_speed, kokopop::StreamMode stream_mode,
                kokopop::HttpRequest & req, kokopop::HttpResponse & res) {
    // Parse JSON body with yyjson
    yyjson_read_err err;
    yyjson_doc * doc = yyjson_read_opts(static_cast<char *>(req.body.data()), req.body.size(),
                                        YYJSON_READ_ALLOW_TRAILING_COMMAS, nullptr, &err);
    if (!doc) {
        res.status_code = 400;
        res.status_text = "Bad Request";
        res.set_json_string(json_error(std::string("JSON parse error: ") + err.msg));
        return false;
    }

    yyjson_val * root = yyjson_doc_get_root(doc);

    // Extract fields
    const char * text = yyjson_get_str(yyjson_obj_get(root, "text"));
    const char * voice = yyjson_get_str(yyjson_obj_get(root, "voice"));
    const char * mode_str = yyjson_get_str(yyjson_obj_get(root, "mode"));

    // Speed
    float speed = default_speed;
    yyjson_val * speed_val = yyjson_obj_get(root, "speed");
    if (speed_val && !yyjson_is_null(speed_val)) {
        speed = (float)yyjson_get_num(speed_val);
    }

    // Voice
    std::string current_voice = voice ? voice : default_voice.c_str();

    // Mode
    kokopop::StreamMode current_mode = stream_mode;
    if (mode_str) {
        if (std::string(mode_str) == "long_form") {
            current_mode = kokopop::StreamMode::LongForm;
        } else if (std::string(mode_str) != "interactive") {
            // Invalid mode
            yyjson_doc_free(doc);
            res.status_code = 400;
            res.status_text = "Bad Request";
            res.set_json_string(json_error(std::string("Unknown mode: ") + mode_str));
            return false;
        }
    }

    // Validate text
    if (!text || std::string(text).empty()) {
        yyjson_doc_free(doc);
        res.status_code = 400;
        res.status_text = "Bad Request";
        res.set_json_string(json_error("Missing or empty 'text' field"));
        return false;
    }

    // Save text BEFORE freeing the doc (yyjson strings are owned by the doc)
    std::string text_str(text);
    yyjson_doc_free(doc);
    if (text_str.size() > 100000) {
        res.status_code = 413;
        res.status_text = "Payload Too Large";
        res.set_json_string(json_error("Text too long (max 100,000 chars)"));
        return false;
    }

    std::fprintf(stderr, "[http] POST /tts: %zu chars, voice=%s, speed=%.1f\n",
                text_str.size(), current_voice.c_str(), speed);

    // Perform synthesis
    HttpStreamContext ctx;
    ctx.sample_rate = model->sample_rate;

    auto handle = kokopop::stream_synthesize(
        *model,
        text_str,
        current_voice,
        speed,
        current_mode,
        http_audio_callback,
        &ctx);

    handle.join();

    std::fprintf(stderr, "[http] POST /tts: done, %zu samples\n", ctx.samples.size());

    // Build WAV response
    if (ctx.samples.empty()) {
        res.status_code = 500;
        res.status_text = "Internal Server Error";
        res.set_json_string(json_error("Synthesis produced no audio"));
        return false;
    }

    // Build complete WAV
    auto wav_header = build_wav_header(ctx.sample_rate, ctx.samples.size());
    res.body = wav_header;
    res.body.insert(res.body.end(),
                    reinterpret_cast<const char *>(ctx.samples.data()),
                    reinterpret_cast<const char *>(ctx.samples.data()) + ctx.samples.size() * sizeof(float));
    res.set_content_type("audio/wav");
    res.headers["Content-Length"] = std::to_string(res.body.size());

    return false; // close connection after response
}

bool handle_health(kokopop::Model * model, kokopop::HttpRequest & /*req*/, kokopop::HttpResponse & res) {
    std::string json = "{\"status\":\"ready\",\"sample_rate\":" + std::to_string(model->sample_rate) + "}";
    res.set_json_string(json);
    return true; // keep-alive
}

bool handle_voices(kokopop::Model * /*model*/, kokopop::HttpRequest & /*req*/, kokopop::HttpResponse & res) {
    // The model doesn't expose voice names — return empty list with a note
    std::string json = "{\"voices\":null,\"note\":\"Voice names are not exposed by the model. Use the --voice flag to set the default voice.\"}";
    res.set_json_string(json);
    return true; // keep-alive
}

bool handle_not_found(kokopop::HttpRequest & /*req*/, kokopop::HttpResponse & res) {
    res.status_code = 404;
    res.status_text = "Not Found";
    res.set_json_string(json_error("not_found"));
    return false; // close
}

} // namespace

static int run_http_mode(kokopop::Model * model, const std::string & default_voice,
                         float speed, kokopop::StreamMode stream_mode,
                         const std::string & bind_addr, int port) {
    kokopop::HttpServer server;

    // Setup routes with lambdas that capture model
    server.route("/tts",
        [model, default_voice, speed, stream_mode](kokopop::HttpRequest & req, kokopop::HttpResponse & res) {
            if (req.method != "POST") {
                res.status_code = 405;
                res.status_text = "Method Not Allowed";
                res.set_json_string(json_error("POST required"));
                return false;
            }
            return handle_tts(model, default_voice, speed, stream_mode, req, res);
        });

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

    // Wait until server stops (e.g., SIGINT)
    server.join();

    return 0;
}

// ============================================================================
// Main
// ============================================================================

static volatile std::atomic<bool> g_running{true};

#ifdef _WIN32
#include <signal.h>
static void signal_handler(int) {
    g_running.store(false);
}
#else
#include <csignal>
static void signal_handler(int) {
    g_running.store(false);
}
#endif

int main(int argc, char ** argv) {
    std::string model_path;
    std::string voice;
    float speed = 1.0f;
    std::string mode_str = "interactive";
    std::string out_path;
    int threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));

    // HTTP mode
    bool http_mode = false;
    int http_port = 8080;
    std::string http_bind = "127.0.0.1";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0) {
            model_path = arg_value(i, argc, argv);
        } else if (std::strcmp(argv[i], "--voice") == 0) {
            voice = arg_value(i, argc, argv);
        } else if (std::strcmp(argv[i], "--speed") == 0) {
            speed = std::stof(arg_value(i, argc, argv));
        } else if (std::strcmp(argv[i], "--mode") == 0) {
            mode_str = arg_value(i, argc, argv);
        } else if (std::strcmp(argv[i], "--out") == 0) {
            out_path = arg_value(i, argc, argv);
        } else if (std::strcmp(argv[i], "--threads") == 0) {
            threads = std::stoi(arg_value(i, argc, argv));
        } else if (std::strcmp(argv[i], "--http") == 0) {
            http_mode = true;
        } else if (std::strcmp(argv[i], "--port") == 0) {
            http_port = std::stoi(arg_value(i, argc, argv));
        } else if (std::strcmp(argv[i], "--bind") == 0) {
            http_bind = arg_value(i, argc, argv);
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty() || voice.empty()) {
        usage(argv[0]);
        return 2;
    }

    // Validate mode
    kokopop::StreamMode stream_mode = kokopop::StreamMode::Interactive;
    if (mode_str == "long_form") {
        stream_mode = kokopop::StreamMode::LongForm;
    } else if (mode_str != "interactive") {
        std::fprintf(stderr, "Unknown mode: %s (use 'interactive' or 'long_form')\n", mode_str.c_str());
        return 2;
    }

    // Load model
    std::fprintf(stderr, "[kokopop] Loading model: %s\n", model_path.c_str());
    kokopop_model_options options{};
    options.n_threads = threads;
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

    // Setup signal handler for graceful shutdown
#ifdef _WIN32
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
#else
    struct sigaction sa{};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
#endif

    if (http_mode) {
        std::fprintf(stderr, "[kokopop] Starting HTTP server on %s:%d\n", http_bind.c_str(), http_port);
        return run_http_mode(model, voice, speed, stream_mode, http_bind, http_port);
    }

    // STDIO mode (default)
    if (isatty(STDIN_FILENO)) {
        std::fprintf(stderr, "[kokopop] Warning: stdin is a terminal. Pipe JSON input or use --http mode.\n");
    }
    return run_stdio_mode(model, voice, speed, stream_mode, out_path);
}
