#include "kokopop.h"
// kokopop_internal.h removed - use kokopop.h directly

#include "synthesis/chunker/chunker.h"
#include "streaming/streaming.h"
#include "playback/playback.h"

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

namespace {

void usage(const char * argv0) {
    std::fputs("usage: ", stderr);
    std::fputs(argv0, stderr);
    std::fputs(" --model kokoro.gguf --voice ff_siwis [--speed 1.0] "
        "[--mode interactive|long_form] [--play] [--out out.wav] [--threads N]\n"
        "\n"
        "Reads JSON commands from stdin and streams TTS audio to stdout.\n"
        "\n"
        "JSON protocol (one command per line):\n"
        "  {\"text\": \"Hello world\"}              >> accumulate text\n"
        "  {\"text\": \"Hello\", \"flush\": true}    >> add text and generate\n"
        "  {\"flush\": true}                        >> generate all accumulated text\n"
        "  {\"stop\": true}                         >> stop streaming\n"
        "\n"
        "Options:\n"
        "  --model PATH    Path to Kokoro GGUF model\n"
        "  --voice NAME    Voice name (e.g., ff_siwis)\n"
        "  --speed FLOAT   Synthesis speed (default: 1.0)\n"
        "  --mode MODE     interactive (default) or long_form\n"
        "  --play          Play audio directly (Core Audio on macOS, stdout elsewhere)\n"
        "  --out PATH      Save full audio to WAV file\n"
        "  --threads N     Number of threads (default: min(4, hw_concurrency))\n"
        "\n"
        "Examples:\n",
        stderr);
    std::fputs("  echo '{{\"text\":\"Hello\", \"flush\":true}}' | ", stderr);
    std::fputs(argv0, stderr);
    std::fputs(" --model m.gguf --voice ff_siwis\n"
        "  ... then pipe to ffplay for playback\n", stderr);
}

const char * arg_value(int & i, int argc, char ** argv) {
    if (i + 1 >= argc) return nullptr;
    ++i;
    return argv[i];
}

// ---------------------------------------------------------------------------
// Simple JSON field extractor (no external dependency)
// ---------------------------------------------------------------------------
std::string extract_json_string(const std::string & line, const char * key) {
    std::string search = "\"";
    search += key;
    search += "\"";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return "";

    pos = line.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    ++pos;

    // Skip whitespace
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) ++pos;
    if (pos >= line.size() || line[pos] != '"') return "";
    ++pos; // skip opening quote

    std::string result;
    while (pos < line.size()) {
        if (line[pos] == '\\') {
            ++pos;
            if (pos < line.size()) {
                if (line[pos] == '"') result += '"';
                else if (line[pos] == '\\') result += '\\';
                else if (line[pos] == 'n') result += '\n';
                else if (line[pos] == 't') result += '\t';
                else if (line[pos] == 'r') result += '\r';
                else result += line[pos];
            }
        } else if (line[pos] == '"') {
            return result;
        } else {
            result += line[pos];
        }
        ++pos;
    }
    return result;
}

bool extract_json_bool(const std::string & line, const char * key, bool default_val = false) {
    std::string search = "\"";
    search += key;
    search += "\"";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return default_val;

    pos = line.find(':', pos + search.size());
    if (pos == std::string::npos) return default_val;
    ++pos;

    // Skip whitespace
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) ++pos;

    if (pos + 4 <= line.size() &&
        line[pos] == 't' && line[pos+1] == 'r' && line[pos+2] == 'u' && line[pos+3] == 'e') {
        return true;
    }
    if (pos + 5 <= line.size() &&
        line[pos] == 'f' && line[pos+1] == 'a' && line[pos+2] == 'l' && line[pos+3] == 's' && line[pos+4] == 'e') {
        return false;
    }
    return default_val;
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

} // namespace

// Global callback state
struct CallbackState {
    bool use_playback = false;
    kokopop::AudioPlayback * playback = nullptr;
    bool use_stdout = true;
    WavAccumulator * wav = nullptr;
    int sample_rate = 24000;
    std::atomic<int> chunk_count{0};
};

static bool audio_callback(const float * data, size_t n_samples, int chunk_index, void * user_data) {
    auto * state = static_cast<CallbackState *>(user_data);

    int count = state->chunk_count.fetch_add(1) + 1;
    std::fprintf(stderr, "[kokopop] audio_chunk[%d]: %zu samples (%.1fms)\n",
                chunk_index, n_samples, (double)n_samples / state->sample_rate * 1000.0);

    // Write to stdout (unless using playback only)
    if (state->use_stdout && !state->use_playback) {
        fwrite(data, sizeof(float), n_samples, stdout);
        fflush(stdout);
    }

    // Playback
    if (state->use_playback && state->playback) {
        state->playback->write(data, n_samples);
    }

    // WAV accumulation
    if (state->wav) {
        state->wav->add(data, n_samples);
    }

    return true;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string voice;
    float speed = 1.0f;
    std::string mode_str = "interactive";
    bool use_playback = false;
    std::string out_path;
    int threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0) {
            model_path = arg_value(i, argc, argv);
        } else if (std::strcmp(argv[i], "--voice") == 0) {
            voice = arg_value(i, argc, argv);
        } else if (std::strcmp(argv[i], "--speed") == 0) {
            speed = std::stof(arg_value(i, argc, argv));
        } else if (std::strcmp(argv[i], "--mode") == 0) {
            mode_str = arg_value(i, argc, argv);
        } else if (std::strcmp(argv[i], "--play") == 0) {
            use_playback = true;
        } else if (std::strcmp(argv[i], "--out") == 0) {
            out_path = arg_value(i, argc, argv);
        } else if (std::strcmp(argv[i], "--threads") == 0) {
            threads = std::stoi(arg_value(i, argc, argv));
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

    // Setup playback
    kokopop::AudioPlayback * playback = nullptr;
    if (use_playback) {
        playback = kokopop::create_default_playback();
        if (!playback->start(model->sample_rate)) {
            std::fprintf(stderr, "Failed to start audio playback\n");
            kokopop::free_playback(playback);
            return 3;
        }
    }

    // Setup WAV accumulator
    WavAccumulator wav_accum;
    wav_accum.path = out_path;
    wav_accum.sample_rate = model->sample_rate;

    // Callback state
    CallbackState state;
    state.use_playback = use_playback;
    state.playback = playback;
    state.use_stdout = !use_playback;
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

        // Extract fields
        std::string text = extract_json_string(line, "text");
        bool flush = extract_json_bool(line, "flush", false);
        bool stop = extract_json_bool(line, "stop", false);

        // Handle stop
        if (stop) {
            // Flush remaining buffer
            if (!text_buffer.empty()) {
                text_buffer += text;
                text.clear();
                flush = true;
            } else {
                break;
            }
        }

        // Accumulate text
        if (!text.empty()) {
            text_buffer += text;
        }

        // Process if flush
        if (flush && !text_buffer.empty()) {
            // Join previous stream before starting a new one.
            // The Model object (scratch arenas, temp buffers) is not thread-safe,
            // so concurrent synthesis would cause data corruption / segfault.
            if (!handles.empty()) {
                std::fprintf(stderr, "[kokopop]   joining previous stream...\n");
                handles.back().join();
                handles.pop_back();
            }

            std::string text_to_process = std::move(text_buffer);
            text_buffer.clear();

            std::fprintf(stderr, "[kokopop] Synthesizing: %zu chars, voice=%s, speed=%.1f, mode=%s\n",
                        text_to_process.size(), voice.c_str(), speed,
                        mode_str.c_str());

            auto handle = kokopop::stream_synthesize(
                *model,
                text_to_process,
                voice,
                speed,
                stream_mode,
                audio_callback,
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
        std::fprintf(stderr, "[kokopop] Synthesizing (EOF flush): %zu chars, voice=%s, speed=%.1f, mode=%s\n",
                    text_buffer.size(), voice.c_str(), speed, mode_str.c_str());
        auto handle = kokopop::stream_synthesize(
            *model,
            text_buffer,
            voice,
            speed,
            stream_mode,
            audio_callback,
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

    // Stop playback
    if (playback) {
        playback->stop();
        playback->wait();
        kokopop::free_playback(playback);
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
