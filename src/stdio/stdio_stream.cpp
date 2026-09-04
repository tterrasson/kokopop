#include "stdio/stdio_stream.h"

#include "kokopop.h"
#include "yyjson.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

namespace kokopop {

// ---------------------------------------------------------------------------
// WavAccumulator
// ---------------------------------------------------------------------------

void WavAccumulator::add(const float * data, size_t n) {
    std::lock_guard<std::mutex> lock(mutex);
    samples.insert(samples.end(), data, data + n);
}

bool WavAccumulator::save() {
    std::lock_guard<std::mutex> lock(mutex);
    if (samples.empty() || sample_rate <= 0) return true;

    kokopop_audio audio{};
    audio.samples = samples.data();
    audio.n_samples = samples.size();
    audio.sample_rate = sample_rate;

    int rc = kokopop_write_wav(path.c_str(), &audio);
    return rc == KOKOPOP_OK;
}

// ---------------------------------------------------------------------------
// StdioStreamer
// ---------------------------------------------------------------------------

StdioStreamer::StdioStreamer(
    Model & model,
    const std::string & voice,
    float speed,
    StreamMode mode,
    const std::string & out_path)
    : _model(model)
    , _voice(voice)
    , _speed(speed)
    , _mode(mode)
    , _out_path(out_path)
    , _sample_rate(model.sample_rate())
{
    _wav_accum.path = out_path;
    _wav_accum.sample_rate = model.sample_rate();
}

StdioStreamer::~StdioStreamer() {
    join();
}

void StdioStreamer::run() {
    std::string line;
    while (std::getline(std::cin, line)) {
        _process_line(line);
    }

    // Flush any remaining accumulated text on EOF
    if (!_text_buffer.empty()) {
        _synthesize(_text_buffer);
        _text_buffer.clear();
    }
}

void StdioStreamer::join() {
    for (auto & handle : _handles) {
        handle.join();
    }
    _handles.clear();

    // Save WAV if requested
    if (!_out_path.empty()) {
        if (_wav_accum.save()) {
            std::fprintf(stderr, "[kokopop] WAV saved: %s (%zu samples)\n",
                        _out_path.c_str(), _wav_accum.samples.size());
        } else {
            std::fprintf(stderr, "[kokopop] Failed to save WAV file\n");
        }
    }
}

void StdioStreamer::_on_audio(const float * data, size_t n_samples, int chunk_index, void * user_data) {
    auto * self = static_cast<StdioStreamer *>(user_data);

    self->_chunk_count.fetch_add(1);
    std::fprintf(stderr, "[kokopop] audio_chunk[%d]: %zu samples (%.1fms)\n",
                chunk_index, n_samples,
                (double)n_samples / self->_sample_rate * 1000.0);

    // Write raw float32 to stdout
    fwrite(data, sizeof(float), n_samples, stdout);
    fflush(stdout);

    // WAV accumulation
    self->_wav_accum.add(data, n_samples);
}

void StdioStreamer::_synthesize(const std::string & text) {
    // Join previous stream if any
    if (!_handles.empty()) {
        std::fprintf(stderr, "[kokopop] joining previous stream...\n");
        _handles.back().join();
        _handles.pop_back();
    }

    std::fprintf(stderr, "[kokopop] Synthesizing: %zu chars, voice=%s, speed=%.1f, mode=%s\n",
                text.size(), _voice.c_str(), _speed,
                _mode == StreamMode::LongForm ? "long_form" : "adaptative");

    auto handle = stream_synthesize(
        _model,
        text,
        _voice,
        _speed,
        _mode,
        [](const float * data, size_t n_samples, int chunk_index, void * user_data) {
            static_cast<StdioStreamer *>(user_data)->_on_audio(data, n_samples, chunk_index, user_data);
            return true;
        },
        this);

    _handles.push_back(std::move(handle));
}

void StdioStreamer::_process_line(const std::string & raw_line) {
    // Trim
    size_t start = raw_line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return;
    std::string line = raw_line.substr(start);
    size_t end = line.find_last_not_of(" \t\r\n");
    if (end != std::string::npos) line = line.substr(0, end + 1);

    if (line.empty()) return;

    // Parse JSON with yyjson
    yyjson_read_err err;
    yyjson_doc * doc = yyjson_read_opts(const_cast<char *>(line.c_str()), line.size(),
                                        YYJSON_READ_ALLOW_TRAILING_COMMAS, nullptr, &err);
    if (!doc) {
        std::fprintf(stderr, "[kokopop] JSON parse error: %s\n", err.msg);
        return;
    }

    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        std::fprintf(stderr, "[kokopop] JSON is not an object\n");
        yyjson_doc_free(doc);
        return;
    }

    // Extract fields using yyjson
    // Note: strings from yyjson are owned by the doc, so copy them before freeing
    // yyjson_get_str returns null for a missing or non-string key, and std::string(nullptr) is undefined
    const char * text_ptr = yyjson_get_str(yyjson_obj_get(root, "text"));
    std::string text_str = text_ptr != nullptr ? std::string(text_ptr) : std::string();
    bool flush = yyjson_obj_get(root, "flush")
                 && !yyjson_is_null(yyjson_obj_get(root, "flush"))
                 && yyjson_get_bool(yyjson_obj_get(root, "flush"));
    bool stop = yyjson_obj_get(root, "stop")
                && !yyjson_is_null(yyjson_obj_get(root, "stop"))
                && yyjson_get_bool(yyjson_obj_get(root, "stop"));

    yyjson_doc_free(doc);

    // Handle stop
    if (stop) {
        if (!_text_buffer.empty() && !text_str.empty()) {
            _text_buffer += text_str;
            flush = true;
        } else {
            return;
        }
    }

    // Accumulate text
    if (!text_str.empty()) {
        _text_buffer += text_str;
    }

    // Process if flush
    if (flush && !_text_buffer.empty()) {
        _synthesize(_text_buffer);
        _text_buffer.clear();
    }
}

} // namespace kokopop
