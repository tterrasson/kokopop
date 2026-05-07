#pragma once

// STDIO mode: reads JSON commands from stdin, streams raw audio (float32) to stdout.
//
// JSON protocol (one command per line):
//   {"text": "Hello world"}              >> accumulate text
//   {"text": "Hello", "flush": true}    >> add text and generate
//   {"flush": true}                      >> generate all accumulated text
//   {"stop": true}                       >> stop streaming
//
// This module is used by tools/kokopop_stream.cpp when NOT in --http mode.

#include "streaming/streaming.h"

#include <atomic>
#include <string>
#include <vector>
#include <mutex>

namespace kokopop {

/// WAV file accumulator — collects audio samples and writes to WAV on save().
struct WavAccumulator {
    std::mutex mutex;
    std::string path;
    std::vector<float> samples;
    int sample_rate = 0;

    /// Add a chunk of audio samples
    void add(const float * data, size_t n);

    /// Save accumulated samples as WAV file. Returns true on success.
    bool save();
};

/// STDIO streamer — processes JSON commands from stdin and streams audio.
///
/// Usage:
///   StdioStreamer streamer(model, voice, speed, mode, out_path);
///   streamer.run();  // blocks until stdin EOF or {"stop":true}
///   streamer.join(); // wait for final synthesis
class StdioStreamer {
public:
    StdioStreamer(
        Model & model,
        const std::string & voice,
        float speed,
        StreamMode mode,
        const std::string & out_path);

    ~StdioStreamer();

    StdioStreamer(const StdioStreamer &) = delete;
    StdioStreamer & operator=(const StdioStreamer &) = delete;

    /// Run the main loop: read stdin, process commands, synthesize.
    /// Blocks until stdin EOF or {"stop": true} received.
    /// Does NOT wait for the final synthesis — call join() for that.
    void run();

    /// Wait for all pending syntheses to complete and save WAV if requested.
    void join();

private:
    Model & _model;
    std::string _voice;
    float _speed;
    StreamMode _mode;
    std::string _out_path;

    WavAccumulator _wav_accum;
    int _sample_rate;
    std::atomic<int> _chunk_count{0};

    std::string _text_buffer;
    std::vector<StreamHandle> _handles;

    void _on_audio(const float * data, size_t n_samples, int chunk_index, void * user_data);
    void _synthesize(const std::string & text);
    void _process_line(const std::string & line);
};

} // namespace kokopop
