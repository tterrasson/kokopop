#pragma once

#include "model/model.h"
#include "synthesis/chunker/chunker.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace kokopop {

/// Mode selection for chunking presets
enum class StreamMode {
    Interactive,  // Fast TTFB, smaller chunks
    LongForm      // Larger chunks, better prosody
};

/// Callback invoked when a chunk of audio is ready
/// Returns true to continue streaming, false to stop early
using AudioCallback = std::function<bool(
    const float * data, size_t n_samples, int chunk_index, void * user_data)>;

/// Shared state for a streaming session
struct StreamState {
    std::atomic<bool> done{false};
    std::atomic<bool> stopped{false};
};

/// Handle for an active streaming session
struct StreamHandle {
    std::shared_ptr<StreamState> state;
    std::shared_ptr<std::thread> thread;

    StreamHandle() = default;
    StreamHandle(StreamHandle &&) = default;
    StreamHandle & operator=(StreamHandle &&) = default;
    StreamHandle(const StreamHandle &) = default;
    StreamHandle & operator=(const StreamHandle &) = default;

    /// Wait for streaming to complete
    void join() {
        if (thread && thread->joinable()) {
            thread->join();
        }
    }

    /// Request early stop
    void stop() {
        if (state) state->stopped.store(true);
    }

    /// Check if streaming is done
    bool is_done() const {
        return state && state->done.load();
    }
};

// ---------------------------------------------------------------------------
// Full streaming synthesis
// ---------------------------------------------------------------------------

StreamHandle stream_synthesize(
    kokopop::Model & model,
    const std::string & text,
    const std::string & voice,
    float speed,
    StreamMode mode,
    AudioCallback callback,
    void * user_data);

// ---------------------------------------------------------------------------
// Incremental streaming — text arrives in fragments
// ---------------------------------------------------------------------------

class IncrementalStreamer {
public:
    IncrementalStreamer(
        kokopop::Model & model,
        const std::string & voice,
        float speed,
        StreamMode mode,
        AudioCallback callback,
        void * user_data);

    ~IncrementalStreamer();

    IncrementalStreamer(const IncrementalStreamer &) = delete;
    IncrementalStreamer & operator=(const IncrementalStreamer &) = delete;

    /// Add text to the buffer
    void feed_text(const std::string & text);

    /// Generate audio for all buffered text
    void flush();

    /// Stop and cleanup
    void stop();

private:
    kokopop::Model & model_;
    std::string voice_;
    float speed_;
    StreamMode mode_;
    AudioCallback callback_;
    void * user_data_;
    std::string buffer_;
    std::atomic<bool> stopped_{false};
    int chunk_counter_ = 0;
};

} // namespace kokopop
