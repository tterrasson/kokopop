#pragma once

#include "streaming/streaming.h"
#include "synthesis/synthesis_session.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

namespace kokopop {

/// Per-request context for HTTP synthesis.
///
/// Shared between:
///   - SynthesisScheduler worker thread (writes output_queue, updates state)
///   - AsyncHttpServer event loop (reads output_queue, sends HTTP chunks)
///
/// Thread-safety:
///   - All state transitions use std::atomic<State>.
///   - output_queue is protected by output_mutex.
///   - output_has_data is an atomic flag set by the worker after pushing.
struct RequestContext {
    uint32_t request_id = 0;

    // ----------------------------------------------------------------
    // State machine: PREPARING → INFERRING → DONE / ERROR / CANCELLED
    // ----------------------------------------------------------------
    enum class State {
        PREPARING,   // Phase 1: chunking + phonemization (in worker)
        INFERRING,   // Phase 2: inferring chunks (in worker)
        DONE,        // All chunks inferred
        ERROR,       // Synthesis failed
        CANCELLED    // Client disconnected or cancelled
    };

    std::atomic<State> state{State::PREPARING};
    std::atomic<bool> cancelled{false};
    std::atomic<bool> output_has_data{false};

    // ----------------------------------------------------------------
    // Request configuration
    // ----------------------------------------------------------------
    enum class AudioFormat { PCM, WAV, OGG_OPUS };

    std::string text;
    std::string voice;
    float speed = 1.0f;
    StreamMode mode = StreamMode::Adaptative;
    AudioFormat format = AudioFormat::PCM;
    ChunkConfig chunk_config;      // override fields (defaults = no-op)
    bool has_chunk_config = false;
    KokoroDiffusionOptions diffusion{};  // disabled by default
    std::unique_ptr<SynthesisSession> synthesis;

    bool is_streaming() const {
        return format == AudioFormat::PCM || format == AudioFormat::OGG_OPUS;
    }

    // ----------------------------------------------------------------
    // Synthesis plan (filled during PREPARING phase)
    // ----------------------------------------------------------------
    std::shared_ptr<SynthesisPlan> plan;
    std::vector<Unit> adaptative_units;
    size_t adaptative_next_unit = 0;
    AdaptativeChunkController adaptative_controller;

    // ----------------------------------------------------------------
    // Progress
    // ----------------------------------------------------------------
    std::atomic<int> chunks_completed{0};
    int chunks_total{0};
    int sample_rate{24000};

    // ----------------------------------------------------------------
    // Output queue (written by worker, read by event loop)
    // ----------------------------------------------------------------
    struct AudioChunk {
        std::vector<float> samples;
        int chunk_index;
    };

    /// Maximum number of chunks that can accumulate in the output queue.
    /// When reached, the worker back-pressures by re-enqueuing this
    /// request at the end of the scheduler queue.
    static constexpr size_t MAX_PENDING = 3;

    std::mutex output_mutex;
    std::queue<AudioChunk> output_queue;

    /// Try to pop the next audio chunk. Returns true if a chunk was available.
    bool try_pop(AudioChunk & out) {
        std::lock_guard<std::mutex> lock(output_mutex);
        if (output_queue.empty()) return false;
        out = std::move(output_queue.front());
        output_queue.pop();
        if (output_queue.empty()) {
            output_has_data.store(false);
        }
        return true;
    }

    /// Check if the output queue has room for another chunk (back-pressure).
    bool output_has_room() {
        std::lock_guard<std::mutex> lock(output_mutex);
        return output_queue.size() < MAX_PENDING;
    }

    /// Push an audio chunk into the output queue.
    void push_audio(std::vector<float> && samples, int chunk_index) {
        std::lock_guard<std::mutex> lock(output_mutex);
        output_queue.push({std::move(samples), chunk_index});
        output_has_data.store(true);
    }

    // ----------------------------------------------------------------
    // Crossfade state (kept between chunks of the same request)
    // ----------------------------------------------------------------
    std::vector<float> prev_tail;

    // ----------------------------------------------------------------
    // WAV accumulation (for format=wav)
    // ----------------------------------------------------------------
    std::vector<float> wav_accumulator;

    // Ogg/Opus prebuffer configuration (encoder is owned by the HTTP loop)
    int ogg_prebuffer_chunks = 0;

    // ----------------------------------------------------------------
    // Error reporting
    // ----------------------------------------------------------------
    std::string error;
};

} // namespace kokopop
