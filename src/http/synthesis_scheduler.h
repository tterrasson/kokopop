#pragma once

#include "http/request_context.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace kokopop {

/// SynthesisScheduler: round-robin worker for interleaved chunk inference.
///
/// Architecture:
///   - One worker thread that processes requests in round-robin order.
///   - Each request goes through two phases:
///       1. PREPARING: chunking + phonemization (no GGML)
///       2. INFERRING: one chunk at a time (GGML, serialized)
///   - After inferring a chunk, the request is re-enqueued at the end
///     if it has more chunks (round-robin interleaving).
///   - Back-pressure: if a request's output queue is full (client is slow),
///     the request is re-enqueued at the end to let other requests progress.
///
/// Thread-safety:
///   - The worker loop is the only thread that calls infer_chunk().
///   - submit() can be called from any thread.
///   - flush_completed_chunks() is called from the event loop to check
///     if any requests have data ready.
class SynthesisScheduler {
public:
    explicit SynthesisScheduler(Model & model);
    ~SynthesisScheduler();

    SynthesisScheduler(const SynthesisScheduler &) = delete;
    SynthesisScheduler & operator=(const SynthesisScheduler &) = delete;

    /// Submit a new synthesis request. Returns the RequestContext.
    std::shared_ptr<RequestContext> submit(
        const std::string & text,
        const std::string & voice,
        float speed,
        StreamMode mode,
        RequestContext::AudioFormat format,
        const ChunkConfig & chunk_config_override = ChunkConfig{},
        bool has_chunk_config_override = false);

    /// Stop the worker thread. Does not wait — call join() for that.
    void stop();

    /// Wait for the worker thread to exit.
    void join();

    /// Get the next available request ID.
    uint32_t next_request_id();

private:
    void _worker_loop();

    Model & _model;
    std::atomic<uint32_t> _next_id{1};

    // Queue of requests to process (round-robin)
    std::deque<std::shared_ptr<RequestContext>> _pending;
    std::mutex _queue_mutex;
    std::condition_variable _queue_cv;

    std::thread _worker;
    std::atomic<bool> _running{false};
};

} // namespace kokopop
