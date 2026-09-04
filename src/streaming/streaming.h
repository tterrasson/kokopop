#pragma once

#include "arch/kokoro/kokoro.h"
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
    Adaptative,  // Fast TTFB, dynamic chunk sizing
    LongForm     // Larger chunks, better prosody
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
// Two-phase synthesis API (for interleaving / async scheduling)
// ---------------------------------------------------------------------------

/// Prepared synthesis plan — output of the chunking/phonemization phase.
/// This struct contains everything needed for inference: chunks with
/// phonemes + tokens, voice name, speed, and post-processing config.
///
/// Thread-safety: once constructed, this struct is read-only and can be
/// safely shared across threads.  The only non-const operation is
/// clear_chunk_data() which is intended to be called after a chunk has
/// been inferred to free memory.
struct SynthesisPlan {
    std::vector<Chunk> chunks;
    std::string voice;
    float speed;
    StreamMode mode;
    ChunkConfig config;
    KokoroDiffusionOptions diffusion;

    /// sanoTTS's deterministic noise seed. Absent means "derive one from the
    /// voice", which is what makes two runs of the same text sound identical
    /// without the caller having to pick a number.
    bool     has_noise_seed = false;
    uint64_t noise_seed     = 0;

    /// Architecture-specific inputs for the chunk at `seq_index` in the
    /// utterance. `kokoro_style_len` is filled in by `synthesize_chunk()`.
    SynthesisExtras chunk_extras(uint32_t seq_index) const;

    /// Estimate total output samples (rough: 0.035s per token / speed)
    size_t estimated_total_samples(int sample_rate) const;
};

/// Phase 1 — Prepare a synthesis plan.
///
/// This function performs chunking, phonemization and tokenization.
/// It does NOT touch the GGML backend scheduler, so it is safe to call
/// concurrently from multiple threads (eSpeak-ng has its own internal
/// mutex, and model->tokenize_phonemes() is read-only).
///
/// Returns an empty plan on error (error string set).
SynthesisPlan prepare_synthesis(
    Model & model,
    const std::string & text,
    const std::string & voice,
    float speed,
    StreamMode mode,
    std::string & error,
    const ChunkConfig * chunk_config_override = nullptr);

/// Phase 2 — Infer a single chunk of audio.
///
/// This function calls into the GGML backend (synthesize_phonemes +
/// postprocessing + crossfade).  It is NOT thread-safe: only one call
/// to infer_chunk() should be active at a time across all requests.
///
/// Parameters:
///   plan       — prepared synthesis plan (read-only)
///   chunk_idx  — index of the chunk to infer (0-based)
///   prev_tail  — tail samples from the previous chunk (for crossfade).
///                Pass empty vector for the first chunk.
///   out_tail   — output: tail samples from this chunk for the next crossfade.
///                Only populated if crossfade_ms > 0.
///   seq_index  — position of the chunk in the whole utterance, used to seed
///                sanoTTS's per-chunk noise. Callers that hand over a plan
///                holding a single chunk at a time (adaptative, incremental)
///                must pass their own running counter, otherwise every chunk
///                would seed as chunk 0. -1 means "use chunk_idx".
///
/// Returns the processed audio samples on success, empty vector on error.
std::vector<float> infer_chunk(
    Model & model,
    const SynthesisPlan & plan,
    int chunk_idx,
    const std::vector<float> & prev_tail,
    std::vector<float> & out_tail,
    std::string & error,
    int seq_index = -1);

// ---------------------------------------------------------------------------
// Full streaming synthesis (backward-compatible wrapper)
// ---------------------------------------------------------------------------

/// `has_noise_seed` / `noise_seed` pin sanoTTS's deterministic noise for the
/// whole utterance; leave them at their defaults to let the decoder derive a
/// seed from the voice. Kokoro voices ignore them.
StreamHandle stream_synthesize(
    kokopop::Model & model,
    const std::string & text,
    const std::string & voice,
    float speed,
    StreamMode mode,
    AudioCallback callback,
    void * user_data,
    bool has_noise_seed = false,
    uint64_t noise_seed = 0);

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
        void * user_data,
        bool has_noise_seed = false,
        uint64_t noise_seed = 0);

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
    bool has_noise_seed_ = false;
    uint64_t noise_seed_ = 0;
    std::string buffer_;
    std::atomic<bool> stopped_{false};
    int chunk_counter_ = 0;
};

} // namespace kokopop
