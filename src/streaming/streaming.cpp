#include "streaming.h"

#include "synthesis/synth.h"
#include "audio/audio_postprocess.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace kokopop {

// ---------------------------------------------------------------------------
// SynthesisPlan helpers
// ---------------------------------------------------------------------------

size_t SynthesisPlan::estimated_total_samples(int sample_rate) const {
    // Rough estimate: ~0.035s per token at speed 1.0
    size_t total_tokens = 0;
    for (const auto & chunk : chunks) {
        total_tokens += static_cast<size_t>(chunk.n_tokens);
    }
    const double duration_s = static_cast<double>(total_tokens) * 0.035 / speed;
    return static_cast<size_t>(duration_s * sample_rate);
}

// ---------------------------------------------------------------------------
// Phase 1 — prepare_synthesis (chunking + phonemization)
// ---------------------------------------------------------------------------

SynthesisPlan prepare_synthesis(
    Model & model,
    const std::string & text,
    const std::string & voice,
    float speed,
    StreamMode mode,
    std::string & error)
{
    SynthesisPlan plan;

    // Select config
    plan.config = (mode == StreamMode::Interactive)
        ? make_interactive_config()
        : make_long_form_config();

    // Tokenize function wrapper
    TokenizeFn tokenize_fn = [&model](const std::string & phonemes,
                                       std::vector<uint32_t> & ids,
                                       std::string & err) -> bool {
        return model.tokenize_phonemes(phonemes, ids, err);
    };

    // Chunk the text
    plan.chunks = chunk_text(text, voice, plan.config, tokenize_fn, error);
    if (plan.chunks.empty()) {
        return plan;
    }

    plan.voice = voice;
    plan.speed = speed;
    plan.mode = mode;

    std::fprintf(stderr, "[kokopop] prepared: %zu chunks, %s mode\n",
                plan.chunks.size(),
                (mode == StreamMode::Interactive) ? "interactive" : "long_form");

    return plan;
}

// ---------------------------------------------------------------------------
// Phase 2 — infer_chunk (single chunk inference + postprocessing)
// ---------------------------------------------------------------------------

std::vector<float> infer_chunk(
    Model & model,
    const SynthesisPlan & plan,
    int chunk_idx,
    const std::vector<float> & prev_tail,
    std::vector<float> & out_tail,
    std::string & error)
{
    if (chunk_idx < 0 || chunk_idx >= static_cast<int>(plan.chunks.size())) {
        error = "chunk index out of range";
        return {};
    }

    const auto & chunk = plan.chunks[chunk_idx];

    std::fprintf(stderr, "[kokopop] chunk[%d/%zu]: %d tokens, phonemes=%zu chars\n",
                chunk_idx + 1, plan.chunks.size(),
                chunk.n_tokens, chunk.phonemes.size());

    // --- Synthesize raw audio ---
    kokopop_audio raw{};
    if (!synthesize_phonemes(model, chunk.phonemes, plan.voice, plan.speed, raw, error)) {
        return {};
    }

    if (raw.n_samples == 0) {
        kokopop_audio_free(&raw);
        error = "synthesis produced no audio";
        return {};
    }

    // Copy to vector for postprocessing
    std::vector<float> raw_audio(raw.n_samples);
    std::copy(raw.samples, raw.samples + raw.n_samples, raw_audio.begin());
    kokopop_audio_free(&raw);

    std::fprintf(stderr, "[kokopop] chunk[%d] synthesized: %zu samples (%.1fms)\n",
                chunk_idx + 1, raw_audio.size(),
                (double)raw_audio.size() / model.sample_rate * 1000.0);

    // --- Post-process ---
    auto processed = postprocess_chunk_audio(
        raw_audio, chunk, chunk_idx,
        static_cast<int>(plan.chunks.size()),
        plan.config, model.sample_rate);

    // --- Crossfade with previous chunk ---
    if (!prev_tail.empty() && !processed.empty()) {
        auto crossed = apply_crossfade_smart(
            prev_tail, processed,
            chunk.boundary_after,
            plan.config.crossfade_ms,
            model.sample_rate);
        processed = std::move(crossed);
    }

    // --- Save tail for next crossfade ---
    out_tail.clear();
    const int crossfade_samples = (plan.config.crossfade_ms * model.sample_rate) / 1000;
    if (!processed.empty() && crossfade_samples > 0) {
        size_t tail_start = processed.size() > static_cast<size_t>(crossfade_samples)
            ? processed.size() - crossfade_samples
            : 0;
        out_tail.assign(processed.begin() + tail_start, processed.end());
    }

    return processed;
}

// ---------------------------------------------------------------------------
// Full streaming synthesis (backward-compatible wrapper)
// ---------------------------------------------------------------------------

StreamHandle stream_synthesize(
    kokopop::Model & model,
    const std::string & text,
    const std::string & voice,
    float speed,
    StreamMode mode,
    AudioCallback callback,
    void * user_data) {

    StreamHandle handle;
    handle.state = std::make_shared<StreamState>();

    // Copy data that must survive after stream_synthesize returns
    std::string text_copy = text;
    std::string voice_copy = voice;
    std::shared_ptr<AudioCallback> cb_shared = std::make_shared<AudioCallback>(std::move(callback));
    std::shared_ptr<StreamState> state_shared = handle.state;

    // Capture model explicitly as a raw, non-owning reference.
    //
    // SAFETY CONTRACT: the caller must guarantee that `model` outlives the
    // streaming thread (i.e. call `handle.join()` or `handle.stop()` before
    // destroying the Model).  A std::shared_ptr<Model> is not feasible here
    // because the caller owns the Model exclusively and we cannot share
    // ownership without changing the public API.
    //
    // We use `[]` (empty default capture) so nothing else slips in by
    // reference accidentally.  Every variable the lambda needs is listed
    // explicitly below.
    handle.thread = std::make_shared<std::thread>(
        [model = &model, text_copy, voice_copy, cb_shared, state_shared,
         speed, mode, user_data]() {

            // Phase 1: prepare
            std::string error;
            auto plan = prepare_synthesis(*model, text_copy, voice_copy, speed, mode, error);
            if (plan.chunks.empty()) {
                std::fprintf(stderr, "[kokopop] prepare_synthesis failed: %s\n", error.c_str());
                state_shared->done.store(true);
                return;
            }

            // Phase 2: infer each chunk
            std::vector<float> prev_tail;

            for (int i = 0; i < static_cast<int>(plan.chunks.size()); ++i) {
                if (state_shared->stopped.load()) break;

                std::vector<float> out_tail;
                auto processed = infer_chunk(
                    *model, plan, i, prev_tail, out_tail, error);

                if (processed.empty()) {
                    std::fprintf(stderr, "[kokopop] WARNING chunk[%d]: %s — skipping\n",
                                i, error.c_str());
                    continue;
                }

                prev_tail = std::move(out_tail);

                // Call callback
                bool continue_streaming = (*cb_shared)(
                    processed.data(), processed.size(), i, user_data);
                if (!continue_streaming) {
                    state_shared->stopped.store(true);
                    break;
                }
            }

            state_shared->done.store(true);
        });

    return handle;
}

// ---------------------------------------------------------------------------
// Incremental streamer
// ---------------------------------------------------------------------------

IncrementalStreamer::IncrementalStreamer(
    kokopop::Model & model,
    const std::string & voice,
    float speed,
    StreamMode mode,
    AudioCallback callback,
    void * user_data)
    : model_(model)
    , voice_(voice)
    , speed_(speed)
    , mode_(mode)
    , callback_(callback)
    , user_data_(user_data) {
}

IncrementalStreamer::~IncrementalStreamer() {
    stop();
}

void IncrementalStreamer::feed_text(const std::string & text) {
    buffer_ += text;
}

void IncrementalStreamer::flush() {
    if (stopped_.load() || buffer_.empty()) return;

    std::string text = std::move(buffer_);
    buffer_.clear();

    // Phase 1: prepare
    std::string error;
    auto plan = prepare_synthesis(model_, text, voice_, speed_, mode_, error);
    if (plan.chunks.empty()) {
        if (!error.empty()) {
            std::fprintf(stderr, "[kokopop] prepare error: %s\n", error.c_str());
        }
        return;
    }

    // Phase 2: infer each chunk
    std::vector<float> prev_tail;

    for (int i = 0; i < static_cast<int>(plan.chunks.size()); ++i) {
        if (stopped_.load()) break;

        std::vector<float> out_tail;
        auto processed = infer_chunk(
            model_, plan, i, prev_tail, out_tail, error);

        if (processed.empty()) {
            std::fprintf(stderr, "[kokopop] WARNING chunk[%d]: %s — skipping\n",
                        chunk_counter_, error.c_str());
            continue;
        }

        prev_tail = std::move(out_tail);

        callback_(processed.data(), processed.size(), chunk_counter_, user_data_);
        ++chunk_counter_;
    }
}

void IncrementalStreamer::stop() {
    stopped_.store(true);
}

} // namespace kokopop
