#include "streaming.h"

#include "synthesis/synth.h"
#include "audio/audio_postprocess.h"

#include <algorithm>
#include <atomic>
#include <future>
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
    std::string & error,
    const ChunkConfig * chunk_config_override)
{
    SynthesisPlan plan;

    // Select config
    if (mode == StreamMode::Interactive) {
        plan.config = make_interactive_config();
    } else if (mode == StreamMode::UltraFast) {
        plan.config = make_ultra_fast_config();
    } else {
        plan.config = make_long_form_config();
    }

    if (chunk_config_override) {
        plan.config = merge_chunk_config(plan.config, *chunk_config_override);
    }

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

    std::string mode_name = "unknown";
    if (mode == StreamMode::Interactive) mode_name = "interactive";
    else if (mode == StreamMode::UltraFast) mode_name = "ultra_fast";
    else if (mode == StreamMode::LongForm) mode_name = "long_form";

    std::fprintf(stderr, "[kokopop] prepared: %zu chunks, %s mode\n",
                plan.chunks.size(), mode_name.c_str());

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

    // Move samples into a vector for postprocessing in a single uninitialised
    // memcpy (constructor form skips the zero-init pass of resize/std::copy).
    std::vector<float> raw_audio(raw.samples, raw.samples + raw.n_samples);
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

            // Phase 2: pipelined inference.
            //
            // Two-stage pipeline: a single async worker produces the next
            // chunk's raw audio while this thread post-processes / crossfades
            // / runs the user callback for the current chunk. The worker uses
            // the model (not thread-safe), this thread does not — so only one
            // thread touches the model at a time.
            //
            // Queue depth = 1 lookahead, which bounds peak memory and keeps
            // stop-on-cancel responsive: at most one chunk of inference is
            // running ahead when the callback returns false.
            const int n_chunks    = static_cast<int>(plan.chunks.size());
            const int sample_rate = model->sample_rate;

            struct RawResult {
                std::vector<float> samples;
                std::string        error;
            };

            auto synthesize_one = [&](int idx) -> RawResult {
                RawResult r;
                if (state_shared->stopped.load()) {
                    r.error = "stopped";
                    return r;
                }
                kokopop_audio raw{};
                if (!synthesize_phonemes(*model,
                                          plan.chunks[static_cast<size_t>(idx)].phonemes,
                                          plan.voice, plan.speed, raw, r.error)) {
                    return r;
                }
                if (raw.n_samples > 0 && raw.samples != nullptr) {
                    r.samples.assign(raw.samples, raw.samples + raw.n_samples);
                }
                kokopop_audio_free(&raw);
                return r;
            };

            std::vector<float> prev_tail;
            std::future<RawResult> next_future;
            if (n_chunks > 0) {
                next_future = std::async(std::launch::async, synthesize_one, 0);
            }

            for (int i = 0; i < n_chunks; ++i) {
                if (state_shared->stopped.load()) {
                    // Drain the in-flight future so its worker thread exits
                    // cleanly before we leave the scope.
                    if (next_future.valid()) (void)next_future.get();
                    break;
                }

                RawResult cur = next_future.get();

                // Kick off inference for chunk i+1 in parallel with this
                // chunk's post-processing and callback.
                if (i + 1 < n_chunks && !state_shared->stopped.load()) {
                    next_future = std::async(std::launch::async, synthesize_one, i + 1);
                }

                if (!cur.error.empty() || cur.samples.empty()) {
                    std::fprintf(stderr, "[kokopop] WARNING chunk[%d]: %s — skipping\n",
                                i, cur.error.empty() ? "no audio" : cur.error.c_str());
                    continue;
                }

                std::fprintf(stderr, "[kokopop] chunk[%d] synthesized: %zu samples (%.1fms)\n",
                            i + 1, cur.samples.size(),
                            (double)cur.samples.size() / sample_rate * 1000.0);

                auto processed = postprocess_chunk_audio(
                    cur.samples, plan.chunks[static_cast<size_t>(i)], i,
                    n_chunks, plan.config, sample_rate);

                if (!prev_tail.empty() && !processed.empty()) {
                    processed = apply_crossfade_smart(
                        prev_tail, processed,
                        plan.chunks[static_cast<size_t>(i)].boundary_after,
                        plan.config.crossfade_ms, sample_rate);
                }

                // Save tail for next chunk's crossfade.
                const int crossfade_samples =
                    (plan.config.crossfade_ms * sample_rate) / 1000;
                if (!processed.empty() && crossfade_samples > 0) {
                    const size_t tail_start =
                        processed.size() > static_cast<size_t>(crossfade_samples)
                            ? processed.size() - static_cast<size_t>(crossfade_samples)
                            : 0;
                    prev_tail.assign(processed.begin() + static_cast<ptrdiff_t>(tail_start),
                                     processed.end());
                } else {
                    prev_tail.clear();
                }

                const bool continue_streaming = (*cb_shared)(
                    processed.data(), processed.size(), i, user_data);
                if (!continue_streaming) {
                    state_shared->stopped.store(true);
                    // Let the in-flight worker finish; it will see the flag
                    // and return quickly (synthesize_phonemes is not
                    // interruptible mid-graph but the next iteration will
                    // exit before doing any more work).
                }
            }

            // Ensure any lingering worker is reaped before declaring done.
            if (next_future.valid()) (void)next_future.get();

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
