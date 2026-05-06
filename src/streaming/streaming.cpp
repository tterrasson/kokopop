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
// Internal: generate a single chunk of audio
// ---------------------------------------------------------------------------
namespace {

bool generate_chunk(
    kokopop::Model & model,
    const Chunk & chunk,
    const std::string & voice,
    float speed,
    std::vector<float> & out_audio,
    std::string & error) {

    // Synthesize phonemes
    kokopop_audio raw{};
    if (!synthesize_phonemes(model, chunk.phonemes, voice, speed, raw, error)) {
        return false;
    }

    if (raw.n_samples == 0) {
        kokopop_audio_free(&raw);
        error = "synthesis produced no audio";
        return false;
    }

    // Copy to vector for postprocessing
    out_audio.assign(raw.samples, raw.samples + raw.n_samples);
    kokopop_audio_free(&raw);

    return true;
}

} // namespace

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
            // Select config
            ChunkConfig config = (mode == StreamMode::Interactive)
                ? make_interactive_config()
                : make_long_form_config();

            // Tokenize function wrapper
            TokenizeFn tokenize_fn = [model](const std::string & phonemes,
                                           std::vector<uint32_t> & ids,
                                           std::string & error) -> bool {
                return model->tokenize_phonemes(phonemes, ids, error);
            };

            // Chunk the text
            std::string error;
            auto chunks = chunk_text(text_copy, voice_copy, config, tokenize_fn, error);
            if (chunks.empty()) {
                state_shared->done.store(true);
                return;
            }

            // Generate audio for each chunk
            std::vector<float> prev_tail; // for crossfade
            const int crossfade_samples = (config.crossfade_ms * model->sample_rate) / 1000;

            std::fprintf(stderr, "[kokopop] streaming: %zu chunks total\n", chunks.size());

            for (int i = 0; i < static_cast<int>(chunks.size()); ++i) {
                if (state_shared->stopped.load()) break;

                auto & chunk = chunks[i];

                std::fprintf(stderr, "[kokopop] chunk[%d/%d]: %d tokens, phonemes=%zu chars\n",
                            i, static_cast<int>(chunks.size()),
                            chunk.n_tokens, chunk.phonemes.size());

                // Generate raw audio
                std::vector<float> raw_audio;
                if (!generate_chunk(*model, chunk, voice_copy, speed, raw_audio, error)) {
                    std::fprintf(stderr, "[kokopop] WARNING chunk[%d]: %s — skipping\n", i, error.c_str());
                    continue;
                }

                std::fprintf(stderr, "[kokopop] chunk[%d] synthesized: %zu samples (%.1fms)\n",
                            i, raw_audio.size(),
                            (double)raw_audio.size() / model->sample_rate * 1000.0);

                // Post-process
                auto processed = postprocess_chunk_audio(
                    raw_audio, chunk, i, static_cast<int>(chunks.size()),
                    config, model->sample_rate);

                // Apply crossfade with previous chunk
                if (!prev_tail.empty() && !processed.empty()) {
                    auto crossed = apply_crossfade_smart(
                        prev_tail, processed,
                        chunk.boundary_after,
                        config.crossfade_ms,
                        model->sample_rate);
                    processed = std::move(crossed);
                }

                // Save tail for next crossfade
                if (!processed.empty() && crossfade_samples > 0) {
                    size_t tail_start = processed.size() > static_cast<size_t>(crossfade_samples)
                        ? processed.size() - crossfade_samples
                        : 0;
                    prev_tail.assign(processed.begin() + tail_start, processed.end());
                }

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

    ChunkConfig config = (mode_ == StreamMode::Interactive)
        ? make_interactive_config()
        : make_long_form_config();

    TokenizeFn tokenize_fn = [&](const std::string & phonemes,
                                   std::vector<uint32_t> & ids,
                                   std::string & error) -> bool {
        return model_.tokenize_phonemes(phonemes, ids, error);
    };

    std::string error;
    auto chunks = chunk_text(text, voice_, config, tokenize_fn, error);
    if (chunks.empty()) return;

    std::vector<float> prev_tail;
    const int crossfade_samples = (config.crossfade_ms * model_.sample_rate) / 1000;

    for (int i = 0; i < static_cast<int>(chunks.size()); ++i) {
        if (stopped_.load()) break;

        auto & chunk = chunks[i];
        std::vector<float> raw_audio;
        if (!generate_chunk(model_, chunk, voice_, speed_, raw_audio, error)) {
            std::fprintf(stderr, "[kokopop] WARNING chunk[%d]: %s — skipping\n",
                        chunk_counter_, error.c_str());
            continue;
        }

        auto processed = postprocess_chunk_audio(
            raw_audio, chunk, chunk_counter_,
            chunk_counter_ + static_cast<int>(chunks.size()),
            config, model_.sample_rate);

        if (!prev_tail.empty() && !processed.empty()) {
            auto crossed = apply_crossfade_smart(
                prev_tail, processed,
                chunk.boundary_after,
                config.crossfade_ms,
                model_.sample_rate);
            processed = std::move(crossed);
        }

        if (!processed.empty() && crossfade_samples > 0) {
            size_t tail_start = processed.size() > static_cast<size_t>(crossfade_samples)
                ? processed.size() - crossfade_samples
                : 0;
            prev_tail.assign(processed.begin() + tail_start, processed.end());
        }

        callback_(processed.data(), processed.size(), chunk_counter_, user_data_);
        ++chunk_counter_;
    }
}

void IncrementalStreamer::stop() {
    stopped_.store(true);
}

} // namespace kokopop
