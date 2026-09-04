#include "http/synthesis_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

namespace kokopop {

SynthesisScheduler::SynthesisScheduler(Model & model)
    : _model(model)
{
    _running.store(true);
    _worker = std::thread(&SynthesisScheduler::_worker_loop, this);
}

SynthesisScheduler::~SynthesisScheduler() {
    stop();
    join();
}

std::shared_ptr<RequestContext> SynthesisScheduler::submit(
    const std::string & text,
    const std::string & voice,
    float speed,
    StreamMode mode,
    RequestContext::AudioFormat format,
    int ogg_prebuffer_chunks,
    const ChunkConfig & chunk_config_override,
    bool has_chunk_config_override,
    const KokoroDiffusionOptions & diffusion)
{
    auto ctx = std::make_shared<RequestContext>();
    ctx->request_id = next_request_id();
    ctx->text = text;
    ctx->voice = voice;
    ctx->speed = speed;
    ctx->mode = mode;
    ctx->format = format;
    ctx->ogg_prebuffer_chunks = std::max(0, ogg_prebuffer_chunks);
    ctx->chunk_config = chunk_config_override;
    ctx->has_chunk_config = has_chunk_config_override;
    ctx->diffusion = diffusion;
    ctx->sample_rate = _model.sample_rate();

    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        _pending.push_back(ctx);
    }
    _queue_cv.notify_one();

    const char * fmt_name =
        format == RequestContext::AudioFormat::WAV     ? "wav"
      : format == RequestContext::AudioFormat::OGG_OPUS ? "ogg"
      :                                                   "pcm";
    std::fprintf(stderr, "[scheduler] request #%u submitted: %zu chars, voice=%s, format=%s\n",
                ctx->request_id, text.size(), voice.c_str(), fmt_name);

    return ctx;
}

void SynthesisScheduler::stop() {
    _running.store(false);
    _queue_cv.notify_all();
}

void SynthesisScheduler::join() {
    if (_worker.joinable()) {
        _worker.join();
    }
}

uint32_t SynthesisScheduler::next_request_id() {
    return _next_id.fetch_add(1);
}

void SynthesisScheduler::_worker_loop() {
    while (_running.load()) {
        std::shared_ptr<RequestContext> ctx;
        size_t queued_after_pop = 0;
        {
            std::unique_lock<std::mutex> lock(_queue_mutex);
            _queue_cv.wait(lock, [&] {
                return !_running.load() || !_pending.empty();
            });
            if (!_running.load()) break;
            if (_pending.empty()) continue;

            ctx = _pending.front();
            _pending.pop_front();
            queued_after_pop = _pending.size();
        }

        // Check cancellation
        if (ctx->cancelled.load()) {
            ctx->state.store(RequestContext::State::CANCELLED);
            continue;
        }

        // ---- Phase 1: PREPARING (chunking + phonemization) ----
        if (ctx->state.load() == RequestContext::State::PREPARING) {
            std::string error;
            SynthesisSessionOptions options;
            options.voice = ctx->voice;
            options.speed = ctx->speed;
            options.mode = ctx->mode;
            options.chunk_config = ctx->chunk_config;
            options.has_chunk_config = ctx->has_chunk_config;
            options.diffusion = ctx->diffusion;

            ctx->synthesis.reset(new SynthesisSession(_model, options));
            if (!ctx->synthesis->push_text(ctx->text, error) ||
                !ctx->synthesis->finish_input(error) ||
                !ctx->synthesis->prepare(error)) {
                ctx->error = error.empty() ? "no chunks produced" : error;
                ctx->state.store(RequestContext::State::ERROR);
                std::fprintf(stderr, "[scheduler] request #%u prepare failed: %s\n",
                            ctx->request_id, ctx->error.c_str());
                continue;
            }
            ctx->chunks_total = ctx->synthesis->chunks_total();

            // Client may have disconnected during preparation
            if (ctx->cancelled.load()) {
                ctx->state.store(RequestContext::State::CANCELLED);
                std::fprintf(stderr, "[scheduler] request #%u cancelled after prepare\n",
                            ctx->request_id);
                continue;
            }

            ctx->state.store(RequestContext::State::INFERRING);
            if (ctx->mode == StreamMode::LongForm) {
                std::fprintf(stderr, "[scheduler] request #%u prepared: %d chunks\n",
                            ctx->request_id, ctx->chunks_total);
            } else {
                std::fprintf(stderr, "[scheduler] request #%u prepared: adaptative mode\n",
                            ctx->request_id);
            }
        }

        // ---- Phase 2: INFERRING (one chunk at a time) ----
        if (ctx->state.load() == RequestContext::State::INFERRING) {
            // Back-pressure: if output queue full, re-enqueue at end
            if (!ctx->output_has_room()) {
                std::lock_guard<std::mutex> lock(_queue_mutex);
                _pending.push_back(ctx);
                continue;
            }

            // Check cancellation before expensive inference
            if (ctx->cancelled.load()) {
                ctx->state.store(RequestContext::State::CANCELLED);
                continue;
            }

            std::string error;
            int idx = ctx->chunks_completed.load();
            std::vector<SynthesisAudioChunk> chunks;
            if (!ctx->synthesis->next(1, queued_after_pop, chunks, error)) {
                ctx->error = error.empty() ? "inference failed" : error;
                ctx->state.store(RequestContext::State::ERROR);
                std::fprintf(stderr, "[scheduler] request #%u chunk %d failed: %s\n",
                            ctx->request_id, idx + 1, ctx->error.c_str());
                continue;
            }
            if (chunks.empty()) {
                ctx->state.store(RequestContext::State::DONE);
                continue;
            }

            for (auto & chunk : chunks) {
                const bool is_last_chunk = chunk.is_final;
                if (is_last_chunk && ctx->chunks_total == 0) {
                    ctx->chunks_total = chunk.chunk_index + 1;
                }
                ctx->push_audio(std::move(chunk.samples), chunk.chunk_index);
                ctx->chunks_completed.fetch_add(1);

                if (ctx->mode == StreamMode::LongForm || is_last_chunk) {
                    std::fprintf(stderr, "[scheduler] request #%u chunk %d/%d\n",
                                ctx->request_id, chunk.chunk_index + 1, ctx->chunks_total);
                } else {
                    std::fprintf(stderr, "[scheduler] request #%u chunk %d/?\n",
                                ctx->request_id, chunk.chunk_index + 1);
                }

                if (is_last_chunk) {
                    ctx->state.store(RequestContext::State::DONE);
                    std::fprintf(stderr, "[scheduler] request #%u DONE\n", ctx->request_id);
                } else {
                    std::lock_guard<std::mutex> lock(_queue_mutex);
                    _pending.push_back(ctx);
                }
            }
        }
    }
}

} // namespace kokopop
