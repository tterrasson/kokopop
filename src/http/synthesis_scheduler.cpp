#include "http/synthesis_scheduler.h"

#include <algorithm>
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
    RequestContext::AudioFormat format)
{
    auto ctx = std::make_shared<RequestContext>();
    ctx->request_id = next_request_id();
    ctx->text = text;
    ctx->voice = voice;
    ctx->speed = speed;
    ctx->mode = mode;
    ctx->format = format;
    ctx->sample_rate = _model.sample_rate;

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
        {
            std::unique_lock<std::mutex> lock(_queue_mutex);
            _queue_cv.wait(lock, [&] {
                return !_running.load() || !_pending.empty();
            });
            if (!_running.load()) break;
            if (_pending.empty()) continue;

            ctx = _pending.front();
            _pending.pop_front();
        }

        // Check cancellation
        if (ctx->cancelled.load()) {
            ctx->state.store(RequestContext::State::CANCELLED);
            continue;
        }

        // ---- Phase 1: PREPARING (chunking + phonemization) ----
        if (ctx->state.load() == RequestContext::State::PREPARING) {
            std::string error;
            auto plan = prepare_synthesis(
                _model, ctx->text, ctx->voice, ctx->speed, ctx->mode, error);

            if (plan.chunks.empty()) {
                ctx->error = error.empty() ? "no chunks produced" : error;
                ctx->state.store(RequestContext::State::ERROR);
                std::fprintf(stderr, "[scheduler] request #%u prepare failed: %s\n",
                            ctx->request_id, ctx->error.c_str());
                continue;
            }

            ctx->plan = std::make_shared<SynthesisPlan>(std::move(plan));
            ctx->chunks_total = static_cast<int>(ctx->plan->chunks.size());
            ctx->state.store(RequestContext::State::INFERRING);

            std::fprintf(stderr, "[scheduler] request #%u prepared: %d chunks\n",
                        ctx->request_id, ctx->chunks_total);
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

            int idx = ctx->chunks_completed.load();
            if (idx >= ctx->chunks_total) {
                // Shouldn't happen but safety check
                ctx->state.store(RequestContext::State::DONE);
                continue;
            }

            std::vector<float> out_tail;
            std::string error;
            auto audio = infer_chunk(
                _model, *ctx->plan, idx, ctx->prev_tail, out_tail, error);

            if (!audio.empty()) {
                ctx->prev_tail = std::move(out_tail);
                ctx->push_audio(std::move(audio), idx);
                ctx->chunks_completed.fetch_add(1);

                std::fprintf(stderr, "[scheduler] request #%u chunk %d/%d: %d tokens\n",
                            ctx->request_id, idx + 1, ctx->chunks_total,
                            ctx->plan->chunks[idx].n_tokens);

                // Check if this was the last chunk
                if (idx >= ctx->chunks_total - 1) {
                    ctx->state.store(RequestContext::State::DONE);
                    std::fprintf(stderr, "[scheduler] request #%u DONE\n", ctx->request_id);
                } else {
                    // More chunks — re-enqueue at end (round-robin)
                    std::lock_guard<std::mutex> lock(_queue_mutex);
                    _pending.push_back(ctx);
                }
            } else {
                ctx->error = error.empty() ? "inference failed" : error;
                ctx->state.store(RequestContext::State::ERROR);
                std::fprintf(stderr, "[scheduler] request #%u chunk %d failed: %s\n",
                            ctx->request_id, idx + 1, ctx->error.c_str());
            }
        }
    }
}

} // namespace kokopop
