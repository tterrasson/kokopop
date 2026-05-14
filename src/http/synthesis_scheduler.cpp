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
    bool has_chunk_config_override)
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
            if (ctx->mode == StreamMode::LongForm) {
                const ChunkConfig * override_ptr = ctx->has_chunk_config ? &ctx->chunk_config : nullptr;
                auto plan = prepare_synthesis(
                    _model, ctx->text, ctx->voice, ctx->speed, ctx->mode, error, override_ptr);

                if (plan.chunks.empty()) {
                    ctx->error = error.empty() ? "no chunks produced" : error;
                    ctx->state.store(RequestContext::State::ERROR);
                    std::fprintf(stderr, "[scheduler] request #%u prepare failed: %s\n",
                                ctx->request_id, ctx->error.c_str());
                    continue;
                }

                ctx->plan = std::make_shared<SynthesisPlan>(std::move(plan));
                ctx->chunks_total = static_cast<int>(ctx->plan->chunks.size());
            } else {
                ChunkConfig cfg = make_adaptative_config();
                if (ctx->has_chunk_config) {
                    cfg = merge_chunk_config(cfg, ctx->chunk_config);
                }
                ctx->chunk_config = cfg;
                ctx->adaptative_controller.min_tokens = cfg.target_min_tokens;
                ctx->adaptative_controller.max_tokens = cfg.target_max_tokens;

                TokenizeFn tokenize_fn = [this](const std::string & phonemes,
                                                std::vector<uint32_t> & ids,
                                                std::string & err) -> bool {
                    return _model.tokenize_phonemes(phonemes, ids, err);
                };
                ctx->adaptative_units = prepare_chunk_units(
                    ctx->text, ctx->voice, cfg, tokenize_fn, error);
                if (ctx->adaptative_units.empty()) {
                    ctx->error = error.empty() ? "no chunks produced" : error;
                    ctx->state.store(RequestContext::State::ERROR);
                    std::fprintf(stderr, "[scheduler] request #%u adaptative prepare failed: %s\n",
                                ctx->request_id, ctx->error.c_str());
                    continue;
                }
                ctx->chunks_total = 0;
            }

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
                std::fprintf(stderr, "[scheduler] request #%u prepared: %zu adaptative units\n",
                            ctx->request_id, ctx->adaptative_units.size());
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

            std::vector<float> out_tail;
            std::string error;
            int idx = ctx->chunks_completed.load();
            int n_tokens = 0;
            bool is_last_chunk = false;
            std::vector<float> audio;

            if (ctx->mode == StreamMode::LongForm) {
                if (idx >= ctx->chunks_total) {
                    ctx->state.store(RequestContext::State::DONE);
                    continue;
                }
                n_tokens = ctx->plan->chunks[static_cast<size_t>(idx)].n_tokens;
                audio = infer_chunk(
                    _model, *ctx->plan, idx, ctx->prev_tail, out_tail, error);
                is_last_chunk = idx >= ctx->chunks_total - 1;
            } else {
                const int target_tokens =
                    ctx->adaptative_controller.target_tokens(queued_after_pop);
                Chunk chunk = build_adaptative_chunk(
                    ctx->adaptative_units,
                    ctx->adaptative_next_unit,
                    ctx->chunk_config,
                    target_tokens,
                    idx == 0);
                if (chunk.tokens.empty()) {
                    ctx->state.store(RequestContext::State::DONE);
                    continue;
                }

                n_tokens = chunk.n_tokens;
                is_last_chunk = chunk.is_last;
                SynthesisPlan one_chunk_plan;
                one_chunk_plan.chunks.push_back(std::move(chunk));
                one_chunk_plan.voice = ctx->voice;
                one_chunk_plan.speed = ctx->speed;
                one_chunk_plan.mode = ctx->mode;
                one_chunk_plan.config = ctx->chunk_config;

                auto start = std::chrono::steady_clock::now();
                audio = infer_chunk(
                    _model, one_chunk_plan, 0, ctx->prev_tail, out_tail, error);
                const auto end = std::chrono::steady_clock::now();
                const double generation_ms =
                    std::chrono::duration<double, std::milli>(end - start).count();
                ctx->adaptative_controller.observe(n_tokens, generation_ms);
                ctx->chunks_total = is_last_chunk ? idx + 1 : 0;
                const double audio_ms = static_cast<double>(audio.size()) /
                    static_cast<double>(_model.sample_rate) * 1000.0;
                const int next_target = ctx->adaptative_controller.target_tokens(
                    queued_after_pop);
                std::fprintf(stderr,
                    "[scheduler] request #%u chunk %d: gen=%.0fms audio=%.0fms"
                    " rtf=%.2f → next_target=%d tokens\n",
                    ctx->request_id, idx + 1,
                    generation_ms, audio_ms,
                    generation_ms / (audio_ms > 0 ? audio_ms : 1.0),
                    next_target);
            }

            if (!audio.empty()) {
                ctx->prev_tail = std::move(out_tail);
                ctx->push_audio(std::move(audio), idx);
                ctx->chunks_completed.fetch_add(1);

                if (ctx->mode == StreamMode::LongForm || is_last_chunk) {
                    std::fprintf(stderr, "[scheduler] request #%u chunk %d/%d: %d tokens\n",
                                ctx->request_id, idx + 1, ctx->chunks_total,
                                n_tokens);
                } else {
                    std::fprintf(stderr, "[scheduler] request #%u chunk %d/?: %d tokens\n",
                                ctx->request_id, idx + 1, n_tokens);
                }

                // Check if this was the last chunk
                if (is_last_chunk) {
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
