#include "test_helpers.h"
#include "sanotts_fixtures.h"
#include "http/http_audio_stream_encoder.h"
#include "http/synthesis_scheduler.h"

#include <chrono>
#include <thread>

// ---- Mixed sample rates through the HTTP path ----
//
// A sanoTTS pack carries 22050 Hz Piperlite voices next to 24000 Hz Vocos
// ones, and the server serves several requests at once from one scheduler.
// The rate therefore has to belong to the request, not to the model: a
// single shared value would stamp one voice's audio with the other's rate.

namespace {

/// Drain one scheduler request to completion, encoding as the async server
/// does: one encoder per request, built from that request's own rate.
std::vector<char> drain_encoded(kokopop::SynthesisScheduler & scheduler,
                                const std::shared_ptr<kokopop::RequestContext> & ctx,
                                int & out_sample_rate) {
    auto encoder = kokopop::make_http_audio_stream_encoder(
        ctx->format, ctx->sample_rate, ctx->ogg_prebuffer_chunks);
    REQUIRE(encoder != nullptr);
    out_sample_rate = ctx->sample_rate;

    std::vector<char> body;
    encoder->start(body);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    bool done = false;
    while (!done && std::chrono::steady_clock::now() < deadline) {
        kokopop::RequestContext::AudioChunk chunk;
        while (ctx->try_pop(chunk)) {
            encoder->write(std::move(chunk), false, body);
        }
        const auto state = ctx->state.load();
        done = state == kokopop::RequestContext::State::DONE ||
               state == kokopop::RequestContext::State::ERROR ||
               state == kokopop::RequestContext::State::CANCELLED;
        if (!done) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    kokopop::RequestContext::AudioChunk chunk;
    while (ctx->try_pop(chunk)) {
        encoder->write(std::move(chunk), false, body);
    }
    CHECK_EQ(ctx->state.load(), kokopop::RequestContext::State::DONE);
    encoder->finish(true, body);
    (void)scheduler;
    return body;
}

int32_t wav_sample_rate(const std::vector<char> & body) {
    // Byte 24 of a canonical RIFF/WAVE header, little endian.
    if (body.size() < 28) return 0;
    if (std::memcmp(body.data(), "RIFF", 4) != 0) return 0;
    int32_t rate = 0;
    std::memcpy(&rate, body.data() + 24, sizeof(rate));
    return rate;
}

} // namespace

TEST_CASE("sanotts_http_mixed_rates_wav_pcm_and_opus_in_flight") {
    std::string why;
    kokopop::test::SanoModel loaded;
    if (!loaded.load("mixed", why)) {
        MESSAGE("skipped: " << why);
        return;
    }

    const bool opus_available = kokopop::http_audio_format_available(
        kokopop::RequestContext::AudioFormat::OGG_OPUS);

    kokopop::SynthesisScheduler scheduler(*loaded.model);

    // Three requests submitted before any is drained, so they are all in the
    // scheduler's queue at the same time.
    auto wav_ctx = scheduler.submit(
        "Hello there, this is the vocos voice.", "heart", 1.0f,
        kokopop::StreamMode::LongForm, kokopop::RequestContext::AudioFormat::WAV);
    auto pcm_ctx = scheduler.submit(
        "Hello there, this is the piperlite voice.", "amy", 1.0f,
        kokopop::StreamMode::LongForm, kokopop::RequestContext::AudioFormat::PCM);
    std::shared_ptr<kokopop::RequestContext> ogg_ctx;
    if (opus_available) {
        ogg_ctx = scheduler.submit(
            "Hello there, once more with the piperlite voice.", "amy", 1.0f,
            kokopop::StreamMode::LongForm,
            kokopop::RequestContext::AudioFormat::OGG_OPUS);
    }

    // The rate is fixed at submit time, from the request's own voice.
    CHECK_EQ(wav_ctx->sample_rate, 24000);
    CHECK_EQ(pcm_ctx->sample_rate, 22050);

    int wav_rate = 0;
    int pcm_rate = 0;
    const std::vector<char> wav_body = drain_encoded(scheduler, wav_ctx, wav_rate);
    const std::vector<char> pcm_body = drain_encoded(scheduler, pcm_ctx, pcm_rate);
    std::vector<char> ogg_body;
    int ogg_rate = 0;
    if (ogg_ctx) {
        ogg_body = drain_encoded(scheduler, ogg_ctx, ogg_rate);
    }

    scheduler.stop();
    scheduler.join();

    CHECK_EQ(wav_rate, 24000);
    CHECK_EQ(pcm_rate, 22050);
    CHECK_EQ(wav_sample_rate(wav_body), 24000);

    // Raw PCM carries no rate of its own; what matters is that the WAV
    // container built next to it did not borrow the other voice's.
    CHECK(pcm_body.size() > 4096);
    CHECK_EQ(pcm_body.size() % sizeof(float), 0u);

    if (ogg_ctx) {
        CHECK_EQ(ogg_rate, 22050);
        REQUIRE(ogg_body.size() > 4);
        CHECK(std::memcmp(ogg_body.data(), "OggS", 4) == 0);
    }
}

TEST_CASE("sanotts_adaptative_session_keeps_its_voice_across_chunks") {
    // The adaptative path stores the frontend's tokenize closure in the
    // session and calls it once per chunk, long after the VoiceFrontend it
    // was built from went out of scope. A closure holding a pointer into that
    // frontend reads a destroyed descriptor here; Kokoro's tokenizer ignores
    // the voice and hides it, sanoTTS's does not.
    std::string why;
    kokopop::test::SanoModel loaded;
    if (!loaded.load("mixed", why)) {
        MESSAGE("skipped: " << why);
        return;
    }

    kokopop::SynthesisSessionOptions options;
    options.voice = "amy";
    options.speed = 1.0f;
    options.mode = kokopop::StreamMode::Adaptative;

    kokopop::SynthesisSession session(*loaded.model, options);
    std::string error;
    REQUIRE(session.push_text(
        "Hello there. This is the second sentence. And a third one to chunk.",
        error));
    REQUIRE(session.finish_input(error));
    CHECK_EQ(session.sample_rate(), 22050);

    size_t total = 0;
    int chunks_seen = 0;
    while (!session.done()) {
        std::vector<kokopop::SynthesisAudioChunk> chunks;
        REQUIRE_MESSAGE(session.next(1, 0, chunks, error), error);
        for (const auto & chunk : chunks) {
            CHECK(chunk.samples.size() > 0);
            total += chunk.samples.size();
            ++chunks_seen;
        }
    }
    CHECK(chunks_seen > 0);
    CHECK(total > 4096);
}

TEST_CASE("voice_frontend_closures_outlive_the_frontend") {
    std::string why;
    kokopop::test::SanoModel loaded;
    if (!loaded.load("mixed", why)) {
        MESSAGE("skipped: " << why);
        return;
    }

    kokopop::PhonemizeFn phonemize;
    kokopop::TokenizeFn tokenize;
    {
        kokopop::VoiceFrontend frontend;
        std::string error;
        REQUIRE(kokopop::make_voice_frontend(*loaded.model, "amy", frontend, error));
        phonemize = frontend.phonemize;
        tokenize = frontend.tokenize;
    }

    std::string phonemes;
    std::string error;
    REQUIRE_MESSAGE(phonemize("Hello there.", phonemes, error), error);
    CHECK(!phonemes.empty());
    std::vector<uint32_t> ids;
    REQUIRE_MESSAGE(tokenize(phonemes, ids, error), error);
    CHECK(!ids.empty());
}
