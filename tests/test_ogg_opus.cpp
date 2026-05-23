#include "test_helpers.h"

#include <cmath>

namespace {

// Returns true if any 5-byte window in `bytes` matches the Ogg page capture
// pattern "OggS". The encoder may emit other Ogg containers/pages before our
// data, so we just verify the magic appears at least once.
bool contains_ogg_magic(const uint8_t * data, size_t size) {
    if (size < 4) return false;
    for (size_t i = 0; i + 4 <= size; ++i) {
        if (data[i] == 'O' && data[i + 1] == 'g' && data[i + 2] == 'g' && data[i + 3] == 'S') {
            return true;
        }
    }
    return false;
}

} // namespace

TEST_CASE("ogg_opus_encoder_roundtrip") {
    kokopop_encoder_options options{};
    options.format = KOKOPOP_AUDIO_OGG_OPUS;
    options.sample_rate = 24000;
    options.ogg_prebuffer_chunks = 0;

    kokopop_audio_encoder * encoder = nullptr;
    int rc = kokopop_audio_encoder_create(&options, &encoder);
    if (rc == KOKOPOP_ERROR_IO) {
        // Build without libopusenc — Ogg/Opus path unavailable. Verify the
        // error message is set and skip the rest.
        CHECK(std::strlen(kokopop_last_error()) > 0);
        return;
    }
    REQUIRE_EQ(rc, KOKOPOP_OK);
    REQUIRE(encoder != nullptr);

    std::vector<uint8_t> stream;
    kokopop_bytes hdr{};
    CHECK_EQ(kokopop_audio_encoder_start(encoder, &hdr), KOKOPOP_OK);
    stream.insert(stream.end(), hdr.data, hdr.data + hdr.size);
    kokopop_bytes_free(&hdr);

    // Feed ~0.5s of a 440Hz sine at 24kHz. Use several pushes to exercise
    // page accumulation and the final-flag path.
    constexpr int sr = 24000;
    constexpr int total = sr / 2;
    constexpr int block = 1024;
    std::vector<float> samples(block);
    int written = 0;
    int idx = 0;
    while (written < total) {
        int n = std::min(block, total - written);
        for (int i = 0; i < n; ++i) {
            samples[i] = 0.25f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(idx++) / sr);
        }
        const bool is_final = (written + n) >= total;
        kokopop_bytes chunk{};
        CHECK_EQ(kokopop_audio_encoder_push(encoder, samples.data(), n, is_final ? 1 : 0, &chunk), KOKOPOP_OK);
        stream.insert(stream.end(), chunk.data, chunk.data + chunk.size);
        kokopop_bytes_free(&chunk);
        written += n;
    }

    kokopop_bytes tail{};
    CHECK_EQ(kokopop_audio_encoder_finish(encoder, 1, &tail), KOKOPOP_OK);
    stream.insert(stream.end(), tail.data, tail.data + tail.size);
    kokopop_bytes_free(&tail);

    CHECK(stream.size() > 0);
    CHECK(contains_ogg_magic(stream.data(), stream.size()));

    kokopop_audio_encoder_free(encoder);
}

TEST_CASE("ogg_opus_encoder_invalid_format") {
    kokopop_encoder_options options{};
    options.format = 999; // not a valid enum value
    options.sample_rate = 24000;
    kokopop_audio_encoder * encoder = nullptr;
    CHECK_EQ(kokopop_audio_encoder_create(&options, &encoder), KOKOPOP_ERROR_INVALID_ARGUMENT);
    CHECK(encoder == nullptr);
}
