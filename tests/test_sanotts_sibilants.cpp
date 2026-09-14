#include "arch/sanotts/sano_arch.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Sibilant noise injection.
//
// A deterministic acoustic student regresses the *mean* latent, and the mean of
// broadband hiss is a whistle, so the reference puts the teacher's per-channel
// variance back at /s z ʃ ʒ/ frames before decoding. What has to hold here is
// not a particular noise stream — the draw is independent per frame and per
// channel, so this runtime and the reference's numpy generator render different
// noise and the same voice — but *where* the noise lands, *how loud* it is, and
// that the same request renders identically twice.

namespace {

/// A voice with nothing but a sibilant calibration: the injection reads the
/// channel count from `channel_std` and the frames from the durations, so no
/// weights are involved.
kokopop::SanoVoice sibilant_voice(float beta, const std::vector<float> & std_dev,
                                  const std::vector<uint32_t> & ids) {
    kokopop::SanoVoice voice;
    voice.sib.beta = beta;
    voice.sib.channel_std = std_dev;
    voice.sib.phoneme_ids = ids;  // the loader sorts; these are already sorted
    return voice;
}

} // namespace

TEST_CASE("sanotts_sibilant_noise_lands_only_on_the_calibrated_phonemes") {
    const int64_t channels = 8;
    const std::vector<int32_t> durations = {3, 2, 4, 1};
    const std::vector<uint32_t> ids = {10, 31, 11, 38};  // tokens 1 and 3 hiss
    const int64_t frames = 10;

    const kokopop::SanoVoice voice =
        sibilant_voice(2.0f, std::vector<float>(channels, 0.5f), {31, 38});

    std::vector<float> latent(static_cast<size_t>(channels * frames), 0.0f);
    std::string error;
    REQUIRE(kokopop::sano_inject_sibilant_noise(voice, ids, durations, frames, 7,
                                                latent, error));

    // Tokens 0 and 2 cover frames [0,3) and [5,9); 1 and 3 cover [3,5) and [9,10).
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t f = 0; f < frames; ++f) {
            const bool hissing = (f >= 3 && f < 5) || f == 9;
            const float value = latent[static_cast<size_t>(c * frames + f)];
            if (hissing) {
                CHECK(value != 0.0f);
            } else {
                CHECK_EQ(value, 0.0f);
            }
        }
    }
}

TEST_CASE("sanotts_sibilant_noise_is_scaled_per_channel") {
    // One long sibilant token, so the per-channel standard deviation of what
    // lands is measurable: it must be `beta * channel_std[c]`, not a single
    // scale shared across channels.
    const int64_t channels = 4;
    const int64_t frames = 4096;
    const std::vector<float> std_dev = {0.25f, 0.5f, 1.0f, 2.0f};
    const kokopop::SanoVoice voice = sibilant_voice(3.0f, std_dev, {31});

    std::vector<float> latent(static_cast<size_t>(channels * frames), 0.0f);
    std::string error;
    REQUIRE(kokopop::sano_inject_sibilant_noise(
        voice, {31}, {static_cast<int32_t>(frames)}, frames, 99, latent, error));

    for (int64_t c = 0; c < channels; ++c) {
        double sum = 0.0;
        const float * row = latent.data() + c * frames;
        for (int64_t f = 0; f < frames; ++f) {
            sum += static_cast<double>(row[f]) * row[f];
        }
        const double measured = std::sqrt(sum / static_cast<double>(frames));
        const double expected = 3.0 * static_cast<double>(std_dev[static_cast<size_t>(c)]);
        // 4096 samples: the sample standard deviation is within ~1.1% of the
        // true one with room to spare.
        CHECK(std::fabs(measured - expected) < expected * 0.05);
    }
}

TEST_CASE("sanotts_sibilant_noise_repeats_for_the_same_seed_and_not_across_seeds") {
    const int64_t channels = 4;
    const int64_t frames = 64;
    const kokopop::SanoVoice voice =
        sibilant_voice(1.0f, std::vector<float>(channels, 1.0f), {31});

    std::vector<float> first(static_cast<size_t>(channels * frames), 0.0f);
    std::vector<float> again = first;
    std::vector<float> other = first;
    std::string error;
    const std::vector<uint32_t> ids = {31};
    const std::vector<int32_t> durations = {static_cast<int32_t>(frames)};

    REQUIRE(kokopop::sano_inject_sibilant_noise(voice, ids, durations, frames, 4242,
                                                first, error));
    REQUIRE(kokopop::sano_inject_sibilant_noise(voice, ids, durations, frames, 4242,
                                                again, error));
    REQUIRE(kokopop::sano_inject_sibilant_noise(voice, ids, durations, frames, 4243,
                                                other, error));
    CHECK(first == again);
    CHECK(first != other);
}

// Which frames hiss must not change the stream: a caller who split a line into
// chunks at a different place would otherwise get a different rendering of the
// half that did not move.
TEST_CASE("sanotts_sibilant_noise_indexes_one_draw_rather_than_drawing_per_run") {
    const int64_t channels = 4;
    const int64_t frames = 32;
    const kokopop::SanoVoice voice =
        sibilant_voice(1.0f, std::vector<float>(channels, 1.0f), {31});

    std::vector<float> whole(static_cast<size_t>(channels * frames), 0.0f);
    std::vector<float> tail_only = whole;
    std::string error;

    REQUIRE(kokopop::sano_inject_sibilant_noise(
        voice, {31}, {static_cast<int32_t>(frames)}, frames, 5, whole, error));
    // Same line, but only its second half is a sibilant.
    REQUIRE(kokopop::sano_inject_sibilant_noise(
        voice, {10, 31}, {16, 16}, frames, 5, tail_only, error));

    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t f = 16; f < frames; ++f) {
            const size_t i = static_cast<size_t>(c * frames + f);
            CHECK_EQ(tail_only[i], whole[i]);
        }
    }
}

TEST_CASE("sanotts_a_voice_without_a_calibration_is_left_alone") {
    kokopop::SanoVoice voice;  // no calibration at all
    std::vector<float> latent(64, 1.5f);
    const std::vector<float> before = latent;
    std::string error;
    REQUIRE(kokopop::sano_inject_sibilant_noise(voice, {31}, {8}, 8, 1, latent, error));
    CHECK(latent == before);
}
