#include "test_helpers.h"
#include "sanotts_fixtures.h"

#include "arch/sanotts/sano_arch.h"
#include "model/model.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

// Decoder parity.
//
// The gates are per family, because the two are not being asked the same
// question:
//
//   * `heart` (F32) and the piperlite voices are compared against a reference
//     that computes the same arithmetic in the same precision, so anything
//     short of correlation 0.999 / RMS error 1e-3 is a bug;
//   * `heartnano` is a dequantised int8 model compared against a runtime that
//     also quantises its *activations*. 0.98 is the gate upstream holds its
//     own lineages to, and it is deliberately looser.
//
// Every case runs with `dur_override`, so a duration difference cannot be
// mistaken for a decoder regression: the two are separate tests.

namespace {

std::string row_stem(int row) {
    char stem[16];
    std::snprintf(stem, sizeof stem, "r%02d", row);
    return stem;
}

/// The per-row noise seeds of an upstream vocos fixture, from its `rows.txt`.
std::vector<uint64_t> read_seeds(const std::string & voice) {
    std::vector<uint64_t> seeds;
    const std::string path = kokopop::test::sanotts_fixture_path(voice, "rows.txt");
    if (path.empty()) {
        return seeds;
    }
    std::ifstream in(path);
    std::string id;
    long long tokens = 0;
    long long frames = 0;
    long long samples = 0;
    unsigned long long seed = 0;
    while (in >> id >> tokens >> frames >> samples >> seed) {
        seeds.push_back(seed);
    }
    return seeds;
}

/// One fixture row rendered with its durations pinned.
bool render_row(kokopop::SanoArch & arch, const kokopop::VoiceDesc & voice,
                const std::vector<int32_t> & ids, const std::vector<int32_t> & durations,
                bool has_seed, uint64_t seed, std::vector<float> & audio,
                std::string & error) {
    kokopop::SynthesisExtras extras;
    extras.dur_override = durations;
    extras.has_noise_seed = has_seed;
    extras.noise_seed = seed;

    kokopop::SanoProbe probe;
    if (!arch.run(std::vector<uint32_t>(ids.begin(), ids.end()), voice, 1.0f,
                  extras, probe, error)) {
        return false;
    }
    audio = std::move(probe.audio);
    return true;
}

/// Runs every row of an upstream vocos fixture and returns the worst metrics.
void check_vocos_fixture(const std::string & model_name, const std::string & voice_name,
                         const std::string & fixture, double min_correlation,
                         double max_rms_error) {
    kokopop::test::SanoModel loaded;
    std::string why;
    if (!loaded.load(model_name, why)) {
        MESSAGE("skipped: " << why);
        return;
    }
    const std::vector<uint64_t> seeds = read_seeds(fixture);
    if (seeds.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_fixture_hint());
        return;
    }

    const kokopop::VoiceDesc * voice = loaded.arch->find_voice(voice_name);
    REQUIRE(voice != nullptr);

    int rows = 0;
    for (size_t row = 0; row < seeds.size(); ++row) {
        const std::string stem = row_stem(static_cast<int>(row));
        const std::string ids_path =
            kokopop::test::sanotts_fixture_path(fixture, stem + "_ids.bin");
        const std::string durs_path =
            kokopop::test::sanotts_fixture_path(fixture, stem + "_durs.bin");
        const std::string audio_path =
            kokopop::test::sanotts_fixture_path(fixture, stem + "_audio.bin");
        if (ids_path.empty() || durs_path.empty() || audio_path.empty()) {
            continue;
        }

        std::vector<int32_t> ids;
        std::vector<int32_t> durations;
        std::vector<float> expected;
        REQUIRE(kokopop::test::read_i32_file(ids_path, ids));
        REQUIRE(kokopop::test::read_i32_file(durs_path, durations));
        REQUIRE(kokopop::test::read_f32_file(audio_path, expected));

        std::vector<float> audio;
        std::string error;
        REQUIRE_MESSAGE(render_row(*loaded.arch, *voice, ids, durations, true,
                                   seeds[row], audio, error), error);

        // Length is part of the contract, not something to align away: the
        // iSTFT of `frames` frames is exactly `(frames - 1) * hop` samples.
        CHECK_EQ(audio.size(), expected.size());

        const auto comparison = kokopop::test::compare_audio(audio, expected);
        INFO(voice_name << " " << stem << ": correlation " << comparison.correlation
             << ", RMS error " << comparison.rms_error
             << ", RMS ratio " << comparison.rms_ratio);
        CHECK(comparison.correlation >= min_correlation);
        CHECK(comparison.rms_error <= max_rms_error);
        // A gain or DC drift would leave the correlation intact while making
        // the rendering wrong, so both are checked separately.
        CHECK(std::fabs(comparison.rms_ratio - 1.0) <= 0.02);
        ++rows;
    }
    CHECK_EQ(rows, static_cast<int>(seeds.size()));
}

} // namespace

TEST_CASE("sanotts_vocos_decoder_matches_the_heart_golden_fixture") {
    check_vocos_fixture("heart", "heart", "en_us_r227f32", 0.999, 1e-3);
}

// heartnano is the dequantised int8 lineage against a reference whose activations are quantised too
TEST_CASE("sanotts_vocos_decoder_matches_the_heartnano_golden_fixture") {
    check_vocos_fixture("heartnano", "heartnano", "en_us_e13b", 0.98, 1e-2);
}

TEST_CASE("sanotts_piperlite_decoder_matches_the_reference_forward_pass") {
    kokopop::test::SanoModel loaded;
    std::string why;
    if (!loaded.load("mixed", why)) {
        MESSAGE("skipped: " << why);
        return;
    }
    if (kokopop::test::sanotts_piperlite_fixture_path("amy", "r00_audio.bin").empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_piperlite_fixture_hint());
        return;
    }

    int rows = 0;
    for (const char * name : {"amy", "kristin"}) {
        const kokopop::VoiceDesc * voice = loaded.arch->find_voice(name);
        REQUIRE(voice != nullptr);

        for (int row = 0; row < 8; ++row) {
            const std::string stem = row_stem(row);
            const std::string ids_path =
                kokopop::test::sanotts_piperlite_fixture_path(name, stem + "_ids.bin");
            const std::string durs_path =
                kokopop::test::sanotts_piperlite_fixture_path(name, stem + "_durs.bin");
            const std::string audio_path =
                kokopop::test::sanotts_piperlite_fixture_path(name, stem + "_audio.bin");
            if (ids_path.empty() || durs_path.empty() || audio_path.empty()) {
                continue;
            }

            std::vector<int32_t> ids;
            std::vector<int32_t> durations;
            std::vector<float> expected;
            REQUIRE(kokopop::test::read_i32_file(ids_path, ids));
            REQUIRE(kokopop::test::read_i32_file(durs_path, durations));
            REQUIRE(kokopop::test::read_f32_file(audio_path, expected));

            std::vector<float> audio;
            std::string error;
            REQUIRE_MESSAGE(render_row(*loaded.arch, *voice, ids, durations, false, 0,
                                       audio, error), error);
            CHECK_EQ(audio.size(), expected.size());

            const auto comparison = kokopop::test::compare_audio(audio, expected);
            INFO(name << " " << stem << ": correlation " << comparison.correlation
                 << ", RMS error " << comparison.rms_error);
            CHECK(comparison.correlation >= 0.999);
            CHECK(comparison.rms_error <= 1e-3);
            CHECK(std::fabs(comparison.rms_ratio - 1.0) <= 0.02);
            ++rows;
        }
    }
    CHECK_EQ(rows, 16);
}

// The whole point of the seed contract: the same request renders identically,
// and a different chunk index does not.
TEST_CASE("sanotts_vocos_noise_seed_governs_the_rendering") {
    kokopop::test::SanoModel loaded;
    std::string why;
    if (!loaded.load("heart", why)) {
        MESSAGE("skipped: " << why);
        return;
    }
    const std::string ids_path =
        kokopop::test::sanotts_fixture_path("en_us_r227f32", "r00_ids.bin");
    const std::string durs_path =
        kokopop::test::sanotts_fixture_path("en_us_r227f32", "r00_durs.bin");
    if (ids_path.empty() || durs_path.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_fixture_hint());
        return;
    }

    std::vector<int32_t> ids;
    std::vector<int32_t> durations;
    REQUIRE(kokopop::test::read_i32_file(ids_path, ids));
    REQUIRE(kokopop::test::read_i32_file(durs_path, durations));

    const kokopop::VoiceDesc * voice = loaded.arch->find_voice("heart");
    REQUIRE(voice != nullptr);

    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    std::string error;
    REQUIRE(render_row(*loaded.arch, *voice, ids, durations, true, 7ull, a, error));
    REQUIRE(render_row(*loaded.arch, *voice, ids, durations, true, 7ull, b, error));
    REQUIRE(render_row(*loaded.arch, *voice, ids, durations, true, 8ull, c, error));

    CHECK_EQ(a, b);
    CHECK(a != c);

    // No seed at all is still reproducible: it is derived from the model's
    // provenance, the voice and the ids.
    kokopop::SynthesisExtras extras;
    extras.dur_override = durations;
    kokopop::SanoProbe first;
    kokopop::SanoProbe second;
    const std::vector<uint32_t> id_vector(ids.begin(), ids.end());
    REQUIRE(loaded.arch->run(id_vector, *voice, 1.0f, extras, first, error));
    REQUIRE(loaded.arch->run(id_vector, *voice, 1.0f, extras, second, error));
    CHECK_EQ(first.audio, second.audio);

    // A later chunk of the same utterance must not reuse the draw.
    extras.chunk_index = 1;
    kokopop::SanoProbe later;
    REQUIRE(loaded.arch->run(id_vector, *voice, 1.0f, extras, later, error));
    CHECK(later.audio != first.audio);
}

TEST_CASE("sanotts_rejects_a_dur_override_that_does_not_match_the_tokens") {
    kokopop::test::SanoModel loaded;
    std::string why;
    if (!loaded.load("heart", why)) {
        MESSAGE("skipped: " << why);
        return;
    }
    const kokopop::VoiceDesc * voice = loaded.arch->find_voice("heart");
    REQUIRE(voice != nullptr);

    const std::vector<uint32_t> ids{1, 20, 30, 2};
    kokopop::SanoProbe probe;
    std::string error;

    kokopop::SynthesisExtras wrong_length;
    wrong_length.dur_override = {4, 4};
    CHECK_FALSE(loaded.arch->run(ids, *voice, 1.0f, wrong_length, probe, error));
    CHECK(error.find("dur_override") != std::string::npos);

    kokopop::SynthesisExtras out_of_range;
    out_of_range.dur_override = {4, 4, 4, 100000};
    error.clear();
    CHECK_FALSE(loaded.arch->run(ids, *voice, 1.0f, out_of_range, probe, error));
    CHECK(error.find("dur_override") != std::string::npos);

    kokopop::SynthesisExtras zero;
    zero.dur_override = {4, 4, 4, 0};
    error.clear();
    CHECK_FALSE(loaded.arch->run(ids, *voice, 1.0f, zero, probe, error));
}
