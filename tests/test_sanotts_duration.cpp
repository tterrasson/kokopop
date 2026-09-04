#include "test_helpers.h"
#include "sanotts_fixtures.h"

#include "arch/sanotts/sano_arch.h"
#include "model/model.h"

#include <memory>
#include <numeric>
#include <string>
#include <vector>

// Duration parity.
//
// `ids -> durations` is the one stage of the pipeline with no tolerance to
// grant: the values are integers, they set the length of everything
// downstream, and a single token off by one desynchronises the rest of the
// chunk. So this compares them exactly.
//
// The vocos fixtures are upstream's own; the piperlite ones come from the
// reference numpy forward pass (tools/gen_sanotts_fixtures.py). Both skip when
// the data is absent, which is the normal state of a fresh clone.

namespace {

std::vector<uint32_t> as_ids(const std::vector<int32_t> & values) {
    return std::vector<uint32_t>(values.begin(), values.end());
}

} // namespace

TEST_CASE("sanotts_duration_matches_the_heart_golden_fixture_exactly") {
    kokopop::test::SanoModel loaded;
    std::string why;
    if (!loaded.load("heart", why)) {
        MESSAGE("skipped: " << why);
        return;
    }
    const std::string probe =
        kokopop::test::sanotts_fixture_path("en_us_r227f32", "r00_ids.bin");
    if (probe.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_fixture_hint());
        return;
    }

    const kokopop::VoiceDesc * voice = loaded.arch->find_voice("heart");
    REQUIRE(voice != nullptr);

    int rows = 0;
    for (int row = 0; row < 8; ++row) {
        char stem[16];
        std::snprintf(stem, sizeof stem, "r%02d", row);

        std::vector<int32_t> ids;
        std::vector<int32_t> expected;
        const std::string ids_path =
            kokopop::test::sanotts_fixture_path("en_us_r227f32", std::string(stem) + "_ids.bin");
        const std::string durs_path =
            kokopop::test::sanotts_fixture_path("en_us_r227f32", std::string(stem) + "_durs.bin");
        if (ids_path.empty() || durs_path.empty()) {
            continue;
        }
        REQUIRE(kokopop::test::read_i32_file(ids_path, ids));
        REQUIRE(kokopop::test::read_i32_file(durs_path, expected));
        REQUIRE(ids.size() == expected.size());

        std::vector<int32_t> got;
        std::string error;
        const kokopop::SanoVoice * weights = loaded.arch->voice_for(*voice);
        REQUIRE(weights != nullptr);
        REQUIRE_MESSAGE(kokopop::sano_run_duration(*loaded.arch, *weights, as_ids(ids),
                                                   voice->length_scale, got, error),
                        error);

        CHECK_EQ(got, expected);
        ++rows;
    }
    CHECK_EQ(rows, 8);
}

// `speed` enters the pipeline through `length_scale`, so a faster rendering is
// a strictly shorter one — token by token, not merely on average.
TEST_CASE("sanotts_duration_scales_monotonically_with_speed") {
    kokopop::test::SanoModel loaded;
    std::string why;
    if (!loaded.load("heart", why)) {
        MESSAGE("skipped: " << why);
        return;
    }
    const std::string ids_path =
        kokopop::test::sanotts_fixture_path("en_us_r227f32", "r00_ids.bin");
    if (ids_path.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_fixture_hint());
        return;
    }

    std::vector<int32_t> ids;
    REQUIRE(kokopop::test::read_i32_file(ids_path, ids));

    const kokopop::VoiceDesc * voice = loaded.arch->find_voice("heart");
    REQUIRE(voice != nullptr);
    const kokopop::SanoVoice * weights = loaded.arch->voice_for(*voice);
    REQUIRE(weights != nullptr);

    std::vector<int32_t> slow;
    std::vector<int32_t> normal;
    std::vector<int32_t> fast;
    std::string error;
    REQUIRE(kokopop::sano_run_duration(*loaded.arch, *weights, as_ids(ids), 1.5f, slow, error));
    REQUIRE(kokopop::sano_run_duration(*loaded.arch, *weights, as_ids(ids), 1.0f, normal, error));
    REQUIRE(kokopop::sano_run_duration(*loaded.arch, *weights, as_ids(ids), 0.5f, fast, error));

    for (size_t i = 0; i < normal.size(); ++i) {
        CHECK(slow[i] >= normal[i]);
        CHECK(fast[i] <= normal[i]);
    }
    const auto sum = [](const std::vector<int32_t> & v) {
        return std::accumulate(v.begin(), v.end(), 0);
    };
    CHECK(sum(slow) > sum(normal));
    CHECK(sum(fast) < sum(normal));
}

TEST_CASE("sanotts_duration_rejects_a_sequence_past_the_voice_ceiling") {
    kokopop::test::SanoModel loaded;
    std::string why;
    if (!loaded.load("heart", why)) {
        MESSAGE("skipped: " << why);
        return;
    }
    const kokopop::VoiceDesc * voice = loaded.arch->find_voice("heart");
    REQUIRE(voice != nullptr);
    const kokopop::SanoVoice * weights = loaded.arch->voice_for(*voice);
    REQUIRE(weights != nullptr);

    const std::vector<uint32_t> too_many(weights->dur.max_tokens + 1, 5u);
    std::vector<int32_t> durations;
    std::string error;
    CHECK_FALSE(kokopop::sano_run_duration(*loaded.arch, *weights, too_many, 1.0f,
                                           durations, error));
    CHECK(error.find("exceeds") != std::string::npos);

    error.clear();
    CHECK_FALSE(kokopop::sano_run_duration(*loaded.arch, *weights, {}, 1.0f,
                                           durations, error));
    CHECK(error.find("empty") != std::string::npos);
}

TEST_CASE("sanotts_duration_matches_the_piperlite_reference") {
    kokopop::test::SanoModel loaded;
    std::string why;
    if (!loaded.load("mixed", why)) {
        MESSAGE("skipped: " << why);
        return;
    }
    if (kokopop::test::sanotts_piperlite_fixture_path("amy", "r00_ids.bin").empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_piperlite_fixture_hint());
        return;
    }

    // Two ties out of ~1,200 tokens differ from the reference: the GGUF stores
    // these weights in F16 while the reference forward pass is F32, and
    // `round(exp(x))` turns a 1e-4 difference at a k+0.5 boundary into a whole
    // frame. The bound is on the count, not a tolerance on the values.
    int mismatches = 0;
    int tokens = 0;
    for (const char * name : {"amy", "kristin"}) {
        const kokopop::VoiceDesc * voice = loaded.arch->find_voice(name);
        REQUIRE(voice != nullptr);
        const kokopop::SanoVoice * weights = loaded.arch->voice_for(*voice);
        REQUIRE(weights != nullptr);

        for (int row = 0; row < 8; ++row) {
            char stem[16];
            std::snprintf(stem, sizeof stem, "r%02d", row);

            const std::string ids_path = kokopop::test::sanotts_piperlite_fixture_path(
                name, std::string(stem) + "_ids.bin");
            const std::string durs_path = kokopop::test::sanotts_piperlite_fixture_path(
                name, std::string(stem) + "_durs.bin");
            if (ids_path.empty() || durs_path.empty()) {
                continue;
            }

            std::vector<int32_t> ids;
            std::vector<int32_t> expected;
            REQUIRE(kokopop::test::read_i32_file(ids_path, ids));
            REQUIRE(kokopop::test::read_i32_file(durs_path, expected));

            std::vector<int32_t> got;
            std::string error;
            REQUIRE_MESSAGE(kokopop::sano_run_duration(*loaded.arch, *weights, as_ids(ids),
                                                       voice->length_scale, got, error),
                            error);
            REQUIRE(got.size() == expected.size());
            tokens += static_cast<int>(got.size());
            for (size_t i = 0; i < got.size(); ++i) {
                if (got[i] != expected[i]) {
                    ++mismatches;
                    CHECK(std::abs(got[i] - expected[i]) == 1);
                }
            }
        }
    }
    REQUIRE(tokens > 0);
    CHECK(mismatches * 200 <= tokens);
}
