#include "test_helpers.h"
#include "sanotts_fixtures.h"

// ---- Model introspection (arch, voice table, per-voice rate) ----

TEST_CASE("api_model_arch_reports_kokoro") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    REQUIRE_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    REQUIRE(model != nullptr);

    CHECK_EQ(kokopop_model_arch(model), KOKOPOP_ARCH_KOKORO);
    CHECK_EQ(std::string(kokopop_model_arch_name(model)), "kokoro-82m");

    const size_t n = kokopop_model_voice_count(model);
    CHECK(n > 0);
    for (size_t i = 0; i < n; ++i) {
        const char * name = kokopop_model_voice_name(model, i);
        REQUIRE(name != nullptr);
        CHECK(kokopop_model_voice_sample_rate(model, name) > 0);
    }
    CHECK(kokopop_model_voice_name(model, n) == nullptr);
    CHECK_EQ(kokopop_model_voice_sample_rate(model, "no_such_voice"), 0);
    CHECK_EQ(kokopop_model_voice_sample_rate(model, nullptr), 0);

    kokopop_model_free(model);
}

TEST_CASE("api_model_arch_null_model_is_unknown") {
    CHECK_EQ(kokopop_model_arch(nullptr), KOKOPOP_ARCH_UNKNOWN);
    CHECK_EQ(std::string(kokopop_model_arch_name(nullptr)), "unknown");
    CHECK_EQ(kokopop_model_voice_count(nullptr), 0u);
    CHECK(kokopop_model_voice_name(nullptr, 0) == nullptr);
    CHECK_EQ(kokopop_model_voice_sample_rate(nullptr, "heart"), 0);
}

TEST_CASE("api_model_voice_sample_rate_sanotts_mixed_pack") {
    const std::string path = kokopop::test::sanotts_model_path("mixed");
    if (path.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_model_hint("mixed"));
        return;
    }
    kokopop_model_options options{};
    options.backend = KOKOPOP_BACKEND_CPU;
    kokopop_model * model = nullptr;
    REQUIRE_EQ(kokopop_model_load(path.c_str(), &options, &model), KOKOPOP_OK);

    CHECK_EQ(kokopop_model_arch(model), KOKOPOP_ARCH_SANOTTS);
    CHECK_EQ(std::string(kokopop_model_arch_name(model)), "sanotts");

    // The point of the accessor: one file, two rates. The model-level one
    // only ever answers for the default voice.
    CHECK_EQ(kokopop_model_voice_sample_rate(model, "heart"), 24000);
    CHECK_EQ(kokopop_model_voice_sample_rate(model, "amy"), 22050);
    const size_t n = kokopop_model_voice_count(model);
    CHECK_EQ(n, 4u);
    std::vector<std::string> names;
    for (size_t i = 0; i < n; ++i) {
        names.emplace_back(kokopop_model_voice_name(model, i));
    }
    CHECK(std::find(names.begin(), names.end(), "heart") != names.end());
    CHECK(std::find(names.begin(), names.end(), "amy") != names.end());

    // The model-level rate is the default voice's, which is not necessarily
    // the first of the file, so it is only ever one of the voices' rates.
    bool matches_a_voice = false;
    for (const std::string & name : names) {
        matches_a_voice |= kokopop_model_voice_sample_rate(model, name.c_str()) ==
                           kokopop_model_sample_rate(model);
    }
    CHECK(matches_a_voice);

    kokopop_model_free(model);
}

// ---- sanoTTS synthesis options ----

TEST_CASE("api_synthesis_sano_noise_seed_drives_the_decoder") {
    const std::string path = kokopop::test::sanotts_model_path("mixed");
    if (path.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_model_hint("mixed"));
        return;
    }
    kokopop_model_options model_options{};
    model_options.n_threads = 1;
    model_options.backend = KOKOPOP_BACKEND_CPU;
    kokopop_model * model = nullptr;
    REQUIRE_EQ(kokopop_model_load(path.c_str(), &model_options, &model), KOKOPOP_OK);

    auto render = [&](bool has_seed, uint64_t seed) {
        kokopop_synthesis_options opts{};
        opts.voice = "heart";
        opts.speed = 1.0f;
        opts.mode = KOKOPOP_SYNTH_LONG_FORM;
        opts.has_sano_noise_seed = has_seed ? 1 : 0;
        opts.sano_noise_seed = seed;

        kokopop_synthesis * synth = nullptr;
        REQUIRE_EQ(kokopop_synthesis_create(model, &opts, &synth), KOKOPOP_OK);
        REQUIRE_EQ(kokopop_synthesis_push_text(synth, "Hello there."), KOKOPOP_OK);
        REQUIRE_EQ(kokopop_synthesis_finish_input(synth), KOKOPOP_OK);
        kokopop_audio_chunk * chunks = nullptr;
        size_t n_chunks = 0;
        REQUIRE_EQ(kokopop_synthesis_next(synth, 4, &chunks, &n_chunks), KOKOPOP_OK);
        std::vector<float> pcm;
        for (size_t i = 0; i < n_chunks; ++i) {
            CHECK_EQ(chunks[i].sample_rate, 24000);
            pcm.insert(pcm.end(), chunks[i].samples, chunks[i].samples + chunks[i].n_samples);
        }
        kokopop_audio_chunks_free(chunks, n_chunks);
        kokopop_synthesis_free(synth);
        return pcm;
    };

    const std::vector<float> seeded_a = render(true, 42);
    const std::vector<float> seeded_b = render(true, 42);
    const std::vector<float> other    = render(true, 43);
    const std::vector<float> unseeded = render(false, 0);
    const std::vector<float> zero     = render(true, 0);

    REQUIRE(!seeded_a.empty());
    CHECK(seeded_a == seeded_b);
    REQUIRE_EQ(seeded_a.size(), other.size());
    CHECK(seeded_a != other);

    // An explicit 0 is a seed, not "unset": it reaches the decoder instead of
    // using the seed derived from the voice.
    REQUIRE_EQ(unseeded.size(), zero.size());
    CHECK(unseeded != zero);

    kokopop_model_free(model);
}
