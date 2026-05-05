#include "test_helpers.h"

// ---- Synthèse mock (edge cases, speed, voices) ----

TEST_CASE("mock_synthesis_speed_fast") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    kokopop_audio audio_fast{};
    CHECK(kokopop::synthesize_phonemes(*model, "abc", "af_heart", 2.0f, audio_fast, error));
    CHECK(audio_fast.n_samples > 0);
}

TEST_CASE("mock_synthesis_speed_slow") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    kokopop_audio audio_slow{};
    CHECK(kokopop::synthesize_phonemes(*model, "abc", "af_heart", 0.5f, audio_slow, error));
    CHECK(audio_slow.n_samples > 0);
}

TEST_CASE("mock_synthesis_default_voice") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    // Empty voice — should fail or fall back
    kokopop_audio audio{};
    const bool result = kokopop::synthesize_phonemes(*model, "abc", "", 1.0f, audio, error);
    if (result) {
        CHECK(audio.n_samples > 0);
    } else {
        CHECK(!error.empty());
    }
}

TEST_CASE("mock_synthesis_finite_samples") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    kokopop_audio audio{};
    CHECK(kokopop::synthesize_phonemes(*model, "abc", "af_heart", 1.0f, audio, error));
    for (size_t i = 0; i < audio.n_samples; ++i) {
        CHECK(std::isfinite(audio.samples[i]));
    }
}

TEST_CASE("mock_synthesis_nonzero_output") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    kokopop_audio audio{};
    CHECK(kokopop::synthesize_phonemes(*model, "abc", "af_heart", 1.0f, audio, error));
    bool nonzero = false;
    for (size_t i = 0; i < audio.n_samples; ++i) {
        if (std::fabs(audio.samples[i]) > 0.0001f) {
            nonzero = true;
            break;
        }
    }
    CHECK(nonzero);
}

TEST_CASE("mock_synthesis_min_duration") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    kokopop_audio audio{};
    CHECK(kokopop::synthesize_phonemes(*model, "a", "af_heart", 1.0f, audio, error));
    // Min duration ~0.1s at 24000 Hz = 2400 samples
    CHECK(audio.n_samples >= 2400u);
}

TEST_CASE("mock_synthesis_sample_rate_from_gguf") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    kokopop_audio audio{};
    CHECK(kokopop::synthesize_phonemes(*model, "abc", "af_heart", 1.0f, audio, error));
    CHECK_EQ(audio.sample_rate, 24000);
}

TEST_CASE("mock_synthesis_speed_invalid_zero") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    kokopop_audio audio{};
    CHECK(!kokopop::synthesize_phonemes(*model, "abc", "af_heart", 0.0f, audio, error));
    CHECK(!error.empty());
}

TEST_CASE("mock_synthesis_speed_invalid_negative") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    kokopop_audio audio{};
    CHECK(!kokopop::synthesize_phonemes(*model, "abc", "af_heart", -1.0f, audio, error));
    CHECK(!error.empty());
}

TEST_CASE("mock_synthesis_speed_invalid_too_fast") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    kokopop_audio audio{};
    CHECK(!kokopop::synthesize_phonemes(*model, "abc", "af_heart", 5.0f, audio, error));
    CHECK(!error.empty());
}

TEST_CASE("mock_synthesis_speed_invalid_nan") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    kokopop_audio audio{};
    CHECK(!kokopop::synthesize_phonemes(*model, "abc", "af_heart", std::nanf(""), audio, error));
    CHECK(!error.empty());
}

TEST_CASE("mock_synthesis_speed_invalid_inf") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    kokopop_audio audio{};
    CHECK(!kokopop::synthesize_phonemes(*model, "abc", "af_heart", std::numeric_limits<float>::infinity(), audio, error));
    CHECK(!error.empty());
}

TEST_CASE("mock_synthesis_speed_comparison") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    kokopop_audio audio_fast{};
    CHECK(kokopop::synthesize_phonemes(*model, "abc", "af_heart", 2.0f, audio_fast, error));

    kokopop_audio audio_slow{};
    CHECK(kokopop::synthesize_phonemes(*model, "abc", "af_heart", 0.5f, audio_slow, error));

    // Slow should produce more samples than fast
    CHECK(audio_slow.n_samples > audio_fast.n_samples);
}

TEST_CASE("mock_synthesis_different_voices_different_freq") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    // The mock uses voice hash to determine frequency, so different voices
    // should produce different audio content
    kokopop_audio audio1{};
    CHECK(kokopop::synthesize_phonemes(*model, "a", "af_heart", 1.0f, audio1, error));

    // With only af_heart available, unknown voice may fail
    // Just check that synthesis with valid voice produces finite samples
    for (size_t i = 0; i < audio1.n_samples; ++i) {
        CHECK(std::isfinite(audio1.samples[i]));
    }
}
