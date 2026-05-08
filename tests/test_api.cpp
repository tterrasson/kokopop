#include "test_helpers.h"

// ---- API C publique ----

TEST_CASE("api_model_load_and_mock_synthesis") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    options.n_threads = 2;
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), &options, &model), KOKOPOP_OK);
    CHECK(model != nullptr);

    kokopop_audio audio{};
    CHECK_EQ(kokopop_synthesize_phonemes(model, "abc", "af_heart", 1.0f, &audio), KOKOPOP_OK);
    CHECK(audio.samples != nullptr);
    CHECK(audio.n_samples > 1000);
    CHECK_EQ(audio.sample_rate, 24000);
    float peak = 0.0f;
    for (size_t i = 0; i < audio.n_samples; ++i) {
        CHECK(std::isfinite(audio.samples[i]));
        peak = std::max(peak, std::fabs(audio.samples[i]));
    }
    CHECK(peak > 0.001f);

    CHECK_EQ(kokopop_write_wav("tests/kokopop_mock_test.wav", &audio), KOKOPOP_OK);
    std::ifstream wav("tests/kokopop_mock_test.wav", std::ios::binary);
    char riff[4] = {};
    wav.read(riff, 4);
    CHECK(std::memcmp(riff, "RIFF", 4) == 0);

    kokopop_audio_free(&audio);
    kokopop_model_free(model);
}

TEST_CASE("api_error_null_model") {
    kokopop_audio audio{};
    CHECK_EQ(kokopop_synthesize_phonemes(nullptr, "abc", "af_heart", 1.0f, &audio), KOKOPOP_ERROR_INVALID_ARGUMENT);
}

TEST_CASE("api_error_empty_phonemes") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    kokopop_audio audio{};
    CHECK_EQ(kokopop_synthesize_phonemes(model, "", "af_heart", 1.0f, &audio), KOKOPOP_ERROR_INVALID_ARGUMENT);
    kokopop_model_free(model);
}

TEST_CASE("api_error_missing_voice") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    kokopop_audio audio{};
    CHECK_EQ(kokopop_synthesize_phonemes(model, "abc", "missing", 1.0f, &audio), KOKOPOP_ERROR_INFERENCE);
    CHECK(std::strlen(kokopop_last_error()) > 0);
    kokopop_model_free(model);
}

TEST_CASE("api_synthesize_text") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    kokopop_audio audio{};
    const int rc = kokopop_synthesize_text(model, "hello", "af_heart", 1.0f, &audio);
    if (rc == KOKOPOP_OK) {
        CHECK(audio.n_samples > 1000);
        kokopop_audio_free(&audio);
    } else {
        const bool rc_valid = (rc == KOKOPOP_ERROR_PHONEMIZER || rc == KOKOPOP_ERROR_INFERENCE);
        CHECK(rc_valid);
    }
    kokopop_model_free(model);
}

TEST_CASE("api_error_null_path") {
    kokopop_model * model = nullptr;
    // nullptr and "" path both return INVALID_ARGUMENT
    CHECK_EQ(kokopop_model_load(nullptr, nullptr, &model), KOKOPOP_ERROR_INVALID_ARGUMENT);
    CHECK(model == nullptr);

    CHECK_EQ(kokopop_model_load("", nullptr, &model), KOKOPOP_ERROR_INVALID_ARGUMENT);
    CHECK(model == nullptr);
}

TEST_CASE("api_error_null_audio_output") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    CHECK_EQ(kokopop_synthesize_phonemes(model, "a", "af_heart", 1.0f, nullptr), KOKOPOP_ERROR_INVALID_ARGUMENT);
    kokopop_model_free(model);
}

TEST_CASE("api_error_null_text") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    CHECK_EQ(kokopop_synthesize_text(model, nullptr, "af_heart", 1.0f, nullptr), KOKOPOP_ERROR_INVALID_ARGUMENT);
    kokopop_model_free(model);
}

TEST_CASE("api_error_whitespace_only_text") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    kokopop_audio audio{};
    const int rc = kokopop_synthesize_text(model, "   ", "af_heart", 1.0f, &audio);
    // Should fail: empty text after trim → INVALID_ARGUMENT
    CHECK(rc != KOKOPOP_OK);
    kokopop_model_free(model);
}

TEST_CASE("api_write_wav_null_args") {
    CHECK_EQ(kokopop_write_wav(nullptr, nullptr), KOKOPOP_ERROR_INVALID_ARGUMENT);

    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    kokopop_audio audio{};
    CHECK_EQ(kokopop_synthesize_phonemes(model, "abc", "af_heart", 1.0f, &audio), KOKOPOP_OK);
    CHECK_EQ(kokopop_write_wav(nullptr, &audio), KOKOPOP_ERROR_INVALID_ARGUMENT);
    kokopop_audio_free(&audio);
    kokopop_model_free(model);
}

TEST_CASE("api_write_wav_empty_audio") {
    kokopop_audio audio{};
    audio.samples = nullptr;
    audio.n_samples = 0;
    audio.sample_rate = 24000;
    // wav_bytes produces a valid header with empty data chunk
    CHECK_EQ(kokopop_write_wav("tests/kokopop_empty_audio.wav", &audio), KOKOPOP_OK);
    std::remove("tests/kokopop_empty_audio.wav");
}

TEST_CASE("api_last_error_after_failure") {
    kokopop_audio audio{};
    CHECK_EQ(kokopop_synthesize_phonemes(nullptr, "abc", "af_heart", 1.0f, &audio), KOKOPOP_ERROR_INVALID_ARGUMENT);
    CHECK(std::strlen(kokopop_last_error()) > 0);
}

TEST_CASE("api_model_free_null") {
    kokopop_model_free(nullptr);
}

TEST_CASE("api_audio_free_null") {
    kokopop_audio_free(nullptr);
}

TEST_CASE("api_speed_boundaries") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);

    kokopop_audio audio1{};
    CHECK_EQ(kokopop_synthesize_phonemes(model, "abc", "af_heart", 0.01f, &audio1), KOKOPOP_OK);
    CHECK(audio1.n_samples > 0);
    kokopop_audio_free(&audio1);

    kokopop_audio audio2{};
    CHECK_EQ(kokopop_synthesize_phonemes(model, "abc", "af_heart", 1.0f, &audio2), KOKOPOP_OK);
    CHECK(audio2.n_samples > 0);
    kokopop_audio_free(&audio2);

    kokopop_audio audio3{};
    CHECK_EQ(kokopop_synthesize_phonemes(model, "abc", "af_heart", 4.0f, &audio3), KOKOPOP_OK);
    CHECK(audio3.n_samples > 0);
    kokopop_audio_free(&audio3);

    kokopop_model_free(model);
}

TEST_CASE("api_speed_invalid") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);

    // Speed validation happens in the inference layer → INFERENCE error
    kokopop_audio audio{};
    CHECK_EQ(kokopop_synthesize_phonemes(model, "abc", "af_heart", 0.0f, &audio), KOKOPOP_ERROR_INFERENCE);
    CHECK_EQ(kokopop_synthesize_phonemes(model, "abc", "af_heart", -1.0f, &audio), KOKOPOP_ERROR_INFERENCE);
    CHECK_EQ(kokopop_synthesize_phonemes(model, "abc", "af_heart", std::nanf(""), &audio), KOKOPOP_ERROR_INFERENCE);
    CHECK_EQ(kokopop_synthesize_phonemes(model, "abc", "af_heart", std::numeric_limits<float>::infinity(), &audio), KOKOPOP_ERROR_INFERENCE);

    kokopop_model_free(model);
}

// ---- Additional C API guards ----

TEST_CASE("api_model_load_nonexistent_path") {
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load("/tmp/kokopop_does_not_exist.gguf", nullptr, &model), KOKOPOP_ERROR_MODEL);
    CHECK(model == nullptr);
    CHECK(std::strlen(kokopop_last_error()) > 0);
}

TEST_CASE("api_model_load_out_model_null") {
    const std::string & gguf = shared_mock_gguf();
    // Passing nullptr for out_model should return INVALID_ARGUMENT
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, nullptr), KOKOPOP_ERROR_INVALID_ARGUMENT);
}

TEST_CASE("api_synthesize_text_null_audio_output") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    // out_audio == nullptr → INVALID_ARGUMENT
    CHECK_EQ(kokopop_synthesize_text(model, "hello", "af_heart", 1.0f, nullptr), KOKOPOP_ERROR_INVALID_ARGUMENT);
    kokopop_model_free(model);
}

TEST_CASE("api_model_sample_rate_mock") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    CHECK_EQ(kokopop_model_sample_rate(model), 24000);
    kokopop_model_free(model);
}

TEST_CASE("api_model_sample_rate_null") {
    CHECK_EQ(kokopop_model_sample_rate(nullptr), 0);
}

TEST_CASE("api_model_get_impl_null") {
    CHECK_EQ(kokopop_model_get_impl(nullptr), nullptr);
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    CHECK(kokopop_model_get_impl(model) != nullptr);
    kokopop_model_free(model);
}

TEST_CASE("api_synthesize_phonemes_null_phonemes") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    kokopop_audio audio{};
    CHECK_EQ(kokopop_synthesize_phonemes(model, nullptr, "af_heart", 1.0f, &audio), KOKOPOP_ERROR_INVALID_ARGUMENT);
    kokopop_model_free(model);
}

TEST_CASE("api_synthesize_text_null_voice") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    kokopop_audio audio{};
    // nullptr voice → should use empty string → falls back to first voice
    const int rc = kokopop_synthesize_text(model, "hello", nullptr, 1.0f, &audio);
    if (rc == KOKOPOP_OK) {
        CHECK(audio.n_samples > 0);
        kokopop_audio_free(&audio);
    } else {
        // rc is one of OK, PHONEMIZER, or INFERENCE
        int valid = (rc == KOKOPOP_OK)
                  + (rc == KOKOPOP_ERROR_PHONEMIZER)
                  + (rc == KOKOPOP_ERROR_INFERENCE);
        CHECK_EQ(valid, 1);
    }
    kokopop_model_free(model);
}
