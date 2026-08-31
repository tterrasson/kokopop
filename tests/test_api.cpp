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

    CHECK_EQ(kokopop_write_wav("kokopop_mock_test.wav", &audio), KOKOPOP_OK);
    std::ifstream wav("kokopop_mock_test.wav", std::ios::binary);
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

TEST_CASE("api_model_backend_reports_resolved_backend") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    kokopop_model_options options{};
    options.backend = KOKOPOP_BACKEND_AUTO;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), &options, &model), KOKOPOP_OK);
    REQUIRE(model != nullptr);

    const int32_t resolved = kokopop_model_backend(model);
    // AUTO is resolved at load time; the accessor never reports it back.
    CHECK_NE(resolved, KOKOPOP_BACKEND_AUTO);
    CHECK_GE(resolved, KOKOPOP_BACKEND_CPU);
    CHECK_LE(resolved, KOKOPOP_BACKEND_OPENCL);
    CHECK_EQ(resolved, kokopop_model_get_impl(model)->backend_type);
    kokopop_model_free(model);

    // Null model is CPU rather than a crash or a sentinel.
    CHECK_EQ(kokopop_model_backend(nullptr), KOKOPOP_BACKEND_CPU);
}

TEST_CASE("api_model_backend_cpu_request") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    kokopop_model_options options{};
    options.backend = KOKOPOP_BACKEND_CPU;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), &options, &model), KOKOPOP_OK);
    REQUIRE(model != nullptr);
    CHECK_EQ(kokopop_model_backend(model), KOKOPOP_BACKEND_CPU);
    kokopop_model_free(model);
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
    CHECK_EQ(kokopop_write_wav("kokopop_empty_audio.wav", &audio), KOKOPOP_OK);
    std::remove("kokopop_empty_audio.wav");
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

TEST_CASE("api_synthesis_session_pull_chunks") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);

    kokopop_synthesis_options opts{};
    opts.voice = "af_heart";
    opts.speed = 1.0f;
    opts.mode = KOKOPOP_SYNTH_ADAPTATIVE;

    kokopop_synthesis * synth = nullptr;
    CHECK_EQ(kokopop_synthesis_create(model, &opts, &synth), KOKOPOP_OK);
    REQUIRE(synth != nullptr);

    CHECK_EQ(kokopop_synthesis_push_text(synth, "Hello, "), KOKOPOP_OK);
    CHECK_EQ(kokopop_synthesis_push_text(synth, "world."), KOKOPOP_OK);
    CHECK_EQ(kokopop_synthesis_finish_input(synth), KOKOPOP_OK);

    kokopop_audio_chunk * chunks = nullptr;
    size_t n_chunks = 0;
    CHECK_EQ(kokopop_synthesis_next(synth, 1, &chunks, &n_chunks), KOKOPOP_OK);
    REQUIRE_EQ(n_chunks, 1u);
    CHECK(chunks[0].samples != nullptr);
    CHECK(chunks[0].n_samples > 0);
    CHECK_EQ(chunks[0].sample_rate, 24000);
    kokopop_audio_chunks_free(chunks, n_chunks);

    CHECK_EQ(kokopop_synthesis_push_text(synth, "too late"), KOKOPOP_ERROR_INVALID_ARGUMENT);
    kokopop_synthesis_free(synth);
    kokopop_model_free(model);
}

TEST_CASE("api_synthesis_session_diffusion_options_disabled_are_noop") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);

    kokopop_synthesis_options opts{};
    opts.voice = "af_heart";
    opts.speed = 1.0f;
    opts.mode = KOKOPOP_SYNTH_LONG_FORM;
    opts.enable_diffusion = 0;
    opts.diffusion_seed = 1234;
    opts.diffusion_steps = 7;
    opts.diffusion_alpha = 0.2f;
    opts.diffusion_beta = 0.6f;
    opts.diffusion_embedding_scale = 1.5f;

    kokopop_synthesis * synth = nullptr;
    CHECK_EQ(kokopop_synthesis_create(model, &opts, &synth), KOKOPOP_OK);
    REQUIRE(synth != nullptr);
    CHECK_EQ(kokopop_synthesis_push_text(synth, "Hello world."), KOKOPOP_OK);
    CHECK_EQ(kokopop_synthesis_finish_input(synth), KOKOPOP_OK);

    kokopop_audio_chunk * chunks = nullptr;
    size_t n_chunks = 0;
    CHECK_EQ(kokopop_synthesis_next(synth, 1, &chunks, &n_chunks), KOKOPOP_OK);
    REQUIRE_EQ(n_chunks, 1u);
    CHECK(chunks[0].n_samples > 0);
    kokopop_audio_chunks_free(chunks, n_chunks);

    kokopop_synthesis_free(synth);
    kokopop_model_free(model);
}

TEST_CASE("api_synthesis_session_next_multiple_chunks") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);

    kokopop_synthesis_options opts{};
    opts.voice = "af_heart";
    opts.speed = 1.0f;
    opts.mode = KOKOPOP_SYNTH_LONG_FORM;
    opts.target_min_tokens = 1;
    opts.target_max_tokens = 12;
    opts.soft_max_tokens = 16;
    opts.hard_max_tokens = 32;

    kokopop_synthesis * synth = nullptr;
    CHECK_EQ(kokopop_synthesis_create(model, &opts, &synth), KOKOPOP_OK);
    REQUIRE(synth != nullptr);
    CHECK_EQ(kokopop_synthesis_push_text(
        synth, "Alpha sentence. Beta sentence. Gamma sentence. Delta sentence."), KOKOPOP_OK);
    CHECK_EQ(kokopop_synthesis_finish_input(synth), KOKOPOP_OK);

    kokopop_audio_chunk * chunks = nullptr;
    size_t n_chunks = 0;
    CHECK_EQ(kokopop_synthesis_next(synth, 4, &chunks, &n_chunks), KOKOPOP_OK);
    CHECK(n_chunks >= 1u);
    for (size_t i = 0; i < n_chunks; ++i) {
        CHECK(chunks[i].n_samples > 0);
        CHECK_EQ(chunks[i].chunk_index, static_cast<int32_t>(i));
    }
    kokopop_audio_chunks_free(chunks, n_chunks);

    kokopop_synthesis_free(synth);
    kokopop_model_free(model);
}

TEST_CASE("api_synthesis_session_invalid_args") {
    kokopop_synthesis * synth = nullptr;
    kokopop_synthesis_options opts{};
    opts.mode = 99;
    CHECK_EQ(kokopop_synthesis_create(nullptr, &opts, &synth), KOKOPOP_ERROR_INVALID_ARGUMENT);

    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    CHECK_EQ(kokopop_synthesis_create(model, &opts, &synth), KOKOPOP_ERROR_INVALID_ARGUMENT);

    opts.mode = KOKOPOP_SYNTH_ADAPTATIVE;
    CHECK_EQ(kokopop_synthesis_create(model, &opts, &synth), KOKOPOP_OK);
    CHECK_EQ(kokopop_synthesis_push_text(synth, ""), KOKOPOP_ERROR_INVALID_ARGUMENT);
    CHECK_EQ(kokopop_synthesis_finish_input(synth), KOKOPOP_ERROR_INVALID_ARGUMENT);
    kokopop_synthesis_free(synth);
    kokopop_model_free(model);
}

TEST_CASE("api_audio_encoder_pcm_and_wav") {
    const float samples[] = {0.0f, 0.25f, -0.25f, 1.0f};

    kokopop_encoder_options pcm_opts{};
    pcm_opts.format = KOKOPOP_AUDIO_PCM_F32LE;
    pcm_opts.sample_rate = 24000;
    kokopop_audio_encoder * enc = nullptr;
    CHECK_EQ(kokopop_audio_encoder_create(&pcm_opts, &enc), KOKOPOP_OK);
    REQUIRE(enc != nullptr);
    kokopop_bytes bytes{};
    CHECK_EQ(kokopop_audio_encoder_push(enc, samples, 4, 1, &bytes), KOKOPOP_OK);
    CHECK_EQ(bytes.size, sizeof(samples));
    kokopop_bytes_free(&bytes);
    kokopop_audio_encoder_free(enc);

    kokopop_encoder_options wav_opts{};
    wav_opts.format = KOKOPOP_AUDIO_WAV_PCM16;
    wav_opts.sample_rate = 24000;
    enc = nullptr;
    CHECK_EQ(kokopop_audio_encoder_create(&wav_opts, &enc), KOKOPOP_OK);
    REQUIRE(enc != nullptr);
    CHECK_EQ(kokopop_audio_encoder_push(enc, samples, 4, 1, &bytes), KOKOPOP_OK);
    CHECK_EQ(bytes.size, 0u);
    kokopop_bytes_free(&bytes);
    CHECK_EQ(kokopop_audio_encoder_finish(enc, 1, &bytes), KOKOPOP_OK);
    REQUIRE(bytes.size >= 12u);
    CHECK(std::memcmp(bytes.data, "RIFF", 4) == 0);
    CHECK(std::memcmp(bytes.data + 8, "WAVE", 4) == 0);
    kokopop_bytes_free(&bytes);
    kokopop_audio_encoder_free(enc);
}

TEST_CASE("api_audio_encoder_ogg_availability") {
    kokopop_encoder_options opts{};
    opts.format = KOKOPOP_AUDIO_OGG_OPUS;
    opts.sample_rate = 24000;
    kokopop_audio_encoder * enc = nullptr;
    const int rc = kokopop_audio_encoder_create(&opts, &enc);
#ifdef KOKOPOP_HAS_OPUS
    CHECK_EQ(rc, KOKOPOP_OK);
    REQUIRE(enc != nullptr);
    kokopop_bytes bytes{};
    CHECK_EQ(kokopop_audio_encoder_start(enc, &bytes), KOKOPOP_OK);
    CHECK(bytes.size > 0);
    kokopop_bytes_free(&bytes);
    kokopop_audio_encoder_free(enc);
#else
    CHECK_EQ(rc, KOKOPOP_ERROR_IO);
    CHECK(enc == nullptr);
    CHECK(std::strlen(kokopop_last_error()) > 0);
#endif
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
