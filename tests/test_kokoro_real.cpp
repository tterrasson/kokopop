#include "test_helpers.h"

// ---- Tests avec modèle réel kokoro.gguf ----
// Le modèle est chargé une seule fois via shared_real_model() et réutilisé
// par toutes les fonctions de test de ce fichier.

TEST_CASE("real_kokoro_frontend_probe") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::vector<uint32_t> ids;
    std::string error;
    CHECK(model->tokenize_phonemes("bɔ̃ʒˈuʁ", ids, error));
    CHECK_EQ(ids.size(), static_cast<size_t>(9));

    kokopop::KokoroFrontendProbe probe;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "ff_siwis", probe, error));
    CHECK_EQ(probe.n_tokens, 9);
    CHECK_EQ(probe.hidden_dim, 640);
    CHECK_EQ(probe.hidden.size(), static_cast<size_t>(5760));
    CHECK_EQ(probe.durations.size(), static_cast<size_t>(9));

    const float expected[] = {19.342f, 2.9142f, 3.1861f, 3.7248f, 2.8947f, 3.8151f, 7.1699f, 7.412f, 3.7483f};
    int rounded_sum = 0;
    for (size_t i = 0; i < probe.durations.size(); ++i) {
        CHECK_NEAR(probe.durations[i], expected[i], 0.5f);
        rounded_sum += std::max(1, static_cast<int>(std::lrint(probe.durations[i])));
    }
    CHECK_EQ(rounded_sum, 54);
}

TEST_CASE("real_kokoro_frontend_probe_handles_punctuation") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::string phonemes;
    std::string error;
    CHECK(kokopop::phonemize_text("Tu aimes les films ? Oui ? Non ? Bonjour, comment tu vas ?", "ff_siwis", phonemes, error));

    std::vector<uint32_t> ids;
    CHECK(model->tokenize_phonemes(phonemes, ids, error));
    CHECK(ids.size() > 20);

    kokopop::KokoroFrontendProbe probe;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "ff_siwis", probe, error));
    CHECK_EQ(probe.n_tokens, static_cast<int64_t>(ids.size()));
    CHECK_EQ(probe.durations.size(), ids.size());
}

TEST_CASE("real_kokoro_generation_probe") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::vector<uint32_t> ids;
    std::string error;
    CHECK(model->tokenize_phonemes("bɔ̃ʒˈuʁ", ids, error));

    kokopop::KokoroFrontendProbe frontend;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "ff_siwis", frontend, error));
    kokopop::KokoroGenerationProbe gen;
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "ff_siwis", 1.0f, frontend, gen, error));

    // Baseline v3 (mixed quantization Q5_K/Q6_K/Q8_0/F16)
    CHECK_EQ(gen.n_frames, 54);
    CHECK_EQ(gen.f0.size(), static_cast<size_t>(108));
    CHECK_EQ(gen.noise.size(), static_cast<size_t>(108));
    CHECK_EQ(gen.asr.size(), static_cast<size_t>(27648));
    CHECK_EQ(gen.decoder.size(), static_cast<size_t>(55296));
    CHECK_EQ(gen.audio.size(), static_cast<size_t>(32400));

    const Stats f0 = stats(gen.f0);
    const Stats noise = stats(gen.noise);
    const Stats decoder = stats(gen.decoder);
    const Stats audio = stats(gen.audio);
    CHECK_NEAR(f0.mean, 114.21, 0.5);
    CHECK_NEAR(f0.rms, 158.5, 0.5);
    CHECK_NEAR(noise.mean, -0.522, 0.05);
    CHECK_NEAR(noise.rms, 8.692, 0.05);
    CHECK_NEAR(decoder.mean, -1.766, 0.05);
    CHECK_NEAR(decoder.rms, 3.378, 0.05);
    CHECK(std::fabs(audio.mean) < 0.05);
    CHECK(audio.rms > 0.02);
    CHECK(audio.rms < 1.2);
    CHECK(audio.peak > 0.35f);
}

TEST_CASE("real_kokoro_synthesize_stabilizes_af_heart_short_unpunctuated_text") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::string phonemes;
    std::string error;
    CHECK(kokopop::phonemize_text("Hello world", "af_heart", phonemes, error));

    kokopop_audio audio{};
    CHECK(kokopop::synthesize_phonemes(*model, phonemes, "af_heart", 1.0f, audio, error));

    std::vector<float> samples(audio.samples, audio.samples + audio.n_samples);
    kokopop_audio_free(&audio);

    const Stats audio_stats = stats(samples);
    CHECK(audio_stats.rms > 0.02);
    CHECK(audio_stats.rms < 0.2);
    CHECK(audio_stats.peak < 0.6f);
}

TEST_CASE("real_kokoro_scratch_reuse") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::vector<uint32_t> ids;
    std::string error;
    CHECK(model->tokenize_phonemes("bɔ̃ʒˈuʁ", ids, error));

    kokopop::KokoroFrontendProbe front1;
    kokopop::KokoroGenerationProbe gen1;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "ff_siwis", front1, error));
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "ff_siwis", 1.0f, front1, gen1, error));

    const size_t frontend_cap   = model->frontend_scratch.capacity();
    const size_t generation_cap = model->generation_scratch.capacity();
    const size_t generator_cap  = model->generator_scratch.capacity();
    CHECK(frontend_cap > 0);
    CHECK(generation_cap > 0);
    CHECK(generator_cap > 0);

    kokopop::KokoroFrontendProbe front2;
    kokopop::KokoroGenerationProbe gen2;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "ff_siwis", front2, error));
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "ff_siwis", 1.0f, front2, gen2, error));

    // Same-sized input: scratch buffers must not reallocate.
    CHECK_EQ(model->frontend_scratch.capacity(),   frontend_cap);
    CHECK_EQ(model->generation_scratch.capacity(), generation_cap);
    CHECK_EQ(model->generator_scratch.capacity(),  generator_cap);
    CHECK_EQ(gen2.audio.size(), gen1.audio.size());
    CHECK_NEAR(stats(gen2.audio).rms, stats(gen1.audio).rms, 0.002);
}

TEST_CASE("real_kokoro_cpu_metal_parity_when_available") {
    auto * cpu_model = shared_real_model();
    if (!cpu_model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::string error;
    kokopop_model_options metal_options{};
    metal_options.n_threads = 1;
    metal_options.backend = KOKOPOP_BACKEND_METAL;
    std::unique_ptr<kokopop::Model> metal_model;
    if (!kokopop::load_model_from_gguf(real_model_path(), &metal_options, metal_model, error)) {
        MESSAGE("skipping: Metal backend unavailable: " << error);
        return;
    }

    std::vector<uint32_t> ids;
    CHECK(cpu_model->tokenize_phonemes("bɔ̃ʒˈuʁ", ids, error));

    kokopop::KokoroFrontendProbe cpu_front;
    kokopop::KokoroFrontendProbe metal_front;
    CHECK(kokopop::run_kokoro_frontend_probe(*cpu_model, ids, "ff_siwis", cpu_front, error));
    CHECK(kokopop::run_kokoro_frontend_probe(*metal_model, ids, "ff_siwis", metal_front, error));
    CHECK_EQ(metal_front.durations.size(), cpu_front.durations.size());
    for (size_t i = 0; i < cpu_front.durations.size(); ++i) {
        CHECK_NEAR(metal_front.durations[i], cpu_front.durations[i], 1.5f);
    }

    kokopop::KokoroGenerationProbe cpu_gen;
    kokopop::KokoroGenerationProbe metal_gen;
    CHECK(kokopop::run_kokoro_generation_probe(*cpu_model, ids, "ff_siwis", 1.0f, cpu_front, cpu_gen, error));
    CHECK(kokopop::run_kokoro_generation_probe(*metal_model, ids, "ff_siwis", 1.0f, metal_front, metal_gen, error));
    CHECK(metal_gen.n_frames > 0);
    CHECK(metal_gen.audio.size() > 0);
    CHECK_NEAR(stats(metal_gen.audio).rms, stats(cpu_gen.audio).rms, 0.05);
}

TEST_CASE("real_model_token_counts_consistency") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::vector<uint32_t> ids;
    std::string error;
    CHECK(model->tokenize_phonemes("bɔ̃ʒˈuʁ", ids, error));

    kokopop::KokoroFrontendProbe probe;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "ff_siwis", probe, error));
    CHECK_EQ(probe.n_tokens, static_cast<int64_t>(ids.size()));
}

TEST_CASE("real_model_durations_all_positive") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::vector<uint32_t> ids;
    std::string error;
    CHECK(model->tokenize_phonemes("bɔ̃ʒˈuʁ", ids, error));

    kokopop::KokoroFrontendProbe probe;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "ff_siwis", probe, error));
    for (float d : probe.durations) {
        CHECK(d > 0.0f);
    }
}

TEST_CASE("real_model_f0_reasonable_range") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::vector<uint32_t> ids;
    std::string error;
    CHECK(model->tokenize_phonemes("bɔ̃ʒˈuʁ", ids, error));

    kokopop::KokoroFrontendProbe frontend;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "ff_siwis", frontend, error));
    kokopop::KokoroGenerationProbe gen;
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "ff_siwis", 1.0f, frontend, gen, error));

    bool has_nonzero = false;
    for (float f : gen.f0) {
        CHECK(std::isfinite(f));
        if (std::fabs(f) > 0.001f) has_nonzero = true;
    }
    CHECK(has_nonzero);
}

TEST_CASE("real_model_speed_affects_duration") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::vector<uint32_t> ids;
    std::string error;
    CHECK(model->tokenize_phonemes("bɔ̃ʒˈuʁ", ids, error));

    kokopop::KokoroFrontendProbe frontend;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "ff_siwis", frontend, error));

    kokopop::KokoroGenerationProbe gen_slow;
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "ff_siwis", 0.5f, frontend, gen_slow, error));

    kokopop::KokoroGenerationProbe gen_fast;
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "ff_siwis", 2.0f, frontend, gen_fast, error));

    CHECK(gen_slow.n_frames > gen_fast.n_frames);
    const double ratio = static_cast<double>(gen_slow.n_frames) / static_cast<double>(gen_fast.n_frames);
    CHECK(ratio > 2.0);
    CHECK(ratio < 8.0);
}

TEST_CASE("real_model_multiple_voices") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }
    CHECK(model->voices.size() > 1);
}

TEST_CASE("real_model_audio_finite") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::vector<uint32_t> ids;
    std::string error;
    CHECK(model->tokenize_phonemes("bɔ̃ʒˈuʁ", ids, error));

    kokopop::KokoroFrontendProbe frontend;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "ff_siwis", frontend, error));
    kokopop::KokoroGenerationProbe gen;
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "ff_siwis", 1.0f, frontend, gen, error));

    for (float sample : gen.audio) {
        CHECK(std::isfinite(sample));
    }
}

TEST_CASE("real_model_mandarin_multichunk_text_synthesis_stable") {
    kokopop_model_options options{};
    options.n_threads = 4;
    options.backend = KOKOPOP_BACKEND_CPU;

    kokopop_model * model = nullptr;
    CHECK_EQ(kokopop_model_load(real_model_path().c_str(), &options, &model), KOKOPOP_OK);
    REQUIRE(model != nullptr);

    kokopop_audio audio{};
    CHECK_EQ(
        kokopop_synthesize_text(
            model,
            "真正的友谊不仅在于分享快乐，更在于能在彼此遇到困难时互相支持。",
            "zf_xiaoni",
            1.0f,
            &audio),
        KOKOPOP_OK);
    CHECK(audio.samples != nullptr);
    CHECK(audio.n_samples > 0);
    bool all_finite = true;
    for (size_t i = 0; i < audio.n_samples; ++i) {
        if (!std::isfinite(audio.samples[i])) {
            all_finite = false;
            break;
        }
    }
    CHECK(all_finite);

    kokopop_audio_free(&audio);
    kokopop_model_free(model);
}
