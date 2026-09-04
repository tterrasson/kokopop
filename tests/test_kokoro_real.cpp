#include "test_helpers.h"
#include "synthesis/chunker/chunker.h"
#include "synthesis/synth.h"

// ---- Tests avec modèle réel kokoro.gguf ----
// Le modèle est chargé une seule fois via shared_real_model() et réutilisé
// par toutes les fonctions de test de ce fichier.

TEST_CASE("real_kokoro_frontend_probe") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::vector<uint32_t> ids;
    std::string error;
    CHECK(model->tokenize_phonemes("bɔ̃ʒˈuʁ", ids, error));
    CHECK(ids.size() > 0);

    kokopop::KokoroFrontendProbe probe;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "af_heart", probe, error));

    // Structural invariants: sizes depend on token count and architecture constants
    CHECK_EQ(probe.n_tokens, static_cast<int64_t>(ids.size()));
    CHECK_EQ(probe.hidden_dim, 640);
    CHECK_EQ(probe.hidden.size(), static_cast<size_t>(probe.n_tokens * probe.hidden_dim));
    CHECK_EQ(probe.durations.size(), ids.size());

    int rounded_sum = 0;
    for (size_t i = 0; i < probe.durations.size(); ++i) {
        CHECK(probe.durations[i] > 0.0f);
        rounded_sum += std::max(1, static_cast<int>(std::lrint(probe.durations[i])));
    }
    CHECK(rounded_sum > 0);
    CHECK(rounded_sum < 200);
}

TEST_CASE("real_kokoro_frontend_probe_handles_punctuation") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::string phonemes;
    std::string error;
    CHECK(kokopop::phonemize_text("Tu aimes les films ? Oui ? Non ? Bonjour, comment tu vas ?", "af_heart", phonemes, error));

    std::vector<uint32_t> ids;
    CHECK(model->tokenize_phonemes(phonemes, ids, error));
    CHECK(ids.size() > 20);

    kokopop::KokoroFrontendProbe probe;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "af_heart", probe, error));
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
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "af_heart", frontend, error));
    kokopop::KokoroGenerationProbe gen;
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "af_heart", 1.0f, frontend, gen, error));

    // Structural invariants: sizes are architecture constants × n_frames
    // (512 ASR channels, 1024 decoder channels, 600 audio samples per frame)
    CHECK(gen.n_frames > 0);
    CHECK_EQ(gen.f0.size(),      static_cast<size_t>(gen.n_frames * 2));
    CHECK_EQ(gen.noise.size(),   static_cast<size_t>(gen.n_frames * 2));
    CHECK_EQ(gen.asr.size(),     static_cast<size_t>(gen.n_frames * 512));
    CHECK_EQ(gen.decoder.size(), static_cast<size_t>(gen.n_frames * 1024));
    CHECK_EQ(gen.audio.size(),   static_cast<size_t>(gen.n_frames * 600));

    // n_frames must match the rounded duration sum from the frontend
    int rounded_sum = 0;
    for (float d : frontend.durations)
        rounded_sum += std::max(1, static_cast<int>(std::lrint(d)));
    CHECK_EQ(gen.n_frames, rounded_sum);

    // Sanity checks on audio output (not quantization-sensitive)
    const Stats audio = stats(gen.audio);
    CHECK(std::fabs(audio.mean) < 0.05);
    CHECK(audio.rms > 0.01);
    CHECK(audio.rms < 1.5);
    CHECK(audio.peak > 0.1f);
}

TEST_CASE("real_kokoro_text_frontend_adds_stable_end_for_short_unpunctuated_text") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::string error;
    kokopop::TokenizeFn tokenize_fn =
        [model](const std::string & phonemes,
                std::vector<uint32_t> & ids,
                std::string & token_error) {
            return model->tokenize_phonemes(phonemes, ids, token_error);
        };
    kokopop::PhonemizeFn phonemize_fn =
        [](const std::string & text,
           std::string & phonemes,
           std::string & ph_error) {
            return kokopop::phonemize_text(text, "gmw/en-US", 'a', phonemes, ph_error);
        };

    auto chunks = kokopop::chunk_text(
        "Hello world",
        kokopop::make_long_form_config(), phonemize_fn, tokenize_fn, error);
    REQUIRE_EQ(chunks.size(), 1);
    CHECK_EQ(chunks.front().boundary_after, kokopop::Boundary::Sentence);
    CHECK(chunks.front().phonemes.find('.') != std::string::npos);

    kokopop_audio audio{};
    CHECK(kokopop::synthesize_phonemes(*model, chunks.front().phonemes, "af_heart", 1.0f, audio, error));

    std::vector<float> samples(audio.samples, audio.samples + audio.n_samples);
    kokopop_audio_free(&audio);

    const Stats audio_stats = stats(samples);
    CHECK(audio_stats.rms > 0.02);
    CHECK(audio_stats.rms < 0.2);
    CHECK(audio_stats.peak < 0.95f);
}

TEST_CASE("real_kokoro_text_short_comma_stays_stable") {
    if (!real_model_available()) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    kokopop_model_options options{};
    options.n_threads = 1;
    options.backend = KOKOPOP_BACKEND_CPU;

    kokopop_model * model = nullptr;
    REQUIRE_EQ(kokopop_model_load(real_model_path().c_str(), &options, &model), KOKOPOP_OK);

    kokopop_audio audio{};
    CHECK_EQ(kokopop_synthesize_text(model, "Hello, world.", "af_heart", 1.0f, &audio), KOKOPOP_OK);

    std::vector<float> samples(audio.samples, audio.samples + audio.n_samples);
    kokopop_audio_free(&audio);
    kokopop_model_free(model);

    const Stats audio_stats = stats(samples);
    CHECK(audio_stats.rms > 0.02);
    CHECK(audio_stats.rms < 0.2);
    CHECK(audio_stats.peak < 0.95f);
}

TEST_CASE("real_kokoro_scratch_reuse") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::vector<uint32_t> ids;
    std::string error;
    CHECK(model->tokenize_phonemes("bɔ̃ʒˈuʁ", ids, error));

    kokopop::KokoroFrontendProbe front1;
    kokopop::KokoroGenerationProbe gen1;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "af_heart", front1, error));
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "af_heart", 1.0f, front1, gen1, error));

    kokopop::KokoroArch * arch = kokopop::kokoro_arch(*model);
    REQUIRE(arch != nullptr);
    const size_t frontend_cap   = arch->frontend_scratch.capacity();
    const size_t generation_cap = arch->generation_scratch.capacity();
    const size_t generator_cap  = arch->generator_scratch.capacity();
    CHECK(frontend_cap > 0);
    CHECK(generation_cap > 0);
    CHECK(generator_cap > 0);

    kokopop::KokoroFrontendProbe front2;
    kokopop::KokoroGenerationProbe gen2;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "af_heart", front2, error));
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "af_heart", 1.0f, front2, gen2, error));

    // Same-sized input: scratch buffers must not reallocate.
    CHECK_EQ(arch->frontend_scratch.capacity(),   frontend_cap);
    CHECK_EQ(arch->generation_scratch.capacity(), generation_cap);
    CHECK_EQ(arch->generator_scratch.capacity(),  generator_cap);
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
    CHECK(kokopop::run_kokoro_frontend_probe(*cpu_model, ids, "af_heart", cpu_front, error));
    CHECK(kokopop::run_kokoro_frontend_probe(*metal_model, ids, "af_heart", metal_front, error));
    CHECK_EQ(metal_front.durations.size(), cpu_front.durations.size());
    for (size_t i = 0; i < cpu_front.durations.size(); ++i) {
        CHECK_NEAR(metal_front.durations[i], cpu_front.durations[i], 1.5f);
    }

    kokopop::KokoroGenerationProbe cpu_gen;
    kokopop::KokoroGenerationProbe metal_gen;
    CHECK(kokopop::run_kokoro_generation_probe(*cpu_model, ids, "af_heart", 1.0f, cpu_front, cpu_gen, error));
    CHECK(kokopop::run_kokoro_generation_probe(*metal_model, ids, "af_heart", 1.0f, metal_front, metal_gen, error));
    CHECK(metal_gen.n_frames > 0);
    CHECK(metal_gen.audio.size() > 0);
    CHECK_NEAR(stats(metal_gen.audio).rms, stats(cpu_gen.audio).rms, 0.05);
}

// Regression: on long inputs, the Vulkan/MoltenVK scheduler
// aliased the buffers of intermediate AdaIN-block outputs in the predictor
// (F0/N) and decoder sections of the generation graph, producing audibly
// deformed audio (metallic / phased). The fix is `ggml_set_output` on those
// intermediate outputs so the allocator cannot reuse their buffers. Without
// the fix the decoder tensor RMS / peak diverges dramatically from CPU; with
// the fix the gap is dominated by harmless upstream fp drift.
//
// Two phrases are checked because they exercise different aliasing paths:
// the first triggered only the decoder-chain bug, the second additionally
// triggered the F0/N predictor-chain bug.
static void check_vulkan_long_input(
    kokopop::Model & cpu_model,
    kokopop::Model & vk_model,
    const std::string & phonemes,
    const std::string & voice) {

    std::string error;
    std::vector<uint32_t> ids;
    CHECK(cpu_model.tokenize_phonemes(phonemes, ids, error));
    CHECK(ids.size() > 50);  // bug 3 only triggers on long inputs

    kokopop::KokoroFrontendProbe cpu_front, vk_front;
    CHECK(kokopop::run_kokoro_frontend_probe(cpu_model, ids, voice, cpu_front, error));
    CHECK(kokopop::run_kokoro_frontend_probe(vk_model,  ids, voice, vk_front,  error));

    kokopop::KokoroGenerationProbe cpu_gen, vk_gen;
    CHECK(kokopop::run_kokoro_generation_probe(cpu_model, ids, voice, 1.0f, cpu_front, cpu_gen, error));
    CHECK(kokopop::run_kokoro_generation_probe(vk_model,  ids, voice, 1.0f, vk_front,  vk_gen,  error));

    // Decoder tensor RMS / peak must match CPU closely. Without the fix
    // Vulkan diverges by tens of percent.
    const Stats cpu_dec = stats(cpu_gen.decoder);
    const Stats vk_dec  = stats(vk_gen.decoder);
    CHECK(cpu_dec.rms > 0.5);
    CHECK_NEAR(vk_dec.rms,  cpu_dec.rms,  0.10);
    CHECK_NEAR(vk_dec.peak, cpu_dec.peak, 5.0);

    // Audio-level sanity: same overall energy.
    CHECK_NEAR(stats(vk_gen.audio).rms, stats(cpu_gen.audio).rms, 0.05);
}

TEST_CASE("real_kokoro_cpu_vulkan_decoder_parity_long_input") {
    auto * cpu_model = shared_real_model();
    if (!cpu_model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::string error;
    kokopop_model_options vk_options{};
    vk_options.n_threads = 1;
    vk_options.backend = KOKOPOP_BACKEND_VULKAN;
    std::unique_ptr<kokopop::Model> vk_model;
    if (!kokopop::load_model_from_gguf(real_model_path(), &vk_options, vk_model, error)) {
        MESSAGE("skipping: Vulkan backend unavailable: " << error);
        return;
    }

    // Phrase 1 — triggered the decoder-chain aliasing (without the F0/N fix
    // it would still pass this case).
    const std::string phonemes_decoder =
        "ˈA dˈɪɹ ɪmˈɜɹʤd fɹʌm ðə ʃˈædOz, pˈɔzɪŋ tə ɡlˈæns. ɪʦ ˈæmbəɹɹ ˈIz "
        "mˈɛt mˈIn fəɹɹə hˈɑɹtbit, ðˈɛn ɪt flˈɛd. fˈɜɹðəɹɹ əhˈɛd, ən "
        "əbˈændənd stˈOn kˈɑTɪʤ stˈʊd. vˈInz klˈAmd ðə ɹˈuf";
    check_vulkan_long_input(*cpu_model, *vk_model, phonemes_decoder, "af_heart");

    // Phrase 2 — additionally triggered the F0/N predictor-chain aliasing.
    const std::string phonemes_f0 =
        "ðə wˈɪnd ɹˈʌsᵊld ðə ˈOld kˈɜɹtənz, ˈɛkOɪŋ sˈɔft lˈʌləbˌIz. "
        "sˈʌnlIt fˈɪltəɹd θɹu kɹˈækt wˈɪndOz, kˈæstɪŋ pˈætəɹnz. "
        "I fˈɛlt ə pɹəfˈWnd pˈis, æz ɪf tˈIm ɪʦˈɛlf sˈɔfənd. "
        "ðə pˈæθ klˈImd stˈipəɹ, ɹᵻvˈilɪŋ ˈA vˈæli bᵻlˈO";
    check_vulkan_long_input(*cpu_model, *vk_model, phonemes_f0, "af_heart");
}

TEST_CASE("real_model_token_counts_consistency") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::vector<uint32_t> ids;
    std::string error;
    CHECK(model->tokenize_phonemes("bɔ̃ʒˈuʁ", ids, error));

    kokopop::KokoroFrontendProbe probe;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "af_heart", probe, error));
    CHECK_EQ(probe.n_tokens, static_cast<int64_t>(ids.size()));
}

TEST_CASE("real_model_durations_all_positive") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::vector<uint32_t> ids;
    std::string error;
    CHECK(model->tokenize_phonemes("bɔ̃ʒˈuʁ", ids, error));

    kokopop::KokoroFrontendProbe probe;
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "af_heart", probe, error));
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
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "af_heart", frontend, error));
    kokopop::KokoroGenerationProbe gen;
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "af_heart", 1.0f, frontend, gen, error));

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
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "af_heart", frontend, error));

    kokopop::KokoroGenerationProbe gen_slow;
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "af_heart", 0.5f, frontend, gen_slow, error));

    kokopop::KokoroGenerationProbe gen_fast;
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "af_heart", 2.0f, frontend, gen_fast, error));

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
    CHECK(kokopop::run_kokoro_frontend_probe(*model, ids, "af_heart", frontend, error));
    kokopop::KokoroGenerationProbe gen;
    CHECK(kokopop::run_kokoro_generation_probe(*model, ids, "af_heart", 1.0f, frontend, gen, error));

    for (float sample : gen.audio) {
        CHECK(std::isfinite(sample));
    }
}

TEST_CASE("real_model_mandarin_multichunk_text_synthesis_stable") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    // Pick the first Mandarin voice available in this model file
    std::string zh_voice;
    for (const auto & kv : model->voices) {
        if (!kv.first.empty() && kv.first[0] == 'z') {
            zh_voice = kv.first;
            break;
        }
    }
    if (zh_voice.empty()) { MESSAGE("skipping: no Mandarin (z*) voice found in model"); return; }

    // Phonemize and synthesize each chunk independently
    std::string phonemes;
    std::string error;
    CHECK(kokopop::phonemize_text(
        "真正的友谊不仅在于分享快乐，更在于能在彼此遇到困难时互相支持。",
        zh_voice, phonemes, error));
    CHECK(!phonemes.empty());

    kokopop_audio audio{};
    CHECK(kokopop::synthesize_phonemes(*model, phonemes, zh_voice, 1.0f, audio, error));
    CHECK(audio.samples != nullptr);
    CHECK(audio.n_samples > 0);
    bool all_finite = true;
    for (size_t i = 0; i < audio.n_samples; ++i) {
        if (!std::isfinite(audio.samples[i])) { all_finite = false; break; }
    }
    CHECK(all_finite);
    kokopop_audio_free(&audio);
}

// ---- Chunk token gate: the ids that were budgeted are the ids
// that reach inference. `Chunk::tokens` must be the chunk's phoneme string
// tokenized once — not a concatenation of per-unit sequences, each carrying its
// own framing — in both chunking modes.

TEST_CASE("chunk_tokens_are_the_canonical_sequence_long_form") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::string error;
    kokopop::VoiceFrontend frontend;
    REQUIRE(kokopop::make_voice_frontend(*model, "af_heart", frontend, error));

    const std::string text =
        "This is a first sentence, with a clause. And a second one follows here. "
        "A third sentence, longer than the others, closes the paragraph. "
        "Then a fourth. And a fifth, to force several chunks.";

    auto cfg = kokopop::make_long_form_config();
    cfg.target_min_tokens = 20;
    cfg.target_max_tokens = 40;
    cfg.soft_max_tokens   = 60;
    auto chunks = kokopop::chunk_text(text, cfg, frontend.phonemize,
                                      frontend.tokenize, error);
    REQUIRE(chunks.size() > 1);

    for (const auto & chunk : chunks) {
        std::vector<uint32_t> expected;
        REQUIRE(frontend.tokenize(chunk.phonemes, expected, error));
        CHECK_EQ(chunk.tokens, expected);
        CHECK_EQ(chunk.n_tokens, static_cast<int>(chunk.tokens.size()));
        // A concatenation of framed unit sequences would repeat the sentinels.
        for (size_t i = 1; i + 1 < chunk.tokens.size(); ++i) {
            CHECK_NE(chunk.tokens[i], chunk.tokens.front());
        }
        // Intermediate chunks carry no trailing punctuation.
        if (!chunk.is_last) {
            std::string trimmed = chunk.phonemes;
            kokopop::trim_trailing_chunk_punctuation(trimmed);
            CHECK_EQ(trimmed, chunk.phonemes);
        }
    }
}

TEST_CASE("chunk_tokens_are_the_canonical_sequence_adaptative") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::string error;
    kokopop::VoiceFrontend frontend;
    REQUIRE(kokopop::make_voice_frontend(*model, "af_heart", frontend, error));

    const std::string text =
        "This is a first sentence, with a clause. And a second one follows here. "
        "A third sentence, longer than the others, closes the paragraph.";

    auto cfg = kokopop::make_adaptative_config();
    auto units = kokopop::prepare_chunk_units(text, cfg, frontend.phonemize,
                                              frontend.tokenize, error);
    REQUIRE(units.size() > 1);

    size_t next = 0;
    int n_chunks = 0;
    while (next < units.size()) {
        auto chunk = kokopop::build_adaptative_chunk(
            units, next, cfg, 40, n_chunks == 0, frontend.tokenize, error);
        REQUIRE(!chunk.tokens.empty());
        ++n_chunks;

        std::vector<uint32_t> expected;
        REQUIRE(frontend.tokenize(chunk.phonemes, expected, error));
        CHECK_EQ(chunk.tokens, expected);
        CHECK_EQ(chunk.n_tokens, static_cast<int>(chunk.tokens.size()));
    }
    CHECK(n_chunks > 1);
}

TEST_CASE("synthesize_chunk_matches_synthesize_phonemes_on_the_chunk_string") {
    auto * model = shared_real_model();
    if (!model) { MESSAGE("skipping: models/kokoro-md.gguf not found"); return; }

    std::string error;
    kokopop::VoiceFrontend frontend;
    REQUIRE(kokopop::make_voice_frontend(*model, "af_heart", frontend, error));

    auto cfg = kokopop::make_long_form_config();
    auto chunks = kokopop::chunk_text(
        "Hello world, this is a chunk. And here is a second one.",
        cfg, frontend.phonemize, frontend.tokenize, error);
    REQUIRE(!chunks.empty());
    const auto & chunk = chunks.front();

    kokopop_audio from_ids{};
    REQUIRE(kokopop::synthesize_chunk(*model, chunk, "af_heart", 1.0f,
                                      kokopop::KokoroDiffusionOptions{}, 0,
                                      from_ids, error));
    kokopop_audio from_phonemes{};
    REQUIRE(kokopop::synthesize_phonemes(*model, chunk.phonemes, "af_heart", 1.0f,
                                         from_phonemes, error));

    REQUIRE_EQ(from_ids.n_samples, from_phonemes.n_samples);
    bool identical = true;
    for (size_t i = 0; i < from_ids.n_samples; ++i) {
        if (from_ids.samples[i] != from_phonemes.samples[i]) { identical = false; break; }
    }
    CHECK(identical);
    kokopop_audio_free(&from_ids);
    kokopop_audio_free(&from_phonemes);
}
