#include "test_helpers.h"
#include "synthesis/chunker/chunker.h"
#include "streaming/streaming.h"

// ---- ChunkConfig merge ----

TEST_CASE("merge_chunk_config_no_override_keeps_base") {
    kokopop::ChunkConfig base = kokopop::make_adaptative_config();
    kokopop::ChunkConfig override{};  // all defaults — should not change base
    kokopop::ChunkConfig merged = kokopop::merge_chunk_config(base, override);
    CHECK_EQ(merged.target_min_tokens, base.target_min_tokens);
    CHECK_EQ(merged.target_max_tokens, base.target_max_tokens);
    CHECK_EQ(merged.soft_max_tokens,   base.soft_max_tokens);
    CHECK_EQ(merged.crossfade_ms,      base.crossfade_ms);
}

TEST_CASE("merge_chunk_config_target_min_override") {
    kokopop::ChunkConfig base = kokopop::make_adaptative_config();
    kokopop::ChunkConfig override{};
    override.target_min_tokens = 5;
    kokopop::ChunkConfig merged = kokopop::merge_chunk_config(base, override);
    CHECK_EQ(merged.target_min_tokens, 5);
    // Other fields unchanged
    CHECK_EQ(merged.target_max_tokens, base.target_max_tokens);
}

TEST_CASE("merge_chunk_config_target_max_override") {
    kokopop::ChunkConfig base = kokopop::make_adaptative_config();
    kokopop::ChunkConfig override{};
    override.target_max_tokens = 42;
    kokopop::ChunkConfig merged = kokopop::merge_chunk_config(base, override);
    CHECK_EQ(merged.target_max_tokens, 42);
    CHECK_EQ(merged.target_min_tokens, base.target_min_tokens);
}

TEST_CASE("merge_chunk_config_multiple_overrides") {
    kokopop::ChunkConfig base = kokopop::make_long_form_config();
    kokopop::ChunkConfig override{};
    override.target_min_tokens   = 3;
    override.target_max_tokens   = 10;
    override.crossfade_ms        = 5;
    override.sentence_pause_ms   = 50;
    kokopop::ChunkConfig merged = kokopop::merge_chunk_config(base, override);
    CHECK_EQ(merged.target_min_tokens,  3);
    CHECK_EQ(merged.target_max_tokens, 10);
    CHECK_EQ(merged.crossfade_ms,       5);
    CHECK_EQ(merged.sentence_pause_ms, 50);
    // Unset fields keep base values
    CHECK_EQ(merged.hard_max_tokens, base.hard_max_tokens);
}

// ---- prepare_synthesis respects chunk config override ----

TEST_CASE("prepare_synthesis_chunk_config_override_changes_chunk_count") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    REQUIRE(kokopop::load_model_from_gguf(gguf, &options, model, error));

    const std::string text =
        "This is a sentence. This is another sentence. And yet another one here. "
        "A fourth sentence follows. The fifth sentence ends the paragraph.";

    // Small chunks: target_max = 10 tokens → many chunks
    kokopop::ChunkConfig small_cfg{};
    small_cfg.target_min_tokens = 1;
    small_cfg.target_max_tokens = 10;
    auto plan_small = kokopop::prepare_synthesis(
        *model, text, "af_heart", 1.0f, kokopop::StreamMode::Adaptative, error, &small_cfg);

    // Large chunks: target_max = 300 tokens → fewer chunks
    kokopop::ChunkConfig large_cfg{};
    large_cfg.target_min_tokens = 100;
    large_cfg.target_max_tokens = 300;
    auto plan_large = kokopop::prepare_synthesis(
        *model, text, "af_heart", 1.0f, kokopop::StreamMode::Adaptative, error, &large_cfg);

    CHECK(plan_small.chunks.size() > plan_large.chunks.size());
}

TEST_CASE("prepare_synthesis_no_override_uses_mode_defaults") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    REQUIRE(kokopop::load_model_from_gguf(gguf, &options, model, error));

    const std::string text = "Hello world. This is a test sentence for chunking.";

    auto plan_adaptative = kokopop::prepare_synthesis(
        *model, text, "af_heart", 1.0f, kokopop::StreamMode::Adaptative, error, nullptr);
    auto plan_longform = kokopop::prepare_synthesis(
        *model, text, "af_heart", 1.0f, kokopop::StreamMode::LongForm, error, nullptr);

    // Both should produce at least one chunk
    CHECK(plan_adaptative.chunks.size() > 0);
    CHECK(plan_longform.chunks.size() > 0);
    // Configs reflect the correct mode
    CHECK_EQ(plan_adaptative.config.target_min_tokens,
             kokopop::make_adaptative_config().target_min_tokens);
    CHECK_EQ(plan_longform.config.target_min_tokens,
             kokopop::make_long_form_config().target_min_tokens);
}

TEST_CASE("chunk_text_adds_terminal_sentence_boundary_for_complete_text") {
    std::string error;
    auto tokenize = [](const std::string & phonemes,
                       std::vector<uint32_t> & ids,
                       std::string &) {
        ids.assign(std::max<size_t>(1, phonemes.size() / 2), 1);
        return true;
    };

    auto chunks = kokopop::chunk_text(
        "Hello world", "af_heart",
        kokopop::make_long_form_config(), tokenize, error);

    REQUIRE_EQ(chunks.size(), 1);
    CHECK_EQ(chunks.front().text, std::string("Hello world."));
    CHECK_EQ(chunks.front().boundary_after, kokopop::Boundary::Sentence);
    CHECK(chunks.front().phonemes.find('.') != std::string::npos);
}

TEST_CASE("adaptative_first_chunk_stops_at_short_natural_pause") {
    std::string error;
    auto tokenize = [](const std::string & phonemes,
                       std::vector<uint32_t> & ids,
                       std::string &) {
        ids.assign(std::max<size_t>(1, phonemes.size() / 4), 1);
        return true;
    };

    auto cfg = kokopop::make_adaptative_config();
    auto units = kokopop::prepare_chunk_units(
        "Bonjour, comment allez-vous ?", "ff_siwis", cfg, tokenize, error);
    REQUIRE(units.size() >= 2);

    size_t next = 0;
    auto chunk = kokopop::build_adaptative_chunk(units, next, cfg, 96, true);
    CHECK_EQ(chunk.text, std::string("Bonjour,"));
    CHECK_EQ(chunk.boundary_after, kokopop::Boundary::ClauseWeak);
    CHECK_EQ(next, 1u);
}

TEST_CASE("adaptative_first_chunk_keeps_long_clause_until_first_pause") {
    std::string error;
    auto tokenize = [](const std::string & phonemes,
                       std::vector<uint32_t> & ids,
                       std::string &) {
        ids.assign(std::max<size_t>(1, phonemes.size() / 4), 1);
        return true;
    };

    auto cfg = kokopop::make_adaptative_config();
    auto units = kokopop::prepare_chunk_units(
        "Bonjour Monsieur Jean qui aime les fraises, comment allez-vous ?",
        "ff_siwis", cfg, tokenize, error);
    REQUIRE(units.size() >= 2);

    size_t next = 0;
    auto chunk = kokopop::build_adaptative_chunk(units, next, cfg, 32, true);
    CHECK_EQ(chunk.text, std::string("Bonjour Monsieur Jean qui aime les fraises,"));
    CHECK_EQ(chunk.boundary_after, kokopop::Boundary::ClauseWeak);
    CHECK_EQ(next, 1u);
}

TEST_CASE("adaptative_long_unpunctuated_text_caps_at_target_max") {
    std::vector<kokopop::Unit> units;
    for (int i = 0; i < 20; ++i) {
        kokopop::Unit unit;
        unit.text = "word ";
        unit.phonemes = "word";
        unit.tokens.assign(5, 1);
        unit.n_tokens = 5;
        unit.boundary_after = kokopop::Boundary::None;
        units.push_back(std::move(unit));
    }

    auto cfg = kokopop::make_adaptative_config();
    cfg.target_min_tokens = 10;
    cfg.target_max_tokens = 20;
    cfg.soft_max_tokens = 25;
    cfg.hard_max_tokens = 30;

    size_t next = 0;
    auto chunk = kokopop::build_adaptative_chunk(units, next, cfg, 20, false);
    // Caps at target_max_tokens=20 even without a boundary.
    CHECK_EQ(chunk.n_tokens, 20);
    CHECK_EQ(next, 4u);
}

TEST_CASE("adaptative_controller_adjusts_budget_from_generation_speed") {
    kokopop::AdaptativeChunkController controller;
    controller.min_tokens = 10;
    controller.max_tokens = 200;
    CHECK_EQ(controller.target_tokens(), 58);

    controller.observe(100, 300.0, 3000.0);
    // Fast synthesis: target rises above the conservative baseline (58).
    // Cold-start seeds the EWMA at 12 ms/token so the jump is dampened.
    CHECK_GT(controller.target_tokens(), 58);

    controller.observe(20, 1000.0, 600.0);
    // Slow synthesis: target falls back below the initial baseline.
    CHECK_LT(controller.target_tokens(), 58);
}

// ---- /voices: model exposes its voices map ----

TEST_CASE("mock_model_has_voices") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    REQUIRE(kokopop::load_model_from_gguf(gguf, &options, model, error));
    // The mock GGUF embeds at least one voice
    CHECK(model->voices.size() > 0);
    for (const auto & kv : model->voices) {
        CHECK(!kv.first.empty());
        CHECK_EQ(kv.first, kv.second.name);
    }
}
