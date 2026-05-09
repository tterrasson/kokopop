#include "test_helpers.h"
#include "synthesis/chunker/chunker.h"
#include "streaming/streaming.h"

// ---- ChunkConfig merge ----

TEST_CASE("merge_chunk_config_no_override_keeps_base") {
    kokopop::ChunkConfig base = kokopop::make_interactive_config();
    kokopop::ChunkConfig override{};  // all defaults — should not change base
    kokopop::ChunkConfig merged = kokopop::merge_chunk_config(base, override);
    CHECK_EQ(merged.target_min_tokens, base.target_min_tokens);
    CHECK_EQ(merged.target_max_tokens, base.target_max_tokens);
    CHECK_EQ(merged.soft_max_tokens,   base.soft_max_tokens);
    CHECK_EQ(merged.crossfade_ms,      base.crossfade_ms);
}

TEST_CASE("merge_chunk_config_target_min_override") {
    kokopop::ChunkConfig base = kokopop::make_interactive_config();
    kokopop::ChunkConfig override{};
    override.target_min_tokens = 5;
    kokopop::ChunkConfig merged = kokopop::merge_chunk_config(base, override);
    CHECK_EQ(merged.target_min_tokens, 5);
    // Other fields unchanged
    CHECK_EQ(merged.target_max_tokens, base.target_max_tokens);
}

TEST_CASE("merge_chunk_config_target_max_override") {
    kokopop::ChunkConfig base = kokopop::make_interactive_config();
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
        *model, text, "af_heart", 1.0f, kokopop::StreamMode::Interactive, error, &small_cfg);

    // Large chunks: target_max = 300 tokens → fewer chunks
    kokopop::ChunkConfig large_cfg{};
    large_cfg.target_min_tokens = 100;
    large_cfg.target_max_tokens = 300;
    auto plan_large = kokopop::prepare_synthesis(
        *model, text, "af_heart", 1.0f, kokopop::StreamMode::Interactive, error, &large_cfg);

    CHECK(plan_small.chunks.size() > plan_large.chunks.size());
}

TEST_CASE("prepare_synthesis_no_override_uses_mode_defaults") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    REQUIRE(kokopop::load_model_from_gguf(gguf, &options, model, error));

    const std::string text = "Hello world. This is a test sentence for chunking.";

    auto plan_interactive = kokopop::prepare_synthesis(
        *model, text, "af_heart", 1.0f, kokopop::StreamMode::Interactive, error, nullptr);
    auto plan_longform = kokopop::prepare_synthesis(
        *model, text, "af_heart", 1.0f, kokopop::StreamMode::LongForm, error, nullptr);

    // Both should produce at least one chunk
    CHECK(plan_interactive.chunks.size() > 0);
    CHECK(plan_longform.chunks.size() > 0);
    // Configs reflect the correct mode
    CHECK_EQ(plan_interactive.config.target_min_tokens,
             kokopop::make_interactive_config().target_min_tokens);
    CHECK_EQ(plan_longform.config.target_min_tokens,
             kokopop::make_long_form_config().target_min_tokens);
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
