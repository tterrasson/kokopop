#include "test_helpers.h"

// ---- Tokenisation phonèmes ----

TEST_CASE("tokenize_phonemes_bonjour") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    std::vector<uint32_t> ids;
    CHECK(model->tokenize_phonemes("b", ids, error));
    CHECK(error.empty());
    // vocab: "", "a", "b", "c", " ", "ɑ", "ɔ", "ʃ"
    // "b" = token id 2, plus BOS/EOS
    CHECK_EQ(ids.size(), 3u); // BOS(0) + "b"(2) + EOS(0)
}

TEST_CASE("tokenize_phonemes_starts_with_zero") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    std::vector<uint32_t> ids;
    CHECK(model->tokenize_phonemes("abc", ids, error));
    CHECK_EQ(ids.front(), 0u); // BOS
}

TEST_CASE("tokenize_phonemes_ends_with_zero") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    std::vector<uint32_t> ids;
    CHECK(model->tokenize_phonemes("abc", ids, error));
    CHECK_EQ(ids.back(), 0u); // EOS
}

TEST_CASE("tokenize_phonemes_unknown_token") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    // vocab: "", "a", "b", "c", " ", "ɑ", "ɔ", "ʃ"
    // "z" is unknown → should be ignored, only BOS/EOS
    std::vector<uint32_t> ids;
    CHECK(model->tokenize_phonemes("z", ids, error));
    CHECK_EQ(ids.size(), 2u); // only BOS + EOS
}

TEST_CASE("tokenize_phonemes_empty_string") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    std::vector<uint32_t> ids;
    CHECK(model->tokenize_phonemes("", ids, error));
    CHECK_EQ(ids.size(), 2u); // BOS + EOS only
}

TEST_CASE("tokenize_phonemes_context_overflow") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    // > 511 phonemes should produce an error
    std::string long_phonemes(512, 'a');
    std::vector<uint32_t> ids;
    CHECK(!model->tokenize_phonemes(long_phonemes, ids, error));
    CHECK(!error.empty());
}

TEST_CASE("tokenize_phonemes_invalid_utf8") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    // Invalid UTF-8: truncated sequence
    std::string invalid("\xC3", 1);
    std::vector<uint32_t> ids;
    CHECK(!model->tokenize_phonemes(invalid, ids, error));
    CHECK(!error.empty());
}

TEST_CASE("tokenize_phonemes_single_known") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    // "a" = token id 1
    std::vector<uint32_t> ids;
    CHECK(model->tokenize_phonemes("a", ids, error));
    CHECK_EQ(ids.size(), 3u); // BOS + "a" + EOS
    CHECK_EQ(ids[1], 1u);
}

TEST_CASE("tokenize_phonemes_all_unknown") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    // All unknown characters → BOS + EOS only
    std::vector<uint32_t> ids;
    CHECK(model->tokenize_phonemes("xyz", ids, error));
    CHECK_EQ(ids.size(), 2u);
}

TEST_CASE("tokenize_phonemes_mixed_known_unknown") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));

    // "azb" — "a" and "b" known, "z" unknown
    std::vector<uint32_t> ids;
    CHECK(model->tokenize_phonemes("azb", ids, error));
    // Should have BOS + "a" + "b" + EOS (z ignored)
    CHECK_EQ(ids.size(), 4u);
    CHECK_EQ(ids[0], 0u); // BOS
    CHECK_EQ(ids[1], 1u); // "a"
    CHECK_EQ(ids[2], 2u); // "b"
    CHECK_EQ(ids[3], 0u); // EOS
}
