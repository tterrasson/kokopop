#include "test_helpers.h"

// ---- Phonemizer ----

TEST_CASE("french_phonemizer_matches_kokoro") {
    std::string phonemes;
    std::string error;
    CHECK(kokopop::phonemize_text("Bonjour", "ff_siwis", phonemes, error));
    CHECK_EQ(phonemes, std::string("bɔ̃ʒˈuʁ"));
}

TEST_CASE("phonemizer_preserves_prosody_punctuation") {
    std::string question;
    std::string statement;
    std::string error;
    CHECK(kokopop::phonemize_text("Tu aimes les films ?", "ff_siwis", question, error));
    CHECK_EQ(question, std::string("ty ˈɛm le fˈilm?"));
    CHECK(kokopop::phonemize_text("Tu aimes les films.", "ff_siwis", statement, error));
    CHECK_EQ(statement, std::string("ty ˈɛm le fˈilm."));
    CHECK(question != statement);
}

TEST_CASE("espeak_voice_for_kokoro_voice mappings") {
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("af_heart"),   std::string("gmw/en-US"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("bf_emma"),    std::string("gmw/en"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("ef_dora"),    std::string("roa/es"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("ff_siwis"),   std::string("roa/fr"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("hf_alpha"),   std::string("inc/hi"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("if_sara"),    std::string("roa/it"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("jf_alpha"),   std::string("jpx/ja"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("pf_dora"),    std::string("roa/pt-BR"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("zf_xiaobei"), std::string("sit/cmn"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("xf_unknown"), std::string("gmw/en-US"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice(""),           std::string("gmw/en-US"));
}

TEST_CASE("normalize_espeak_phonemes diphthong substitutions") {
    CHECK_EQ(kokopop::normalize_espeak_phonemes("aɪ", 'a'), std::string("I"));
    CHECK_EQ(kokopop::normalize_espeak_phonemes("aʊ", 'a'), std::string("W"));
    CHECK_EQ(kokopop::normalize_espeak_phonemes("eɪ", 'a'), std::string("A"));
    CHECK_EQ(kokopop::normalize_espeak_phonemes("oʊ", 'a'), std::string("O"));
    CHECK_EQ(kokopop::normalize_espeak_phonemes("tʃ", 'a'), std::string("ʧ"));
    CHECK_EQ(kokopop::normalize_espeak_phonemes("dʒ", 'a'), std::string("ʤ"));
    CHECK_EQ(kokopop::normalize_espeak_phonemes("aɪ  aʊ", 'a'), std::string("I W"));
    CHECK_EQ(kokopop::normalize_espeak_phonemes("a-b^c", 'a'), std::string("abc"));
    CHECK_EQ(kokopop::normalize_espeak_phonemes("ɔ̃", 'f'), std::string("ɔ̃"));
    CHECK_EQ(kokopop::normalize_espeak_phonemes("ɔ̃", 'a'), std::string("ɔ"));
}

TEST_CASE("phonemize_text english produces non-empty phonemes") {
    std::string phonemes, error;
    bool ok = kokopop::phonemize_text("Hello world", "af_heart", phonemes, error);
    CHECK(ok);
    CHECK(error.empty());
    CHECK(!phonemes.empty());
}

TEST_CASE("phonemize_text unknown voice char falls back to en-US and succeeds") {
    std::string phonemes, error;
    bool ok = kokopop::phonemize_text("Hello", "?unknown", phonemes, error);
    CHECK(ok);
    CHECK(!phonemes.empty());
}

// ---- New tests ----

TEST_CASE("phonemize_text_empty_input") {
    std::string phonemes, error;
    bool ok = kokopop::phonemize_text("", "af_heart", phonemes, error);
    // Empty input should fail or produce empty phonemes
    if (!ok) {
        CHECK(!error.empty());
    } else {
        CHECK(phonemes.empty());
    }
}

TEST_CASE("phonemize_text_whitespace_only") {
    std::string phonemes, error;
    bool ok = kokopop::phonemize_text("   ", "af_heart", phonemes, error);
    if (!ok) {
        CHECK(!error.empty());
    } else {
        // Whitespace-only should produce empty or minimal phonemes
        CHECK(phonemes.empty());
    }
}

TEST_CASE("phonemize_text_single_char") {
    std::string phonemes, error;

    // "a"
    CHECK(kokopop::phonemize_text("a", "af_heart", phonemes, error));
    CHECK(!phonemes.empty());

    // "x"
    CHECK(kokopop::phonemize_text("x", "af_heart", phonemes, error));
    CHECK(!phonemes.empty());
}

TEST_CASE("phonemize_text_punctuation_only") {
    std::string phonemes, error;
    bool ok = kokopop::phonemize_text("?,!;:.", "af_heart", phonemes, error);
    if (ok) {
        CHECK(error.empty());
    }
}

TEST_CASE("phonemize_text_mixed_script") {
    std::string phonemes, error;
    // "Hello 世界" — behavior with unsupported characters
    bool ok = kokopop::phonemize_text("Hello \u4e16\u754c", "af_heart", phonemes, error);
    // May or may not succeed depending on espeak-ng support
    if (ok) {
        CHECK(!phonemes.empty());
    }
}

TEST_CASE("phonemize_text_long_text") {
    std::string phonemes, error;
    std::string long_text = "This is a very long sentence that exceeds two hundred characters in total length. "
        "It should be phonemized correctly without any issues or truncation problems whatsoever. "
        "The phonemizer must handle this gracefully and produce valid output phonemes for the entire text.";
    CHECK(kokopop::phonemize_text(long_text, "af_heart", phonemes, error));
    CHECK(!phonemes.empty());
}

TEST_CASE("phonemize_text_preserves_multiple_punctuations") {
    std::string phonemes, error;
    CHECK(kokopop::phonemize_text("Hello! World? Yes!", "af_heart", phonemes, error));
    CHECK(phonemes.find('!') != std::string::npos);
    CHECK(phonemes.find('?') != std::string::npos);
}

TEST_CASE("normalize_espeak_phonemes_empty") {
    CHECK_EQ(kokopop::normalize_espeak_phonemes("", 'a'), std::string(""));
}

TEST_CASE("normalize_espeak_phonemes_multiple_spaces") {
    // Whitespace should be collapsed
    CHECK_EQ(kokopop::normalize_espeak_phonemes("a  b   c", 'a'), std::string("a b c"));
}

TEST_CASE("normalize_espeak_phonemes_trailing_space") {
    // Should not have trailing space
    std::string result = kokopop::normalize_espeak_phonemes("abc ", 'a');
    CHECK_EQ(result, std::string("abc"));
}

TEST_CASE("normalize_espeak_phonemes_tilde_removal_non_french") {
    // Non-French: tilde (combining tilde) should be removed
    CHECK_EQ(kokopop::normalize_espeak_phonemes("\u0254\u0303", 'a'), std::string("\u0254"));
}

TEST_CASE("normalize_espeak_phonemes_french_preserves_tilde") {
    // French: tilde should be preserved
    CHECK_EQ(kokopop::normalize_espeak_phonemes("\u0254\u0303", 'f'), std::string("\u0254\u0303"));
}

TEST_CASE("normalize_espeak_phonemes_carriage_return") {
    // \n and whitespace are normalized to spaces
    std::string result = kokopop::normalize_espeak_phonemes("abc\ndef", 'a');
    CHECK(result.find('\n') == std::string::npos);
    CHECK(result.find(' ') != std::string::npos);
}

TEST_CASE("normalize_espeak_phonemes_angle_quotes") {
    // « » should become ()
    std::string result = kokopop::normalize_espeak_phonemes("\u00ab\u00bb", 'a');
    CHECK_EQ(result, std::string("()"));
}

TEST_CASE("normalize_espeak_phonemes_caret_hyphen_removal") {
    // ^ and - should be removed
    CHECK_EQ(kokopop::normalize_espeak_phonemes("a^b-c", 'a'), std::string("abc"));
}

TEST_CASE("espeak_voice_for_kokoro_voice_all_prefixes") {
    // Test exhaustive prefix coverage: a/b/e/f/h/i/j/p/z + unknown
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("af_test"),  std::string("gmw/en-US"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("bf_test"),  std::string("gmw/en"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("ef_test"),  std::string("roa/es"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("ff_test"),  std::string("roa/fr"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("hf_test"),  std::string("inc/hi"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("if_test"),  std::string("roa/it"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("jf_test"),  std::string("jpx/ja"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("pf_test"),  std::string("roa/pt-BR"));
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("zf_test"),  std::string("sit/cmn"));
    // Unknown prefix falls back to en-US
    CHECK_EQ(kokopop::espeak_voice_for_kokoro_voice("xx_test"),  std::string("gmw/en-US"));
}

TEST_CASE("phonemizer_different_voices_give_different_phonemes") {
    // Regression: the phoneme cache key must include the voice so that the
    // same text phonemized for two different languages stays independent.
    std::string fr_phonemes, en_phonemes, error;
    CHECK(kokopop::phonemize_text("Bonjour", "ff_siwis", fr_phonemes, error));
    CHECK(kokopop::phonemize_text("Bonjour", "af_heart", en_phonemes, error));
    // French and English phonemizations of the same word must differ
    CHECK(fr_phonemes != en_phonemes);
}

TEST_CASE("phonemizer_same_voice_same_text_is_deterministic") {
    std::string p1, p2, error;
    CHECK(kokopop::phonemize_text("Hello world", "af_heart", p1, error));
    CHECK(kokopop::phonemize_text("Hello world", "af_heart", p2, error));
    CHECK_EQ(p1, p2);
}
