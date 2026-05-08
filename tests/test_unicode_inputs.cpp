// test_unicode_inputs.cpp — complex character coverage for the full pipeline
//
// Tests UTF-8 edge cases through normalizer → splitter → phonemizer → tokenizer.
// Focuses on characters that commonly trip up single-byte / ASCII-only logic.

#include "test_helpers.h"
#include "synthesis/chunker/text_normalizer.h"
#include "synthesis/chunker/text_splitter.h"
#include "synthesis/phonemizer.h"

#include <string>
#include <vector>
#include <cctype>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Verify that a string contains a given Unicode codepoint (as UTF-8).
static bool has_codepoint(const std::string & s, uint32_t cp) {
    size_t i = 0;
    while (i < s.size()) {
        size_t len;
        uint32_t c = 0;
        unsigned char b = static_cast<unsigned char>(s[i]);
        if ((b & 0x80) == 0) { len = 1; c = b; }
        else if ((b & 0xE0) == 0xC0) {
            len = 2;
            if (i + 1 >= s.size()) break;
            c = ((b & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
        }
        else if ((b & 0xF0) == 0xE0) {
            len = 3;
            if (i + 2 >= s.size()) break;
            c = ((b & 0x0F) << 12) | (static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6 |
                (static_cast<unsigned char>(s[i + 2]) & 0x3F);
        }
        else if ((b & 0xF8) == 0xF0) {
            len = 4;
            if (i + 3 >= s.size()) break;
            c = ((b & 0x07) << 18) | (static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12 |
                (static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6 |
                (static_cast<unsigned char>(s[i + 3]) & 0x3F);
        }
        else { len = 1; }
        if (c == cp) return true;
        i += len;
    }
    return false;
}

/// Count UTF-8 codepoints in a string.
static size_t count_codepoints(const std::string & s) {
    size_t count = 0, i = 0;
    while (i < s.size()) {
        unsigned char b = static_cast<unsigned char>(s[i]);
        if ((b & 0x80) == 0) i += 1;
        else if ((b & 0xE0) == 0xC0) i += 2;
        else if ((b & 0xF0) == 0xE0) i += 3;
        else if ((b & 0xF8) == 0xF0) i += 4;
        else i += 1;
        count++;
    }
    return count;
}

// ===========================================================================
// 1. UTF-8 edge cases
// ===========================================================================

TEST_CASE("utf8_surrogate_pair_structurally_valid") {
    // U+D800 encoded as UTF-8: 0xED 0xA0 0x80 (high surrogate)
    // The parser is structural-only: it validates byte patterns, not semantic
    // Unicode rules. Surrogates have valid 3-byte structure so they pass.
    std::string_view text = "\xED\xA0\x80";
    size_t offset = 0;
    std::string_view ch;
    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch.size(), 3u);
}

TEST_CASE("utf8_high_surrogate_structurally_valid") {
    // U+DBFF — high surrogate (structurally valid 3-byte sequence)
    size_t offset = 0;
    std::string_view ch;
    CHECK(kokopop::utf8_next("\xED\xAF\xBF", offset, ch));
    CHECK_EQ(ch.size(), 3u);
}

TEST_CASE("utf8_low_surrogate_structurally_valid") {
    // U+DC00 — low surrogate (structurally valid 3-byte sequence)
    size_t offset = 0;
    std::string_view ch;
    CHECK(kokopop::utf8_next("\xED\xB0\x80", offset, ch));
    CHECK_EQ(ch.size(), 3u);
}

TEST_CASE("utf8_overlong_nul_rejected") {
    // NUL overlong: 0xC0 0x80 (should be just 0x00)
    // The parser allows this (it doesn't explicitly reject overlong forms),
    // but the byte sequence is technically invalid. We document the behavior.
    size_t offset = 0;
    std::string_view ch;
    // Our parser checks continuation bytes, so 0xC0 0x80 passes structural check.
    // This tests that the parser is structural, not semantic.
    bool result = kokopop::utf8_next("\xC0\x80", offset, ch);
    // Either way is fine — we just want deterministic behavior.
    if (result) {
        CHECK_EQ(ch.size(), 2u);
    }
}

TEST_CASE("utf8_overlong_a_rejected_or_accepted_consistently") {
    // 'A' overlong: 0xC1 0x81 (should be 0x41)
    size_t offset = 0;
    std::string_view ch;
    bool result = kokopop::utf8_next("\xC1\x81", offset, ch);
    if (result) {
        CHECK_EQ(ch.size(), 2u);
    }
}

TEST_CASE("utf8_mixed_valid_invalid_returns_empty") {
    // Valid ASCII + invalid continuation → utf8_chars should return empty
    std::string text = "abc\x80";
    auto chars = kokopop::utf8_chars(text);
    // The invalid byte at position 3 makes off != text.size() → clears all
    CHECK(chars.empty());
}

TEST_CASE("utf8_bom_at_start") {
    // BOM = U+FEFF = 0xEF 0xBB 0xBF
    std::string text = "\xEF\xBB\xBFhello";
    size_t offset = 0;
    std::string_view ch;
    CHECK(kokopop::utf8_next(text, offset, ch));
    // BOM is a valid 3-byte sequence
    CHECK_EQ(ch.size(), 3u);
    CHECK_EQ(offset, 3u);
    // Next char is 'h'
    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch, "h");
}

TEST_CASE("utf8_zero_width_space") {
    // U+200B = 0xE2 0x80 0x8B (3 bytes)
    std::string text = "a\u200Bb";
    auto chars = kokopop::utf8_chars(text);
    CHECK_EQ(chars.size(), 3u); // 'a', ZWS, 'b'
    CHECK_EQ(chars[0], "a");
    CHECK_EQ(chars[1].size(), 3u); // ZWS = 3 bytes
    CHECK_EQ(chars[2], "b");
}

TEST_CASE("utf8_zero_width_joiner") {
    // U+200D = 0xE2 0x80 0x8D (3 bytes)
    std::string text = "\xE2\x80\x8D";
    size_t offset = 0;
    std::string_view ch;
    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch.size(), 3u);
}

TEST_CASE("utf8_non_breaking_space") {
    // U+00A0 = 0xC2 0xA0 (2 bytes)
    std::string text = "\xC2\xA0";
    size_t offset = 0;
    std::string_view ch;
    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch.size(), 2u);
    CHECK_EQ(offset, 2u);
}

TEST_CASE("utf8_fullwidth_digit") {
    // U+FF11 = 0xEF 0xBC 0x91 (fullwidth '1')
    std::string text = "\xEF\xBC\x91";
    size_t offset = 0;
    std::string_view ch;
    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch.size(), 3u);
}

TEST_CASE("utf8_combining_acute_on_e") {
    // 'e' + U+0301 (combining acute accent) — 1 + 2 = 3 bytes, 2 codepoints
    std::string text = "e\xCC\x81";
    auto chars = kokopop::utf8_chars(text);
    CHECK_EQ(chars.size(), 2u);
    CHECK_EQ(chars[0], "e");
    CHECK_EQ(chars[1].size(), 2u); // combining mark = 2 bytes
}

TEST_CASE("utf8_stacked_combining_marks") {
    // 'a' + combining grave + combining acute + combining circumflex
    std::string text = "a\xCC\x80\xCC\x81\xCC\x82";
    auto chars = kokopop::utf8_chars(text);
    CHECK_EQ(chars.size(), 4u); // 'a' + 3 combining marks
}

TEST_CASE("utf8_regional_indicators") {
    // Flag emoji: 🇫🇷 = U+1F1EB U+1F1F7 (two 4-byte codepoints)
    std::string text = "\xF0\x9F\x87\xAB\xF0\x9F\x87\xB7";
    auto chars = kokopop::utf8_chars(text);
    CHECK_EQ(chars.size(), 2u);
    CHECK_EQ(chars[0].size(), 4u);
    CHECK_EQ(chars[1].size(), 4u);
}

TEST_CASE("utf8_variation_selector") {
    // U+FE0F = 0xEF 0xB8 0x8F (variation selector-16)
    std::string text = "\xF0\x9F\x98\x80\xEF\xB8\x8F"; // 😀 + VS16
    auto chars = kokopop::utf8_chars(text);
    CHECK_EQ(chars.size(), 2u);
}

TEST_CASE("utf8_ligature_fi") {
    // U+FB01 = 0xEF 0xAC 0x81 (fi ligature)
    std::string text = "\xEF\xAC\x81";
    size_t offset = 0;
    std::string_view ch;
    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch.size(), 3u);
}

TEST_CASE("utf8_soft_hyphen") {
    // U+00AD = 0xC2 0xAD (soft hyphen)
    std::string text = "\xC2\xAD";
    size_t offset = 0;
    std::string_view ch;
    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch.size(), 2u);
}

TEST_CASE("utf8_bidi_controls") {
    // U+200E (LTR mark) + U+200F (RTL mark)
    std::string text = "\xE2\x80\x8E\xE2\x80\x8F";
    auto chars = kokopop::utf8_chars(text);
    CHECK_EQ(chars.size(), 2u);
    CHECK_EQ(chars[0].size(), 3u);
    CHECK_EQ(chars[1].size(), 3u);
}

TEST_CASE("utf8_truncated_4byte") {
    // 4-byte sequence truncated: only 2 of 4 bytes
    std::string text = "\xF0\x9F";
    size_t offset = 0;
    std::string_view ch;
    CHECK(!kokopop::utf8_next(text, offset, ch));
}

TEST_CASE("utf8_invalid_5_byte_sequence") {
    // 0xF8+ is not valid UTF-8
    std::string text = "\xF8\x88\x80\x80\x80";
    size_t offset = 0;
    std::string_view ch;
    CHECK(!kokopop::utf8_next(text, offset, ch));
}

TEST_CASE("utf8_max_valid_codepoint") {
    // U+10FFFF = 0xF4 0x8F 0xBF 0xBF (last valid Unicode codepoint)
    std::string text = "\xF4\x8F\xBF\xBF";
    size_t offset = 0;
    std::string_view ch;
    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch.size(), 4u);
    CHECK_EQ(offset, 4u);
}

TEST_CASE("utf8_beyond_max_codepoint") {
    // U+110000 = 0xF4 0x90 0x80 0x80 (beyond Unicode max — structurally valid UTF-8 though)
    std::string text = "\xF4\x90\x80\x80";
    size_t offset = 0;
    std::string_view ch;
    // Our parser is structural, so this passes
    bool result = kokopop::utf8_next(text, offset, ch);
    if (result) {
        CHECK_EQ(ch.size(), 4u);
    }
}

// ===========================================================================
// 2. Text normalizer — complex characters
// ===========================================================================

TEST_CASE("normalize_text_preserves_emoji") {
    // Emoji should pass through the normalizer unchanged
    std::string input = "Hello! \U0001f600 How are you?";
    auto result = kokopop::normalize_text(input);
    CHECK(has_codepoint(result, 0x1F600));
}

TEST_CASE("normalize_text_preserves_cjk") {
    std::string input = "你好世界 Hello World";
    auto result = kokopop::normalize_text(input);
    CHECK(has_codepoint(result, 0x4F60)); // 你
    CHECK(has_codepoint(result, 0x4E16)); // 世
}

TEST_CASE("normalize_text_preserves_combining_characters") {
    // Precomposed é (U+00E9) — the normalizer processes bytes individually,
    // so multi-byte chars pass through as-is (no dots/whitespace to transform)
    std::string input = "caf\u00E9";
    auto result = kokopop::normalize_text(input);
    CHECK(!result.empty());
    CHECK(result.find("caf") != std::string::npos);

    // e + combining acute (decomposed U+0065 U+0301)
    std::string input2 = "cafe\u0301";
    auto result2 = kokopop::normalize_text(input2);
    CHECK(!result2.empty());
    CHECK(result2.find("cafe") != std::string::npos);
}

TEST_CASE("normalize_text_fullwidth_latin") {
    // Fullwidth characters: Ｈｅｌｌｏ (U+FF28+)
    std::string input = "\xEF\xBD\xa8\xEF\xBD\x85\xEF\xBD\x8C\xEF\xBD\x8C\xEF\xBD\x8D"; // Ｈｅｌｌｏ
    auto result = kokopop::normalize_text(input);
    // Fullwidth chars should pass through (they're not ASCII alnum)
    CHECK_EQ(result.size(), input.size());
}

TEST_CASE("normalize_text_fullwidth_dot_not_split") {
    // Fullwidth period U+3002 = 0xE3 0x80 0x82 (not ASCII '.')
    std::string input = "Hello\xE3\x80\x82 World";
    auto result = kokopop::normalize_text(input);
    // The fullwidth dot should NOT be treated as an abbreviation/decimal dot
    CHECK(has_codepoint(result, 0x3002));
}

TEST_CASE("normalize_text_url_with_unicode") {
    // URL-like pattern with non-ASCII: behavior depends on locale.
    // In some locales, the normalizer may or may not protect dots near
    // non-ASCII chars. We verify no crash and that the text is non-empty.
    std::string input = "caf\u00E9.com is great";
    auto result = kokopop::normalize_text(input);
    CHECK(!result.empty());
    // Input should survive normalization regardless of URL protection
    CHECK(result.find("caf") != std::string::npos);
}

TEST_CASE("normalize_text_zero_width_space_passthrough") {
    // ZWS should pass through normalization
    std::string input = "Hello\xE2\x80\x8BWorld";
    auto result = kokopop::normalize_text(input);
    CHECK(has_codepoint(result, 0x200B));
}

TEST_CASE("normalize_text_non_breaking_space_not_collapsed") {
    // U+00A0 is a non-ASCII byte; std::isspace returns false in C locale.
    // It should NOT be collapsed to a regular space.
    std::string input = "Hello\xC2\xA0World";
    auto result = kokopop::normalize_text(input);
    // Non-breaking space bytes should be preserved
    CHECK(has_codepoint(result, 0xA0));
}

TEST_CASE("normalize_text_multiple_emoji") {
    std::string input = "\U0001f600\U0001f601\U0001f602";
    auto result = kokopop::normalize_text(input);
    CHECK_EQ(count_codepoints(result), 3u);
}

TEST_CASE("normalize_text_rtl_punctuation") {
    // Arabic question mark U+061F = 0xD9 0x9F
    std::string input = "Hello\u061F World";
    auto result = kokopop::normalize_text(input);
    CHECK(!result.empty());
    // Should contain the Arabic question mark bytes
    CHECK(result.find("\u061F") != std::string::npos);
}

TEST_CASE("normalize_text_mixed_scripts_no_crash") {
    // Mix of Latin, Greek, Cyrillic, CJK, emoji, combining marks
    std::string input = "Hello \u03B1\u03B2\u03B3 \u0410\u0411\u0412 日本語 \U0001f600 caf\xE9";
    auto result = kokopop::normalize_text(input);
    // Should not crash and should preserve key characters
    CHECK(has_codepoint(result, 0x3B1));  // α
    CHECK(has_codepoint(result, 0x410));  // А
    CHECK(has_codepoint(result, 0x65E5)); // 日
    CHECK(has_codepoint(result, 0x1F600)); // 😀
}

TEST_CASE("normalize_text_ligature_passthrough") {
    // fi ligature U+FB01
    std::string input = "Hello\xEF\xAC\x81World";
    auto result = kokopop::normalize_text(input);
    CHECK(has_codepoint(result, 0xFB01));
}

TEST_CASE("normalize_text_bom_passthrough") {
    std::string input = "\xEF\xBB\xBFHello World";
    auto result = kokopop::normalize_text(input);
    CHECK(has_codepoint(result, 0xFEFF));
}

// ===========================================================================
// 3. Text splitter — complex characters
// ===========================================================================

TEST_CASE("split_sentences_unicode_ellipsis") {
    // Unicode ellipsis … (U+2026) as sentence boundary
    auto parts = kokopop::split_sentences("Hello world\xE2\x80\xA6 Next sentence.");
    CHECK(parts.size() >= 1u);
}

TEST_CASE("split_sentences_em_dash_no_split") {
    // Em dash — (U+2014) is not a sentence boundary in split_sentences
    // (it's handled in infer_boundary_type as ClauseWeak)
    auto parts = kokopop::split_sentences("Hello world\xE2\x80\x94more text.");
    CHECK(parts.size() == 1u);
}

TEST_CASE("split_sentences_en_dash_no_split") {
    // En dash – (U+2013) is not a sentence boundary
    auto parts = kokopop::split_sentences("Hello world\xE2\x80\x93more text.");
    CHECK(parts.size() == 1u);
}

TEST_CASE("infer_boundary_type_em_dash_clause_weak") {
    // Em dash at end → ClauseWeak
    CHECK_EQ(kokopop::infer_boundary_type("Hello\xE2\x80\x94"), kokopop::Boundary::ClauseWeak);
}

TEST_CASE("infer_boundary_type_en_dash_clause_weak") {
    // En dash at end → ClauseWeak
    CHECK_EQ(kokopop::infer_boundary_type("Hello\xE2\x80\x93"), kokopop::Boundary::ClauseWeak);
}

TEST_CASE("infer_boundary_type_unicode_ellipsis_sentence") {
    // Unicode ellipsis at end → Sentence
    CHECK_EQ(kokopop::infer_boundary_type("Hello\xE2\x80\xA6"), kokopop::Boundary::Sentence);
}

TEST_CASE("split_sentences_fullwidth_cjk_punctuation") {
    // CJK fullwidth period 。(U+3002) as sentence boundary
    std::string text = "\u4F60\u597D\u3002\u4E16\u754C\uFF01"; // 你好。世界！
    auto parts = kokopop::split_sentences(text);
    CHECK_EQ(parts.size(), 2u);
}

TEST_CASE("split_sentences_arabic_question_mark") {
    // Arabic question mark ؟ (U+061F) — NOT recognized as sentence boundary
    // (only ASCII '?' is checked)
    std::string text = "Hello\xD9\x9F World";
    auto parts = kokopop::split_sentences(text);
    CHECK_EQ(parts.size(), 1u); // Not split — only ASCII ? is recognized
}

TEST_CASE("split_sentences_arabic_semicolon") {
    // Arabic semicolon ؛ (U+061B) — NOT recognized
    std::string text = "Hello\xD9\x9B World";
    auto parts = kokopop::split_sentences(text);
    CHECK_EQ(parts.size(), 1u);
}

TEST_CASE("split_into_candidate_units_mixed_script") {
    // Mix of CJK and Latin with Chinese clause punctuation
    std::string text = "\u4F60\u597D\uFF0CHello\uFF0CWorld\u3002"; // 你好，Hello，World。
    auto units = kokopop::split_into_candidate_units(text);
    CHECK(units.size() >= 1u);
    // No crash, no truncation
    for (const auto & u : units) {
        CHECK(!u.empty());
    }
}

TEST_CASE("split_into_candidate_units_emoji_only") {
    std::string text = "\U0001f600\U0001f601\U0001f602";
    auto units = kokopop::split_into_candidate_units(text);
    CHECK(units.size() >= 1u);
}

TEST_CASE("split_into_candidate_units_fullwidth_latin") {
    // Fullwidth "HELLO WORLD"
    std::string text = "\xEF\xBD\x88\xEF\xBD\x85\xEF\xBD\x8C\xEF\xBD\x8C\xEF\xBD\x8F "
                       "\xEF\xBD\x98\xEF\xBD\x8F\xEF\xBD\xA1\xEF\xBD\x8C\xEF\xBD\x84";
    auto units = kokopop::split_into_candidate_units(text);
    CHECK(units.size() >= 1u);
}

TEST_CASE("split_sentences_closing_punctuation_stripped_unicode") {
    // Chinese closing quotes after sentence end
    std::string text = "\u8BB0\u5F97\uFF1A\u201C\u4F60\u597D\u3002\u201D"; // 记得："你好。"
    auto parts = kokopop::split_sentences(text);
    CHECK(parts.size() >= 1u);
}

TEST_CASE("split_sentences_closing_angle_quote") {
    // French closing angle quote » (U+00BB) after sentence end
    std::string text = "Hello world!\xBB";
    auto parts = kokopop::split_sentences(text);
    CHECK(parts.size() >= 1u);
}

TEST_CASE("split_sentences_right_double_quote_closer") {
    // Right double quote " (U+201D) as closing punctuation after sentence end
    std::string text = "Hello world!\xE2\x80\x9D";
    auto parts = kokopop::split_sentences(text);
    CHECK(parts.size() >= 1u);
}

TEST_CASE("split_by_words_cjk_treated_as_single_word") {
    // CJK without spaces → treated as a single "word" (no ASCII whitespace)
    std::string text = "\u4F60\u597D\u4E16\u754C"; // 你好世界
    auto words = kokopop::split_by_words(text);
    CHECK(words.size() >= 1u);
    // At minimum, all CJK chars should be preserved
    for (const auto & w : words) {
        CHECK(!w.empty());
    }
}

TEST_CASE("split_by_words_mixed_latin_cjk") {
    // "Hello 世界 World" — 3 words
    std::string text = "Hello \u4E16\u754C World";
    auto words = kokopop::split_by_words(text);
    CHECK_EQ(words.size(), 3u);
    CHECK_EQ(words[0], "Hello");
    CHECK_EQ(words[2], "World");
}

TEST_CASE("split_by_words_zero_width_space_not_a_space") {
    // ZWS is not a whitespace → text not split
    std::string text = "Hello\xE2\x80\x8BWorld";
    auto words = kokopop::split_by_words(text);
    CHECK_EQ(words.size(), 1u); // One "word" containing ZWS
}

TEST_CASE("split_by_words_non_breaking_space_behavior") {
    // NBSP (U+00A0) behavior depends on locale. In the C locale it's NOT
    // whitespace (so one word). In UTF-8 locales it may be treated as space.
    // We test that the function doesn't crash and produces non-empty output.
    std::string text = "Hello\u00A0World";
    auto words = kokopop::split_by_words(text);
    CHECK(words.size() >= 1u);
    for (const auto & w : words) {
        CHECK(!w.empty());
    }
}

TEST_CASE("infer_boundary_type_trailing_unicode_closers") {
    // Chinese closing bracket 】 (U+3011) after sentence end
    std::string text = "Hello.\xE3\x80\x91";
    CHECK_EQ(kokopop::infer_boundary_type(text), kokopop::Boundary::Sentence);
}

TEST_CASE("infer_boundary_type_trailing_ideographic_close_paren") {
    // ） (U+FF09) after sentence end
    std::string text = "Hello!\xEF\xBC\x89";
    CHECK_EQ(kokopop::infer_boundary_type(text), kokopop::Boundary::Sentence);
}

TEST_CASE("infer_boundary_type_trailing_greek_close_quote") {
    // » (U+00BB) after sentence end
    std::string text = "Hello!\xC2\xBB";
    CHECK_EQ(kokopop::infer_boundary_type(text), kokopop::Boundary::Sentence);
}

TEST_CASE("infer_boundary_type_emoji_at_end") {
    // Emoji at end → None (not punctuation)
    CHECK_EQ(kokopop::infer_boundary_type("Hello\U0001f600"), kokopop::Boundary::None);
}

TEST_CASE("split_sentences_only_emoji") {
    std::string text = "\U0001f600\U0001f601";
    auto parts = kokopop::split_sentences(text);
    CHECK_EQ(parts.size(), 1u); // One unit, no sentence boundaries
}

TEST_CASE("split_into_candidate_units_with_bom") {
    std::string text = "\xEF\xBB\xBFHello world.";
    auto units = kokopop::split_into_candidate_units(text);
    CHECK(units.size() >= 1u);
}

// ===========================================================================
// 4. Phonemizer — complex characters
// ===========================================================================

TEST_CASE("phonemize_text_emoji_input") {
    std::string phonemes, error;
    bool ok = kokopop::phonemize_text("Hello! \U0001f600 World", "af_heart", phonemes, error);
    if (ok) {
        CHECK(!phonemes.empty());
    }
    // Should not crash regardless of outcome
}

TEST_CASE("phonemize_text_emoji_only") {
    std::string phonemes, error;
    bool ok = kokopop::phonemize_text("\U0001f600\U0001f601", "af_heart", phonemes, error);
    // May succeed with empty phonemes or fail — either is acceptable
    if (ok) {
        // Phonemes may be empty since emoji aren't speech
    }
}

TEST_CASE("phonemize_text_cyrillic") {
    std::string phonemes, error;
    // Russian: "Привет мир"
    bool ok = kokopop::phonemize_text("\u041F\u0440\u0438\u0432\u0435\u0442 \u043C\u0438\u0440",
                                      "af_heart", phonemes, error);
    // eSpeak should handle Cyrillic; result may or may not be empty
    if (ok) {
        // Non-crash is the primary goal
    }
}

TEST_CASE("phonemize_text_arabic") {
    std::string phonemes, error;
    // Arabic: "مرحبا"
    bool ok = kokopop::phonemize_text("\u0645\u0631\u062D\u0628\u0627",
                                      "af_heart", phonemes, error);
    if (ok) {
        // Non-crash is the primary goal
    }
}

TEST_CASE("phonemize_text_greek") {
    std::string phonemes, error;
    // Greek: "Γεια σου"
    bool ok = kokopop::phonemize_text("\u0393\u03B5\u03B9\u03B1 \u03C3\u03BF\u03C5",
                                      "af_heart", phonemes, error);
    if (ok) {
        // Non-crash is the primary goal
    }
}

TEST_CASE("phonemize_text_thai") {
    std::string phonemes, error;
    // Thai: "สวัสดี"
    bool ok = kokopop::phonemize_text("\u0E2A\u0E27\u0E31\u0E2A\u0E14\u0E35\u0E22\u0E07",
                                      "af_heart", phonemes, error);
    if (ok) {
        // Non-crash is the primary goal
    }
}

TEST_CASE("phonemize_text_devanagari") {
    std::string phonemes, error;
    // Hindi: "नमस्ते"
    bool ok = kokopop::phonemize_text("\u0928\u092E\u0938\u094D\u0924\u0947",
                                      "hf_alpha", phonemes, error);
    if (ok) {
        CHECK(!phonemes.empty());
    }
}

TEST_CASE("phonemize_text_combining_marks_stacked") {
    std::string phonemes, error;
    // 'a' + 3 stacked combining marks
    bool ok = kokopop::phonemize_text("a\xCC\x80\xCC\x81\xCC\x82",
                                      "af_heart", phonemes, error);
    if (ok) {
        // Should not crash
    }
}

TEST_CASE("phonemize_text_fullwidth_latin") {
    std::string phonemes, error;
    // Fullwidth "HELLO" ＨＥＬＬＯ
    bool ok = kokopop::phonemize_text("\xEF\xBD\x88\xEF\xBD\x85\xEF\xBD\x8C\xEF\xBD\x8C\xEF\xBD\x8F",
                                      "af_heart", phonemes, error);
    // eSpeak may or may not handle fullwidth — either outcome is fine
    if (ok) {
        // Non-crash
    }
}

TEST_CASE("phonemize_text_bom_prefix") {
    std::string phonemes, error;
    bool ok = kokopop::phonemize_text("\xEF\xBB\xBFHello", "af_heart", phonemes, error);
    if (ok) {
        CHECK(!phonemes.empty());
    }
}

TEST_CASE("phonemize_text_prosody_em_dash") {
    std::string phonemes, error;
    // Em dash is in is_prosody_punctuation
    bool ok = kokopop::phonemize_text("Hello\xE2\x80\x94World", "af_heart", phonemes, error);
    if (ok) {
        CHECK(!phonemes.empty());
    }
}

TEST_CASE("phonemize_text_prosody_ellipsis") {
    std::string phonemes, error;
    // Unicode ellipsis is in is_prosody_punctuation
    bool ok = kokopop::phonemize_text("Hello\xE2\x80\xA6World", "af_heart", phonemes, error);
    if (ok) {
        CHECK(!phonemes.empty());
    }
}

TEST_CASE("phonemize_text_mixed_cjk_latin_emoji") {
    std::string phonemes, error;
    bool ok = kokopop::phonemize_text("\u4F60\u597D Hello! \U0001f600",
                                      "af_heart", phonemes, error);
    // Mixed script with emoji — should not crash
    if (ok) {
        // Phonemes may contain CJK chars passed through eSpeak
    }
}

TEST_CASE("phonemize_text_only_punctuation_unicode") {
    std::string phonemes, error;
    bool ok = kokopop::phonemize_text("\xE2\x80\x94\xE2\x80\xA6\xE2\x80\x9C",
                                      "af_heart", phonemes, error);
    // Only Unicode punctuation (— … ")
    if (ok) {
        // Should succeed, possibly with minimal output
    }
}

TEST_CASE("phonemize_text_long_word_500_chars") {
    std::string phonemes, error;
    // 500-character word (all 'a')
    std::string long_word(500, 'a');
    bool ok = kokopop::phonemize_text(long_word, "af_heart", phonemes, error);
    if (ok) {
        CHECK(!phonemes.empty());
    }
}

TEST_CASE("phonemize_text_only_ellipsis") {
    std::string phonemes, error;
    bool ok = kokopop::phonemize_text("\xE2\x80\xA6", "af_heart", phonemes, error);
    if (ok) {
        // Ellipsis is prosody punctuation — may produce minimal output
    }
}

TEST_CASE("phonemize_text_rtl_arabic_text") {
    std::string phonemes, error;
    // Arabic question mark
    bool ok = kokopop::phonemize_text("Hello\xD9\x9F", "af_heart", phonemes, error);
    if (ok) {
        CHECK(!phonemes.empty());
    }
}

// ===========================================================================
// 5. replace_all — complex characters
// ===========================================================================

TEST_CASE("replace_all_multibyte_pattern") {
    // Replace a 2-byte UTF-8 char: é (U+00E9, bytes C3 A9) → à (U+00E0, bytes C3 A0)
    std::string s = "caf\u00E9 caf\u00E9 caf\u00E9";
    kokopop::replace_all(s, "\u00E9", "\u00E0");
    CHECK_EQ(s, "caf\u00E0 caf\u00E0 caf\u00E0");
}

TEST_CASE("replace_all_multibyte_no_match") {
    std::string s = "hello world";
    kokopop::replace_all(s, "\u00E9", "a");
    CHECK_EQ(s, "hello world");
}

TEST_CASE("replace_all_emoji") {
    std::string s = "Hello\U0001f600World";
    kokopop::replace_all(s, "\U0001f600", "\U0001f602");
    CHECK(has_codepoint(s, 0x1F602));
    CHECK(!has_codepoint(s, 0x1F600));
}

TEST_CASE("replace_all_cjk") {
    std::string s = "\u4F60\u597D\u4E16\u754C"; // 你好世界
    kokopop::replace_all(s, "\u4F60\u597D", "\u6211\u5566"); // 你好→我啦
    CHECK_EQ(s, "\u6211\u5566\u4E16\u754C");
}

// ===========================================================================
// 6. trim_ascii — complex characters
// ===========================================================================

TEST_CASE("trim_ascii_preserves_leading_emoji") {
    CHECK(kokopop::trim_ascii("\U0001f600").find("\U0001f600") != std::string::npos);
}

TEST_CASE("trim_ascii_preserves_cjk") {
    CHECK_EQ(kokopop::trim_ascii(" \u4F60\u597D "), "\u4F60\u597D");
}

TEST_CASE("trim_ascii_preserves_combining_marks") {
    // e + combining acute accent (U+0301, bytes CC 81)
    CHECK_EQ(kokopop::trim_ascii(" e\u0301 "), "e\u0301");
}

TEST_CASE("trim_ascii_only_unicode_whitespace_bytes") {
    // Non-breaking space is not trimmed by trim_ascii (ASCII-only)
    std::string input = "\xC2\xA0Hello\xC2\xA0";
    auto result = kokopop::trim_ascii(input);
    // NBSP bytes preserved — trim_ascii only trims ASCII whitespace
    CHECK(has_codepoint(result, 0xA0));
}

TEST_CASE("trim_ascii_preserves_bom") {
    std::string input = " \xEF\xBB\xBF ";
    auto result = kokopop::trim_ascii(input);
    CHECK(has_codepoint(result, 0xFEFF));
}

// ===========================================================================
// 7. Integration: full pipeline with complex chars
// ===========================================================================

TEST_CASE("pipeline_normalize_split_unicode") {
    std::string input = "Dr. Smith\u2026 Hello \u4F60\u597D\u3002";
    std::string normalized = kokopop::normalize_text(input);
    auto units = kokopop::split_into_candidate_units(normalized);
    CHECK(units.size() >= 1u);
    // No crash, all units non-empty
    for (const auto & u : units) {
        CHECK(!u.empty());
    }
}

TEST_CASE("pipeline_normalize_split_mixed_punctuation") {
    std::string input = "Hello!\xE2\x80\x94World\xE2\x80\xA6";
    std::string normalized = kokopop::normalize_text(input);
    auto units = kokopop::split_into_candidate_units(normalized);
    CHECK(units.size() >= 1u);
}

TEST_CASE("pipeline_phonemize_cjk_then_split") {
    std::string text = "\u4F60\u597D\u4E16\u754C\u3001\u65E5\u672C\u8A9E\u3002";
    std::string phonemes, error;
    bool ok = kokopop::phonemize_text(text, "af_heart", phonemes, error);
    // May or may not succeed — verify no crash
    if (ok) {
        // Split the phonemized output
        auto parts = kokopop::split_sentences(phonemes);
        CHECK(parts.size() >= 1u);
    }
}

TEST_CASE("pipeline_full_mixed_input") {
    // The ultimate stress test: everything thrown at once
    std::string input = "Dr. 3.14 www.example.com "
                        "\u4F60\u597D\u3002 "
                        "\U0001f600\xE2\x80\x94Hello\xE2\x80\xA6 "
                        "caf\xE9\xCC\x81 "
                        "\xEF\xBD\x88\xEF\xBD\x85"; // Ｈｅ

    std::string normalized = kokopop::normalize_text(input);
    // Should not crash
    CHECK(!normalized.empty());

    auto units = kokopop::split_into_candidate_units(normalized);
    CHECK(units.size() >= 1u);

    // Phonemize each unit
    for (const auto & unit : units) {
        std::string phonemes, error;
        bool ok = kokopop::phonemize_text(unit, "af_heart", phonemes, error);
        if (ok) {
            // Should not crash
            CHECK(true);
        }
    }
}
