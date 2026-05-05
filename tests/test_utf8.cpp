#include "test_helpers.h"

// ---- Utilitaires UTF-8 ----

TEST_CASE("utf8_next_single_byte") {
    std::string_view text = "hello";
    size_t offset = 0;
    std::string_view ch;

    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch, "h");
    CHECK_EQ(offset, 1u);

    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch, "e");
    CHECK_EQ(offset, 2u);

    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch, "l");
    CHECK_EQ(offset, 3u);

    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch, "l");
    CHECK_EQ(offset, 4u);

    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch, "o");
    CHECK_EQ(offset, 5u);

    // End of string
    CHECK(!kokopop::utf8_next(text, offset, ch));
}

TEST_CASE("utf8_next_two_byte") {
    // "café" — é = 2 bytes (0xC3 0xA9)
    std::string_view text = "caf\u00e9";
    size_t offset = 0;
    std::string_view ch;

    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch, "c");
    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch, "a");
    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch, "f");
    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch, "\u00e9"); // é
    CHECK_EQ(ch.size(), 2u); // 2 bytes for é
    CHECK(!kokopop::utf8_next(text, offset, ch));
}

TEST_CASE("utf8_next_three_byte") {
    // "日本語" — 3 bytes each
    std::string_view text = "\u65e5\u672c\u8a9e";
    size_t offset = 0;
    std::string_view ch;

    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch.size(), 3u);

    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch.size(), 3u);

    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch.size(), 3u);

    CHECK(!kokopop::utf8_next(text, offset, ch));
}

TEST_CASE("utf8_next_four_byte") {
    // "😀" — 4 bytes
    std::string_view text = "\U0001f600";
    size_t offset = 0;
    std::string_view ch;

    CHECK(kokopop::utf8_next(text, offset, ch));
    CHECK_EQ(ch.size(), 4u);
    CHECK(!kokopop::utf8_next(text, offset, ch));
}

TEST_CASE("utf8_next_empty_string") {
    std::string_view text = "";
    size_t offset = 0;
    std::string_view ch;
    CHECK(!kokopop::utf8_next(text, offset, ch));
}

TEST_CASE("utf8_next_truncated_two_byte") {
    // 2-byte sequence truncated: only 1 of 2 bytes
    std::string_view text = "\xC3"; // leading byte of é without continuation
    size_t offset = 0;
    std::string_view ch;
    CHECK(!kokopop::utf8_next(text, offset, ch));
}

TEST_CASE("utf8_next_truncated_three_byte") {
    // 3-byte sequence truncated: only 2 of 3 bytes
    std::string_view text = "\xE6\x5E"; // partial 日本語
    size_t offset = 0;
    std::string_view ch;
    CHECK(!kokopop::utf8_next(text, offset, ch));
}

TEST_CASE("utf8_next_contbyte_error") {
    // Continuation byte at start (invalid)
    std::string_view text = "\x80";
    size_t offset = 0;
    std::string_view ch;
    CHECK(!kokopop::utf8_next(text, offset, ch));
}

TEST_CASE("utf8_next_invalid_leading_byte") {
    // 0xC2 without valid continuation
    std::string_view text = "\xC2\x00";
    size_t offset = 0;
    std::string_view ch;
    CHECK(!kokopop::utf8_next(text, offset, ch));
}

TEST_CASE("utf8_chars_ascii") {
    auto chars = kokopop::utf8_chars("abc");
    CHECK_EQ(chars.size(), 3u);
    CHECK_EQ(chars[0], "a");
    CHECK_EQ(chars[1], "b");
    CHECK_EQ(chars[2], "c");
}

TEST_CASE("utf8_chars_mixed") {
    auto chars = kokopop::utf8_chars("caf\u00e9");
    CHECK_EQ(chars.size(), 4u);
    CHECK_EQ(chars[0], "c");
    CHECK_EQ(chars[1], "a");
    CHECK_EQ(chars[2], "f");
    CHECK_EQ(chars[3], "\u00e9");
}

TEST_CASE("utf8_chars_empty") {
    auto chars = kokopop::utf8_chars("");
    CHECK(chars.empty());
}

TEST_CASE("utf8_chars_truncated") {
    auto chars = kokopop::utf8_chars("\xC3");
    CHECK(chars.empty());
}

TEST_CASE("utf8_chars_emoji") {
    auto chars = kokopop::utf8_chars("\U0001f600\U0001f389");
    CHECK_EQ(chars.size(), 2u);
    CHECK_EQ(chars[0], "\U0001f600");
    CHECK_EQ(chars[1], "\U0001f389");
}

TEST_CASE("trim_ascii_no_trim") {
    CHECK_EQ(kokopop::trim_ascii("hello"), "hello");
}

TEST_CASE("trim_ascii_leading_spaces") {
    CHECK_EQ(kokopop::trim_ascii("  hello"), "hello");
}

TEST_CASE("trim_ascii_trailing_spaces") {
    CHECK_EQ(kokopop::trim_ascii("hello  "), "hello");
}

TEST_CASE("trim_ascii_both_sides") {
    CHECK_EQ(kokopop::trim_ascii("  hello  "), "hello");
}

TEST_CASE("trim_ascii_only_spaces") {
    CHECK_EQ(kokopop::trim_ascii("   "), "");
}

TEST_CASE("trim_ascii_tabs_newlines") {
    CHECK_EQ(kokopop::trim_ascii("\t\n hello \n\t"), "hello");
}

TEST_CASE("trim_ascii_empty") {
    CHECK_EQ(kokopop::trim_ascii(""), "");
}

TEST_CASE("trim_ascii_preserves_unicode") {
    CHECK_EQ(kokopop::trim_ascii("  caf\u00e9  "), "caf\u00e9");
}
