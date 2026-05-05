// Test: text_normalizer.cpp — single-pass normalize_text correctness
//
// Strategy: compare the optimized single-pass normalize_text against the
// original multi-pass pipeline (normalize_line_endings → collapse_spaces →
// protect_abbreviations → protect_decimals → protect_urls) for every test case.

#include "synthesis/chunker/text_normalizer.h"

#include <string>

// Macro: assert that the single-pass result matches the original multi-pass result.
#define CHECK_NORMALIZE_EQ(input)                                                  \
    do {                                                                           \
        std::string inp = (input);                                                 \
        std::string single_pass = kokopop::normalize_text(inp);                       \
        std::string multi_pass =                                                   \
            kokopop::protect_urls(                                                    \
                kokopop::protect_decimals(                                            \
                    kokopop::protect_abbreviations(                                   \
                        kokopop::collapse_spaces(                                     \
                            kokopop::normalize_line_endings(inp)))))                  \
        ;                                                                          \
        CHECK(single_pass == multi_pass);                                          \
    } while (0)

TEST_CASE("text_normalizer: empty and whitespace") {
    CHECK_NORMALIZE_EQ("");
    CHECK_NORMALIZE_EQ("   ");
    CHECK_NORMALIZE_EQ("\t\t");
    CHECK_NORMALIZE_EQ(" \t ");
    CHECK_NORMALIZE_EQ("hello   world");
    CHECK_NORMALIZE_EQ("  hello world  ");
}

TEST_CASE("text_normalizer: line endings") {
    CHECK_NORMALIZE_EQ("hello\r\nworld");
    CHECK_NORMALIZE_EQ("hello\rworld");
    CHECK_NORMALIZE_EQ("hello\nworld");
    CHECK_NORMALIZE_EQ("\r\n\r\n");
    CHECK_NORMALIZE_EQ("a\r\nb\r\nc");
}

TEST_CASE("text_normalizer: abbreviations") {
    CHECK_NORMALIZE_EQ("Mr. Smith went there");
    CHECK_NORMALIZE_EQ("Mrs. and Mr. Jones");
    CHECK_NORMALIZE_EQ("Dr. Evans");
    CHECK_NORMALIZE_EQ("Prof. Williams");
    CHECK_NORMALIZE_EQ("e.g. 3.14");
    CHECK_NORMALIZE_EQ("i.e. the answer");
    CHECK_NORMALIZE_EQ("a.m. and p.m.");
    CHECK_NORMALIZE_EQ("vs. the other team");
    CHECK_NORMALIZE_EQ("etc. you know");
    CHECK_NORMALIZE_EQ("Fig. 1 shows");
    CHECK_NORMALIZE_EQ("St. Louis");
    CHECK_NORMALIZE_EQ("No. 5");
    CHECK_NORMALIZE_EQ("Vol. 3");
    CHECK_NORMALIZE_EQ("Ch. 12");
    CHECK_NORMALIZE_EQ("Sr. and Jr.");
    CHECK_NORMALIZE_EQ("Eq. 2.5");
    CHECK_NORMALIZE_EQ("Ms. Johnson");
}

TEST_CASE("text_normalizer: abbreviations at various positions") {
    CHECK_NORMALIZE_EQ("Mr.");
    CHECK_NORMALIZE_EQ("Hello Mr.");
    CHECK_NORMALIZE_EQ("Dr.");
    CHECK_NORMALIZE_EQ("See e.g.");
}

TEST_CASE("text_normalizer: decimals") {
    CHECK_NORMALIZE_EQ("3.14");
    CHECK_NORMALIZE_EQ("The value is 2.718");
    CHECK_NORMALIZE_EQ("1.0 and 2.0");
    CHECK_NORMALIZE_EQ("0.5");
    CHECK_NORMALIZE_EQ("100.200.300"); // multiple decimals
}

TEST_CASE("text_normalizer: URLs") {
    CHECK_NORMALIZE_EQ("www.example.com");
    CHECK_NORMALIZE_EQ("See www.example.com for more");
    CHECK_NORMALIZE_EQ("a.b.c");
    CHECK_NORMALIZE_EQ("http.example.org/path");
    CHECK_NORMALIZE_EQ("example.com");
    CHECK_NORMALIZE_EQ("sub.domain.example.com");
}

TEST_CASE("text_normalizer: abbreviations + URLs combined") {
    CHECK_NORMALIZE_EQ("Mr. example.com");
    CHECK_NORMALIZE_EQ("Dr. www.site.org");
    CHECK_NORMALIZE_EQ("Mr. and Mrs. example.com");
}

TEST_CASE("text_normalizer: abbreviations + decimals combined") {
    CHECK_NORMALIZE_EQ("Dr. 3.14");
    CHECK_NORMALIZE_EQ("e.g. 2.5 and 3.7");
    CHECK_NORMALIZE_EQ("Mr. said 1.5");
}

TEST_CASE("text_normalizer: mixed complex input") {
    CHECK_NORMALIZE_EQ("Dr. Smith's e-mail is dr.smith@example.com, see e.g. Fig. 1, p. 3.14");
    CHECK_NORMALIZE_EQ("Mr. and Mrs. Jones, Vol. 2, Ch. 5, Eq. 3.14");
    CHECK_NORMALIZE_EQ("See St. Louis, www.stlouis.com for info");
}

TEST_CASE("text_normalizer: sentence-ending dots") {
    CHECK_NORMALIZE_EQ("Hello world.");
    CHECK_NORMALIZE_EQ("First sentence. Second sentence.");
    CHECK_NORMALIZE_EQ("What? Really! OK.");
}

TEST_CASE("text_normalizer: dots that should NOT be protected") {
    CHECK_NORMALIZE_EQ("Hello.");
    CHECK_NORMALIZE_EQ("End of file.");
    CHECK_NORMALIZE_EQ("Wait... what?");
}

TEST_CASE("text_normalizer: edge cases") {
    CHECK_NORMALIZE_EQ(".");
    CHECK_NORMALIZE_EQ("..");
    CHECK_NORMALIZE_EQ("...");
    CHECK_NORMALIZE_EQ("a.");
    CHECK_NORMALIZE_EQ(".a");
    CHECK_NORMALIZE_EQ("a.b");
    CHECK_NORMALIZE_EQ("1.2.3.4.5");
    CHECK_NORMALIZE_EQ("a. b.");
    CHECK_NORMALIZE_EQ("a. b. c.");
}

TEST_CASE("text_normalizer: restore_protected roundtrip") {
    auto roundtrip = [](const std::string & input) {
        std::string normalized = kokopop::normalize_text(input);
        std::string restored = kokopop::restore_protected(normalized);
        // After restore, all placeholders are back to dots.
        // The restored text should match the space-collapsed, line-ending-normalized input.
        std::string expected = kokopop::collapse_spaces(
            kokopop::normalize_line_endings(input));
        CHECK(restored == expected);
    };

    roundtrip("Mr. Smith went there");
    roundtrip("See www.example.com");
    roundtrip("The value is 3.14");
    roundtrip("Dr. said e.g. 2.5");
    roundtrip("Hello world.");
    roundtrip("Mr.\r\nSmith");
    roundtrip("  a.b.c  ");
}

TEST_CASE("text_normalizer: abbreviation not mid-word") {
    // "xMr." should NOT match "Mr" abbreviation (mid-word).
    CHECK_NORMALIZE_EQ("xMr. Smith");
    // "aMr." should NOT match.
    CHECK_NORMALIZE_EQ("aMr.");
}

TEST_CASE("text_normalizer: URL at end of string") {
    CHECK_NORMALIZE_EQ("See www.example.com");
    CHECK_NORMALIZE_EQ("Visit a.b");
}

TEST_CASE("text_normalizer: URL followed by punctuation") {
    CHECK_NORMALIZE_EQ("See www.example.com.");
    CHECK_NORMALIZE_EQ("Visit a.b!");
}
