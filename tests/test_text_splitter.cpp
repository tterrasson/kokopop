// test_text_splitter.cpp — tests pour les utilitaires de découpe de texte.
// Converti du format standalone (assert/main) vers doctest.

#include "synthesis/chunker/text_splitter.h"

#include <string>
#include <vector>

using namespace kokopop;

// ---- split_keep_delimiter ----

TEST_CASE("split_keep_delimiter basic") {
    auto parts = split_keep_delimiter("a;b;c", {";"});
    CHECK_EQ(parts.size(), 3u);
    CHECK_EQ(parts[0], "a;");
    CHECK_EQ(parts[1], "b;");
    CHECK_EQ(parts[2], "c");
}

TEST_CASE("split_keep_delimiter multi-char delimiter") {
    auto parts = split_keep_delimiter("para1\n\npara2\n\npara3", {"\n\n"});
    CHECK_EQ(parts.size(), 3u);
    CHECK_EQ(parts[0], "para1\n\n");
    CHECK_EQ(parts[1], "para2\n\n");
    CHECK_EQ(parts[2], "para3");
}

TEST_CASE("split_keep_delimiter first-match priority") {
    // longer delimiter listed first → wins
    {
        auto parts = split_keep_delimiter("a; b; c", {"; ", ";"});
        CHECK_EQ(parts.size(), 3u);
        CHECK_EQ(parts[0], "a; ");
        CHECK_EQ(parts[1], "b; ");
        CHECK_EQ(parts[2], "c");
    }
    // shorter delimiter listed first → wins, space stays in next part
    {
        auto parts = split_keep_delimiter("a; b; c", {";", "; "});
        CHECK_EQ(parts.size(), 3u);
        CHECK_EQ(parts[0], "a;");
        CHECK_EQ(parts[1], " b;");
        CHECK_EQ(parts[2], " c");
    }
}

TEST_CASE("split_keep_delimiter empty input") {
    auto parts = split_keep_delimiter("", {";"});
    CHECK_EQ(parts.size(), 0u);
}

TEST_CASE("split_keep_delimiter no match") {
    auto parts = split_keep_delimiter("hello world", {";"});
    CHECK_EQ(parts.size(), 1u);
    CHECK_EQ(parts[0], "hello world");
}

TEST_CASE("split_keep_delimiter delimiter at start") {
    auto parts = split_keep_delimiter(";a", {";"});
    CHECK_EQ(parts.size(), 2u);
    CHECK_EQ(parts[0], ";");
    CHECK_EQ(parts[1], "a");
}

TEST_CASE("split_keep_delimiter delimiter at end") {
    auto parts = split_keep_delimiter("a;", {";"});
    CHECK_EQ(parts.size(), 1u);
    CHECK_EQ(parts[0], "a;");
}

// ---- infer_boundary_type ----

TEST_CASE("infer_boundary_type sentence endings") {
    CHECK_EQ(infer_boundary_type("Hello world!"), Boundary::Sentence);
    CHECK_EQ(infer_boundary_type("Hello world?"), Boundary::Sentence);
    CHECK_EQ(infer_boundary_type("Hello world."), Boundary::Sentence);
}

TEST_CASE("infer_boundary_type clause endings") {
    CHECK_EQ(infer_boundary_type("Hello world;"), Boundary::ClauseStrong);
    CHECK_EQ(infer_boundary_type("Hello world,"), Boundary::ClauseWeak);
    CHECK_EQ(infer_boundary_type("Hello world:"), Boundary::ClauseStrong);
}

TEST_CASE("infer_boundary_type paragraph and newline") {
    CHECK_EQ(infer_boundary_type("Hello\n\n"), Boundary::Paragraph);
    CHECK_EQ(infer_boundary_type("Hello\n"),  Boundary::Newline);
}

TEST_CASE("infer_boundary_type none and edge cases") {
    CHECK_EQ(infer_boundary_type("Hello world"),    Boundary::None);
    CHECK_EQ(infer_boundary_type(""),               Boundary::None);
    CHECK_EQ(infer_boundary_type("   "),            Boundary::None);
    CHECK_EQ(infer_boundary_type("Hello world!   "), Boundary::Sentence);
    CHECK_EQ(infer_boundary_type("Hello world!\""), Boundary::Sentence);
    CHECK_EQ(infer_boundary_type("Hello world!)"),  Boundary::Sentence);
}

// ---- split_sentences ----

TEST_CASE("split_sentences basic period and exclamation") {
    {
        auto parts = split_sentences("Hello world. Goodbye world.");
        CHECK_EQ(parts.size(), 2u);
        CHECK_EQ(parts[0], "Hello world. ");
        CHECK_EQ(parts[1], "Goodbye world.");
    }
    {
        auto parts = split_sentences("Hello! How are you?");
        CHECK_EQ(parts.size(), 2u);
        CHECK_EQ(parts[0], "Hello! ");
        CHECK_EQ(parts[1], "How are you?");
    }
}

TEST_CASE("split_sentences protected abbreviation placeholder") {
    auto parts = split_sentences("M<ABBR_DOT>r Smith went home.");
    CHECK_EQ(parts.size(), 1u);
    CHECK_EQ(parts[0], "M<ABBR_DOT>r Smith went home.");
}

TEST_CASE("split_sentences protected single uppercase letter dot") {
    auto parts = split_sentences("A. B. C.");
    CHECK_EQ(parts.size(), 1u);
}

TEST_CASE("split_sentences protected decimal") {
    auto parts = split_sentences("The value is 3.14.");
    CHECK_EQ(parts.size(), 1u);
    CHECK_EQ(parts[0], "The value is 3.14.");
}

TEST_CASE("split_sentences lowercase abbreviation not protected") {
    // 'r' is lowercase → not caught by fallback → splits at "Mr."
    auto parts = split_sentences("Mr. Smith.");
    CHECK_EQ(parts.size(), 2u);
}

TEST_CASE("split_sentences multiple types") {
    auto parts = split_sentences("Hello! World. How?");
    CHECK_EQ(parts.size(), 3u);
    CHECK_EQ(parts[0], "Hello! ");
    CHECK_EQ(parts[1], "World. ");
    CHECK_EQ(parts[2], "How?");
}

TEST_CASE("split_sentences empty") {
    auto parts = split_sentences("");
    CHECK_EQ(parts.size(), 0u);
}

TEST_CASE("split_sentences closing quote after sentence end") {
    auto parts = split_sentences("Hello world!\" Next.");
    CHECK_EQ(parts.size(), 2u);
    CHECK_EQ(parts[0], "Hello world!\" ");
    CHECK_EQ(parts[1], "Next.");
}

// ---- split_into_candidate_units ----

TEST_CASE("split_into_candidate_units basic") {
    auto units = split_into_candidate_units("Hello world. Goodbye world.\n\nNew para. End.");
    CHECK_EQ(units.size(), 4u);
    CHECK_EQ(units[0], "Hello world.");
    CHECK_EQ(units[1], "Goodbye world.");
    CHECK_EQ(units[2], "New para.");
    CHECK_EQ(units[3], "End.");
}

TEST_CASE("split_into_candidate_units decimal dot protected") {
    auto units = split_into_candidate_units("Value is 3.14. End.");
    CHECK_EQ(units.size(), 2u);
    CHECK_EQ(units[0], "Value is 3.14.");
    CHECK_EQ(units[1], "End.");
}

TEST_CASE("split_into_candidate_units pure decimal no sentence end") {
    auto units = split_into_candidate_units("The ratio is 3.14");
    CHECK_EQ(units.size(), 1u);
    CHECK_EQ(units[0], "The ratio is 3.14");
}

TEST_CASE("split_into_candidate_units placeholder dots") {
    auto units = split_into_candidate_units("D<ABBR_DOT>r Evans said hello. Fine.");
    CHECK_EQ(units.size(), 2u);
    CHECK_EQ(units[0], "D<ABBR_DOT>r Evans said hello.");
    CHECK_EQ(units[1], "Fine.");
}

// ---- force_split_unit ----

TEST_CASE("force_split_unit strong delimiters") {
    {
        auto parts = force_split_unit("First; second: third");
        CHECK_EQ(parts.size(), 3u);
        CHECK_EQ(parts[0], "First; ");
        CHECK_EQ(parts[1], "second: ");
        CHECK_EQ(parts[2], "third");
    }
    {
        auto parts = force_split_unit("First; second");
        CHECK_EQ(parts.size(), 2u);
        CHECK_EQ(parts[0], "First; ");
        CHECK_EQ(parts[1], "second");
    }
}

TEST_CASE("force_split_unit comma delimiter") {
    auto parts = force_split_unit("a, b, c");
    CHECK_EQ(parts.size(), 3u);
    CHECK_EQ(parts[0], "a, ");
    CHECK_EQ(parts[1], "b, ");
    CHECK_EQ(parts[2], "c");
}

TEST_CASE("force_split_unit word split fallback") {
    auto parts = force_split_unit("hello world foo");
    CHECK_EQ(parts.size(), 3u);
    CHECK_EQ(parts[0], "hello");
    CHECK_EQ(parts[1], "world");
    CHECK_EQ(parts[2], "foo");
}

// ---- Public API helpers ----

TEST_CASE("text_splitter is_sentence_boundary") {
    CHECK(is_sentence_boundary("Hello.", 5)         == true);
    CHECK(is_sentence_boundary("Hello.", 4)         == false);
    CHECK(is_sentence_boundary("Hello!", 5)         == true);
    CHECK(is_sentence_boundary("M<ABBR_DOT>r", 2)  == false);
    CHECK(is_sentence_boundary("3.14", 1)           == false);
}

TEST_CASE("text_splitter is_protected_dot") {
    CHECK(is_protected_dot("3.14", 1)    == true);
    CHECK(is_protected_dot("A. B", 1)    == true);
    CHECK(is_protected_dot("Hello.", 5)  == false);
}

TEST_CASE("text_splitter boundary classification") {
    CHECK(is_strong_boundary(Boundary::Sentence)         == true);
    CHECK(is_strong_boundary(Boundary::ClauseStrong)     == false);
    CHECK(is_reasonable_boundary(Boundary::ClauseStrong) == true);
    CHECK(is_reasonable_boundary(Boundary::ClauseWeak)   == false);
}
