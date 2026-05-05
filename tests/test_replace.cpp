#include "test_helpers.h"
#include "core/replace.h"

// ---- replace_all ----

TEST_CASE("replace_all_basic") {
    std::string s = "hello world";
    replace_all(s, "world", "earth");
    CHECK_EQ(s, "hello earth");
}

TEST_CASE("replace_all_multiple_occurrences") {
    std::string s = "aaa";
    replace_all(s, "a", "e");
    CHECK_EQ(s, "eee");
}

TEST_CASE("replace_all_no_match") {
    std::string s = "hello world";
    replace_all(s, "xyz", "abc");
    CHECK_EQ(s, "hello world");
}

TEST_CASE("replace_all_empty_from") {
    std::string s = "hello";
    replace_all(s, "", "x");
    CHECK_EQ(s, "hello"); // empty from is no-op
}

TEST_CASE("replace_all_empty_to") {
    std::string s = "hello world world";
    replace_all(s, " world", "");
    CHECK_EQ(s, "hello");
}

TEST_CASE("replace_all_overlapping") {
    std::string s = "aaa";
    replace_all(s, "aa", "b");
    // First match at pos 0 replaces "aa" → "b", pos advances by 1 → "ba"
    // No more "aa" found
    CHECK_EQ(s, "ba");
}

TEST_CASE("replace_all_same_string") {
    std::string s = "abc";
    replace_all(s, "abc", "abc");
    CHECK_EQ(s, "abc");
}
