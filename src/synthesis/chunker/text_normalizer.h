#pragma once

#include <string>

namespace kokopop {

// ---------------------------------------------------------------------------
// Text normalization — prepare text before splitting
//
// normalize_text() is the production path: a single pass that normalizes
// line endings (\r\n, \r → \n), collapses runs of spaces/tabs and protects
// dots inside abbreviations, decimals and URLs.
//
// The individual helpers below (normalize_line_endings, collapse_spaces,
// protect_*, restore_protected) are the original multi-pass pipeline. They
// are no longer called by production code; tests keep them as a reference
// oracle to differential-test normalize_text() against.
// ---------------------------------------------------------------------------

/// Full normalization pipeline (single pass)
std::string normalize_text(const std::string & text);

/// Reference-pipeline helpers (test oracle — see tests/test_text_normalizer.cpp)
std::string protect_abbreviations(const std::string & text);
std::string protect_decimals(const std::string & text);
std::string protect_urls(const std::string & text);
std::string restore_protected(const std::string & text);
std::string collapse_spaces(const std::string & text);
std::string normalize_line_endings(const std::string & text);

} // namespace kokopop
