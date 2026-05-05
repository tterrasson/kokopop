#pragma once

#include <string>

namespace kokopop {

// ---------------------------------------------------------------------------
// Text normalization — prepare text before splitting
//
// 1. Normalize unicode (NFC)
// 2. Replace \r\n → \n
// 3. Collapse spaces (except around newlines)
// 4. Normalize quotes to standard forms
// 5. Normalize ellipsis to …
// 6. Protect abbreviations, decimals, URLs (replace dots with placeholders)
//
// The protection tokens (<ABBR_DOT>, etc.) are restored after splitting.
// ---------------------------------------------------------------------------

/// Full normalization pipeline
std::string normalize_text(const std::string & text);

/// Protect special patterns so they are not split on dots
std::string protect_abbreviations(const std::string & text);
std::string protect_decimals(const std::string & text);
std::string protect_urls(const std::string & text);

/// Restore protected tokens to their original form
std::string restore_protected(const std::string & text);

/// Collapse consecutive spaces, but preserve newlines
std::string collapse_spaces(const std::string & text);

/// Normalize CRLF and standalone CR to LF
std::string normalize_line_endings(const std::string & text);

} // namespace kokopop
