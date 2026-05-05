#include "text_normalizer.h"
#include "core/replace.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace kokopop {

using ::kokopop::replace_all;

/* ----------------------------------------------------------------------- */
/*  Individual helpers — kept for backward API compatibility               */
/* ----------------------------------------------------------------------- */

std::string normalize_line_endings(const std::string & text) {
    std::string out = text;
    replace_all(out, "\r\n", "\n");
    replace_all(out, "\r", "\n");
    return out;
}

std::string collapse_spaces(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    bool prev_space = false;
    for (char c : text) {
        if (c == ' ' || c == '\t') {
            if (!prev_space && !out.empty()) {
                out.push_back(' ');
            }
            prev_space = true;
        } else {
            out.push_back(c);
            prev_space = false;
        }
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

std::string protect_abbreviations(const std::string & text) {
    std::string out = text;

    static const char * common_abbrs[] = {
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
        "vs.", "etc.", "e.g.", "i.e.", "a.m.", "p.m.",
        "Fig.", "Eq.", "No.", "Vol.", "Ch.",
    };
    for (const char * abbr : common_abbrs) {
        std::string placeholder(abbr);
        replace_all(placeholder, ".", "<ABBR_DOT>");
        replace_all(out, abbr, placeholder);
    }

    for (size_t i = 0; i + 2 < out.size(); ++i) {
        if (std::isupper(static_cast<unsigned char>(out[i])) &&
            out[i + 1] == '.' &&
            std::isspace(static_cast<unsigned char>(out[i + 2]))) {
            out.replace(i + 1, 1, "<ABBR_DOT>");
            i += 10;
        }
    }

    return out;
}

std::string protect_decimals(const std::string & text) {
    std::string out = text;
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] == '.' && i > 0 && i + 1 < out.size()) {
            if (std::isdigit(static_cast<unsigned char>(out[i - 1])) &&
                std::isdigit(static_cast<unsigned char>(out[i + 1]))) {
                out.replace(i, 1, "<DECIMAL_DOT>");
                i += 13;
            }
        }
    }
    return out;
}

std::string protect_urls(const std::string & text) {
    std::string out = text;
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] == '.' && i > 0 && i + 1 < out.size()) {
            char prev = out[i - 1];
            char next = out[i + 1];
            if (std::isalnum(static_cast<unsigned char>(prev)) &&
                std::isalnum(static_cast<unsigned char>(next))) {
                size_t start = i;
                while (start > 0 && std::isalnum(static_cast<unsigned char>(out[start - 1])))
                    --start;
                size_t end = i + 1;
                while (end < out.size() &&
                       (std::isalnum(static_cast<unsigned char>(out[end])) || out[end] == '.'))
                    ++end;
                std::string candidate = out.substr(start, end - start);
                size_t dot_count = 0;
                for (char c : candidate) if (c == '.') ++dot_count;
                if (dot_count >= 1) {
                    std::string protected_candidate = candidate;
                    replace_all(protected_candidate, ".", "<URL_DOT>");
                    out.replace(start, candidate.size(), protected_candidate);
                    i = start + protected_candidate.size();
                }
            }
        }
    }
    return out;
}

std::string restore_protected(const std::string & text) {
    std::string out = text;
    replace_all(out, "<ABBR_DOT>", ".");
    replace_all(out, "<DECIMAL_DOT>", ".");
    replace_all(out, "<URL_DOT>", ".");
    return out;
}

/* ----------------------------------------------------------------------- */
/*  Single-pass normalize_text (optimizations 4.1 + 4.2)                   */
/* ----------------------------------------------------------------------- */

// Abbreviation table (letters only, no dots).
struct AbbrevEntry {
    const char * text;
    uint8_t len;
};

static constexpr AbbrevEntry ABBREV_TABLE[] = {
    {"Mr", 2},   {"Mrs", 3},  {"Ms", 2},   {"Dr", 2},   {"Prof", 4},
    {"Sr", 2},   {"Jr", 2},   {"St", 2},   {"vs", 2},   {"etc", 3},
    {"e.g", 3},  {"i.e", 3},  {"a.m", 3},  {"p.m", 3},  {"Fig", 3},
    {"Eq", 2},   {"No", 2},   {"Vol", 3},  {"Ch", 2},
};
static constexpr size_t ABBREV_COUNT = sizeof(ABBREV_TABLE) / sizeof(AbbrevEntry);

/// Check if position 'dot_pos' (which is a '.') is the final dot of a known
/// abbreviation.  Handles both single-dot ("Mr.") and multi-dot ("e.g.") forms.
/// Returns the abbreviation start position if matched, npos otherwise.
static size_t find_abbr(const std::string & s, size_t dot_pos) {
    if (dot_pos == 0) return std::string::npos;

    // Find the start of the current alphanumeric/dot span.
    // Scan back over alphanumerics and dots (for multi-dot abbreviations).
    size_t span_start = dot_pos;
    while (span_start > 0 &&
           (std::isalnum(static_cast<unsigned char>(s[span_start - 1])) ||
            s[span_start - 1] == '.')) {
        --span_start;
    }

    // Count dots within this span (before dot_pos).
    size_t dots_in_span = 0;
    for (size_t k = span_start; k < dot_pos; ++k) {
        if (s[k] == '.') ++dots_in_span;
    }

    // Must not be preceded by an alphanumeric (mid-word guard).
    if (span_start > 0 &&
        std::isalnum(static_cast<unsigned char>(s[span_start - 1]))) {
        return std::string::npos;
    }

    for (size_t idx = 0; idx < ABBREV_COUNT; ++idx) {
        const auto & entry = ABBREV_TABLE[idx];

        // Count dots inside this abbreviation (e.g. "e.g" → 1).
        size_t dots_in_abbr = 0;
        for (uint8_t k = 0; k < entry.len; ++k) {
            if (entry.text[k] == '.') ++dots_in_abbr;
        }

        // Dots in the current span must match dots in the abbreviation.
        if (dots_in_span != dots_in_abbr) continue;

        // Total span before the final dot = entry.len (which already includes internal dots).
        if ((dot_pos - span_start) < entry.len) continue;

        size_t abbr_start = dot_pos - entry.len;

        // Match: interleave abbreviation chars and dots.
        // entry.text contains letters and dots (e.g. "e.g").
        // s[abbr_start..dot_pos) contains the same pattern.
        bool match = true;
        size_t tp = abbr_start;
        uint8_t ek = 0;
        while (ek < entry.len) {
            if (tp >= dot_pos) { match = false; break; }
            if (entry.text[ek] == '.') {
                // Entry expects a dot here — text must also have a dot.
                if (s[tp] != '.') { match = false; break; }
                ++tp;
            } else {
                // Entry expects a letter here.
                if (s[tp] == '.') { ++tp; continue; }  // skip text dot
                if (s[tp] != entry.text[ek]) { match = false; break; }
                ++tp;
            }
            ++ek;
        }
        if (!match || tp != dot_pos) continue;

        return abbr_start;
    }
    return std::string::npos;
}

/// Is this dot a potential "intermediate" dot of any multi-dot abbreviation?
/// (i.e. it could be the 1st of 2 dots, but not yet the final dot.)
/// If so, we should NOT treat it as a URL dot.
/// All multi-dot abbreviations in our table are letter.letter patterns,
/// so we check both the preceding char AND the following char.
static bool could_be_multi_dot_abbr(const std::string & s, size_t dot_pos) {
    if (dot_pos == 0 || dot_pos + 1 >= s.size()) return false;

    char prev_c = s[dot_pos - 1];
    char next_c = s[dot_pos + 1];
    if (!std::isalpha(static_cast<unsigned char>(prev_c))) return false;
    if (!std::isalpha(static_cast<unsigned char>(next_c))) return false;

    for (size_t idx = 0; idx < ABBREV_COUNT; ++idx) {
        const auto & entry = ABBREV_TABLE[idx];
        if (entry.len < 3) continue;

        // Find the dot position in this entry.
        size_t dot_in_entry = entry.len;
        for (uint8_t k = 0; k < entry.len; ++k) {
            if (entry.text[k] == '.') {
                dot_in_entry = k;
                break;
            }
        }
        if (dot_in_entry == entry.len) continue; // no dot in entry

        // Check: char before dot matches, char after dot matches.
        if (entry.text[dot_in_entry - 1] == prev_c &&
            entry.text[dot_in_entry + 1] == next_c) {
            return true;
        }
    }
    return false;
}

/// Flush accumulated chars in text[buf_start, flush_end) to out.
/// If protect_dots is true, dots are replaced with dot_placeholder.
/// Word boundary tracking uses input positions.
static void flush_buf(
    const std::string & text,
    std::string & out,
    size_t buf_start, size_t flush_end,
    bool protect_dots,
    const char * dot_placeholder,
    size_t & in_word_start, size_t & in_word_end)
{
    for (size_t k = buf_start; k < flush_end; ++k) {
        unsigned char uc = static_cast<unsigned char>(text[k]);
        char c = text[k];

        if (protect_dots && c == '.') {
            out.append(dot_placeholder);
        } else {
            out.push_back(c);
        }

        // Update word boundary (input positions).
        if (std::isalnum(uc)) {
            if (in_word_start >= in_word_end) {
                in_word_start = k;
            }
            in_word_end = k + 1;
        } else {
            in_word_start = k + 1;
            in_word_end = k + 1;
        }
    }
}

std::string normalize_text(const std::string & text) {
    //
    // Single-pass normalization:
    //  1. Line ending normalization (CRLF → LF, CR → LF)
    //  2. Space collapsing (consecutive whitespace → single space)
    //  3. Dot protection (abbreviations → decimals → URLs)
    //
    // State machine approach:
    // - Characters are accumulated in a buffer [buf_start, i).
    // - Dots, spaces, and non-alphanumerics trigger a flush.
    // - URL protection: when a URL dot is detected, we flush up to it
    //   (with <URL_DOT>), then set protect_url_dot. Subsequent alphanumerics
    //   and dots are accumulated. A non-alnum, non-dot char triggers a flush
    //   with protect_dots=true, protecting all dots in that span.
    //

    std::string out;
    out.reserve(text.size() + 64);

    size_t i = 0;
    const size_t n = text.size();

    // Word boundary tracking (input positions).
    size_t in_word_start = 0;
    size_t in_word_end = 0;

    // URL protection state.
    bool protect_url_dot = false;

    // Accumulation buffer start (input position).
    size_t buf_start = 0;

    auto flush_to = [&](size_t flush_end, bool force_protect_dots = false) {
        bool do_protect = protect_url_dot || force_protect_dots;
        const char * placeholder = force_protect_dots ? "<ABBR_DOT>" : "<URL_DOT>";
        flush_buf(text, out, buf_start, flush_end,
                  do_protect, placeholder,
                  in_word_start, in_word_end);
        buf_start = flush_end;
    };

    while (i < n) {
        unsigned char uc = static_cast<unsigned char>(text[i]);
        char c = text[i];

        // --- Line ending normalization ---
        if (c == '\r') {
            flush_to(i);
            out.push_back('\n');
            if (i + 1 < n && text[i + 1] == '\n') {
                ++i;
            }
            ++i;
            buf_start = i;
            in_word_start = in_word_end = i;
            protect_url_dot = false;
            continue;
        }

        // --- Dot handling (priority: abbr → decimal → URL) ---
        if (c == '.') {
            // If we're in URL-protection mode, flush the dot with protection.
            if (protect_url_dot) {
                flush_to(i + 1);
                ++i;
                buf_start = i;
                continue;
            }

            bool handled = false;

            // 1. Known abbreviation (handles both single and multi-dot forms)?
            size_t abbr_start = find_abbr(text, i);
            if (abbr_start != std::string::npos) {
                // For multi-dot abbreviations, the buffer contains earlier dots
                // that also need protection as ABBR_DOT.
                bool multi_dot = false;
                for (size_t k = buf_start; k < i; ++k) {
                    if (text[k] == '.') { multi_dot = true; break; }
                }
                flush_to(i, multi_dot);
                out.append("<ABBR_DOT>");
                in_word_start = in_word_end = i + 1;
                ++i;
                buf_start = i;
                handled = true;
            }

            // 2. Decimal digit.digit?
            if (!handled && i > 0 && i + 1 < n &&
                std::isdigit(static_cast<unsigned char>(text[i - 1])) &&
                std::isdigit(static_cast<unsigned char>(text[i + 1]))) {
                flush_to(i);
                out.append("<DECIMAL_DOT>");
                in_word_start = in_word_end = i + 1;
                ++i;
                buf_start = i;
                handled = true;
            }

            // 3. URL dot (between alphanumerics, with preceding word)?
            //    BUT skip if this dot could be an intermediate dot of a
            //    multi-dot abbreviation (e.g. the first dot of "e.g.").
            if (!handled && in_word_start < in_word_end && i + 1 < n &&
                std::isalnum(static_cast<unsigned char>(text[i + 1])) &&
                !could_be_multi_dot_abbr(text, i)) {
                flush_to(i);
                out.append("<URL_DOT>");
                in_word_start = in_word_end = i + 1;
                protect_url_dot = true;
                ++i;
                buf_start = i;
                handled = true;
            }

            if (!handled) {
                // If this could be an intermediate dot of a multi-dot abbreviation
                // (e.g. the dot between 'e' and 'g' in "e.g."), accumulate it in
                // the buffer so the final dot can protect it with ABBR_DOT.
                if (could_be_multi_dot_abbr(text, i)) {
                    ++i;  // dot stays in the buffer [buf_start, i)
                } else {
                    flush_to(i);
                    out.push_back('.');
                    in_word_start = in_word_end = i + 1;
                    ++i;
                    buf_start = i;
                }
            }
            continue;
        }

        // --- Space / tab (collapse consecutive whitespace) ---
        if (c == ' ' || c == '\t') {
            flush_to(i);
            bool prev_space = (!out.empty() &&
                               (out.back() == ' ' || out.back() == '\t'));
            // Only add a space if we've seen actual content (not just whitespace).
            // prev_space check handles collapse; out.empty() prevents leading space.
            if (!prev_space && !out.empty()) {
                out.push_back(' ');
            }
            ++i;
            buf_start = i;
            in_word_start = in_word_end = i;
            protect_url_dot = false;
            continue;
        }

        // --- Other non-alphanumeric character ---
        if (!std::isalnum(uc)) {
            flush_to(i);
            out.push_back(c);
            ++i;
            buf_start = i;
            protect_url_dot = false;
            in_word_start = in_word_end = i;
            continue;
        }

        // --- Alphanumeric character ---
        if (in_word_start >= in_word_end) {
            in_word_start = i;
        }
        in_word_end = i + 1;
        ++i;
    }

    // Flush remaining buffer.
    flush_to(n);

    // Trim trailing spaces.
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }

    return out;
}

} // namespace kokopop
