#include "synthesis/chunker/text_splitter.h"
#include "core/utf8.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace kokopop {
namespace {

static bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool starts_with(const std::string & s, const std::string & prefix) {
    return s.size() >= prefix.size() &&
           s.compare(0, prefix.size(), prefix) == 0;
}

/* ---- UTF-8 multi-byte character helpers (shared everywhere) ---- */

static bool char_is_ellipsis(const std::string & s, size_t pos) {
    return pos + 2 < s.size() &&
           static_cast<unsigned char>(s[pos]) == 0xE2 &&
           static_cast<unsigned char>(s[pos+1]) == 0x80 &&
           static_cast<unsigned char>(s[pos+2]) == 0xA6;
}

static bool char_is_em_dash(const std::string & s, size_t pos) {
    return pos + 2 < s.size() &&
           static_cast<unsigned char>(s[pos]) == 0xE2 &&
           static_cast<unsigned char>(s[pos+1]) == 0x80 &&
           static_cast<unsigned char>(s[pos+2]) == 0x94;
}

static bool char_is_en_dash(const std::string & s, size_t pos) {
    return pos + 2 < s.size() &&
           static_cast<unsigned char>(s[pos]) == 0xE2 &&
           static_cast<unsigned char>(s[pos+1]) == 0x80 &&
           static_cast<unsigned char>(s[pos+2]) == 0x93;
}

static bool matches_at(const std::string & s, size_t pos, const char * utf8) {
    const size_t n = std::strlen(utf8);
    return pos + n <= s.size() && s.compare(pos, n, utf8) == 0;
}

static size_t utf8_char_start_at_or_before_end(const std::string & s) {
    if (s.empty()) return 0;
    size_t pos = s.size() - 1;
    while (pos > 0 &&
           (static_cast<unsigned char>(s[pos]) & 0xC0) == 0x80) {
        --pos;
    }
    return pos;
}

static bool char_is_zh_sentence_punct(const std::string & s, size_t pos) {
    return matches_at(s, pos, "。") ||
           matches_at(s, pos, "！") ||
           matches_at(s, pos, "？");
}

static bool char_is_zh_clause_strong(const std::string & s, size_t pos) {
    return matches_at(s, pos, "；") ||
           matches_at(s, pos, "：");
}

static bool char_is_zh_clause_weak(const std::string & s, size_t pos) {
    return matches_at(s, pos, "，") ||
           matches_at(s, pos, "、");
}

static bool char_is_closing_punctuation(const std::string & s, size_t pos, size_t & len) {
    const char c = s[pos];
    if (c == '"' || c == ')' || c == '\'') {
        len = 1;
        return true;
    }
    const char * closers[] = {
        "”", "’", "）", "】", "》", "〉", "」", "』", "»",
    };
    for (const char * closer : closers) {
        const size_t n = std::strlen(closer);
        if (pos + n <= s.size() && s.compare(pos, n, closer) == 0) {
            len = n;
            return true;
        }
    }
    return false;
}

/* ---- 3.2: View-based trim (zero allocation) ---- */

static std::string_view trim_view(std::string_view sv) {
    size_t first = 0;
    while (first < sv.size() && std::isspace(static_cast<unsigned char>(sv[first]))) ++first;
    size_t last = sv.size();
    while (last > first && std::isspace(static_cast<unsigned char>(sv[last - 1]))) --last;
    return sv.substr(first, last - first);
}

/* ---- 3.2: infer_boundary_type — string_view, no string copies (static impl) ---- */

static Boundary infer_boundary_type_impl(const std::string & text) {
    std::string_view sv(text);

    // Trim — zero allocation
    std::string_view trimmed = trim_view(sv);
    if (trimmed.empty()) return Boundary::None;

    // Paragraph / newline boundaries (must check original text for \n\n)
    if (ends_with(text, "\n\n")) return Boundary::Paragraph;
    if (ends_with(text, "\n")) return Boundary::Newline;

    // Walk backwards from end of trimmed, stripping closing punctuation
    while (!trimmed.empty()) {
        std::string tmp(trimmed);
        size_t closer_len = 0;
        bool found_closer = false;
        for (size_t back = 1; back <= 4 && back <= tmp.size(); ++back) {
            const size_t pos = tmp.size() - back;
            if (char_is_closing_punctuation(tmp, pos, closer_len) &&
                pos + closer_len == tmp.size()) {
                found_closer = true;
                break;
            }
        }
        if (found_closer) {
            trimmed.remove_suffix(closer_len);
            trimmed = trim_view(trimmed);
            if (trimmed.empty()) return Boundary::None;
        } else {
            break;
        }
    }
    if (trimmed.empty()) return Boundary::None;

    // Build a temporary std::string only for UTF-8 multi-byte helpers
    std::string str(trimmed);
    size_t last_pos = utf8_char_start_at_or_before_end(str);

    if (str[last_pos] == '.' || str[last_pos] == '!' ||
        str[last_pos] == '?' || char_is_ellipsis(str, last_pos) ||
        char_is_zh_sentence_punct(str, last_pos)) {
        return Boundary::Sentence;
    }
    if (str[last_pos] == ';' || str[last_pos] == ':' ||
        char_is_zh_clause_strong(str, last_pos)) {
        return Boundary::ClauseStrong;
    }
    if (str[last_pos] == ',' ||
        char_is_em_dash(str, last_pos) ||
        char_is_en_dash(str, last_pos) ||
        char_is_zh_clause_weak(str, last_pos)) {
        return Boundary::ClauseWeak;
    }
    return Boundary::None;
}

/* ---- 3.3: is_protected_token (public API uses this; internal uses pre-compute) ---- */

static bool is_protected_token(const std::string & text, size_t dot_pos) {
    // Placeholder tags: <ABBR_DOT> etc. (12 chars from dot position)
    if (dot_pos + 12 <= text.size()) {
        std::string sub = text.substr(dot_pos, 12);
        if (starts_with(sub, "<ABBR_DOT>") ||
            starts_with(sub, "<DECIMAL_DOT>") ||
            starts_with(sub, "<URL_DOT>")) {
            return true;
        }
    }
    // Decimal number (digit.digit)
    if (dot_pos > 0 && dot_pos + 1 < text.size()) {
        char prev = text[dot_pos - 1];
        char next = text[dot_pos + 1];
        if (std::isdigit(static_cast<unsigned char>(prev)) &&
            std::isdigit(static_cast<unsigned char>(next))) {
            return true;
        }
    }
    // Abbreviation pattern (uppercase dot whitespace)
    if (dot_pos > 0 && dot_pos + 1 < text.size()) {
        char prev = text[dot_pos - 1];
        char next = text[dot_pos + 1];
        if (std::isupper(static_cast<unsigned char>(prev)) &&
            std::isspace(static_cast<unsigned char>(next))) {
            return true;
        }
    }
    return false;
}

/* ---- 3.3: Pre-compute protected dot positions in one pass (O(n)) ---- */

static std::vector<bool> precompute_protected_positions(const std::string & text) {
    std::vector<bool> protected_pos(text.size(), false);

    // 1. Scan for placeholder tags (<ABBR_DOT>, <DECIMAL_DOT>, <URL_DOT>)
    //    These are 12-char markers; check char[1..4] after the dot
    size_t pos = 0;
    while ((pos = text.find('.', pos)) != std::string::npos) {
        if (pos + 12 <= text.size()) {
            const char * p = text.data() + pos;
            if ((p[1] == 'A' && p[2] == 'B' && p[3] == 'B' && p[4] == 'R') ||
                (p[1] == 'D' && p[2] == 'E' && p[3] == 'C') ||
                (p[1] == 'U' && p[2] == 'R' && p[3] == 'L')) {
                protected_pos[pos] = true;
            }
        }
        ++pos;
    }

    // 2. Scan for digit.digit and uppercase-dot-whitespace patterns
    for (size_t i = 1; i + 1 < text.size(); ++i) {
        if (text[i] == '.' && !protected_pos[i]) {
            char prev = text[i - 1];
            char next = text[i + 1];
            if ((std::isdigit(static_cast<unsigned char>(prev)) &&
                 std::isdigit(static_cast<unsigned char>(next))) ||
                (std::isupper(static_cast<unsigned char>(prev)) &&
                 std::isspace(static_cast<unsigned char>(next)))) {
                protected_pos[i] = true;
            }
        }
    }
    return protected_pos;
}

/* ---- 3.3: split_sentences using pre-computed positions (O(1) lookup per char) ---- */

static std::vector<std::string> split_sentences_internal(
    const std::string & text,
    const std::vector<bool> & pre_protected) {
    std::vector<std::string> parts;
    if (text.empty()) return parts;

    size_t start = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        bool is_boundary = false;
        size_t boundary_len = 1;

        if (c == '.' && !pre_protected[i]) {
            if (i + 1 >= text.size()) {
                is_boundary = true;
            } else {
                char next = text[i + 1];
                if (next == ' ' || next == '"' || next == ')') {
                    is_boundary = true;
                } else if (i + 3 < text.size() &&
                           static_cast<unsigned char>(text[i + 1]) == 0xE2 &&
                           static_cast<unsigned char>(text[i + 2]) == 0x80 &&
                           static_cast<unsigned char>(text[i + 3]) == 0x9D) {
                    is_boundary = true;
                }
            }
        } else if (c == '!' || c == '?') {
            is_boundary = true;
        } else if (char_is_ellipsis(text, i)) {
            is_boundary = true;
            boundary_len = 3;
        } else if (char_is_zh_sentence_punct(text, i)) {
            is_boundary = true;
            boundary_len = 3;
        }

        if (is_boundary) {
            size_t end = i + boundary_len;

            // Include trailing closing quotes/parentheses
            size_t closer_len = 0;
            while (end < text.size() &&
                   char_is_closing_punctuation(text, end, closer_len)) {
                end += closer_len;
            }
            // Include trailing space
            if (end < text.size() && text[end] == ' ') ++end;

            parts.push_back(text.substr(start, end - start));
            start = end;
            i = end - 1;
        }
    }
    if (start < text.size()) {
        parts.push_back(text.substr(start));
    }
    return parts;
}

} // namespace anonymous

// ============================================================================
// Public API — declared in text_splitter.h
// ============================================================================

Boundary infer_boundary_type(const std::string & text) {
    return infer_boundary_type_impl(text);
}

bool is_protected_dot(const std::string & text, size_t pos) {
    return is_protected_token(text, pos);
}

bool is_sentence_boundary(const std::string & text, size_t pos) {
    if (pos >= text.size()) return false;
    char c = text[pos];
    if (char_is_zh_sentence_punct(text, pos)) return true;
    if (c == '.') return !is_protected_token(text, pos) && (pos + 1 >= text.size() ||
        text[pos+1] == ' ' || text[pos+1] == '"' || text[pos+1] == ')' ||
        (pos + 3 < text.size() && static_cast<unsigned char>(text[pos+1]) == 0xE2 &&
         static_cast<unsigned char>(text[pos+2]) == 0x80 &&
         static_cast<unsigned char>(text[pos+3]) == 0x9D));
    if (c == '!' || c == '?') return true;
    return char_is_ellipsis(text, pos);
}

/* ---- 3.1: split_keep_delimiter with first-char lookup table (O(n+m)) ---- */

std::vector<std::string> split_keep_delimiter(
    const std::string & text,
    const std::vector<std::string> & delimiters) {
    std::vector<std::string> parts;
    if (text.empty() || delimiters.empty()) return parts;

    // Build lookup: which ASCII chars can start a delimiter
    bool can_start[256] = {};
    for (const auto & d : delimiters) {
        if (!d.empty()) {
            can_start[static_cast<unsigned char>(d[0])] = true;
        }
    }

    size_t start = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (!can_start[c]) continue;

        // Try delimiters in order (preserves original first-match semantics)
        bool matched = false;
        for (const auto & delim : delimiters) {
            if (static_cast<unsigned char>(delim[0]) == c &&
                i + delim.size() <= text.size() &&
                text.compare(i, delim.size(), delim) == 0) {
                parts.push_back(text.substr(start, i - start + delim.size()));
                start = i + delim.size();
                i += delim.size();
                matched = true;
                break;
            }
        }
        if (matched) continue; // i was advanced; outer ++i skips past delimiter
    }
    if (start < text.size()) {
        parts.push_back(text.substr(start));
    }
    return parts;
}

std::vector<std::string> split_paragraphs(const std::string & text) {
    return split_keep_delimiter(text, { "\n\n" });
}

std::vector<std::string> split_sentences(const std::string & text) {
    auto protected_pos = precompute_protected_positions(text);
    return split_sentences_internal(text, protected_pos);
}

std::vector<std::string> split_into_candidate_units(const std::string & text) {
    std::vector<std::string> units;
    if (text.empty()) return units;

    auto paragraphs = split_paragraphs(text);
    for (const auto & para : paragraphs) {
        auto sentences = split_sentences(para);
        for (const auto & sent : sentences) {
            auto clauses = split_keep_delimiter(sent, { "；", "：", "，", "、" });
            for (const auto & clause : clauses) {
                std::string s = trim_ascii(clause);
                if (!s.empty()) {
                    units.push_back(s);
                }
            }
        }
    }
    return units;
}

std::vector<std::string> force_split_unit(const std::string & text) {
    auto strong_parts = split_keep_delimiter(text, { "; ", ": ", ";", ":", "；", "：" });
    if (strong_parts.size() > 1) return strong_parts;

    auto weak_parts = split_keep_delimiter(text, { ", ", "，", "、" });
    if (weak_parts.size() > 1) return weak_parts;

    return split_by_words(text);
}

std::vector<std::string> split_by_words(const std::string & text) {
    std::vector<std::string> words;
    if (text.empty()) return words;

    std::string current;
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current.empty()) {
                words.push_back(std::move(current));
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) {
        words.push_back(std::move(current));
    }
    return words;
}

bool is_strong_boundary(Boundary b) {
    return b == Boundary::Sentence ||
           b == Boundary::Paragraph ||
           b == Boundary::Newline;
}

bool is_reasonable_boundary(Boundary b) {
    return b == Boundary::Sentence ||
           b == Boundary::ClauseStrong ||
           b == Boundary::Paragraph ||
           b == Boundary::Newline;
}

} // namespace kokopop
