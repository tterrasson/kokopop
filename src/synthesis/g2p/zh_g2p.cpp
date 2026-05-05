#include "zh_g2p.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "pinyin_dict.h"
#include "pinyin_tables.h"
#include "num2cn.h"

namespace kokopop::g2p::zh {
namespace {

// ────────────────────────────────────────────────────────────────────────
// UTF-8 helpers
// ────────────────────────────────────────────────────────────────────────

inline uint32_t utf8_decode(const char * p, size_t & byte_len) {
    unsigned char c = static_cast<unsigned char>(p[0]);
    if ((c & 0x80) == 0) {
        byte_len = 1;
        return static_cast<uint32_t>(c);
    }
    if ((c & 0xE0) == 0xC0) {
        byte_len = 2;
        return (static_cast<uint32_t>(c & 0x1F) << 6) |
               (static_cast<uint32_t>(p[1] & 0x3F));
    }
    if ((c & 0xF0) == 0xE0) {
        byte_len = 3;
        return (static_cast<uint32_t>(c & 0x0F) << 12) |
               (static_cast<uint32_t>(p[1] & 0x3F) << 6) |
               (static_cast<uint32_t>(p[2] & 0x3F));
    }
    if ((c & 0xF8) == 0xF0) {
        byte_len = 4;
        return (static_cast<uint32_t>(c & 0x07) << 18) |
               (static_cast<uint32_t>(p[1] & 0x3F) << 12) |
               (static_cast<uint32_t>(p[2] & 0x3F) << 6) |
               (static_cast<uint32_t>(p[3] & 0x3F));
    }
    byte_len = 1;
    return 0xFFFFFFFF;
}

inline bool is_cjk_ideograph(uint32_t cp) {
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    if (cp >= 0x20000 && cp <= 0x2A6DF) return true;
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    return false;
}

// ────────────────────────────────────────────────────────────────────────
// Pinyin dictionary lookup (binary search)
// ────────────────────────────────────────────────────────────────────────

inline std::string_view pinyin_for_cp(uint32_t cp) {
    const auto & dict = PINYIN_CP;
    const size_t n = PINYIN_DICT_SIZE;

    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (dict[mid] < cp) {
            lo = mid + 1;
        } else if (dict[mid] > cp) {
            hi = mid;
        } else {
            size_t offset = PINYIN_OFFSET[mid];
            const char * start =
                reinterpret_cast<const char *>(PINYIN_DATA + offset);
            size_t len = 0;
            while (len < 20 && start[len] != '\0') ++len;
            return std::string_view(start, len);
        }
    }
    return {};
}

// ────────────────────────────────────────────────────────────────────────
// Punctuation mapping (Chinese → ASCII equivalents)
// ────────────────────────────────────────────────────────────────────────

inline std::string map_punctuation(const std::string & text) {
    std::string result;
    result.reserve(text.size() + 32);

    size_t i = 0;
    while (i < text.size()) {
        size_t char_len;
        uint32_t cp = utf8_decode(text.data() + i, char_len);

        switch (cp) {
            case 0x3001: result += ", "; break;
            case 0x3002: result += ". "; break;
            case 0xFF0C: result += ", "; break;
            case 0xFF0E: result += ". "; break;
            case 0xFF01: result += "! "; break;
            case 0xFF1A: result += ": "; break;
            case 0xFF1B: result += "; "; break;
            case 0xFF1F: result += "? "; break;
            case 0x300A: case 0x3008: result += " \" "; break;
            case 0x300B: case 0x3009: result += " \" "; break;
            case 0x3010: result += " \" "; break;
            case 0x3011: result += " \" "; break;
            case 0xFF08: result += " ( "; break;
            case 0xFF09: result += " ) "; break;
            case 0x00AB: result += " \" "; break;
            case 0x00BB: result += " \" "; break;
            default:
                if (cp >= 0xFF01 && cp <= 0xFF5E) {
                    char ascii = static_cast<char>(cp - 0xFEE0);
                    result += ascii;
                    if (!std::isalnum(static_cast<unsigned char>(ascii))) {
                        result += ' ';
                    }
                } else {
                    result.append(text.data() + i, char_len);
                }
                break;
        }
        i += char_len;
    }

    while (!result.empty() && (result.back() == ' ' || result.back() == '\t')) {
        result.pop_back();
    }

    return result;
}

// ────────────────────────────────────────────────────────────────────────
// Strip combining marks (U+0300–U+036F) from result
// ────────────────────────────────────────────────────────────────────────

inline std::string strip_combining_marks(const std::string & text) {
    std::string result;
    result.reserve(text.size());

    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == 0xCC && i + 1 < text.size() &&
            static_cast<unsigned char>(text[i + 1]) >= 0x80 &&
            static_cast<unsigned char>(text[i + 1]) <= 0xBF) {
            ++i;
            continue;
        }
        result.push_back(text[i]);
    }
    return result;
}

// ────────────────────────────────────────────────────────────────────────
// Replace syllabic consonants with Kokoro-compatible IPA
// ɻ̩ / ɹ̩ → ɨ (U+0268)
// ────────────────────────────────────────────────────────────────────────

inline std::string replace_syllabic_consonants(const std::string & text) {
    std::string result;
    result.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        // ɻ̩ = U+027B (0xC9 0xBB) + U+0329 (0xCC 0xA9)
        if (i + 3 < text.size() &&
            static_cast<unsigned char>(text[i])     == 0xC9 &&
            static_cast<unsigned char>(text[i + 1]) == 0xBB &&
            static_cast<unsigned char>(text[i + 2]) == 0xCC &&
            static_cast<unsigned char>(text[i + 3]) == 0xA9) {
            result += "\xC9\xA8"; // ɨ (U+0268)
            i += 4;
            continue;
        }
        // ɹ̩ = U+0279 (0xC9 0xB9) + U+0329 (0xCC 0xA9)
        if (i + 3 < text.size() &&
            static_cast<unsigned char>(text[i])     == 0xC9 &&
            static_cast<unsigned char>(text[i + 1]) == 0xB9 &&
            static_cast<unsigned char>(text[i + 2]) == 0xCC &&
            static_cast<unsigned char>(text[i + 3]) == 0xA9) {
            result += "\xC9\xA8"; // ɨ (U+0268)
            i += 4;
            continue;
        }
        result.push_back(text[i++]);
    }
    return result;
}

// ────────────────────────────────────────────────────────────────────────
// Pinyin parsing helpers
// ────────────────────────────────────────────────────────────────────────

inline char extract_tone(std::string_view py) {
    if (!py.empty() && py.back() >= '1' && py.back() <= '5') {
        return py.back();
    }
    return '5';
}

inline std::string_view strip_tone(std::string_view py) {
    if (!py.empty() && py.back() >= '1' && py.back() <= '5') {
        return py.substr(0, py.size() - 1);
    }
    return py;
}

/// Find initial in the initials table. Returns empty if none found.
inline std::string_view find_initial(std::string_view py) {
    if (py.size() >= 2) {
        std::string_view two = py.substr(0, 2);
        for (const auto & entry : g_initials) {
            if (entry.pinyin.size() == 2 && two == entry.pinyin) {
                return entry.pinyin;
            }
        }
    }
    if (py.size() >= 1) {
        std::string_view one = py.substr(0, 1);
        for (const auto & entry : g_initials) {
            if (entry.pinyin.size() == 1 && one == entry.pinyin) {
                return entry.pinyin;
            }
        }
    }
    return {};
}

inline std::string_view lookup_initial_ipa(std::string_view initial) {
    for (const auto & entry : g_initials) {
        if (entry.pinyin == initial) return entry.ipa;
    }
    return {};
}

inline std::string_view lookup_final_ipa(std::string_view final) {
    for (const auto & entry : g_finals) {
        if (entry.pinyin == final) return entry.ipa;
    }
    return {};
}

// ────────────────────────────────────────────────────────────────────────
// Normalize pinyin abbreviations: iu→iou, ui→uei, un→uen
// ────────────────────────────────────────────────────────────────────────

inline std::string expand_pinyin_abbrevs(std::string_view py_with_tone) {
    char tone = extract_tone(py_with_tone);
    std::string_view py = strip_tone(py_with_tone);
    std::string tone_suffix;
    if (tone != '5') tone_suffix = tone;

    std::string_view init = find_initial(py);
    std::string_view fin;
    if (!init.empty()) fin = py.substr(init.size());
    else fin = py;

    if (fin == "iu") return std::string(init) + "iou" + tone_suffix;
    if (fin == "ui") return std::string(init) + "uei" + tone_suffix;
    if (fin == "un") return std::string(init) + "uen" + tone_suffix;
    return std::string(py_with_tone);
}

// ────────────────────────────────────────────────────────────────────────
// Normalize y/w prefixes
// ────────────────────────────────────────────────────────────────────────

inline std::string normalize_yw_prefix(std::string_view py_with_tone) {
    char tone = extract_tone(py_with_tone);
    std::string_view py = strip_tone(py_with_tone);
    std::string tone_suffix;
    if (tone != '5') tone_suffix = tone;

    std::string result;
    if (!py.empty() && py[0] == 'y') {
        if (py == "yi") result = "i";
        else if (py == "yin") result = "in";
        else if (py == "ying") result = "ing";
        else if (py == "yu") result = "v";
        else if (py == "yuan") result = "van";
        else if (py == "yue") result = "ve";
        else if (py == "yun") result = "vn";
        else if (py == "yan") result = "ian";
        else if (py == "yang") result = "iang";
        else if (py == "yao") result = "iao";
        else if (py == "ye") result = "ie";
        else if (py == "yong") result = "iong";
        else if (py == "you") result = "iou";
        else if (py.size() >= 3 && py.substr(0, 2) == "yu") {
            result = "v" + std::string(py.substr(2));
        }
    } else if (!py.empty() && py[0] == 'w') {
        if (py == "wu") result = "u";
        else if (py == "wo") result = "uo";
        else {
            result = "u" + std::string(py.substr(1));
        }
    }

    if (result.empty()) return std::string(py_with_tone);
    return result + tone_suffix;
}

} // namespace

// ────────────────────────────────────────────────────────────────────────
// pinyin_to_ipa: TONE3 pinyin → Kokoro IPA
// ────────────────────────────────────────────────────────────────────────

std::string pinyin_to_ipa(std::string_view pinyin_tone3) {
    if (pinyin_tone3.empty()) return {};

    // Step 1: Expand abbreviations (iu→iou, ui→uei, un→uen)
    std::string expanded = expand_pinyin_abbrevs(pinyin_tone3);

    // Step 2: Normalize y/w prefixes
    std::string normalized = normalize_yw_prefix(expanded);

    char tone = extract_tone(normalized);
    std::string_view py = strip_tone(normalized);
    std::string_view tm = tone_marker(tone);

    // ── Check interjections ──────────────────────────────────────
    for (const auto & entry : g_interjections) {
        if (py == entry.pinyin) return apply_tone(entry.ipa, tm);
    }

    // ── Check syllabic consonants ────────────────────────────────
    for (const auto & entry : g_syllabic_consonants) {
        if (py == entry.pinyin) return apply_tone(entry.ipa, tm);
    }

    // ── Find initial + final ────────────────────────────────────
    std::string_view initial = find_initial(py);
    std::string_view final;
    if (!initial.empty()) final = py.substr(initial.size());
    else final = py;

    // ── Build IPA ────────────────────────────────────────────────
    std::string ipa;

    if (!initial.empty()) {
        ipa.append(lookup_initial_ipa(initial));
    }

    std::string_view final_ipa;
    if (!initial.empty() && final == "i" && is_zh_ch_sh_initial(initial)) {
        final_ipa = FINAL_I_AFTER_ZH_CH_SH_R;
    } else if (!initial.empty() && final == "i" && is_z_c_s_initial(initial)) {
        final_ipa = FINAL_I_AFTER_Z_C_S;
    } else {
        final_ipa = lookup_final_ipa(final);
    }

    if (!final_ipa.empty()) {
        ipa.append(apply_tone(final_ipa, tm));
    }

    // ── Post-processing: replace syllabic consonants → ɨ ────────
    ipa = replace_syllabic_consonants(ipa);

    return ipa;
}

// ────────────────────────────────────────────────────────────────────────
// Main entry point: g2p_chinese
// ────────────────────────────────────────────────────────────────────────

bool g2p_chinese(const std::string & text, std::string & phonemes, std::string & error) {
    if (text.empty()) {
        phonemes.clear();
        return true;
    }

    error.clear();

    // Step 1: Number → Chinese
    std::string normalized = num_to_chinese(text);

    // Step 2: Map Chinese punctuation → ASCII
    normalized = map_punctuation(normalized);

    // Step 3: Process character by character, grouping CJK segments
    std::string result;
    std::string zh_segment_phonemes;

    size_t i = 0;
    bool in_zh_segment = false;

    while (i < normalized.size()) {
        size_t char_len;
        uint32_t cp = utf8_decode(normalized.data() + i, char_len);
        bool is_zh = is_cjk_ideograph(cp);

        if (is_zh && !in_zh_segment) {
            in_zh_segment = true;
            zh_segment_phonemes.clear();
        }

        if (is_zh) {
            std::string_view pinyin = pinyin_for_cp(cp);
            if (!pinyin.empty()) {
                std::string ipa = pinyin_to_ipa(pinyin);
                if (!ipa.empty()) {
                    if (!zh_segment_phonemes.empty()) {
                        zh_segment_phonemes.push_back(' ');
                    }
                    zh_segment_phonemes += ipa;
                }
            }
        } else {
            if (in_zh_segment) {
                in_zh_segment = false;
                if (!zh_segment_phonemes.empty()) {
                    if (!result.empty() && result.back() != ' ') {
                        result.push_back(' ');
                    }
                    result += zh_segment_phonemes;
                    zh_segment_phonemes.clear();
                }
            }
            result.append(normalized.data() + i, char_len);
        }
        i += char_len;
    }

    if (!zh_segment_phonemes.empty()) {
        if (!result.empty() && result.back() != ' ') {
            result.push_back(' ');
        }
        result += zh_segment_phonemes;
    }

    // Step 4: Post-processing
    result = replace_syllabic_consonants(result);
    result = strip_combining_marks(result);

    // Collapse whitespace, strip leading/trailing
    std::string collapsed;
    collapsed.reserve(result.size());
    bool prev_space = true;
    for (unsigned char c : result) {
        if (c == ' ' || c == '\t') {
            if (!prev_space) collapsed.push_back(' ');
            prev_space = true;
        } else {
            collapsed.push_back(static_cast<char>(c));
            prev_space = false;
        }
    }
    while (!collapsed.empty() && collapsed.back() == ' ') {
        collapsed.pop_back();
    }

    phonemes = std::move(collapsed);
    return true;
}

} // namespace kokopop::g2p::zh
