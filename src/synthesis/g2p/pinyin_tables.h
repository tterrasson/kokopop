#pragma once
// Ported from misaki/transcription.py — pinyin → IPA mapping tables.
//
// This file contains:
//   - INITIAL_MAPPING: 21 initials (b, c, ch, d, ..., z, zh) → IPA
//   - FINAL_MAPPING: 34 finals (a, ai, an, ..., ü, üe, üan, ün) → IPA
//   - SYLLABIC_CONSONANT_MAPPINGS: hm, hng, m, n, ng → IPA
//   - INTERJECTION_MAPPINGS: er, o, io → IPA
//   - Special final overrides for zh/ch/sh/r + z/c/s contexts
//   - TONE_MAPPING: tone digit → Kokoro tone marker
//
// IPA uses Kokoro-compatible symbols:
//   Tones: → (1st), ↗ (2nd), ↓ (3rd), ↘ (4th), "" (neutral/5th)
//   Special: ꭧ (tʂ), ʨ (tɕ), ɕ, ʂ, ɻ, ə, ɤ, ɨ, etc.

#include <array>
#include <string_view>

namespace kokopop::g2p::zh {

// ────────────────────────────────────────────────────────────────────────
// Tone mapping: digit char → Kokoro/misaki tone marker
//
// The Kokoro model was trained with misaki's arrow-style tone markers:
//   → (1st, high flat)   = U+2192 (UTF-8: 0xE2 0x86 0x92)
//   ↗ (2nd, rising)      = U+2197 (UTF-8: 0xE2 0x86 0x97)
//   ↓ (3rd, dipping)     = U+2193 (UTF-8: 0xE2 0x86 0x93)
//   ↘ (4th, falling)     = U+2198 (UTF-8: 0xE2 0x86 0x98)
//   (empty) (5th, neutral)
// ────────────────────────────────────────────────────────────────────────
inline constexpr std::string_view tone_marker(char tone_digit) {
    switch (tone_digit) {
        case '1': return "\xE2\x86\x92";   // → (arrow right)
        case '2': return "\xE2\x86\x97";   // ↗ (arrow NE)
        case '3': return "\xE2\x86\x93";   // ↓ (arrow down)
        case '4': return "\xE2\x86\x98";   // ↘ (arrow SE)
        case '5': return "";
        default:  return "";
    }
}

// ────────────────────────────────────────────────────────────────────────
// Initial consonants → IPA (first alternative only, most common)
// ────────────────────────────────────────────────────────────────────────
struct InitialEntry {
    const char * pinyin;
    const char * ipa;
};

// Sorted by length descending for matching (zh/ch before single chars)
inline constexpr InitialEntry g_initials[] = {
    {"zh", "ꭧ"},      // tʂ
    {"ch", "ꭧʰ"},     // tʂʰ
    {"sh", "ʂ"},
    {"b",  "p"},
    {"c",  "ʦʰ"},     // tsʰ
    {"d",  "t"},
    {"f",  "f"},
    {"g",  "k"},
    {"h",  "x"},
    {"j",  "ʨ"},      // tɕ
    {"k",  "kʰ"},
    {"l",  "l"},
    {"m",  "m"},
    {"n",  "n"},
    {"p",  "pʰ"},
    {"q",  "ʨʰ"},     // tɕʰ
    {"r",  "ɻ"},
    {"s",  "s"},
    {"t",  "tʰ"},
    {"x",  "ɕ"},
    {"z",  "ʦ"},      // ts
};
static_assert(sizeof(g_initials) / sizeof(g_initials[0]) == 21);

inline bool is_zh_ch_sh_initial(std::string_view p) {
    return p == "zh" || p == "ch" || p == "sh" || p == "r";
}
inline bool is_z_c_s_initial(std::string_view p) {
    return p == "z" || p == "c" || p == "s";
}

// ────────────────────────────────────────────────────────────────────────
// Finals → IPA components (stored as flat strings, tone placeholder = "0")
// ────────────────────────────────────────────────────────────────────────
// We store each final's IPA as a single concatenated string with "0" as tone
// placeholder. The apply_tone() function replaces "0" with the actual tone.
//
// For finals with multiple parts (e.g. "ian" → j + ɛ0 + n), the "0" appears
// on the vowel part. apply_tone scans and replaces only the first "0".

struct FinalEntry {
    const char * pinyin;
    const char * ipa;   // IPA with "0" as tone placeholder
};

inline constexpr FinalEntry g_finals[] = {
    {"a",   "a0"},
    {"ai",  "ai̯0"},
    {"an",  "a0n"},
    {"ang", "a0ŋ"},
    {"ao",  "au̯0"},
    {"e",   "ɤ0"},
    {"ei",  "ei̯0"},
    {"en",  "ə0n"},
    {"eng", "ə0ŋ"},
    {"i",   "i0"},
    {"ia",  "ja0"},
    {"ian", "jɛ0n"},
    {"iang","ja0ŋ"},
    {"iao", "jau̯0"},
    {"ie",  "je0"},
    {"in",  "i0n"},
    {"iou", "jou̯0"},
    {"ing", "i0ŋ"},
    {"iong","jʊ0ŋ"},
    {"ong", "ʊ0ŋ"},
    {"ou",  "ou̯0"},
    {"u",   "u0"},
    {"uei", "wei̯0"},
    {"ua",  "wa0"},
    {"uai", "wai̯0"},
    {"uan", "wa0n"},
    {"uen", "wə0n"},
    {"uang","wa0ŋ"},
    {"ueng","wə0ŋ"},
    {"uo",  "wo0"},
    {"o",   "wo0"},
    // v = ü (pypinyin convention)
    {"v",   "y0"},
    {"ve",  "ɥe0"},
    {"van", "ɥɛ0n"},
    {"vn",  "y0n"},
};
static_assert(sizeof(g_finals) / sizeof(g_finals[0]) == 35);

// ────────────────────────────────────────────────────────────────────────
// Special finals after zh, ch, sh, r: "i" → ɻ̩0 (syllabic retroflex)
// ────────────────────────────────────────────────────────────────────────
inline constexpr std::string_view FINAL_I_AFTER_ZH_CH_SH_R = "ɻ̩0";

// After z, c, s: "i" → ɹ̩0 (syllabic denti-alveolar)
inline constexpr std::string_view FINAL_I_AFTER_Z_C_S = "ɹ̩0";

// ────────────────────────────────────────────────────────────────────────
// Syllabic consonants (standalone, no vowel)
// ────────────────────────────────────────────────────────────────────────
struct SyllabicEntry {
    const char * pinyin;
    const char * ipa;
};

inline constexpr SyllabicEntry g_syllabic_consonants[] = {
    {"hm",  "hm0"},
    {"hng", "hŋ0"},
    {"m",   "m0"},
    {"n",   "n0"},
    {"ng",  "ŋ0"},
};

// ────────────────────────────────────────────────────────────────────────
// Interjections
// ────────────────────────────────────────────────────────────────────────
struct InterjectionEntry {
    const char * pinyin;
    const char * ipa;
};

inline constexpr InterjectionEntry g_interjections[] = {
    {"er",  "ɚ0"},
    {"o",   "ɔ0"},
    {"io",  "jɔ0"},
};

// ────────────────────────────────────────────────────────────────────────
// Helper: Apply tone to IPA — replace first "0" with tone marker
// For neutral tone (empty string), the "0" is simply removed.
// ────────────────────────────────────────────────────────────────────────
inline std::string apply_tone(std::string_view ipa_with_placeholder,
                               std::string_view tone) {
    std::string result;
    result.reserve(ipa_with_placeholder.size() + tone.size());
    bool replaced = false;
    for (size_t i = 0; i < ipa_with_placeholder.size(); ++i) {
        if (!replaced && ipa_with_placeholder[i] == '0') {
            result.append(tone);
            replaced = true;
        } else {
            result.push_back(ipa_with_placeholder[i]);
        }
    }
    return result;
}

// ────────────────────────────────────────────────────────────────────────
// Pinyin → IPA conversion (single syllable)
// Input: pinyin in TONE3 format, e.g. "zhi3", "hao4", "lv2"
// Output: Kokoro-compatible IPA string, e.g. "ꭧɨ↓", "xau̯↘", "ɥe↗"
// ────────────────────────────────────────────────────────────────────────
std::string pinyin_to_ipa(std::string_view pinyin_tone3);

} // namespace kokopop::g2p::zh
