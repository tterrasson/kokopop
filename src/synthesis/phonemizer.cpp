#include "phonemizer.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string_view>
#include <unordered_map>

#include <espeak-ng/speak_lib.h>

#include "synthesis/g2p/zh_g2p.h"

namespace kokopop {
namespace {

bool is_prosody_punctuation(std::string_view ch) {
    return ch == ";" || ch == ":" || ch == "," || ch == "." || ch == "!" ||
           ch == "?" || ch == "—" || ch == "…";
}

void append_part(std::string & out, const std::string & part) {
    if (part.empty()) {
        return;
    }
    if (!out.empty() && out.back() != ' ') {
        out.push_back(' ');
    }
    out += part;
}

bool phonemize_fragment(const std::string & text, std::string & out) {
    const char * input = text.c_str();
    const void * input_ptr = input;
    for (;;) {
        const char * p = espeak_TextToPhonemes(&input_ptr, espeakCHARS_UTF8, espeakPHONEMES_IPA);
        if (p == nullptr || *p == '\0') {
            break;
        }
        append_part(out, p);
        if (input_ptr == nullptr || static_cast<const char *>(input_ptr)[0] == '\0') {
            break;
        }
    }
    return true;
}

// ─── Single-pass phoneme normalization ───────────────────────────────

struct Rule {
    std::string_view pattern;
    std::string_view replacement;
};

static const Rule common_rules[] = {
    // Diphthongs & affricates — longest first
    {"ɔɪ", "Y"},
    {"əʊ", "Q"},
    {"dʒ", "ʤ"},
    {"tʃ", "ʧ"},
    {"aɪ", "I"},
    {"aʊ", "W"},
    {"eɪ", "A"},
    {"oʊ", "O"},
    {"ts", "ʦ"},
    {"dz", "ʣ"},
    {"ss", "S"},
    {"«", "("},
    {"»", ")"},
    {"\n", " "},
    {"\r", " "},
    {"^", ""},
    {"-", ""},
};

static const Rule english_rules[] = {
    // Glottal + syllabic-n: pre-composed with later ʔ→t rule
    {"ʔˌn̩", "tn"},
    {"ʔn̩",   "tn"},
    {"ʲə", "jə"},
    // Syllabic consonants
    {"n̩", "ᵊn"},
    {"l̩", "ᵊl"},
    {"m̩", "ᵊm"},
    {"ʲo", "jo"},
    {"əl", "ᵊl"},
    // Single-char substitutions
    {"ɚ", "əɹ"},
    {"ɐ", "ə"},
    {"ɬ", "l"},
    {"ç", "k"},
    {"ɾ", "T"},
    {"ʔ", "t"},
    {"ʲ", ""},
    {"r", "ɹ"},
    {"x", "k"},
    {"o", "ɔ"},
};

static const Rule american_rules[] = {
    {"ɜːɹ", "ɜɹ"},
    {"ɜː",   "ɜɹ"},
    {"ɪə",   "iə"},
    {"ː",    ""},
};

static const Rule british_rules[] = {
    {"eə", "ɛː"},
    {"iə", "ɪə"},
};

static const Rule non_english_rules[] = {
    {"\xcc\xa9", ""},  // U+0329 combining vertical line below (syllabic marker)
    {"ʲ",        ""},  // palatalization marker
};

template <typename T, size_t N>
constexpr size_t countof(const T (&)[N]) {
    return N;
}

} // namespace

std::string espeak_voice_for_kokoro_voice(const std::string & voice) {
    const char lang = voice.empty() ? 'a' : voice[0];
    switch (lang) {
    case 'a': return "gmw/en-US";
    case 'b': return "gmw/en";
    case 'e': return "roa/es";
    case 'f': return "roa/fr";
    case 'h': return "inc/hi";
    case 'i': return "roa/it";
    case 'j': return "jpx/ja";
    case 'p': return "roa/pt-BR";
    case 'z': return "sit/cmn";
    default: return "gmw/en-US";
    }
}

std::string normalize_espeak_phonemes(std::string ps, char kokoro_lang) {
    // Select language-specific rule groups
    const Rule * lang_rules   = nullptr;
    size_t        lang_count  = 0;
    const Rule * dialect_rules = nullptr;
    size_t        dialect_count = 0;

    if (kokoro_lang == 'a') {
        lang_rules   = english_rules;
        lang_count   = countof(english_rules);
        dialect_rules = american_rules;
        dialect_count = countof(american_rules);
    } else if (kokoro_lang == 'b') {
        lang_rules   = english_rules;
        lang_count   = countof(english_rules);
        dialect_rules = british_rules;
        dialect_count = countof(british_rules);
    } else {
        lang_rules   = non_english_rules;
        lang_count   = countof(non_english_rules);
    }

    // ── Single-pass scan ──────────────────────────────────────────────
    std::string result;
    result.reserve(ps.size());

    const size_t common_count = countof(common_rules);

    for (size_t i = 0; i < ps.size();) {
        bool matched = false;

        // 1. Common rules (all languages)
        for (size_t j = 0; j < common_count; ++j) {
            const auto & rule = common_rules[j];
            const size_t len = rule.pattern.size();
            if (i + len <= ps.size() && ps.compare(i, len, rule.pattern) == 0) {
                result.append(rule.replacement);
                i += len;
                matched = true;
                break;
            }
        }

        // 2. Combining tilde removal (non-French only)
        if (!matched && kokoro_lang != 'f') {
            const unsigned char tilde_hi = 0xcc, tilde_lo = 0x83;
            if (i + 1 < ps.size() &&
                static_cast<unsigned char>(ps[i])   == tilde_hi &&
                static_cast<unsigned char>(ps[i + 1]) == tilde_lo) {
                // U+0303 combining tilde — strip silently
                i += 2;
                matched = true;
            }
        }

        // 3. Language-specific rules
        if (!matched) {
            for (size_t j = 0; j < lang_count; ++j) {
                const auto & rule = lang_rules[j];
                const size_t len = rule.pattern.size();
                if (i + len <= ps.size() && ps.compare(i, len, rule.pattern) == 0) {
                    result.append(rule.replacement);
                    i += len;
                    matched = true;
                    break;
                }
            }
        }

        // 4. Dialect-specific rules (American / British)
        if (!matched && dialect_rules) {
            for (size_t j = 0; j < dialect_count; ++j) {
                const auto & rule = dialect_rules[j];
                const size_t len = rule.pattern.size();
                if (i + len <= ps.size() && ps.compare(i, len, rule.pattern) == 0) {
                    result.append(rule.replacement);
                    i += len;
                    matched = true;
                    break;
                }
            }
        }

        // 5. No rule matched — copy byte verbatim
        if (!matched) {
            result.push_back(ps[i++]);
        }
    }

    // ── Collapse whitespace + strip trailing space ────────────────────
    std::string collapsed;
    collapsed.reserve(result.size());
    bool prev_space = true;

    for (size_t i = 0; i < result.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(result[i]);
        if (c == ' ' || c == '\t') {
            if (!prev_space) {
                collapsed.push_back(' ');
            }
            prev_space = true;
        } else {
            collapsed.push_back(static_cast<char>(c));
            prev_space = false;
        }
    }

    while (!collapsed.empty() && collapsed.back() == ' ') {
        collapsed.pop_back();
    }

    return collapsed;
}

bool phonemize_text(const std::string & text, const std::string & voice, std::string & phonemes, std::string & error) {
    const char lang = voice.empty() ? 'a' : voice[0];

    // Route Mandarin (z) to pure C++ G2P — no eSpeak needed
    if (lang == 'z') {
        return g2p::zh::g2p_chinese(text, phonemes, error);
    }

    // All other languages: eSpeak-ng path
    static std::once_flag init_once;
    static int init_result = 0;
    static std::mutex espeak_mutex;
    std::call_once(init_once, [] {
        init_result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, nullptr, 0);
    });
    if (init_result < 0) {
        error = "failed to initialize eSpeak-ng";
        return false;
    }

    std::string out;
    {
        std::lock_guard<std::mutex> lock(espeak_mutex);
        const std::string espeak_voice = espeak_voice_for_kokoro_voice(voice);
        if (espeak_SetVoiceByName(espeak_voice.c_str()) != EE_OK) {
            error = "failed to select eSpeak-ng voice: " + espeak_voice;
            return false;
        }

        std::string fragment;
        fragment.reserve(text.size());
        size_t off = 0;
        while (off < text.size()) {
            const size_t start = off;
            unsigned char c = static_cast<unsigned char>(text[off]);
            size_t len = 1;
            if ((c & 0x80) != 0) {
                if      ((c & 0xE0) == 0xC0) len = 2;
                else if ((c & 0xF0) == 0xE0) len = 3;
                else if ((c & 0xF8) == 0xF0) len = 4;
            }
            if (start + len > text.size()) {
                len = 1;
            }

            std::string_view ch(text.data() + start, len);
            if (is_prosody_punctuation(ch)) {
                const bool had_content = !fragment.empty();
                phonemize_fragment(fragment, out);
                fragment.clear();
                if (had_content) {
                    append_part(out, std::string(ch));
                }
            } else {
                fragment.append(ch);
            }
            off += len;
        }
        phonemize_fragment(fragment, out);
    }
    phonemes = normalize_espeak_phonemes(out, lang);
    return true;
}

} // namespace kokopop
