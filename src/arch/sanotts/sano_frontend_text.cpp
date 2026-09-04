#include "arch/sanotts/sano_frontend_text.h"

#include "core/utf8.h"
#include "synthesis/phonemizer.h"

#include <espeak-ng/speak_lib.h>

#include <algorithm>
#include <string>
#include <vector>

namespace kokopop::sano {

// phonemizer's default marks. Multi-byte ones are spelled out so the source
// stays readable next to the ASCII half.
const char * const MISAKI_PUNCTUATION_MARKS = ";:,.!?¡¿—…\"«»“”(){}[]";

// Piper's set: the ASCII punctuation that appears in its phoneme_id_map.
const char * const PIPER_PUNCTUATION_MARKS = "!'(),-.:;?\"";

namespace {

/// espeak phoneme mode: IPA, plus the tie character in bits 8-23.
int ipa_mode_with_tie(char tie) {
    int mode = espeakPHONEMES_IPA;
    if (tie != '\0') {
        mode |= espeakPHONEMES_TIE | (static_cast<int>(static_cast<unsigned char>(tie)) << 8);
    }
    return mode;
}

bool is_ascii_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

/// Splits `marks`, a UTF-8 string of single-code-point marks, into a set.
std::vector<std::string> mark_set(std::string_view marks) {
    std::vector<std::string> out;
    size_t offset = 0;
    std::string_view ch;
    while (utf8_next(marks, offset, ch)) {
        out.emplace_back(ch);
    }
    return out;
}

bool contains(const std::vector<std::string> & set, std::string_view value) {
    return std::find(set.begin(), set.end(), value) != set.end();
}

/// One punctuation run, with the whitespace that surrounds it.
struct MarkRun {
    size_t begin = 0;  // byte offset in the input
    size_t end = 0;
    std::string text;
};

/// What a code point is worth to the run scanner.
enum class CharKind { Other, Space, Mark };

struct Unit {
    size_t begin = 0;   // byte offset in the input
    size_t end = 0;
    CharKind kind = CharKind::Other;
};

/// Classifies every code point of `text` once. Fails on invalid UTF-8.
bool classify(std::string_view text, const std::vector<std::string> & marks,
              std::vector<Unit> & units, std::string & error) {
    units.clear();
    size_t offset = 0;
    std::string_view ch;
    for (;;) {
        const size_t begin = offset;
        if (!utf8_next(text, offset, ch)) {
            break;
        }
        Unit unit;
        unit.begin = begin;
        unit.end = offset;
        if (ch.size() == 1 && is_ascii_space(ch[0])) {
            unit.kind = CharKind::Space;
        } else if (contains(marks, ch)) {
            unit.kind = CharKind::Mark;
        }
        units.push_back(unit);
    }
    if (offset != text.size()) {
        error = "text is not valid UTF-8";
        return false;
    }
    return true;
}

/// Finds every maximal run matching `\s*[marks]+\s*`, repeated.
///
/// This is the shape phonemizer's regex matches, and the surrounding
/// whitespace being part of the run is what makes the restored output read
/// `"ph, ph"` rather than `"ph ,ph"`. Such a run is exactly a maximal stretch
/// of whitespace-or-mark code points containing at least one mark, so one
/// forward pass over the classified units finds them all.
bool find_mark_runs(std::string_view text, const std::vector<std::string> & marks,
                    std::vector<MarkRun> & runs, std::string & error) {
    runs.clear();
    std::vector<Unit> units;
    if (!classify(text, marks, units, error)) {
        return false;
    }

    size_t i = 0;
    while (i < units.size()) {
        if (units[i].kind == CharKind::Other) {
            ++i;
            continue;
        }
        const size_t begin = i;
        bool has_mark = false;
        while (i < units.size() && units[i].kind != CharKind::Other) {
            has_mark = has_mark || units[i].kind == CharKind::Mark;
            ++i;
        }
        // A stretch of pure whitespace is not a run: it belongs to whichever
        // fragment espeak phonemizes.
        if (!has_mark) {
            continue;
        }
        MarkRun run;
        run.begin = units[begin].begin;
        run.end = units[i - 1].end;
        run.text = std::string(text.substr(run.begin, run.end - run.begin));
        runs.push_back(std::move(run));
    }
    return true;
}

/// Collapses runs of whitespace inside a mark to a single space.
///
/// The reference replaces the word separator into the mark; with a
/// single-space separator that is what this comes to.
std::string normalize_mark(const std::string & mark) {
    std::string out;
    out.reserve(mark.size());
    bool in_space = false;
    for (char c : mark) {
        if (is_ascii_space(c)) {
            if (!in_space) {
                out.push_back(' ');
                in_space = true;
            }
            continue;
        }
        in_space = false;
        out.push_back(c);
    }
    return out;
}

std::string trim(std::string_view text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && is_ascii_space(text[begin])) {
        ++begin;
    }
    while (end > begin && is_ascii_space(text[end - 1])) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

void replace_all(std::string & text, std::string_view needle,
                 std::string_view replacement) {
    if (needle.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        text.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
}

// misaki's E2M table, in the order misaki produces: sorted by decreasing key
// length, stable over the dict's insertion order. The order is part of the
// contract — `e^ɪ -> A` has to fire before the bare `e -> A`.
struct Rewrite {
    std::string_view from;
    std::string_view to;
};

constexpr const char * SYLLABIC = "\xcc\xa9";  // U+0329 COMBINING VERTICAL LINE BELOW

const Rewrite E2M[] = {
    {"\xca\x94\xcb\x8c\x6e\xcc\xa9", "\xca\x94\x6e"},   // U+0294 U+02CC n U+0329 -> U+0294 n
    {"\xca\x94\x6e\xcc\xa9", "\xca\x94\x6e"},              // U+0294 n U+0329       -> U+0294 n
    {"\x61\x5e\xc9\xaa", "\x49"},                            // a^U+026A -> I
    {"\x61\x5e\xca\x8a", "\x57"},                            // a^U+028A -> W
    {"\x64\x5e\xca\x92", "\xca\xa4"},                        // d^U+0292 -> U+02A4
    {"\x65\x5e\xc9\xaa", "\x41"},                            // e^U+026A -> A
    {"\x74\x5e\xca\x83", "\xca\xa7"},                        // t^U+0283 -> U+02A7
    {"\xc9\x94\x5e\xc9\xaa", "\x59"},                        // U+0254^U+026A -> Y
    {"\xc9\x99\x5e\x6c", "\xe1\xb5\x8a\x6c"},               // U+0259^l -> U+1D4A l
    {"\xca\xb2\x6f", "\x6a\x6f"},                            // U+02B2 o -> j o
    {"\xca\xb2\xc9\x99", "\x6a\xc9\x99"},                    // U+02B2 U+0259 -> j U+0259
    {"\x65", "\x41"},                                          // e -> A
    {"\xca\xb2", ""},                                          // U+02B2 -> (dropped)
    {"\xc9\x9a", "\xc9\x99\xc9\xb9"},                         // U+025A -> U+0259 U+0279
    {"\x72", "\xc9\xb9"},                                      // r -> U+0279
    {"\x78", "\x6b"},                                          // x -> k
    {"\xc3\xa7", "\x6b"},                                      // U+00E7 -> k
    {"\xc9\x90", "\xc9\x99"},                                  // U+0250 -> U+0259
    {"\xc9\xac", "\x6c"},                                      // U+026C -> l
    {"\xcc\x83", ""},                                          // U+0303 combining tilde
};


/// `re.sub(r'(\S)̩', r'ᵊ\1', ps)`: a syllabic consonant becomes a schwa
/// followed by that consonant.
std::string rewrite_syllabic(const std::string & text) {
    const std::string_view marker(SYLLABIC);
    std::string out;
    out.reserve(text.size());

    size_t offset = 0;
    std::string_view ch;
    std::string previous;      // last emitted code point, when non-space
    size_t previous_len = 0;   // its length in `out`, 0 when not applicable
    while (offset < text.size()) {
        if (!utf8_next(text, offset, ch)) {
            // Not valid UTF-8. Copying the byte through and moving on keeps
            // this a rewrite rather than a truncation; the tokenizer is the
            // layer that rejects the string.
            out.push_back(text[offset]);
            ++offset;
            previous.clear();
            previous_len = 0;
            continue;
        }
        if (ch == marker && previous_len > 0) {
            // Replace the pending non-space code point with `ᵊ` + itself.
            out.resize(out.size() - previous_len);
            out += "\xe1\xb5\x8a";  // U+1D4A MODIFIER LETTER SMALL SCHWA
            out += previous;
            previous.clear();
            previous_len = 0;
            continue;
        }
        const bool space = ch.size() == 1 && is_ascii_space(ch[0]);
        out += ch;
        if (space || ch == marker) {
            previous.clear();
            previous_len = 0;
        } else {
            previous.assign(ch);
            previous_len = ch.size();
        }
    }
    return out;
}

} // namespace

std::string normalize_misaki_e2m(std::string ps) {
    ps = trim(ps);
    for (const Rewrite & rule : E2M) {
        replace_all(ps, rule.from, rule.to);
    }
    ps = rewrite_syllabic(ps);
    // Any syllabic marker that had no preceding consonant is dropped.
    replace_all(ps, SYLLABIC, "");

    // American English tail. `british` is false for every sanoTTS vocos voice.
    replace_all(ps, "\x6f\x5e\xca\x8a", "\x4f");                 // o^U+028A -> O
    replace_all(ps, "\xc9\x9c\xcb\x90\xc9\xb9", "\xc9\x9c\xc9\xb9");  // U+025C U+02D0 U+0279
    replace_all(ps, "\xc9\x9c\xcb\x90", "\xc9\x9c\xc9\xb9");     // U+025C U+02D0
    replace_all(ps, "\xc9\xaa\xc9\x99", "\x69\xc9\x99");         // U+026A U+0259 -> i U+0259
    replace_all(ps, "\xcb\x90", "");                            // U+02D0 length mark

    // Kept from the reference: harmless on espeak >= 1.52, where `o` no longer
    // appears, and required on older builds.
    replace_all(ps, "\x6f", "\xc9\x94");                        // o -> U+0254
    replace_all(ps, "\xc9\xbe", "\x54");                        // U+027E flap -> T
    replace_all(ps, "\xca\x94", "\x74");                        // U+0294 glottal -> t
    replace_all(ps, "\x5e", "");                                // the tie has done its job
    return ps;
}

bool phonemize_preserving_punctuation(const std::string & text,
                                      const std::string & espeak_voice,
                                      int phoneme_mode,
                                      std::string_view marks,
                                      std::string & phonemes,
                                      std::string & error) {
    phonemes.clear();
    const std::vector<std::string> mark_list = mark_set(marks);
    std::vector<MarkRun> runs;
    if (!find_mark_runs(text, mark_list, runs, error)) {
        return false;
    }

    if (runs.empty()) {
        return espeak_phonemize_raw(text, espeak_voice, phoneme_mode, phonemes, error);
    }

    // The input is nothing but punctuation: there is nothing to phonemize, so
    // the marks are the whole answer.
    if (runs.size() == 1 && runs.front().begin == 0 && runs.front().end == text.size()) {
        phonemes = normalize_mark(runs.front().text);
        return true;
    }

    std::string out;
    size_t cursor = 0;
    for (const MarkRun & run : runs) {
        const std::string fragment = text.substr(cursor, run.begin - cursor);
        if (!trim(fragment).empty()) {
            std::string part;
            if (!espeak_phonemize_raw(fragment, espeak_voice, phoneme_mode, part, error)) {
                return false;
            }
            // espeak emits no trailing separator, so nothing to strip here.
            out += part;
        }
        out += normalize_mark(run.text);
        cursor = run.end;
    }
    const std::string tail = text.substr(cursor);
    if (!trim(tail).empty()) {
        std::string part;
        if (!espeak_phonemize_raw(tail, espeak_voice, phoneme_mode, part, error)) {
            return false;
        }
        out += part;
    }

    phonemes = std::move(out);
    return true;
}

bool phonemize_misaki_sanotts(const std::string & text,
                              const std::string & espeak_voice,
                              std::string & phonemes,
                              std::string & error) {
    std::string raw;
    if (!phonemize_preserving_punctuation(text, espeak_voice, ipa_mode_with_tie('^'),
                                          MISAKI_PUNCTUATION_MARKS, raw, error)) {
        return false;
    }
    phonemes = normalize_misaki_e2m(std::move(raw));
    if (phonemes.empty()) {
        error = "phonemization produced nothing for: " + text;
        return false;
    }
    return true;
}

bool phonemize_piper_sanotts(const std::string & text,
                             const std::string & espeak_voice,
                             std::string & phonemes,
                             std::string & error) {
    std::string raw;
    if (!phonemize_preserving_punctuation(text, espeak_voice, ipa_mode_with_tie('\0'),
                                          PIPER_PUNCTUATION_MARKS, raw, error)) {
        return false;
    }
    // The reference right-strips before decomposing: phonemizer emits a
    // trailing separator space that Piper's own bridge does not. This is the
    // frontend's own contract — the phoneme string it returns is the one the
    // reference produces, and the corpus gate compares it byte for byte — so
    // it stands even though `tokenize_piper()` strips again for callers that
    // hand it a string from somewhere else.
    while (!raw.empty() && is_ascii_space(raw.back())) {
        raw.pop_back();
    }
    phonemes = std::move(raw);
    if (phonemes.empty()) {
        error = "phonemization produced nothing for: " + text;
        return false;
    }
    return true;
}

} // namespace kokopop::sano
