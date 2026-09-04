#include "arch/sanotts/sano_tokenizer.h"

#include "core/utf8.h"

#include <algorithm>

namespace kokopop::sano {
namespace {

/// Index of `cp` in a sorted u32 array, or -1.
ptrdiff_t sorted_index(const uint32_t * values, size_t count, uint32_t cp) {
    if (values == nullptr || count == 0) {
        return -1;
    }
    const uint32_t * end = values + count;
    const uint32_t * it = std::lower_bound(values, end, cp);
    if (it == end || *it != cp) {
        return -1;
    }
    return it - values;
}

/// Appends the UTF-8 encoding of `cp`.
void append_utf8(uint32_t cp, std::string & out) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

uint32_t decode_utf8(std::string_view bytes) {
    const unsigned char first = static_cast<unsigned char>(bytes[0]);
    if (first < 0x80) {
        return first;
    }
    size_t expected = 0;
    uint32_t cp = 0;
    if ((first & 0xE0) == 0xC0) {
        expected = 2;
        cp = first & 0x1Fu;
    } else if ((first & 0xF0) == 0xE0) {
        expected = 3;
        cp = first & 0x0Fu;
    } else if ((first & 0xF8) == 0xF0) {
        expected = 4;
        cp = first & 0x07u;
    } else {
        return 0xFFFD;
    }
    if (bytes.size() != expected) {
        return 0xFFFD;
    }
    for (size_t i = 1; i < expected; ++i) {
        const unsigned char byte = static_cast<unsigned char>(bytes[i]);
        if ((byte & 0xC0) != 0x80) {
            return 0xFFFD;
        }
        cp = (cp << 6) | (byte & 0x3Fu);
    }
    return cp;
}

/// Strips trailing spaces and tabs.
///
/// `phonemizer` appends a separator space that Piper's own clause-based bridge
/// does not, and that one space becomes a `(space_id, PAD)` pair — two extra
/// ids on every utterance.
std::string_view rstrip(std::string_view text) {
    size_t end = text.size();
    while (end > 0 && (text[end - 1] == ' ' || text[end - 1] == '\t' ||
                       text[end - 1] == '\n' || text[end - 1] == '\r')) {
        --end;
    }
    return text.substr(0, end);
}

} // namespace

// ---------------------------------------------------------------------------
// NfdTable
// ---------------------------------------------------------------------------

bool NfdTable::validate(std::string & error) const {
    if (count == 0) {
        if (n_values != 0) {
            error = "NFD table has decomposition values but no code points";
            return false;
        }
        return true;
    }
    if (codepoints == nullptr || offsets == nullptr || values == nullptr) {
        error = "NFD table is missing one of its three arrays";
        return false;
    }
    // Checked before anything indexes `offsets`: every bound below, including
    // `offsets[count]`, is only in range once this holds.
    if (n_offsets != count + 1) {
        error = "NFD table has " + std::to_string(n_offsets) + " offsets for " +
                std::to_string(count) + " code points; it needs one more than the count";
        return false;
    }
    if (offsets[0] != 0) {
        error = "NFD table offsets must start at 0";
        return false;
    }
    if (offsets[count] != n_values) {
        error = "NFD table's last offset does not match the value count";
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (offsets[i + 1] <= offsets[i]) {
            error = "NFD table offsets must be strictly increasing";
            return false;
        }
        if (offsets[i + 1] > n_values) {
            error = "NFD table offset points past the value array";
            return false;
        }
        if (i > 0 && codepoints[i] <= codepoints[i - 1]) {
            error = "NFD table code points must be sorted and unique";
            return false;
        }
        if (codepoints[i] > 0x10FFFF) {
            error = "NFD table contains a code point above U+10FFFF";
            return false;
        }
    }
    for (size_t i = 0; i < n_values; ++i) {
        if (values[i] > 0x10FFFF) {
            error = "NFD table decomposes to a code point above U+10FFFF";
            return false;
        }
    }
    if (ccc_count > 0) {
        if (ccc_codepoints == nullptr || ccc_classes == nullptr) {
            error = "NFD combining-class table is missing one of its arrays";
            return false;
        }
        if (n_ccc_classes != ccc_count) {
            error = "NFD combining-class table has " + std::to_string(n_ccc_classes) +
                    " classes for " + std::to_string(ccc_count) + " code points";
            return false;
        }
        for (size_t i = 0; i < ccc_count; ++i) {
            if (i > 0 && ccc_codepoints[i] <= ccc_codepoints[i - 1]) {
                error = "NFD combining-class code points must be sorted and unique";
                return false;
            }
            if (ccc_classes[i] == 0 || ccc_classes[i] > 255) {
                error = "NFD combining class must be in 1..255";
                return false;
            }
        }
    }
    return true;
}

uint32_t NfdTable::combining_class(uint32_t cp) const {
    const ptrdiff_t index = sorted_index(ccc_codepoints, ccc_count, cp);
    return index < 0 ? 0u : ccc_classes[static_cast<size_t>(index)];
}

void NfdTable::decompose(uint32_t cp, std::vector<uint32_t> & out) const {
    const ptrdiff_t index = sorted_index(codepoints, count, cp);
    if (index < 0) {
        out.push_back(cp);
        return;
    }
    const uint32_t start = offsets[static_cast<size_t>(index)];
    const uint32_t end = offsets[static_cast<size_t>(index) + 1];
    // The table already holds the *full* decomposition, so no recursion.
    out.insert(out.end(), values + start, values + end);
}

bool NfdTable::normalize(std::string_view text, std::vector<uint32_t> & out,
                         std::string & error) const {
    out.clear();
    size_t offset = 0;
    std::string_view ch;
    while (utf8_next(text, offset, ch)) {
        decompose(decode_utf8(ch), out);
    }
    if (offset != text.size()) {
        error = "phoneme string is not valid UTF-8";
        return false;
    }

    // Canonical ordering: within each maximal run of non-starters, sort by
    // combining class, stably. A plain std::stable_sort over the run is
    // equivalent to the canonical-ordering algorithm's bubble pass.
    size_t i = 0;
    while (i < out.size()) {
        if (combining_class(out[i]) == 0) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < out.size() && combining_class(out[j]) != 0) {
            ++j;
        }
        std::stable_sort(out.begin() + static_cast<ptrdiff_t>(i),
                         out.begin() + static_cast<ptrdiff_t>(j),
                         [this](uint32_t a, uint32_t b) {
                             return combining_class(a) < combining_class(b);
                         });
        i = j;
    }
    return true;
}

// ---------------------------------------------------------------------------
// TokenTable
// ---------------------------------------------------------------------------

bool TokenTable::validate(std::string & error) const {
    if (to_id.empty()) {
        error = "voice has an empty token table";
        return false;
    }
    if (max_tokens != 0 && max_tokens < 3) {
        error = "voice max_tokens must leave room for BOS, one phoneme and EOS";
        return false;
    }

    uint32_t highest = 0;
    for (const auto & entry : to_id) {
        highest = std::max(highest, entry.second);
    }
    if (bos_id > highest || eos_id > highest) {
        error = "voice BOS/EOS id is outside its own token table";
        return false;
    }
    if (bos_id == eos_id) {
        error = "voice BOS and EOS ids are identical";
        return false;
    }
    if (pad_id >= 0) {
        if (static_cast<uint32_t>(pad_id) > highest) {
            error = "voice PAD id is outside its own token table";
            return false;
        }
        if (static_cast<uint32_t>(pad_id) == bos_id ||
            static_cast<uint32_t>(pad_id) == eos_id) {
            error = "voice PAD id collides with BOS or EOS";
            return false;
        }
    }
    if (fallback_id >= 0 && static_cast<uint32_t>(fallback_id) > highest) {
        error = "voice fallback id is outside its own token table";
        return false;
    }
    return true;
}

const uint32_t * TokenTable::find(std::string_view symbol) const {
    const auto it = to_id.find(std::string(symbol));
    return it == to_id.end() ? nullptr : &it->second;
}

bool TokenTable::is_special(std::string_view symbol) const {
    return special_symbols.find(std::string(symbol)) != special_symbols.end();
}

bool TokenTable::is_framing_id(uint32_t id) const {
    return id == bos_id || id == eos_id ||
           (pad_id >= 0 && id == static_cast<uint32_t>(pad_id));
}

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------

namespace {

bool over_budget(const TokenTable & table, size_t n_ids, std::string & error) {
    if (table.max_tokens != 0 && n_ids > table.max_tokens) {
        error = "phoneme sequence yields " + std::to_string(n_ids) +
                " token ids including framing; this voice's limit is " +
                std::to_string(table.max_tokens);
        return true;
    }
    return false;
}

} // namespace

bool tokenize_piper(std::string_view phonemes, const TokenTable & table,
                    const NfdTable & nfd, std::vector<uint32_t> & ids,
                    std::string & error) {
    ids.clear();
    if (table.pad_id < 0) {
        error = "the Piper framing needs a PAD id, but this voice declares none";
        return false;
    }

    std::vector<uint32_t> decomposed;
    if (!nfd.normalize(rstrip(phonemes), decomposed, error)) {
        return false;
    }

    const uint32_t pad = static_cast<uint32_t>(table.pad_id);
    // Upper bound: two ids per code point plus three framing ids.
    ids.reserve(decomposed.size() * 2 + 3);
    ids.push_back(table.bos_id);
    ids.push_back(pad);

    std::string symbol;
    size_t kept = 0;
    for (uint32_t cp : decomposed) {
        symbol.clear();
        append_utf8(cp, symbol);
        const uint32_t * id = table.find(symbol);
        // Piper skips a phoneme that is not in the map, with a warning. Doing
        // the same keeps a stray character from shifting every later id. The
        // id check is what actually stops a sentinel being forged from text;
        // `is_special` only covers the symbols the voice names explicitly.
        if (id == nullptr || table.is_special(symbol) || table.is_framing_id(*id)) {
            continue;
        }
        ids.push_back(*id);
        ids.push_back(pad);
        ++kept;
    }

    if (kept == 0) {
        ids.clear();
        error = "phonemization produced no symbol present in this voice's table";
        return false;
    }
    ids.push_back(table.eos_id);

    if (over_budget(table, ids.size(), error)) {
        ids.clear();
        return false;
    }
    return true;
}

bool tokenize_misaki(std::string_view phonemes, const TokenTable & table,
                     std::vector<uint32_t> & ids, std::string & error) {
    ids.clear();
    ids.reserve(phonemes.size() / 2 + 2);
    ids.push_back(table.bos_id);

    size_t offset = 0;
    size_t kept = 0;
    std::string_view ch;
    while (utf8_next(phonemes, offset, ch)) {
        const uint32_t * id = table.find(ch);
        // Unknown symbols are dropped, and so is anything that would land on a
        // sentinel. Filtering on the *id* is what makes that guarantee hold:
        // this loop looks up one code point at a time, so a multi-code-point
        // entry in `special_symbols` could never match `is_special`.
        if (id == nullptr || table.is_special(ch) || table.is_framing_id(*id)) {
            continue;
        }
        ids.push_back(*id);
        ++kept;
    }
    if (offset != phonemes.size()) {
        ids.clear();
        error = "phoneme string is not valid UTF-8";
        return false;
    }
    if (kept == 0) {
        ids.clear();
        error = "phonemization produced no symbol present in this voice's vocabulary";
        return false;
    }
    ids.push_back(table.eos_id);

    if (over_budget(table, ids.size(), error)) {
        ids.clear();
        return false;
    }
    return true;
}

bool clamp_ids_to_vocab(const std::vector<uint32_t> & ids, uint32_t vocab_size,
                        const TokenTable & table, const char * component,
                        std::vector<uint32_t> & out, std::string & error) {
    if (vocab_size == 0) {
        error = std::string("the ") + component + " component declares a zero vocabulary";
        return false;
    }
    out = ids;
    for (uint32_t & id : out) {
        if (id < vocab_size) {
            continue;
        }
        if (table.fallback_id < 0) {
            out.clear();
            error = "token id " + std::to_string(id) + " is outside the " +
                    component + " vocabulary of " + std::to_string(vocab_size) +
                    ", and this voice forbids a fallback";
            return false;
        }
        const uint32_t fallback = static_cast<uint32_t>(table.fallback_id);
        if (fallback >= vocab_size) {
            out.clear();
            error = "the fallback id " + std::to_string(fallback) +
                    " is itself outside the " + component + " vocabulary of " +
                    std::to_string(vocab_size);
            return false;
        }
        id = fallback;
    }
    return true;
}

} // namespace kokopop::sano
