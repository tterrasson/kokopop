#pragma once

// sanoTTS text tokenizers.
//
// The two voice families do not share a tokenizer, and the difference is not
// cosmetic — it is the difference between correct audio and confident noise:
//
//   piper (amy, kristin, hfc, vi, id, ...)
//       raw espeak IPA, right-stripped, decomposed to NFD, one id per code
//       point, framed as [BOS, PAD] + (id, PAD) per phoneme + [EOS]. An id
//       outside a component's trained vocabulary falls back to schwa.
//
//   misaki (heart, heartnano)
//       misaki-normalised IPA over a 62-symbol character vocabulary, framed as
//       [BOS] + ids + [EOS] with nothing interleaved and *no* schwa fallback:
//       id 59 is the em dash in that vocabulary, so remapping to it would
//       silently insert punctuation.
//
// NFD is the awkward part. kokopop has no Unicode dependency and cannot know
// which precomposed characters exist, so the converter serialises the full
// canonical-decomposition table (plus the combining classes canonical ordering
// needs) into the GGUF, and this module reads it from there.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kokopop::sano {

/// The model's canonical-decomposition table, as flat views into GGUF metadata.
///
/// `codepoints` is sorted ascending, so lookups binary-search. Row `i`
/// decomposes to `values[offsets[i] .. offsets[i + 1])`; `offsets` therefore
/// has `count + 1` entries. A code point absent from `codepoints` decomposes
/// to itself.
///
/// Every array carries its own length. That is not redundancy: `validate()`
/// is the only thing standing between a hostile GGUF and an out-of-bounds
/// read, and it cannot check that `offsets` really has `count + 1` entries
/// unless the loader tells it how many it found.
struct NfdTable {
    const uint32_t * codepoints = nullptr;   // count entries
    const uint32_t * offsets = nullptr;      // n_offsets == count + 1 entries
    const uint32_t * values = nullptr;       // n_values entries
    size_t count = 0;
    size_t n_offsets = 0;
    size_t n_values = 0;

    /// Code points with a non-zero canonical combining class, sorted, and
    /// their classes. Needed because NFD is decomposition *and* canonical
    /// ordering: two adjacent marks of different classes must be sorted, or
    /// they map to a different id sequence.
    const uint32_t * ccc_codepoints = nullptr;   // ccc_count entries
    const uint32_t * ccc_classes = nullptr;      // n_ccc_classes entries
    size_t ccc_count = 0;
    size_t n_ccc_classes = 0;

    /// True when the table is present. An empty table is valid and means "no
    /// code point decomposes", which is only correct for a model whose
    /// tokenizer never needs NFD.
    bool present() const { return count > 0; }

    /// Structural validation. A GGUF is untrusted input, and a truncated or
    /// non-monotonic offset array would read out of bounds. Call this before
    /// any other member: nothing else re-checks the invariants.
    bool validate(std::string & error) const;

    /// Canonical combining class of `cp`, 0 when it has none.
    uint32_t combining_class(uint32_t cp) const;

    /// Full canonical decomposition of `cp`, appended to `out`.
    void decompose(uint32_t cp, std::vector<uint32_t> & out) const;

    /// NFD of a UTF-8 string: decompose every code point, then canonically
    /// order each run of combining marks. Returns false on invalid UTF-8.
    bool normalize(std::string_view text, std::vector<uint32_t> & out,
                   std::string & error) const;
};

/// One voice's symbol table and framing rules, read from the GGUF.
struct TokenTable {
    /// Symbol (UTF-8) to id. Owns its keys: the GGUF strings it was built from
    /// are not guaranteed to outlive the load.
    ///
    /// Heterogeneous lookup on unordered_map is C++20, so `find()` below builds
    /// a std::string from the view. Every symbol is a single code point, which
    /// is at most 4 bytes and therefore always inside the small-string buffer:
    /// no allocation on the tokenizing path.
    std::unordered_map<std::string, uint32_t> to_id;

    uint32_t bos_id = 1;
    uint32_t eos_id = 2;

    /// Id interleaved after every phoneme, or -1 when the framing has none.
    int32_t pad_id = -1;

    /// Id substituted for a symbol whose id is outside a component's trained
    /// vocabulary, or -1 when that must be an error instead.
    int32_t fallback_id = -1;

    /// Budget in *final* ids, framing and interleaved PAD included. 0 means
    /// unbounded.
    uint32_t max_tokens = 0;

    /// Framing symbols that must never be emitted from input text, even when
    /// the input happens to contain them.
    ///
    /// A set rather than a list: both tokenizers consult it once per code
    /// point, and nothing downstream depends on the declaration order.
    std::unordered_set<std::string> special_symbols;

    bool validate(std::string & error) const;
    const uint32_t * find(std::string_view symbol) const;
    bool is_special(std::string_view symbol) const;

    /// True when `id` is one of this voice's framing sentinels.
    ///
    /// `is_special()` can only reject a *symbol*, and both tokenizers look up
    /// one code point at a time, so a multi-code-point entry in
    /// `special_symbols` could never match there. This is the check that
    /// actually holds: whatever the input text says, no id emitted from it is
    /// allowed to be BOS, EOS or PAD.
    bool is_framing_id(uint32_t id) const;
};

/// Piper framing: `[BOS, PAD] + (id, PAD) per phoneme + [EOS]`.
///
/// `phonemes` is raw espeak IPA. Trailing whitespace is stripped (phonemizer
/// emits a trailing space that Piper does not), the rest is decomposed to NFD
/// and mapped one id per code point. A code point absent from the table is
/// skipped, matching Piper.
bool tokenize_piper(std::string_view phonemes, const TokenTable & table,
                    const NfdTable & nfd, std::vector<uint32_t> & ids,
                    std::string & error);

/// misaki framing: `[BOS] + ids + [EOS]`, no interleaving, no NFD.
///
/// Symbols absent from the vocabulary are dropped, as are declared special
/// symbols and anything that would resolve to a framing id, so no input text
/// can forge a sentinel.
bool tokenize_misaki(std::string_view phonemes, const TokenTable & table,
                     std::vector<uint32_t> & ids, std::string & error);

/// Clamp `ids` to `vocab_size`, substituting `table.fallback_id`.
///
/// The duration and acoustic components of a piperlite voice are trained with
/// smaller vocabularies than the shared code-point table, so an id can be
/// valid for the tokenizer and out of range for a component. Fails when the
/// table forbids a fallback.
bool clamp_ids_to_vocab(const std::vector<uint32_t> & ids, uint32_t vocab_size,
                        const TokenTable & table, const char * component,
                        std::vector<uint32_t> & out, std::string & error);

} // namespace kokopop::sano
