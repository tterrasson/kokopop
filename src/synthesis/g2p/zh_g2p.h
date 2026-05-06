#pragma once
// Mandarin Chinese G2P (Grapheme-to-Phoneme)
//
// Pipeline:
//   text → number/date/time normalization → map_punctuation()
//     → longest-match lexical segmentation + char fallback
//     → contextual pinyin overrides + tone sandhi → pinyin_to_ipa
//     → strip combining marks → join → phonemes
//
// No runtime dependencies: segmentation and disambiguation use small embedded
// tables plus deterministic fallback to the generated pinyin dictionary.

#include <string>

namespace kokopop::g2p::zh {

/// Convert Chinese text to Kokoro-compatible IPA phonemes.
///
/// @param text  Input text (UTF-8, may contain mixed scripts)
/// @param[out] phonemes  Output phoneme string (Kokoro IPA format with tone markers)
/// @param[out] error  Error message if conversion fails
/// @return true on success
bool g2p_chinese(const std::string & text, std::string & phonemes, std::string & error);

/// Convert Chinese text to normalized TONE3 pinyin after lexical overrides and
/// tone sandhi. Non-Chinese text and punctuation are preserved for diagnostics.
bool g2p_chinese_to_pinyin(const std::string & text, std::string & pinyin, std::string & error);

/// Pinyin TONE3 → Kokoro IPA (single syllable).
/// E.g. "zhi3" → "ꭧɨ↓", "hao4" → "xau̯↘"
std::string pinyin_to_ipa(std::string_view pinyin_tone3);

} // namespace kokopop::g2p::zh
