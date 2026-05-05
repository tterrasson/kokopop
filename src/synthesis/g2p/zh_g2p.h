#pragma once
// Mandarin Chinese G2P (Grapheme-to-Phoneme)
//
// Pipeline:
//   text → cn2an(digit→chinese) → map_punctuation() → split_zh_non_zh()
//     → for each Chinese segment: char → pinyin lookup → pinyin_to_ipa
//     → for each non-Chinese segment: keep as-is
//     → strip combining marks → join → phonemes
//
// Ported from misaki.zh.ZHG2P (legacy mode, no jieba).
// Character-by-character processing — pypinyin works per-character.

#include <string>

namespace kokopop::g2p::zh {

/// Convert Chinese text to Kokoro-compatible IPA phonemes.
///
/// @param text  Input text (UTF-8, may contain mixed scripts)
/// @param[out] phonemes  Output phoneme string (Kokoro IPA format with tone markers)
/// @param[out] error  Error message if conversion fails
/// @return true on success
bool g2p_chinese(const std::string & text, std::string & phonemes, std::string & error);

/// Pinyin TONE3 → Kokoro IPA (single syllable).
/// E.g. "zhi3" → "ꭧɨ↓", "hao4" → "xau̯↘"
std::string pinyin_to_ipa(std::string_view pinyin_tone3);

} // namespace kokopop::g2p::zh
