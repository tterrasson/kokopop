#pragma once

// sanoTTS text -> phonemes.
//
// Two contracts, neither of them kokopop's Kokoro path:
//
//   misaki (heart, heartnano)
//       espeak-ng with `tie='^'`, punctuation hidden from espeak and restored
//       afterwards, then misaki's E2M rewrite applied to the whole line. The
//       tie is load-bearing: `ə^l` is a tied syllabic-l and rewrites to `ᵊl`,
//       while an untied `əl` is two phonemes and must stay as it is. Without
//       the tie the two are indistinguishable and half the vowels come out
//       wrong.
//
//   piper (amy, kristin, hfc, vi, id, ...)
//       espeak-ng with no tie, Piper's own punctuation set preserved, and no
//       normalisation at all — the tokenizer maps raw IPA code points.
//
// The reference implementations are `phonemizer` (GPL-3.0) and `misaki`
// (Apache-2.0), driven by `pypkg/sanotts/frontend.py` (MIT). Nothing here is a
// port of the GPL code: the punctuation handling is written from its observable
// behaviour and gated against the reference output in
// tests/data/sanotts_g2p_corpus.json, which
// tools/gen_sanotts_g2p_corpus.py regenerates. misaki's rewrite table is
// reproduced from misaki itself, which is Apache-2.0.

#include <string>
#include <string_view>

namespace kokopop::sano {

/// misaki's E2M rewrite plus its American-English tail, applied to a whole
/// phonemized line.
///
/// Input is espeak IPA *with* the `^` tie still present; output is the
/// 62-symbol misaki alphabet. The rewrites are sequential whole-string
/// replacements in a fixed order — not a single longest-match pass — because
/// later rules deliberately see the output of earlier ones.
std::string normalize_misaki_e2m(std::string phonemes);

/// Text -> misaki-normalised IPA for a sanoTTS vocos voice.
bool phonemize_misaki_sanotts(const std::string & text,
                              const std::string & espeak_voice,
                              std::string & phonemes,
                              std::string & error);

/// Text -> raw espeak IPA for a sanoTTS piperlite voice.
///
/// No normalisation, Piper's punctuation set (`!'(),-.:;?"`) preserved, and the
/// trailing separator space that the reference strips is stripped here too.
bool phonemize_piper_sanotts(const std::string & text,
                             const std::string & espeak_voice,
                             std::string & phonemes,
                             std::string & error);

// ---------------------------------------------------------------------------
// Exposed for testing
// ---------------------------------------------------------------------------

/// The punctuation marks each contract hides from espeak.
///
/// misaki uses phonemizer's default set; Piper passes its own, which is exactly
/// the ASCII punctuation present in its phoneme tables.
extern const char * const MISAKI_PUNCTUATION_MARKS;
extern const char * const PIPER_PUNCTUATION_MARKS;

/// Phonemize `text` while keeping its punctuation.
///
/// espeak silently drops punctuation, so it is cut out first, each remaining
/// fragment is phonemized on its own, and the marks are spliced back at the
/// same positions. Phonemizing the fragments separately is not an
/// optimisation — it is what changes espeak's clause context, and therefore its
/// stress assignment, to match the reference.
bool phonemize_preserving_punctuation(const std::string & text,
                                      const std::string & espeak_voice,
                                      int phoneme_mode,
                                      std::string_view marks,
                                      std::string & phonemes,
                                      std::string & error);

} // namespace kokopop::sano
