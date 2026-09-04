#pragma once

#include <string>

namespace kokopop {

/// Apply misaki's phoneme normalisation rules to raw espeak output.
///
/// `normalization_lang` selects the rule set: 'a' American English, 'b'
/// British English, 'f' French (keeps the combining tilde), anything else the
/// generic non-English rules.
std::string normalize_espeak_phonemes(std::string phonemes, char normalization_lang);

/// Text -> misaki-normalised IPA phonemes.
///
/// Both the espeak voice and the normalisation language are explicit: deriving
/// them from a voice *name* only works for Kokoro's naming convention, and
/// would mis-route a sanoTTS voice such as `heart` to Hindi. Architectures pass
/// what their voice descriptor says (see ModelArch::phonemize).
///
/// `normalization_lang == 'z'` routes to the pure-C++ Mandarin G2P instead of
/// espeak; `espeak_voice` is then unused.
bool phonemize_text(const std::string & text,
                    const std::string & espeak_voice,
                    char normalization_lang,
                    std::string & phonemes,
                    std::string & error);

/// Raw espeak-ng phonemization, with no normalisation of any kind.
///
/// Selects `espeak_voice`, then loops `espeak_TextToPhonemes` over the clauses
/// espeak cuts the input into, joining them with a single space. Punctuation is
/// *not* preserved: espeak drops it. Callers that need it in the output put it
/// back themselves, which is what both misaki-style contracts do.
///
/// `phoneme_mode` is espeak's own bitfield: `espeakPHONEMES_IPA`, optionally
/// `espeakPHONEMES_TIE` with the tie character in bits 8-23. The tie matters —
/// it is what distinguishes a tied diphthong from two adjacent phonemes, and
/// misaki's rewrite table is written against the tied form.
bool espeak_phonemize_raw(const std::string & text,
                          const std::string & espeak_voice,
                          int phoneme_mode,
                          std::string & phonemes,
                          std::string & error);

} // namespace kokopop
