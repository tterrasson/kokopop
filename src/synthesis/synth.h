#pragma once

#include "model/model.h"
#include "inference/kokoro.h"

#include <string>

namespace kokopop {

bool synthesize_phonemes(
    Model & model, const std::string & phonemes,
    const std::string & voice, float speed,
    kokopop_audio & out, std::string & error);

bool synthesize_phonemes(
    Model & model, const std::string & phonemes,
    const std::string & voice, float speed,
    const KokoroDiffusionOptions & diffusion,
    kokopop_audio & out, std::string & error);

// Strip trailing punctuation (and trailing spaces) from a phoneme string.
// Used at intermediate chunk boundaries: keeping the punctuation tokens there
// destabilises the Kokoro model without improving prosody, since the chunk
// pause is already added by audio post-processing.
void trim_trailing_chunk_punctuation(std::string & phonemes);

} // namespace kokopop
