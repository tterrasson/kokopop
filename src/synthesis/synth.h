#pragma once

#include "model/model.h"
#include "inference/kokoro.h"

#include <string>

namespace kokopop {

bool synthesize_phonemes(
    Model & model, const std::string & phonemes,
    const std::string & voice, float speed,
    kokopop_audio & out, std::string & error);

} // namespace kokopop

