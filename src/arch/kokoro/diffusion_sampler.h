#pragma once

#include "inference/kokoro.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kokopop {

bool apply_diffusion_style_options(
    Model & model,
    const KokoroDiffusionOptions * options,
    std::vector<float> & style,
    const std::vector<float> & embedding,
    int64_t n_tokens,
    int64_t embedding_dim,
    std::string & error);

} // namespace kokopop
