#pragma once

#include "model/model.h"
#include "arch/kokoro/kokoro.h"
#include "synthesis/chunker/chunker.h"

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

/// Synthesize one chunk from the ids the chunker already produced.
///
/// The streaming paths use this rather than `synthesize_phonemes()`: the chunk
/// carries the sequence that was budgeted, and re-tokenizing its phoneme string
/// here could yield a different one. `chunk.phonemes` is still read, for the
/// code-point count that picks Kokoro's style row.
bool synthesize_chunk(
    Model & model, const Chunk & chunk,
    const std::string & voice, float speed,
    const KokoroDiffusionOptions & diffusion,
    kokopop_audio & out, std::string & error);

} // namespace kokopop
