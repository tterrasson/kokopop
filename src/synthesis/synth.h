#pragma once

#include "model/model.h"
#include "arch/kokoro/kokoro.h"
#include "synthesis/chunker/chunker.h"

#include <cstdint>
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
///
/// `extras` carries the architecture-specific inputs the caller owns:
/// Kokoro's diffusion options, sanoTTS's noise seed, and `chunk_index`, the
/// chunk's position in the utterance, which is folded into the per-chunk seed
/// so two chunks with identical ids still decode differently.
/// `kokoro_style_len` is derived from `chunk.phonemes` here and any value the
/// caller set is ignored.
bool synthesize_chunk(
    Model & model, const Chunk & chunk,
    const std::string & voice, float speed,
    const SynthesisExtras & extras,
    kokopop_audio & out, std::string & error);

} // namespace kokopop
