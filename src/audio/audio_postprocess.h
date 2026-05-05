#pragma once

#include "chunker.h"

#include <cstddef>
#include <vector>

namespace kokopop {

// ---------------------------------------------------------------------------
// Audio post-processing per chunk
//
// Operations:
//   - Trim leading/trailing silence
//   - Append silence for boundary pauses
//   - Crossfade between chunks
// ---------------------------------------------------------------------------

/// Apply linear crossfade between the tail of prev and the head of next.
/// Returns a new vector containing the merged audio.
/// If either buffer is too short for crossfade, simple concatenation is used.
std::vector<float> apply_crossfade(
    const std::vector<float> & prev,
    const std::vector<float> & next,
    int crossfade_ms,
    int sample_rate);

/// Apply crossfade only if the boundary is weak (no crossfade on sentence ends)
std::vector<float> apply_crossfade_smart(
    const std::vector<float> & prev,
    const std::vector<float> & next,
    Boundary boundary,
    int crossfade_ms,
    int sample_rate);

/// Trim leading silence from audio (up to max_ms)
std::vector<float> trim_leading_silence(
    const std::vector<float> & audio,
    int max_ms,
    int sample_rate);

/// Trim trailing silence from audio (up to max_ms)
std::vector<float> trim_trailing_silence(
    const std::vector<float> & audio,
    int max_ms,
    int sample_rate);

/// Append silence (silence = zeros) for the given duration in ms
void append_silence(
    std::vector<float> & audio,
    int ms,
    int sample_rate);

/// Calculate pause duration in ms for a boundary type
int pause_for_boundary(Boundary boundary, const ChunkConfig & config);

/// Full post-processing pipeline for a single chunk
/// - Trim silence
/// - Add boundary pause
std::vector<float> postprocess_chunk_audio(
    const std::vector<float> & audio,
    const Chunk & chunk,
    int chunk_index,
    int total_chunks,
    const ChunkConfig & config,
    int sample_rate);

} // namespace kokopop
