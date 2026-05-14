#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace kokopop {

// ---------------------------------------------------------------------------
// Boundary types (linguistic frontier strength)
// ---------------------------------------------------------------------------
enum class Boundary {
    None = 0,
    ClauseWeak,      // comma, em-dash, parenthetical
    ClauseStrong,    // semicolon, colon
    Sentence,        // . ! ? …
    Newline,         // single \n
    Paragraph        // double \n
};

// ---------------------------------------------------------------------------
// Chunking configuration
// ---------------------------------------------------------------------------
struct ChunkConfig {
    // Token budget
    int target_min_tokens = 80;
    int target_max_tokens = 180;
    int soft_max_tokens = 260;
    int hard_max_tokens = 510;

    // First chunk policy (reduce TTFB for interactive mode)
    int first_chunk_target_max_tokens = 100;
    bool allow_short_first_chunk = true;

    // Pause durations (ms) per boundary type
    int comma_pause_ms = 80;
    int sentence_pause_ms = 180;
    int paragraph_pause_ms = 400;

    // Crossfade between chunks (ms)
    int crossfade_ms = 25;

    // Silence trimming
    bool trim_silence = true;
    int max_silence_trim_ms = 120;
};

/// Default "interactive" config — fast first audio, smaller chunks
ChunkConfig make_interactive_config();

/// Default "long-form" config — larger chunks, better prosody
ChunkConfig make_long_form_config();

/// "Ultra-fast" config — minimal TTFB, very small first chunk
ChunkConfig make_ultra_fast_config();

/// Merge `overrides` into `base`: only non-default fields in overrides
/// replace the corresponding fields in base.  This lets a client send a
/// partial ChunkConfig that patches the preset.
ChunkConfig merge_chunk_config(ChunkConfig base, ChunkConfig overrides);

// ---------------------------------------------------------------------------
// Unit — a text fragment with phonemes and tokens (pre-chunking)
// ---------------------------------------------------------------------------
struct Unit {
    std::string text;
    std::string phonemes;
    std::vector<uint32_t> tokens;
    int n_tokens = 0;
    Boundary boundary_after = Boundary::None;
};

// ---------------------------------------------------------------------------
// Chunk — a group of units ready for inference
// ---------------------------------------------------------------------------
struct Chunk {
    std::string text;
    std::string phonemes;
    std::vector<uint32_t> tokens;
    int n_tokens = 0;
    Boundary boundary_after = Boundary::None;
    bool is_first = false;
    bool is_last = false;
};

// ---------------------------------------------------------------------------
// Main pipeline: text → chunks ready for inference
//
// This function:
//   1. Normalizes the text (unicode, whitespace, protections)
//   2. Splits into candidate units (paragraphs → sentences → clauses)
//   3. Phonemizes each unit and tokenizes
//   4. Force-splits oversized units (> hard_max_tokens)
//   5. Assembles units into chunks respecting token budgets
//   6. Rebalances tiny chunks
//
// The caller is expected to pass `tokenize_fn` which converts a phoneme
// string into token ids (using the model's tokenizer).
// ---------------------------------------------------------------------------
using TokenizeFn = std::function<bool(const std::string & phonemes,
                                       std::vector<uint32_t> & ids,
                                       std::string & error)>;

std::vector<Chunk> chunk_text(
    const std::string & text,
    const std::string & voice,
    const ChunkConfig & config,
    TokenizeFn tokenize_fn,
    std::string & error);

// ---------------------------------------------------------------------------
// Memory management helpers
// ---------------------------------------------------------------------------

/// Free the internal data of a chunk after it has been inferred.
/// This releases the phonemes string and tokens vector to reduce memory
/// usage during long streaming sessions.  The n_tokens and boundary_after
/// fields are preserved for diagnostic purposes.
///
/// Call this AFTER infer_chunk() has completed for the given chunk.
void clear_chunk_data(Chunk & chunk);

// ---------------------------------------------------------------------------
// Boundary helpers
// ---------------------------------------------------------------------------
bool is_strong_boundary(Boundary b);
bool is_reasonable_boundary(Boundary b);

} // namespace kokopop
