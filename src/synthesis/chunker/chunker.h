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

    // First chunk policy (reduce TTFB for adaptative mode)
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

/// Default "adaptative" config — fast first audio, dynamic chunk sizing
ChunkConfig make_adaptative_config();

/// Default "long-form" config — larger chunks, better prosody
ChunkConfig make_long_form_config();

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

/// Prepare natural text units for adaptative chunking. The returned units are
/// already phonemized/tokenized and split on natural pauses where possible.
std::vector<Unit> prepare_chunk_units(
    const std::string & text,
    const std::string & voice,
    const ChunkConfig & config,
    TokenizeFn tokenize_fn,
    std::string & error);

/// Build the next adaptative chunk from prepared units. `next_unit` is advanced
/// past consumed units. The caller decides `target_tokens` from live timings.
Chunk build_adaptative_chunk(
    const std::vector<Unit> & units,
    size_t & next_unit,
    const ChunkConfig & config,
    int target_tokens,
    bool is_first);

/// Per-request live controller for adaptative chunk sizing.
///
/// Closed-loop control on the audio buffer ahead of playback (`lead_ms`):
///   lead_ms = Σ audio_ms_i  −  Σ generation_ms_i (chunks 2..N)
/// The first chunk's generation does not count: the client only starts
/// playing once it has been delivered.
///
/// A dynamic cap `growth_max_tokens` is adjusted AIMD-style after every chunk:
///   - lead_ms <  safety_floor_ms  → multiplicative shrink (×shrink_factor)
///   - lead_ms ≥  comfort_lead_ms  → additive grow (+grow_step_tokens)
///   - in between                  → half-speed grow (+grow_step_tokens/2)
struct AdaptativeChunkController {
    // Cost model
    double target_generation_ms = 700.0;
    double ms_per_token_ewma = 0.0;
    int min_tokens = 32;
    int max_tokens = 220;            // hard absolute ceiling (soft_max_tokens)

    // Closed-loop state
    double cumulative_audio_ms = 0.0;
    double cumulative_gen_ms_after_first = 0.0;
    bool first_observed = false;
    int growth_max_tokens = 80;      // dynamic cap, starts at target_max_tokens

    // Tunables for the AIMD loop
    double safety_floor_ms = 250.0;
    double comfort_lead_ms = 1500.0;
    int grow_step_tokens = 16;
    double shrink_factor = 0.5;

    int target_tokens(size_t queued_requests = 0) const;
    void observe(int n_tokens, double generation_ms, double audio_ms);
    double lead_ms() const { return cumulative_audio_ms - cumulative_gen_ms_after_first; }
};

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
