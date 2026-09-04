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
    ClauseWeak,      // comma, em dash, parenthetical
    ClauseStrong,    // semicolon, colon
    Sentence,        // . ! ? ...
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

    // Tokens allowed past the controller target while looking for a stronger
    // boundary. Still capped by target_max_tokens and hard_max_tokens.
    int target_overshoot_tokens = 24;

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

/// Default "adaptative" config: fast first audio, dynamic chunk sizing
ChunkConfig make_adaptative_config();

/// Default "long-form" config: larger chunks, better prosody
ChunkConfig make_long_form_config();

/// Merge `overrides` into `base`: only fields that differ from the ChunkConfig
/// defaults replace the base value, so a client can send a partial config.
ChunkConfig merge_chunk_config(ChunkConfig base, ChunkConfig overrides);

// ---------------------------------------------------------------------------
// Unit: a text fragment with its phonemes (pre-chunking)
// ---------------------------------------------------------------------------
struct Unit {
    std::string text;
    std::string phonemes;
    /// Token count of `phonemes` tokenized alone, framing included. Used as the
    /// budget estimate when assembling chunks.
    int n_tokens = 0;
    Boundary boundary_after = Boundary::None;
};

// ---------------------------------------------------------------------------
// Chunk: a group of units ready for inference
// ---------------------------------------------------------------------------
struct Chunk {
    std::string text;
    std::string phonemes;
    /// `phonemes` tokenized once, framing included. This is the exact sequence
    /// passed to inference.
    std::vector<uint32_t> tokens;
    /// `tokens.size()` once the chunk is built.
    int n_tokens = 0;
    Boundary boundary_after = Boundary::None;
    bool is_first = false;
    bool is_last = false;
};

// ---------------------------------------------------------------------------
// Main pipeline: text -> chunks ready for inference
//
//   1. Normalize the text
//   2. Split into candidate units (paragraphs -> sentences -> clauses)
//   3. Phonemize and tokenize each unit, splitting those above target_max_tokens
//   4. Assemble units into chunks within the token budget
//   5. Rebalance tiny chunks
//   6. Trim boundary punctuation and tokenize each chunk once
//
// Both closures are bound by the caller to the resolved voice: the phonemizer
// and tokenizer belong to the architecture, so the chunker has no voice
// parameter. Budgets are counted in final ids, framing included.
// ---------------------------------------------------------------------------
using PhonemizeFn = std::function<bool(const std::string & text,
                                       std::string & phonemes,
                                       std::string & error)>;
using TokenizeFn = std::function<bool(const std::string & phonemes,
                                       std::vector<uint32_t> & ids,
                                       std::string & error)>;

std::vector<Chunk> chunk_text(
    const std::string & text,
    const ChunkConfig & config,
    const PhonemizeFn & phonemize_fn,
    const TokenizeFn & tokenize_fn,
    std::string & error);

/// Steps 1 to 3 only: units for adaptative chunking, split on natural pauses.
std::vector<Unit> prepare_chunk_units(
    const std::string & text,
    const ChunkConfig & config,
    const PhonemizeFn & phonemize_fn,
    const TokenizeFn & tokenize_fn,
    std::string & error);

/// Build the next adaptative chunk from prepared units and advance `next_unit`.
/// The caller picks `target_tokens` from live timings. `tokenize_fn` must be
/// the closure the units were built with. Returns an empty chunk when the
/// units are exhausted, or on failure with `error` set.
Chunk build_adaptative_chunk(
    const std::vector<Unit> & units,
    size_t & next_unit,
    const ChunkConfig & config,
    int target_tokens,
    bool is_first,
    const TokenizeFn & tokenize_fn,
    std::string & error);

/// Per-request live controller for adaptative chunk sizing.
///
/// Closed-loop control on the audio buffered ahead of playback:
///   lead_ms = sum(audio_ms) - sum(generation_ms of chunks 2..N) - first_gen_ms
/// The first chunk's generation is the TTFB cost: playback starts after it.
///
/// `growth_max_tokens` is adjusted AIMD-style after every chunk:
///   lead < safety_floor_ms, or rtf > 1   : multiplicative shrink
///   lead >= comfort_lead_ms and rtf < 0.9: additive grow
///   lead >= comfort_lead_ms otherwise    : half-step grow
struct AdaptativeChunkController {
    // Cost model
    double target_generation_ms = 700.0;
    double ms_per_token_ewma = 0.0;
    int min_tokens = 32;
    int max_tokens = 220;            // absolute ceiling (soft_max_tokens)

    // Closed-loop state
    double cumulative_audio_ms = 0.0;
    double cumulative_gen_ms_after_first = 0.0;
    double first_gen_ms = 0.0;
    bool first_observed = false;
    int growth_max_tokens = 80;      // dynamic cap, starts at target_max_tokens

    // Tunables for the AIMD loop
    double safety_floor_ms = 250.0;
    double comfort_lead_ms = 1500.0;
    int grow_step_tokens = 16;
    double shrink_factor = 0.5;

    int target_tokens(size_t queued_requests = 0) const;
    void observe(int n_tokens, double generation_ms, double audio_ms);
    double lead_ms() const { return cumulative_audio_ms - cumulative_gen_ms_after_first - first_gen_ms; }
};

// ---------------------------------------------------------------------------
// Memory management helpers
// ---------------------------------------------------------------------------

/// Release a chunk's phonemes and tokens after inference. `n_tokens` and
/// `boundary_after` are kept for diagnostics.
void clear_chunk_data(Chunk & chunk);

// ---------------------------------------------------------------------------
// Boundary helpers
// ---------------------------------------------------------------------------

/// Strip trailing punctuation and spaces from a phoneme string. Applied to
/// intermediate chunks before tokenizing: boundary punctuation destabilises the
/// model, and the pause is re-added by audio post-processing.
void trim_trailing_chunk_punctuation(std::string & phonemes);

bool is_strong_boundary(Boundary b);
bool is_reasonable_boundary(Boundary b);

/// Boundary quality, higher = stronger prosodic break.
///   None=0, ClauseWeak=1, ClauseStrong=2, Newline=3, Sentence=4, Paragraph=5.
int boundary_score(Boundary b);

} // namespace kokopop
