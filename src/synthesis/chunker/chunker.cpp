#include "chunker.h"

#include "synthesis/chunker/text_normalizer.h"
#include "synthesis/chunker/text_splitter.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace kokopop {

ChunkConfig make_adaptative_config() {
    ChunkConfig cfg;
    cfg.target_min_tokens = 28;
    cfg.target_max_tokens = 80;
    cfg.soft_max_tokens = 320;
    cfg.hard_max_tokens = 510;
    cfg.first_chunk_target_max_tokens = 0; // 0 = flush at the first boundary
    cfg.allow_short_first_chunk = true;
    cfg.comma_pause_ms = 70;
    cfg.sentence_pause_ms = 170;
    cfg.paragraph_pause_ms = 350;
    cfg.crossfade_ms = 25;
    cfg.trim_silence = true;
    cfg.max_silence_trim_ms = 120;
    return cfg;
}

ChunkConfig make_long_form_config() {
    ChunkConfig cfg;
    cfg.target_min_tokens = 140;
    cfg.target_max_tokens = 260;
    cfg.soft_max_tokens = 360;
    cfg.hard_max_tokens = 510;
    cfg.first_chunk_target_max_tokens = 220;
    cfg.allow_short_first_chunk = false;
    cfg.comma_pause_ms = 90;
    cfg.sentence_pause_ms = 220;
    cfg.paragraph_pause_ms = 450;
    cfg.crossfade_ms = 35;
    cfg.trim_silence = true;
    cfg.max_silence_trim_ms = 160;
    return cfg;
}

ChunkConfig merge_chunk_config(ChunkConfig base, ChunkConfig overrides) {
    ChunkConfig def; // all defaults
    if (overrides.target_min_tokens != def.target_min_tokens)
        base.target_min_tokens = overrides.target_min_tokens;
    if (overrides.target_max_tokens != def.target_max_tokens)
        base.target_max_tokens = overrides.target_max_tokens;
    if (overrides.soft_max_tokens != def.soft_max_tokens)
        base.soft_max_tokens = overrides.soft_max_tokens;
    if (overrides.hard_max_tokens != def.hard_max_tokens)
        base.hard_max_tokens = overrides.hard_max_tokens;
    if (overrides.first_chunk_target_max_tokens != def.first_chunk_target_max_tokens)
        base.first_chunk_target_max_tokens = overrides.first_chunk_target_max_tokens;
    if (overrides.allow_short_first_chunk != def.allow_short_first_chunk)
        base.allow_short_first_chunk = overrides.allow_short_first_chunk;
    if (overrides.target_overshoot_tokens != def.target_overshoot_tokens)
        base.target_overshoot_tokens = overrides.target_overshoot_tokens;
    if (overrides.comma_pause_ms != def.comma_pause_ms)
        base.comma_pause_ms = overrides.comma_pause_ms;
    if (overrides.sentence_pause_ms != def.sentence_pause_ms)
        base.sentence_pause_ms = overrides.sentence_pause_ms;
    if (overrides.paragraph_pause_ms != def.paragraph_pause_ms)
        base.paragraph_pause_ms = overrides.paragraph_pause_ms;
    if (overrides.crossfade_ms != def.crossfade_ms)
        base.crossfade_ms = overrides.crossfade_ms;
    if (overrides.trim_silence != def.trim_silence)
        base.trim_silence = overrides.trim_silence;
    if (overrides.max_silence_trim_ms != def.max_silence_trim_ms)
        base.max_silence_trim_ms = overrides.max_silence_trim_ms;
    return base;
}

// ---------------------------------------------------------------------------
// Boundary punctuation
// ---------------------------------------------------------------------------

namespace {

bool ends_with(const std::string & text, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return text.size() >= n && std::memcmp(text.data() + text.size() - n, suffix, n) == 0;
}

} // namespace

void trim_trailing_chunk_punctuation(std::string & phonemes) {
    static constexpr const char * multibyte[] = {
        "\xe2\x80\xa6", // ellipsis
        "\xe2\x80\x94", // em dash
    };
    for (;;) {
        while (!phonemes.empty() && phonemes.back() == ' ') {
            phonemes.pop_back();
        }
        if (phonemes.empty()) return;

        const char c = phonemes.back();
        if (c == ',' || c == '.' || c == ';' || c == ':' || c == '!' || c == '?') {
            phonemes.pop_back();
            continue;
        }
        bool trimmed = false;
        for (const char * s : multibyte) {
            if (ends_with(phonemes, s)) {
                phonemes.erase(phonemes.size() - std::strlen(s));
                trimmed = true;
                break;
            }
        }
        if (!trimmed) return;
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

/// The two voice-bound closures, passed together to shorten signatures.
struct Frontend {
    const PhonemizeFn & phonemize;
    const TokenizeFn & tokenize;
};

/// Phonemize and tokenize one text fragment. `n_tokens` is the budget estimate
/// for this fragment alone, framing included.
bool make_unit(const std::string & text, const Frontend & fe, Unit & out, std::string & error) {
    std::string phonemes;
    std::vector<uint32_t> ids;
    if (!fe.phonemize(text, phonemes, error) || !fe.tokenize(phonemes, ids, error)) {
        return false;
    }
    out.text = text;
    out.phonemes = std::move(phonemes);
    out.n_tokens = static_cast<int>(ids.size());
    out.boundary_after = infer_boundary_type(text);
    return true;
}

std::string ensure_terminal_sentence_boundary(std::string text) {
    if (infer_boundary_type(text) == Boundary::None) {
        text.push_back('.');
    }
    return text;
}

void append_piece(Chunk & chunk, const std::string & phonemes, const std::string & text,
                  int n_tokens, Boundary boundary_after) {
    if (!chunk.phonemes.empty() && !phonemes.empty()) {
        chunk.phonemes.push_back(' ');
    }
    chunk.phonemes.append(phonemes);
    if (!chunk.text.empty() && !text.empty()) {
        chunk.text.push_back(' ');
    }
    chunk.text.append(text);
    chunk.n_tokens += n_tokens;
    chunk.boundary_after = boundary_after;
}

void append_unit(Chunk & chunk, const Unit & unit) {
    append_piece(chunk, unit.phonemes, unit.text, unit.n_tokens, unit.boundary_after);
}

void append_chunk(Chunk & chunk, const Chunk & other) {
    append_piece(chunk, other.phonemes, other.text, other.n_tokens, other.boundary_after);
}

/// Tokenize the assembled chunk once. `n_tokens` was a sum of per-unit
/// estimates, each with its own framing; after this call it is the size of the
/// sequence that is actually inferred. Intermediate chunks lose their trailing
/// punctuation first, the pause is re-added by audio post-processing.
bool finalize_chunk(Chunk & chunk, const TokenizeFn & tokenize, std::string & error) {
    if (!chunk.is_last) {
        std::string trimmed = chunk.phonemes;
        trim_trailing_chunk_punctuation(trimmed);
        if (!trimmed.empty()) {
            chunk.phonemes = std::move(trimmed);
        }
    }
    if (!tokenize(chunk.phonemes, chunk.tokens, error)) {
        return false;
    }
    chunk.n_tokens = static_cast<int>(chunk.tokens.size());
    return true;
}

/// Split a unit that exceeds `max_tokens`: on strong then weak punctuation
/// first, then by words. A single word above the budget is kept whole.
std::vector<Unit> split_oversized_unit(const std::string & text, int max_tokens,
                                       const Frontend & fe, std::string & error) {
    std::vector<Unit> result;

    auto parts = force_split_unit(text);
    if (parts.size() > 1) {
        for (const auto & part : parts) {
            Unit u;
            std::string part_error;
            if (make_unit(part, fe, u, part_error) && u.n_tokens <= max_tokens) {
                result.push_back(std::move(u));
                continue;
            }
            auto sub = split_oversized_unit(part, max_tokens, fe, error);
            result.insert(result.end(),
                          std::make_move_iterator(sub.begin()),
                          std::make_move_iterator(sub.end()));
        }
        return result;
    }

    // Word level: grow `current` one word at a time while it fits.
    std::string current;
    Unit current_unit;
    for (const auto & word : split_by_words(text)) {
        const std::string trial = current.empty() ? word : current + " " + word;
        Unit trial_unit;
        std::string trial_error;
        if (make_unit(trial, fe, trial_unit, trial_error) && trial_unit.n_tokens <= max_tokens) {
            current = trial;
            current_unit = std::move(trial_unit);
            continue;
        }
        if (!current.empty()) {
            result.push_back(std::move(current_unit));
            current.clear();
        }
        Unit word_unit;
        if (!make_unit(word, fe, word_unit, error)) {
            std::fprintf(stderr, "[chunker] skipping word (cannot tokenize): %s\n", error.c_str());
            continue;
        }
        if (word_unit.n_tokens > max_tokens) {
            result.push_back(std::move(word_unit));
        } else {
            current = word;
            current_unit = std::move(word_unit);
        }
    }
    if (!current.empty()) {
        result.push_back(std::move(current_unit));
    }
    return result;
}

/// Merge chunks below `target_min_tokens` into a neighbour when the pair stays
/// within `target_max_tokens`. The first chunk is exempt when it may be short.
std::vector<Chunk> rebalance_tiny_chunks(std::vector<Chunk> chunks, const ChunkConfig & config) {
    if (chunks.size() <= 1) return chunks;

    std::vector<Chunk> result;
    result.reserve(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
        Chunk & cur = chunks[i];
        const bool exempt = (i == 0 && config.allow_short_first_chunk);
        if (exempt || cur.n_tokens >= config.target_min_tokens) {
            result.push_back(std::move(cur));
            continue;
        }
        if (!result.empty() &&
            result.back().n_tokens + cur.n_tokens <= config.target_max_tokens) {
            append_chunk(result.back(), cur);
            continue;
        }
        if (i + 1 < chunks.size() &&
            cur.n_tokens + chunks[i + 1].n_tokens <= config.target_max_tokens) {
            append_chunk(cur, chunks[i + 1]);
            ++i;
        }
        result.push_back(std::move(cur));
    }
    return result;
}

} // namespace

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------

std::vector<Unit> prepare_chunk_units(
    const std::string & text,
    const ChunkConfig & config,
    const PhonemizeFn & phonemize_fn,
    const TokenizeFn & tokenize_fn,
    std::string & error) {
    const Frontend fe{phonemize_fn, tokenize_fn};

    const std::string normalized = ensure_terminal_sentence_boundary(normalize_text(text));

    std::vector<Unit> units;
    for (const auto & raw : split_into_candidate_units(normalized)) {
        std::vector<Unit> made;
        Unit u;
        std::string unit_error;
        if (make_unit(raw, fe, u, unit_error) && u.n_tokens <= config.target_max_tokens) {
            made.push_back(std::move(u));
        } else {
            made = split_oversized_unit(raw, config.target_max_tokens, fe, error);
            if (made.empty()) {
                std::fprintf(stderr, "[chunker] skipping unit (cannot split): %s\n",
                             unit_error.empty() ? error.c_str() : unit_error.c_str());
                continue;
            }
        }

        // The first unit bounds TTFB: cut it down to the first-chunk budget.
        const int first_max = config.first_chunk_target_max_tokens;
        if (units.empty() && config.allow_short_first_chunk && first_max > 0 &&
            made.front().n_tokens > first_max) {
            auto head = split_oversized_unit(made.front().text, first_max, fe, error);
            if (!head.empty()) {
                made.erase(made.begin());
                made.insert(made.begin(),
                            std::make_move_iterator(head.begin()),
                            std::make_move_iterator(head.end()));
            }
        }

        for (auto & unit : made) {
            if (unit.phonemes.empty()) {
                // Nothing to say; keep its boundary on the previous unit.
                if (!units.empty() &&
                    boundary_score(unit.boundary_after) > boundary_score(units.back().boundary_after)) {
                    units.back().boundary_after = unit.boundary_after;
                }
                continue;
            }
            units.push_back(std::move(unit));
        }
    }
    return units;
}

std::vector<Chunk> chunk_text(
    const std::string & text,
    const ChunkConfig & config,
    const PhonemizeFn & phonemize_fn,
    const TokenizeFn & tokenize_fn,
    std::string & error) {
    const std::vector<Unit> units =
        prepare_chunk_units(text, config, phonemize_fn, tokenize_fn, error);

    std::vector<Chunk> chunks;
    Chunk current;
    auto flush = [&]() {
        if (current.n_tokens > 0) {
            chunks.push_back(std::move(current));
        }
        current = Chunk{};
    };

    for (const Unit & unit : units) {
        const bool first = chunks.empty();
        const int max_this = first && config.allow_short_first_chunk
            ? config.first_chunk_target_max_tokens
            : config.target_max_tokens;
        const int trial = current.n_tokens + unit.n_tokens;

        if (current.n_tokens > 0) {
            const bool full = current.n_tokens >= config.target_min_tokens;
            const bool flush_before =
                trial > config.hard_max_tokens ||
                (full && (trial > max_this ||
                          unit.n_tokens > config.target_max_tokens ||
                          (trial > config.target_max_tokens &&
                           is_strong_boundary(current.boundary_after))));
            if (flush_before) flush();
        }

        append_unit(current, unit);

        bool flush_after = false;
        if (chunks.empty() && config.allow_short_first_chunk) {
            flush_after = current.n_tokens >= config.first_chunk_target_max_tokens ||
                          (current.n_tokens >= config.target_min_tokens &&
                           is_strong_boundary(current.boundary_after));
        } else {
            flush_after = (current.n_tokens >= config.target_max_tokens &&
                           is_reasonable_boundary(current.boundary_after)) ||
                          (current.n_tokens >= config.soft_max_tokens &&
                           current.boundary_after != Boundary::None) ||
                          current.n_tokens >= config.hard_max_tokens;
        }
        if (flush_after) flush();
    }
    flush();

    chunks = rebalance_tiny_chunks(std::move(chunks), config);
    if (chunks.empty()) return chunks;

    chunks.front().is_first = true;
    chunks.back().is_last = true;
    for (Chunk & chunk : chunks) {
        if (!finalize_chunk(chunk, tokenize_fn, error)) {
            return {};
        }
    }
    return chunks;
}

// ---------------------------------------------------------------------------
// Adaptative mode
// ---------------------------------------------------------------------------

int AdaptativeChunkController::target_tokens(size_t queued_requests) const {
    const double baseline = ms_per_token_ewma > 0.01 ? ms_per_token_ewma : 12.0;
    double adjusted_target_ms = target_generation_ms;
    if (queued_requests > 0) {
        adjusted_target_ms /= 1.0 + std::min<double>(2.0, queued_requests * 0.35);
    }
    const int estimate = static_cast<int>(adjusted_target_ms / baseline);
    const int cap = std::max(min_tokens, std::min(max_tokens, growth_max_tokens));
    // With a comfortable buffer, aim for the cap: on fast hardware the estimate
    // falls below min_tokens and would otherwise make the AIMD cap irrelevant.
    const int base = (lead_ms() >= comfort_lead_ms) ? cap : estimate;
    return std::max(min_tokens, std::min(cap, base));
}

void AdaptativeChunkController::observe(int n_tokens, double generation_ms, double audio_ms) {
    if (n_tokens <= 0 || generation_ms <= 0.0) return;
    const double sample = generation_ms / static_cast<double>(n_tokens);
    constexpr double alpha = 0.30;
    // Seed the EWMA at the default baseline: a single fast first chunk (warmup)
    // would otherwise make the next chunk too large.
    if (ms_per_token_ewma <= 0.0) ms_per_token_ewma = 12.0;
    ms_per_token_ewma = alpha * sample + (1.0 - alpha) * ms_per_token_ewma;

    cumulative_audio_ms += audio_ms;
    if (first_observed) {
        cumulative_gen_ms_after_first += generation_ms;
    } else {
        first_observed = true;
        first_gen_ms = generation_ms;
    }

    const double lead = lead_ms();
    const double rtf = audio_ms > 0.0 ? generation_ms / audio_ms : 1.0;

    // Shrink when the buffer is critically low or synthesis is losing ground,
    // whatever the current depth, so the cap drops before the buffer drains.
    const bool buffer_critical = lead < safety_floor_ms;
    const bool losing_ground = rtf > 1.0;
    if (buffer_critical || losing_ground) {
        const int shrunk = static_cast<int>(growth_max_tokens * shrink_factor);
        growth_max_tokens = std::max(min_tokens, shrunk);
    } else if (lead >= comfort_lead_ms && rtf < 0.9) {
        growth_max_tokens = std::min(max_tokens, growth_max_tokens + grow_step_tokens);
    } else if (lead >= comfort_lead_ms) {
        growth_max_tokens = std::min(max_tokens, growth_max_tokens + grow_step_tokens / 2);
    }
    // Otherwise (between floor and comfort, rtf <= 1): hold steady.
}

Chunk build_adaptative_chunk(
    const std::vector<Unit> & units,
    size_t & next_unit,
    const ChunkConfig & config,
    int target_tokens,
    bool is_first,
    const TokenizeFn & tokenize_fn,
    std::string & error) {
    Chunk chunk;
    if (next_unit >= units.size()) return chunk;

    const int target = std::max(config.target_min_tokens,
                                std::min(config.hard_max_tokens, target_tokens));
    const int overshoot = std::max(0, config.target_overshoot_tokens);

    while (next_unit < units.size()) {
        const Unit & unit = units[next_unit];
        if (chunk.n_tokens > 0 && chunk.n_tokens + unit.n_tokens > config.hard_max_tokens) {
            break;
        }
        append_unit(chunk, unit);
        ++next_unit;

        const Boundary b = chunk.boundary_after;
        if (chunk.n_tokens >= config.hard_max_tokens) break;

        if (is_first) {
            // Flush at the first boundary, or once the first-chunk budget is met.
            if (b != Boundary::None &&
                (config.first_chunk_target_max_tokens <= 0 ||
                 chunk.n_tokens >= config.first_chunk_target_max_tokens)) {
                break;
            }
            continue;
        }

        const bool past_min = chunk.n_tokens >= config.target_min_tokens;
        if (past_min && is_strong_boundary(b)) break;

        // Past target: break on a sentence or stronger, otherwise keep going up
        // to `overshoot` tokens in the hope of a stronger boundary.
        if (chunk.n_tokens >= target && b != Boundary::None) {
            const bool strong = boundary_score(b) >= boundary_score(Boundary::Sentence);
            if (strong || chunk.n_tokens >= target + overshoot) break;
        }

        // A large controller target may exceed the static target_max_tokens.
        if (chunk.n_tokens >= std::max(config.target_max_tokens, target)) break;
        if (chunk.n_tokens >= config.soft_max_tokens && b != Boundary::None) break;
    }

    // Absorb a tiny tail rather than emit an undersized final chunk.
    if (!is_first && chunk.n_tokens > 0 && next_unit < units.size()) {
        int remaining = 0;
        for (size_t i = next_unit; i < units.size(); ++i) {
            remaining += units[i].n_tokens;
        }
        if (remaining > 0 && remaining < config.target_min_tokens &&
            chunk.n_tokens + remaining <= config.hard_max_tokens) {
            for (; next_unit < units.size(); ++next_unit) {
                append_unit(chunk, units[next_unit]);
            }
        }
    }

    chunk.is_first = is_first;
    chunk.is_last = next_unit >= units.size();
    if (!finalize_chunk(chunk, tokenize_fn, error)) {
        return Chunk{};
    }
    return chunk;
}

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

void clear_chunk_data(Chunk & chunk) {
    // Swap with empties to release the heap blocks; keep n_tokens and
    // boundary_after for diagnostics.
    std::string().swap(chunk.phonemes);
    std::vector<uint32_t>().swap(chunk.tokens);
}

} // namespace kokopop
