#include "chunker.h"

#include "synthesis/chunker/text_normalizer.h"
#include "synthesis/chunker/text_splitter.h"

#include "synthesis/phonemizer.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kokopop {

ChunkConfig make_adaptative_config() {
    ChunkConfig cfg;
    cfg.target_min_tokens = 28;
    cfg.target_max_tokens = 80;
    cfg.soft_max_tokens = 320;
    cfg.hard_max_tokens = 510;
    cfg.first_chunk_target_max_tokens = 0; // 0 = flush at first boundary (default TTFB behaviour)
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
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

/// Phonemize and tokenize a single text fragment
bool make_unit(const std::string & text,
               const std::string & voice,
               TokenizeFn tokenize_fn,
               Unit & out,
               std::string & error) {
    std::string phonemes;
    if (!kokopop::phonemize_text(text, voice, phonemes, error)) {
        return false;
    }

    if (!tokenize_fn(phonemes, out.tokens, error)) {
        return false;
    }

    out.text = text;
    out.phonemes = std::move(phonemes);
    out.n_tokens = static_cast<int>(out.tokens.size());
    out.boundary_after = infer_boundary_type(text);
    return true;
}

std::string ensure_terminal_sentence_boundary(std::string text) {
    if (infer_boundary_type(text) != Boundary::None) {
        return text;
    }
    text.push_back('.');
    return text;
}

/// Phonemize and tokenize with a small cache to avoid repeated work.
/// When force_split_unit recurses, the same substring can be phonemized
/// multiple times; the cache intercepts identical text keys.
bool make_unit_cached(const std::string & text,
                      const std::string & voice,
                      TokenizeFn tokenize_fn,
                      Unit & out,
                      std::string & error) {
    static thread_local std::unordered_map<std::string,
                                            std::pair<std::string, std::vector<uint32_t>>>
        phoneme_cache;
    constexpr size_t cache_max_size = 256;

    // Key includes voice so that the same text phonemized for different
    // languages (e.g. ff_siwis → French, af_heart → English) is cached separately.
    std::string cache_key;
    cache_key.reserve(voice.size() + 1 + text.size());
    cache_key = voice;
    cache_key += '\0';
    cache_key += text;

    auto it = phoneme_cache.find(cache_key);
    if (it != phoneme_cache.end()) {
        out.text = text;
        out.phonemes = it->second.first;
        out.tokens = it->second.second;
        out.n_tokens = static_cast<int>(out.tokens.size());
        out.boundary_after = infer_boundary_type(text);
        return true;
    }

    std::string phonemes;
    if (!kokopop::phonemize_text(text, voice, phonemes, error)) {
        return false;
    }
    std::vector<uint32_t> tokens;
    if (!tokenize_fn(phonemes, tokens, error)) {
        return false;
    }

    // Store in cache (evict oldest by clearing when full)
    if (phoneme_cache.size() >= cache_max_size) {
        phoneme_cache.clear();
    }
    phoneme_cache.emplace(cache_key, std::make_pair(phonemes, tokens));

    out.text = text;
    out.phonemes = std::move(phonemes);
    out.tokens = std::move(tokens);
    out.n_tokens = static_cast<int>(out.tokens.size());
    out.boundary_after = infer_boundary_type(text);
    return true;
}

/// Force-split an oversized unit by trying progressively coarser boundaries
std::vector<Unit> force_split_unit(
    const std::string & text,
    const std::string & voice,
    const ChunkConfig & config,
    TokenizeFn tokenize_fn,
    std::string & error) {
    std::vector<Unit> result;

    // Try splitting on punctuation (uses cached make_unit)
    auto parts = ::kokopop::force_split_unit(text);
    if (parts.size() > 1) {
        for (const auto & part : parts) {
            Unit u;
            if (make_unit_cached(part, voice, tokenize_fn, u, error)) {
                if (u.n_tokens <= config.target_max_tokens) {
                    result.push_back(std::move(u));
                } else {
                    // Recursively split punctuation parts until they fit the
                    // active chunk target. This matters for adaptative, where
                    // waiting on a long comma-clause can dominate TTFB.
                    auto sub = force_split_unit(part, voice, config, tokenize_fn, error);
                    result.insert(result.end(),
                                  std::make_move_iterator(sub.begin()),
                                  std::make_move_iterator(sub.end()));
                }
            }
        }
        if (!result.empty()) return result;
    }

    // Try word-level splitting
    // Keep current_unit so we don't re-phonemize on flush
    auto words = ::kokopop::split_by_words(text);
    std::string current;
    Unit current_unit;
    for (const auto & word : words) {
        std::string trial = current + (current.empty() ? "" : " ") + word;
        Unit trial_unit;
        std::string trial_error;
        if (make_unit_cached(trial, voice, tokenize_fn, trial_unit, trial_error)) {
            if (trial_unit.n_tokens > config.target_max_tokens) {
                // Flush current — reuse already-phonemized unit if available
                if (!current.empty()) {
                    result.push_back(std::move(current_unit));
                    current = word;
                    if (!make_unit_cached(current, voice, tokenize_fn, current_unit, error)) {
                        current_unit = Unit{};
                    }
                } else {
                    // Single word too big — split by characters
                    result.push_back(std::move(trial_unit));
                    current.clear();
                    current_unit = Unit{};
                }
            } else {
                current = trial;
                current_unit = std::move(trial_unit);
            }
        } else {
            current = trial;
            current_unit = Unit{};
        }
    }
    // Flush remainder — reuse cached unit
    if (!current.empty()) {
        result.push_back(std::move(current_unit));
    }

    return result;
}

/// Rebalance tiny chunks by merging with neighbors
std::vector<Chunk> rebalance_tiny_chunks(
    std::vector<Chunk> chunks,
    const ChunkConfig & config) {
    if (chunks.size() <= 1) return chunks;

    // Pre-allocate buffers for merge targets to avoid repeated reallocs
    std::string phoneme_merge_buf;
    phoneme_merge_buf.reserve(2048);
    std::string text_merge_buf;
    text_merge_buf.reserve(2048);

    std::vector<Chunk> result;
    result.reserve(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
        // First chunk can be small in adaptative mode
        bool is_first = (i == 0);
        if (is_first && config.allow_short_first_chunk) {
            result.push_back(std::move(chunks[i]));
            continue;
        }

        if (chunks[i].n_tokens >= config.target_min_tokens) {
            result.push_back(std::move(chunks[i]));
            continue;
        }

        // Tiny chunk — try merging with previous
        if (!result.empty()) {
            Chunk & prev = result.back();
            if (prev.n_tokens + chunks[i].n_tokens <= config.target_max_tokens) {
                // Merge phonemes — avoid temporary " " + phonemes
                if (!prev.phonemes.empty() && !chunks[i].phonemes.empty()) {
                    prev.phonemes.push_back(' ');
                    prev.phonemes.append(chunks[i].phonemes);
                }
                prev.tokens.insert(prev.tokens.end(),
                                   std::make_move_iterator(chunks[i].tokens.begin()),
                                   std::make_move_iterator(chunks[i].tokens.end()));
                prev.n_tokens += chunks[i].n_tokens;
                prev.text.append(chunks[i].text);
                prev.boundary_after = chunks[i].boundary_after;
                continue;
            }
        }

        // Try merging with next
        if (i + 1 < chunks.size()) {
            Chunk merged = chunks[i];
            Chunk & next = chunks[i + 1];
            if (merged.n_tokens + next.n_tokens <= config.target_max_tokens) {
                if (!merged.phonemes.empty() && !next.phonemes.empty()) {
                    merged.phonemes.push_back(' ');
                    merged.phonemes.append(next.phonemes);
                }
                merged.tokens.insert(merged.tokens.end(),
                                     std::make_move_iterator(next.tokens.begin()),
                                     std::make_move_iterator(next.tokens.end()));
                merged.n_tokens += next.n_tokens;
                merged.text.append(next.text);
                merged.boundary_after = next.boundary_after;
                result.push_back(std::move(merged));
                ++i; // skip next
                continue;
            }
        }

        // Can't merge — keep as is
        result.push_back(std::move(chunks[i]));
    }
    return result;
}

} // namespace

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------

std::vector<Unit> prepare_chunk_units(
    const std::string & text,
    const std::string & voice,
    const ChunkConfig & config,
    TokenizeFn tokenize_fn,
    std::string & error) {

    // Step 1: Normalize text
    std::string normalized = ensure_terminal_sentence_boundary(normalize_text(text));

    // Step 2: Split into candidate units
    auto raw_units = split_into_candidate_units(normalized);

    // Step 3: Phonemize and tokenize each unit
    std::vector<Unit> units;
    for (const auto & raw : raw_units) {
        Unit u;
        std::string unit_error;
        if (!make_unit(raw, voice, tokenize_fn, u, unit_error)) {
            // Tokenization failed (likely exceeds context length) — try force-split
            // before giving up, so long sentences without punctuation still work.
            auto split = force_split_unit(raw, voice, config, tokenize_fn, error);
            if (units.empty() &&
                config.allow_short_first_chunk &&
                config.first_chunk_target_max_tokens > 0 &&
                !split.empty() &&
                split.front().n_tokens > config.first_chunk_target_max_tokens) {
                ChunkConfig first_cfg = config;
                first_cfg.target_min_tokens = 1;
                first_cfg.target_max_tokens = config.first_chunk_target_max_tokens;
                first_cfg.soft_max_tokens = config.first_chunk_target_max_tokens;
                first_cfg.hard_max_tokens = config.first_chunk_target_max_tokens;
                auto first_split = force_split_unit(
                    split.front().text, voice, first_cfg, tokenize_fn, error);
                if (!first_split.empty()) {
                    std::vector<Unit> refined;
                    refined.reserve(first_split.size() + split.size() - 1);
                    refined.insert(refined.end(),
                                   std::make_move_iterator(first_split.begin()),
                                   std::make_move_iterator(first_split.end()));
                    refined.insert(refined.end(),
                                   std::make_move_iterator(split.begin() + 1),
                                   std::make_move_iterator(split.end()));
                    split = std::move(refined);
                }
            }
            if (split.empty()) {
                std::fprintf(stderr, "[chunker] skipping unit (cannot split): %s\n",
                             unit_error.c_str());
                continue;
            }
            units.insert(units.end(),
                        std::make_move_iterator(split.begin()),
                        std::make_move_iterator(split.end()));
            continue;
        }

        // Step 4: Force-split oversized units.
        // In adaptative mode, target_max_tokens is small (≈80) so the
        // controller has fine-grained unit granularity. In long-form mode,
        // target_max is larger but still below soft_max, acting as a
        // reasonable memory and latency guard.
        if (u.n_tokens > config.target_max_tokens) {
            auto split = force_split_unit(raw, voice, config, tokenize_fn, error);
            units.insert(units.end(),
                        std::make_move_iterator(split.begin()),
                        std::make_move_iterator(split.end()));
        } else {
            units.push_back(std::move(u));
        }
    }

    return units;
}

std::vector<Chunk> chunk_text(
    const std::string & text,
    const std::string & voice,
    const ChunkConfig & config,
    TokenizeFn tokenize_fn,
    std::string & error) {

    std::vector<Unit> units = prepare_chunk_units(text, voice, config, tokenize_fn, error);

    // Step 5: Assemble units into chunks
    // Pre-compute upper bounds for reserve() — avoids repeated reallocations
    // as units are accumulated into each chunk.
    size_t remaining_phonemes = 0;
    size_t remaining_text = 0;
    for (const auto & unit : units) {
        remaining_phonemes += unit.phonemes.size() + 1; // +1 for space separator
        remaining_text += unit.text.size();
    }

    std::vector<Chunk> chunks;
    chunks.reserve(4); // typical lower bound; grows as needed
    Chunk current;

    auto flush = [&]() {
        if (!current.tokens.empty()) {
            current.is_last = false;
            chunks.push_back(std::move(current));
        }
    };

    auto reserve_chunk = [&]() {
        // Reserve a reasonable portion of the remaining budget.
        // Each chunk targets ~target_max_tokens, and phoneme strings are
        // roughly 2-3 chars per token, so we reserve proportionally.
        current.phonemes.reserve(remaining_phonemes);
        current.text.reserve(remaining_text);
    };

    for (const auto & unit : units) {
        int max_this = chunks.empty() && config.allow_short_first_chunk
            ? config.first_chunk_target_max_tokens
            : config.target_max_tokens;

        int trial = current.n_tokens + unit.n_tokens;

        // Check if we should flush before adding
        if (!current.tokens.empty()) {
            if (trial > config.hard_max_tokens) {
                flush();
            } else if (trial > max_this &&
                       current.n_tokens >= config.target_min_tokens) {
                flush();
            } else if (current.n_tokens >= config.target_min_tokens &&
                       is_strong_boundary(current.boundary_after) &&
                       trial > config.target_max_tokens) {
                flush();
            }
            if (!current.tokens.empty() &&
                unit.n_tokens > config.target_max_tokens &&
                current.n_tokens >= config.target_min_tokens) {
                flush();
            }
        }

        // Add unit to current chunk
        if (current.tokens.empty()) {
            current = Chunk{};
            current.is_first = chunks.empty();
            reserve_chunk();
        }

        if (!current.phonemes.empty() && !unit.phonemes.empty()) {
            current.phonemes.push_back(' ');
            current.phonemes.append(unit.phonemes);
        } else if (unit.phonemes.empty()) {
            // keep current phonemes
        } else {
            current.phonemes = unit.phonemes;
        }

        remaining_phonemes -= unit.phonemes.size() + 1;
        remaining_text -= unit.text.size();

        current.tokens.insert(current.tokens.end(),
                             std::make_move_iterator(unit.tokens.begin()),
                             std::make_move_iterator(unit.tokens.end()));
        current.n_tokens += unit.n_tokens;
        current.text.append(unit.text);
        current.boundary_after = unit.boundary_after;

        // Check if we should flush after adding
        bool is_first = chunks.empty();
        if (is_first && config.allow_short_first_chunk) {
            if (current.n_tokens >= config.first_chunk_target_max_tokens) {
                flush();
            } else if (current.n_tokens >= config.target_min_tokens &&
                       is_strong_boundary(current.boundary_after)) {
                flush();
            }
        } else {
            if (current.n_tokens >= config.target_max_tokens &&
                is_reasonable_boundary(current.boundary_after)) {
                flush();
            } else if (current.n_tokens >= config.soft_max_tokens &&
                       current.boundary_after != Boundary::None) {
                flush();
            } else if (current.n_tokens >= config.hard_max_tokens) {
                flush();
            }
        }
    }

    flush();

    // Step 6: Rebalance tiny chunks
    chunks = rebalance_tiny_chunks(std::move(chunks), config);

    // Set first/last flags
    if (!chunks.empty()) {
        chunks.front().is_first = true;
        chunks.back().is_last = true;
    }

    return chunks;
}

int AdaptativeChunkController::target_tokens(size_t queued_requests) const {
    const double baseline = ms_per_token_ewma > 0.01 ? ms_per_token_ewma : 12.0;
    double adjusted_target_ms = target_generation_ms;
    if (queued_requests > 0) {
        adjusted_target_ms /= 1.0 + std::min<double>(2.0, queued_requests * 0.35);
    }
    const int estimate = static_cast<int>(adjusted_target_ms / baseline);
    const int cap = std::max(min_tokens, std::min(max_tokens, growth_max_tokens));
    // When the buffer is comfortable, aim for the cap directly.
    // On fast hardware (RTF << 1) estimate often falls below min_tokens,
    // making the AIMD cap irrelevant. Using cap as the target lets chunks
    // grow as the buffer deepens and amortizes per-chunk synthesis overhead.
    const int base = (lead_ms() >= comfort_lead_ms) ? cap : estimate;
    return std::max(min_tokens, std::min(cap, base));
}

void AdaptativeChunkController::observe(int n_tokens, double generation_ms, double audio_ms) {
    if (n_tokens <= 0 || generation_ms <= 0.0) return;
    const double sample = generation_ms / static_cast<double>(n_tokens);
    constexpr double alpha = 0.30;
    // Seed the EWMA at the default baseline rather than cold-starting at the
    // first observation. A single fast first chunk (JIT/Metal warmup) would
    // otherwise overshoot the estimate and make the next chunk too large,
    // causing a gap when subsequent synthesis is slower (O(n^2) attention).
    if (ms_per_token_ewma <= 0.0) ms_per_token_ewma = 12.0;
    ms_per_token_ewma = alpha * sample + (1.0 - alpha) * ms_per_token_ewma;

    // Closed-loop: update lead_ms and adjust the dynamic cap AIMD-style.
    cumulative_audio_ms += audio_ms;
    if (first_observed) {
        cumulative_gen_ms_after_first += generation_ms;
    } else {
        first_observed = true;
        first_gen_ms = generation_ms;  // TTFB cost: excluded from gen_after but deducted from lead
    }

    const double lead = lead_ms();
    const double rtf = audio_ms > 0.0 ? generation_ms / audio_ms : 1.0;

    // Shrink when the buffer is critically low, OR when synthesis is losing
    // ground (rtf > 1) and the buffer isn't already comfortable. Looking at
    // rtf in addition to absolute lead lets us react one chunk earlier when
    // generation slows down — otherwise we keep growing the cap until the
    // buffer has already drained.
    const bool buffer_critical = lead < safety_floor_ms;
    const bool losing_ground = rtf > 1.0 && lead < comfort_lead_ms;
    if (buffer_critical || losing_ground) {
        const int shrunk = static_cast<int>(growth_max_tokens * shrink_factor);
        growth_max_tokens = std::max(min_tokens, shrunk);
    } else if (lead >= comfort_lead_ms && rtf < 0.9) {
        growth_max_tokens = std::min(max_tokens, growth_max_tokens + grow_step_tokens);
    } else if (lead >= comfort_lead_ms) {
        growth_max_tokens = std::min(max_tokens, growth_max_tokens + grow_step_tokens / 2);
    }
    // Otherwise (between floor and comfort, rtf ≤ 1): hold steady.
}

Chunk build_adaptative_chunk(
    const std::vector<Unit> & units,
    size_t & next_unit,
    const ChunkConfig & config,
    int target_tokens,
    bool is_first) {
    Chunk chunk;
    if (next_unit >= units.size()) return chunk;

    const int target = std::max(config.target_min_tokens,
                                std::min(config.hard_max_tokens, target_tokens));
    auto append_unit = [&](const Unit & unit) {
        if (!chunk.phonemes.empty() && !unit.phonemes.empty()) {
            chunk.phonemes.push_back(' ');
        }
        chunk.phonemes.append(unit.phonemes);
        chunk.tokens.insert(chunk.tokens.end(), unit.tokens.begin(), unit.tokens.end());
        chunk.n_tokens += unit.n_tokens;
        chunk.text.append(unit.text);
        chunk.boundary_after = unit.boundary_after;
    };

    while (next_unit < units.size()) {
        const Unit & unit = units[next_unit];
        const int trial = chunk.n_tokens + unit.n_tokens;
        if (!chunk.tokens.empty() && trial > config.hard_max_tokens) {
            break;
        }

        append_unit(unit);
        ++next_unit;

        if (is_first) {
            if (chunk.boundary_after != Boundary::None) {
                // first_chunk_target_max_tokens == 0  → flush at first boundary (default)
                // first_chunk_target_max_tokens  > 0  → keep accumulating until that threshold
                if (config.first_chunk_target_max_tokens <= 0 ||
                    chunk.n_tokens >= config.first_chunk_target_max_tokens) break;
            }
            if (chunk.n_tokens >= config.hard_max_tokens) break;
            continue;
        }

        // Break at any strong (sentence/paragraph) boundary once we are past
        // target_min_tokens. Avoids orphan-tail bugs and yields natural cuts.
        if (chunk.n_tokens >= config.target_min_tokens &&
            is_strong_boundary(chunk.boundary_after)) {
            break;
        }

        // Boundary-scored break around target:
        //   - Past `target` with a strong boundary  → break (good cut).
        //   - Past `target` with a weak boundary    → only break if we already
        //     overshot beyond `target + overshoot`, otherwise keep going to
        //     try to reach a stronger boundary.
        // The overshoot is bounded by target_max_tokens / soft_max_tokens
        // below so we don't snowball.
        const int overshoot = std::max(0, config.target_overshoot_tokens);
        if (chunk.n_tokens >= target && chunk.boundary_after != Boundary::None) {
            const int score = boundary_score(chunk.boundary_after);
            const bool strong = score >= boundary_score(Boundary::Sentence);
            const bool past_overshoot = chunk.n_tokens >= target + overshoot;
            if (strong || past_overshoot) {
                break;
            }
        }

        // Hard cap: max(target, target_max_tokens) lets a large controller
        // target override the static 80-token cap when the buffer is deep.
        if (chunk.n_tokens >= std::max(config.target_max_tokens, target)) break;
        if (chunk.n_tokens >= config.soft_max_tokens &&
            chunk.boundary_after != Boundary::None) {
            break;
        }
        if (chunk.n_tokens >= config.hard_max_tokens) break;
    }

    // Tail-merge: if the units remaining after this chunk would form a single
    // tiny last chunk (< target_min_tokens), absorb them now so we don't end
    // up with a dry, undersized final chunk. Bounded by hard_max_tokens.
    if (!is_first && next_unit < units.size() && !chunk.tokens.empty()) {
        int remaining = 0;
        for (size_t i = next_unit; i < units.size(); ++i) {
            remaining += units[i].n_tokens;
        }
        if (remaining > 0 && remaining < config.target_min_tokens &&
            chunk.n_tokens + remaining <= config.hard_max_tokens) {
            while (next_unit < units.size()) {
                append_unit(units[next_unit]);
                ++next_unit;
            }
        }
    }

    chunk.is_first = is_first;
    chunk.is_last = next_unit >= units.size();
    return chunk;
}

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

void clear_chunk_data(Chunk & chunk) {
    // Free phonemes and tokens — swap with empty releases the heap allocation.
    { std::string       empty; empty.swap(chunk.phonemes); }
    { std::vector<uint32_t> empty; empty.swap(chunk.tokens); }
    // Keep n_tokens and boundary_after for diagnostics
}

} // namespace kokopop
