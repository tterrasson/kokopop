#include "audio_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "core/constants.h"

namespace kokopop {
namespace {

static constexpr float SILENCE_THRESHOLD = 0.01f;

static int ms_to_samples(int ms, int sample_rate) {
    return (ms * sample_rate) / 1000;
}

// ---------------------------------------------------------------------------
// Sliding-window silence detection via prefix-sum.
//
// Instead of rescanning every window independently (which re-evaluates fabs
// for the same samples multiple times), we build a prefix-sum of "non-silent"
// booleans once.  A window [lo, hi) contains non-silent audio iff
//   prefix[hi] - prefix[lo] > 0
//
// This reduces the total work from O(k · window) to O(n) where n is the
// number of samples and k is the number of window probes.
// ---------------------------------------------------------------------------

// Find the first non-silent sample from the start
static size_t find_leading_edge(const std::vector<float> & audio, size_t max_samples) {
    size_t check = std::min(audio.size(), max_samples);
    if (check == 0) return 0;

    const size_t window = static_cast<size_t>(50 * KOKOPOP_SAMPLE_RATE / 1000); // 50ms
    const size_t step   = std::max(size_t(1), window / 4);

    // Build prefix-sum: prefix[i] = count of non-silent samples in audio[0..i)
    std::vector<uint32_t> prefix(check + 1, 0);
    for (size_t i = 0; i < check; ++i) {
        prefix[i + 1] = prefix[i] + (std::fabs(audio[i]) > SILENCE_THRESHOLD ? 1u : 0u);
    }

    // Scan with sliding windows; return the start of the first non-silent window
    for (size_t i = 0; i < check; i += step) {
        size_t end = std::min(i + window, check);
        if (prefix[end] - prefix[i] > 0) {
            return i;
        }
    }
    return check;
}

// Find the last non-silent sample from the end
static size_t find_trailing_edge(const std::vector<float> & audio, size_t max_samples) {
    size_t check_end = std::min(audio.size(), max_samples);
    if (check_end == 0) return 0;

    const size_t window = static_cast<size_t>(50 * KOKOPOP_SAMPLE_RATE / 1000); // 50ms
    const size_t step   = std::max(size_t(1), window / 4);

    // Build prefix-sum: prefix[i] = count of non-silent samples in audio[0..i)
    std::vector<uint32_t> prefix(check_end + 1, 0);
    for (size_t i = 0; i < check_end; ++i) {
        prefix[i + 1] = prefix[i] + (std::fabs(audio[i]) > SILENCE_THRESHOLD ? 1u : 0u);
    }

    // Scan backwards; return the end of the first non-silent window found
    for (size_t i = check_end; i > 0; ) {
        size_t start = (i > window) ? i - window : 0;
        if (prefix[i] - prefix[start] > 0) {
            return i;
        }
        i -= step;
    }
    return 0;
}

} // namespace

std::vector<float> apply_crossfade(
    const std::vector<float> & prev,
    const std::vector<float> & next,
    int crossfade_ms,
    int sample_rate) {
    if (prev.empty()) return next;
    if (next.empty()) return prev;

    int n = ms_to_samples(crossfade_ms, sample_rate);
    if (n <= 0 ||
        static_cast<int>(prev.size()) < n ||
        static_cast<int>(next.size()) < n) {
        // Too short for crossfade, simple concat
        std::vector<float> result;
        result.reserve(prev.size() + next.size());
        result.insert(result.end(), prev.begin(), prev.end());
        result.insert(result.end(), next.begin(), next.end());
        return result;
    }

    // Pre-size result exactly: (prev - n) + n (crossfade) + (next - n)
    std::vector<float> result;
    result.resize(prev.size() + next.size() - n);

    // Main part of prev (everything except tail)
    size_t offset = 0;
    size_t main_len = prev.size() - static_cast<size_t>(n);
    std::memcpy(result.data() + offset, prev.data(), main_len * sizeof(float));
    offset += main_len;

    // Crossfade region — write directly into pre-sized buffer
    if (n == 1) {
        // Edge case: single-sample overlap, just pick the next-head sample
        result[offset] = next[0];
    } else {
        float inv_n_minus_1 = 1.0f / static_cast<float>(n - 1);
        for (int i = 0; i < n; ++i) {
            float t = static_cast<float>(i) * inv_n_minus_1;
            result[offset + i] =
                prev[prev.size() - n + i] * (1.0f - t) +
                next[i] * t;
        }
    }
    offset += static_cast<size_t>(n);

    // Rest of next (everything except head)
    size_t tail_len = next.size() - static_cast<size_t>(n);
    std::memcpy(result.data() + offset, next.data() + n, tail_len * sizeof(float));

    return result;
}

std::vector<float> apply_crossfade_smart(
    const std::vector<float> & prev,
    const std::vector<float> & next,
    Boundary boundary,
    int crossfade_ms,
    int sample_rate) {
    // Don't crossfade after strong boundaries — natural pause is better
    if (boundary == Boundary::Sentence ||
        boundary == Boundary::Paragraph ||
        boundary == Boundary::Newline) {
        std::vector<float> result;
        result.reserve(prev.size() + next.size());
        result.insert(result.end(), prev.begin(), prev.end());
        result.insert(result.end(), next.begin(), next.end());
        return result;
    }

    return apply_crossfade(prev, next, crossfade_ms, sample_rate);
}

std::vector<float> trim_leading_silence(
    const std::vector<float> & audio,
    int max_ms,
    int sample_rate) {
    if (audio.empty()) return audio;

    int max_samples = ms_to_samples(max_ms, sample_rate);
    size_t edge = find_leading_edge(audio, static_cast<size_t>(max_samples));

    if (edge == 0) return audio;

    return std::vector<float>(audio.begin() + edge, audio.end());
}

std::vector<float> trim_trailing_silence(
    const std::vector<float> & audio,
    int max_ms,
    int sample_rate) {
    if (audio.empty()) return audio;

    int max_samples = ms_to_samples(max_ms, sample_rate);
    size_t edge = find_trailing_edge(audio, static_cast<size_t>(max_samples));

    if (edge >= audio.size()) return audio;

    std::vector<float> result(audio.begin(), audio.begin() + edge);
    return result;
}

void append_silence(
    std::vector<float> & audio,
    int ms,
    int sample_rate) {
    if (ms <= 0) return;
    int n = ms_to_samples(ms, sample_rate);
    audio.insert(audio.end(), n, 0.0f);
}

int pause_for_boundary(Boundary boundary, const ChunkConfig & config) {
    switch (boundary) {
        case Boundary::Paragraph:
            return config.paragraph_pause_ms;
        case Boundary::Newline:
            return config.paragraph_pause_ms / 2; // ~half of paragraph pause
        case Boundary::Sentence:
            return config.sentence_pause_ms;
        case Boundary::ClauseStrong:
            return static_cast<int>(config.sentence_pause_ms * 0.7f);
        case Boundary::ClauseWeak:
            return config.comma_pause_ms;
        case Boundary::None:
        default:
            return 0;
    }
}

std::vector<float> postprocess_chunk_audio(
    const std::vector<float> & audio,
    const Chunk & chunk,
    int /*chunk_index*/,
    int /*total_chunks*/,
    const ChunkConfig & config,
    int sample_rate) {
    std::vector<float> result = audio;
    if (result.empty()) return result;

    // Trim leading silence (skip on first chunk to preserve natural intro)
    if (config.trim_silence && !chunk.is_first) {
        result = trim_leading_silence(result, config.max_silence_trim_ms, sample_rate);
    }

    // Trim trailing silence (less aggressively if there's a boundary pause)
    if (config.trim_silence) {
        int trim_ms = config.max_silence_trim_ms;
        if (chunk.boundary_after != Boundary::None) {
            trim_ms = std::max(1, trim_ms / 2);
        }
        result = trim_trailing_silence(result, trim_ms, sample_rate);
    }

    // Append boundary pause
    int pause_ms = pause_for_boundary(chunk.boundary_after, config);
    // Don't add pause after the last chunk
    if (pause_ms > 0 && !chunk.is_last) {
        append_silence(result, pause_ms, sample_rate);
    }

    return result;
}

} // namespace kokopop
