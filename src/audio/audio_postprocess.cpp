#include "audio_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "core/constants.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace kokopop {
namespace {

static constexpr float SILENCE_THRESHOLD = 0.01f;

static int ms_to_samples(int ms, int sample_rate) {
    return (ms * sample_rate) / 1000;
}

// Returns true if any |x| > SILENCE_THRESHOLD in [data, data+count).
// SIMD with per-lane early exit; falls back to scalar.
static bool any_above_threshold(const float * data, size_t count) {
    size_t i = 0;
#ifdef __ARM_NEON
    const float32x4_t vt = vdupq_n_f32(SILENCE_THRESHOLD);
    for (; i + 3 < count; i += 4) {
        uint32x4_t cmp = vcgtq_f32(vabsq_f32(vld1q_f32(data + i)), vt);
        if (vgetq_lane_u32(cmp, 0) | vgetq_lane_u32(cmp, 1) |
            vgetq_lane_u32(cmp, 2) | vgetq_lane_u32(cmp, 3)) return true;
    }
#elif defined(__AVX2__)
    const __m256 vt       = _mm256_set1_ps(SILENCE_THRESHOLD);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (; i + 7 < count; i += 8) {
        __m256 v = _mm256_and_ps(_mm256_loadu_ps(data + i), abs_mask);
        if (_mm256_movemask_ps(_mm256_cmp_ps(v, vt, _CMP_GT_OQ))) return true;
    }
#endif
    for (; i < count; ++i)
        if (std::fabs(data[i]) > SILENCE_THRESHOLD) return true;
    return false;
}

// Pointer-based variants — operate on a raw range so callers can scan a view of
// a larger buffer without materialising it first.
static size_t scan_leading_edge(const float * data, size_t count, size_t max_samples) {
    const size_t check  = std::min(count, max_samples);
    if (check == 0) return 0;
    const size_t window = static_cast<size_t>(50 * KOKOPOP_SAMPLE_RATE / 1000);
    const size_t step   = std::max(size_t(1), window / 4);
    for (size_t i = 0; i < check; i += step) {
        if (any_above_threshold(data + i, std::min(window, check - i))) return i;
    }
    return check;
}

// Returns the position past the last non-silent sample, considering only the
// trailing `max_samples` of the range. If the entire trailing region is silent,
// returns count - max_samples (or 0 if count is shorter). If the last sample is
// non-silent, returns count (no trim).
static size_t scan_trailing_edge(const float * data, size_t count, size_t max_samples) {
    if (count == 0 || max_samples == 0) return count;
    const size_t scan_floor = count > max_samples ? count - max_samples : 0;

    // Walk backwards in blocks; SIMD-skip blocks that are entirely silent,
    // then locate the exact non-silent sample within the first hit.
    constexpr size_t BLOCK = 256;
    size_t end = count;
    while (end > scan_floor) {
        const size_t block_start = (end > scan_floor + BLOCK) ? end - BLOCK : scan_floor;
        if (any_above_threshold(data + block_start, end - block_start)) {
            for (size_t i = end; i > block_start; --i) {
                if (std::fabs(data[i - 1]) > SILENCE_THRESHOLD) return i;
            }
        }
        end = block_start;
    }
    return scan_floor;
}

static size_t find_leading_edge(const std::vector<float> & audio, size_t max_samples) {
    return scan_leading_edge(audio.data(), audio.size(), max_samples);
}

static size_t find_trailing_edge(const std::vector<float> & audio, size_t max_samples) {
    return scan_trailing_edge(audio.data(), audio.size(), max_samples);
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
        const float inv_n_minus_1 = 1.0f / static_cast<float>(n - 1);
        const float * prev_tail   = prev.data() + (prev.size() - static_cast<size_t>(n));
        const float * next_head   = next.data();
        float       * out         = result.data() + offset;
        int i = 0;

#ifdef __ARM_NEON
        {
            float32x4_t vi    = { 0.f, 1.f, 2.f, 3.f };
            const float32x4_t vstep  = vdupq_n_f32(4.f);
            const float32x4_t vscale = vdupq_n_f32(inv_n_minus_1);
            const float32x4_t vone   = vdupq_n_f32(1.f);
            for (; i + 3 < n; i += 4) {
                float32x4_t vt   = vmulq_f32(vi, vscale);
                float32x4_t v1_t = vsubq_f32(vone, vt);
                float32x4_t vp   = vld1q_f32(prev_tail + i);
                float32x4_t vn   = vld1q_f32(next_head + i);
                vst1q_f32(out + i, vmlaq_f32(vmulq_f32(vp, v1_t), vn, vt));
                vi = vaddq_f32(vi, vstep);
            }
        }
#elif defined(__AVX2__)
        {
            __m256 vi    = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
            const __m256 vstep  = _mm256_set1_ps(8.f);
            const __m256 vscale = _mm256_set1_ps(inv_n_minus_1);
            const __m256 vone   = _mm256_set1_ps(1.f);
            for (; i + 7 < n; i += 8) {
                __m256 vt   = _mm256_mul_ps(vi, vscale);
                __m256 v1_t = _mm256_sub_ps(vone, vt);
                __m256 vp   = _mm256_loadu_ps(prev_tail + i);
                __m256 vn   = _mm256_loadu_ps(next_head + i);
                _mm256_storeu_ps(out + i, _mm256_fmadd_ps(vp, v1_t, _mm256_mul_ps(vn, vt)));
                vi = _mm256_add_ps(vi, vstep);
            }
        }
#endif
        for (; i < n; ++i) {
            const float t = static_cast<float>(i) * inv_n_minus_1;
            out[i] = prev_tail[i] * (1.0f - t) + next_head[i] * t;
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
    if (audio.empty()) return {};

    // Compute trim offsets on the input directly — no intermediate allocations.
    size_t start = 0;
    size_t end   = audio.size();

    if (config.trim_silence) {
        // Leading trim: skipped on the very first chunk to preserve natural intro.
        if (!chunk.is_first) {
            const size_t max_lead = static_cast<size_t>(
                ms_to_samples(config.max_silence_trim_ms, sample_rate));
            start = scan_leading_edge(audio.data(), audio.size(), max_lead);
        }

        // Trailing trim is performed on the post-leading-trim view (matches
        // the legacy sequential behaviour bit-for-bit).
        int trim_ms = config.max_silence_trim_ms;
        if (chunk.boundary_after != Boundary::None) {
            trim_ms = std::max(1, trim_ms / 2);
        }
        const size_t max_trail = static_cast<size_t>(ms_to_samples(trim_ms, sample_rate));
        const size_t view_size = audio.size() - start;
        const size_t edge = scan_trailing_edge(audio.data() + start, view_size, max_trail);
        end = start + edge;
    }

    // Boundary pause samples (don't append after the last chunk).
    size_t pause_n = 0;
    const int pause_ms = pause_for_boundary(chunk.boundary_after, config);
    if (pause_ms > 0 && !chunk.is_last) {
        pause_n = static_cast<size_t>(ms_to_samples(pause_ms, sample_rate));
    }

    const size_t body = (end > start) ? (end - start) : 0;
    std::vector<float> result(body + pause_n);  // pause tail zero-initialised
    if (body > 0) {
        std::memcpy(result.data(), audio.data() + start, body * sizeof(float));
    }
    return result;
}

} // namespace kokopop
