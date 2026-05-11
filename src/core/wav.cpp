#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace kokopop {
// ---------------------------------------------------------------------------
// Little-endian writers — operate on a raw pointer and advance it.
// Avoids per-byte push_back on a std::vector.
// ---------------------------------------------------------------------------
namespace {

void put_u16(uint8_t * & dst, uint16_t v) {
    dst[0] = static_cast<uint8_t>(v & 0xffu);
    dst[1] = static_cast<uint8_t>((v >> 8u) & 0xffu);
    dst += 2;
}

void put_u32(uint8_t * & dst, uint32_t v) {
    dst[0] = static_cast<uint8_t>(v & 0xffu);
    dst[1] = static_cast<uint8_t>((v >> 8u) & 0xffu);
    dst[2] = static_cast<uint8_t>((v >> 16u) & 0xffu);
    dst[3] = static_cast<uint8_t>((v >> 24u) & 0xffu);
    dst += 4;
}

void put_ascii(uint8_t * & dst, const char * s) {
    while (*s) {
        *dst++ = static_cast<uint8_t>(*s++);
    }
}

int16_t float_to_s16(float sample) {
    if (!std::isfinite(sample)) {
        sample = 0.0f;
    }
    sample = std::max(-1.0f, std::min(1.0f, sample));
    return static_cast<int16_t>(std::lrintf(sample * 32767.0f));
}

// SIMD float→s16 bulk conversion. Bit-exact equivalent of repeated float_to_s16:
//   1. NaN/Inf → 0 (non-finite mask via self-compare)
//   2. clamp to [-1, 1]
//   3. multiply by 32767
//   4. round-to-nearest-even (matches lrintf default rounding mode)
//   5. store as little-endian int16
// Little-endian targets only (NEON / AVX2 platforms qualify).
static_assert(static_cast<uint16_t>(0x0102) == 0x0102, "unexpected endianness");

void pcm_f32_to_s16_le(const float * src, uint8_t * dst, size_t n) {
    size_t i = 0;

#ifdef __ARM_NEON
    const float32x4_t vlo    = vdupq_n_f32(-1.0f);
    const float32x4_t vhi    = vdupq_n_f32( 1.0f);
    const float32x4_t vscale = vdupq_n_f32(32767.0f);
    for (; i + 7 < n; i += 8) {
        float32x4_t v0 = vld1q_f32(src + i);
        float32x4_t v1 = vld1q_f32(src + i + 4);

        // Zero out NaN/Inf lanes: finite ⇔ (x - x == 0), which is false for NaN and Inf.
        float32x4_t d0 = vsubq_f32(v0, v0);
        float32x4_t d1 = vsubq_f32(v1, v1);
        uint32x4_t fmask0 = vceqq_f32(d0, vdupq_n_f32(0.0f));
        uint32x4_t fmask1 = vceqq_f32(d1, vdupq_n_f32(0.0f));
        v0 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(v0), fmask0));
        v1 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(v1), fmask1));

        v0 = vminq_f32(vmaxq_f32(v0, vlo), vhi);
        v1 = vminq_f32(vmaxq_f32(v1, vlo), vhi);
        v0 = vmulq_f32(v0, vscale);
        v1 = vmulq_f32(v1, vscale);

        // vcvtnq_s32_f32 rounds to nearest, ties to even (IEEE-754 default).
        int32x4_t i32_0 = vcvtnq_s32_f32(v0);
        int32x4_t i32_1 = vcvtnq_s32_f32(v1);
        int16x4_t s0 = vqmovn_s32(i32_0);
        int16x4_t s1 = vqmovn_s32(i32_1);
        int16x8_t out = vcombine_s16(s0, s1);
        vst1q_s16(reinterpret_cast<int16_t *>(dst + 2 * i), out);
    }
#elif defined(__AVX2__)
    const __m256 vlo    = _mm256_set1_ps(-1.0f);
    const __m256 vhi    = _mm256_set1_ps( 1.0f);
    const __m256 vscale = _mm256_set1_ps(32767.0f);
    const __m256 vzero  = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        // finite ⇔ v == v && |v| != +Inf. (v - v == 0) is true exactly for finite.
        __m256 finite_mask = _mm256_cmp_ps(_mm256_sub_ps(v, v), vzero, _CMP_EQ_OQ);
        v = _mm256_and_ps(v, finite_mask);
        v = _mm256_min_ps(_mm256_max_ps(v, vlo), vhi);
        v = _mm256_mul_ps(v, vscale);
        // _mm256_cvtps_epi32 uses MXCSR rounding mode (default: round-to-nearest-even).
        __m256i i32 = _mm256_cvtps_epi32(v);
        __m128i lo = _mm256_castsi256_si128(i32);
        __m128i hi = _mm256_extracti128_si256(i32, 1);
        __m128i s16 = _mm_packs_epi32(lo, hi);  // saturating int32 → int16
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 2 * i), s16);
    }
#endif

    for (; i < n; ++i) {
        const int16_t s = float_to_s16(src[i]);
        // Little-endian store, matches put_u16.
        std::memcpy(dst + 2 * i, &s, sizeof(int16_t));
    }
}

} // namespace

std::vector<uint8_t> wav_bytes(const kokopop_audio & audio) {
    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t byte_rate = static_cast<uint32_t>(audio.sample_rate) * block_align;
    const uint32_t data_bytes = static_cast<uint32_t>(audio.n_samples * block_align);

    // Allocate the entire WAV buffer upfront: 44-byte header + payload.
    // Then write directly through a raw pointer — zero push_back overhead.
    const size_t total_size = 44u + data_bytes;
    std::vector<uint8_t> out(total_size);
    uint8_t * p = out.data();

    put_ascii(p, "RIFF");
    put_u32(p, 36u + data_bytes);
    put_ascii(p, "WAVE");
    put_ascii(p, "fmt ");
    put_u32(p, 16);
    put_u16(p, 1);
    put_u16(p, channels);
    put_u32(p, static_cast<uint32_t>(audio.sample_rate));
    put_u32(p, byte_rate);
    put_u16(p, block_align);
    put_u16(p, bits_per_sample);
    put_ascii(p, "data");
    put_u32(p, data_bytes);

    pcm_f32_to_s16_le(audio.samples, p, audio.n_samples);
    return out;
}

bool write_wav_file(const std::string & path, const kokopop_audio & audio, std::string & error) {
    if (path.empty()) {
        error = "output path is empty";
        return false;
    }
    if (audio.sample_rate <= 0 || (audio.n_samples > 0 && audio.samples == nullptr)) {
        error = "invalid audio buffer";
        return false;
    }

    const auto bytes = wav_bytes(audio);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        error = "failed to open output WAV file";
        return false;
    }
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        error = "failed to write output WAV file";
        return false;
    }
    return true;
}

} // namespace kokopop

