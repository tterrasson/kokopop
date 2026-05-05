#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>

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

    for (size_t i = 0; i < audio.n_samples; ++i) {
        put_u16(p, static_cast<uint16_t>(float_to_s16(audio.samples[i])));
    }
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

