#pragma once

#include "kokopop.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kokopop {

bool write_wav_file(const std::string & path, const kokopop_audio & audio, std::string & error);
std::vector<uint8_t> wav_bytes(const kokopop_audio & audio);

/// Minimal parsed view of a RIFF/WAVE file, for reading PCM WAV input
/// (used by kokopop_play). All bounds are validated against the buffer:
/// `data_size` is clamped to what is actually present.
struct WavFileInfo {
    int32_t sample_rate = 0;
    int channels = 1;
    int bits_per_sample = 16;
    bool is_float = false;      // IEEE float samples (format tag 3)
    size_t data_offset = 0;     // offset of the first sample byte
    size_t data_size = 0;       // sample bytes available (clamped to buffer)
};

/// Parse a RIFF/WAVE header. Accepts PCM (format 1) 16/32-bit integer and
/// IEEE float (format 3) 32-bit. Returns false on any malformed input.
bool wav_parse(const uint8_t * data, size_t size, WavFileInfo & out);

} // namespace kokopop
