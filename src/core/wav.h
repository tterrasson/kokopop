#pragma once

#include "kokopop.h"

#include <string>
#include <vector>

namespace kokopop {

bool write_wav_file(const std::string & path, const kokopop_audio & audio, std::string & error);
std::vector<uint8_t> wav_bytes(const kokopop_audio & audio);

} // namespace kokopop
