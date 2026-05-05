#pragma once

#include <string>

namespace kokopop {

std::string espeak_voice_for_kokoro_voice(const std::string & voice);
std::string normalize_espeak_phonemes(std::string phonemes, char kokoro_lang);
bool phonemize_text(const std::string & text, const std::string & voice, std::string & phonemes, std::string & error);

} // namespace kokopop

