#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace kokopop {

bool utf8_next(std::string_view text, size_t & offset, std::string_view & ch);
std::vector<std::string> utf8_chars(std::string_view text);
std::string trim_ascii(std::string_view text);

} // namespace kokopop

