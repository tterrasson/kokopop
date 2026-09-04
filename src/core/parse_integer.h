#pragma once

#include <charconv>
#include <cstdint>
#include <cstring>

namespace kokopop {

// Decimal unsigned CLI values: consume the entire argument, without signs,
// truncation, wraparound or exceptions (including the full uint64_t range).
inline bool parse_u64(const char * text, uint64_t & out) {
    if (text == nullptr || *text == '\0') return false;
    uint64_t value = 0;
    const char * end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, value);
    if (result.ec != std::errc{} || result.ptr != end) return false;
    out = value;
    return true;
}

} // namespace kokopop
