#include "utf8.h"

#include <algorithm>
#include <cctype>

namespace kokopop {

bool utf8_next(std::string_view text, size_t & offset, std::string_view & ch) {
    if (offset >= text.size()) {
        ch = {};
        return false;
    }

    const unsigned char c = static_cast<unsigned char>(text[offset]);
    size_t len = 0;
    if ((c & 0x80u) == 0) {
        len = 1;
    } else if ((c & 0xE0u) == 0xC0u) {
        len = 2;
    } else if ((c & 0xF0u) == 0xE0u) {
        len = 3;
    } else if ((c & 0xF8u) == 0xF0u) {
        len = 4;
    } else {
        return false;
    }

    if (offset + len > text.size()) {
        return false;
    }
    for (size_t i = 1; i < len; ++i) {
        const unsigned char cc = static_cast<unsigned char>(text[offset + i]);
        if ((cc & 0xC0u) != 0x80u) {
            return false;
        }
    }

    ch = text.substr(offset, len);
    offset += len;
    return true;
}

std::vector<std::string> utf8_chars(std::string_view text) {
    std::vector<std::string> chars;
    chars.reserve(text.size());
    size_t off = 0;
    std::string_view ch;
    while (utf8_next(text, off, ch)) {
        chars.emplace_back(ch);
    }
    if (off != text.size()) {
        chars.clear();
    }
    return chars;
}

std::string trim_ascii(std::string_view text) {
    size_t first = 0;
    while (first < text.size() && std::isspace(static_cast<unsigned char>(text[first]))) {
        ++first;
    }
    size_t last = text.size();
    while (last > first && std::isspace(static_cast<unsigned char>(text[last - 1]))) {
        --last;
    }
    return std::string(text.substr(first, last - first));
}

} // namespace kokopop

