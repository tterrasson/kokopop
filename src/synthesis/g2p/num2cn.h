#pragma once
// Simplified number → Chinese character conversion (cn2an port).
//
// Replaces ASCII digits 0-9 with Chinese digit characters.
// This is a minimal version — full cn2an handles complex numbers
// (hundreds, thousands, etc.) but for TTS phonemization, digit-by-digit
// replacement is sufficient since pypinyin handles the rest.
//
// Examples:
//   "123" → "一二三"
//   "2023年" → "二零二三年"
//   "第5名" → "第五名"

#include <string>
#include <array>

namespace kokopop::g2p::zh {

// UTF-8 encoded Chinese digits 零 through 九
inline constexpr std::array chinese_digits = {
    "零", // 0 — U+96F6
    "一", // 1 — U+4E00
    "二", // 2 — U+4E8C
    "三", // 3 — U+4E09
    "四", // 4 — U+56DB
    "五", // 5 — U+4E94
    "六", // 6 — U+516D
    "七", // 7 — U+4E03
    "八", // 8 — U+516B
    "九", // 9 — U+4E5D
};

inline std::string num_to_chinese(const std::string & text) {
    std::string result;
    result.reserve(text.size());

    for (char c : text) {
        if (c >= '0' && c <= '9') {
            result.append(chinese_digits[static_cast<size_t>(c - '0')]);
        } else {
            result.push_back(c);
        }
    }
    return result;
}

} // namespace kokopop::g2p::zh
