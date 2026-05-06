#pragma once
// Lightweight number → Chinese character conversion for Mandarin TTS.
//
// This is intentionally dependency-free. It handles the forms that matter most
// to pronunciation: years, dates, clock times, percentages, decimals, long
// digit strings such as phone numbers, and small cardinal/ordinal numbers.
//
// Examples:
//   "123" → "一百二十三"
//   "2023年" → "二零二三年"
//   "第5名" → "第五名"
//   "14:05" → "十四点零五分"

#include <string>
#include <array>
#include <string_view>
#include <algorithm>

namespace kokopop::g2p::zh {

// UTF-8 encoded Chinese digits 零 through 九
inline constexpr std::array<const char *, 10> chinese_digits = {
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

inline bool starts_with_at(std::string_view text, size_t pos, std::string_view needle) {
    return pos + needle.size() <= text.size() &&
           text.substr(pos, needle.size()) == needle;
}

inline void append_digits_as_chinese(std::string & out, std::string_view digits) {
    for (char c : digits) {
        out.append(chinese_digits[static_cast<size_t>(c - '0')]);
    }
}

inline std::string integer_to_chinese_under_10000(int value) {
    if (value < 0) return std::string(chinese_digits[0]);
    if (value == 0) return std::string(chinese_digits[0]);
    // Defensive: if caller passed >= 10000, fall back to digit-by-digit.
    if (value >= 10000) {
        std::string out;
        std::string digits;
        digits.reserve(8);
        int v = value;
        while (v > 0) { digits.push_back(static_cast<char>('0' + (v % 10))); v /= 10; }
        std::reverse(digits.begin(), digits.end());
        append_digits_as_chinese(out, digits);
        return out;
    }

    struct Unit {
        int value;
        const char * text;
    };
    constexpr Unit units[] = {
        {1000, "千"},
        {100,  "百"},
        {10,   "十"},
        {1,    ""},
    };

    std::string out;
    bool pending_zero = false;
    for (const auto & unit : units) {
        const int digit = value / unit.value;
        value %= unit.value;
        if (digit == 0) {
            if (!out.empty() && value > 0) pending_zero = true;
            continue;
        }
        if (pending_zero) {
            out.append(chinese_digits[0]);
            pending_zero = false;
        }
        if (!(unit.value == 10 && digit == 1 && out.empty())) {
            out.append(chinese_digits[static_cast<size_t>(digit)]);
        }
        out.append(unit.text);
    }
    return out;
}

inline int parse_small_int(std::string_view digits) {
    int value = 0;
    for (char c : digits) {
        value = value * 10 + (c - '0');
        if (value > 9999) return value;
    }
    return value;
}

inline std::string num_to_chinese(const std::string & text) {
    std::string result;
    result.reserve(text.size() * 2);

    size_t i = 0;
    while (i < text.size()) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (ch < '0' || ch > '9') {
            result.push_back(text[i++]);
            continue;
        }

        const size_t start = i;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') ++i;
        const std::string_view digits(text.data() + start, i - start);

        if (i < text.size() && text[i] == ':') {
            size_t min_start = i + 1;
            size_t min_end = min_start;
            while (min_end < text.size() && text[min_end] >= '0' && text[min_end] <= '9') ++min_end;
            if (min_end > min_start) {
                const int hour = parse_small_int(digits);
                const std::string_view minutes(text.data() + min_start, min_end - min_start);
                const int minute = parse_small_int(minutes);
                result.append(integer_to_chinese_under_10000(hour));
                result.append("点");
                if (minute > 0) {
                    if (minutes.size() >= 2 && minutes.front() == '0') {
                        result.append(chinese_digits[0]);
                        append_digits_as_chinese(result, minutes.substr(1));
                    } else {
                        result.append(integer_to_chinese_under_10000(minute));
                    }
                    result.append("分");
                }
                i = min_end;
                continue;
            }
        }

        if (i < text.size() && text[i] == '.') {
            size_t frac_start = i + 1;
            size_t frac_end = frac_start;
            while (frac_end < text.size() && text[frac_end] >= '0' && text[frac_end] <= '9') ++frac_end;
            if (frac_end > frac_start) {
                const int int_part = parse_small_int(digits);
                if (int_part <= 9999) {
                    result.append(integer_to_chinese_under_10000(int_part));
                } else {
                    append_digits_as_chinese(result, digits);
                }
                result.append("点");
                append_digits_as_chinese(result, std::string_view(text.data() + frac_start, frac_end - frac_start));
                i = frac_end;
                continue;
            }
        }

        if (i < text.size() && text[i] == '%') {
            result.append("百分之");
            const int pct_val = parse_small_int(digits);
            if (pct_val <= 9999) {
                result.append(integer_to_chinese_under_10000(pct_val));
            } else {
                append_digits_as_chinese(result, digits);
            }
            ++i;
            continue;
        }

        if (digits.size() == 4 && starts_with_at(text, i, "年")) {
            append_digits_as_chinese(result, digits);
            continue;
        }

        if (digits.size() >= 7) {
            append_digits_as_chinese(result, digits);
            continue;
        }

        const int value = parse_small_int(digits);
        if (value <= 9999) {
            result.append(integer_to_chinese_under_10000(value));
        } else {
            append_digits_as_chinese(result, digits);
        }
    }
    return result;
}

} // namespace kokopop::g2p::zh
