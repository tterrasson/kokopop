#include "zh_g2p.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "pinyin_dict.h"
#include "pinyin_tables.h"
#include "num2cn.h"

namespace kokopop::g2p::zh {
namespace {

// ────────────────────────────────────────────────────────────────────────
// UTF-8 helpers
// ────────────────────────────────────────────────────────────────────────

inline uint32_t utf8_decode(const char * p, size_t & byte_len) {
    unsigned char c = static_cast<unsigned char>(p[0]);
    if ((c & 0x80) == 0) {
        byte_len = 1;
        return static_cast<uint32_t>(c);
    }
    if ((c & 0xE0) == 0xC0) {
        byte_len = 2;
        return (static_cast<uint32_t>(c & 0x1F) << 6) |
               (static_cast<uint32_t>(p[1] & 0x3F));
    }
    if ((c & 0xF0) == 0xE0) {
        byte_len = 3;
        return (static_cast<uint32_t>(c & 0x0F) << 12) |
               (static_cast<uint32_t>(p[1] & 0x3F) << 6) |
               (static_cast<uint32_t>(p[2] & 0x3F));
    }
    if ((c & 0xF8) == 0xF0) {
        byte_len = 4;
        return (static_cast<uint32_t>(c & 0x07) << 18) |
               (static_cast<uint32_t>(p[1] & 0x3F) << 12) |
               (static_cast<uint32_t>(p[2] & 0x3F) << 6) |
               (static_cast<uint32_t>(p[3] & 0x3F));
    }
    byte_len = 1;
    return 0xFFFFFFFF;
}

inline bool is_cjk_ideograph(uint32_t cp) {
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    if (cp >= 0x20000 && cp <= 0x2A6DF) return true;
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    return false;
}

inline bool is_chinese_digit(uint32_t cp) {
    switch (cp) {
        case 0x96F6: // 零
        case 0x3007: // 〇
        case 0x4E00: // 一
        case 0x4E8C: // 二
        case 0x4E09: // 三
        case 0x56DB: // 四
        case 0x4E94: // 五
        case 0x516D: // 六
        case 0x4E03: // 七
        case 0x516B: // 八
        case 0x4E5D: // 九
        case 0x5341: // 十
        case 0x767E: // 百
        case 0x5343: // 千
            return true;
        default:
            return false;
    }
}

struct Utf8Char {
    std::string text;
    uint32_t cp = 0;
    bool is_zh = false;
};

std::vector<Utf8Char> split_utf8_chars(std::string_view text) {
    std::vector<Utf8Char> chars;
    chars.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        size_t len = 1;
        uint32_t cp = utf8_decode(text.data() + i, len);
        if (i + len > text.size()) len = 1;
        chars.push_back({std::string(text.substr(i, len)), cp, is_cjk_ideograph(cp)});
        i += len;
    }
    return chars;
}

// ────────────────────────────────────────────────────────────────────────
// Pinyin dictionary lookup (binary search)
// ────────────────────────────────────────────────────────────────────────

inline std::string_view pinyin_for_cp(uint32_t cp) {
    const auto & dict = PINYIN_CP;
    const size_t n = PINYIN_DICT_SIZE;

    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (dict[mid] < cp) {
            lo = mid + 1;
        } else if (dict[mid] > cp) {
            hi = mid;
        } else {
            size_t offset = PINYIN_OFFSET[mid];
            const char * start =
                reinterpret_cast<const char *>(PINYIN_DATA + offset);
            size_t len = 0;
            while (len < 20 && start[len] != '\0') ++len;
            return std::string_view(start, len);
        }
    }
    return {};
}

struct LexemeEntry {
    std::string_view text;
    std::string_view pinyin;
};

// High-impact Mandarin overrides. Longest-match lookup uses this table before
// falling back to the generated per-character dictionary.
inline constexpr LexemeEntry g_lexicon[] = {
    {"一个", "yi1 ge5"},
    {"一些", "yi1 xie1"},
    {"一点", "yi1 dian3"},
    {"一样", "yi1 yang4"},
    {"一定", "yi1 ding4"},
    {"一直", "yi1 zhi2"},
    {"一起", "yi1 qi3"},
    {"一天", "yi1 tian1"},
    {"一杯", "yi1 bei1"},

    {"不是", "bu2 shi4"},
    {"不好", "bu4 hao3"},
    {"不用", "bu4 yong4"},
    {"不要", "bu2 yao4"},
    {"不会", "bu2 hui4"},

    {"你好", "ni3 hao3"},
    {"很好", "hen3 hao3"},
    {"我想买", "wo3 xiang3 mai3"},
    {"什么", "shen2 me5"},
    {"什么时候", "shen2 me5 shi2 hou5"},
    {"时候", "shi2 hou5"},
    {"我们", "wo3 men5"},
    {"你们", "ni3 men5"},
    {"他们", "ta1 men5"},
    {"她们", "ta1 men5"},
    {"它们", "ta1 men5"},
    {"这个", "zhe4 ge5"},
    {"那个", "na4 ge5"},
    {"哪个", "na3 ge5"},
    {"这里", "zhe4 li5"},
    {"那里", "na4 li5"},
    {"东西", "dong1 xi5"},
    {"朋友", "peng2 you3"},
    {"孩子", "hai2 zi5"},
    {"儿子", "er2 zi5"},
    {"桌子", "zhuo1 zi5"},
    {"椅子", "yi3 zi5"},
    {"杯子", "bei1 zi5"},
    {"妈妈", "ma1 ma5"},
    {"爸爸", "ba4 ba5"},
    {"哥哥", "ge1 ge5"},
    {"姐姐", "jie3 jie5"},
    {"弟弟", "di4 di5"},
    {"妹妹", "mei4 mei5"},

    {"银行", "yin2 hang2"},
    {"行业", "hang2 ye4"},
    {"行长", "hang2 zhang3"},
    {"行走", "xing2 zou3"},
    {"行为", "xing2 wei2"},
    {"旅行", "lv3 xing2"},
    {"不行", "bu4 xing2"},
    {"进行", "jin4 xing2"},

    {"重要", "zhong4 yao4"},
    {"重点", "zhong4 dian3"},
    {"重量", "zhong4 liang4"},
    {"重复", "chong2 fu4"},
    {"重新", "chong2 xin1"},
    {"重庆", "chong2 qing4"},

    {"长大", "zhang3 da4"},
    {"长城", "chang2 cheng2"},
    {"长度", "chang2 du4"},
    {"长短", "chang2 duan3"},
    {"校长", "xiao4 zhang3"},
    {"成长", "cheng2 zhang3"},

    {"只要", "zhi3 yao4"},
    {"只有", "zhi3 you3"},
    {"只是", "zhi3 shi4"},
    {"一只", "yi1 zhi1"},
    {"两只", "liang3 zhi1"},

    {"了解", "liao3 jie3"},
    {"为了", "wei4 le5"},
    {"完了", "wan2 le5"},
    {"走了", "zou3 le5"},
    {"得了", "de2 liao3"},
    {"了不起", "liao3 bu4 qi3"},

    {"得到", "de2 dao4"},
    {"觉得", "jue2 de5"},
    {"跑得快", "pao3 de5 kuai4"},
    {"慢慢地", "man4 man4 de5"},
    {"高兴地", "gao1 xing4 de5"},
    {"目的", "mu4 di4"},

    {"还钱", "huan2 qian2"},
    {"归还", "gui1 huan2"},
    {"还有", "hai2 you3"},
    {"还是", "hai2 shi4"},
    {"还没", "hai2 mei2"},

    {"音乐", "yin1 yue4"},
    {"乐队", "yue4 dui4"},
    {"快乐", "kuai4 le4"},
    {"欢乐", "huan1 le4"},
    {"可乐", "ke3 le4"},

    {"因为", "yin1 wei4"},
    {"以为", "yi3 wei2"},
    {"认为", "ren4 wei2"},
    {"作为", "zuo4 wei2"},
    {"为啥", "wei4 sha2"},
    {"为什么", "wei4 shen2 me5"},
    {"为人民服务", "wei4 ren2 min2 fu2 wu4"},

    {"尽管", "jin3 guan3"},
    {"儘管", "jin3 guan3"},
    {"每天", "mei3 tian1"},
    {"忙着", "mang2 zhe5"},
    {"忙著", "mang2 zhe5"},
    {"工作", "gong1 zuo4"},
    {"学习", "xue2 xi2"},
    {"學習", "xue2 xi2"},

    {"千万", "qian1 wan4"},
    {"千萬", "qian1 wan4"},
    {"忘记", "wang4 ji4"},
    {"忘記", "wang4 ji4"},

    {"时间", "shi2 jian1"},
    {"時間", "shi2 jian1"},
    {"陪伴", "pei2 ban4"},
    {"家人", "jia1 ren2"},
    {"毕竟", "bi4 jing4"},
    {"畢竟", "bi4 jing4"},
    {"生活", "sheng1 huo2"},
    {"中的", "zhong1 de5"},
    {"美好", "mei3 hao3"},
    {"往往", "wang3 wang3"},
    {"来自于", "lai2 zi4 yu2"},
    {"來自於", "lai2 zi4 yu2"},
    {"这些", "zhe4 xie1"},
    {"這些", "zhe4 xie1"},
    {"平凡", "ping2 fan2"},
    {"瞬间", "shun4 jian1"},
    {"瞬間", "shun4 jian1"},

    {"没有", "mei2 you3"},
    {"沒有", "mei2 you3"},
    {"道德", "dao4 de2"},
    {"绑架", "bang3 jia4"},
    {"綁架", "bang3 jia4"},
    {"普通话", "pu3 tong1 hua4"},
    {"普通話", "pu3 tong1 hua4"},

    // Validation phrases: align with Kokoro Python's jieba-style grouping.
    {"随着", "sui2 zhe5"},
    {"人工智能", "ren2 gong1 zhi4 neng2"},
    {"发展", "fa1 zhan3"},
    {"生活", "sheng1 huo2"},
    {"方式", "fang1 shi4"},
    {"发生", "fa1 sheng1"},
    {"巨大", "ju4 da4"},
    {"变化", "bian4 hua4"},
    {"这座", "zhe4 zuo4"},
    {"古老", "gu3 lao3"},
    {"城市", "cheng2 shi4"},
    {"街道", "jie1 dao4"},
    {"两旁", "liang3 pang2"},
    {"种满", "zhong3 man3"},
    {"高大", "gao1 da4"},
    {"法国梧桐", "fa3 guo2 wu2 tong2"},
    {"一种", "yi1 zhong3"},
    {"宁静", "ning2 jing4"},
    {"感觉", "gan3 jue2"},
    {"真正", "zhen1 zheng4"},
    {"友谊", "you3 yi4"},
    {"不仅", "bu4 jin3"},
    {"在于", "zai4 yu2"},
    {"分享", "fen1 xiang3"},
    {"彼此", "bi3 ci3"},
    {"遇到困难", "yu4 dao4 kun4 nan2"},
    {"互相支持", "hu4 xiang1 zhi1 chi2"},
    {"团队", "tuan2 dui4"},
    {"合作", "he2 zuo4"},
    {"项目", "xiang4 mu4"},
    {"成功", "cheng2 gong1"},
    {"关键", "guan1 jian4"},
    {"一位", "yi1 wei4"},
    {"成员", "cheng2 yuan2"},
    {"贡献", "gong4 xian4"},
    {"不可或缺", "bu4 ke3 huo4 que1"},
    {"无论", "wu2 lun4"},
    {"前方", "qian2 fang1"},
    {"道路", "dao4 lu4"},
    {"多么", "duo1 me5"},
    {"曲折", "qu1 zhe2"},
    {"保持", "bao3 chi2"},
    {"能够", "neng2 gou4"},
    {"到达", "dao4 da2"},
    {"梦想", "meng4 xiang3"},
    {"彼岸", "bi3 an4"},

    // Common phrases with polyphonic characters
    {"民以食为天", "min2 yi3 shi2 wei2 tian1"},
    {"中国", "zhong1 guo2"},
    {"中國", "zhong1 guo2"},
    {"饮食文化", "yin3 shi2 wen2 hua4"},
    {"飲食文化", "yin3 shi2 wen2 hua4"},
    {"博大精深", "bo2 da4 jing1 shen1"},
    {"不仅", "bu4 jin3"},
    {"不僅", "bu4 jin3"},
    {"讲究", "jiang3 jiu1"},
    {"講究", "jiang3 jiu1"},
    {"色香味", "se4 xiang1 wei4"},
    {"俱全", "ju4 quan2"},
    {"包含", "bao1 han2"},
    {"丰富", "feng1 fu4"},
    {"豐富", "feng1 fu4"},
    {"历史背景", "li4 shi3 bei4 jing3"},
    {"歷史背景", "li4 shi3 bei4 jing3"},
    {"哲学思想", "zhe2 xue2 si1 xiang3"},
    {"哲學思想", "zhe2 xue2 si1 xiang3"},
    {"包含", "bao1 han2"},
    {"背景", "bei4 jing3"},
    {"思想", "si1 xiang3"},
    {"哲学", "zhe2 xue2"},
    {"哲學", "zhe2 xue2"},
};

inline bool starts_with_chars(const std::vector<Utf8Char> & chars,
                              size_t pos,
                              std::string_view needle) {
    size_t byte_pos = 0;
    for (size_t i = pos; i < chars.size() && byte_pos < needle.size(); ++i) {
        const auto & ch = chars[i].text;
        if (byte_pos + ch.size() > needle.size()) return false;
        if (needle.substr(byte_pos, ch.size()) != ch) return false;
        byte_pos += ch.size();
    }
    return byte_pos == needle.size();
}

inline std::vector<std::string_view> split_pinyin_list(std::string_view pinyin) {
    std::vector<std::string_view> out;
    size_t i = 0;
    while (i < pinyin.size()) {
        while (i < pinyin.size() && pinyin[i] == ' ') ++i;
        const size_t start = i;
        while (i < pinyin.size() && pinyin[i] != ' ') ++i;
        if (i > start) out.push_back(pinyin.substr(start, i - start));
    }
    return out;
}

inline const LexemeEntry * find_lexeme(const std::vector<Utf8Char> & chars,
                                       size_t pos) {
    const LexemeEntry * best = nullptr;
    size_t best_chars = 0;
    for (const auto & entry : g_lexicon) {
        if (!starts_with_chars(chars, pos, entry.text)) continue;
        const auto pinyins = split_pinyin_list(entry.pinyin);
        if (pinyins.size() > best_chars) {
            best = &entry;
            best_chars = pinyins.size();
        }
    }
    return best;
}

// ────────────────────────────────────────────────────────────────────────
// Punctuation mapping (Chinese → ASCII equivalents)
// ────────────────────────────────────────────────────────────────────────

inline std::string map_punctuation(const std::string & text) {
    std::string result;
    result.reserve(text.size() + 32);

    size_t i = 0;
    while (i < text.size()) {
        size_t char_len;
        uint32_t cp = utf8_decode(text.data() + i, char_len);

        switch (cp) {
            case 0x3001: result += ", "; break;
            case 0x3002: result += ". "; break;
            case 0xFF0C: result += ", "; break;
            case 0xFF0E: result += ". "; break;
            case 0xFF01: result += "! "; break;
            case 0xFF1A: result += ": "; break;
            case 0xFF1B: result += "; "; break;
            case 0xFF1F: result += "? "; break;
            case 0x300A: case 0x3008: result += " \" "; break;
            case 0x300B: case 0x3009: result += " \" "; break;
            case 0x3010: result += " \" "; break;
            case 0x3011: result += " \" "; break;
            case 0xFF08: result += " ( "; break;
            case 0xFF09: result += " ) "; break;
            case 0x00AB: result += " \" "; break;
            case 0x00BB: result += " \" "; break;
            default:
                if (cp >= 0xFF01 && cp <= 0xFF5E) {
                    char ascii = static_cast<char>(cp - 0xFEE0);
                    result += ascii;
                    if (!std::isalnum(static_cast<unsigned char>(ascii))) {
                        result += ' ';
                    }
                } else {
                    result.append(text.data() + i, char_len);
                }
                break;
        }
        i += char_len;
    }

    while (!result.empty() && (result.back() == ' ' || result.back() == '\t')) {
        result.pop_back();
    }

    return result;
}

// ────────────────────────────────────────────────────────────────────────
// Strip combining marks (U+0300–U+036F) from result
// ────────────────────────────────────────────────────────────────────────

inline std::string strip_combining_marks(const std::string & text) {
    std::string result;
    result.reserve(text.size());

    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == 0xCC && i + 1 < text.size() &&
            static_cast<unsigned char>(text[i + 1]) >= 0x80 &&
            static_cast<unsigned char>(text[i + 1]) <= 0xBF) {
            ++i;
            continue;
        }
        result.push_back(text[i]);
    }
    return result;
}

// ────────────────────────────────────────────────────────────────────────
// Replace syllabic consonants with Kokoro-compatible IPA
// ɻ̩ / ɹ̩ → ɨ (U+0268)
// ────────────────────────────────────────────────────────────────────────

inline std::string replace_syllabic_consonants(const std::string & text) {
    std::string result;
    result.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        // ɻ̩ = U+027B (0xC9 0xBB) + U+0329 (0xCC 0xA9)
        if (i + 3 < text.size() &&
            static_cast<unsigned char>(text[i])     == 0xC9 &&
            static_cast<unsigned char>(text[i + 1]) == 0xBB &&
            static_cast<unsigned char>(text[i + 2]) == 0xCC &&
            static_cast<unsigned char>(text[i + 3]) == 0xA9) {
            result += "\xC9\xA8"; // ɨ (U+0268)
            i += 4;
            continue;
        }
        // ɹ̩ = U+0279 (0xC9 0xB9) + U+0329 (0xCC 0xA9)
        if (i + 3 < text.size() &&
            static_cast<unsigned char>(text[i])     == 0xC9 &&
            static_cast<unsigned char>(text[i + 1]) == 0xB9 &&
            static_cast<unsigned char>(text[i + 2]) == 0xCC &&
            static_cast<unsigned char>(text[i + 3]) == 0xA9) {
            result += "\xC9\xA8"; // ɨ (U+0268)
            i += 4;
            continue;
        }
        result.push_back(text[i++]);
    }
    return result;
}

// ────────────────────────────────────────────────────────────────────────
// Pinyin parsing helpers
// ────────────────────────────────────────────────────────────────────────

inline char extract_tone(std::string_view py) {
    if (!py.empty() && py.back() >= '1' && py.back() <= '5') {
        return py.back();
    }
    return '5';
}

inline std::string_view strip_tone(std::string_view py) {
    if (!py.empty() && py.back() >= '1' && py.back() <= '5') {
        return py.substr(0, py.size() - 1);
    }
    return py;
}

/// Find initial in the initials table. Returns empty if none found.
inline std::string_view find_initial(std::string_view py) {
    if (py.size() >= 2) {
        std::string_view two = py.substr(0, 2);
        for (const auto & entry : g_initials) {
            std::string_view ep(entry.pinyin);
            if (ep.size() == 2 && two == ep) {
                return ep;
            }
        }
    }
    if (py.size() >= 1) {
        std::string_view one = py.substr(0, 1);
        for (const auto & entry : g_initials) {
            std::string_view ep(entry.pinyin);
            if (ep.size() == 1 && one == ep) {
                return ep;
            }
        }
    }
    return {};
}

inline std::string_view lookup_initial_ipa(std::string_view initial) {
    for (const auto & entry : g_initials) {
        if (entry.pinyin == initial) return entry.ipa;
    }
    return {};
}

inline std::string_view lookup_final_ipa(std::string_view final) {
    for (const auto & entry : g_finals) {
        if (entry.pinyin == final) return entry.ipa;
    }
    return {};
}

// ────────────────────────────────────────────────────────────────────────
// Normalize pinyin abbreviations: iu→iou, ui→uei, un→uen
// ────────────────────────────────────────────────────────────────────────

inline std::string expand_pinyin_abbrevs(std::string_view py_with_tone) {
    char tone = extract_tone(py_with_tone);
    std::string_view py = strip_tone(py_with_tone);
    std::string tone_suffix;
    if (tone != '5') tone_suffix = tone;

    std::string_view init = find_initial(py);
    std::string_view fin;
    if (!init.empty()) fin = py.substr(init.size());
    else fin = py;

    if (fin == "iu") return std::string(init) + "iou" + tone_suffix;
    if (fin == "ui") return std::string(init) + "uei" + tone_suffix;
    if (fin == "un") return std::string(init) + "uen" + tone_suffix;
    return std::string(py_with_tone);
}

// ────────────────────────────────────────────────────────────────────────
// Normalize y/w prefixes
// ────────────────────────────────────────────────────────────────────────

inline std::string normalize_yw_prefix(std::string_view py_with_tone) {
    char tone = extract_tone(py_with_tone);
    std::string_view py = strip_tone(py_with_tone);
    std::string tone_suffix;
    if (tone != '5') tone_suffix = tone;

    std::string result;
    if (!py.empty() && py[0] == 'y') {
        if (py == "yi") result = "i";
        else if (py == "yin") result = "in";
        else if (py == "ying") result = "ing";
        else if (py == "yu") result = "v";
        else if (py == "yuan") result = "van";
        else if (py == "yue") result = "ve";
        else if (py == "yun") result = "vn";
        else if (py == "yan") result = "ian";
        else if (py == "yang") result = "iang";
        else if (py == "yao") result = "iao";
        else if (py == "ye") result = "ie";
        else if (py == "yong") result = "iong";
        else if (py == "you") result = "iou";
        else if (py.size() >= 3 && py.substr(0, 2) == "yu") {
            result = "v" + std::string(py.substr(2));
        }
    } else if (!py.empty() && py[0] == 'w') {
        if (py == "wu") result = "u";
        else if (py == "wo") result = "uo";
        else {
            result = "u" + std::string(py.substr(1));
        }
    }

    if (result.empty()) return std::string(py_with_tone);
    return result + tone_suffix;
}

struct PinyinToken {
    std::string text;
    std::string pinyin;
    uint32_t cp = 0;
    int group = -1;
    bool syllable = false;
};

inline void set_tone(std::string & pinyin, char tone) {
    if (!pinyin.empty() && pinyin.back() >= '1' && pinyin.back() <= '5') {
        pinyin.back() = tone;
    } else {
        pinyin.push_back(tone);
    }
}

inline bool previous_syllable_is_digit(const std::vector<PinyinToken> & tokens,
                                       size_t i) {
    if (i == 0) return false;
    for (size_t j = i; j-- > 0;) {
        if (tokens[j].syllable) return is_chinese_digit(tokens[j].cp);
        if (!tokens[j].text.empty() && !std::isspace(static_cast<unsigned char>(tokens[j].text[0]))) {
            return false;
        }
    }
    return false;
}

inline bool next_syllable(const std::vector<PinyinToken> & tokens,
                          size_t i,
                          bool same_group_only,
                          size_t & next_i) {
    for (size_t j = i + 1; j < tokens.size(); ++j) {
        if (same_group_only && tokens[j].group != tokens[i].group) return false;
        if (tokens[j].syllable) {
            next_i = j;
            return true;
        }
        if (!tokens[j].text.empty() && !std::isspace(static_cast<unsigned char>(tokens[j].text[0]))) {
            return false;
        }
    }
    return false;
}

[[maybe_unused]] inline void apply_tone_sandhi(std::vector<PinyinToken> & tokens) {
    // Third-tone chains: all but the last third tone surface as second tone.
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (!tokens[i].syllable || extract_tone(tokens[i].pinyin) != '3') continue;
        size_t next_i = 0;
        if (next_syllable(tokens, i, true, next_i) &&
            extract_tone(tokens[next_i].pinyin) == '3') {
            set_tone(tokens[i].pinyin, '2');
        }
    }

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (!tokens[i].syllable) continue;

        size_t next_i = 0;
        const bool has_next = next_syllable(tokens, i, false, next_i);
        const char next_tone = has_next ? extract_tone(tokens[next_i].pinyin) : '5';
        const bool in_digit_run =
            is_chinese_digit(tokens[i].cp) &&
            (previous_syllable_is_digit(tokens, i) ||
             (has_next && is_chinese_digit(tokens[next_i].cp)));

        if (tokens[i].text == "一" && !in_digit_run) {
            if (!has_next || (i > 0 && tokens[i - 1].text == "第")) {
                set_tone(tokens[i].pinyin, '1');
            } else if (next_tone == '4') {
                set_tone(tokens[i].pinyin, '2');
            } else if (next_tone == '5') {
                set_tone(tokens[i].pinyin, '2');
            } else {
                set_tone(tokens[i].pinyin, '4');
            }
        } else if (tokens[i].text == "不" && has_next && next_tone == '4') {
            set_tone(tokens[i].pinyin, '2');
        }
    }
}

std::vector<PinyinToken> make_pinyin_tokens(const std::string & text) {
    std::string normalized = num_to_chinese(text);
    normalized = map_punctuation(normalized);
    std::vector<Utf8Char> chars = split_utf8_chars(normalized);

    std::vector<PinyinToken> tokens;
    tokens.reserve(chars.size());

    size_t i = 0;
    int next_group = 0;
    while (i < chars.size()) {
        const auto & ch = chars[i];
        if (!ch.is_zh) {
            if (ch.cp < 0x80 && std::isalnum(static_cast<unsigned char>(ch.text[0]))) {
                std::string run;
                size_t j = i;
                while (j < chars.size() && !chars[j].is_zh &&
                       chars[j].cp < 0x80 &&
                       std::isalnum(static_cast<unsigned char>(chars[j].text[0]))) {
                    run += chars[j].text;
                    ++j;
                }
                tokens.push_back({run, {}, 0, -1, false});
                i = j;
                continue;
            }
            tokens.push_back({ch.text, {}, ch.cp, -1, false});
            ++i;
            continue;
        }

        if (const LexemeEntry * entry = find_lexeme(chars, i)) {
            const auto pinyins = split_pinyin_list(entry->pinyin);
            const int group = next_group++;
            for (size_t j = 0; j < pinyins.size() && i + j < chars.size(); ++j) {
                tokens.push_back({chars[i + j].text, std::string(pinyins[j]), chars[i + j].cp, group, true});
            }
            i += pinyins.size();
            continue;
        }

        std::string_view pinyin = pinyin_for_cp(ch.cp);
        if (!pinyin.empty()) {
            tokens.push_back({ch.text, std::string(pinyin), ch.cp, next_group++, true});
        }
        ++i;
    }

    // Kokoro was trained on pypinyin output WITHOUT tone sandhi.
    // Applying sandhi produces phonemes the model doesn't recognize well.
    // apply_tone_sandhi(tokens);
    return tokens;
}

inline bool is_ascii_punctuation(std::string_view s) {
    return s.size() == 1 &&
           (s[0] == ',' || s[0] == '.' || s[0] == '!' ||
            s[0] == '?' || s[0] == ':' || s[0] == ';');
}

inline void trim_trailing_spaces(std::string & out) {
    while (!out.empty() && out.back() == ' ') out.pop_back();
}

inline void append_rendered_part(std::string & out,
                                 std::string_view part,
                                 bool syllable) {
    if (part.empty()) return;

    if (syllable) {
        if (!out.empty() && out.back() != ' ') out.push_back(' ');
        out.append(part);
        return;
    }

    if (part.size() == 1 && std::isspace(static_cast<unsigned char>(part[0]))) {
        if (!out.empty() && out.back() != ' ') out.push_back(' ');
        return;
    }

    if (is_ascii_punctuation(part)) {
        trim_trailing_spaces(out);
        out.append(part);
        out.push_back(' ');
        return;
    }

    if (!out.empty() && out.back() != ' ') out.push_back(' ');
    out.append(part);
}

inline std::string collapse_spaces(std::string_view text) {
    std::string collapsed;
    collapsed.reserve(text.size());
    bool prev_space = true;
    for (unsigned char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!prev_space) collapsed.push_back(' ');
            prev_space = true;
        } else {
            collapsed.push_back(static_cast<char>(c));
            prev_space = false;
        }
    }
    trim_trailing_spaces(collapsed);
    return collapsed;
}

} // namespace

// ────────────────────────────────────────────────────────────────────────
// pinyin_to_ipa: TONE3 pinyin → Kokoro IPA
// ────────────────────────────────────────────────────────────────────────

std::string pinyin_to_ipa(std::string_view pinyin_tone3) {
    if (pinyin_tone3.empty()) return {};

    // Step 1: Expand abbreviations (iu→iou, ui→uei, un→uen)
    std::string expanded = expand_pinyin_abbrevs(pinyin_tone3);

    // Step 2: Normalize y/w prefixes
    std::string normalized = normalize_yw_prefix(expanded);

    char tone = extract_tone(normalized);
    std::string_view py = strip_tone(normalized);
    std::string_view tm = tone_marker(tone);

    // ── Check interjections ──────────────────────────────────────
    for (const auto & entry : g_interjections) {
        if (py == entry.pinyin) return apply_tone(entry.ipa, tm);
    }

    // ── Check syllabic consonants ────────────────────────────────
    for (const auto & entry : g_syllabic_consonants) {
        if (py == entry.pinyin) return apply_tone(entry.ipa, tm);
    }

    // ── Find initial + final ────────────────────────────────────
    std::string_view initial = find_initial(py);
    std::string final_storage;
    std::string_view final;
    if (!initial.empty()) final = py.substr(initial.size());
    else final = py;

    // In pinyin, j/q/x combine with ü but are written with plain u.
    if ((initial == "j" || initial == "q" || initial == "x") &&
        (final == "u" || final == "ue" || final == "uan" ||
         final == "un" || final == "uen")) {
        final_storage = final == "uen" ? "vn" : "v";
        if (final != "uen") final_storage.append(final.substr(1));
        final = final_storage;
    }

    // ── Build IPA ────────────────────────────────────────────────
    std::string ipa;

    if (!initial.empty()) {
        ipa.append(lookup_initial_ipa(initial));
    }

    std::string_view final_ipa;
    if (!initial.empty() && final == "i" && is_zh_ch_sh_initial(initial)) {
        final_ipa = FINAL_I_AFTER_ZH_CH_SH_R;
    } else if (!initial.empty() && final == "i" && is_z_c_s_initial(initial)) {
        final_ipa = FINAL_I_AFTER_Z_C_S;
    } else {
        final_ipa = lookup_final_ipa(final);
    }

    if (!final_ipa.empty()) {
        ipa.append(apply_tone(final_ipa, tm));
    }

    // ── Post-processing: replace syllabic consonants → ɨ ────────
    ipa = replace_syllabic_consonants(ipa);

    return ipa;
}

// ────────────────────────────────────────────────────────────────────────
// Main entry point: g2p_chinese
// ────────────────────────────────────────────────────────────────────────

bool g2p_chinese(const std::string & text, std::string & phonemes, std::string & error) {
    if (text.empty()) {
        phonemes.clear();
        return true;
    }

    error.clear();

    std::string result;
    int prev_group = -2;  // sentinel: no previous group
    bool prev_was_syllable = false;

    for (const auto & token : make_pinyin_tokens(text)) {
        if (token.syllable) {
            std::string ipa = pinyin_to_ipa(token.pinyin);
            if (ipa.empty()) continue;

            const bool same_group = prev_was_syllable && token.group == prev_group;
            if (!same_group && !result.empty() && result.back() != ' ') {
                result.push_back(' ');
            }

            result.append(ipa);
            prev_group = token.group;
            prev_was_syllable = true;
        } else {
            append_rendered_part(result, token.text, false);
            prev_group = -2;
            prev_was_syllable = false;
        }
    }

    result = replace_syllabic_consonants(result);
    result = strip_combining_marks(result);
    phonemes = collapse_spaces(result);
    return true;
}

bool g2p_chinese_to_pinyin(const std::string & text, std::string & pinyin, std::string & error) {
    if (text.empty()) {
        pinyin.clear();
        return true;
    }

    error.clear();

    std::string result;
    for (const auto & token : make_pinyin_tokens(text)) {
        if (token.syllable) {
            append_rendered_part(result, token.pinyin, true);
        } else {
            append_rendered_part(result, token.text, false);
        }
    }

    pinyin = collapse_spaces(result);
    return true;
}

} // namespace kokopop::g2p::zh
