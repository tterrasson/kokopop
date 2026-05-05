#include "test_helpers.h"
#include "synthesis/g2p/zh_g2p.h"

// Helper: check if string contains any tone marker
static bool has_tone_marker(const std::string & s) {
    return s.find("→") != std::string::npos ||
           s.find("↗") != std::string::npos ||
           s.find("↓") != std::string::npos ||
           s.find("↘") != std::string::npos;
}

// Helper: check if string contains UTF-8 encoded character
static bool contains_utf8(const std::string & s, uint32_t cp) {
    size_t i = 0;
    while (i < s.size()) {
        size_t len;
        uint32_t c = 0;
        unsigned char b = static_cast<unsigned char>(s[i]);
        if ((b & 0x80) == 0) { len = 1; c = b; }
        else if ((b & 0xE0) == 0xC0) { len = 2; c = ((b & 0x1F) << 6) | (static_cast<unsigned char>(s[i+1]) & 0x3F); }
        else if ((b & 0xF0) == 0xE0) { len = 3; c = ((b & 0x0F) << 12) | (static_cast<unsigned char>(s[i+1]) & 0x3F) << 6 | (static_cast<unsigned char>(s[i+2]) & 0x3F); }
        else if ((b & 0xF8) == 0xF0) { len = 4; c = ((b & 0x07) << 18) | (static_cast<unsigned char>(s[i+1]) & 0x3F) << 12 | (static_cast<unsigned char>(s[i+2]) & 0x3F) << 6 | (static_cast<unsigned char>(s[i+3]) & 0x3F); }
        else { len = 1; }
        if (c == cp) return true;
        i += len;
    }
    return false;
}

// ─── Chinese G2P tests ─────────────────────────────────────────────────

TEST_CASE("zh_g2p_chinese_simple_sentence") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("你好世界", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
    CHECK(has_tone_marker(phonemes));
}

TEST_CASE("zh_g2p_complex_sentence") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese(
        "只要我沒有道德，就没有人能绑架我。", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
    CHECK(phonemes.find(",") != std::string::npos);
    CHECK(phonemes.find(".") != std::string::npos);
}

TEST_CASE("zh_g2p_punctuation_mapping") {
    std::string phonemes, error;

    CHECK(kokopop::g2p::zh::g2p_chinese("你好，世界", phonemes, error));
    CHECK(error.empty());
    CHECK(phonemes.find(",") != std::string::npos);

    CHECK(kokopop::g2p::zh::g2p_chinese("你好。世界", phonemes, error));
    CHECK(error.empty());
    CHECK(phonemes.find(".") != std::string::npos);
}

TEST_CASE("zh_g2p_numbers_converted") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("123", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
}

TEST_CASE("zh_g2p_mixed_script") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("Hello你好", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
    CHECK(phonemes.find("Hello") != std::string::npos);
}

TEST_CASE("zh_g2p_empty_input") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("", phonemes, error));
    CHECK(error.empty());
    CHECK(phonemes.empty());
}

TEST_CASE("zh_g2p_tone1_character") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("一", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
    CHECK(has_tone_marker(phonemes));
}

TEST_CASE("zh_g2p_tone4_character") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("二", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
    CHECK(has_tone_marker(phonemes));
}

TEST_CASE("zh_g2p_tone3_character") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("我", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
    CHECK(has_tone_marker(phonemes));
}

TEST_CASE("zh_g2p_iu_abbreviation") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("就", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
    // 就 → jiu4 → iu→iou → ʨ + jou̯↘ → ʨjou↘ (after stripping combining marks)
    CHECK(phonemes.find("jou") != std::string::npos);
    CHECK(phonemes.find("↘") != std::string::npos);
}

TEST_CASE("zh_g2p_liu") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("六", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
    // 六 → liu4 → l + iou↘ → liou↘ → ljou↘
    CHECK(phonemes.find("jou") != std::string::npos);
    CHECK(phonemes.find("↘") != std::string::npos);
}

TEST_CASE("zh_g2p_special_syllables_zhi") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("知", phonemes, error));
    CHECK(error.empty());
    CHECK(contains_utf8(phonemes, 0xAB67)); // ꭧ = tʂ
    CHECK(contains_utf8(phonemes, 0x0268)); // ɨ
}

TEST_CASE("zh_g2p_special_syllables_chi") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("吃", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
}

TEST_CASE("zh_g2p_special_syllables_shi") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("是", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
}

TEST_CASE("zh_g2p_zi_special_final") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("子", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
    CHECK(contains_utf8(phonemes, 0x0268)); // ɨ
}

TEST_CASE("zh_g2p_er_interjection") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("二", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
}

TEST_CASE("zh_g2p_ü_finals") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("女", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());

    CHECK(kokopop::g2p::zh::g2p_chinese("绿", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
}

// ─── Integration: phonemize_text with zh voice ────────────────────────

TEST_CASE("phonemize_text_zh_routes_to_g2p_chinese") {
    std::string phonemes, error;
    CHECK(kokopop::phonemize_text("你好", "zf_xiaoni", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
    CHECK(has_tone_marker(phonemes));
}

TEST_CASE("phonemize_text_zh_full_sentence") {
    std::string phonemes, error;
    CHECK(kokopop::phonemize_text(
        "只要我沒有道德，就没有人能绑架我。", "zf_xiaoni", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
}

TEST_CASE("phonemize_text_non_zh_unchanged") {
    std::string phonemes, error;
    CHECK(kokopop::phonemize_text("Hello world", "af_heart", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());

    CHECK(kokopop::phonemize_text("Bonjour", "ff_siwis", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
}

// ─── pinyin_to_ipa unit tests ─────────────────────────────────────────

TEST_CASE("pinyin_to_ipa_ba1") {
    using kokopop::g2p::zh::pinyin_to_ipa;
    CHECK_EQ(pinyin_to_ipa("ba1"), "pa→");
}

TEST_CASE("pinyin_to_ipa_da1") {
    using kokopop::g2p::zh::pinyin_to_ipa;
    CHECK_EQ(pinyin_to_ipa("da1"), "ta→");
}

TEST_CASE("pinyin_to_ipa_de1") {
    using kokopop::g2p::zh::pinyin_to_ipa;
    CHECK_EQ(pinyin_to_ipa("de1"), "tɤ→");
}

TEST_CASE("pinyin_to_ipa_tone1") {
    using kokopop::g2p::zh::pinyin_to_ipa;
    CHECK(pinyin_to_ipa("ma1").find("→") != std::string::npos);
}

TEST_CASE("pinyin_to_ipa_tone2") {
    using kokopop::g2p::zh::pinyin_to_ipa;
    CHECK(pinyin_to_ipa("ma2").find("↗") != std::string::npos);
}

TEST_CASE("pinyin_to_ipa_tone3") {
    using kokopop::g2p::zh::pinyin_to_ipa;
    CHECK(pinyin_to_ipa("ma3").find("↓") != std::string::npos);
}

TEST_CASE("pinyin_to_ipa_tone4") {
    using kokopop::g2p::zh::pinyin_to_ipa;
    CHECK(pinyin_to_ipa("ma4").find("↘") != std::string::npos);
}

TEST_CASE("pinyin_to_ipa_zhi1_syllabic") {
    using kokopop::g2p::zh::pinyin_to_ipa;
    std::string zhi = pinyin_to_ipa("zhi1");
    CHECK(contains_utf8(zhi, 0xAB67)); // ꭧ = tʂ
    CHECK(contains_utf8(zhi, 0x0268)); // ɨ
}

TEST_CASE("pinyin_to_ipa_zi1_syllabic") {
    using kokopop::g2p::zh::pinyin_to_ipa;
    std::string zi = pinyin_to_ipa("zi1");
    CHECK(contains_utf8(zi, 0x0268)); // ɨ
}

TEST_CASE("pinyin_to_ipa_diphthong_hao4") {
    using kokopop::g2p::zh::pinyin_to_ipa;
    CHECK(pinyin_to_ipa("hao4").find("au̯↘") != std::string::npos);
}
