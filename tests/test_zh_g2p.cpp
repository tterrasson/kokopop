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

static std::string zh_pinyin(const std::string & text) {
    std::string pinyin, error;
    CHECK(kokopop::g2p::zh::g2p_chinese_to_pinyin(text, pinyin, error));
    CHECK(error.empty());
    return pinyin;
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

TEST_CASE("zh_g2p_pinyin_diagnostic_simple_sandhi") {
    CHECK_EQ(zh_pinyin("你好"), "ni2 hao3");
    CHECK_EQ(zh_pinyin("很好"), "hen2 hao3");
    CHECK_EQ(zh_pinyin("我想买"), "wo2 xiang2 mai3");
}

TEST_CASE("zh_g2p_yi_and_bu_sandhi") {
    CHECK_EQ(zh_pinyin("一杯 一个 一天 一样"),
             "yi4 bei1 yi2 ge5 yi4 tian1 yi2 yang4");
    CHECK_EQ(zh_pinyin("不是 不好 不要"),
             "bu2 shi4 bu4 hao3 bu2 yao4");
    CHECK_EQ(zh_pinyin("第一名"), "di4 yi1 ming2");
}

TEST_CASE("zh_g2p_polyphonic_lexicon") {
    CHECK_EQ(zh_pinyin("银行 旅行 行为"),
             "yin2 hang2 lv3 xing2 xing2 wei2");
    CHECK_EQ(zh_pinyin("重庆 重要 重复"),
             "chong2 qing4 zhong4 yao4 chong2 fu4");
    CHECK_EQ(zh_pinyin("长大 长城 校长"),
             "zhang3 da4 chang2 cheng2 xiao4 zhang3");
    CHECK_EQ(zh_pinyin("只要 一只 音乐 快乐"),
             "zhi3 yao4 yi4 zhi1 yin1 yue4 kuai4 le4");
    CHECK_EQ(zh_pinyin("为了 因为 还钱 还有 慢慢地"),
             "wei4 le5 yin1 wei4 huan2 qian2 hai2 you3 man4 man4 de5");
}

TEST_CASE("zh_g2p_numbers_dates_times_percentages") {
    CHECK_EQ(zh_pinyin("2026年5月6日"),
             "er4 ling2 er4 liu4 nian2 wu3 yue4 liu4 ri4");
    CHECK_EQ(zh_pinyin("14:05"), "shi2 si4 dian3 ling2 wu3 fen1");
    CHECK_EQ(zh_pinyin("12%"), "bai3 fen1 zhi1 shi2 er4");
    CHECK_EQ(zh_pinyin("13800138000"),
             "yi1 san1 ba1 ling2 ling2 yi1 san1 ba1 ling2 ling2 ling2");
    CHECK_EQ(zh_pinyin("12.50元"), "shi2 er4 dian3 wu3 ling2 yuan2");
}

TEST_CASE("zh_g2p_long_sentence_lexical_boundaries") {
    CHECK_EQ(zh_pinyin("尽管 我们每天都在忙着工作和学习"),
             "jin2 guan3 wo3 men5 mei3 tian1 dou1 zai4 mang2 zhe5 gong1 zuo4 he2 xue2 xi2");
    CHECK_EQ(zh_pinyin("美好往往来自于这些平凡的瞬间"),
             "mei2 hao3 wang2 wang3 lai2 zi4 yu2 zhe4 xie1 ping2 fan2 de5 shun4 jian1");
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

TEST_CASE("pinyin_to_ipa_jqx_umlaut_finals") {
    using kokopop::g2p::zh::pinyin_to_ipa;
    CHECK(pinyin_to_ipa("xue2").find("ɥe↗") != std::string::npos);
    CHECK(pinyin_to_ipa("juan4").find("ɥɛ↘n") != std::string::npos);
    CHECK(pinyin_to_ipa("qun2").find("y↗n") != std::string::npos);
}

// ─── num_to_chinese — edge cases (bug fix: 10000 segfault) ────────────

TEST_CASE("zh_g2p_num_cardinals") {
    CHECK_EQ(zh_pinyin("0"), "ling2");
    // 1个: 一→yi1, sandhi before neutral→yi2
    CHECK_EQ(zh_pinyin("1个"), "yi2 ge5");
    // 10个: 十→shi2, 个→ge4 (non-lexicon lookup)
    CHECK_EQ(zh_pinyin("10个"), "shi2 ge4");
    // 100个: 一百→yi1 bai3, 个→ge4
    CHECK_EQ(zh_pinyin("100个"), "yi1 bai3 ge4");
    // 101个: 一百零一→yi1 bai3 ling2 yi1, 个→ge5 (lexicon "一个" matches)
    CHECK_EQ(zh_pinyin("101个"), "yi1 bai3 ling2 yi1 ge5");
    CHECK_EQ(zh_pinyin("9999"), "jiu3 qian1 jiu3 bai3 jiu3 shi2 jiu3");
    // 10000 must not segfault — digit-by-digit fallback
    CHECK_EQ(zh_pinyin("10000"), "yi1 ling2 ling2 ling2 ling2");
}

TEST_CASE("zh_g2p_num_percentages") {
    CHECK_EQ(zh_pinyin("0%"), "bai3 fen1 zhi1 ling2");
    CHECK_EQ(zh_pinyin("12%"), "bai3 fen1 zhi1 shi2 er4");
    // 100 = 一百 → yi1 bai3
    CHECK_EQ(zh_pinyin("100%"), "bai3 fen1 zhi1 yi1 bai3");
    // 10000% must not segfault — digit-by-digit fallback
    CHECK_EQ(zh_pinyin("10000%"), "bai3 fen1 zhi1 yi1 ling2 ling2 ling2 ling2");
}

TEST_CASE("zh_g2p_num_decimals") {
    CHECK_EQ(zh_pinyin("0.5元"), "ling2 dian3 wu3 yuan2");
    CHECK_EQ(zh_pinyin("3.14"), "san1 dian3 yi1 si4");
    CHECK_EQ(zh_pinyin("12.50元"), "shi2 er4 dian3 wu3 ling2 yuan2");
    // Large int part must not segfault
    CHECK_EQ(zh_pinyin("10000.5"), "yi1 ling2 ling2 ling2 ling2 dian3 wu3");
}

TEST_CASE("zh_g2p_num_times") {
    CHECK_EQ(zh_pinyin("0:00"), "ling2 dian3");
    CHECK_EQ(zh_pinyin("9:05"), "jiu3 dian3 ling2 wu3 fen1");
    CHECK_EQ(zh_pinyin("14:05"), "shi2 si4 dian3 ling2 wu3 fen1");
    CHECK_EQ(zh_pinyin("23:59"), "er4 shi2 san1 dian3 wu3 shi2 jiu3 fen1");
}

TEST_CASE("zh_g2p_num_years") {
    // 三 in digit-by-digit → san1 (dictionary tone)
    CHECK_EQ(zh_pinyin("2023年"), "er4 ling2 er4 san1 nian2");
    CHECK_EQ(zh_pinyin("2026年5月6日"), "er4 ling2 er4 liu4 nian2 wu3 yue4 liu4 ri4");
}

// ─── Tone sandhi — comprehensive ──────────────────────────────────────

TEST_CASE("zh_g2p_yi_sandhi_comprehensive") {
    // 一 alone → yi1 (citation tone)
    CHECK_EQ(zh_pinyin("一"), "yi1");
    // 一 at end of phrase (no next syllable) → yi1
    CHECK_EQ(zh_pinyin("唯一"), "wei2 yi1");
    // 第一 → yi1 (after 第)
    CHECK_EQ(zh_pinyin("第一名"), "di4 yi1 ming2");
    // 一 before tone-4 → yi2 (lexicon "一定")
    CHECK_EQ(zh_pinyin("一定"), "yi2 ding4");
    // 一 before tone-1 → yi4 (lexicon "一天")
    CHECK_EQ(zh_pinyin("一天"), "yi4 tian1");
    // 一 before neutral → yi2 (lexicon "一个")
    CHECK_EQ(zh_pinyin("一个"), "yi2 ge5");
}

TEST_CASE("zh_g2p_bu_sandhi_comprehensive") {
    // 不 alone → bu4 (citation tone)
    CHECK_EQ(zh_pinyin("不"), "bu4");
    // 不 before tone-4 → bu2 (lexicon "不是")
    CHECK_EQ(zh_pinyin("不是"), "bu2 shi4");
    // 不 before other tone → bu4 (lexicon "不好")
    CHECK_EQ(zh_pinyin("不好"), "bu4 hao3");
}

TEST_CASE("zh_g2p_third_tone_sandhi_chains") {
    // 你好 → ni3+3 → ni2 hao3 (lexicon + sandhi)
    CHECK_EQ(zh_pinyin("你好"), "ni2 hao3");
    // 很好 → hen3+3 → hen2 hao3 (lexicon + sandhi)
    CHECK_EQ(zh_pinyin("很好"), "hen2 hao3");
    // 我想买: 我→wo3 sandhi before 3rd→wo2, 想→xiang3 sandhi before 3rd→xiang2
    CHECK_EQ(zh_pinyin("我想买"), "wo2 xiang2 mai3");
}

// ─── Lexicon — longest match & mixed script ───────────────────────────

TEST_CASE("zh_g2p_lexicon_longest_match") {
    // 4-char entry should match before shorter entries
    CHECK_EQ(zh_pinyin("为人民服务"), "wei4 ren2 min2 fu2 wu4");
    // 3-char entry "为什么" vs "为" + "什么"
    CHECK_EQ(zh_pinyin("为什么"), "wei4 shen2 me5");
}

TEST_CASE("zh_g2p_traditional_simplified_mixed") {
    CHECK_EQ(zh_pinyin("學習時間"), "xue2 xi2 shi2 jian1");
    CHECK_EQ(zh_pinyin("学习時間"), "xue2 xi2 shi2 jian1");
    CHECK_EQ(zh_pinyin("學習时间"), "xue2 xi2 shi2 jian1");
    CHECK_EQ(zh_pinyin("没有沒有"), "mei2 you3 mei2 you3");
    // 尽管: 尽→jin3 sandhi before 3rd→jin2, 管→guan3
    CHECK_EQ(zh_pinyin("儘管"), "jin2 guan3");
    CHECK_EQ(zh_pinyin("尽管"), "jin2 guan3");
}

// ─── j/q/x + ü finals — all forms ─────────────────────────────────────

TEST_CASE("zh_g2p_jqx_u_finals_all_forms") {
    // j + u/ue/uan/un → ü
    CHECK_EQ(zh_pinyin("居"), "ju1");
    CHECK_EQ(zh_pinyin("决"), "jue2");
    CHECK_EQ(zh_pinyin("卷"), "juan3");
    CHECK_EQ(zh_pinyin("军"), "jun1");

    // q + u/ue/uan/un → ü
    CHECK_EQ(zh_pinyin("区"), "qu1");
    CHECK_EQ(zh_pinyin("全"), "quan2");
    CHECK_EQ(zh_pinyin("却"), "que4");
    CHECK_EQ(zh_pinyin("群"), "qun2");

    // x + u/ue/uan/un → ü
    CHECK_EQ(zh_pinyin("需"), "xu1");
    CHECK_EQ(zh_pinyin("选"), "xuan3");
    CHECK_EQ(zh_pinyin("学"), "xue2");
    CHECK_EQ(zh_pinyin("寻"), "xun2");
}

// ─── pinyin_to_ipa — j/q/x umlaut full coverage ───────────────────────

TEST_CASE("pinyin_to_ipa_jqx_umlaut_full") {
    using kokopop::g2p::zh::pinyin_to_ipa;
    // ju1 = j + vü → ʨ + y→
    CHECK(pinyin_to_ipa("ju1").find("y") != std::string::npos);
    CHECK(pinyin_to_ipa("ju1").find("→") != std::string::npos);

    // quan2 = q + van → ʨʰ + ɥɛ↗n
    CHECK(pinyin_to_ipa("quan2").find("ɥ") != std::string::npos);

    // xun2 = x + vn → ɕ + y↗n
    CHECK(pinyin_to_ipa("xun2").find("y") != std::string::npos);

    // jue2 = j + vüe → ʨ + ɥe↗
    CHECK(pinyin_to_ipa("jue2").find("ɥe") != std::string::npos);

    // juan3 = j + vüan → ʨ + ɥɛ↓n
    CHECK(pinyin_to_ipa("juan3").find("ɥɛ") != std::string::npos);
}

// ─── Full pipeline integration ────────────────────────────────────────

TEST_CASE("zh_g2p_full_pipeline_numbers_mixed") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("我有100块钱，买了3个苹果。", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
    CHECK(has_tone_marker(phonemes));
}

TEST_CASE("zh_g2p_full_pipeline_time_date") {
    std::string phonemes, error;
    CHECK(kokopop::g2p::zh::g2p_chinese("2024年12月25日14:30", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
}

TEST_CASE("zh_g2p_full_pipeline_10000_no_crash") {
    std::string phonemes, error;
    // Must not crash with 10000 in any numeric context
    CHECK(kokopop::g2p::zh::g2p_chinese("10000%", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());

    CHECK(kokopop::g2p::zh::g2p_chinese("10000.5", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());

    CHECK(kokopop::g2p::zh::g2p_chinese("价格是10000元", phonemes, error));
    CHECK(error.empty());
    CHECK(!phonemes.empty());
}
