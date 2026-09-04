#include "arch/sanotts/sano_frontend_text.h"
#include "arch/sanotts/sano_tokenizer.h"
#include "sanotts_corpus.h"

#include <map>
#include <string>
#include <vector>

namespace sano = kokopop::sano;

// ---------------------------------------------------------------------------
// The G2P parity gate
//
// Wrong phoneme ids produce confident, wrong audio with no error anywhere, so
// this is the project's highest-risk surface and the gate is exact equality —
// not a similarity threshold that could hide a systematic error on one
// language. tests/data/sanotts_g2p_corpus.json is the reference side, produced
// by tools/gen_sanotts_g2p_corpus.py from phonemizer + misaki against a pinned
// espeak-ng.
//
// A divergence is either fixed or recorded here as a known incompatibility
// that blocks the voice. It is never absorbed by relaxing the comparison.
// ---------------------------------------------------------------------------

TEST_CASE("sanotts_g2p_corpus_is_present_and_pinned") {
    const kokopop::test::G2pCorpus & corpus = kokopop::test::sanotts_g2p_corpus();
    REQUIRE_MESSAGE(corpus.loaded, corpus.load_error);
    CHECK(corpus.entries.size() > 100);
    // The gate is only meaningful against a known espeak-ng.
    CHECK_EQ(corpus.espeak_version, "1.52.0");
    CHECK_FALSE(corpus.unicode_version.empty());

    size_t misaki = 0;
    size_t piper = 0;
    std::map<std::string, size_t> voices;
    for (const auto & entry : corpus.entries) {
        (entry.frontend == "misaki" ? misaki : piper) += 1;
        voices[entry.voice] += 1;
    }
    CHECK(misaki > 40);
    CHECK(piper > 90);
    // English piperlite, plus the two non-English packs, plus the vocos voice.
    CHECK(voices.count("heart") == 1);
    CHECK(voices.count("amy") == 1);
    CHECK(voices.count("kristin") == 1);
    CHECK(voices.count("vi") == 1);
    CHECK(voices.count("id") == 1);
}

TEST_CASE("sanotts_misaki_phonemization_matches_the_reference_exactly") {
    const kokopop::test::G2pCorpus & corpus = kokopop::test::sanotts_g2p_corpus();
    REQUIRE_MESSAGE(corpus.loaded, corpus.load_error);

    size_t checked = 0;
    std::vector<std::string> divergences;
    for (const auto & entry : corpus.entries) {
        if (entry.frontend != "misaki") {
            continue;
        }
        std::string phonemes;
        std::string error;
        REQUIRE_MESSAGE(
            sano::phonemize_misaki_sanotts(entry.text, entry.espeak_voice, phonemes, error),
            error);
        if (phonemes != entry.phonemes) {
            divergences.push_back("text=[" + entry.text + "] got=[" + phonemes +
                                  "] want=[" + entry.phonemes + "]");
        }
        ++checked;
    }
    CHECK(checked > 40);
    for (const std::string & line : divergences) {
        MESSAGE(line);
    }
    CHECK_EQ(divergences.size(), 0u);
}

TEST_CASE("sanotts_piper_phonemization_matches_the_reference_exactly") {
    const kokopop::test::G2pCorpus & corpus = kokopop::test::sanotts_g2p_corpus();
    REQUIRE_MESSAGE(corpus.loaded, corpus.load_error);

    size_t checked = 0;
    std::vector<std::string> divergences;
    for (const auto & entry : corpus.entries) {
        if (entry.frontend != "piper") {
            continue;
        }
        std::string phonemes;
        std::string error;
        REQUIRE_MESSAGE(
            sano::phonemize_piper_sanotts(entry.text, entry.espeak_voice, phonemes, error),
            error);
        if (phonemes != entry.phonemes) {
            divergences.push_back(entry.voice + " text=[" + entry.text + "] got=[" +
                                  phonemes + "] want=[" + entry.phonemes + "]");
        }
        ++checked;
    }
    CHECK(checked > 90);
    for (const std::string & line : divergences) {
        MESSAGE(line);
    }
    CHECK_EQ(divergences.size(), 0u);
}

// The ids are what reaches the model, so they are gated separately from the
// phoneme strings: an identical string tokenized differently is still wrong
// audio.
TEST_CASE("sanotts_g2p_ids_match_the_reference_exactly") {
    const kokopop::test::G2pCorpus & corpus = kokopop::test::sanotts_g2p_corpus();
    REQUIRE_MESSAGE(corpus.loaded, corpus.load_error);

    const sano::NfdTable & nfd = kokopop::test::sanotts_nfd_table();
    REQUIRE(nfd.present());

    size_t checked = 0;
    std::vector<std::string> divergences;
    for (const auto & entry : corpus.entries) {
        const sano::TokenTable * table = kokopop::test::sanotts_token_table(entry.voice);
        if (table == nullptr) {
            continue;
        }
        std::string phonemes;
        std::string error;
        const bool ok = entry.frontend == "misaki"
            ? sano::phonemize_misaki_sanotts(entry.text, entry.espeak_voice, phonemes, error)
            : sano::phonemize_piper_sanotts(entry.text, entry.espeak_voice, phonemes, error);
        REQUIRE_MESSAGE(ok, error);

        std::vector<uint32_t> ids;
        const bool tokenized = entry.frontend == "misaki"
            ? sano::tokenize_misaki(phonemes, *table, ids, error)
            : sano::tokenize_piper(phonemes, *table, nfd, ids, error);
        if (!tokenized) {
            divergences.push_back(entry.voice + " text=[" + entry.text +
                                  "] tokenizer failed: " + error);
            continue;
        }
        if (ids != entry.ids) {
            std::string got;
            for (uint32_t id : ids) {
                got += std::to_string(id) + " ";
            }
            std::string want;
            for (uint32_t id : entry.ids) {
                want += std::to_string(id) + " ";
            }
            divergences.push_back(entry.voice + " text=[" + entry.text + "] got=[" +
                                  got + "] want=[" + want + "]");
        }
        ++checked;
    }
    CHECK(checked > 100);
    for (const std::string & line : divergences) {
        MESSAGE(line);
    }
    CHECK_EQ(divergences.size(), 0u);
}

// ---------------------------------------------------------------------------
// The pieces of the contract, pinned individually
// ---------------------------------------------------------------------------

// misaki's rewrites are sequential whole-string replacements in a fixed order,
// not one longest-match pass. These cases are the ones where the two differ.
TEST_CASE("sanotts_misaki_e2m_applies_its_rules_in_order") {
    // `e^U+026A -> A` must fire before the bare `e -> A`. The other way
    // round, `e^U+026A` would become `A^U+026A` and leave a stray vowel.
    CHECK_EQ(sano::normalize_misaki_e2m("e^\xc9\xaa"), "A");
    CHECK_EQ(sano::normalize_misaki_e2m("de^\xc9\xaa" "d"), "dAd");
    // A bare `e` still maps, so an untied one is not left as-is...
    CHECK_EQ(sano::normalize_misaki_e2m("de"), "dA");
    // ...while a bare U+026A is untouched, which is what makes the
    // ordering observable rather than a matter of taste.
    CHECK_EQ(sano::normalize_misaki_e2m("d\xc9\xaa" "d"), "d\xc9\xaa" "d");

    // Tied forms.
    CHECK_EQ(sano::normalize_misaki_e2m("a^ɪ"), "I");
    CHECK_EQ(sano::normalize_misaki_e2m("a^ʊ"), "W");
    CHECK_EQ(sano::normalize_misaki_e2m("d^ʒ"), "ʤ");
    CHECK_EQ(sano::normalize_misaki_e2m("e^ɪ"), "A");
    CHECK_EQ(sano::normalize_misaki_e2m("t^ʃ"), "ʧ");
    CHECK_EQ(sano::normalize_misaki_e2m("ɔ^ɪ"), "Y");
    CHECK_EQ(sano::normalize_misaki_e2m("o^ʊ"), "O");

    // A tied syllabic l becomes a superscript schwa plus l; an untied one is
    // two phonemes and must survive untouched. This is the case that makes the
    // tie mandatory rather than cosmetic.
    CHECK_EQ(sano::normalize_misaki_e2m("mˈɑːdə^l"), "mˈɑdᵊl");
    CHECK_EQ(sano::normalize_misaki_e2m("hˈændəl"), "hˈændəl");

    // The tie is stripped last, so a tie that no rule consumed leaves no trace.
    CHECK_EQ(sano::normalize_misaki_e2m("k^p"), "kp");
}

TEST_CASE("sanotts_misaki_e2m_rewrites_syllabic_consonants") {
    // re.sub(r'(\S)̩', r'ᵊ\1'): the marker moves in front as ᵊ.
    CHECK_EQ(sano::normalize_misaki_e2m("bʌn̩"), "bʌᵊn");
    CHECK_EQ(sano::normalize_misaki_e2m("mɛtəl̩"), "mɛtəᵊl");
    CHECK_EQ(sano::normalize_misaki_e2m("\xcc\xa9"), "");
    // The pattern is `(\S)U+0329`, so a marker with no non-space before it
    // matches nothing and is dropped instead of becoming a stray schwa.
    CHECK_EQ(sano::normalize_misaki_e2m("a \xcc\xa9" "b"), "a b");
}

TEST_CASE("sanotts_misaki_e2m_applies_the_american_tail") {
    CHECK_EQ(sano::normalize_misaki_e2m("wˈɜːɹd"), "wˈɜɹd");
    CHECK_EQ(sano::normalize_misaki_e2m("hˈɜːd"), "hˈɜɹd");
    CHECK_EQ(sano::normalize_misaki_e2m("hɪəɹ"), "hiəɹ");
    // The length mark goes away entirely.
    CHECK_EQ(sano::normalize_misaki_e2m("tˈuː"), "tˈu");
    // Flap and glottal stop.
    CHECK_EQ(sano::normalize_misaki_e2m("ˈa^ɪɾəm"), "ˈITəm");
    CHECK_EQ(sano::normalize_misaki_e2m("ʔʌ"), "tʌ");
    // Single-character mappings.
    CHECK_EQ(sano::normalize_misaki_e2m("ɐ"), "ə");
    CHECK_EQ(sano::normalize_misaki_e2m("ɚ"), "əɹ");
    CHECK_EQ(sano::normalize_misaki_e2m("r"), "ɹ");
    CHECK_EQ(sano::normalize_misaki_e2m("x"), "k");
    CHECK_EQ(sano::normalize_misaki_e2m("ç"), "k");
    CHECK_EQ(sano::normalize_misaki_e2m("ɬ"), "l");
    // The combining tilde is dropped; the letter it sat on is untouched,
    // since a bare `a` has no rewrite of its own.
    CHECK_EQ(sano::normalize_misaki_e2m("a\xcc\x83"), "a");
    // A precomposed character is not decomposed first, so it survives: the
    // misaki path does no NFD, only the Piper tokenizer does.
    CHECK_EQ(sano::normalize_misaki_e2m("\xc3\xa3"), "\xc3\xa3");
}

TEST_CASE("sanotts_misaki_e2m_trims_its_input") {
    CHECK_EQ(sano::normalize_misaki_e2m("  bʌ  "), "bʌ");
    CHECK_EQ(sano::normalize_misaki_e2m(""), "");
    CHECK_EQ(sano::normalize_misaki_e2m("   "), "");
}

// espeak drops punctuation, so it has to be hidden and restored. Splitting the
// input at the marks is not an optimisation: it changes espeak's clause
// context, and therefore its stress assignment, to match the reference.
TEST_CASE("sanotts_punctuation_is_restored_where_it_was") {
    std::string phonemes;
    std::string error;

    REQUIRE(sano::phonemize_misaki_sanotts("Hello, world.", "gmw/en-US", phonemes, error));
    CHECK(phonemes.find(',') != std::string::npos);
    CHECK_EQ(phonemes.back(), '.');

    REQUIRE(sano::phonemize_misaki_sanotts("One. Two. Three.", "gmw/en-US", phonemes, error));
    // Three sentence marks survive, in order.
    size_t dots = 0;
    for (char c : phonemes) {
        dots += (c == '.') ? 1u : 0u;
    }
    CHECK_EQ(dots, 3u);

    // Punctuation-only input has nothing to phonemize; the marks are the answer.
    REQUIRE(sano::phonemize_misaki_sanotts("...", "gmw/en-US", phonemes, error));
    CHECK_EQ(phonemes, "...");

    // Leading punctuation stays in front.
    REQUIRE(sano::phonemize_misaki_sanotts("\"quoted\" text", "gmw/en-US", phonemes, error));
    CHECK_EQ(phonemes.front(), '"');
}

// Piper's set is narrower than misaki's: an em dash is a phoneme-table symbol
// for misaki and not punctuation for Piper.
TEST_CASE("sanotts_the_two_contracts_use_different_punctuation_sets") {
    const std::string misaki_marks = sano::MISAKI_PUNCTUATION_MARKS;
    const std::string piper_marks = sano::PIPER_PUNCTUATION_MARKS;
    CHECK(misaki_marks.find("—") != std::string::npos);  // em dash
    CHECK(piper_marks.find("—") == std::string::npos);
    CHECK(piper_marks.find('\'') != std::string::npos);       // apostrophe
    CHECK(misaki_marks.find('\'') == std::string::npos);
}

// Malformed input used to be dropped silently at the first bad byte, which
// turned a broken request into a shorter, plausible utterance.
TEST_CASE("sanotts_phonemization_rejects_invalid_utf8_input") {
    std::string phonemes;
    std::string error;
    const std::string truncated = "hello \xc3";  // a lone UTF-8 lead byte
    CHECK_FALSE(sano::phonemize_misaki_sanotts(truncated, "gmw/en-US", phonemes, error));
    CHECK(error.find("UTF-8") != std::string::npos);

    CHECK_FALSE(sano::phonemize_piper_sanotts(truncated, "gmw/en-US", phonemes, error));
    CHECK(error.find("UTF-8") != std::string::npos);
}

TEST_CASE("sanotts_phonemization_rejects_an_unknown_espeak_voice") {
    std::string phonemes;
    std::string error;
    CHECK_FALSE(sano::phonemize_misaki_sanotts("hello", "no/such-voice", phonemes, error));
    CHECK(error.find("voice") != std::string::npos);
}
