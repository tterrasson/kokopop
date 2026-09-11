// Emotion styles: the text tags that select them, and their resolution.
//
// Neither needs a trained emotion pack. The tag splitter is driven by the same
// closure the architecture binds, and the resolver reads a `SanoEmotionWeights`
// a test can fill in — which is the point: what these gate is the contract
// between a name in the text and a vector, not the acoustics it produces.

#include "arch/sanotts/sano_arch.h"
#include "synthesis/chunker/chunker.h"

#include <doctest/doctest.h>

#include <string>
#include <vector>

namespace {

/// The style vocabulary of the shipped French pack, without its weights.
kokopop::SanoArch make_styled_arch() {
    kokopop::SanoArch arch;
    kokopop::SanoVoice voice;
    voice.desc.name = "fr";
    voice.desc.sample_rate = 22050;
    voice.emo.dim = 4;
    voice.emo.styles = {"neutral", "angry", "sad", "joyful", "laughing"};
    voice.emo.vectors = {
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    voice.emo.alias_names = {"chuckles", "excited"};
    voice.emo.alias_styles = {"laughing", "joyful"};
    voice.emo.default_style = "neutral";
    voice.desc.styles = voice.emo.styles;
    voice.desc.default_style = voice.emo.default_style;

    arch.voice_descs.push_back(voice.desc);
    arch.voice_weights.push_back(std::move(voice));
    return arch;
}

/// A phonemizer that is its own input, and a tokenizer counting characters:
/// the chunker's budgets are all that matter here, not the phonemes.
kokopop::PhonemizeFn echo_phonemizer() {
    return [](const std::string & text, std::string & phonemes, std::string &) {
        phonemes = text;
        return true;
    };
}

kokopop::TokenizeFn counting_tokenizer() {
    return [](const std::string & phonemes, std::vector<uint32_t> & ids, std::string &) {
        ids.assign(std::max<size_t>(1, phonemes.size() / 4), 1);
        return true;
    };
}

} // namespace

TEST_CASE("sanotts_style_tags_select_a_style_and_leave_the_text_to_the_phonemizer") {
    kokopop::SanoArch arch = make_styled_arch();
    kokopop::ModelArch * base = &arch;
    const kokopop::VoiceDesc desc = arch.voice_descs.front();
    kokopop::StyleTagFn style_tag = [base, desc](std::string_view tag, std::string & style) {
        return base->resolve_style_tag(desc, tag, style);
    };

    std::string error;
    auto chunks = kokopop::chunk_text(
        "Bonjour tout le monde. [sad] Il est parti hier soir. "
        "[chuckles] Enfin, presque.",
        kokopop::make_long_form_config(), echo_phonemizer(), counting_tokenizer(),
        style_tag, error);

    REQUIRE_EQ(chunks.size(), 3);
    // The first segment carries no tag: its style is the request's, not one
    // this text named.
    CHECK_EQ(chunks[0].style, std::string(""));
    CHECK_EQ(chunks[1].style, std::string("sad"));
    // An ElevenLabs audio tag resolves through the pack's alias table.
    CHECK_EQ(chunks[2].style, std::string("laughing"));

    for (const auto & chunk : chunks) {
        CHECK(chunk.text.find('[') == std::string::npos);
        CHECK(chunk.phonemes.find('[') == std::string::npos);
    }
}

TEST_CASE("sanotts_unknown_bracket_tags_stay_in_the_text") {
    kokopop::SanoArch arch = make_styled_arch();
    kokopop::ModelArch * base = &arch;
    const kokopop::VoiceDesc desc = arch.voice_descs.front();
    kokopop::StyleTagFn style_tag = [base, desc](std::string_view tag, std::string & style) {
        return base->resolve_style_tag(desc, tag, style);
    };

    std::string error;
    auto chunks = kokopop::chunk_text(
        "Voir la note [12] du rapport [tres long commentaire entre crochets].",
        kokopop::make_long_form_config(), echo_phonemizer(), counting_tokenizer(),
        style_tag, error);

    REQUIRE(!chunks.empty());
    std::string text;
    for (const auto & chunk : chunks) {
        text += chunk.text;
        CHECK_EQ(chunk.style, std::string(""));
    }
    CHECK(text.find("[12]") != std::string::npos);
    CHECK(text.find("[tres long commentaire entre crochets]") != std::string::npos);
}

TEST_CASE("sanotts_a_chunk_never_mixes_two_styles") {
    kokopop::SanoArch arch = make_styled_arch();
    kokopop::ModelArch * base = &arch;
    const kokopop::VoiceDesc desc = arch.voice_descs.front();
    kokopop::StyleTagFn style_tag = [base, desc](std::string_view tag, std::string & style) {
        return base->resolve_style_tag(desc, tag, style);
    };

    // Two very short sentences the long-form budget would happily merge.
    std::string error;
    auto chunks = kokopop::chunk_text(
        "Oui. [angry] Non.",
        kokopop::make_long_form_config(), echo_phonemizer(), counting_tokenizer(),
        style_tag, error);
    REQUIRE_EQ(chunks.size(), 2);
    CHECK_EQ(chunks[0].style, std::string(""));
    CHECK_EQ(chunks[1].style, std::string("angry"));

    // Same through the adaptative path, which assembles units one chunk at a
    // time and would otherwise keep pulling across the tag.
    auto cfg = kokopop::make_adaptative_config();
    auto units = kokopop::prepare_chunk_units("Oui. [angry] Non.", cfg,
                                              echo_phonemizer(), counting_tokenizer(),
                                              style_tag, error);
    REQUIRE(units.size() >= 2);
    size_t next = 0;
    auto first = kokopop::build_adaptative_chunk(units, next, cfg, 200, true,
                                                 counting_tokenizer(), error);
    CHECK_EQ(first.style, std::string(""));
    CHECK(next < units.size());
    auto second = kokopop::build_adaptative_chunk(units, next, cfg, 200, false,
                                                  counting_tokenizer(), error);
    CHECK_EQ(second.style, std::string("angry"));
}

TEST_CASE("sanotts_empty_style_segments_produce_no_chunks") {
    kokopop::StyleTagFn tag = [](std::string_view name, std::string & style) {
        if (name != "sad" && name != "angry") return false;
        style = std::string(name);
        return true;
    };
    std::string error;
    for (const std::string text : {"[sad] Bonjour.",
                                   " \n [angry] \t [sad] Bonjour. [angry] \n "}) {
        auto cfg = kokopop::make_long_form_config();
        auto units = kokopop::prepare_chunk_units(text, cfg, echo_phonemizer(),
                                                   counting_tokenizer(), tag, error);
        REQUIRE_EQ(units.size(), 1);
        CHECK_EQ(units[0].text, "Bonjour.");
        CHECK_EQ(units[0].style, "sad");
        auto chunks = kokopop::chunk_text(text, cfg, echo_phonemizer(),
                                          counting_tokenizer(), tag, error);
        REQUIRE_EQ(chunks.size(), 1);
        CHECK_EQ(chunks[0].text, "Bonjour.");
        CHECK_EQ(chunks[0].style, "sad");
    }
    auto chunks = kokopop::chunk_text("[sad] \n [angry]", kokopop::make_long_form_config(),
                                      echo_phonemizer(), counting_tokenizer(), tag, error);
    CHECK(chunks.empty());
}

TEST_CASE("sanotts_adaptative_tiny_tail_preserves_every_style_boundary") {
    auto cfg = kokopop::make_adaptative_config();
    std::string error;
    for (const bool change_style : {false, true}) {
        std::vector<kokopop::Unit> units(3);
        for (auto & unit : units) {
            unit.text = unit.phonemes = "Bonjour.";
            unit.n_tokens = 2;
            unit.boundary_after = kokopop::Boundary::Sentence;
        }
        // Stop the main loop before the tail, then exercise its absorption.
        units[0].n_tokens = cfg.target_min_tokens;
        units[2].style = change_style ? "sad" : "";
        size_t next = 0;
        auto chunk = kokopop::build_adaptative_chunk(units, next, cfg,
            cfg.target_max_tokens, false, counting_tokenizer(), error);
        CHECK_EQ(next, change_style ? 1 : 3);
        CHECK_EQ(chunk.is_last, !change_style);
        CHECK(chunk.style.empty());
        if (change_style) {
            chunk = kokopop::build_adaptative_chunk(units, next, cfg,
                cfg.target_max_tokens, false, counting_tokenizer(), error);
            CHECK_EQ(next, 2);
            CHECK(chunk.style.empty());
            chunk = kokopop::build_adaptative_chunk(units, next, cfg,
                cfg.target_max_tokens, false, counting_tokenizer(), error);
            CHECK_EQ(next, 3);
            CHECK_EQ(chunk.style, "sad");
            CHECK(chunk.is_last);
        }
    }
}

TEST_CASE("sanotts_style_resolution_yields_a_vector_and_the_neutral_bypass") {
    kokopop::SanoArch arch = make_styled_arch();
    const kokopop::SanoVoice & voice = arch.voice_weights.front();

    std::string error;
    kokopop::SanoStyle style;

    // An empty request is the pack's own default, and that default is neutral:
    // the vector is present and zero, and the duration branch is bypassed.
    REQUIRE_MESSAGE(arch.resolve_style(voice, "", style, error), error);
    CHECK_EQ(style.name, std::string("neutral"));
    CHECK_EQ(style.vector, std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f});
    CHECK(style.neutral);

    // Any other style conditions both branches on the same vector.
    REQUIRE_MESSAGE(arch.resolve_style(voice, "sad", style, error), error);
    CHECK_EQ(style.vector, std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f});
    CHECK_FALSE(style.neutral);

    // Case folding, and the alias table.
    REQUIRE_MESSAGE(arch.resolve_style(voice, "Angry", style, error), error);
    CHECK_EQ(style.name, std::string("angry"));
    REQUIRE_MESSAGE(arch.resolve_style(voice, "excited", style, error), error);
    CHECK_EQ(style.name, std::string("joyful"));
}

TEST_CASE("sanotts_an_unrenderable_style_fails_the_request") {
    kokopop::SanoArch arch = make_styled_arch();
    const kokopop::SanoVoice & styled = arch.voice_weights.front();

    std::string error;
    kokopop::SanoStyle style;
    CHECK_FALSE(arch.resolve_style(styled, "sarcastic", style, error));
    CHECK(error.find("sarcastic") != std::string::npos);
    // The message says what the voice does have, so the caller can correct it.
    CHECK(error.find("joyful") != std::string::npos);

    // A voice with no emotion pack accepts no style at all: rendering it
    // neutral instead would be a silently different voice.
    kokopop::SanoVoice plain;
    plain.desc.name = "amy";
    error.clear();
    CHECK(arch.resolve_style(plain, "", style, error));
    CHECK(style.vector.empty());
    CHECK(style.neutral);
    CHECK_FALSE(arch.resolve_style(plain, "sad", style, error));
    CHECK(error.find("no emotion pack") != std::string::npos);
}
