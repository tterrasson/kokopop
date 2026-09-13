// Emotion styles: the text tags that select them, and their resolution.
//
// Neither needs a trained emotion pack. The tag splitter is driven by the same
// closure the architecture binds, and the resolver reads a `SanoEmotionWeights`
// a test can fill in — which is the point: what these gate is the contract
// between a name in the text and a vector, not the acoustics it produces.

#include "arch/sanotts/sano_arch.h"
#include "synthesis/chunker/chunker.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

/// The style vocabulary of the shipped French pack, with a style encoder small
/// enough to predict by hand.
///
/// The encoder is the identity on the first `style_code` components — no bias
/// anywhere, which is what makes a zero vector encode to zero — so a resolved
/// code is `silu` of the direction and nothing else. That is enough to gate
/// the split the two heads consume without training anything.
kokopop::SanoArch make_styled_arch(float axis_gain = 1.0f) {
    kokopop::SanoArch arch;
    kokopop::SanoVoice voice;
    voice.desc.name = "fr";
    voice.desc.sample_rate = 22050;
    voice.emo.dim = 4;
    voice.emo.cond_hidden = 4;
    voice.emo.style_code = 4;
    voice.emo.axis_gain.assign(4, axis_gain);
    voice.emo.style_w0.assign(4 * 4, 0.0f);
    voice.emo.style_w2.assign(4 * 4, 0.0f);
    for (size_t i = 0; i < 4; ++i) {
        voice.emo.style_w0[i * 4 + i] = 1.0f;
        voice.emo.style_w2[i * 4 + i] = 1.0f;
    }
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
    // the vector is present and zero, both heads are bypassed, and nothing is
    // resolved that either of them could read.
    REQUIRE_MESSAGE(arch.resolve_style(voice, "", style, error), error);
    CHECK_EQ(style.name, std::string("neutral"));
    CHECK_EQ(style.vector, std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f});
    CHECK(style.neutral);
    CHECK(style.code.empty());
    CHECK_EQ(style.amount, 0.0f);

    // Any other style drives both heads from the same split: a direction
    // through the encoder, and one scalar intensity multiplying its output.
    REQUIRE_MESSAGE(arch.resolve_style(voice, "sad", style, error), error);
    CHECK_EQ(style.vector, std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f});
    CHECK_FALSE(style.neutral);
    REQUIRE_EQ(style.code.size(), 4);
    // A unit basis vector normalizes to itself, and `silu(1) = 1 / (1 + e^-1)`.
    CHECK(style.code[1] == doctest::Approx(1.0f / (1.0f + std::exp(-1.0f))));
    CHECK_EQ(style.code[0], 0.0f);
    CHECK_EQ(style.amount, doctest::Approx(1.0f));

    // Case folding, and the alias table.
    REQUIRE_MESSAGE(arch.resolve_style(voice, "Angry", style, error), error);
    CHECK_EQ(style.name, std::string("angry"));
    REQUIRE_MESSAGE(arch.resolve_style(voice, "excited", style, error), error);
    CHECK_EQ(style.name, std::string("joyful"));
}

TEST_CASE("sanotts_intensity_is_radial_and_separable_from_the_direction") {
    kokopop::SanoArch arch = make_styled_arch();
    kokopop::SanoVoice & voice = arch.voice_weights.front();
    std::string error;

    // Two vectors on one ray. Only the intensity differs, and it differs by
    // exactly the ratio of the vectors: what the heads displace by is linear
    // in it, which is the property the direction/amount split exists to give.
    kokopop::SanoStyle unit, half;
    voice.emo.vectors.assign({0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f});
    voice.emo.styles = {"neutral", "angry"};
    REQUIRE_MESSAGE(arch.resolve_style(voice, "angry", unit, error), error);
    voice.emo.vectors[4] = 0.5f;
    REQUIRE_MESSAGE(arch.resolve_style(voice, "angry", half, error), error);

    CHECK_EQ(half.amount, doctest::Approx(unit.amount * 0.5f));
    // The direction, and so the code the nonlinear encoder produces from it,
    // is untouched: half a vector buys half an effect, not a different one.
    REQUIRE_EQ(half.code.size(), unit.code.size());
    for (size_t i = 0; i < unit.code.size(); ++i) {
        CHECK_EQ(half.code[i], doctest::Approx(unit.code[i]));
    }

    // Calibration scales the intensity of an axis. It cannot rotate one style
    // into another, because it never reaches the direction.
    kokopop::SanoArch loud = make_styled_arch(2.0f);
    kokopop::SanoStyle scaled;
    REQUIRE_MESSAGE(loud.resolve_style(loud.voice_weights.front(), "angry", scaled, error),
                    error);
    CHECK_EQ(scaled.amount, doctest::Approx(unit.amount * 2.0f));
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

// ---------------------------------------------------------------------------
// The two heads
//
// Both are pure host functions over a `SanoVoice`, so they gate without a
// GGUF: what they need is a style code, a conditioning, and the frozen model's
// own output. The weights below are chosen so that the composition — which
// phone mean pairs with which channel, what the intensity multiplies, what the
// centering is weighted by — is readable, not so that the arithmetic is.
// ---------------------------------------------------------------------------

namespace {

/// A voice whose acoustic head has no contextual term: `context_bound` of zero
/// switches it off, leaving the phrase-global pair alone.
kokopop::SanoVoice make_film_voice() {
    kokopop::SanoVoice voice;
    voice.desc.name = "fr";
    voice.ac.out_channels = 2;
    voice.emo.dim = 1;
    voice.emo.cond_hidden = 2;
    voice.emo.style_code = 1;
    voice.emo.channels = 2;
    voice.emo.context_rank = 1;
    voice.emo.max_delta = 1.0f;
    voice.emo.max_log_gain = 0.3f;
    voice.emo.context_bound = 0.0f;
    voice.emo.donor_projection = 0.0f;
    voice.emo.donor_axis.assign(2, 0.0f);
    // `[style_code, 2 * channels]`: the log gain of both channels, then their
    // shifts.
    voice.emo.global_film = {0.2f, -0.2f, 0.5f, -0.5f};
    voice.emo.max_log_ratio = 0.6931471805599453f;
    voice.emo.local_bound = 0.2f;
    voice.emo.tempo_w = {0.5f};
    // `output(tokens * style_local(code))` collapses to reading the first
    // channel of the conditioning and nothing else.
    voice.emo.style_local = {1.0f, 0.0f};
    voice.emo.dur_output_w = {1.0f, 1.0f};
    voice.emo.observed_ids.assign(4, 1.0f);
    return voice;
}

kokopop::SanoStyle unit_style() {
    kokopop::SanoStyle style;
    style.name = "sad";
    style.vector = {1.0f};
    style.code = {1.0f};
    style.amount = 1.0f;
    style.neutral = false;
    return style;
}

} // namespace

TEST_CASE("sanotts_the_acoustic_head_films_each_phone_by_its_own_mean") {
    const kokopop::SanoVoice voice = make_film_voice();
    const kokopop::SanoStyle style = unit_style();

    // `[frames, channels]` with frames fastest: channel 0 is 1, 2, 3 and
    // channel 1 is 4, 5, 6. The first phone spans two frames, the second one.
    std::vector<float> latent = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<int32_t> durations = {2, 1};
    const std::vector<float> conditioned = {0.0f, 0.0f, 0.0f, 0.0f};

    std::string error;
    REQUIRE_MESSAGE(kokopop::sano_apply_emotion_latent(voice, durations, style,
                                                       conditioned, 3, latent, error),
                    error);

    auto delta = [&](float log_gain, float shift, float mean) {
        const float bounded = 0.3f * std::tanh(log_gain / 0.3f);
        return 1.0f * std::tanh(std::expm1(bounded) * mean + shift);
    };
    // At `frame_gain == 0` the pair sees nothing but the mean of the phone's
    // *own* frames, and the same correction reaches every frame of it: the
    // ablation of the frame-adaptive term, and what the retired v1 packs did
    // at every frame_gain.
    const float first0 = delta(0.2f, 0.5f, 1.5f);
    const float second0 = delta(0.2f, 0.5f, 3.0f);
    const float first1 = delta(-0.2f, -0.5f, 4.5f);
    const float second1 = delta(-0.2f, -0.5f, 6.0f);

    CHECK(latent[0] == doctest::Approx(1.0f + first0));
    CHECK(latent[1] == doctest::Approx(2.0f + first0));
    CHECK(latent[2] == doctest::Approx(3.0f + second0));
    CHECK(latent[3] == doctest::Approx(4.0f + first1));
    CHECK(latent[4] == doctest::Approx(5.0f + first1));
    CHECK(latent[5] == doctest::Approx(6.0f + second1));
}

namespace {

/// The raw, unbounded residual of one frame under `make_film_voice`.
float film_raw(float log_gain, float shift, float mean, float value, float frame_gain) {
    const float bounded = 0.3f * std::tanh(log_gain / 0.3f);
    return std::expm1(bounded) * (mean + frame_gain * (value - mean)) + shift;
}

} // namespace

TEST_CASE("sanotts_the_film_gain_reaches_the_variation_inside_a_phone") {
    kokopop::SanoVoice voice = make_film_voice();
    voice.emo.frame_gain = 1.0f;
    const kokopop::SanoStyle style = unit_style();

    // One phone, three frames, so nothing but the frame-adaptive term can make
    // its three corrections differ.
    std::vector<float> latent = {1.0f, 2.0f, 6.0f, 0.0f, 0.0f, 0.0f};
    std::string error;
    REQUIRE_MESSAGE(kokopop::sano_apply_emotion_latent(voice, {3}, style,
                                                       {0.0f, 0.0f}, 3, latent, error),
                    error);

    // At a gain of 1 the pair is evaluated on each frame itself, not on the
    // mean of 3 the phone would otherwise stand for.
    const std::vector<float> base = {1.0f, 2.0f, 6.0f};
    for (size_t f = 0; f < 3; ++f) {
        const float expected =
            std::tanh(film_raw(0.2f, 0.5f, 3.0f, base[f], 1.0f));
        CHECK(latent[f] == doctest::Approx(base[f] + expected));
    }
    CHECK(latent[0] != doctest::Approx(latent[1] - 1.0f));
}

TEST_CASE("sanotts_a_partial_film_gain_sits_between_the_mean_and_the_frame") {
    const kokopop::SanoStyle style = unit_style();
    const std::vector<float> base = {1.0f, 2.0f, 6.0f, 0.0f, 0.0f, 0.0f};

    auto render = [&](float frame_gain) {
        kokopop::SanoVoice voice = make_film_voice();
        voice.emo.frame_gain = frame_gain;
        std::vector<float> latent = base;
        std::string error;
        REQUIRE_MESSAGE(kokopop::sano_apply_emotion_latent(
                            voice, {3}, style, {0.0f, 0.0f}, 3, latent, error),
                        error);
        return latent;
    };

    const std::vector<float> quarter = render(0.25f);
    // The declared formula, not merely "something in between": a reader that
    // interpolated the *bounded* deltas instead would also land between them.
    for (size_t f = 0; f < 3; ++f) {
        const float expected =
            std::tanh(film_raw(0.2f, 0.5f, 3.0f, base[f], 0.25f));
        CHECK(quarter[f] == doctest::Approx(base[f] + expected));
    }
}

TEST_CASE("sanotts_the_crossfade_joins_the_boundary_and_leaves_the_interiors") {
    kokopop::SanoVoice voice = make_film_voice();
    voice.emo.frame_gain = 1.0f;
    voice.emo.transition_frames = 1;
    const kokopop::SanoStyle style = unit_style();

    // Two phones of four frames each, on channel 0 only.
    const std::vector<float> base = {1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 8.0f, 7.0f, 6.0f};
    std::vector<float> latent(base);
    latent.resize(16, 0.0f);
    std::vector<float> plain(latent);

    std::string error;
    REQUIRE_MESSAGE(kokopop::sano_apply_emotion_latent(voice, {4, 4}, style,
                                                       std::vector<float>(4, 0.0f),
                                                       8, latent, error),
                    error);
    kokopop::SanoVoice unjoined = voice;
    unjoined.emo.transition_frames = 0;
    REQUIRE_MESSAGE(kokopop::sano_apply_emotion_latent(unjoined, {4, 4}, style,
                                                       std::vector<float>(4, 0.0f),
                                                       8, plain, error),
                    error);

    std::vector<float> raw(8);
    for (size_t f = 0; f < 8; ++f) {
        const float mean = f < 4 ? 2.5f : 7.5f;
        raw[f] = film_raw(0.2f, 0.5f, mean, base[f], 1.0f);
    }

    // Only the two frames either side of the single boundary move, and the
    // window is half a raised cosine over `2 * width + 1` positions.
    for (size_t f : {0u, 1u, 2u, 5u, 6u, 7u}) {
        CHECK(latent[f] == doctest::Approx(plain[f]));
    }
    auto blend = [&](float position) {
        const float weight = 0.5f - 0.5f * std::cos(
            static_cast<float>(M_PI) * position / 3.0f);
        return std::tanh(raw[3] + weight * (raw[4] - raw[3]));
    };
    CHECK(latent[3] == doctest::Approx(base[3] + blend(1.0f)));
    CHECK(latent[4] == doctest::Approx(base[4] + blend(2.0f)));
    CHECK(latent[3] != doctest::Approx(plain[3]));
    CHECK(latent[4] != doctest::Approx(plain[4]));
}

TEST_CASE("sanotts_a_short_phone_is_never_smeared_across_its_neighbour") {
    kokopop::SanoVoice voice = make_film_voice();
    voice.emo.frame_gain = 1.0f;
    voice.emo.transition_frames = 2;
    const kokopop::SanoStyle style = unit_style();

    // The window is capped at half of *either* phone, and half of a one-frame
    // phone is zero: the boundary is left alone rather than blended with a
    // neighbour four times its length.
    std::vector<float> latent = {1.0f, 9.0f, 8.0f, 7.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> plain(latent);
    std::string error;
    REQUIRE_MESSAGE(kokopop::sano_apply_emotion_latent(voice, {1, 4}, style,
                                                       std::vector<float>(4, 0.0f),
                                                       5, latent, error),
                    error);
    kokopop::SanoVoice unjoined = voice;
    unjoined.emo.transition_frames = 0;
    REQUIRE_MESSAGE(kokopop::sano_apply_emotion_latent(unjoined, {1, 4}, style,
                                                       std::vector<float>(4, 0.0f),
                                                       5, plain, error),
                    error);
    for (size_t f = 0; f < 10; ++f) {
        CHECK(latent[f] == doctest::Approx(plain[f]));
    }
}

TEST_CASE("sanotts_a_silent_position_does_not_separate_the_phones_it_sits_between") {
    kokopop::SanoVoice voice = make_film_voice();
    voice.emo.frame_gain = 1.0f;
    voice.emo.transition_frames = 1;
    const kokopop::SanoStyle style = unit_style();

    // A zero-duration position occupies no frame, so the phones either side of
    // it are still neighbours and are still joined.
    std::vector<float> spaced = {1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 8.0f, 7.0f, 6.0f};
    spaced.resize(16, 0.0f);
    std::vector<float> adjacent(spaced);

    std::string error;
    REQUIRE_MESSAGE(kokopop::sano_apply_emotion_latent(voice, {4, 0, 4}, style,
                                                       std::vector<float>(6, 0.0f),
                                                       8, spaced, error),
                    error);
    REQUIRE_MESSAGE(kokopop::sano_apply_emotion_latent(voice, {4, 4}, style,
                                                       std::vector<float>(4, 0.0f),
                                                       8, adjacent, error),
                    error);
    for (size_t f = 0; f < 16; ++f) {
        CHECK(spaced[f] == doctest::Approx(adjacent[f]));
    }
}

TEST_CASE("sanotts_the_contextual_term_moves_one_phone_and_not_the_others") {
    kokopop::SanoVoice voice = make_film_voice();
    voice.emo.context_bound = 0.25f;
    voice.emo.global_film.assign(4, 0.0f);
    // Rank 1, reading the first channel of the conditioning and writing the
    // first channel's shift: the smallest factorization that is still a
    // style x context x channel product.
    voice.emo.style_rank = {1.0f};
    voice.emo.context_coeff_w = {1.0f, 0.0f};
    voice.emo.context_coeff_b = {0.0f};
    voice.emo.context_basis = {0.0f, 0.0f, 1.0f, 0.0f};

    const kokopop::SanoStyle style = unit_style();
    std::vector<float> latent(4, 0.0f);
    std::string error;
    // Two phones, one frame each; only the first is conditioned nonzero.
    REQUIRE_MESSAGE(kokopop::sano_apply_emotion_latent(
                        voice, {1, 1}, style, {0.4f, 0.0f, 0.0f, 0.0f}, 2, latent, error),
                    error);

    const float shift = 0.25f * std::tanh(0.4f / 0.25f);
    CHECK(latent[0] == doctest::Approx(std::tanh(shift)));
    CHECK(latent[1] == doctest::Approx(0.0f));
    // The second channel has no contextual coefficient and no global pair, so
    // it is left exactly where the frozen model put it.
    CHECK(latent[2] == doctest::Approx(0.0f));
    CHECK(latent[3] == doctest::Approx(0.0f));
}

TEST_CASE("sanotts_the_acoustic_head_is_linear_in_the_intensity_at_small_amounts") {
    // The bound is a tanh, so this is not an identity at any amount; near zero
    // it is, and that is where the radial promise has to hold.
    kokopop::SanoVoice voice = make_film_voice();
    voice.emo.global_film = {0.0f, 0.0f, 1.0f, 1.0f};

    std::string error;
    std::vector<float> full = {0.0f, 0.0f}, half = {0.0f, 0.0f};
    kokopop::SanoStyle style = unit_style();

    style.amount = 1e-3f;
    REQUIRE_MESSAGE(kokopop::sano_apply_emotion_latent(voice, {1}, style,
                                                       {0.0f, 0.0f}, 1, full, error),
                    error);
    style.amount = 5e-4f;
    REQUIRE_MESSAGE(kokopop::sano_apply_emotion_latent(voice, {1}, style,
                                                       {0.0f, 0.0f}, 1, half, error),
                    error);
    CHECK(half[0] == doctest::Approx(full[0] * 0.5f).epsilon(1e-4));
}

TEST_CASE("sanotts_the_duration_head_redistributes_around_the_frozen_attention") {
    const kokopop::SanoVoice voice = make_film_voice();
    const kokopop::SanoStyle style = unit_style();
    const std::vector<uint32_t> ids = {0, 1};
    // `[cond_hidden, n_tokens]`: the head reads the first channel of each.
    const std::vector<float> conditioned = {1.0f, 0.0f, -1.0f, 0.0f};

    std::string error;
    std::vector<float> residual;

    // Equal neutral predictions: the two tokens carry equal weight, the
    // redistribution is already mean-free, and only the tempo is left common.
    REQUIRE_MESSAGE(kokopop::sano_emotion_duration_residual(
                        voice, ids, style, conditioned, {0.0f, 0.0f}, residual, error),
                    error);
    const float local = std::tanh(1.0f) * 0.1f;
    auto bound = [](float x) { return 0.6931471805599453f * std::tanh(x / 0.6931471805599453f); };
    REQUIRE_EQ(residual.size(), 2);
    CHECK(residual[0] == doctest::Approx(bound(0.5f + local)));
    CHECK(residual[1] == doctest::Approx(bound(0.5f - local)));

    // A longer first token weighs three times as much in the centering, so the
    // pair no longer averages to zero over the tokens but over the *time*.
    REQUIRE_MESSAGE(kokopop::sano_emotion_duration_residual(
                        voice, ids, style, conditioned,
                        {std::log(3.0f), 0.0f}, residual, error),
                    error);
    const float mean = 0.75f * local + 0.25f * -local;
    CHECK(residual[0] == doctest::Approx(bound(0.5f + (local - mean))));
    CHECK(residual[1] == doctest::Approx(bound(0.5f + (-local - mean))));
}

TEST_CASE("sanotts_an_unsupervised_phoneme_keeps_the_tempo_and_loses_the_rest") {
    kokopop::SanoVoice voice = make_film_voice();
    // The emotional corpus never carried this id, so nothing measured how its
    // duration should move relative to its neighbours.
    voice.emo.observed_ids[1] = 0.0f;
    const kokopop::SanoStyle style = unit_style();

    std::string error;
    std::vector<float> residual;
    REQUIRE_MESSAGE(kokopop::sano_emotion_duration_residual(
                        voice, {0, 1}, style, {1.0f, 0.0f, -1.0f, 0.0f},
                        {0.0f, 0.0f}, residual, error),
                    error);
    auto bound = [](float x) { return 0.6931471805599453f * std::tanh(x / 0.6931471805599453f); };
    // Only the first token takes part in the centering, so it is centered
    // against itself and left with the tempo alone — as is the second.
    CHECK(residual[0] == doctest::Approx(bound(0.5f)));
    CHECK(residual[1] == doctest::Approx(bound(0.5f)));
}

TEST_CASE("sanotts_the_heads_refuse_a_conditioning_that_does_not_fit") {
    const kokopop::SanoVoice voice = make_film_voice();
    const kokopop::SanoStyle style = unit_style();

    std::string error;
    std::vector<float> residual;
    CHECK_FALSE(kokopop::sano_emotion_duration_residual(
        voice, {0, 1}, style, {1.0f, 0.0f}, {0.0f, 0.0f}, residual, error));
    CHECK(error.find("does not match") != std::string::npos);

    // Durations that do not add up to the frames the latent carries would
    // otherwise walk off the end of it.
    std::vector<float> latent = {0.0f, 0.0f, 0.0f, 0.0f};
    error.clear();
    CHECK_FALSE(kokopop::sano_apply_emotion_latent(voice, {1, 5}, style,
                                                   {0.0f, 0.0f, 0.0f, 0.0f}, 2,
                                                   latent, error));
    CHECK(error.find("do not fill") != std::string::npos);
}
