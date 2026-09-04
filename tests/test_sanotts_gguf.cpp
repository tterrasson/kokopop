#include "test_helpers.h"
#include "sanotts_fixtures.h"

#include "arch/sanotts/sano_arch.h"
#include "model/model.h"
#include "synthesis/synthesis_session.h"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

// GGUF loading: what a well-formed sanoTTS file must yield, and what a
// malformed one must be rejected for.
//
// The rejection cases are built by rewriting one key of a real file rather
// than by synthesising a whole GGUF: the point is that the loader notices the
// *disagreement* between metadata and tensors, which a hand-built file with no
// tensors could not exercise.

namespace {

/// Rewrites the little-endian u32 value of `key` in a copy of `source`.
///
/// Only patches values, never sizes, so the file stays structurally valid and
/// the loader fails on semantics rather than on a truncated read.
bool patch_u32_key(const std::string & source, const std::string & destination,
                   const std::string & key, uint32_t value) {
    std::ifstream in(source, std::ios::binary);
    if (!in) {
        return false;
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());

    // The key appears as a length-prefixed string followed by the type tag 4
    // (UINT32) and the value.
    std::vector<char> needle(8, 0);
    const uint64_t length = key.size();
    std::memcpy(needle.data(), &length, sizeof length);
    needle.insert(needle.end(), key.begin(), key.end());

    const auto it = std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end());
    if (it == bytes.end()) {
        return false;
    }
    const size_t offset = static_cast<size_t>(it - bytes.begin()) + needle.size();
    uint32_t type = 0;
    std::memcpy(&type, bytes.data() + offset, sizeof type);
    if (type != 4) {
        return false;
    }
    std::memcpy(bytes.data() + offset + 4, &value, sizeof value);

    std::ofstream out(destination, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return out.good();
}

bool load(const std::string & path, std::unique_ptr<kokopop::Model> & model,
          std::string & error) {
    kokopop_model_options options{};
    options.n_threads = 1;
    options.backend = KOKOPOP_BACKEND_CPU;
    return kokopop::load_model_from_gguf(path, &options, model, error);
}

} // namespace

TEST_CASE("sanotts_gguf_rejects_malformed_metadata_types_and_special_ids") {
    const auto path = kokopop::test::sanotts_model_path("heart");
    if (path.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_model_hint("heart"));
        return;
    }
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    REQUIRE_MESSAGE(load(path, model, error), error);
    auto * arch = kokopop::sano_arch(*model);
    auto * meta = model->gguf_ctx;
    SUBCASE("wrong scalar type") {
        gguf_set_val_str(meta, "kokopop.sanotts.version", "1");
    }
    SUBCASE("wrong array element type") {
        const char * strings[] = {"a"};
        gguf_set_arr_str(meta, "kokopop.sanotts.nfd_codepoints", strings, 1);
    }
    SUBCASE("missing NFD array") {
        gguf_remove_key(meta, "kokopop.sanotts.nfd_values");
    }
    SUBCASE("special id outside embeddings") {
        gguf_set_val_u32(meta, "kokopop.sanotts.voice.0.bos_id", 999);
    }
    SUBCASE("wrong operator selector type") {
        gguf_set_val_str(meta, "kokopop.sanotts.voice.0.dec.norm_type", "0");
    }
    SUBCASE("F16 bias unsupported by F32 addition") {
        auto * tensor = arch->voice_weights[0].dur.input_proj_b;
        const auto type = tensor->type;
        tensor->type = GGML_TYPE_F16;
        CHECK_FALSE(arch->load(*model, error));
        CHECK(error.find("F32") != std::string::npos);
        tensor->type = type;
        return;
    }
    SUBCASE("unsupported quantized tensor") {
        auto * tensor = arch->voice_weights[0].dur.embedding;
        const auto type = tensor->type;
        tensor->type = GGML_TYPE_Q8_0;
        CHECK_FALSE(arch->load(*model, error));
        CHECK(error.find("F32 or F16") != std::string::npos);
        tensor->type = type;
        return;
    }
    CHECK_FALSE(arch->load(*model, error));
    CHECK_FALSE(error.empty());
}

TEST_CASE("sanotts_duration_override_cannot_bypass_token_budget") {
    kokopop::test::SanoModel loaded;
    std::string error;
    if (!loaded.load("heart", error)) {
        MESSAGE("skipped: " << error);
        return;
    }
    const auto & voice = *loaded.arch->default_voice();
    const std::vector<uint32_t> ids(static_cast<size_t>(voice.max_tokens) + 1, 1);
    kokopop::SynthesisExtras extras;
    extras.dur_override.assign(ids.size(), 1);
    kokopop::SanoProbe probe;
    CHECK_FALSE(loaded.arch->run(ids, voice, 1.0f, extras, probe, error));
    CHECK(error.find("limit") != std::string::npos);
    CHECK(probe.audio.empty());
}

TEST_CASE("sanotts_gguf_cannot_silently_disable_a_learned_post_filter") {
    kokopop::test::SanoModel loaded;
    std::string error;
    if (!loaded.load("mixed", error)) {
        MESSAGE("skipped: " << error);
        return;
    }
    REQUIRE_EQ(loaded.arch->voices()[1].name, "kristin");
    gguf_set_val_u32(loaded.model->gguf_ctx,
                    "kokopop.sanotts.voice.1.dec.post_filter_channels", 0);
    CHECK_FALSE(loaded.arch->load(*loaded.model, error));
    CHECK(error.find("post-filter") != std::string::npos);
}

TEST_CASE("sanotts_gguf_loads_and_exposes_its_voices") {
    const std::string path = kokopop::test::sanotts_model_path("mixed");
    if (path.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_model_hint("mixed"));
        return;
    }

    std::unique_ptr<kokopop::Model> model;
    std::string error;
    REQUIRE_MESSAGE(load(path, model, error), error);

    kokopop::SanoArch * arch = kokopop::sano_arch(*model);
    REQUIRE(arch != nullptr);
    CHECK_EQ(std::string(arch->name()), "sanotts");
    CHECK(arch->arch() == kokopop::Arch::SanoTTS);
    CHECK_EQ(arch->voices().size(), 4u);

    // File order, which multi-voice enumeration depends on.
    CHECK_EQ(arch->voices()[0].name, "amy");
    CHECK_EQ(arch->voices()[3].name, "heartnano");

    // A GGUF may mix sample rates; the rate belongs to the voice, and the
    // model-level accessor is documented as "the default voice's".
    CHECK_EQ(arch->find_voice("amy")->sample_rate, 22050);
    CHECK_EQ(arch->find_voice("heart")->sample_rate, 24000);
    CHECK_EQ(model->sample_rate("amy"), 22050);
    CHECK_EQ(model->sample_rate("heart"), 24000);
    CHECK_EQ(model->sample_rate(), model->sample_rate(arch->default_voice()->name));

    // Aliases resolve to the canonical voice, not a copy of it.
    CHECK_EQ(arch->find_voice("heart-nano")->name, "heartnano");
    CHECK(arch->find_voice("nope") == nullptr);

    // Frontend and decoder are per voice, and they do not have to agree
    // across a file.
    CHECK(arch->find_voice("amy")->frontend == kokopop::FrontendKind::Piper);
    CHECK(arch->find_voice("amy")->decoder == kokopop::DecoderKind::PiperLite);
    CHECK(arch->find_voice("heart")->frontend == kokopop::FrontendKind::Misaki);
    CHECK(arch->find_voice("heart")->decoder == kokopop::DecoderKind::Vocos);
}

TEST_CASE("sanotts_gguf_carries_the_decomposition_table_the_piper_tokenizer_needs") {
    const std::string path = kokopop::test::sanotts_model_path("mixed");
    if (path.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_model_hint("mixed"));
        return;
    }
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    REQUIRE_MESSAGE(load(path, model, error), error);

    kokopop::SanoArch * arch = kokopop::sano_arch(*model);
    REQUIRE(arch != nullptr);
    CHECK(arch->nfd.present());
    CHECK_EQ(arch->nfd.n_offsets, arch->nfd.count + 1);
    CHECK(arch->nfd.validate(error));

    // A precomposed character must decompose; a plain one must not.
    std::vector<uint32_t> out;
    arch->nfd.decompose(0x00C0, out);   // LATIN CAPITAL LETTER A WITH GRAVE
    CHECK_EQ(out, std::vector<uint32_t>{0x0041, 0x0300});
    out.clear();
    arch->nfd.decompose(0x0041, out);
    CHECK_EQ(out, std::vector<uint32_t>{0x0041});
}

TEST_CASE("sanotts_gguf_chunk_budget_follows_the_voice") {
    const std::string path = kokopop::test::sanotts_model_path("mixed");
    if (path.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_model_hint("mixed"));
        return;
    }
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    REQUIRE_MESSAGE(load(path, model, error), error);

    kokopop::SanoArch * arch = kokopop::sano_arch(*model);
    REQUIRE(arch != nullptr);

    // The Kokoro presets budget 510 tokens; no sanoTTS voice was trained that
    // far, and the duration model's length hint is a function of its own
    // ceiling, so the preset has to come down rather than be exceeded.
    const kokopop::ChunkConfig preset = kokopop::make_long_form_config();
    const kokopop::ChunkConfig heart =
        arch->adjust_chunk_config(preset, *arch->find_voice("heart"));
    CHECK(heart.hard_max_tokens <= arch->find_voice("heart")->max_tokens);
    CHECK(heart.soft_max_tokens <= heart.hard_max_tokens);
    CHECK(heart.target_max_tokens <= heart.soft_max_tokens);
    CHECK(heart.target_min_tokens <= heart.target_max_tokens);
    CHECK(heart.first_chunk_target_max_tokens <= heart.target_max_tokens);

    // A voice with a larger ceiling keeps more of the preset.
    const kokopop::ChunkConfig amy =
        arch->adjust_chunk_config(preset, *arch->find_voice("amy"));
    CHECK(amy.hard_max_tokens > heart.hard_max_tokens);
}

TEST_CASE("sanotts_gguf_rejects_an_unsupported_file_version") {
    const std::string source = kokopop::test::sanotts_model_path("heart");
    if (source.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_model_hint("heart"));
        return;
    }
    const std::string path = "kokopop_sanotts_bad_version.gguf";
    if (!patch_u32_key(source, path, "kokopop.sanotts.version", 99)) {
        MESSAGE("skipped: could not patch the version key");
        return;
    }

    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK_FALSE(load(path, model, error));
    CHECK(error.find("version") != std::string::npos);
    std::remove(path.c_str());
}

// DyT normalisation and ReLU are a real upstream arm that no shipped voice
// uses. Running such a file as LayerNorm + GELU would be a silent
// substitution, so the loader must refuse it.
TEST_CASE("sanotts_gguf_refuses_the_unimplemented_operator_arm") {
    const std::string source = kokopop::test::sanotts_model_path("heart");
    if (source.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_model_hint("heart"));
        return;
    }
    const std::string path = "kokopop_sanotts_bad_norm.gguf";
    if (!patch_u32_key(source, path, "kokopop.sanotts.voice.0.dec.norm_type", 1)) {
        MESSAGE("skipped: could not patch the norm_type key");
        return;
    }

    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK_FALSE(load(path, model, error));
    CHECK(error.find("norm_type") != std::string::npos);
    std::remove(path.c_str());
}

TEST_CASE("sanotts_gguf_rejects_metadata_that_contradicts_the_tensors") {
    const std::string source = kokopop::test::sanotts_model_path("heart");
    if (source.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_model_hint("heart"));
        return;
    }

    struct Case {
        const char * key;
        uint32_t value;
        const char * expected;
    };
    // Each of these leaves the tensors untouched, so the only way to catch it
    // is to check the declared dimension against the shape actually stored.
    const Case cases[] = {
        {"kokopop.sanotts.voice.0.dur.hidden", 65, "dur.embedding.weight"},
        {"kokopop.sanotts.voice.0.ac.out_channels", 99, "ac.output.weight"},
        {"kokopop.sanotts.voice.0.dec.dim", 191, "dec.embed.weight"},
        {"kokopop.sanotts.voice.0.dec.bins", 512, "n_fft/2 + 1"},
        {"kokopop.sanotts.voice.0.dur.kernel", 4, "must be odd"},
        {"kokopop.sanotts.voice.0.dec.blocks", 6, "dec.blocks.5"},
        {"kokopop.sanotts.voice.0.dur.hidden", 0, "outside the supported range"},
    };

    for (const Case & entry : cases) {
        const std::string path = "kokopop_sanotts_bad_meta.gguf";
        if (!patch_u32_key(source, path, entry.key, entry.value)) {
            MESSAGE("skipped: could not patch " << entry.key);
            continue;
        }
        std::unique_ptr<kokopop::Model> model;
        std::string error;
        INFO(std::string(entry.key) << " = " << entry.value);
        CHECK_FALSE(load(path, model, error));
        CHECK(error.find(entry.expected) != std::string::npos);
        std::remove(path.c_str());
    }
}

TEST_CASE("sanotts_gguf_refuses_a_chunk_too_short_to_render") {
    const std::string path = kokopop::test::sanotts_model_path("heart");
    if (path.empty()) {
        MESSAGE("skipped: " << kokopop::test::sanotts_model_hint("heart"));
        return;
    }
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    REQUIRE_MESSAGE(load(path, model, error), error);

    kokopop::SanoArch * arch = kokopop::sano_arch(*model);
    REQUIRE(arch != nullptr);
    const kokopop::VoiceDesc * voice = arch->find_voice("heart");
    REQUIRE(voice != nullptr);

    // One frame cannot be inverse-transformed: `(frames - 1) * hop` is zero
    // samples. The pipeline says so rather than returning an empty buffer.
    kokopop::SynthesisExtras extras;
    extras.dur_override = {1};
    kokopop::SanoProbe probe;
    CHECK_FALSE(arch->run({5u}, *voice, 1.0f, extras, probe, error));
    CHECK(error.find("two frames") != std::string::npos);

    // Three frames is legal for the transform but below the 16 values torch
    // draws its Gaussian noise in one go; the decoder refuses rather than
    // guessing at torch's scalar path.
    extras.dur_override = {3};
    error.clear();
    CHECK_FALSE(arch->run({5u}, *voice, 1.0f, extras, probe, error));
    CHECK(error.find("vocos decoder") != std::string::npos);

    // Four frames is the smallest chunk that renders.
    extras.dur_override = {4};
    error.clear();
    REQUIRE_MESSAGE(arch->run({5u}, *voice, 1.0f, extras, probe, error), error);
    CHECK_EQ(probe.audio.size(), 3u * 256u);
}

// The graph arenas hold ggml metadata, not activations, so their size follows
// the shape of the graph and not the length of the chunk. That is the whole
// reason one arena can serve all four graphs.
TEST_CASE("sanotts_graph_arena_does_not_grow_with_the_chunk") {
    kokopop::test::SanoModel loaded;
    std::string why;
    if (!loaded.load("mixed", why)) {
        MESSAGE("skipped: " << why);
        return;
    }

    for (const char * name : {"amy", "heart"}) {
        const kokopop::VoiceDesc * voice = loaded.arch->find_voice(name);
        REQUIRE(voice != nullptr);

        std::string error;
        std::string phonemes;
        std::vector<uint32_t> short_ids;
        std::vector<uint32_t> long_ids;
        REQUIRE(loaded.arch->phonemize("Hi.", *voice, phonemes, error));
        REQUIRE(loaded.arch->tokenize(phonemes, *voice, short_ids, error));
        REQUIRE(loaded.arch->phonemize(
            "The quick brown fox jumps over the lazy dog, twice.",
            *voice, phonemes, error));
        REQUIRE(loaded.arch->tokenize(phonemes, *voice, long_ids, error));
        REQUIRE(long_ids.size() > short_ids.size() * 3);

        kokopop::SynthesisExtras extras;
        kokopop::SanoProbe probe;
        REQUIRE_MESSAGE(loaded.arch->run(short_ids, *voice, 1.0f, extras, probe, error), error);
        const size_t after_short = loaded.arch->graph_scratch.high_water;

        REQUIRE_MESSAGE(loaded.arch->run(long_ids, *voice, 1.0f, extras, probe, error), error);
        INFO(std::string(name) << ": arena " << after_short << " -> "
             << loaded.arch->graph_scratch.high_water);
        CHECK_EQ(loaded.arch->graph_scratch.high_water, after_short);
        CHECK(after_short > 0);
    }
}

// A deeper voice needs a bigger graph, and the budget has to say so: ggml
// answers an undersized `ggml_new_graph_custom` with an abort, not an error.
TEST_CASE("sanotts_graph_budget_follows_the_declared_depths") {
    kokopop::test::SanoModel loaded;
    std::string why;
    if (!loaded.load("mixed", why)) {
        MESSAGE("skipped: " << why);
        return;
    }

    const kokopop::SanoVoice * amy =
        loaded.arch->voice_for(*loaded.arch->find_voice("amy"));
    const kokopop::SanoVoice * kristin =
        loaded.arch->voice_for(*loaded.arch->find_voice("kristin"));
    const kokopop::SanoVoice * heart =
        loaded.arch->voice_for(*loaded.arch->find_voice("heart"));
    const kokopop::SanoVoice * nano =
        loaded.arch->voice_for(*loaded.arch->find_voice("heartnano"));
    REQUIRE(amy != nullptr);
    REQUIRE(kristin != nullptr);
    REQUIRE(heart != nullptr);
    REQUIRE(nano != nullptr);

    // kristin has one more acoustic frame block than amy, and a post filter.
    REQUIRE(kristin->ac.depth > amy->ac.depth);
    CHECK(kokopop::sano_acoustic_frame_budget(*kristin).nodes >
          kokopop::sano_acoustic_frame_budget(*amy).nodes);
    CHECK(kokopop::sano_piperlite_budget(*kristin).nodes >
          kokopop::sano_piperlite_budget(*amy).nodes);

    // heart has five ConvNeXt blocks, heartnano four.
    REQUIRE(heart->vocos.blocks > nano->vocos.blocks);
    CHECK(kokopop::sano_vocos_budget(*heart).nodes >
          kokopop::sano_vocos_budget(*nano).nodes);

    // The measured node counts on these voices, with the headroom the
    // formulas are meant to leave.
    CHECK(kokopop::sano_duration_budget(*heart).nodes >= 96);
    CHECK(kokopop::sano_vocos_budget(*heart).nodes >= 200);
    CHECK(kokopop::sano_piperlite_budget(*kristin).nodes >= 300);
}

// A session's rate is its voice's, not the model's. `kokopop_model_sample_rate`
// keeps answering for the default voice — documented as such — but everything
// that has resolved a voice must use that voice's rate, or a 22050 Hz
// rendering is written into a 24000 Hz container and plays back too fast.
TEST_CASE("sanotts_session_sample_rate_follows_its_own_voice") {
    kokopop::test::SanoModel loaded;
    std::string why;
    if (!loaded.load("mixed", why)) {
        MESSAGE("skipped: " << why);
        return;
    }

    const int default_rate = loaded.model->sample_rate();
    const int amy_rate = loaded.model->sample_rate("amy");
    const int heart_rate = loaded.model->sample_rate("heart");
    REQUIRE(amy_rate != heart_rate);

    kokopop::SynthesisSessionOptions amy_options;
    amy_options.voice = "amy";
    kokopop::SynthesisSession amy(*loaded.model, amy_options);
    CHECK_EQ(amy.sample_rate(), amy_rate);

    kokopop::SynthesisSessionOptions heart_options;
    heart_options.voice = "heart";
    kokopop::SynthesisSession heart(*loaded.model, heart_options);
    CHECK_EQ(heart.sample_rate(), heart_rate);

    // An unnamed voice still resolves to something deterministic.
    kokopop::SynthesisSessionOptions unnamed;
    kokopop::SynthesisSession fallback(*loaded.model, unnamed);
    CHECK_EQ(fallback.sample_rate(), default_rate);

    // An alias is the same voice, so it must carry the same rate.
    CHECK_EQ(loaded.model->sample_rate("heart-nano"),
             loaded.model->sample_rate("heartnano"));
}
