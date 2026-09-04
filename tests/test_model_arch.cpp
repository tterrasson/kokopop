#include "test_helpers.h"

#include "arch/kokoro/kokoro_arch.h"
#include "model/arch.h"
#include "model/gguf_util.h"

#include <ggml.h>
#include <gguf.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

using kokopop::Arch;


TEST_CASE("gguf_typed_readers_reject_wrong_types_without_aborting") {
    auto * meta = gguf_init_empty();
    REQUIRE(meta != nullptr);
    gguf_set_val_str(meta, "scalar", "wrong type");
    gguf_set_val_u32(meta, "number", 7);
    const uint32_t data[] = {1, 2};
    gguf_set_arr_data(meta, "numbers", GGUF_TYPE_UINT32, data, 2);
    const char * strings[] = {"a", "b"};
    gguf_set_arr_str(meta, "strings", strings, 2);
    uint32_t u = 0;
    int32_t i = 0;
    bool b = false;
    float f = 0;
    std::string s;
    std::vector<uint32_t> us;
    std::vector<std::string> ss;
    CHECK_FALSE(kokopop::gguf_get_u32(meta, "scalar", u));
    CHECK_FALSE(kokopop::gguf_get_i32(meta, "scalar", i));
    CHECK_FALSE(kokopop::gguf_get_bool(meta, "scalar", b));
    CHECK_FALSE(kokopop::gguf_get_f32(meta, "scalar", f));
    CHECK_FALSE(kokopop::gguf_get_str(meta, "number", s));
    CHECK_FALSE(kokopop::gguf_get_str_array(meta, "scalar", ss));
    CHECK_FALSE(kokopop::gguf_get_str_array(meta, "numbers", ss));
    CHECK_FALSE(kokopop::gguf_get_u32_array(meta, "number", us));
    CHECK_FALSE(kokopop::gguf_get_u32_array(meta, "strings", us));
    CHECK(kokopop::gguf_get_u32_array(meta, "numbers", us));
    CHECK_EQ(us, std::vector<uint32_t>{1, 2});
    CHECK(kokopop::gguf_get_str_array(meta, "strings", ss));
    CHECK_EQ(ss, std::vector<std::string>{"a", "b"});
    gguf_free(meta);
}

// ---------------------------------------------------------------------------
// Architecture detection
// ---------------------------------------------------------------------------

namespace {

// Opens a GGUF for metadata only, the way load_model_from_gguf does before it
// knows which architecture it is dealing with.
struct MetaOnly {
    ggml_context * weights = nullptr;
    gguf_context * meta = nullptr;

    explicit MetaOnly(const std::string & path) {
        gguf_init_params params;
        params.no_alloc = true;
        params.ctx = &weights;
        meta = gguf_init_from_file(path.c_str(), params);
    }
    ~MetaOnly() {
        if (meta != nullptr) gguf_free(meta);
        if (weights != nullptr) ggml_free(weights);
    }
};

// A GGUF carrying only `general.alignment`: no architecture marker at all.
std::string write_archless_gguf() {
    const std::string path = "kokopop_archless_test.gguf";
    std::vector<uint8_t> bytes;
    bytes.insert(bytes.end(), {'G', 'G', 'U', 'F'});
    put_u32(bytes, 3);
    put_u64(bytes, 0);  // no tensors
    put_u64(bytes, 1);  // one kv
    put_kv_u32(bytes, "general.alignment", 32);
    align_to(bytes, 32);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return path;
}

} // namespace

TEST_CASE("arch_name_covers_every_enumerator") {
    CHECK_EQ(std::string(kokopop::arch_name(Arch::Kokoro)), "kokoro-82m");
    CHECK_EQ(std::string(kokopop::arch_name(Arch::SanoTTS)), "sanotts");
    CHECK_EQ(std::string(kokopop::arch_name(Arch::Unknown)), "unknown");
}

// The Kokoro GGUFs already distributed predate `kokopop.arch`; they must keep
// loading without a reconversion.
TEST_CASE("peek_arch_infers_kokoro_from_version_key_alone") {
    MetaOnly gguf(shared_mock_gguf());
    REQUIRE(gguf.meta != nullptr);
    CHECK_EQ(gguf_find_key(gguf.meta, "kokopop.arch"), -1);
    CHECK(kokopop::peek_arch(gguf.meta) == Arch::Kokoro);
}

TEST_CASE("peek_arch_returns_unknown_without_any_marker") {
    MetaOnly gguf(write_archless_gguf());
    REQUIRE(gguf.meta != nullptr);
    CHECK(kokopop::peek_arch(gguf.meta) == Arch::Unknown);
}

TEST_CASE("create_arch_rejects_a_file_with_no_marker") {
    MetaOnly gguf(write_archless_gguf());
    REQUIRE(gguf.meta != nullptr);
    std::string error;
    auto arch = kokopop::create_arch(gguf.meta, error);
    CHECK(arch == nullptr);
    CHECK(error.find("kokopop.arch") != std::string::npos);
}

TEST_CASE("create_arch_builds_kokoro_for_a_legacy_gguf") {
    MetaOnly gguf(shared_mock_gguf());
    REQUIRE(gguf.meta != nullptr);
    std::string error;
    auto arch = kokopop::create_arch(gguf.meta, error);
    REQUIRE(arch != nullptr);
    CHECK(error.empty());
    CHECK(arch->arch() == Arch::Kokoro);
    CHECK_EQ(std::string(arch->name()), "kokoro-82m");
}

// ---------------------------------------------------------------------------
// Loaded model: the neutral surface delegates to the architecture
// ---------------------------------------------------------------------------

TEST_CASE("loaded_mock_model_exposes_its_kokoro_arch") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    REQUIRE(kokopop::load_model_from_gguf(gguf, &options, model, error));

    REQUIRE(model->arch != nullptr);
    CHECK(model->arch->arch() == Arch::Kokoro);
    CHECK(kokopop::kokoro_arch(*model) != nullptr);

    // The voice list is ordered; the map is only an index over it.
    const auto & voices = model->arch->voices();
    REQUIRE_EQ(voices.size(), 1u);
    CHECK_EQ(voices.front().name, "af_heart");
    CHECK_EQ(model->voices.size(), voices.size());

    const kokopop::VoiceDesc * found = model->arch->find_voice("af_heart");
    REQUIRE(found != nullptr);
    CHECK_EQ(found->espeak_voice, "gmw/en-US");
    CHECK_EQ(found->normalization_lang, 'a');
    CHECK(found->frontend == kokopop::FrontendKind::Misaki);
    CHECK(found->decoder == kokopop::DecoderKind::Kokoro);

    CHECK(model->arch->find_voice("nope") == nullptr);
    CHECK(model->arch->default_voice() == found);
}

// The default voice used to be whichever entry an unordered_map happened to
// yield first, which could differ between runs and builds.
TEST_CASE("default_voice_follows_the_kokopop_voices_order") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::string error;

    std::string first;
    for (int run = 0; run < 3; ++run) {
        std::unique_ptr<kokopop::Model> model;
        REQUIRE(kokopop::load_model_from_gguf(gguf, &options, model, error));
        const std::string resolved = kokopop::resolve_voice_name("", *model);
        if (run == 0) {
            first = resolved;
            CHECK_EQ(first, "af_heart");
        } else {
            CHECK_EQ(resolved, first);
        }
    }
}

TEST_CASE("sample_rate_falls_back_to_the_default_voice") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    REQUIRE(kokopop::load_model_from_gguf(gguf, &options, model, error));

    CHECK_EQ(model->sample_rate(), 24000);
    CHECK_EQ(model->sample_rate("af_heart"), 24000);
    // Unknown voice: the default voice's rate, not zero.
    CHECK_EQ(model->sample_rate("does_not_exist"), 24000);
    CHECK_EQ(model->sample_rate(""), 24000);
}

TEST_CASE("tokenize_phonemes_goes_through_the_arch") {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    REQUIRE(kokopop::load_model_from_gguf(gguf, &options, model, error));

    std::vector<uint32_t> via_model;
    REQUIRE(model->tokenize_phonemes("abc", via_model, error));

    std::vector<uint32_t> via_arch;
    const kokopop::VoiceDesc * voice = model->arch->default_voice();
    REQUIRE(voice != nullptr);
    REQUIRE(model->arch->tokenize("abc", *voice, via_arch, error));

    CHECK_EQ(via_model, via_arch);
    // Kokoro frames with the 0 sentinel on both sides.
    REQUIRE_EQ(via_model.size(), 5u);
    CHECK_EQ(via_model.front(), 0u);
    CHECK_EQ(via_model.back(), 0u);
}

TEST_CASE("kokoro_adjust_chunk_config_is_the_identity") {
    kokopop::KokoroArch arch;
    kokopop::VoiceDesc voice;
    const kokopop::ChunkConfig in = kokopop::make_long_form_config();
    const kokopop::ChunkConfig out = arch.adjust_chunk_config(in, voice);
    CHECK_EQ(out.hard_max_tokens, in.hard_max_tokens);
    CHECK_EQ(out.soft_max_tokens, in.soft_max_tokens);
    CHECK_EQ(out.target_max_tokens, in.target_max_tokens);
    CHECK_EQ(out.target_min_tokens, in.target_min_tokens);
}

// ---------------------------------------------------------------------------
// Graph context sizing
//
// The three Kokoro formulas moved out of Backend. Their results must not have
// moved with them: these are the exact byte counts the pipeline reserved
// before the refactor, so a change here is a change in Kokoro's memory
// profile and should be a deliberate one.
// ---------------------------------------------------------------------------

TEST_CASE("kokoro_context_bytes_match_the_pre_refactor_formulas") {
    std::string error;
    auto cpu = kokopop::create_cpu_backend(1);
    REQUIRE(cpu != nullptr);

    constexpr size_t mib = 1024u * 1024u;

    CHECK_EQ(kokopop::kokoro_frontend_context_bytes(*cpu), 16u * mib);

    const int64_t frames = 1200;
    const int64_t tokens = 64;
    const size_t expected_generation =
        static_cast<size_t>(frames) * static_cast<size_t>(tokens) * sizeof(float)
        + 512u * static_cast<size_t>(tokens) * sizeof(float)
        + static_cast<size_t>(frames) * 4096u * sizeof(float)
        + 64u * mib;
    CHECK_EQ(kokopop::kokoro_generation_context_bytes(*cpu, frames, tokens),
             expected_generation);

    const int64_t decoder_len = 640;
    const size_t out_frames = static_cast<size_t>(decoder_len) * 60u;
    const size_t expected_generator =
        64u * static_cast<size_t>(decoder_len) * sizeof(float)
        + 22u * (out_frames + 1) * sizeof(float)
        + out_frames * sizeof(float)
        + 256u * out_frames * sizeof(float) * 3u
        + 16u * mib;
    CHECK_EQ(kokopop::kokoro_generator_context_bytes(*cpu, decoder_len),
             expected_generator);
}

// A degenerate request must still reserve something usable rather than 0.
TEST_CASE("kokoro_context_bytes_clamp_non_positive_sizes") {
    auto cpu = kokopop::create_cpu_backend(1);
    REQUIRE(cpu != nullptr);
    CHECK_EQ(kokopop::kokoro_generation_context_bytes(*cpu, 0, 0),
             kokopop::kokoro_generation_context_bytes(*cpu, 1, 1));
    CHECK_EQ(kokopop::kokoro_generator_context_bytes(*cpu, -5),
             kokopop::kokoro_generator_context_bytes(*cpu, 1));
}

// The generic hook must never return less than the metadata a graph of that
// size actually needs, margin or no margin.
TEST_CASE("graph_context_bytes_grows_past_the_backend_margin") {
    auto cpu = kokopop::create_cpu_backend(1);
    REQUIRE(cpu != nullptr);

    const size_t small = cpu->graph_context_bytes(16, 32);
    CHECK_EQ(small, 16u * 1024u * 1024u);  // floored at the margin

    const size_t huge_tensors = 1u << 20;
    const size_t huge = cpu->graph_context_bytes(huge_tensors, 1u << 20);
    CHECK(huge > small);
    CHECK(huge >= huge_tensors * ggml_tensor_overhead());
}

TEST_CASE("create_backend_auto_keeps_cpu_for_sanotts") {
    std::string error;
    auto backend = kokopop::create_backend(KOKOPOP_BACKEND_AUTO, 1,
                                           Arch::SanoTTS, error);
    REQUIRE(backend != nullptr);
    CHECK_EQ(backend->type(), KOKOPOP_BACKEND_CPU);
}
