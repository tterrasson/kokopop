#include "test_helpers.h"
#include "sanotts_fixtures.h"

#include "arch/sanotts/sano_arch.h"
#include "core/backend_names.h"
#include "model/model.h"

#include <memory>
#include <string>
#include <vector>

// Cross-backend parity.
//
// Durations must be *identical*, not close: they are computed on the CPU
// sub-backend on every backend precisely so that this holds, and a difference
// here is a design bug rather than a tolerance to widen. The waveform is held
// to the same gate as the golden fixtures.
//
// Every backend the build was configured with is tried; one that fails to
// initialise on this machine is skipped, not failed.

namespace {

bool load_on(int backend, std::unique_ptr<kokopop::Model> & model, std::string & error) {
    const std::string path = kokopop::test::sanotts_model_path("mixed");
    if (path.empty()) {
        error = kokopop::test::sanotts_model_hint("mixed");
        return false;
    }
    kokopop_model_options options{};
    options.n_threads = 1;
    options.backend = backend;
    return kokopop::load_model_from_gguf(path, &options, model, error);
}

/// One deterministic rendering: a pinned seed and pinned durations, so the
/// only variable between two runs is the backend.
bool render(kokopop::Model & model, const std::string & voice_name,
            kokopop::SanoProbe & probe, std::string & error) {
    kokopop::SanoArch * arch = kokopop::sano_arch(model);
    if (arch == nullptr) {
        error = "not a sanoTTS model";
        return false;
    }
    const kokopop::VoiceDesc * voice = arch->find_voice(voice_name);
    if (voice == nullptr) {
        error = "unknown voice " + voice_name;
        return false;
    }

    std::string phonemes;
    std::vector<uint32_t> ids;
    if (!arch->phonemize("The quick brown fox jumps over the lazy dog.", *voice,
                         phonemes, error) ||
        !arch->tokenize(phonemes, *voice, ids, error)) {
        return false;
    }

    kokopop::SynthesisExtras extras;
    extras.has_noise_seed = true;
    extras.noise_seed = 20260904ull;
    return arch->run(ids, *voice, 1.0f, extras, probe, error);
}

void check_backend_matches_cpu(int backend) {
    std::unique_ptr<kokopop::Model> cpu_model;
    std::string error;
    if (!load_on(KOKOPOP_BACKEND_CPU, cpu_model, error)) {
        MESSAGE("skipped: " << error);
        return;
    }

    std::unique_ptr<kokopop::Model> other_model;
    if (!load_on(backend, other_model, error)) {
        MESSAGE("skipped: " << std::string(kokopop::backend_name(backend)) << " unavailable: " << error);
        return;
    }
    if (other_model->backend_type == KOKOPOP_BACKEND_CPU) {
        MESSAGE("skipped: " << std::string(kokopop::backend_name(backend)) << " resolved to CPU");
        return;
    }

    for (const char * voice : {"amy", "kristin", "heart", "heartnano"}) {
        kokopop::SanoProbe reference;
        kokopop::SanoProbe candidate;
        REQUIRE_MESSAGE(render(*cpu_model, voice, reference, error), error);
        REQUIRE_MESSAGE(render(*other_model, voice, candidate, error), error);

        INFO(std::string(kokopop::backend_name(backend)) << " vs CPU, voice " << voice);
        CHECK_EQ(candidate.durations, reference.durations);
        CHECK_EQ(candidate.frames, reference.frames);
        CHECK_EQ(candidate.audio.size(), reference.audio.size());

        const auto comparison = kokopop::test::compare_audio(candidate.audio, reference.audio);
        INFO("correlation " << comparison.correlation
             << ", RMS error " << comparison.rms_error);
        CHECK(comparison.correlation >= 0.999);
        CHECK(comparison.rms_error <= 1e-3);
    }
}

} // namespace

// `AUTO` picks CPU for sanoTTS for now
TEST_CASE("sanotts_auto_backend_resolves_to_cpu") {
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    if (!load_on(KOKOPOP_BACKEND_AUTO, model, error)) {
        MESSAGE("skipped: " << error);
        return;
    }
    CHECK_EQ(model->backend_type, KOKOPOP_BACKEND_CPU);
}

TEST_CASE("sanotts_metal_matches_cpu") {
    check_backend_matches_cpu(KOKOPOP_BACKEND_METAL);
}

TEST_CASE("sanotts_cuda_matches_cpu") {
    check_backend_matches_cpu(KOKOPOP_BACKEND_CUDA);
}

TEST_CASE("sanotts_vulkan_matches_cpu") {
    check_backend_matches_cpu(KOKOPOP_BACKEND_VULKAN);
}

TEST_CASE("sanotts_opencl_matches_cpu") {
    check_backend_matches_cpu(KOKOPOP_BACKEND_OPENCL);
}

// Two synthesis sessions on one model share the arch's iSTFT workspace and
// graph arena, so a rendering must not depend on what ran before it.
TEST_CASE("sanotts_rendering_does_not_depend_on_what_ran_before") {
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    if (!load_on(KOKOPOP_BACKEND_CPU, model, error)) {
        MESSAGE("skipped: " << error);
        return;
    }

    kokopop::SanoProbe first;
    kokopop::SanoProbe interleaved;
    kokopop::SanoProbe again;
    REQUIRE_MESSAGE(render(*model, "heart", first, error), error);
    REQUIRE_MESSAGE(render(*model, "amy", interleaved, error), error);
    REQUIRE_MESSAGE(render(*model, "heart", again, error), error);

    CHECK_EQ(again.durations, first.durations);
    CHECK_EQ(again.audio, first.audio);
}
