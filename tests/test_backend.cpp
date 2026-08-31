#include "test_helpers.h"
#include "backend/backend.h"
#include "core/backend_names.h"

// ---- Backend factory + options de chargement ----

TEST_CASE("backend_options") {
    const std::string & gguf = shared_mock_gguf();
    std::string error;
    std::unique_ptr<kokopop::Model> model;

    kokopop_model_options default_options{};
    CHECK(kokopop::load_model_from_gguf(gguf, &default_options, model, error));
    CHECK(model != nullptr);
    CHECK(model->backend != nullptr);
    // backend_type is one of the public backend enum values.
    int valid = (model->backend_type == KOKOPOP_BACKEND_CPU)
              + (model->backend_type == KOKOPOP_BACKEND_METAL)
              + (model->backend_type == KOKOPOP_BACKEND_CUDA)
              + (model->backend_type == KOKOPOP_BACKEND_VULKAN)
              + (model->backend_type == KOKOPOP_BACKEND_OPENCL);
    CHECK_EQ(valid, 1);
}

TEST_CASE("backend_auto_selects_available") {
    const std::string & gguf = shared_mock_gguf();
    std::string error;
    std::unique_ptr<kokopop::Model> model;

    kokopop_model_options opts{};
    opts.backend = KOKOPOP_BACKEND_AUTO;
    CHECK(kokopop::load_model_from_gguf(gguf, &opts, model, error));
    CHECK(model != nullptr);
    CHECK(model->backend != nullptr);
    // backend_type stores the *actual* backend used, not what was requested.
    // AUTO is resolved at runtime, so backend_type should never be AUTO after load.
    CHECK_NE(model->backend_type, KOKOPOP_BACKEND_AUTO);
    // backend_type should be a concrete backend enum value.
    CHECK_GE(model->backend_type, KOKOPOP_BACKEND_CPU);
    CHECK_LE(model->backend_type, KOKOPOP_BACKEND_OPENCL);
    // The actual backend is non-null.
    CHECK(model->backend != nullptr);
}

TEST_CASE("backend_cpu_always_works") {
    const std::string & gguf = shared_mock_gguf();
    std::string error;
    std::unique_ptr<kokopop::Model> model;

    kokopop_model_options opts{};
    opts.backend = KOKOPOP_BACKEND_CPU;
    CHECK(kokopop::load_model_from_gguf(gguf, &opts, model, error));
    CHECK(model != nullptr);
    CHECK_EQ(model->backend_type, KOKOPOP_BACKEND_CPU);
    CHECK(model->backend != nullptr);
}

TEST_CASE("backend_invalid_option") {
    const std::string & gguf = shared_mock_gguf();
    std::string error;
    std::unique_ptr<kokopop::Model> model;

    kokopop_model_options opts{};
    opts.backend = 99; // invalid
    CHECK(!kokopop::load_model_from_gguf(gguf, &opts, model, error));
    CHECK(!error.empty());
}

TEST_CASE("backend_vulkan_request_is_explicit") {
    const std::string & gguf = shared_mock_gguf();
    std::string error;
    std::unique_ptr<kokopop::Model> model;

    kokopop_model_options opts{};
    opts.backend = KOKOPOP_BACKEND_VULKAN;

    const bool ok = kokopop::load_model_from_gguf(gguf, &opts, model, error);
#ifdef KOKOPOP_HAS_VULKAN
    if (ok) {
        REQUIRE(model != nullptr);
        CHECK_EQ(model->backend_type, KOKOPOP_BACKEND_VULKAN);
    } else {
        CHECK(model == nullptr);
        CHECK(!error.empty());
    }
#else
    CHECK(!ok);
    CHECK(model == nullptr);
    CHECK(!error.empty());
#endif
}

TEST_CASE("backend_zero_threads") {
    const std::string & gguf = shared_mock_gguf();
    std::string error;
    std::unique_ptr<kokopop::Model> model;

    kokopop_model_options opts{};
    opts.n_threads = 0;
    opts.backend = KOKOPOP_BACKEND_CPU;
    CHECK(kokopop::load_model_from_gguf(gguf, &opts, model, error));
    CHECK(model != nullptr);
    CHECK(model->n_threads > 0);
}

TEST_CASE("backend_negative_threads") {
    const std::string & gguf = shared_mock_gguf();
    std::string error;
    std::unique_ptr<kokopop::Model> model;

    kokopop_model_options opts{};
    opts.n_threads = -1;
    opts.backend = KOKOPOP_BACKEND_CPU;
    CHECK(kokopop::load_model_from_gguf(gguf, &opts, model, error));
    CHECK(model != nullptr);
    CHECK(model->n_threads > 0);
}

TEST_CASE("backend_load_mock_cpu") {
    const std::string & gguf = shared_mock_gguf();
    std::string error;
    std::unique_ptr<kokopop::Model> model;

    kokopop_model_options opts{};
    opts.backend = KOKOPOP_BACKEND_CPU;
    opts.n_threads = 2;
    CHECK(kokopop::load_model_from_gguf(gguf, &opts, model, error));
    CHECK(model != nullptr);
    CHECK(model->is_mock);
    CHECK_EQ(model->backend_type, KOKOPOP_BACKEND_CPU);
}

TEST_CASE("backend_load_mock_auto") {
    const std::string & gguf = shared_mock_gguf();
    std::string error;
    std::unique_ptr<kokopop::Model> model;

    kokopop_model_options opts{};
    opts.backend = KOKOPOP_BACKEND_AUTO;
    CHECK(kokopop::load_model_from_gguf(gguf, &opts, model, error));
    CHECK(model != nullptr);
    CHECK(model->is_mock);
}

TEST_CASE("backend_opencl_request_is_explicit") {
    const std::string & gguf = shared_mock_gguf();
    std::string error;
    std::unique_ptr<kokopop::Model> model;

    kokopop_model_options opts{};
    opts.backend = KOKOPOP_BACKEND_OPENCL;

    const bool ok = kokopop::load_model_from_gguf(gguf, &opts, model, error);
#ifdef KOKOPOP_HAS_OPENCL
    if (ok) {
        REQUIRE(model != nullptr);
        CHECK_EQ(model->backend_type, KOKOPOP_BACKEND_OPENCL);
    } else {
        CHECK(model == nullptr);
        CHECK(!error.empty());
    }
#else
    CHECK(!ok);
    CHECK(model == nullptr);
    CHECK(!error.empty());
#endif
}

// Vulkan bring-up on Adreno crashed with a SIGSEGV inside vkCreateFence on the
// *second* init attempt after the first was refused: ggml had left its instance
// singleton half-built. Asking twice in a row must stay a clean refusal (or a
// clean load), never a crash.
TEST_CASE("backend_opencl_repeated_request_does_not_crash") {
    const std::string & gguf = shared_mock_gguf();

    for (int attempt = 0; attempt < 2; ++attempt) {
        std::string error;
        std::unique_ptr<kokopop::Model> model;

        kokopop_model_options opts{};
        opts.backend = KOKOPOP_BACKEND_OPENCL;

        if (kokopop::load_model_from_gguf(gguf, &opts, model, error)) {
            REQUIRE(model != nullptr);
            CHECK_EQ(model->backend_type, KOKOPOP_BACKEND_OPENCL);
        } else {
            CHECK(model == nullptr);
            CHECK(!error.empty());
        }
    }
}

// ---- Shared --backend name mapping (src/core/backend_names.h) ----

TEST_CASE("backend_names_round_trip") {
    const int32_t values[] = {
        KOKOPOP_BACKEND_CPU,
        KOKOPOP_BACKEND_METAL,
        KOKOPOP_BACKEND_CUDA,
        KOKOPOP_BACKEND_VULKAN,
        KOKOPOP_BACKEND_OPENCL,
    };

    for (int32_t value : values) {
        int32_t parsed = KOKOPOP_BACKEND_AUTO;
        REQUIRE(kokopop::backend_from_name(kokopop::backend_name(value), parsed));
        CHECK_EQ(parsed, value);
        CHECK(std::strlen(kokopop::backend_display_name(value)) > 0);
    }
}

TEST_CASE("backend_names_reject_unknown") {
    int32_t parsed = KOKOPOP_BACKEND_METAL;
    CHECK(!kokopop::backend_from_name("rocm", parsed));
    CHECK(!kokopop::backend_from_name(nullptr, parsed));
    // Rejected input leaves the caller's value untouched.
    CHECK_EQ(parsed, KOKOPOP_BACKEND_METAL);
    // AUTO is not a spelling the tools accept.
    CHECK(!kokopop::backend_from_name("auto", parsed));
    CHECK_EQ(std::string(kokopop::backend_name(KOKOPOP_BACKEND_AUTO)), "auto");
    // Every accepted name appears in the usage list.
    const std::string list = kokopop::backend_name_list();
    CHECK(list.find("opencl") != std::string::npos);
    CHECK(list.find("vulkan") != std::string::npos);
}
