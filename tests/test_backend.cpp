#include "test_helpers.h"
#include "backend/backend.h"

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
              + (model->backend_type == KOKOPOP_BACKEND_VULKAN);
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
    CHECK_LE(model->backend_type, KOKOPOP_BACKEND_VULKAN);
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
