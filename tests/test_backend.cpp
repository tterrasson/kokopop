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
    // backend_type is one of AUTO, CPU, or METAL
    int valid = (model->backend_type == KOKOPOP_BACKEND_AUTO)
              + (model->backend_type == KOKOPOP_BACKEND_CPU)
              + (model->backend_type == KOKOPOP_BACKEND_METAL);
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
    // backend_type stores the *actual* backend used (METAL or CPU), not what was requested.
    // AUTO is resolved at runtime, so backend_type should never be AUTO after load.
    CHECK_NE(model->backend_type, KOKOPOP_BACKEND_AUTO);
    // backend_type should be either CPU (1) or METAL (2)
    CHECK_GE(model->backend_type, KOKOPOP_BACKEND_CPU);
    CHECK_LE(model->backend_type, KOKOPOP_BACKEND_METAL);
    // The actual backend is non-null (CPU or Metal resolved at runtime)
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
