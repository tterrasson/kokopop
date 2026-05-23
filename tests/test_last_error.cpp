#include "test_helpers.h"

#include <thread>
#include <string>
#include <atomic>

TEST_CASE("last_error_is_set_on_failure") {
    kokopop_model * model = nullptr;
    const int rc = kokopop_model_load("does-not-exist.gguf", nullptr, &model);
    CHECK(rc != KOKOPOP_OK);
    const char * msg = kokopop_last_error();
    REQUIRE(msg != nullptr);
    CHECK(std::strlen(msg) > 0);
}

TEST_CASE("last_error_persists_until_next_error") {
    // Trigger one error.
    kokopop_audio audio{};
    CHECK_EQ(kokopop_synthesize_phonemes(nullptr, "abc", "af_heart", 1.0f, &audio),
             KOKOPOP_ERROR_INVALID_ARGUMENT);
    const std::string first = kokopop_last_error();
    CHECK(!first.empty());

    // A successful call does not clear the previous error message — current
    // behavior documented here so a future change is intentional.
    const std::string & gguf = shared_mock_gguf();
    kokopop_model * model = nullptr;
    REQUIRE_EQ(kokopop_model_load(gguf.c_str(), nullptr, &model), KOKOPOP_OK);
    const std::string after_success = kokopop_last_error();
    CHECK_EQ(after_success, first);

    // A new failure overwrites the message.
    CHECK_EQ(kokopop_synthesize_phonemes(model, "", "af_heart", 1.0f, &audio),
             KOKOPOP_ERROR_INVALID_ARGUMENT);
    const std::string second = kokopop_last_error();
    CHECK(!second.empty());

    kokopop_model_free(model);
}

TEST_CASE("last_error_is_thread_local") {
    // Set a known error on the main thread.
    kokopop_model * dummy = nullptr;
    CHECK(kokopop_model_load("main-thread-error.gguf", nullptr, &dummy) != KOKOPOP_OK);
    const std::string main_msg = kokopop_last_error();
    REQUIRE(!main_msg.empty());

    // A fresh thread should observe an empty thread-local error string until
    // it triggers its own failure.
    std::string worker_initial;
    std::string worker_after;
    std::thread t([&]() {
        worker_initial = kokopop_last_error();
        kokopop_model * m = nullptr;
        kokopop_model_load("worker-thread-error.gguf", nullptr, &m);
        worker_after = kokopop_last_error();
    });
    t.join();

    CHECK(worker_initial.empty());
    CHECK(!worker_after.empty());
    // The worker's error must not leak into the main thread's slot.
    CHECK_EQ(std::string(kokopop_last_error()), main_msg);
}
