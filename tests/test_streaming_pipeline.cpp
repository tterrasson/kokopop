#include "test_helpers.h"
#include "streaming/streaming.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ---- Regression coverage for the two-stage streaming pipeline (G8) ----
//
// The pipeline kicks off chunk N+1's inference in a worker thread while
// the main thread post-processes / runs the callback for chunk N. These
// tests pin down four invariants the pipeline must preserve:
//
//   1. Callback receives chunks in monotonically increasing index order.
//   2. Every chunk in the plan produces a callback before is_done() flips.
//   3. Returning false from the callback stops streaming promptly (within
//      one chunk of lookahead).
//   4. join() completes even after early stop — no leaked threads.
//
// All tests run against the mock model so they finish in well under a
// second; the mock synth path is deterministic and produces non-empty audio.

namespace {

struct Capture {
    std::mutex mu;
    std::vector<int> indices;
    std::vector<size_t> sizes;
    bool keep_going = true;
    int  stop_after_index = -1;
};

bool record_callback(const float * data, size_t n_samples, int chunk_index, void * user_data) {
    auto * cap = static_cast<Capture *>(user_data);
    std::lock_guard<std::mutex> lk(cap->mu);
    cap->indices.push_back(chunk_index);
    cap->sizes.push_back(n_samples);
    (void)data;
    if (cap->stop_after_index >= 0 && chunk_index >= cap->stop_after_index) {
        return false;
    }
    return cap->keep_going;
}

std::unique_ptr<kokopop::Model> load_mock_model() {
    const std::string & gguf = shared_mock_gguf();
    kokopop_model_options options{};
    options.n_threads = 1;
    options.backend   = KOKOPOP_BACKEND_CPU;
    std::unique_ptr<kokopop::Model> model;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(gguf, &options, model, error));
    REQUIRE(model != nullptr);
    return model;
}

} // namespace

TEST_CASE("streaming_pipeline_delivers_all_chunks_in_order") {
    auto model = load_mock_model();
    // A multi-sentence text guarantees several chunks under the default
    // interactive chunker config — exercising the pipeline lookahead.
    const std::string text =
        "Premier chunk. Deuxième chunk. Troisième chunk. Quatrième chunk. Cinquième chunk.";

    Capture cap;
    auto handle = kokopop::stream_synthesize(
        *model, text, "af_heart", 1.0f, kokopop::StreamMode::Interactive,
        record_callback, &cap);
    handle.join();

    REQUIRE_FALSE(cap.indices.empty());
    // Monotonically increasing — pipeline must not deliver chunk N+1 before N.
    for (size_t i = 1; i < cap.indices.size(); ++i) {
        CHECK_LT(cap.indices[i - 1], cap.indices[i]);
    }
    // Done flag must be set after join().
    CHECK(handle.is_done());
    // Every delivered chunk had some samples.
    for (size_t n : cap.sizes) CHECK_GT(n, 0u);
}

TEST_CASE("streaming_pipeline_stops_promptly_when_callback_returns_false") {
    auto model = load_mock_model();
    const std::string text =
        "Alpha. Beta. Gamma. Delta. Epsilon. Zeta. Eta. Theta. Iota. Kappa.";

    Capture cap;
    cap.stop_after_index = 1;  // stop after the second chunk's callback fires

    auto handle = kokopop::stream_synthesize(
        *model, text, "af_heart", 1.0f, kokopop::StreamMode::Interactive,
        record_callback, &cap);
    handle.join();

    REQUIRE(cap.indices.size() >= 1u);
    // With a 1-slot lookahead the worker may have produced one extra chunk
    // before the stop flag was observed, so we allow at most one callback
    // beyond the requested stop index. Anything more means the pipeline
    // ignored the stop signal.
    const int last = cap.indices.back();
    CHECK_LE(last, cap.stop_after_index + 1);
    CHECK(handle.is_done());
}

TEST_CASE("streaming_pipeline_handles_single_chunk_text") {
    // Edge case: only one chunk → the worker is started once, the pipeline
    // never queues a lookahead. Must still produce exactly one callback and
    // tear down cleanly.
    auto model = load_mock_model();
    Capture cap;
    auto handle = kokopop::stream_synthesize(
        *model, "abc", "af_heart", 1.0f, kokopop::StreamMode::Interactive,
        record_callback, &cap);
    handle.join();

    CHECK_EQ(cap.indices.size(), 1u);
    CHECK_EQ(cap.indices[0], 0);
    CHECK(handle.is_done());
}

TEST_CASE("streaming_pipeline_external_stop_is_honoured") {
    // The user can also stop via StreamHandle::stop() (independent of the
    // callback's return value). The worker must reap cleanly without
    // leaking threads or hanging.
    auto model = load_mock_model();
    const std::string text =
        "First. Second. Third. Fourth. Fifth. Sixth. Seventh. Eighth.";

    Capture cap;
    auto handle = kokopop::stream_synthesize(
        *model, text, "af_heart", 1.0f, kokopop::StreamMode::Interactive,
        record_callback, &cap);
    handle.stop();
    handle.join();
    CHECK(handle.is_done());
    // Indices, if any, must still be in order.
    for (size_t i = 1; i < cap.indices.size(); ++i) {
        CHECK_LT(cap.indices[i - 1], cap.indices[i]);
    }
}
