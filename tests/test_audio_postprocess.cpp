#include "test_helpers.h"
#include "audio/audio_postprocess.h"
#include "synthesis/chunker/chunker.h"

using namespace kokopop;

// ---- apply_crossfade ----

TEST_CASE("apply_crossfade prev empty") {
    std::vector<float> prev;
    std::vector<float> next = {0.1f, 0.2f, 0.3f};
    auto result = apply_crossfade(prev, next, 100, 24000);
    CHECK_EQ(result.size(), 3u);
    CHECK_EQ(result[0], 0.1f);
    CHECK_EQ(result[1], 0.2f);
    CHECK_EQ(result[2], 0.3f);
}

TEST_CASE("apply_crossfade next empty") {
    std::vector<float> prev = {0.1f, 0.2f, 0.3f};
    std::vector<float> next;
    auto result = apply_crossfade(prev, next, 100, 24000);
    CHECK_EQ(result.size(), 3u);
    CHECK_EQ(result[0], 0.1f);
    CHECK_EQ(result[1], 0.2f);
    CHECK_EQ(result[2], 0.3f);
}

TEST_CASE("apply_crossfade too_short_concat") {
    std::vector<float> prev = {0.1f, 0.2f};
    std::vector<float> next = {0.3f, 0.4f};
    // 100ms at 24000 = 2400 samples needed, but we only have 2 each
    auto result = apply_crossfade(prev, next, 100, 24000);
    CHECK_EQ(result.size(), 4u);
    CHECK_EQ(result[0], 0.1f);
    CHECK_EQ(result[1], 0.2f);
    CHECK_EQ(result[2], 0.3f);
    CHECK_EQ(result[3], 0.4f);
}

TEST_CASE("apply_crossfade n_zero_concat") {
    std::vector<float> prev = {0.1f, 0.2f};
    std::vector<float> next = {0.3f, 0.4f};
    auto result = apply_crossfade(prev, next, 0, 24000);
    CHECK_EQ(result.size(), 4u);
    CHECK_EQ(result[0], 0.1f);
    CHECK_EQ(result[3], 0.4f);
}

TEST_CASE("apply_crossfade_correct_overlay") {
    // Use a crossfade_ms that produces n=24 at 24000 Hz
    const int crossfade_ms = 1;
    const int n = (crossfade_ms * 24000) / 1000; // 24
    std::vector<float> prev(n + 10, 0.0f);
    std::vector<float> next(n + 10, 0.0f);
    // Fill tail of prev with 1.0
    for (int i = 0; i < n; ++i) {
        prev[prev.size() - n + i] = 1.0f;
    }
    // Fill ALL of next with 2.0 so the post-fade tail is also 2.0
    for (int i = 0; i < static_cast<int>(next.size()); ++i) {
        next[i] = 2.0f;
    }

    auto result = apply_crossfade(prev, next, crossfade_ms, 24000);

    // Result size: prev.size() + next.size() - n = (n+10) + (n+10) - n = n + 20
    CHECK_EQ(result.size(), static_cast<size_t>(n + 20));

    // First part: prev[0..9] = 0.0f (pre-fade)
    CHECK_EQ(result[0], 0.0f);

    // Crossfade region: n samples with blended values
    // First crossfade sample: t=0, prev_tail=prev[10]=1.0, next_head=next[0]=2.0 -> 1.0
    CHECK_EQ(result[static_cast<size_t>(10)], 1.0f);

    // Rest of next: next[n..n+9] = 2.0f
    CHECK_EQ(result[static_cast<size_t>(n + 10)], 2.0f);
}

// ---- apply_crossfade_smart ----

TEST_CASE("apply_crossfade_smart Sentence no crossfade") {
    std::vector<float> prev = {1.0f, 1.0f, 1.0f};
    std::vector<float> next = {2.0f, 2.0f, 2.0f};
    auto result = apply_crossfade_smart(prev, next, Boundary::Sentence, 100, 24000);
    CHECK_EQ(result.size(), 6u);
    CHECK_EQ(result[0], 1.0f);
    CHECK_EQ(result[3], 2.0f);
}

TEST_CASE("apply_crossfade_smart Paragraph no crossfade") {
    std::vector<float> prev = {1.0f};
    std::vector<float> next = {2.0f};
    auto result = apply_crossfade_smart(prev, next, Boundary::Paragraph, 100, 24000);
    CHECK_EQ(result.size(), 2u);
    CHECK_EQ(result[0], 1.0f);
    CHECK_EQ(result[1], 2.0f);
}

TEST_CASE("apply_crossfade_smart Newline no crossfade") {
    std::vector<float> prev = {1.0f};
    std::vector<float> next = {2.0f};
    auto result = apply_crossfade_smart(prev, next, Boundary::Newline, 100, 24000);
    CHECK_EQ(result.size(), 2u);
    CHECK_EQ(result[0], 1.0f);
    CHECK_EQ(result[1], 2.0f);
}

TEST_CASE("apply_crossfade_smart ClauseWeak crossfade applied") {
    const int crossfade_ms = 1;
    const int n = (crossfade_ms * 24000) / 1000;
    std::vector<float> prev(n + 10, 0.5f);
    std::vector<float> next(n + 10, 1.0f);
    auto result = apply_crossfade_smart(prev, next, Boundary::ClauseWeak,
                                         crossfade_ms, 24000);
    // Crossfaded: size = (n+10) + (n+10) - n = n + 20, NOT simple concat 2*(n+10)
    CHECK_EQ(result.size(), static_cast<size_t>(n + 20));
    // First sample is from prev (pre-fade)
    CHECK_EQ(result[0], 0.5f);
}

// ---- trim_leading_silence ----

TEST_CASE("trim_leading_silence_no_silence") {
    std::vector<float> audio = {0.5f, -0.3f, 0.1f, 0.0f};
    auto result = trim_leading_silence(audio, 500, 24000);
    CHECK_EQ(result.size(), audio.size());
    CHECK_EQ(result[0], 0.5f);
}

TEST_CASE("trim_leading_silence_leading_silence") {
    // Window = 50ms = 1200 samples at 24000 Hz, step = 300 samples.
    // The function returns audio[edge..end] where edge is the start of
    // the first window that contains non-silent data. edge can be up to
    // 300 samples before the actual non-silent data.
    // Use 1500 silence + 2 non-silent = 1502 total.
    // i=0: [0..1200] — silent
    // i=300: [300..1500] — silent (samples 300-1499 are zeros)
    // i=600: [600..1502] — has non-silent at 1500, returns 600
    // result = audio[600..1501]. Samples 600-1499 are 0, 1500-1501 are non-silent.
    std::vector<float> audio(1500, 0.0f);
    audio.push_back(0.5f);
    audio.push_back(-0.3f);
    auto result = trim_leading_silence(audio, 500, 24000);
    CHECK(result.size() < audio.size());
    // The last sample should be non-silent (it's preserved)
    size_t last_idx = result.size() - 1;
    CHECK(std::fabs(result[last_idx]) > 0.01f);
}

TEST_CASE("trim_leading_silence_empty_input") {
    std::vector<float> audio;
    auto result = trim_leading_silence(audio, 500, 24000);
    CHECK(result.empty());
}

// ---- trim_trailing_silence ----

TEST_CASE("trim_trailing_silence_no_silence") {
    std::vector<float> audio = {0.5f, -0.3f, 0.1f};
    auto result = trim_trailing_silence(audio, 500, 24000);
    CHECK_EQ(result.size(), audio.size());
    CHECK_EQ(result[0], 0.5f);
}

TEST_CASE("trim_trailing_silence_trailing_silence") {
    // Same 50ms window logic but from end.
    // Need >= 12000 non-silent samples so the first window from the end
    // (at i=12000) covers only non-silent data.
    // check_end = min(non_silent + silence, 12000).
    // If non_silent >= 12000: check_end = 12000, i=12000: check [10800..12000) all non-silent.
    // Returns 12000. result = audio[0..11999], last = audio[11999] = non-silent.
    const int non_silent_count = 12000;
    std::vector<float> audio(non_silent_count, 0.5f);
    for (int i = 0; i < 200; ++i) {
        audio.push_back(0.0f);
    }
    auto result = trim_trailing_silence(audio, 500, 24000);
    CHECK(result.size() < audio.size());
    CHECK(std::fabs(result.back()) > 0.01f);
}

TEST_CASE("trim_trailing_silence_empty_input") {
    std::vector<float> audio;
    auto result = trim_trailing_silence(audio, 500, 24000);
    CHECK(result.empty());
}

// ---- append_silence ----

TEST_CASE("append_silence_positive_ms") {
    std::vector<float> audio = {0.5f};
    // 100ms at 24000 = 2400 samples
    append_silence(audio, 100, 24000);
    CHECK_EQ(audio.size(), static_cast<size_t>(2401));
    CHECK_EQ(audio[0], 0.5f);
    for (size_t i = 1; i < audio.size(); ++i) {
        CHECK_EQ(audio[i], 0.0f);
    }
}

TEST_CASE("append_silence_zero_ms") {
    std::vector<float> audio = {0.5f};
    append_silence(audio, 0, 24000);
    CHECK_EQ(audio.size(), 1u);
    CHECK_EQ(audio[0], 0.5f);
}

TEST_CASE("append_silence_negative_ms") {
    std::vector<float> audio = {0.5f};
    append_silence(audio, -50, 24000);
    CHECK_EQ(audio.size(), 1u);
    CHECK_EQ(audio[0], 0.5f);
}

// ---- pause_for_boundary ----

TEST_CASE("pause_for_boundary_paragraph") {
    ChunkConfig cfg;
    cfg.paragraph_pause_ms = 400;
    cfg.sentence_pause_ms = 180;
    cfg.comma_pause_ms = 80;
    CHECK_EQ(pause_for_boundary(Boundary::Paragraph, cfg), 400);
}

TEST_CASE("pause_for_boundary_newline") {
    ChunkConfig cfg;
    cfg.paragraph_pause_ms = 400;
    cfg.sentence_pause_ms = 180;
    cfg.comma_pause_ms = 80;
    // Newline = half paragraph pause
    CHECK_EQ(pause_for_boundary(Boundary::Newline, cfg), 200);
}

TEST_CASE("pause_for_boundary_sentence") {
    ChunkConfig cfg;
    cfg.sentence_pause_ms = 180;
    CHECK_EQ(pause_for_boundary(Boundary::Sentence, cfg), 180);
}

TEST_CASE("pause_for_boundary_clause_strong") {
    ChunkConfig cfg;
    cfg.sentence_pause_ms = 180;
    cfg.comma_pause_ms = 80;
    CHECK_EQ(pause_for_boundary(Boundary::ClauseStrong, cfg), static_cast<int>(180 * 0.7f));
}

TEST_CASE("pause_for_boundary_clause_weak") {
    ChunkConfig cfg;
    cfg.comma_pause_ms = 80;
    CHECK_EQ(pause_for_boundary(Boundary::ClauseWeak, cfg), 80);
}

TEST_CASE("pause_for_boundary_none") {
    ChunkConfig cfg;
    CHECK_EQ(pause_for_boundary(Boundary::None, cfg), 0);
}

// ---- postprocess_chunk_audio ----

TEST_CASE("postprocess_chunk_audio_first_chunk_no_trim_leading") {
    std::vector<float> audio = {0.5f, 0.3f, 0.1f, -0.05f};
    Chunk chunk;
    chunk.is_first = true;
    chunk.is_last = false;
    chunk.boundary_after = Boundary::Sentence;

    ChunkConfig cfg;
    cfg.trim_silence = true;
    cfg.max_silence_trim_ms = 120;
    cfg.sentence_pause_ms = 180;

    auto result = postprocess_chunk_audio(audio, chunk, 0, 3, cfg, 24000);

    int pause_samples = (180 * 24000) / 1000; // 4320
    // The result should be longer due to pause addition
    CHECK_GT(result.size(), audio.size());
    // Leading content preserved (not trimmed for first chunk)
    CHECK_EQ(result[0], 0.5f);
}

TEST_CASE("postprocess_chunk_audio_not_first_trims_leading") {
    std::vector<float> audio = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.3f};
    Chunk chunk;
    chunk.is_first = false;
    chunk.is_last = false;
    chunk.boundary_after = Boundary::ClauseWeak;

    ChunkConfig cfg;
    cfg.trim_silence = true;
    cfg.max_silence_trim_ms = 120;
    cfg.comma_pause_ms = 80;

    auto result = postprocess_chunk_audio(audio, chunk, 1, 3, cfg, 24000);
    CHECK(result.size() > 0);
    // Comma pause added
    int comma_samples = (80 * 24000) / 1000; // 1920
    CHECK(result.size() > audio.size());
}

TEST_CASE("postprocess_chunk_audio_last_chunk_no_pause") {
    std::vector<float> audio = {0.5f, -0.3f, 0.1f};
    Chunk chunk;
    chunk.is_first = false;
    chunk.is_last = true;
    chunk.boundary_after = Boundary::Sentence;

    ChunkConfig cfg;
    cfg.trim_silence = true;
    cfg.max_silence_trim_ms = 120;
    cfg.sentence_pause_ms = 180;

    auto result = postprocess_chunk_audio(audio, chunk, 2, 3, cfg, 24000);

    // No pause should be added after last chunk
    // Size should be close to original (minus any trailing silence trim)
    int pause_samples = (180 * 24000) / 1000;
    CHECK_LT(result.size(), static_cast<size_t>(audio.size() + pause_samples));
}

TEST_CASE("postprocess_chunk_audio_disabled_trim") {
    std::vector<float> audio = {0.0f, 0.0f, 0.5f, -0.3f, 0.0f, 0.0f};
    Chunk chunk;
    chunk.is_first = false;
    chunk.is_last = false;
    chunk.boundary_after = Boundary::Sentence;

    ChunkConfig cfg;
    cfg.trim_silence = false;
    cfg.sentence_pause_ms = 180;

    auto result = postprocess_chunk_audio(audio, chunk, 1, 3, cfg, 24000);

    // No trimming should happen
    // Only sentence pause added
    int pause_samples = (180 * 24000) / 1000;
    CHECK_EQ(result.size(), audio.size() + static_cast<size_t>(pause_samples));
}

TEST_CASE("postprocess_chunk_audio_empty_audio") {
    std::vector<float> audio;
    Chunk chunk;
    chunk.is_first = true;
    chunk.is_last = true;
    chunk.boundary_after = Boundary::None;

    ChunkConfig cfg;
    auto result = postprocess_chunk_audio(audio, chunk, 0, 1, cfg, 24000);
    CHECK(result.empty());
}

// ---- Regression coverage for the single-allocation fused postprocess path ----

TEST_CASE("postprocess_chunk_audio_all_silence_trimmed_with_pause_yields_pause_zeros") {
    // The fused path computes (start, end) offsets in-place and then allocates
    // body+pause in one shot. When the audio is fully silent (start == end),
    // body=0 and only the pause-zero tail must remain. Earlier copy-based
    // path produced the same result via a chain of empty vectors; this test
    // pins down the contract.
    std::vector<float> audio(2400, 0.0f);  // 100 ms of silence at 24 kHz
    Chunk chunk;
    chunk.is_first      = false;          // leading trim allowed
    chunk.is_last       = false;          // pause appended
    chunk.boundary_after = Boundary::ClauseWeak;

    ChunkConfig cfg;
    cfg.trim_silence       = true;
    cfg.max_silence_trim_ms = 500;        // window wider than audio
    cfg.comma_pause_ms     = 50;

    auto result = postprocess_chunk_audio(audio, chunk, 1, 3, cfg, 24000);
    const size_t pause_n = static_cast<size_t>((50 * 24000) / 1000);
    CHECK_EQ(result.size(), pause_n);
    for (float v : result) CHECK_EQ(v, 0.0f);
}

TEST_CASE("postprocess_chunk_audio_no_op_when_trim_off_and_no_pause") {
    // Disabled trim, no boundary pause → output must equal input bit-exact.
    // Guards against the fused path inadvertently copying through a path
    // that re-quantizes or zero-fills.
    std::vector<float> audio = {0.123456f, -0.7f, 0.0f, 1e-9f, -1e-9f, 0.999f};
    Chunk chunk;
    chunk.is_first       = true;
    chunk.is_last        = true;
    chunk.boundary_after = Boundary::None;

    ChunkConfig cfg;
    cfg.trim_silence = false;

    auto result = postprocess_chunk_audio(audio, chunk, 0, 1, cfg, 24000);
    REQUIRE_EQ(result.size(), audio.size());
    for (size_t i = 0; i < audio.size(); ++i) {
        // Bit-exact: postprocess must not mutate the signal in the no-op path.
        CHECK_EQ(result[i], audio[i]);
    }
}

TEST_CASE("postprocess_chunk_audio_matches_chained_trim_calls_for_middle_chunk") {
    // The fused postprocess and the public trim_*_silence helpers must agree
    // sample-by-sample when no boundary pause is appended. This pins the
    // refactored fused path against the still-public trim helpers, so any
    // future drift between them gets caught immediately.
    std::vector<float> audio(6000, 0.0f);
    for (int i = 1000; i < 5000; ++i) {
        audio[static_cast<size_t>(i)] = 0.3f * std::sin(0.01f * i);
    }

    Chunk chunk;
    chunk.is_first       = false;
    chunk.is_last        = true;   // no pause appended
    chunk.boundary_after = Boundary::None;

    ChunkConfig cfg;
    cfg.trim_silence        = true;
    cfg.max_silence_trim_ms = 200;

    auto fused = postprocess_chunk_audio(audio, chunk, 1, 3, cfg, 24000);
    auto chained = trim_leading_silence(audio, cfg.max_silence_trim_ms, 24000);
    chained = trim_trailing_silence(chained, cfg.max_silence_trim_ms, 24000);
    REQUIRE_EQ(fused.size(), chained.size());
    for (size_t i = 0; i < fused.size(); ++i) CHECK_EQ(fused[i], chained[i]);
}
