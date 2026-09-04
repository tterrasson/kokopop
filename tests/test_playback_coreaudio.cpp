#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "playback/playback_coreaudio.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

// A device consumes each buffer before returning it. An immediate stop or
// disposal loses pending samples, just as the real AudioQueue does.
struct MockAudioQueue {
    AudioQueueOutputCallback callback;
    void * user;
    std::mutex mutex;
    std::queue<AudioQueueBufferRef> pending;
    std::vector<AudioQueueBufferRef> allocated;
    std::atomic<bool> running{false}, stopping{false};
    std::thread worker;
};
static std::atomic<size_t> played{0};
static bool fail_start = false;
static bool disposed_while_running = false;
OSStatus AudioQueueNewOutput(const AudioStreamBasicDescription *, AudioQueueOutputCallback cb,
                            void * user, void *, void *, UInt32, AudioQueueRef * out) {
    *out = new MockAudioQueue;
    (*out)->callback = cb;
    (*out)->user = user;
    return noErr;
}
OSStatus AudioQueueAllocateBuffer(AudioQueueRef q, UInt32 size, AudioQueueBufferRef * out) {
    *out = new AudioQueueBuffer{std::malloc(size), 0};
    q->allocated.push_back(*out);
    return noErr;
}
OSStatus AudioQueueEnqueueBuffer(AudioQueueRef q, AudioQueueBufferRef b, UInt32, const void *) {
    std::lock_guard<std::mutex> lock(q->mutex);
    q->pending.push(b);
    return noErr;
}
OSStatus AudioQueueStart(AudioQueueRef q, const void *) {
    if (fail_start) return -1;
    q->running = true;
    q->worker = std::thread([q] {
        while (q->running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            AudioQueueBufferRef b = nullptr;
            {
                std::lock_guard<std::mutex> lock(q->mutex);
                if (!q->pending.empty()) {
                    b = q->pending.front();
                    q->pending.pop();
                } else if (q->stopping) {
                    q->running = false;
                }
            }
            if (b && q->running) {
                const auto * pcm = static_cast<int16_t *>(b->mAudioData);
                for (size_t i = 0; i < b->mAudioDataByteSize / 2; ++i)
                    if (pcm[i] != 0) ++played;
                q->callback(q->user, q, b);
            }
        }
    });
    return noErr;
}
OSStatus AudioQueueStop(AudioQueueRef q, bool immediate) {
    q->stopping = true;
    if (immediate) q->running = false;
    return noErr;
}
OSStatus AudioQueueGetProperty(AudioQueueRef q, UInt32, void * out, UInt32 *) {
    *static_cast<UInt32 *>(out) = q->running ? 1 : 0;
    return noErr;
}
OSStatus AudioQueueDispose(AudioQueueRef q, bool) {
    disposed_while_running |= q->running;
    q->running = false;
    if (q->worker.joinable()) q->worker.join();
    for (auto * b : q->allocated) { std::free(b->mAudioData); delete b; }
    delete q;
    return noErr;
}
TEST_CASE("Core Audio drains every sample, including a partial final buffer") {
    for (int rate : {22050, 24000}) {
        for (size_t count : {size_t(1), size_t(rate / 10), size_t(rate + 137)}) {
            CAPTURE(rate);
            CAPTURE(count);
            played = 0;
            disposed_while_running = false;
            kokopop::CoreAudioPlayback playback;
            REQUIRE(playback.start(rate));
            std::vector<float> samples(count, 0.5f);
            playback.write(samples.data(), count);
            playback.stop();
            playback.wait();
            CHECK(played == count);
            CHECK_FALSE(disposed_while_running);
            playback.stop();
        }
    }
}
TEST_CASE("Core Audio handles empty playback, restart and failed start") {
    kokopop::CoreAudioPlayback playback;
    playback.stop();
    playback.wait();
    for (int i = 0; i < 2; ++i) {
        REQUIRE(playback.start(22050));
        playback.stop();
    }
    fail_start = true;
    CHECK_FALSE(playback.start(22050));
    playback.stop();
    playback.wait();
    fail_start = false;
    REQUIRE(playback.start(24000));
    playback.stop();
}
