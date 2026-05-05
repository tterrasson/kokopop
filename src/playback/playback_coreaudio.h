#pragma once

#include "playback.h"

#include <AudioToolbox/AudioToolbox.h>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

namespace kokopop {

/// Playback using macOS Core Audio (AudioQueue)
///
/// Uses a ring buffer of AudioQueue buffers for smooth playback.
class CoreAudioPlayback : public AudioPlayback {
public:
    CoreAudioPlayback();
    ~CoreAudioPlayback() override;

    bool start(int sample_rate) override;
    void write(const float * samples, size_t n) override;
    void stop() override;
    void wait() override;

private:
    void fill_and_enqueue(AudioQueueBufferRef buffer);

    static void callback(void * userData,
                         AudioQueueRef queue,
                         AudioQueueBufferRef buffer);

    AudioQueueRef queue_ = nullptr;
    std::vector<AudioQueueBufferRef> buffers_;
    std::vector<AudioStreamPacketDescription> packet_descs_;

    std::mutex data_mutex_;
    std::condition_variable data_cv_;      // used by wait()
    std::condition_variable drain_cv_;     // used by stop() to wait for drain
    std::queue<std::vector<float>> data_queue_;
    bool running_ = false;
    bool done_ = false;       // no more write() calls will happen
    bool drained_ = false;    // callback confirmed all data has been enqueued
    bool stopped_ = false;    // stop() has fully completed

    int sample_rate_ = 24000;
    int buffer_size_samples_ = 0;
};

} // namespace kokopop
