#include "playback_coreaudio.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

namespace kokopop {
namespace {

static constexpr int BUFFER_COUNT = 3;
static constexpr int BUFFER_DURATION_MS = 100; // 100ms per buffer

static float clamp_sample(float s) {
    return std::fmaxf(-1.0f, std::fminf(1.0f, s));
}

static float float_to_s16(float f) {
    return clamp_sample(f) * 32767.0f;
}

static inline void pb_log(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fputs("[kokopop-playback] ", stderr);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
}

} // namespace

CoreAudioPlayback::CoreAudioPlayback() = default;

CoreAudioPlayback::~CoreAudioPlayback() {
    stop();
}

bool CoreAudioPlayback::start(int sample_rate) {
    if (queue_) return false;

    sample_rate_ = sample_rate;
    buffer_size_samples_ = (sample_rate * BUFFER_DURATION_MS) / 1000;
    pb_log("start: sr=%d buf_samples=%d\n", sample_rate, buffer_size_samples_);

    AudioStreamBasicDescription format;
    format.mSampleRate = sample_rate;
    format.mFormatID = kAudioFormatLinearPCM;
    format.mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked;
    format.mBytesPerPacket = 2;
    format.mFramesPerPacket = 1;
    format.mBytesPerFrame = 2;
    format.mChannelsPerFrame = 1;
    format.mBitsPerChannel = 16;

    OSStatus status = AudioQueueNewOutput(&format, callback, this, nullptr, nullptr, 0, &queue_);
    if (status != noErr) {
        pb_log("AudioQueueNewOutput failed: 0x%08x\n", status);
        return false;
    }
    pb_log("AudioQueueNewOutput OK\n");

    buffers_.reserve(BUFFER_COUNT);
    packet_descs_.reserve(BUFFER_COUNT);
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        AudioQueueBufferRef buf = nullptr;
        AudioQueueAllocateBuffer(queue_, buffer_size_samples_ * 2, &buf);
        if (!buf) {
            pb_log("AudioQueueAllocateBuffer[%d] failed\n", i);
            AudioQueueDispose(queue_, true);
            queue_ = nullptr;
            return false;
        }
        buffers_.push_back(buf);
        packet_descs_.push_back({});
    }
    pb_log("Allocated %d buffers (%d bytes each)\n", BUFFER_COUNT, buffer_size_samples_ * 2);

    running_ = true;
    done_ = false;
    drained_ = false;
    stopped_ = false;

    // Prime buffers (silence — real data hasn't arrived yet)
    std::lock_guard<std::mutex> lock(data_mutex_);
    for (auto & buf : buffers_) {
        fill_and_enqueue(buf);
    }

    status = AudioQueueStart(queue_, nullptr);
    if (status != noErr) {
        pb_log("AudioQueueStart failed: 0x%08x\n", status);
        AudioQueueDispose(queue_, true);
        queue_ = nullptr;
        return false;
    }
    pb_log("AudioQueueStart OK\n");
    return true;
}

void CoreAudioPlayback::write(const float * samples, size_t n) {
    if (!queue_ || !samples || n == 0) return;

    std::lock_guard<std::mutex> lock(data_mutex_);
    data_queue_.emplace(samples, samples + n);
    pb_log("write: +%zu samples (total queued chunks: %zu)\n", n, data_queue_.size());
    data_cv_.notify_one();
}

void CoreAudioPlayback::stop() {
    pb_log("stop() called\n");

    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (stopped_) {
            pb_log("stop: already stopped\n");
            return;
        }
        // Signal: no more data will arrive
        done_ = true;
    }

    // The callback will detect (done_ && data_queue_.empty()) and signal drained_.
    // We wait WITHOUT holding data_mutex_ so the callback can still fire.
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        drain_cv_.wait(lock, [this]() { return drained_; });
    }
    pb_log("stop: all data drained from queue\n");

    // Now stop the queue synchronously.
    // Callbacks will still fire for currently-playing buffers, but we've
    // already released data_mutex_ so they can run.
    if (queue_) {
        pb_log("stop: AudioQueueStop (sync)\n");
        AudioQueueStop(queue_, true);  // blocks until queue fully stops

        pb_log("stop: AudioQueueDispose\n");
        AudioQueueDispose(queue_, true);
        queue_ = nullptr;
        buffers_.clear();
        packet_descs_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        running_ = false;
        stopped_ = true;
    }
    data_cv_.notify_all();
    drain_cv_.notify_all();
    pb_log("stop: done\n");
}

void CoreAudioPlayback::wait() {
    pb_log("wait() entered\n");
    // Wait until stop() has fully completed
    std::unique_lock<std::mutex> lock(data_mutex_);
    data_cv_.wait(lock, [this]() { return stopped_; });
    pb_log("wait() returned\n");
}

void CoreAudioPlayback::callback(void * userData,
                                  AudioQueueRef /*queue*/,
                                  AudioQueueBufferRef buffer) {
    auto * self = static_cast<CoreAudioPlayback *>(userData);
    if (!self) return;

    {
        std::lock_guard<std::mutex> lock(self->data_mutex_);

        bool has_data = !self->data_queue_.empty();
        bool still_playing = self->running_ && self->queue_ != nullptr;
        pb_log("callback: running=%d done=%d has_data=%d\n",
               self->running_, self->done_, has_data);

        if (still_playing || has_data) {
            self->fill_and_enqueue(buffer);
        } else {
            // No more data and queue is stopping — enqueue silence
            std::memset(buffer->mAudioData, 0, self->buffer_size_samples_ * 2);
            buffer->mAudioDataByteSize = self->buffer_size_samples_ * 2;
            if (self->queue_) {
                AudioQueueEnqueueBuffer(self->queue_, buffer, 0, nullptr);
            }
        }

        // Signal drained when: done_ is set AND all data has been consumed
        if (self->done_ && self->data_queue_.empty() && !self->drained_) {
            self->drained_ = true;
            pb_log("callback: signaling drained (all data enqueued)\n");
            self->drain_cv_.notify_one();
        }
    }
}

void CoreAudioPlayback::fill_and_enqueue(AudioQueueBufferRef buffer) {
    if (!buffer || !queue_) return;

    std::vector<float> samples;
    while (!data_queue_.empty()) {
        auto & front = data_queue_.front();
        if (!front.empty()) {
            size_t take = std::min(front.size(), static_cast<size_t>(buffer_size_samples_));
            samples.insert(samples.end(), front.begin(), front.begin() + take);
            front.erase(front.begin(), front.begin() + take);
            if (front.empty()) {
                data_queue_.pop();
            }
            break;
        } else {
            data_queue_.pop();
        }
    }

    if (samples.empty()) {
        std::memset(buffer->mAudioData, 0, buffer_size_samples_ * 2);
        buffer->mAudioDataByteSize = buffer_size_samples_ * 2;
        AudioQueueEnqueueBuffer(queue_, buffer, 0, nullptr);
        pb_log("fill_and_enqueue: silence\n");
        return;
    }

    // Convert float32 to int16
    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        pcm[i] = static_cast<int16_t>(float_to_s16(samples[i]));
    }

    buffer->mAudioDataByteSize = static_cast<UInt32>(samples.size() * sizeof(int16_t));
    std::memcpy(buffer->mAudioData, pcm.data(), buffer->mAudioDataByteSize);
    AudioQueueEnqueueBuffer(queue_, buffer, 0, nullptr);
    pb_log("fill_and_enqueue: %zu samples (%d bytes)\n",
           samples.size(), buffer->mAudioDataByteSize);
}

} // namespace kokopop
