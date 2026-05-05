#include "playback_stdout.h"

#include <cstdio>
#include <mutex>

namespace kokopop {

StdoutPlayback::StdoutPlayback() = default;

StdoutPlayback::~StdoutPlayback() {
    stop();
}

bool StdoutPlayback::start(int sample_rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    sample_rate_ = sample_rate;
    started_ = true;
    // Nothing special to do — we just write to stdout
    return true;
}

void StdoutPlayback::write(const float * samples, size_t n) {
    if (!started_ || !samples || n == 0) return;

    std::lock_guard<std::mutex> lock(mutex_);
    size_t written = fwrite(samples, sizeof(float), n, stdout);
    total_bytes_ += written * sizeof(float);
    fflush(stdout);
}

void StdoutPlayback::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    started_ = false;
    // Flush any remaining data
    fflush(stdout);
}

void StdoutPlayback::wait() {
    // Nothing to wait for — stdout writes are synchronous
}

} // namespace kokopop
