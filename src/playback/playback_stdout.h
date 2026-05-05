#pragma once

#include "playback.h"

#include <cstdio>
#include <mutex>

namespace kokopop {

/// Playback that writes PCM float32 samples to stdout
///
/// Designed for piping:
///   kokopop_stdio_stream ... | ffplay -f f32le -ar 24000 -ac 1 -i -
class StdoutPlayback : public AudioPlayback {
public:
    StdoutPlayback();
    ~StdoutPlayback() override;

    bool start(int sample_rate) override;
    void write(const float * samples, size_t n) override;
    void stop() override;
    void wait() override;

private:
    bool started_ = false;
    int sample_rate_ = 0;
    std::mutex mutex_;
    size_t total_bytes_ = 0;
};

} // namespace kokopop
