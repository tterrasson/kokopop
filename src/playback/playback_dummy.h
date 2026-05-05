#pragma once

#include "playback.h"

namespace kokopop {

/// No-op playback — discards all audio data
/// Used as fallback on platforms without audio support
class DummyPlayback : public AudioPlayback {
public:
    DummyPlayback() = default;
    ~DummyPlayback() override = default;

    bool start(int) override { return true; }
    void write(const float *, size_t) override { /* discard */ }
    void stop() override { }
    void wait() override { }
};

} // namespace kokopop
