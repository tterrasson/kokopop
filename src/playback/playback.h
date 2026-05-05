#pragma once

#include <cstddef>

namespace kokopop {

// ---------------------------------------------------------------------------
// Audio playback — abstract interface for playing audio samples
// ---------------------------------------------------------------------------

class AudioPlayback {
public:
    virtual ~AudioPlayback() = default;

    /// Start the playback device at the given sample rate
    virtual bool start(int sample_rate) = 0;

    /// Write audio samples (float32, range [-1, 1])
    virtual void write(const float * samples, size_t n) = 0;

    /// Stop playback and wait for queue to drain
    virtual void stop() = 0;

    /// Wait for all queued audio to finish playing
    virtual void wait() = 0;
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/// Create a stdout-based playback (writes PCM float32 to stdout)
AudioPlayback * create_stdout_playback();

/// Create a Core Audio playback (macOS only)
AudioPlayback * create_coreaudio_playback();

/// Create a dummy playback (no-op, for platforms without audio support)
AudioPlayback * create_dummy_playback();

/// Try to create the best available playback for this platform
AudioPlayback * create_default_playback();

/// Free a playback instance
void free_playback(AudioPlayback * pb);

} // namespace kokopop
