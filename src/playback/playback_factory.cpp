#include "playback.h"
#include "playback_stdout.h"
#include "playback_dummy.h"

#include <memory>

#if defined(__APPLE__)
#include "playback_coreaudio.h"
#endif

namespace kokopop {

AudioPlayback * create_stdout_playback() {
    return new StdoutPlayback();
}

AudioPlayback * create_dummy_playback() {
    return new DummyPlayback();
}

#if defined(__APPLE__)
AudioPlayback * create_coreaudio_playback() {
    return new CoreAudioPlayback();
}
#endif

AudioPlayback * create_default_playback() {
#if defined(__APPLE__)
    // Try Core Audio first, fallback to stdout
    auto * ca = new CoreAudioPlayback();
    if (ca) {
        // We can't test start() here without a sample rate
        // Just return it and let the caller handle failures
        return ca;
    }
#endif
    return create_stdout_playback();
}

void free_playback(AudioPlayback * pb) {
    delete pb;
}

} // namespace kokopop
