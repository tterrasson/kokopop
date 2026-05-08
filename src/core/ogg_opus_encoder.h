#pragma once

#ifdef KOKOPOP_HAS_OPUS

#include <cstdint>
#include <memory>
#include <vector>

struct OggOpusComments;
struct OggOpusEnc;

namespace kokopop {

// Encodes float32 PCM into an Ogg/Opus stream using libopusenc pull mode.
//
// Pull mode lets the HTTP server decide when complete Ogg pages are sent.
// During live streaming, pages should be pulled without forcing a flush so the
// muxer can keep page timing continuous. The final drain writes the EOS page.
//
// Not thread-safe — used exclusively by the HTTP event loop thread.
class OggOpusEncoder {
public:
    OggOpusEncoder(int sample_rate, int channels = 1);
    ~OggOpusEncoder();

    OggOpusEncoder(const OggOpusEncoder &) = delete;
    OggOpusEncoder & operator=(const OggOpusEncoder &) = delete;

    // Write samples into the encoder without pulling pages yet.
    // Call pull_pages() after writing all available synthesis chunks to batch
    // the page emission (avoids inter-chunk gaps in the Ogg stream).
    void write(const float * samples, int n_samples);

    // Finalize and expose Ogg/Opus header pages before audio is available.
    void flush_header();

    // Pull Ogg pages accumulated since the last pull_pages() call.
    // Use flush=0 while streaming so pages are emitted on the muxer's natural
    // cadence. Use flush=1 only at EOS or when deliberately trading continuity
    // for lower latency.
    void pull_pages(int flush = 0);

    // Flush remaining buffered frames and write EOS page.
    void drain();

    // Returns accumulated Ogg pages and clears the internal buffer.
    std::vector<uint8_t> take_pending();

    bool has_pending() const { return !_page_buffer.empty(); }

private:
    void _pull_pages(int flush);

    OggOpusComments * _comments = nullptr;
    OggOpusEnc *      _enc      = nullptr;
    std::vector<uint8_t> _page_buffer;
};

} // namespace kokopop

#endif // KOKOPOP_HAS_OPUS
