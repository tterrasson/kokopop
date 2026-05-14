#ifdef KOKOPOP_HAS_OPUS

#include "core/ogg_opus_encoder.h"

// pkg_check_modules(OPUSENC) adds …/include/opus as include dir,
// so the header is directly accessible as <opusenc.h>.
#include <opusenc.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace kokopop {

OggOpusEncoder::OggOpusEncoder(int sample_rate, int channels) {
    _comments = ope_comments_create();
    ope_comments_add(_comments, "ENCODER", "kokopop");

    int err = OPE_OK;
    _enc = ope_encoder_create_pull(_comments, sample_rate, channels, 0, &err);
    if (!_enc || err != OPE_OK) {
        ope_comments_destroy(_comments);
        _comments = nullptr;
        throw std::runtime_error(
            std::string("OggOpusEncoder: ope_encoder_create_pull failed: ")
            + ope_strerror(err));
    }

    // Keep encoder-side latency low for live TTS. The defaults are tuned for
    // file output and can withhold most of the first synthesis chunk even after
    // the server has written several seconds of PCM into libopusenc.
    ope_encoder_ctl(_enc, OPE_SET_DECISION_DELAY(0));
    ope_encoder_ctl(_enc, OPE_SET_MUXING_DELAY(40));

    // A typical Opus page is well under 4 KB; reserving 16 KB covers the first
    // few pages emitted in a streaming session and avoids tiny incremental
    // allocations from std::vector::insert.
    _page_buffer.reserve(16 * 1024);
}

OggOpusEncoder::~OggOpusEncoder() {
    if (_enc) {
        ope_encoder_destroy(_enc);
        _enc = nullptr;
    }
    if (_comments) {
        ope_comments_destroy(_comments);
        _comments = nullptr;
    }
}

void OggOpusEncoder::pull_pages(int flush) {
    unsigned char * page = nullptr;
    opus_int32 len = 0;
    while (ope_encoder_get_page(_enc, &page, &len, flush)) {
        _page_buffer.insert(_page_buffer.end(), page, page + len);
    }
}

void OggOpusEncoder::write(const float * samples, int n_samples) {
    if (!_enc) return;
    int ret = ope_encoder_write_float(_enc, samples, n_samples);
    if (ret != OPE_OK) {
        std::fprintf(stderr, "[opus] encode error: %s\n", ope_strerror(ret));
    }
}

void OggOpusEncoder::flush_header() {
    if (!_enc) return;
    int ret = ope_encoder_flush_header(_enc);
    if (ret != OPE_OK) {
        std::fprintf(stderr, "[opus] flush header error: %s\n", ope_strerror(ret));
        return;
    }
    pull_pages(1);
}

void OggOpusEncoder::drain() {
    if (!_enc) return;
    int ret = ope_encoder_drain(_enc);
    if (ret != OPE_OK) {
        std::fprintf(stderr, "[opus] drain error: %s\n", ope_strerror(ret));
    }
    // flush=1 on drain: force out whatever partial page remains at EOS.
    pull_pages(1);
}

std::vector<uint8_t> OggOpusEncoder::take_pending() {
    std::vector<uint8_t> out;
    out.swap(_page_buffer);
    return out;
}

} // namespace kokopop

#endif // KOKOPOP_HAS_OPUS
