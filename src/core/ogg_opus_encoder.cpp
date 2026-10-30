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
    // Allocate C++ storage before acquiring C resources, so allocation failure
    // cannot leak an encoder from this not-yet-constructed object.
    _page_buffer.reserve(16 * 1024);
    _comments = ope_comments_create();
    if (_comments == nullptr) {
        throw std::bad_alloc();
    }
    ope_comments_add(_comments, "ENCODER", "kokopop");

    int err = OPE_OK;
    _enc = ope_encoder_create_pull(_comments, sample_rate, channels, 0, &err);
    if (!_enc || err != OPE_OK) {
        // Release everything acquired so far: _enc may be non-null on failure.
        if (_enc != nullptr) {
            ope_encoder_destroy(_enc);
            _enc = nullptr;
        }
        if (_comments != nullptr) {
            ope_comments_destroy(_comments);
            _comments = nullptr;
        }
        throw std::runtime_error(
            std::string("OggOpusEncoder: ope_encoder_create_pull failed: ")
            + ope_strerror(err));
    }

    // Keep encoder-side latency low for live TTS. The defaults are tuned for
    // file output and can withhold most of the first synthesis chunk even after
    // the server has written several seconds of PCM into libopusenc.
    ope_encoder_ctl(_enc, OPE_SET_DECISION_DELAY(0));
    ope_encoder_ctl(_enc, OPE_SET_MUXING_DELAY(40));

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

bool OggOpusEncoder::write(const float * samples, int n_samples) {
    if (!_enc) return false;
    return ope_encoder_write_float(_enc, samples, n_samples) == OPE_OK;
}

bool OggOpusEncoder::flush_header() {
    if (!_enc) return false;
    if (ope_encoder_flush_header(_enc) != OPE_OK) {
        return false;
    }
    pull_pages(1);
    return true;
}

bool OggOpusEncoder::drain() {
    if (!_enc) return false;
    const bool ok = ope_encoder_drain(_enc) == OPE_OK;
    // flush=1 on drain: force out whatever partial page remains at EOS.
    pull_pages(1);
    return ok;
}

std::vector<uint8_t> OggOpusEncoder::take_pending() {
    std::vector<uint8_t> out;
    out.swap(_page_buffer);
    return out;
}

} // namespace kokopop

#endif // KOKOPOP_HAS_OPUS
