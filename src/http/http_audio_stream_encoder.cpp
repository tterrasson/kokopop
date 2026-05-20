#include "http/http_audio_stream_encoder.h"

#include "core/ogg_opus_encoder.h"

#include <algorithm>

namespace kokopop {

namespace {

class PcmStreamEncoder final : public HttpAudioStreamEncoder {
public:
    std::string content_type() const override { return "application/octet-stream"; }
    void start(std::vector<char> & /*out*/) override {}
    void write(RequestContext::AudioChunk && chunk, bool /*is_terminal*/,
               std::vector<char> & out) override {
        const char * p = reinterpret_cast<const char *>(chunk.samples.data());
        out.insert(out.end(), p, p + chunk.samples.size() * sizeof(float));
    }
    void finish(bool /*success*/, std::vector<char> & /*out*/) override {}
};

#ifdef KOKOPOP_HAS_OPUS
class OggOpusStreamEncoder final : public HttpAudioStreamEncoder {
public:
    OggOpusStreamEncoder(int sample_rate, int prebuffer_chunks)
        : _encoder(sample_rate),
          _prebuffer_target(std::max(0, prebuffer_chunks)) {}

    std::string content_type() const override { return "audio/ogg"; }

    void start(std::vector<char> & out) override {
        _encoder.flush_header();
        take_pending(out);
    }

    void write(RequestContext::AudioChunk && chunk, bool is_terminal,
               std::vector<char> & out) override {
        if (_prebuffer_target > 0 && !_prebuffer_released) {
            _prebuffered.push_back(std::move(chunk));
            const bool ready =
                static_cast<int>(_prebuffered.size()) >= _prebuffer_target ||
                is_terminal;
            if (!ready) return;
            for (auto & buffered : _prebuffered) {
                write_samples(buffered);
            }
            _prebuffered.clear();
            _prebuffer_released = true;
        } else {
            write_samples(chunk);
        }
        _encoder.pull_pages(0);
        take_pending(out);
    }

    void finish(bool success, std::vector<char> & out) override {
        if (success) {
            _encoder.drain();
            take_pending(out);
        }
    }

private:
    void write_samples(const RequestContext::AudioChunk & chunk) {
        _encoder.write(chunk.samples.data(), static_cast<int>(chunk.samples.size()));
    }

    void take_pending(std::vector<char> & out) {
        if (!_encoder.has_pending()) return;
        auto pages = _encoder.take_pending();
        out.insert(out.end(), pages.begin(), pages.end());
    }

    OggOpusEncoder _encoder;
    int _prebuffer_target = 0;
    bool _prebuffer_released = false;
    std::vector<RequestContext::AudioChunk> _prebuffered;
};
#endif

} // namespace

bool http_audio_format_available(RequestContext::AudioFormat format) {
    if (format == RequestContext::AudioFormat::OGG_OPUS) {
#ifdef KOKOPOP_HAS_OPUS
        return true;
#else
        return false;
#endif
    }
    return true;
}

std::unique_ptr<HttpAudioStreamEncoder> make_http_audio_stream_encoder(
    RequestContext::AudioFormat format,
    int sample_rate,
    int ogg_prebuffer_chunks) {
    if (format == RequestContext::AudioFormat::PCM) {
        return std::make_unique<PcmStreamEncoder>();
    }
    if (format == RequestContext::AudioFormat::OGG_OPUS) {
#ifdef KOKOPOP_HAS_OPUS
        return std::make_unique<OggOpusStreamEncoder>(sample_rate, ogg_prebuffer_chunks);
#else
        (void)sample_rate;
        (void)ogg_prebuffer_chunks;
        return nullptr;
#endif
    }
    return nullptr;
}

} // namespace kokopop
