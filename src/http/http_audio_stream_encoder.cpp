#include "http/http_audio_stream_encoder.h"

#include "audio/audio_encoder.h"

#include <utility>

namespace kokopop {

namespace {

EncodedAudioFormat to_encoded_format(RequestContext::AudioFormat format) {
    if (format == RequestContext::AudioFormat::WAV) return EncodedAudioFormat::WavPcm16;
    if (format == RequestContext::AudioFormat::OGG_OPUS) return EncodedAudioFormat::OggOpus;
    return EncodedAudioFormat::PcmF32Le;
}

class GenericHttpAudioStreamEncoder final : public HttpAudioStreamEncoder {
public:
    GenericHttpAudioStreamEncoder(RequestContext::AudioFormat format,
                                  int sample_rate,
                                  int ogg_prebuffer_chunks)
        : _format(to_encoded_format(format)) {
        AudioEncoderOptions options;
        options.format = _format;
        options.sample_rate = sample_rate;
        options.ogg_prebuffer_chunks = ogg_prebuffer_chunks;
        _encoder = std::make_unique<AudioEncoder>(options);
    }

    std::string content_type() const override {
        return AudioEncoder::content_type(_format);
    }

    void start(std::vector<char> & out) override {
        std::vector<uint8_t> bytes;
        std::string error;
        if (_encoder->start(bytes, error)) append(bytes, out);
    }

    void write(RequestContext::AudioChunk && chunk, bool is_terminal,
               std::vector<char> & out) override {
        std::vector<uint8_t> bytes;
        std::string error;
        if (_encoder->push(chunk.samples.data(), chunk.samples.size(),
                           is_terminal, bytes, error)) {
            append(bytes, out);
        }
    }

    void finish(bool success, std::vector<char> & out) override {
        std::vector<uint8_t> bytes;
        std::string error;
        if (_encoder->finish(success, bytes, error)) append(bytes, out);
    }

private:
    static void append(const std::vector<uint8_t> & bytes, std::vector<char> & out) {
        if (bytes.empty()) return;
        out.insert(out.end(), reinterpret_cast<const char *>(bytes.data()),
                   reinterpret_cast<const char *>(bytes.data() + bytes.size()));
    }

    EncodedAudioFormat _format;
    std::unique_ptr<AudioEncoder> _encoder;
};

} // namespace

bool http_audio_format_available(RequestContext::AudioFormat format) {
    return AudioEncoder::available(to_encoded_format(format));
}

std::unique_ptr<HttpAudioStreamEncoder> make_http_audio_stream_encoder(
    RequestContext::AudioFormat format,
    int sample_rate,
    int ogg_prebuffer_chunks) {
    if (!http_audio_format_available(format)) return nullptr;
    return std::make_unique<GenericHttpAudioStreamEncoder>(
        format, sample_rate, ogg_prebuffer_chunks);
}

} // namespace kokopop
