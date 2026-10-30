#pragma once

#include "http/request_context.h"

#include <memory>
#include <string>
#include <vector>

namespace kokopop {

class HttpAudioStreamEncoder {
public:
    virtual ~HttpAudioStreamEncoder() = default;
    virtual std::string content_type() const = 0;
    // All three return false on encoder failure; the caller must then abort
    // the stream (a truncated chunked response signals the error to clients).
    virtual bool start(std::vector<char> & out) = 0;
    virtual bool write(RequestContext::AudioChunk && chunk, bool is_terminal,
                       std::vector<char> & out) = 0;
    virtual bool finish(bool success, std::vector<char> & out) = 0;
};

bool http_audio_format_available(RequestContext::AudioFormat format);
std::unique_ptr<HttpAudioStreamEncoder> make_http_audio_stream_encoder(
    RequestContext::AudioFormat format,
    int sample_rate,
    int ogg_prebuffer_chunks);

} // namespace kokopop
