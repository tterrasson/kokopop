#pragma once

#include "http/http_server.h"

#include <cstddef>
#include <string>

namespace kokopop {

struct AsyncHttpParseResult {
    enum class Status {
        NeedMore,
        Complete,
        Error
    };

    Status status = Status::NeedMore;
    HttpRequest request;
    bool keep_alive = false;
    int status_code = 400;
    std::string status_text = "Bad Request";
    std::string error_message;
};

class AsyncHttpRequestParser {
public:
    AsyncHttpRequestParser(size_t max_header_size, size_t max_body_size);

    void append(const char * data, size_t size);
    AsyncHttpParseResult next();
    void reset_current();
    bool empty() const { return _buffer.empty(); }

private:
    enum class State {
        ReadingHeaders,
        ReadingFixedBody,
        ReadingChunkSize,
        ReadingChunkData,
        ReadingChunkTrailer
    };

    AsyncHttpParseResult _parse_headers(size_t header_end);
    AsyncHttpParseResult _parse_fixed_body();
    AsyncHttpParseResult _parse_chunk_size();
    AsyncHttpParseResult _parse_chunk_data();
    AsyncHttpParseResult _parse_chunk_trailer();
    AsyncHttpParseResult _complete();
    AsyncHttpParseResult _error(int code, const std::string & text,
                                const std::string & message);

    static bool _parse_request_line(const std::string & request_line,
                                    HttpRequest & req,
                                    std::string & version);
    static std::string _lower_ascii(std::string s);
    static std::string _trim_ascii(std::string s);
    static bool _header_has_token(const std::string & value,
                                  const std::string & token);

    size_t _max_header_size = 0;
    size_t _max_body_size = 0;
    State _state = State::ReadingHeaders;
    std::string _buffer;
    HttpRequest _request;
    bool _keep_alive = false;
    size_t _content_length = 0;
    size_t _chunk_size = 0;
};

} // namespace kokopop
