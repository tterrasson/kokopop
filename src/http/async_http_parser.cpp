#include "http/async_http_parser.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace kokopop {

namespace {
constexpr int MAX_HEADER_COUNT = 100;

bool is_http_token_char(unsigned char c) {
    return c > 32 && c < 127 && c != ':';
}
} // namespace

AsyncHttpRequestParser::AsyncHttpRequestParser(size_t max_header_size,
                                               size_t max_body_size)
    : _max_header_size(max_header_size),
      _max_body_size(max_body_size) {
    reset_current();
}

void AsyncHttpRequestParser::append(const char * data, size_t size) {
    _buffer.append(data, size);
}

void AsyncHttpRequestParser::reset_current() {
    _state = State::ReadingHeaders;
    _request = HttpRequest{};
    _request.content_length = -1;
    _keep_alive = false;
    _content_length = 0;
    _chunk_size = 0;
}

AsyncHttpParseResult AsyncHttpRequestParser::next() {
    while (true) {
        if (_state == State::ReadingHeaders) {
            const size_t header_end = _buffer.find("\r\n\r\n");
            if (header_end == std::string::npos) {
                if (_buffer.size() > _max_header_size) {
                    return _error(431, "Request Header Fields Too Large",
                                  "Headers too large");
                }
                return {};
            }
            auto result = _parse_headers(header_end);
            if (result.status != AsyncHttpParseResult::Status::NeedMore) {
                return result;
            }
        } else if (_state == State::ReadingFixedBody) {
            auto result = _parse_fixed_body();
            if (result.status != AsyncHttpParseResult::Status::NeedMore) {
                return result;
            }
            return result;
        } else if (_state == State::ReadingChunkSize) {
            auto result = _parse_chunk_size();
            if (result.status != AsyncHttpParseResult::Status::NeedMore) {
                return result;
            }
            return result;
        } else if (_state == State::ReadingChunkData) {
            auto result = _parse_chunk_data();
            if (result.status != AsyncHttpParseResult::Status::NeedMore) {
                return result;
            }
            return result;
        } else if (_state == State::ReadingChunkTrailer) {
            auto result = _parse_chunk_trailer();
            if (result.status != AsyncHttpParseResult::Status::NeedMore) {
                return result;
            }
            return result;
        }
    }
}

AsyncHttpParseResult AsyncHttpRequestParser::_parse_headers(size_t header_end) {
    std::string header_data = _buffer.substr(0, header_end);
    _buffer.erase(0, header_end + 4);

    size_t line_end = header_data.find("\r\n");
    if (header_data.empty() || line_end == 0) {
        return _error(400, "Bad Request", "Malformed request");
    }
    if (line_end == std::string::npos) {
        line_end = header_data.size();
    }

    std::string version;
    if (!_parse_request_line(header_data.substr(0, line_end), _request, version)) {
        return _error(400, "Bad Request", "Malformed request line");
    }

    std::string header_block;
    if (line_end < header_data.size()) {
        header_block = header_data.substr(line_end + 2);
    }
    std::istringstream iss(header_block);
    std::string line;
    int header_count = 0;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        ++header_count;
        if (header_count > MAX_HEADER_COUNT) {
            return _error(431, "Request Header Fields Too Large",
                          "Too many headers");
        }

        const size_t colon = line.find(':');
        if (colon == std::string::npos || colon == 0) {
            return _error(400, "Bad Request", "Malformed header");
        }

        std::string key = _lower_ascii(line.substr(0, colon));
        for (unsigned char c : key) {
            if (!is_http_token_char(c)) {
                return _error(400, "Bad Request", "Malformed header name");
            }
        }
        _request.headers[key] = _trim_ascii(line.substr(colon + 1));
    }

    const auto conn_it = _request.headers.find("connection");
    const bool close_requested =
        conn_it != _request.headers.end() &&
        _header_has_token(conn_it->second, "close");
    const bool keep_alive_requested =
        conn_it != _request.headers.end() &&
        _header_has_token(conn_it->second, "keep-alive");
    _keep_alive = (version == "HTTP/1.1" && !close_requested) ||
                  (version == "HTTP/1.0" && keep_alive_requested);

    const auto cl_it = _request.headers.find("content-length");
    const auto te_it = _request.headers.find("transfer-encoding");
    if (cl_it != _request.headers.end() && te_it != _request.headers.end()) {
        return _error(400, "Bad Request",
                      "Content-Length and Transfer-Encoding are mutually exclusive");
    }

    if (te_it != _request.headers.end()) {
        if (!_header_has_token(te_it->second, "chunked")) {
            return _error(501, "Not Implemented",
                          "Unsupported Transfer-Encoding");
        }
        std::string te = _trim_ascii(_lower_ascii(te_it->second));
        if (te != "chunked") {
            return _error(501, "Not Implemented",
                          "Unsupported Transfer-Encoding");
        }
        _state = State::ReadingChunkSize;
        return {};
    }

    if (cl_it != _request.headers.end()) {
        char * end = nullptr;
        const std::string value = _trim_ascii(cl_it->second);
        if (value.empty() || value[0] == '-') {
            return _error(400, "Bad Request", "Invalid Content-Length");
        }
        unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
        if (!end || end == value.c_str() || *end != '\0') {
            return _error(400, "Bad Request", "Invalid Content-Length");
        }
        if (parsed > static_cast<unsigned long long>(_max_body_size)) {
            return _error(413, "Payload Too Large", "Body too large");
        }
        _content_length = static_cast<size_t>(parsed);
        _request.content_length = static_cast<int>(_content_length);
        if (_content_length == 0) {
            return _complete();
        }
        _state = State::ReadingFixedBody;
        return _parse_fixed_body();
    }

    return _complete();
}

AsyncHttpParseResult AsyncHttpRequestParser::_parse_fixed_body() {
    if (_buffer.size() < _content_length) return {};
    _request.body.assign(_buffer.begin(),
                         _buffer.begin() + static_cast<ptrdiff_t>(_content_length));
    _buffer.erase(0, _content_length);
    return _complete();
}

AsyncHttpParseResult AsyncHttpRequestParser::_parse_chunk_size() {
    const size_t line_end = _buffer.find("\r\n");
    if (line_end == std::string::npos) {
        if (_buffer.size() > _max_header_size) {
            return _error(400, "Bad Request", "Malformed chunk size");
        }
        return {};
    }

    std::string line = _buffer.substr(0, line_end);
    _buffer.erase(0, line_end + 2);
    const size_t semicolon = line.find(';');
    if (semicolon != std::string::npos) line.resize(semicolon);
    line = _trim_ascii(line);
    if (line.empty()) {
        return _error(400, "Bad Request", "Malformed chunk size");
    }

    size_t value = 0;
    for (unsigned char c : line) {
        int digit = -1;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = 10 + c - 'a';
        else if (c >= 'A' && c <= 'F') digit = 10 + c - 'A';
        else return _error(400, "Bad Request", "Malformed chunk size");

        if (value > (std::numeric_limits<size_t>::max() - static_cast<size_t>(digit)) / 16) {
            return _error(413, "Payload Too Large", "Body too large");
        }
        value = value * 16 + static_cast<size_t>(digit);
    }

    _chunk_size = value;
    if (_chunk_size == 0) {
        _state = State::ReadingChunkTrailer;
        return _parse_chunk_trailer();
    }
    if (_request.body.size() >= _max_body_size ||
        _chunk_size > _max_body_size - _request.body.size()) {
        return _error(413, "Payload Too Large", "Body too large");
    }
    _state = State::ReadingChunkData;
    return _parse_chunk_data();
}

AsyncHttpParseResult AsyncHttpRequestParser::_parse_chunk_data() {
    if (_buffer.size() < 2 || _chunk_size > _buffer.size() - 2) return {};
    if (_buffer[_chunk_size] != '\r' || _buffer[_chunk_size + 1] != '\n') {
        return _error(400, "Bad Request", "Malformed chunk data");
    }
    _request.body.insert(_request.body.end(), _buffer.begin(),
                         _buffer.begin() + static_cast<ptrdiff_t>(_chunk_size));
    _buffer.erase(0, _chunk_size + 2);
    _state = State::ReadingChunkSize;
    return _parse_chunk_size();
}

AsyncHttpParseResult AsyncHttpRequestParser::_parse_chunk_trailer() {
    const size_t trailer_end = _buffer.find("\r\n");
    if (trailer_end == std::string::npos) return {};
    if (trailer_end == 0) {
        _buffer.erase(0, 2);
        _request.content_length = static_cast<int>(_request.body.size());
        return _complete();
    }
    const size_t all_trailers_end = _buffer.find("\r\n\r\n");
    if (all_trailers_end == std::string::npos) {
        if (_buffer.size() > _max_header_size) {
            return _error(431, "Request Header Fields Too Large",
                          "Trailers too large");
        }
        return {};
    }
    _buffer.erase(0, all_trailers_end + 4);
    _request.content_length = static_cast<int>(_request.body.size());
    return _complete();
}

AsyncHttpParseResult AsyncHttpRequestParser::_complete() {
    AsyncHttpParseResult result;
    result.status = AsyncHttpParseResult::Status::Complete;
    result.request = std::move(_request);
    result.keep_alive = _keep_alive;
    reset_current();
    return result;
}

AsyncHttpParseResult AsyncHttpRequestParser::_error(int code, const std::string & text,
                                                    const std::string & message) {
    AsyncHttpParseResult result;
    result.status = AsyncHttpParseResult::Status::Error;
    result.status_code = code;
    result.status_text = text;
    result.error_message = message;
    reset_current();
    return result;
}

bool AsyncHttpRequestParser::_parse_request_line(const std::string & request_line,
                                                 HttpRequest & req,
                                                 std::string & version) {
    const size_t space1 = request_line.find(' ');
    const size_t space2 = request_line.rfind(' ');
    if (space1 == std::string::npos || space2 == std::string::npos || space1 == space2) {
        return false;
    }

    req.method = request_line.substr(0, space1);
    std::string path_and_query = request_line.substr(space1 + 1, space2 - space1 - 1);
    version = request_line.substr(space2 + 1);
    if (req.method.empty() || path_and_query.empty() ||
        (version != "HTTP/1.0" && version != "HTTP/1.1")) {
        return false;
    }

    const size_t qmark = path_and_query.find('?');
    if (qmark != std::string::npos) {
        req.path = path_and_query.substr(0, qmark);
        req.query_string = path_and_query.substr(qmark + 1);
    } else {
        req.path = path_and_query;
        req.query_string.clear();
    }
    return !req.path.empty() && req.path[0] == '/';
}

std::string AsyncHttpRequestParser::_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string AsyncHttpRequestParser::_trim_ascii(std::string s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
        s.erase(s.begin());
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

bool AsyncHttpRequestParser::_header_has_token(const std::string & value,
                                               const std::string & token) {
    std::string lower = _lower_ascii(value);
    size_t start = 0;
    while (start <= lower.size()) {
        size_t comma = lower.find(',', start);
        std::string part = _trim_ascii(lower.substr(start, comma - start));
        if (part == token) return true;
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return false;
}

} // namespace kokopop
