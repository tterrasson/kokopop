#include "http/http_response_writer.h"

#include <cassert>
#include <sstream>

namespace kokopop {

bool http_headers_contain(const std::map<std::string, std::string> & headers,
                          const std::string & name) {
    const std::string target = http_lower_ascii(name);
    for (const auto & kv : headers) {
        if (http_lower_ascii(kv.first) == target) return true;
    }
    return false;
}

bool http_header_value_has_token(const std::string & value,
                                 const std::string & token) {
    const std::string target = http_lower_ascii(token);
    const std::string lower = http_lower_ascii(value);
    size_t start = 0;
    while (start <= lower.size()) {
        const size_t comma = lower.find(',', start);
        const std::string part = http_trim_ascii(lower.substr(start, comma - start));
        if (part == target) return true;
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return false;
}

std::string build_http_response_head(const HttpResponse & res, bool keep_alive) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << res.status_code << " " << res.status_text << "\r\n";
    for (const auto & kv : res.headers) {
        oss << kv.first << ": " << kv.second << "\r\n";
    }
    if (!http_headers_contain(res.headers, "Content-Length")) {
        oss << "Content-Length: " << res.body.size() << "\r\n";
    }
    if (!http_headers_contain(res.headers, "Server")) {
        oss << "Server: kokopop-async/0.1\r\n";
    }
    if (!http_headers_contain(res.headers, "Connection")) {
        oss << "Connection: " << (keep_alive ? "keep-alive" : "close") << "\r\n";
    }
    oss << "\r\n";
    return oss.str();
}

std::string build_streaming_response_head(const std::string & content_type) {
    std::ostringstream oss;
    oss << "HTTP/1.1 200 OK\r\n";
    oss << "Transfer-Encoding: chunked\r\n";
    oss << "Content-Type: " << content_type << "\r\n";
    oss << "Server: kokopop-async/0.1\r\n";
    oss << "Connection: close\r\n";
    oss << "\r\n";
    return oss.str();
}

void append_http_chunk(std::string & out, const std::vector<char> & data) {
    assert(!data.empty() && "append_http_chunk: empty data produces a premature terminal chunk");
    std::ostringstream hdr;
    hdr << std::hex << data.size() << "\r\n";
    out.append(hdr.str());
    out.insert(out.end(), data.begin(), data.end());
    out.append("\r\n");
}

void append_final_http_chunk(std::string & out) {
    out.append("0\r\n\r\n");
}

} // namespace kokopop
