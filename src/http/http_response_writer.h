#pragma once

#include "http/http_server.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace kokopop {

/// Lowercase an ASCII string.
inline std::string http_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

/// Trim HTTP header-value whitespace (space, tab, CR) from both ends.
inline std::string http_trim_ascii(std::string s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r')) {
        s.erase(s.begin());
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

bool http_headers_contain(const std::map<std::string, std::string> & headers,
                          const std::string & name);
bool http_header_value_has_token(const std::string & value,
                                 const std::string & token);
std::string build_http_response_head(const HttpResponse & res, bool keep_alive);
std::string build_streaming_response_head(const std::string & content_type);
void append_http_chunk(std::string & out, const std::vector<char> & data);
void append_final_http_chunk(std::string & out);

} // namespace kokopop
