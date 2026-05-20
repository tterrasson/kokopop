#pragma once

#include "http/http_server.h"

#include <string>
#include <vector>

namespace kokopop {

bool http_headers_contain(const std::map<std::string, std::string> & headers,
                          const std::string & name);
bool http_header_value_has_token(const std::string & value,
                                 const std::string & token);
std::string build_http_response_head(const HttpResponse & res, bool keep_alive);
std::string build_streaming_response_head(const std::string & content_type);
void append_http_chunk(std::string & out, const std::vector<char> & data);
void append_final_http_chunk(std::string & out);

} // namespace kokopop
