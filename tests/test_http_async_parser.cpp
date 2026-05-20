#include "http/async_http_parser.h"
#include "http/http_audio_stream_encoder.h"

#include <string>

namespace {

kokopop::AsyncHttpRequestParser make_parser() {
    return kokopop::AsyncHttpRequestParser(1024, 64);
}

kokopop::AsyncHttpParseResult feed(kokopop::AsyncHttpRequestParser & parser,
                                   const std::string & data) {
    parser.append(data.data(), data.size());
    return parser.next();
}

std::string body_string(const kokopop::HttpRequest & req) {
    return std::string(req.body.begin(), req.body.end());
}

} // namespace

TEST_CASE("async_http_parser fixed body split across reads") {
    auto parser = make_parser();
    auto result = feed(parser,
        "POST /tts HTTP/1.1\r\nHost: example\r\nContent-Length: 11\r\n\r\nhello");
    CHECK(result.status == kokopop::AsyncHttpParseResult::Status::NeedMore);

    result = feed(parser, " world");
    REQUIRE(result.status == kokopop::AsyncHttpParseResult::Status::Complete);
    CHECK(result.request.method == "POST");
    CHECK(result.request.path == "/tts");
    CHECK(result.keep_alive);
    CHECK(body_string(result.request) == "hello world");
}

TEST_CASE("async_http_parser accepts request line without headers") {
    auto parser = make_parser();
    auto result = feed(parser, "GET /health HTTP/1.0\r\n\r\n");

    REQUIRE(result.status == kokopop::AsyncHttpParseResult::Status::Complete);
    CHECK(result.request.method == "GET");
    CHECK(result.request.path == "/health");
    CHECK_FALSE(result.keep_alive);
}

TEST_CASE("async_http_parser chunked body one and many chunks") {
    auto parser = make_parser();
    auto result = feed(parser,
        "POST /tts HTTP/1.1\r\nHost: example\r\nTransfer-Encoding: chunked\r\n\r\n"
        "5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n");

    REQUIRE(result.status == kokopop::AsyncHttpParseResult::Status::Complete);
    CHECK(result.keep_alive);
    CHECK(result.request.content_length == 11);
    CHECK(body_string(result.request) == "hello world");
}

TEST_CASE("async_http_parser chunked body split across reads") {
    auto parser = make_parser();
    auto result = feed(parser,
        "POST /tts HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
        "5\r\nhe");
    CHECK(result.status == kokopop::AsyncHttpParseResult::Status::NeedMore);

    result = feed(parser, "llo\r\n0\r\n\r\n");
    REQUIRE(result.status == kokopop::AsyncHttpParseResult::Status::Complete);
    CHECK(body_string(result.request) == "hello");
}

TEST_CASE("async_http_parser rejects malformed chunk size") {
    auto parser = make_parser();
    auto result = feed(parser,
        "POST /tts HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
        "nope\r\nhello\r\n0\r\n\r\n");

    REQUIRE(result.status == kokopop::AsyncHttpParseResult::Status::Error);
    CHECK(result.status_code == 400);
}

TEST_CASE("async_http_parser rejects missing chunk terminator") {
    auto parser = make_parser();
    auto result = feed(parser,
        "POST /tts HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
        "5\r\nhelloXX");

    REQUIRE(result.status == kokopop::AsyncHttpParseResult::Status::Error);
    CHECK(result.status_code == 400);
}

TEST_CASE("async_http_parser rejects oversized headers and bodies") {
    auto parser = make_parser();
    std::string req = "GET / HTTP/1.1\r\n";
    for (int i = 0; i < 110; ++i) {
        req += "X-Test-" + std::to_string(i) + ": value\r\n";
    }
    req += "\r\n";
    auto result = feed(parser, req);
    REQUIRE(result.status == kokopop::AsyncHttpParseResult::Status::Error);
    CHECK(result.status_code == 431);

    parser = make_parser();
    result = feed(parser,
        "POST /tts HTTP/1.1\r\nContent-Length: 65\r\n\r\n");
    REQUIRE(result.status == kokopop::AsyncHttpParseResult::Status::Error);
    CHECK(result.status_code == 413);

    parser = make_parser();
    result = feed(parser,
        "POST /tts HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
        "41\r\n");
    REQUIRE(result.status == kokopop::AsyncHttpParseResult::Status::Error);
    CHECK(result.status_code == 413);
}

TEST_CASE("async_http_parser rejects content length with transfer encoding") {
    auto parser = make_parser();
    auto result = feed(parser,
        "POST /tts HTTP/1.1\r\nContent-Length: 5\r\n"
        "Transfer-Encoding: chunked\r\n\r\n");

    REQUIRE(result.status == kokopop::AsyncHttpParseResult::Status::Error);
    CHECK(result.status_code == 400);
}

TEST_CASE("async_http_parser rejects invalid content length as bad request") {
    auto parser = make_parser();
    auto result = feed(parser,
        "POST /tts HTTP/1.1\r\nContent-Length: nope\r\n\r\n");

    REQUIRE(result.status == kokopop::AsyncHttpParseResult::Status::Error);
    CHECK(result.status_code == 400);
}

TEST_CASE("async_http_parser keep alive and pipelined requests") {
    auto parser = make_parser();
    auto result = feed(parser,
        "GET /health HTTP/1.1\r\nHost: example\r\n\r\n"
        "GET /voices HTTP/1.1\r\nHost: example\r\nConnection: close\r\n\r\n");

    REQUIRE(result.status == kokopop::AsyncHttpParseResult::Status::Complete);
    CHECK(result.request.path == "/health");
    CHECK(result.keep_alive);

    result = parser.next();
    REQUIRE(result.status == kokopop::AsyncHttpParseResult::Status::Complete);
    CHECK(result.request.path == "/voices");
    CHECK_FALSE(result.keep_alive);
}

TEST_CASE("http_audio_stream_encoder reports opus capability consistently") {
#ifdef KOKOPOP_HAS_OPUS
    CHECK(kokopop::http_audio_format_available(
        kokopop::RequestContext::AudioFormat::OGG_OPUS));
#else
    CHECK_FALSE(kokopop::http_audio_format_available(
        kokopop::RequestContext::AudioFormat::OGG_OPUS));
#endif
    CHECK(kokopop::http_audio_format_available(
        kokopop::RequestContext::AudioFormat::PCM));
    CHECK(kokopop::http_audio_format_available(
        kokopop::RequestContext::AudioFormat::WAV));
}
