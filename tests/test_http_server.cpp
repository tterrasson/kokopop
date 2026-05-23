#include "test_helpers.h"
#include "http/http_server.h"

#include <chrono>
#include <thread>
#include <atomic>

#ifndef _WIN32
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#endif

namespace {

// Probe a free TCP port by binding to port 0 and reading the assigned port.
// Small race window between probe and actual server bind — acceptable for tests.
int find_free_port() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    REQUIRE(bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0);
    socklen_t len = sizeof(addr);
    REQUIRE(getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len) == 0);
    int port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
}

// Connect to 127.0.0.1:port, send raw request, return raw response (best-effort,
// reads until the peer closes or until ~1s of inactivity).
std::string http_roundtrip(int port, const std::string & request) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port));

    // Retry connect for up to ~1s while the server thread is coming up.
    bool connected = false;
    for (int i = 0; i < 50 && !connected; ++i) {
        if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
            connected = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    REQUIRE(connected);

    ssize_t sent = ::send(fd, request.data(), request.size(), 0);
    REQUIRE(sent == static_cast<ssize_t>(request.size()));

    // Read until close or until no data for a short timeout.
    timeval tv{};
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    std::string response;
    char buf[4096];
    while (true) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        response.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);
    return response;
}

} // namespace

TEST_CASE("http_server_routes_and_lifecycle") {
    using namespace kokopop;

    HttpServer server;
    std::atomic<int> hits{0};
    server.route("/ping", [&](HttpRequest & req, HttpResponse & res) {
        ++hits;
        CHECK_EQ(req.method, "GET");
        res.status_code = 200;
        res.set_content_type("text/plain");
        const char body[] = "pong";
        res.body.assign(body, body + sizeof(body) - 1);
        return false; // close connection
    });
    server.route_default([](HttpRequest &, HttpResponse & res) {
        res.status_code = 404;
        res.status_text = "Not Found";
        res.set_content_type("application/json");
        const std::string body = json_error("unknown route");
        res.body.assign(body.begin(), body.end());
        return false;
    });

    int port = find_free_port();
    REQUIRE(server.start("127.0.0.1", port));
    CHECK(server.is_running());

    SUBCASE("known route returns handler body") {
        const std::string req =
            "GET /ping HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
        std::string resp = http_roundtrip(port, req);
        CHECK(resp.find("HTTP/1.1 200") != std::string::npos);
        CHECK(resp.find("pong") != std::string::npos);
        CHECK_EQ(hits.load(), 1);
    }

    SUBCASE("unknown route uses default handler") {
        const std::string req =
            "GET /nope HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
        std::string resp = http_roundtrip(port, req);
        CHECK(resp.find("HTTP/1.1 404") != std::string::npos);
        CHECK(resp.find("unknown route") != std::string::npos);
        CHECK_EQ(hits.load(), 0);
    }

    server.stop();
    server.join();
    CHECK(!server.is_running());
}

TEST_CASE("http_server_invalid_port") {
    kokopop::HttpServer server;
    CHECK(!server.start("127.0.0.1", 0));
    CHECK(!server.start("127.0.0.1", 70000));
    CHECK(!server.is_running());
}
