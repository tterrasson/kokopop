#include "http/http_server.h"
#include "yyjson.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <numeric>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #include <errno.h>
  #include <fcntl.h>
  #include <csignal>
  #include <signal.h>
#endif

namespace {

// ---------------------------------------------------------------------------
// Graceful shutdown callback (called from signal handler)
// ---------------------------------------------------------------------------

static std::function<void()> g_http_shutdown_callback;

// ---------------------------------------------------------------------------
// Limits / constants
// ---------------------------------------------------------------------------

/// Maximum allowed HTTP request body (16 MiB).
/// Prevents a malicious Content-Length from exhausting memory.
static constexpr size_t MAX_BODY_SIZE = 16 * 1024 * 1024;

/// Maximum HTTP header size (64 KiB).
static constexpr size_t MAX_HEADER_SIZE = 64 * 1024;

/// Maximum number of concurrent connections.
/// Prevents thread-explosion DoS.
static constexpr int MAX_CONCURRENT_CONNECTIONS = 64;

/// Connection idle timeout (seconds).  A connection that sits idle for this
/// long is closed.
static constexpr int CONNECTION_TIMEOUT_SECS = 30;

// ---------------------------------------------------------------------------
// Platform helpers
// ---------------------------------------------------------------------------

#ifdef _WIN32
void init_platform() {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
}
#else
void init_platform() {
    // Ignore SIGPIPE so that writing to a broken socket returns EPIPE instead
    // of killing the whole process.
    struct sigaction sa{};
    sa.sa_handler = SIG_IGN;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGPIPE, &sa, nullptr);

    // Handle SIGINT (Ctrl+C) and SIGTERM for graceful shutdown
    // Call the user-provided shutdown callback (e.g. server.stop())
    sa.sa_handler = +[](int) {
        if (g_http_shutdown_callback) {
            g_http_shutdown_callback();
        }
    };
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}
#endif

int socket_close(int fd) {
#ifdef _WIN32
    return closesocket(fd);
#else
    return ::close(fd);
#endif
}

// ---------------------------------------------------------------------------
// Blocking read / write with EINTR retry
//
// Both functions loop on EINTR so that signals (e.g. SIGCHLD) do not
// prematurely terminate the I/O.
// ---------------------------------------------------------------------------

ssize_t socket_read(int fd, void * buf, size_t len) {
#ifdef _WIN32
    return recv(fd, static_cast<char *>(buf), static_cast<int>(len), 0);
#else
    ssize_t n;
    do {
        n = ::read(fd, buf, len);
    } while (n < 0 && errno == EINTR);
    return n;
#endif
}

ssize_t socket_write(int fd, const void * buf, size_t len) {
#ifdef _WIN32
    return send(fd, static_cast<const char *>(buf), static_cast<int>(len), 0);
#else
    ssize_t n;
    do {
        n = ::write(fd, buf, len);
    } while (n < 0 && errno == EINTR);
    return n;
#endif
}

// ---------------------------------------------------------------------------
// Set SO_SNDTIMEO / SO_RCVTIMEO for idle detection
// ---------------------------------------------------------------------------

void set_socket_timeout(int fd, int seconds) {
#ifdef _WIN32
    DWORD ms = static_cast<DWORD>(seconds * 1000);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char *>(&ms), sizeof(ms));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char *>(&ms), sizeof(ms));
#else
    struct timeval tv{};
    tv.tv_sec = seconds;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

// ---------------------------------------------------------------------------
// write_all — blocking, handles EINTR internally
// ---------------------------------------------------------------------------

static bool write_all(int fd, const char * data, size_t len) {
    size_t offset = 0;
    while (offset < len) {
        ssize_t n = socket_write(fd, data + offset, len - offset);
        if (n <= 0) return false;
        offset += static_cast<size_t>(n);
    }
    return true;
}

static bool write_all(int fd, const std::vector<char> & data) {
    return data.empty() ? true : write_all(fd, data.data(), data.size());
}

static bool write_all(int fd, const std::string & s) {
    return write_all(fd, s.data(), s.size());
}

} // anonymous namespace

namespace kokopop {

// ---------------------------------------------------------------------------
// HttpResponse helpers
// ---------------------------------------------------------------------------

void HttpResponse::set_content_type(const std::string & type) {
    headers["Content-Type"] = type;
}

void HttpResponse::set_json(yyjson_doc * doc) {
    if (!doc) return;
    size_t len = 0;
    char * json = yyjson_write(doc, 0, &len);
    if (json && len > 0) {
        body.reserve(body.size() + len);
        body.insert(body.end(), json, json + len);
        free(json);
    }
    set_content_type("application/json");
}

void HttpResponse::set_json_string(const std::string & json_str) {
    body.insert(body.end(), json_str.begin(), json_str.end());
    set_content_type("application/json");
}

// Helper: build a simple JSON error object
std::string json_error(const std::string & msg) {
    // Escape the message for JSON (handle quotes and backslashes)
    std::string escaped;
    escaped.reserve(msg.size() + 2);
    for (char c : msg) {
        if (c == '"') escaped += "\\\"";
        else if (c == '\\') escaped += "\\\\";
        else if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else if (c == '\t') escaped += "\\t";
        else escaped += c;
    }
    return "{\"error\":\"" + escaped + "\"}";
}

// ---------------------------------------------------------------------------
// HttpServer implementation
// ---------------------------------------------------------------------------

HttpServer::HttpServer() {
    init_platform();
}

HttpServer::~HttpServer() {
    stop();
    join();
    if (_server_fd >= 0) {
        socket_close(_server_fd);
        _server_fd = -1;
    }
}

void HttpServer::route(const std::string & path, RequestHandler handler) {
    _routes[path] = std::move(handler);
}

void HttpServer::route_default(RequestHandler handler) {
    _default_handler = std::move(handler);
}

bool HttpServer::start(const std::string & addr, int port) {
    if (port < 1 || port > 65535) {
        std::fprintf(stderr, "[http] Invalid port: %d\n", port);
        return false;
    }

    _addr = addr;
    _port = port;

    _server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (_server_fd < 0) {
        std::fprintf(stderr, "[http] Failed to create socket: %s\n",
#ifdef _WIN32
                     "WSA error"
#else
                     strerror(errno)
#endif
        );
        return false;
    }

    // Allow address reuse
    int opt = 1;
#ifdef _WIN32
    setsockopt(_server_fd, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char *>(&opt), sizeof(opt));
    setsockopt(_server_fd, SOL_SOCKET, SO_KEEPALIVE,
               reinterpret_cast<const char *>(&opt), sizeof(opt));
#else
    setsockopt(_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(_server_fd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt));
#endif

    sockaddr_in addr_in{};
    addr_in.sin_family = AF_INET;
    addr_in.sin_port = htons(static_cast<uint16_t>(port));
    if (addr.empty() || addr == "*" || addr == "0.0.0.0") {
        addr_in.sin_addr.s_addr = INADDR_ANY;
    } else {
        int rv = inet_pton(AF_INET, addr.c_str(), &addr_in.sin_addr);
        if (rv <= 0) {
            std::fprintf(stderr, "[http] Invalid bind address: %s\n", addr.c_str());
            socket_close(_server_fd);
            _server_fd = -1;
            return false;
        }
    }

    if (bind(_server_fd, reinterpret_cast<sockaddr *>(&addr_in), sizeof(addr_in)) < 0) {
        std::fprintf(stderr, "[http] Failed to bind to %s:%d: %s\n",
                     addr.empty() ? "0.0.0.0" : addr.c_str(), port,
#ifdef _WIN32
                     "error"
#else
                     strerror(errno)
#endif
        );
        socket_close(_server_fd);
        _server_fd = -1;
        return false;
    }

    if (listen(_server_fd, 128) < 0) {
        std::fprintf(stderr, "[http] Failed to listen: %s\n",
#ifdef _WIN32
                     "error"
#else
                     strerror(errno)
#endif
        );
        socket_close(_server_fd);
        _server_fd = -1;
        return false;
    }

    _running.store(true);
    _accept_thread = std::thread(&HttpServer::_accept_loop, this);

    std::fprintf(stderr, "[http] Server listening on %s:%d (max %d connections)\n",
                addr.empty() ? "0.0.0.0" : addr.c_str(), port,
                MAX_CONCURRENT_CONNECTIONS);
    return true;
}

void HttpServer::stop() {
    _running.store(false);
    if (_server_fd >= 0) {
        socket_close(_server_fd);
        _server_fd = -1;
    }
}

void HttpServer::set_shutdown_callback(ShutdownCallback cb) {
    g_http_shutdown_callback = std::move(cb);
}

void HttpServer::join() {
    if (_accept_thread.joinable()) {
        _accept_thread.join();
    }
    {
        std::lock_guard<std::mutex> lock(_threads_mutex);
        for (auto & t : _conn_threads) {
            if (t.joinable()) t.join();
        }
        _conn_threads.clear();
    }
}

void HttpServer::_accept_loop() {
    while (_running.load()) {
        // Reject new connections if we're already at max concurrency.
        {
            std::lock_guard<std::mutex> lock(_threads_mutex);
            if (static_cast<int>(_conn_threads.size()) >= MAX_CONCURRENT_CONNECTIONS) {
                // Wait a bit before trying again — prevents busy-wait.
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
        }

        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(_server_fd, reinterpret_cast<sockaddr *>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (_running.load()) {
                std::fprintf(stderr, "[http] accept failed: %s\n",
#ifdef _WIN32
                             "WSA error"
#else
                             strerror(errno)
#endif
                );
                // If the error is transient (EINTR, EAGAIN/EWOULDBLOCK) keep trying.
                // Otherwise break.
#ifndef _WIN32
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
#endif
            }
            break;
        }

        // Disable Nagle's algorithm for lower latency
        int opt = 1;
#ifdef _WIN32
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY,
                   reinterpret_cast<const char *>(&opt), sizeof(opt));
#else
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#endif

        // Set recv timeout so idle connections eventually time out
        set_socket_timeout(client_fd, CONNECTION_TIMEOUT_SECS);

        // Spawn thread for this connection
        {
            std::lock_guard<std::mutex> lock(_threads_mutex);
            if (static_cast<int>(_conn_threads.size()) < MAX_CONCURRENT_CONNECTIONS) {
                _conn_threads.emplace_back(&HttpServer::_handle_connection, this, client_fd);
            } else {
                // Race: thread slots filled between check and lock — reject.
                std::fprintf(stderr, "[http] Rejecting connection (max %d reached)\n",
                             MAX_CONCURRENT_CONNECTIONS);
                socket_close(client_fd);
            }
        }
    }
}

void HttpServer::_handle_connection(int client_fd) {
    try {
        while (_running.load()) {
            HttpRequest req = _parse_request(client_fd);
            if (req.path.empty()) {
                // EOF, parse error, or timeout
                break;
            }

            HttpResponse res;
            res.headers["Server"] = "kokopop/0.1";
            res.headers["Connection"] = "keep-alive";

            // Find handler
            auto it = _routes.find(req.path);
            RequestHandler handler = (it != _routes.end()) ? it->second : _default_handler;

            if (!handler) {
                res.status_code = 404;
                res.status_text = "Not Found";
                std::string err_json = "{\"error\":\"not_found\"}";
                res.body.insert(res.body.end(), err_json.begin(), err_json.end());
                res.set_content_type("application/json");
            } else {
                bool keep_alive = handler(req, res);
                if (!keep_alive) {
                    res.headers["Connection"] = "close";
                    _send_response(client_fd, res, true);
                    break;
                }
            }

            if (!_send_response(client_fd, res, false)) {
                break;
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[http] Connection error: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "[http] Unknown connection error\n");
    }

    socket_close(client_fd);

    // Reap finished threads from the list
    {
        std::lock_guard<std::mutex> lock(_threads_mutex);
        _conn_threads.erase(
            std::remove_if(_conn_threads.begin(), _conn_threads.end(),
                [](std::thread & t) { return !t.joinable(); }),
            _conn_threads.end());
    }
}

HttpRequest HttpServer::_parse_request(int fd) {
    HttpRequest req;
    req.content_length = -1;

    // -------------------------------------------------------------------
    // 1. Read headers byte-by-byte until \r\n\r\n  (bounded)
    // -------------------------------------------------------------------
    std::vector<char> buf(MAX_HEADER_SIZE);
    std::string raw_header_data;
    size_t header_end = 0;

    while (header_end < buf.size()) {
        ssize_t n = socket_read(fd, buf.data() + header_end, 1);
        if (n <= 0) {
            // n == 0  →  peer closed
            // n < 0  →  error (timeout, ECONNRESET, etc.)
            return req;
        }

        // Check for end of headers (\r\n\r\n)
        if (header_end >= 3) {
            const char * p = buf.data() + static_cast<ptrdiff_t>(header_end) - 3;
            if (p[0] == '\r' && p[1] == '\n' && p[2] == '\r' && p[3] == '\n') {
                raw_header_data.assign(buf.data(), header_end - 3);
                break;
            }
        }
        ++header_end;
    }

    if (raw_header_data.empty()) {
        return req; // EOF or timeout
    }

    // -------------------------------------------------------------------
    // 2. Parse request line  —  "METHOD /path?query HTTP/x.x"
    // -------------------------------------------------------------------
    // Find first \r\n or \n to isolate the request line
    auto line_end = raw_header_data.find("\r\n");
    if (line_end == std::string::npos) {
        line_end = raw_header_data.find('\n');
    }
    if (line_end == std::string::npos || line_end == 0) {
        std::fprintf(stderr, "[http] Malformed request (no request line)\n");
        return req;
    }

    std::string request_line = raw_header_data.substr(0, line_end);

    // Split "METHOD /path HTTP/1.1"
    size_t space1 = request_line.find(' ');
    size_t space2 = request_line.rfind(' ');
    if (space1 == std::string::npos || space2 == std::string::npos || space1 == space2) {
        std::fprintf(stderr, "[http] Malformed request line: %.100s\n", request_line.c_str());
        return req;
    }

    req.method = request_line.substr(0, space1);
    std::string path_and_query = request_line.substr(space1 + 1, space2 - space1 - 1);

    // Split path and query string
    size_t qmark = path_and_query.find('?');
    if (qmark != std::string::npos) {
        req.path = path_and_query.substr(0, qmark);
        req.query_string = path_and_query.substr(qmark + 1);
    } else {
        req.path = path_and_query;
    }

    // -------------------------------------------------------------------
    // 3. Parse headers
    // -------------------------------------------------------------------
    std::string header_block = raw_header_data.substr(static_cast<size_t>(line_end) +
                                                      (raw_header_data[line_end] == '\r' ? 2 : 1));
    std::istringstream iss(header_block);
    std::string header_line;
    int header_count = 0;
    while (std::getline(iss, header_line)) {
        ++header_count;
        if (header_count > 100) {
            std::fprintf(stderr, "[http] Too many headers (%d), rejecting\n", header_count);
            return req;
        }

        // Trim trailing \r
        if (!header_line.empty() && header_line.back() == '\r') {
            header_line.pop_back();
        }
        if (header_line.empty()) continue;

        size_t colon = header_line.find(':');
        if (colon == std::string::npos || colon == 0) continue;

        std::string key = header_line.substr(0, colon);
        std::string val = header_line.substr(colon + 1);
        // Trim leading whitespace from value
        while (!val.empty() && (val.front() == ' ' || val.front() == '\t'))
            val.erase(val.begin());

        // Normalize key to lowercase
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        req.headers[key] = val;
    }

    // -------------------------------------------------------------------
    // 4. Read body (if Content-Length is present) — bounded
    // -------------------------------------------------------------------
    auto it = req.headers.find("content-length");
    if (it != req.headers.end()) {
        try {
            // Use stoll so we can detect values that don't fit in int.
            long long cl = std::stoll(it->second);
            if (cl < 0 || static_cast<size_t>(cl) > MAX_BODY_SIZE) {
                std::fprintf(stderr, "[http] Content-Length %lld exceeds limit (%zu), rejecting\n",
                             cl, MAX_BODY_SIZE);
                return req;
            }
            req.content_length = static_cast<int>(cl);
            req.body.resize(static_cast<size_t>(req.content_length));
        } catch (const std::exception &) {
            std::fprintf(stderr, "[http] Invalid Content-Length: %s\n", it->second.c_str());
            return req;
        }

        // Read exactly req.content_length bytes
        size_t offset = 0;
        while (offset < req.body.size()) {
            ssize_t n = socket_read(fd, req.body.data() + offset,
                                    req.body.size() - offset);
            if (n <= 0) break; // timeout or disconnect
            offset += static_cast<size_t>(n);
        }
        // Resize in case we got fewer bytes than expected (incomplete read)
        req.body.resize(offset);
    }

    return req;
}

bool HttpServer::_send_response(int fd, const HttpResponse & res, bool close_conn) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << res.status_code << " " << res.status_text << "\r\n";

    for (auto & [key, val] : res.headers) {
        oss << key << ": " << val << "\r\n";
    }

    // If Content-Length header wasn't explicitly set, add it
    if (res.headers.find("content-length") == res.headers.end()) {
        oss << "Content-Length: " << res.body.size() << "\r\n";
    }

    // Only add Connection: close if not already present
    // Only add Connection: close if not already set (check case-insensitive)
    bool has_connection_header = false;
    for (auto & [k, v] : res.headers) {
        if (k == "connection" || k == "Connection" || k == "CONNECTION") {
            has_connection_header = true;
            break;
        }
    }
    if (close_conn && !has_connection_header) {
        oss << "Connection: close\r\n";
    }

    oss << "\r\n";
    std::string headers = oss.str();

    if (!write_all(fd, headers)) return false;
    if (!res.body.empty() && !write_all(fd, res.body)) return false;

    if (close_conn) {
#ifdef _WIN32
        shutdown(fd, SD_BOTH);
#else
        shutdown(fd, SHUT_RDWR);
#endif
    }
    return true;
}

} // namespace kokopop
