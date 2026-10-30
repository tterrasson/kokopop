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
  #include <poll.h>
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

/// Maximum number of header lines per request.
static constexpr int MAX_HEADER_COUNT = 100;

/// Maximum number of concurrent connections.
/// Prevents thread-explosion DoS.
static constexpr int MAX_CONCURRENT_CONNECTIONS = 64;

/// Connection idle timeout (seconds).  A connection that sits idle for this
/// long is closed.
static constexpr int CONNECTION_TIMEOUT_SECS = 30;

/// How long the accept loop waits for a connection before re-reading the
/// running flag. It bounds shutdown latency on an idle server and costs one
/// `poll()` every tick; it is not a connection timeout.
static constexpr int ACCEPT_POLL_MS = 100;

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

/// Waits until `fd` has a connection to accept, for at most `timeout_ms`.
/// Returns 1 when it does, 0 on timeout, -1 on error.
///
/// This is what makes the accept loop interruptible. Closing the listening
/// socket from another thread does *not* wake a thread already blocked in
/// `accept()` on Linux — the descriptor leaves the table but the syscall stays
/// parked — so a server shut down that way never joins. Waiting with a timeout
/// and only then accepting means `stop()` has to flip one flag and nothing
/// else, which is also the only thing it may safely do from a signal handler.
int wait_for_connection(int fd, int timeout_ms) {
#ifdef _WIN32
    WSAPOLLFD pfd{};
    pfd.fd = static_cast<SOCKET>(fd);
    pfd.events = POLLRDNORM;
    const int rv = WSAPoll(&pfd, 1, timeout_ms);
#else
    struct pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLIN;
    int rv;
    do {
        rv = ::poll(&pfd, 1, timeout_ms);
    } while (rv < 0 && errno == EINTR);
#endif
    return rv > 0 ? 1 : rv;
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
    // Escape the message for JSON. All C0 control characters must be escaped
    // (\u00XX), not just the named ones — anything else produces invalid JSON.
    std::string escaped;
    escaped.reserve(msg.size() + 2);
    char buf[8];
    for (unsigned char c : msg) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            default:
                if (c < 0x20) {
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    escaped += buf;
                } else {
                    escaped += static_cast<char>(c);
                }
        }
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
    // Only the flag: the accept loop polls it, and closing the listening
    // socket here would neither wake that loop (see `wait_for_connection`) nor
    // be safe to do from the signal handler this is called from.
    _running.store(false);
}

void HttpServer::set_shutdown_callback(ShutdownCallback cb) {
    g_http_shutdown_callback = std::move(cb);
}

void HttpServer::join() {
    if (_accept_thread.joinable()) {
        _accept_thread.join();
    }
    // After the accept thread has exited, nothing reads `_server_fd` any more,
    // so this is the one point where closing it races with nobody. A connection
    // thread can still take up to CONNECTION_TIMEOUT_SECS to notice the flag on
    // an idle keep-alive socket; it holds its own descriptor, not this one.
    if (_server_fd >= 0) {
        socket_close(_server_fd);
        _server_fd = -1;
    }
    {
        std::lock_guard<std::mutex> lock(_threads_mutex);
        for (auto & ct : _conn_threads) {
            if (ct.thread.joinable()) ct.thread.join();
        }
        _conn_threads.clear();
    }
}

void HttpServer::_reap_finished_threads() {
    // Caller must hold _threads_mutex. A finished thread has already closed
    // its socket and no longer touches the mutex, so joining here is safe.
    for (auto it = _conn_threads.begin(); it != _conn_threads.end();) {
        if (it->done && it->done->load() && it->thread.joinable()) {
            it->thread.join();
            it = _conn_threads.erase(it);
        } else {
            ++it;
        }
    }
}

void HttpServer::_accept_loop() {
    while (_running.load()) {
        // Reject new connections if we're already at max concurrency.
        {
            std::lock_guard<std::mutex> lock(_threads_mutex);
            _reap_finished_threads();
            if (static_cast<int>(_conn_threads.size()) >= MAX_CONCURRENT_CONNECTIONS) {
                // Wait a bit before trying again — prevents busy-wait.
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
        }

        // Bounded wait, so `stop()` is noticed within one tick even with no
        // traffic at all. An error here is the listening socket itself going
        // bad, which the accept below reports with a message.
        const int ready = wait_for_connection(_server_fd, ACCEPT_POLL_MS);
        if (ready == 0) {
            continue;
        }
        if (ready < 0 && !_running.load()) {
            break;
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
            _reap_finished_threads();
            if (static_cast<int>(_conn_threads.size()) < MAX_CONCURRENT_CONNECTIONS) {
                auto done = std::make_shared<std::atomic<bool>>(false);
                _conn_threads.push_back(ConnThread{
                    std::thread(&HttpServer::_handle_connection, this, client_fd, done),
                    std::move(done)});
            } else {
                // Race: thread slots filled between check and lock — reject.
                std::fprintf(stderr, "[http] Rejecting connection (max %d reached)\n",
                             MAX_CONCURRENT_CONNECTIONS);
                socket_close(client_fd);
            }
        }
    }
}

void HttpServer::_handle_connection(int client_fd, std::shared_ptr<std::atomic<bool>> done) {
    try {
        std::string pending;
        while (_running.load()) {
            HttpRequest req = _parse_request(client_fd, pending);
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

    // Mark finished; the accept loop joins and erases this entry (a thread
    // cannot join itself).
    if (done) done->store(true);
}

HttpRequest HttpServer::_parse_request(int fd, std::string & pending) {
    HttpRequest req;
    req.content_length = -1;

    // -------------------------------------------------------------------
    // 1. Read headers until \r\n\r\n  (bounded, in blocks — one read() per
    //    byte would cost up to 64 KiB of syscalls per request)
    // -------------------------------------------------------------------
    std::string raw_header_data = std::move(pending);
    pending.clear();
    bool found_eoh = false;
    {
        char chunk[4096];
        while (true) {
            const size_t eoh = raw_header_data.find("\r\n\r\n");
            if (eoh != std::string::npos) {
                if (eoh + 4 > MAX_HEADER_SIZE) return {};
                pending = raw_header_data.substr(eoh + 4);
                raw_header_data.resize(eoh);
                found_eoh = true;
                break;
            }
            if (raw_header_data.size() >= MAX_HEADER_SIZE) break;
            const ssize_t n = socket_read(fd, chunk, sizeof(chunk));
            if (n <= 0) return {};
            raw_header_data.append(chunk, static_cast<size_t>(n));
        }
        if (!found_eoh) {
            // EOF/timeout before any byte, or headers alone exceed the limit.
            if (raw_header_data.size() >= MAX_HEADER_SIZE) {
                std::fprintf(stderr, "[http] Headers too large (%zu bytes), rejecting\n",
                             raw_header_data.size());
            }
            return {};
        }
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
        return {};
    }

    std::string request_line = raw_header_data.substr(0, line_end);

    // Split "METHOD /path HTTP/1.1"
    size_t space1 = request_line.find(' ');
    size_t space2 = request_line.rfind(' ');
    if (space1 == std::string::npos || space2 == std::string::npos || space1 == space2) {
        std::fprintf(stderr, "[http] Malformed request line: %.100s\n", request_line.c_str());
        return {};
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
        if (header_count > MAX_HEADER_COUNT) {
            std::fprintf(stderr, "[http] Too many headers (%d), rejecting\n", header_count);
            return {};
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
        if (key == "content-length" && req.headers.count(key) != 0) {
            return {};  // duplicate framing headers are ambiguous
        }
        req.headers[key] = val;
    }

    // -------------------------------------------------------------------
    // 4. Read body (if Content-Length is present) — bounded
    // -------------------------------------------------------------------
    const auto cl_it = req.headers.find("content-length");
    const auto te_it = req.headers.find("transfer-encoding");
    if (cl_it != req.headers.end() && te_it != req.headers.end()) {
        // Ambiguous framing — reject rather than guess (RFC 9112 §6.3).
        std::fprintf(stderr, "[http] Both Content-Length and Transfer-Encoding, rejecting\n");
        return {};
    }
    if (te_it != req.headers.end()) {
        // This server does not implement chunked request bodies.
        std::fprintf(stderr, "[http] Transfer-Encoding request bodies unsupported, rejecting\n");
        return {};
    }

    if (cl_it != req.headers.end()) {
        try {
            if (cl_it->second.empty() ||
                cl_it->second.find_first_not_of("0123456789 \t") != std::string::npos ||
                cl_it->second.find_first_of("0123456789") != 0) {
                return {};
            }
            // Use stoll so we can detect values that don't fit in int, and
            // reject trailing garbage ("12abc") instead of parsing "12".
            size_t pos = 0;
            const long long cl = std::stoll(cl_it->second, &pos);
            while (pos < cl_it->second.size() &&
                   (cl_it->second[pos] == ' ' || cl_it->second[pos] == '\t')) {
                ++pos;
            }
            if (pos != cl_it->second.size()) {
                throw std::invalid_argument("trailing characters");
            }
            if (cl < 0 || static_cast<size_t>(cl) > MAX_BODY_SIZE) {
                std::fprintf(stderr, "[http] Content-Length %lld exceeds limit (%zu), rejecting\n",
                             cl, MAX_BODY_SIZE);
                return {};
            }
            req.content_length = static_cast<int>(cl);
            req.body.resize(static_cast<size_t>(req.content_length));
        } catch (const std::exception &) {
            std::fprintf(stderr, "[http] Invalid Content-Length: %s\n", cl_it->second.c_str());
            return {};
        }

        // Read exactly req.content_length bytes; `pending` holds the bytes
        // that arrived together with the headers.
        size_t offset = 0;
        if (!pending.empty()) {
            const size_t take = std::min(pending.size(), req.body.size());
            std::memcpy(req.body.data(), pending.data(), take);
            pending.erase(0, take);
            offset = take;
        }
        while (offset < req.body.size()) {
            ssize_t n = socket_read(fd, req.body.data() + offset,
                                    req.body.size() - offset);
            if (n <= 0) break; // timeout or disconnect
            offset += static_cast<size_t>(n);
        }
        if (offset != req.body.size()) return {};  // incomplete body
    }

    return req;
}

bool HttpServer::_send_response(int fd, const HttpResponse & res, bool close_conn) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << res.status_code << " " << res.status_text << "\r\n";

    for (auto & [key, val] : res.headers) {
        oss << key << ": " << val << "\r\n";
    }

    // If Content-Length header wasn't explicitly set, add it (case-insensitive check)
    bool has_content_length = false;
    for (auto & [k, v] : res.headers) {
        std::string lower_k = k;
        std::transform(lower_k.begin(), lower_k.end(), lower_k.begin(), ::tolower);
        if (lower_k == "content-length") {
            has_content_length = true;
            break;
        }
    }
    if (!has_content_length) {
        oss << "Content-Length: " << res.body.size() << "\r\n";
    }

    // Only add Connection: close if not already present (case-insensitive)
    bool has_connection_header = false;
    for (auto & [k, v] : res.headers) {
        std::string lower_k = k;
        std::transform(lower_k.begin(), lower_k.end(), lower_k.begin(), ::tolower);
        if (lower_k == "connection") {
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

    // No shutdown() here — the handler thread loops back to _parse_request
    // to read the next keep-alive request.  When the client closes its end,
    // socket_read returns EOF (0) and the loop exits, followed by socket_close().
    return true;
}

} // namespace kokopop
