#include "http/async_server.h"

#include "http/http_server.h"  // for HttpRequest, HttpResponse, json_error
#include "core/wav.h"
#include "yyjson.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
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

namespace kokopop {

// ---------------------------------------------------------------------------
// Platform helpers
// ---------------------------------------------------------------------------

namespace async_socket {

int close_fd(int fd) {
#ifdef _WIN32
    return closesocket(fd);
#else
    return ::close(fd);
#endif
}

ssize_t read_fd(int fd, void * buf, size_t len) {
#ifdef _WIN32
    return recv(fd, static_cast<char *>(buf), static_cast<int>(len), 0);
#else
    ssize_t n;
    do { n = ::read(fd, buf, len); } while (n < 0 && errno == EINTR);
    return n;
#endif
}

ssize_t write_fd(int fd, const void * buf, size_t len) {
#ifdef _WIN32
    return send(fd, static_cast<const char *>(buf), static_cast<int>(len), 0);
#else
    ssize_t n;
    do { n = ::write(fd, buf, len); } while (n < 0 && errno == EINTR);
    return n;
#endif
}

void set_nonblocking(int fd) {
#ifdef _WIN32
    u_long mode = 1;
    ioctlsocket(fd, FIONBIO, &mode);
#else
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
#endif
}

void set_nodelay(int fd) {
#ifdef _WIN32
    int opt = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY,
               reinterpret_cast<const char *>(&opt), sizeof(opt));
#else
    int opt = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#endif
}

void ignore_sigpipe() {
#ifndef _WIN32
    struct sigaction sa{};
    sa.sa_handler = SIG_IGN;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGPIPE, &sa, nullptr);
#endif
}

} // namespace async_socket

// ---------------------------------------------------------------------------
// AsyncHttpServer implementation
// ---------------------------------------------------------------------------

AsyncHttpServer::AsyncHttpServer() {
    async_socket::ignore_sigpipe();
}

AsyncHttpServer::~AsyncHttpServer() {
    stop();
    join();
    if (_server_fd >= 0) {
        async_socket::close_fd(_server_fd);
        _server_fd = -1;
    }
}

void AsyncHttpServer::route(const std::string & path, RequestHandler handler) {
    _routes[path] = std::move(handler);
}

void AsyncHttpServer::route_default(RequestHandler handler) {
    _default_handler = std::move(handler);
}

bool AsyncHttpServer::start(const std::string & addr, int port) {
    if (port < 1 || port > 65535) {
        std::fprintf(stderr, "[http] Invalid port: %d\n", port);
        return false;
    }

    _addr = addr;
    _port = port;

    _server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (_server_fd < 0) {
        std::fprintf(stderr, "[http] Failed to create socket\n");
        return false;
    }

    int opt = 1;
#ifdef _WIN32
    setsockopt(_server_fd, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char *>(&opt), sizeof(opt));
#else
    setsockopt(_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    async_socket::set_nonblocking(_server_fd);

    sockaddr_in addr_in{};
    addr_in.sin_family = AF_INET;
    addr_in.sin_port = htons(static_cast<uint16_t>(port));
    if (addr.empty() || addr == "*" || addr == "0.0.0.0") {
        addr_in.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, addr.c_str(), &addr_in.sin_addr) <= 0) {
            std::fprintf(stderr, "[http] Invalid bind address: %s\n", addr.c_str());
            async_socket::close_fd(_server_fd);
            _server_fd = -1;
            return false;
        }
    }

    if (bind(_server_fd, reinterpret_cast<sockaddr *>(&addr_in), sizeof(addr_in)) < 0) {
        std::fprintf(stderr, "[http] Failed to bind to %s:%d\n",
                     addr.empty() ? "0.0.0.0" : addr.c_str(), port);
        async_socket::close_fd(_server_fd);
        _server_fd = -1;
        return false;
    }

    if (listen(_server_fd, 128) < 0) {
        std::fprintf(stderr, "[http] Failed to listen\n");
        async_socket::close_fd(_server_fd);
        _server_fd = -1;
        return false;
    }

    _running.store(true);
    _loop_thread = std::thread(&AsyncHttpServer::_event_loop, this);

    std::fprintf(stderr, "[http] Async server listening on %s:%d (max %d conns)\n",
                addr.empty() ? "0.0.0.0" : addr.c_str(), port, MAX_CONNECTIONS);
    return true;
}

void AsyncHttpServer::stop() {
    _running.store(false);
    if (_server_fd >= 0) {
        async_socket::close_fd(_server_fd);
        _server_fd = -1;
    }
}

void AsyncHttpServer::join() {
    if (_loop_thread.joinable()) {
        _loop_thread.join();
    }
    for (auto & [fd, conn] : _connections) {
        if (conn.fd >= 0) async_socket::close_fd(conn.fd);
    }
    _connections.clear();
}

// ---------------------------------------------------------------------------
// Event loop
// ---------------------------------------------------------------------------

void AsyncHttpServer::_event_loop() {
    _poll_fds.resize(MAX_CONNECTIONS + 2);

    while (_running.load()) {
        int nfds = 0;

        // Server socket
        if (_server_fd >= 0) {
            _poll_fds[nfds].fd = _server_fd;
            _poll_fds[nfds].events = POLLIN;
            _poll_fds[nfds].revents = 0;
            nfds++;
        }

        // Client connections
        for (auto & [fd, conn] : _connections) {
            if (nfds >= static_cast<int>(_poll_fds.size())) break;

            _poll_fds[nfds].fd = fd;
            _poll_fds[nfds].events = 0;
            _poll_fds[nfds].revents = 0;

            if (conn.state == Connection::STATE_READING_HEADERS ||
                conn.state == Connection::STATE_READING_BODY) {
                _poll_fds[nfds].events |= POLLIN;
            }
            if (!conn.write_buffer.empty() ||
                conn.state == Connection::STATE_STREAMING) {
                _poll_fds[nfds].events |= POLLOUT;
            }

            nfds++;
        }

        int result = poll(_poll_fds.data(), nfds, 100);
        if (result < 0) {
#ifndef _WIN32
            if (errno != EINTR && _running.load()) {
                std::fprintf(stderr, "[http] poll() error: %s\n", strerror(errno));
            }
#endif
            continue;
        }

        // Server accept
        if (_server_fd >= 0 && nfds > 0 && _poll_fds[0].revents & (POLLIN | POLLERR)) {
            _handle_server_accept();
        }

        // Connection events
        for (int i = 1; i < nfds; i++) {
            int fd = _poll_fds[i].fd;
            auto it = _connections.find(fd);
            if (it == _connections.end()) continue;
            Connection & conn = it->second;

            if (_poll_fds[i].revents & (POLLIN | POLLHUP | POLLERR)) {
                _handle_connection_read(fd, conn);
            }
            // Re-check: read handler may have closed the connection
            if (_connections.find(fd) == _connections.end()) continue;
            if (_poll_fds[i].revents & (POLLOUT | POLLHUP | POLLERR)) {
                _handle_connection_write(fd, conn);
            }
        }

        // Drain output queue for streaming connections.
        // Collect fds to close after the loop to avoid iterator invalidation.
        std::vector<int> fds_to_close;
        for (auto & [fd, conn] : _connections) {
            if (conn.state != Connection::STATE_STREAMING || !conn.req_ctx) continue;

            auto rs = conn.req_ctx->state.load();
            bool is_stream = conn.req_ctx->stream_mode;

            RequestContext::AudioChunk chunk;
            while (conn.req_ctx->try_pop(chunk)) {
                if (is_stream) {
                    const char * p = reinterpret_cast<const char *>(chunk.samples.data());
                    std::vector<char> raw(p, p + chunk.samples.size() * sizeof(float));
                    _send_http_chunk(fd, conn, raw);
                } else {
                    auto & acc = conn.req_ctx->wav_accumulator;
                    acc.insert(acc.end(), chunk.samples.begin(), chunk.samples.end());
                }
            }

            bool is_terminal = (rs == RequestContext::State::DONE ||
                                rs == RequestContext::State::ERROR ||
                                rs == RequestContext::State::CANCELLED);
            if (is_terminal && conn.write_buffer.empty()) {
                if (is_stream) {
                    _send_final_chunk(fd, conn);
                    if (rs == RequestContext::State::CANCELLED) {
                        fds_to_close.push_back(fd);
                    }
                } else if (rs == RequestContext::State::DONE) {
                    _send_wav_response(fd, conn);
                } else {
                    const std::string & msg = conn.req_ctx->error;
                    _send_error(fd, conn, 500, "Internal Server Error",
                                json_error(msg.empty() ? "synthesis failed" : msg));
                }
            }
        }
        for (int fd : fds_to_close) {
            _close_connection(fd);
        }
    }
}

void AsyncHttpServer::_handle_server_accept() {
    if (_connections.size() >= static_cast<size_t>(MAX_CONNECTIONS)) {
        sockaddr_in addr{};
        socklen_t len = sizeof(addr);
        int fd = accept(_server_fd, reinterpret_cast<sockaddr *>(&addr), &len);
        if (fd >= 0) async_socket::close_fd(fd);
        return;
    }

    sockaddr_in addr{};
    socklen_t len = sizeof(addr);
    int fd = accept(_server_fd, reinterpret_cast<sockaddr *>(&addr), &len);
    if (fd < 0) {
#ifndef _WIN32
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return;
#endif
        return;
    }

    async_socket::set_nonblocking(fd);
    async_socket::set_nodelay(fd);

    Connection conn;
    conn.fd = fd;
    conn.state = Connection::STATE_READING_HEADERS;
    conn.read_buffer.reserve(4096);
    _connections[fd] = std::move(conn);
}

void AsyncHttpServer::_handle_connection_read(int fd, Connection & conn) {
    char buf[8192];
    ssize_t n = async_socket::read_fd(fd, buf, sizeof(buf));

    if (n <= 0) {
        if (n == 0) {
            // EOF / client disconnect
        } else {
#ifndef _WIN32
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return;
#endif
        }
        if (conn.req_ctx) conn.req_ctx->cancelled.store(true);
        _close_connection(fd);
        return;
    }

    conn.read_buffer.append(buf, static_cast<size_t>(n));

    if (conn.read_buffer.size() > MAX_HEADER_SIZE + MAX_BODY_SIZE) {
        _send_error(fd, conn, 413, "Payload Too Large", json_error("Request too large"));
        _close_connection(fd);
        return;
    }

    // ---- Parse headers ----
    if (conn.state == Connection::STATE_READING_HEADERS) {
        auto hdr_end = conn.read_buffer.find("\r\n\r\n");
        if (hdr_end == std::string::npos) return; // Wait for more data

        // Parse request
        std::string hdr_data(conn.read_buffer.begin(),
                            conn.read_buffer.begin() + static_cast<ptrdiff_t>(hdr_end));

        auto line_end = hdr_data.find("\r\n");
        if (line_end == std::string::npos) {
            _send_error(fd, conn, 400, "Bad Request", json_error("Malformed request"));
            _close_connection(fd);
            return;
        }

        std::string req_line = hdr_data.substr(0, line_end);
        std::string method, path, qs;
        if (!_parse_request_line(req_line, method, path, qs)) {
            _send_error(fd, conn, 400, "Bad Request", json_error("Malformed request line"));
            _close_connection(fd);
            return;
        }

        // Parse headers
        std::string hdr_block = hdr_data.substr(line_end + 2);
        std::istringstream iss(hdr_block);
        std::string line;
        while (std::getline(iss, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            size_t colon = line.find(':');
            if (colon == std::string::npos || colon == 0) continue;
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) val.erase(val.begin());
            std::transform(key.begin(), key.end(), key.begin(), ::tolower);
            conn.request.headers[key] = val;
        }

        conn.request.method = std::move(method);
        conn.request.path = std::move(path);
        conn.request.query_string = std::move(qs);
        conn.request.content_length = -1;

        // Determine body size
        auto cl_it = conn.request.headers.find("content-length");
        if (cl_it != conn.request.headers.end()) {
            try {
                long long cl = std::stoll(cl_it->second);
                if (cl < 0 || cl > static_cast<long long>(MAX_BODY_SIZE)) {
                    _send_error(fd, conn, 413, "Payload Too Large", json_error("Body too large"));
                    _close_connection(fd);
                    return;
                }
                conn.content_length = static_cast<int>(cl);
            } catch (...) {
                _send_error(fd, conn, 400, "Bad Request", json_error("Invalid Content-Length"));
                _close_connection(fd);
                return;
            }
            conn.request.content_length = conn.content_length;
        }

        // Check if body is already in the buffer
        size_t body_start = hdr_end + 4;
        if (conn.content_length > 0) {
            size_t avail = conn.read_buffer.size() - body_start;
            if (avail >= static_cast<size_t>(conn.content_length)) {
                // Full body available
                conn.request.body.assign(conn.read_buffer.begin() + static_cast<ptrdiff_t>(body_start),
                                        conn.read_buffer.begin() + static_cast<ptrdiff_t>(body_start) + conn.content_length);
                conn.read_buffer.clear();
                conn.state = Connection::STATE_PROCESSING;
                _dispatch_request(fd, conn);
            } else {
                // Need more body data
                conn.state = Connection::STATE_READING_BODY;
                conn.body_bytes_read = avail;
                // Trim header data from buffer, keep body data
                if (avail > 0) {
                    conn.read_buffer.erase(0, static_cast<size_t>(body_start));
                } else {
                    conn.read_buffer.clear();
                }
            }
        } else {
            // No body
            conn.read_buffer.clear();
            conn.state = Connection::STATE_PROCESSING;
            _dispatch_request(fd, conn);
        }
    } else if (conn.state == Connection::STATE_READING_BODY) {
        conn.body_bytes_read += static_cast<size_t>(n);
        if (conn.body_bytes_read >= static_cast<size_t>(conn.content_length)) {
            conn.request.body.assign(conn.read_buffer.begin(), conn.read_buffer.end());
            conn.read_buffer.clear();
            conn.state = Connection::STATE_PROCESSING;
            _dispatch_request(fd, conn);
        }
    }
}

void AsyncHttpServer::_handle_connection_write(int fd, Connection & conn) {
    if (conn.write_buffer.empty()) return;

    ssize_t n = async_socket::write_fd(fd, conn.write_buffer.data(),
                                       conn.write_buffer.size());
    if (n > 0) {
        conn.write_buffer.erase(0, static_cast<size_t>(n));
        if (conn.write_buffer.empty() && conn.state != Connection::STATE_STREAMING) {
            _close_connection(fd);
        }
    } else if (n < 0) {
#ifndef _WIN32
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return;
#endif
        _close_connection(fd);
    }
}

// ---------------------------------------------------------------------------
// Request dispatch
// ---------------------------------------------------------------------------

void AsyncHttpServer::_dispatch_request(int fd, Connection & conn) {
    HttpRequest & req = conn.request;

    // ---- /tts: submit to scheduler, streaming response ----
    if (req.path == "/tts") {
        if (req.method != "POST") {
            _send_error(fd, conn, 405, "Method Not Allowed",
                        json_error("POST required"));
            _close_connection(fd);
            return;
        }
        if (!_scheduler) {
            _send_error(fd, conn, 503, "Service Unavailable",
                        json_error("Scheduler not configured"));
            _close_connection(fd);
            return;
        }

        // Parse JSON
        yyjson_read_err err;
        yyjson_doc * doc = yyjson_read_opts(
            static_cast<char *>(req.body.data()), req.body.size(),
            YYJSON_READ_ALLOW_TRAILING_COMMAS, nullptr, &err);
        if (!doc) {
            _send_error(fd, conn, 400, "Bad Request",
                        json_error(std::string("JSON parse error: ") + err.msg));
            _close_connection(fd);
            return;
        }

        yyjson_val * root = yyjson_doc_get_root(doc);
        if (!yyjson_is_obj(root)) {
            yyjson_doc_free(doc);
            _send_error(fd, conn, 400, "Bad Request",
                        json_error("JSON is not an object"));
            _close_connection(fd);
            return;
        }

        const char * text      = yyjson_get_str(yyjson_obj_get(root, "text"));
        const char * voice     = yyjson_get_str(yyjson_obj_get(root, "voice"));
        const char * mode_str  = yyjson_get_str(yyjson_obj_get(root, "mode"));
        const char * fmt_str   = yyjson_get_str(yyjson_obj_get(root, "format"));

        float spd = _default_speed;
        yyjson_val * speed_val = yyjson_obj_get(root, "speed");
        if (speed_val && !yyjson_is_null(speed_val) && yyjson_is_num(speed_val)) {
            spd = (float)yyjson_get_num(speed_val);
        }

        // "format": "wav"  → accumulate and return a complete WAV file
        // "format": "pcm"  → stream raw float32 PCM (default)
        bool use_stream = !(fmt_str && std::string(fmt_str) == "wav");

        std::string current_voice = voice ? voice : _default_voice.c_str();
        StreamMode current_mode = _stream_mode;
        if (mode_str) {
            if (std::string(mode_str) == "long_form") {
                current_mode = StreamMode::LongForm;
            } else if (std::string(mode_str) != "interactive") {
                yyjson_doc_free(doc);
                _send_error(fd, conn, 400, "Bad Request",
                            json_error(std::string("Unknown mode: ") + mode_str));
                _close_connection(fd);
                return;
            }
        }

        if (!text || std::string(text).empty()) {
            yyjson_doc_free(doc);
            _send_error(fd, conn, 400, "Bad Request",
                        json_error("Missing or empty 'text' field"));
            _close_connection(fd);
            return;
        }

        std::string text_str(text);
        yyjson_doc_free(doc);
        if (text_str.size() > 100000) {
            _send_error(fd, conn, 413, "Payload Too Large",
                        json_error("Text too long (max 100,000 chars)"));
            _close_connection(fd);
            return;
        }

        std::fprintf(stderr, "[http] POST /tts: %zu chars, voice=%s, speed=%.1f, format=%s\n",
                     text_str.size(), current_voice.c_str(), spd,
                     use_stream ? "pcm" : "wav");

        // Submit to scheduler (round-robin interleaving)
        auto ctx = _scheduler->submit(
            text_str, current_voice, spd, current_mode, use_stream);

        _send_streaming_response(fd, conn, ctx);
        return;
    }

    // ---- Other routes ----
    auto it = _routes.find(req.path);
    if (it == _routes.end()) {
        if (_default_handler) {
            HttpResponse res;
            _default_handler(req, res);
            _send_response(fd, conn, res);
        } else {
            _send_error(fd, conn, 404, "Not Found", json_error("not_found"));
            _close_connection(fd);
        }
        return;
    }

    HttpResponse res;
    it->second(req, res);
    _send_response(fd, conn, res);
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

void AsyncHttpServer::_send_response(int fd, Connection & conn,
                                     const HttpResponse & res) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << res.status_code << " " << res.status_text << "\r\n";

    for (auto & [key, val] : res.headers) {
        oss << key << ": " << val << "\r\n";
    }

    bool has_cl = false;
    for (auto & [k, v] : res.headers) {
        std::string lk = k;
        std::transform(lk.begin(), lk.end(), lk.begin(), ::tolower);
        if (lk == "content-length") { has_cl = true; break; }
    }
    if (!has_cl) {
        oss << "Content-Length: " << res.body.size() << "\r\n";
    }

    oss << "Server: kokopop-async/0.1\r\n";
    oss << "\r\n";

    conn.write_buffer = oss.str();
    conn.write_buffer.insert(conn.write_buffer.end(),
                            res.body.begin(), res.body.end());
    conn.state = Connection::STATE_WRITING;
}

void AsyncHttpServer::_send_streaming_response(int fd, Connection & conn,
                                                std::shared_ptr<RequestContext> ctx) {
    conn.req_ctx = std::move(ctx);
    if (conn.req_ctx->stream_mode) {
        // PCM float32 stream — send headers immediately, chunks follow
        std::ostringstream oss;
        oss << "HTTP/1.1 200 OK\r\n";
        oss << "Transfer-Encoding: chunked\r\n";
        oss << "Content-Type: application/octet-stream\r\n";
        oss << "Server: kokopop-async/0.1\r\n";
        oss << "\r\n";
        conn.write_buffer = oss.str();
    }
    // WAV mode: no headers yet — they are sent with the complete file in _send_wav_response
    conn.state = Connection::STATE_STREAMING;
}

void AsyncHttpServer::_send_wav_response(int fd, Connection & conn) {
    auto & acc = conn.req_ctx->wav_accumulator;
    kokopop_audio audio{};
    audio.samples     = acc.data();
    audio.n_samples   = static_cast<int>(acc.size());
    audio.sample_rate = conn.req_ctx->sample_rate;

    auto wav = wav_bytes(audio);

    std::ostringstream oss;
    oss << "HTTP/1.1 200 OK\r\n";
    oss << "Content-Type: audio/wav\r\n";
    oss << "Content-Length: " << wav.size() << "\r\n";
    oss << "Server: kokopop-async/0.1\r\n";
    oss << "\r\n";

    conn.write_buffer = oss.str();
    conn.write_buffer.insert(conn.write_buffer.end(),
                             reinterpret_cast<const char *>(wav.data()),
                             reinterpret_cast<const char *>(wav.data() + wav.size()));
    conn.state = Connection::STATE_WRITING;
}

void AsyncHttpServer::_send_http_chunk(int fd, Connection & conn,
                                       const std::vector<char> & data) {
    std::ostringstream hdr;
    hdr << std::hex << data.size() << "\r\n";
    conn.write_buffer.append(hdr.str());
    conn.write_buffer.insert(conn.write_buffer.end(), data.begin(), data.end());
    conn.write_buffer.append("\r\n");
}

void AsyncHttpServer::_send_final_chunk(int fd, Connection & conn) {
    conn.write_buffer.append("0\r\n\r\n");
    conn.state = Connection::STATE_WRITING;
}

void AsyncHttpServer::_send_error(int fd, Connection & conn,
                                  int status_code, const std::string & status_text,
                                  const std::string & error_json) {
    HttpResponse res;
    res.status_code = status_code;
    res.status_text = status_text;
    res.headers["Server"] = "kokopop-async/0.1";
    res.body.insert(res.body.end(), error_json.begin(), error_json.end());
    res.set_content_type("application/json");
    _send_response(fd, conn, res);
}

void AsyncHttpServer::_close_connection(int fd) {
    auto it = _connections.find(fd);
    if (it != _connections.end()) {
        if (it->second.fd >= 0) async_socket::close_fd(it->second.fd);
        _connections.erase(it);
    }
}

bool AsyncHttpServer::_parse_request_line(const std::string & request_line,
                                          std::string & method,
                                          std::string & path,
                                          std::string & query_string) {
    size_t space1 = request_line.find(' ');
    size_t space2 = request_line.rfind(' ');
    if (space1 == std::string::npos || space2 == std::string::npos || space1 == space2) {
        return false;
    }

    method = request_line.substr(0, space1);
    std::string path_and_query = request_line.substr(space1 + 1, space2 - space1 - 1);
    size_t qmark = path_and_query.find('?');
    if (qmark != std::string::npos) {
        path = path_and_query.substr(0, qmark);
        query_string = path_and_query.substr(qmark + 1);
    } else {
        path = path_and_query;
        query_string.clear();
    }
    return true;
}

} // namespace kokopop
