#include "http/async_server.h"

#include "audio/audio_encoder.h"
#include "http/http_response_writer.h"
#include "http/http_server.h"  // for HttpRequest, HttpResponse, json_error
#include "yyjson.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <sstream>

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
    _last_activity_ms.store(_now_ms());
    _loop_thread = std::thread(&AsyncHttpServer::_event_loop, this);

    if (_idle_unload_seconds > 0) {
        _idle_thread = std::thread(&AsyncHttpServer::_idle_loop, this);
        std::fprintf(stderr, "[http] Idle unload enabled: model will be unloaded after %ds of inactivity\n",
                     _idle_unload_seconds);
    }

    std::fprintf(stderr, "[http] Async server listening on %s:%d (max %d conns)\n",
                addr.empty() ? "0.0.0.0" : addr.c_str(), port, MAX_CONNECTIONS);
    return true;
}

void AsyncHttpServer::set_idle_unload(int idle_seconds, ModelReloadFn reload_fn,
                                      ModelUnloadFn unload_fn) {
    _idle_unload_seconds = idle_seconds;
    _reload_fn = std::move(reload_fn);
    _unload_fn = std::move(unload_fn);
}

void AsyncHttpServer::stop() {
    _running.store(false);
    _idle_cv.notify_all();
    if (_server_fd >= 0) {
        async_socket::close_fd(_server_fd);
        _server_fd = -1;
    }
}

void AsyncHttpServer::join() {
    if (_loop_thread.joinable()) {
        _loop_thread.join();
    }
    if (_idle_thread.joinable()) {
        _idle_thread.join();
    }
    for (auto & [fd, conn] : _connections) {
        if (conn.fd >= 0) async_socket::close_fd(conn.fd);
    }
    _connections.clear();
}

// ---------------------------------------------------------------------------
// Idle unload loop
// ---------------------------------------------------------------------------

void AsyncHttpServer::_idle_loop() {
    while (_running.load()) {
        {
            std::unique_lock<std::mutex> lk(_idle_cv_mutex);
            _idle_cv.wait_for(lk, std::chrono::seconds(30),
                              [this] { return !_running.load(); });
        }
        if (!_running.load()) break;
        if (!_model_loaded.load()) continue;
        if (_active_streams.load() > 0) continue;

        int64_t elapsed_ms = _now_ms() - _last_activity_ms.load();
        if (elapsed_ms < static_cast<int64_t>(_idle_unload_seconds) * 1000) continue;

        // Acquire lifecycle lock. If a request just came in, it will hold the lock
        // — try_lock avoids blocking the idle thread indefinitely; we'll retry next round.
        std::unique_lock<std::mutex> lock(_lifecycle_mutex, std::try_to_lock);
        if (!lock.owns_lock()) continue;

        // Re-check under lock
        if (!_model_loaded.load()) continue;
        if (_active_streams.load() > 0) continue;
        elapsed_ms = _now_ms() - _last_activity_ms.load();
        if (elapsed_ms < static_cast<int64_t>(_idle_unload_seconds) * 1000) continue;

        std::fprintf(stderr, "[http] No activity for %llds — unloading model\n",
                     (long long)(elapsed_ms / 1000));

        _model_loaded.store(false);
        _model     = nullptr;
        _scheduler = nullptr;

        // unload_fn stops/joins the scheduler and frees the model.
        // We hold _lifecycle_mutex so any concurrent /tts will wait; this is
        // intentional — the request will then trigger a reload once we release.
        if (_unload_fn) _unload_fn();
    }
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

            // Always watch for read events: needed for reading request data
            // and for detecting client disconnects (POLLHUP/POLLERR) on streaming
            // connections that have no outstanding writes.
            if (conn.state == Connection::STATE_READING_HEADERS ||
                conn.state == Connection::STATE_READING_BODY ||
                conn.state == Connection::STATE_STREAMING) {
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

            // --- Read events ---
            if (_poll_fds[i].revents & (POLLIN | POLLHUP | POLLERR)) {
                auto it = _connections.find(fd);
                if (it != _connections.end()) {
                    _handle_connection_read(fd, it->second);
                }
            }

            // Re-fetch: _handle_connection_read may have closed the connection
            auto it = _connections.find(fd);
            if (it == _connections.end()) continue;

            // --- Write events ---
            if (_poll_fds[i].revents & (POLLOUT | POLLHUP | POLLERR)) {
                _handle_connection_write(fd, it->second);
            }
        }

        // Drain output queue for streaming connections.
        // Collect fds to close after the loop to avoid iterator invalidation.
        std::vector<int> fds_to_close;

        // Sweep for connections stuck in header/body reading past the idle timeout.
        {
            int64_t now = _now_ms();
            for (auto & [fd, conn] : _connections) {
                if ((conn.state == Connection::STATE_READING_HEADERS ||
                     conn.state == Connection::STATE_READING_BODY) &&
                    now - conn.last_activity_ms > CONN_IDLE_TIMEOUT_MS) {
                    fds_to_close.push_back(fd);
                }
            }
            for (int fd : fds_to_close) _close_connection(fd);
            fds_to_close.clear();
        }

        for (auto & [fd, conn] : _connections) {
            if (conn.state != Connection::STATE_STREAMING || !conn.req_ctx) continue;

            // Client disconnected — stop draining, close the connection.
            if (conn.req_ctx->cancelled.load()) {
                fds_to_close.push_back(fd);
                continue;
            }

            auto rs  = conn.req_ctx->state.load();
            auto fmt = conn.req_ctx->format;
            bool is_terminal = (rs == RequestContext::State::DONE ||
                                rs == RequestContext::State::ERROR ||
                                rs == RequestContext::State::CANCELLED);

            RequestContext::AudioChunk chunk;
            while (conn.write_buffer.size() <= WRITE_BUFFER_HIGH_WATER &&
                   conn.req_ctx->try_pop(chunk)) {
                if (conn.stream_encoder) {
                    std::vector<char> encoded;
                    conn.stream_encoder->write(std::move(chunk), is_terminal, encoded);
                    if (!encoded.empty()) {
                        _send_http_chunk(fd, conn, encoded);
                    }
                } else {
                    auto & acc = conn.req_ctx->wav_accumulator;
                    acc.insert(acc.end(), chunk.samples.begin(), chunk.samples.end());
                }
            }

            if (is_terminal && conn.write_buffer.empty()) {
                if (conn.stream_encoder) {
                    std::vector<char> tail;
                    conn.stream_encoder->finish(rs == RequestContext::State::DONE, tail);
                    if (!tail.empty()) {
                        _send_http_chunk(fd, conn, tail);
                    }
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
    conn.last_activity_ms = _now_ms();
    conn.state = Connection::STATE_READING_HEADERS;
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
        conn.write_buffer.clear();  // free orphaned chunks
        _close_connection(fd);
        return;
    }

    conn.last_activity_ms = _now_ms();
    conn.parser.append(buf, static_cast<size_t>(n));
    _process_pending_requests(fd, conn);
}

void AsyncHttpServer::_process_pending_requests(int fd, Connection & conn) {
    while (conn.state == Connection::STATE_READING_HEADERS ||
           conn.state == Connection::STATE_READING_BODY) {
        auto parsed = conn.parser.next();
        if (parsed.status == AsyncHttpParseResult::Status::NeedMore) return;
        if (parsed.status == AsyncHttpParseResult::Status::Error) {
            conn.keep_alive = false;
            conn.close_after_write = true;
            _send_error(fd, conn, parsed.status_code, parsed.status_text,
                        json_error(parsed.error_message));
            return;
        }

        conn.request = std::move(parsed.request);
        conn.keep_alive = parsed.keep_alive;
        conn.close_after_write = !parsed.keep_alive;
        conn.state = Connection::STATE_PROCESSING;
        _dispatch_request(fd, conn);
        if (_connections.find(fd) == _connections.end()) return;
        if (conn.state != Connection::STATE_READING_HEADERS) return;
    }
}

void AsyncHttpServer::_handle_connection_write(int fd, Connection & conn) {
    if (conn.write_buffer.empty()) return;

    ssize_t n = async_socket::write_fd(fd, conn.write_buffer.data(),
                                       conn.write_buffer.size());
    if (n > 0) {
        conn.write_buffer.erase(0, static_cast<size_t>(n));
        if (conn.write_buffer.empty() && conn.state != Connection::STATE_STREAMING) {
            if (conn.close_after_write || !conn.keep_alive) {
                _close_connection(fd);
            } else {
                _prepare_next_request(fd, conn);
            }
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
            return;
        }

        if (req.headers.find("content-length") == req.headers.end() &&
            req.headers.find("transfer-encoding") == req.headers.end()) {
            _send_error(fd, conn, 411, "Length Required",
                        json_error("Content-Length or Transfer-Encoding: chunked required"));
            return;
        }

        // Ensure model is loaded; reload if it was unloaded due to idle timeout.
        if (_idle_unload_seconds > 0) {
            std::unique_lock<std::mutex> lock(_lifecycle_mutex);
            if (!_model_loaded.load()) {
                std::fprintf(stderr, "[http] Reloading model for incoming request\n");
                if (!_reload_fn || !_reload_fn()) {
                    _send_error(fd, conn, 503, "Service Unavailable",
                                json_error("Model reload failed"));
                    return;
                }
                _model_loaded.store(true);
                std::fprintf(stderr, "[http] Model reloaded\n");
            }
            _last_activity_ms.store(_now_ms());
        }

        if (!_scheduler) {
            _send_error(fd, conn, 503, "Service Unavailable",
                        json_error("Scheduler not configured"));
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
            return;
        }

        yyjson_val * root = yyjson_doc_get_root(doc);
        if (!yyjson_is_obj(root)) {
            yyjson_doc_free(doc);
            _send_error(fd, conn, 400, "Bad Request",
                        json_error("JSON is not an object"));
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

        int ogg_prebuffer_chunks = 0;
        yyjson_val * prebuffer_val = yyjson_obj_get(root, "prebuffer_chunks");
        if (prebuffer_val && !yyjson_is_null(prebuffer_val) && yyjson_is_num(prebuffer_val)) {
            ogg_prebuffer_chunks = std::max(0, static_cast<int>(yyjson_get_int(prebuffer_val)));
        }

        int first_chunk_target_tokens = 0;
        yyjson_val * fct_val = yyjson_obj_get(root, "first_chunk_target_tokens");
        if (fct_val && !yyjson_is_null(fct_val) && yyjson_is_num(fct_val)) {
            first_chunk_target_tokens = std::max(1, static_cast<int>(yyjson_get_int(fct_val)));
        }

        // Optional diffusion style sampling. Disabled by default; only the
        // "diffusion" boolean turns it on, the rest are optional tuning knobs.
        KokoroDiffusionOptions diffusion;  // defaults: disabled, steps=5, alpha=0.1, beta=0.5
        yyjson_val * diff_val = yyjson_obj_get(root, "diffusion");
        if (diff_val && yyjson_is_bool(diff_val)) {
            diffusion.enabled = yyjson_get_bool(diff_val);
        } else if (diff_val && !yyjson_is_null(diff_val)) {
            yyjson_doc_free(doc);
            _send_error(fd, conn, 400, "Bad Request",
                        json_error("'diffusion' must be a boolean"));
            return;
        }
        if (diffusion.enabled) {
            yyjson_val * seed_val = yyjson_obj_get(root, "diffusion_seed");
            if (seed_val && yyjson_is_num(seed_val)) {
                diffusion.seed = static_cast<uint32_t>(std::max<int64_t>(0, yyjson_get_sint(seed_val)));
            }
            yyjson_val * steps_val = yyjson_obj_get(root, "diffusion_steps");
            if (steps_val && yyjson_is_num(steps_val)) {
                diffusion.steps = static_cast<int>(yyjson_get_int(steps_val));
            }
            yyjson_val * alpha_val = yyjson_obj_get(root, "diffusion_alpha");
            if (alpha_val && yyjson_is_num(alpha_val)) {
                diffusion.alpha = static_cast<float>(yyjson_get_num(alpha_val));
            }
            yyjson_val * beta_val = yyjson_obj_get(root, "diffusion_beta");
            if (beta_val && yyjson_is_num(beta_val)) {
                diffusion.beta = static_cast<float>(yyjson_get_num(beta_val));
            }
            yyjson_val * escale_val = yyjson_obj_get(root, "diffusion_embedding_scale");
            if (escale_val && yyjson_is_num(escale_val)) {
                diffusion.embedding_scale = static_cast<float>(yyjson_get_num(escale_val));
            }
        }

        // "format": "pcm"     → stream raw float32 PCM (default)
        // "format": "wav"     → accumulate and return a complete WAV file
        // "format": "ogg"     → stream Ogg/Opus with Transfer-Encoding: chunked
        RequestContext::AudioFormat fmt = RequestContext::AudioFormat::PCM;
        if (fmt_str) {
            std::string fs(fmt_str);
            if (fs == "wav")      fmt = RequestContext::AudioFormat::WAV;
            else if (fs == "ogg") fmt = RequestContext::AudioFormat::OGG_OPUS;
            else {
                yyjson_doc_free(doc);
                _send_error(fd, conn, 400, "Bad Request",
                            json_error(std::string("Unknown format: ") + fs));
                return;
            }
        }

        if (!http_audio_format_available(fmt)) {
            yyjson_doc_free(doc);
            _send_error(fd, conn, 501, "Not Implemented",
                        json_error("Ogg/Opus output is not available in this build"));
            return;
        }

        if (!voice || std::string(voice).empty()) {
            yyjson_doc_free(doc);
            _send_error(fd, conn, 400, "Bad Request",
                        json_error("Missing or empty 'voice' field"));
            return;
        }

        std::string current_voice(voice);
        StreamMode current_mode = _stream_mode;
        if (mode_str) {
            std::string ms(mode_str);
            if (ms == "long_form") {
                current_mode = StreamMode::LongForm;
            } else if (ms == "adaptative") {
                current_mode = StreamMode::Adaptative;
            } else {
                yyjson_doc_free(doc);
                _send_error(fd, conn, 400, "Bad Request",
                            json_error(std::string("Unknown mode: ") + ms));
                return;
            }
        }

        if (!text || std::string(text).empty()) {
            yyjson_doc_free(doc);
            _send_error(fd, conn, 400, "Bad Request",
                        json_error("Missing or empty 'text' field"));
            return;
        }

        std::string text_str(text);
        yyjson_doc_free(doc);
        if (text_str.size() > 100000) {
            _send_error(fd, conn, 413, "Payload Too Large",
                        json_error("Text too long (max 100,000 chars)"));
            return;
        }

        // Validate diffusion support up front so we can fail with a clean 4xx
        // instead of aborting mid-stream after headers are already sent.
        if (diffusion.enabled &&
            (!_model || _model->cached_tensor("kokopop.diffusion.to_out.1.weight") == nullptr)) {
            _send_error(fd, conn, 400, "Bad Request",
                        json_error("diffusion requested, but this model has no diffusion tensors"));
            return;
        }

        const char * fmt_name =
            fmt == RequestContext::AudioFormat::WAV      ? "wav"
          : fmt == RequestContext::AudioFormat::OGG_OPUS ? "ogg"
          :                                                "pcm";
        std::fprintf(stderr, "[http] POST /tts: %zu chars, voice=%s, speed=%.1f, format=%s\n",
                     text_str.size(), current_voice.c_str(), spd, fmt_name);

        // Submit to scheduler (round-robin interleaving)
        ChunkConfig chunk_cfg_override;
        bool has_chunk_cfg_override = false;
        if (first_chunk_target_tokens > 0) {
            chunk_cfg_override.first_chunk_target_max_tokens = first_chunk_target_tokens;
            has_chunk_cfg_override = true;
        }
        auto ctx = _scheduler->submit(
            text_str, current_voice, spd, current_mode, fmt,
            ogg_prebuffer_chunks,
            chunk_cfg_override, has_chunk_cfg_override, diffusion);

        _send_streaming_response(fd, conn, ctx);
        return;
    }

    // ---- Other routes ----
    auto it = _routes.find(req.path);
    if (it == _routes.end()) {
        if (_default_handler) {
            HttpResponse res;
            bool keep = _default_handler(req, res);
            if (!keep) {
                conn.keep_alive = false;
                conn.close_after_write = true;
            }
            _send_response(fd, conn, res);
        } else {
            _send_error(fd, conn, 404, "Not Found", json_error("not_found"));
        }
        return;
    }

    HttpResponse res;
    bool keep = it->second(req, res);
    if (!keep) {
        conn.keep_alive = false;
        conn.close_after_write = true;
    }
    _send_response(fd, conn, res);
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

void AsyncHttpServer::_send_response(int fd, Connection & conn,
                                     const HttpResponse & res) {
    auto conn_header = res.headers.find("Connection");
    if (conn_header == res.headers.end()) {
        conn_header = res.headers.find("connection");
    }
    if (conn_header != res.headers.end() &&
        http_header_value_has_token(conn_header->second, "close")) {
        conn.keep_alive = false;
        conn.close_after_write = true;
    }
    conn.write_buffer = build_http_response_head(res, conn.keep_alive && !conn.close_after_write);
    conn.write_buffer.insert(conn.write_buffer.end(),
                            res.body.begin(), res.body.end());
    conn.state = Connection::STATE_WRITING;
}

void AsyncHttpServer::_send_streaming_response(int fd, Connection & conn,
                                                std::shared_ptr<RequestContext> ctx) {
    conn.req_ctx = std::move(ctx);
    conn.was_streaming = true;
    conn.close_after_write = true;
    conn.keep_alive = false;
    _active_streams.fetch_add(1);
    auto fmt = conn.req_ctx->format;

    if (fmt == RequestContext::AudioFormat::PCM ||
        fmt == RequestContext::AudioFormat::OGG_OPUS) {
        conn.stream_encoder = make_http_audio_stream_encoder(
            fmt, conn.req_ctx->sample_rate, conn.req_ctx->ogg_prebuffer_chunks);
        assert(conn.stream_encoder && "encoder should be available — checked by http_audio_format_available");
        conn.write_buffer = build_streaming_response_head(conn.stream_encoder->content_type());
        std::vector<char> initial;
        conn.stream_encoder->start(initial);
        if (!initial.empty()) {
            _send_http_chunk(fd, conn, initial);
        }
    }
    // WAV mode: no headers yet — they are sent with the complete file in _send_wav_response
    conn.state = Connection::STATE_STREAMING;
}

void AsyncHttpServer::_send_wav_response(int fd, Connection & conn) {
    auto & acc = conn.req_ctx->wav_accumulator;
    AudioEncoderOptions enc_options;
    enc_options.format = EncodedAudioFormat::WavPcm16;
    enc_options.sample_rate = conn.req_ctx->sample_rate;
    AudioEncoder encoder(enc_options);

    std::vector<uint8_t> ignored;
    std::vector<uint8_t> wav;
    std::string error;
    if (!encoder.start(ignored, error) ||
        !encoder.push(acc.data(), acc.size(), true, ignored, error) ||
        !encoder.finish(true, wav, error)) {
        _send_error(fd, conn, 500, "Internal Server Error",
                    json_error(error.empty() ? "WAV encoding failed" : error));
        return;
    }

    std::ostringstream oss;
    oss << "HTTP/1.1 200 OK\r\n";
    oss << "Content-Type: audio/wav\r\n";
    oss << "Content-Length: " << wav.size() << "\r\n";
    oss << "Server: kokopop-async/0.1\r\n";
    oss << "Connection: close\r\n";
    oss << "\r\n";

    conn.write_buffer = oss.str();
    conn.write_buffer.insert(conn.write_buffer.end(),
                             reinterpret_cast<const char *>(wav.data()),
                             reinterpret_cast<const char *>(wav.data() + wav.size()));
    conn.state = Connection::STATE_WRITING;
}

void AsyncHttpServer::_send_http_chunk(int fd, Connection & conn,
                                       const std::vector<char> & data) {
    (void)fd;
    append_http_chunk(conn.write_buffer, data);
}

void AsyncHttpServer::_send_final_chunk(int fd, Connection & conn) {
    (void)fd;
    append_final_http_chunk(conn.write_buffer);
    conn.close_after_write = true;
    conn.state = Connection::STATE_WRITING;
}

void AsyncHttpServer::_send_error(int fd, Connection & conn,
                                  int status_code, const std::string & status_text,
                                  const std::string & error_json) {
    HttpResponse res;
    res.status_code = status_code;
    res.status_text = status_text;
    res.body.insert(res.body.end(), error_json.begin(), error_json.end());
    res.set_content_type("application/json");
    _send_response(fd, conn, res);
}

void AsyncHttpServer::_close_connection(int fd) {
    auto it = _connections.find(fd);
    if (it != _connections.end()) {
        if (it->second.was_streaming) {
            int current = _active_streams.load();
            while (current > 0 &&
                   !_active_streams.compare_exchange_weak(current, current - 1)) {
            }
            _last_activity_ms.store(_now_ms());
        }
        if (it->second.fd >= 0) async_socket::close_fd(it->second.fd);
        _connections.erase(it);
    }
}

void AsyncHttpServer::_prepare_next_request(int fd, Connection & conn) {
    conn.request = HttpRequest{};
    conn.req_ctx.reset();
    conn.stream_encoder.reset();
    conn.keep_alive = false;
    conn.close_after_write = false;
    conn.state = Connection::STATE_READING_HEADERS;
    if (!conn.parser.empty()) {
        _process_pending_requests(fd, conn);
    }
}

} // namespace kokopop
