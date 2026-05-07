#pragma once

#include "http/request_context.h"
#include "http/synthesis_scheduler.h"
#include "http/http_server.h"  // for HttpRequest, HttpResponse, json_error, RequestHandler

#include <functional>
#include <map>
#include <poll.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace kokopop {

// ---------------------------------------------------------------------------
// AsyncHttpServer — event-driven HTTP server using poll()
//
// Architecture:
//   - Single thread runs the event loop (poll())
//   - Non-blocking sockets for all connections
//   - Progressive request parsing (headers + body may arrive in multiple reads)
//   - Progressive response writing (streaming with Transfer-Encoding: chunked)
//
// Connections are tracked by file descriptor.
// ---------------------------------------------------------------------------

class AsyncHttpServer {
public:
    AsyncHttpServer();
    ~AsyncHttpServer();

    AsyncHttpServer(const AsyncHttpServer &) = delete;
    AsyncHttpServer & operator=(const AsyncHttpServer &) = delete;

    /// Register a handler for a specific path
    /// Handler fills the HttpResponse; server manages connection lifecycle.
    void route(const std::string & path, RequestHandler handler);

    /// Register a default handler for unmatched paths
    void route_default(RequestHandler handler);

    /// Start serving (blocking). Call stop() to shut down.
    bool start(const std::string & addr, int port);

    /// Signal the server to stop
    void stop();

    /// Wait until the server has stopped
    void join();

    /// Check if server is running
    bool is_running() const { return _running.load(); }

    /// Set the synthesis scheduler (must be set before start())
    void set_scheduler(SynthesisScheduler & scheduler) {
        _scheduler = &scheduler;
    }

    /// Set the model pointer (for health/voices endpoints)
    void set_model(Model * model) {
        _model = model;
    }

    /// Default voice for /tts
    void set_default_voice(const std::string & voice) {
        _default_voice = voice;
    }

    /// Default speed for /tts
    void set_default_speed(float speed) {
        _default_speed = speed;
    }

    /// Default stream mode for /tts
    void set_stream_mode(StreamMode mode) {
        _stream_mode = mode;
    }

    /// Getters for use by route handlers
    SynthesisScheduler * get_scheduler() { return _scheduler; }
    Model * get_model() { return _model; }
    const std::string & get_default_voice() const { return _default_voice; }
    float get_default_speed() const { return _default_speed; }
    StreamMode get_stream_mode() const { return _stream_mode; }

private:
    struct Connection {
        int fd = -1;
        std::string read_buffer;
        HttpRequest request;
        bool request_complete{false};
        std::shared_ptr<RequestContext> req_ctx{nullptr};
        std::string write_buffer;
        bool write_in_progress{false};

        enum {
            STATE_IDLE,
            STATE_READING_HEADERS,
            STATE_READING_BODY,
            STATE_PROCESSING,
            STATE_WRITING,
            STATE_STREAMING  // Writing chunks as they become available
        } state = STATE_IDLE;

        // For body reading
        int content_length = -1;
        size_t body_bytes_read = 0;
    };

    void _event_loop();
    void _handle_server_accept();
    void _handle_connection_read(int fd, Connection & conn);
    void _handle_connection_write(int fd, Connection & conn);
    void _dispatch_request(int fd, Connection & conn);
    void _send_response(int fd, Connection & conn, const HttpResponse & res);
    void _send_http_chunk(int fd, Connection & conn, const std::vector<char> & data);
    void _send_final_chunk(int fd, Connection & conn);
    void _send_streaming_response(int fd, Connection & conn,
                                   std::shared_ptr<RequestContext> ctx);
    void _send_wav_response(int fd, Connection & conn);
    void _send_error(int fd, Connection & conn, int status_code,
                     const std::string & status_text,
                     const std::string & error_json);
    void _close_connection(int fd);

    bool _parse_request_line(const std::string & request_line, std::string & method,
                             std::string & path, std::string & query_string);

    int _server_fd = -1;
    std::string _addr;
    int _port = 0;

    std::atomic<bool> _running{false};
    std::thread _loop_thread;

    std::unordered_map<int, Connection> _connections;
    std::vector<struct pollfd> _poll_fds;
    std::map<std::string, RequestHandler> _routes;
    RequestHandler _default_handler;

    // Shared state with scheduler
    SynthesisScheduler * _scheduler = nullptr;
    Model * _model = nullptr;
    std::string _default_voice;
    float _default_speed = 1.0f;
    StreamMode _stream_mode = StreamMode::Interactive;

    // Connection timeout in milliseconds
    static constexpr int TIMEOUT_MS = 30000;

    // Maximum connections
    static constexpr int MAX_CONNECTIONS = 64;

    // Maximum header size
    static constexpr size_t MAX_HEADER_SIZE = 64 * 1024;

    // Maximum body size
    static constexpr size_t MAX_BODY_SIZE = 16 * 1024 * 1024;
};

// Inline helpers for platform portability
namespace async_socket {

int close_fd(int fd);
ssize_t read_fd(int fd, void * buf, size_t len);
ssize_t write_fd(int fd, const void * buf, size_t len);
void set_nonblocking(int fd);
void set_nodelay(int fd);
void ignore_sigpipe();  // Platform-specific

} // namespace async_socket

} // namespace kokopop
