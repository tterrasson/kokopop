#pragma once

#include "http/async_http_parser.h"
#include "http/http_audio_stream_encoder.h"
#include "http/request_context.h"
#include "http/synthesis_scheduler.h"
#include "http/http_server.h"  // for HttpRequest, HttpResponse, json_error, RequestHandler

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
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

    /// Set the model pointer (for the health endpoint)
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

    /// Enable automatic model unload after idle_seconds of inactivity.
    /// reload_fn  — called on the next request when the model is unloaded;
    ///              must update the server's _model and _scheduler via set_model/set_scheduler
    ///              and return true on success.
    /// unload_fn  — called by the idle monitor; must stop/join the scheduler and free the model.
    using ModelReloadFn = std::function<bool()>;
    using ModelUnloadFn = std::function<void()>;
    void set_idle_unload(int idle_seconds, ModelReloadFn reload_fn, ModelUnloadFn unload_fn);

private:
    // Maximum connections
    static constexpr int MAX_CONNECTIONS = 64;

    // Maximum header size
    static constexpr size_t MAX_HEADER_SIZE = 64 * 1024;

    // Maximum body size
    static constexpr size_t MAX_BODY_SIZE = 16 * 1024 * 1024;

    // Idle connection timeout (headers/body not fully received)
    static constexpr int64_t CONN_IDLE_TIMEOUT_MS = 30000;

    // write_buffer high-water mark: stop draining audio chunks when exceeded
    static constexpr size_t WRITE_BUFFER_HIGH_WATER = 256 * 1024;

    struct Connection {
        int fd = -1;
        int64_t last_activity_ms = 0;
        AsyncHttpRequestParser parser{MAX_HEADER_SIZE, MAX_BODY_SIZE};
        HttpRequest request;
        bool keep_alive = false;
        bool close_after_write = false;
        std::shared_ptr<RequestContext> req_ctx{nullptr};
        std::unique_ptr<HttpAudioStreamEncoder> stream_encoder;
        std::string write_buffer;
        bool was_streaming = false; // set on streaming start, used to update _active_streams on close

        enum {
            STATE_IDLE,
            STATE_READING_HEADERS,
            STATE_READING_BODY,
            STATE_PROCESSING,
            STATE_WRITING,
            STATE_STREAMING  // Writing chunks as they become available
        } state = STATE_IDLE;
    };

    void _event_loop();
    void _idle_loop();
    void _handle_server_accept();
    void _handle_connection_read(int fd, Connection & conn);
    void _handle_connection_write(int fd, Connection & conn);
    void _process_pending_requests(int fd, Connection & conn);
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
    void _prepare_next_request(int fd, Connection & conn);

    static int64_t _now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

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
    StreamMode _stream_mode = StreamMode::Adaptative;

    // Idle unload
    int _idle_unload_seconds = 0;
    ModelReloadFn _reload_fn;
    ModelUnloadFn _unload_fn;
    std::atomic<int64_t> _last_activity_ms{0};
    std::atomic<int>     _active_streams{0};
    std::atomic<bool>    _model_loaded{true};
    std::mutex           _lifecycle_mutex;
    std::thread          _idle_thread;
    std::condition_variable _idle_cv;
    std::mutex              _idle_cv_mutex;

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
