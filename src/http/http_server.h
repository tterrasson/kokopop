#pragma once

#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>

// Forward declare yyjson types (we include yyjson.h in the .cpp)
struct yyjson_doc;

namespace kokopop {

// ---------------------------------------------------------------------------
// HTTP request / response types
// ---------------------------------------------------------------------------

struct HttpRequest {
    std::string method;       // GET, POST, etc.
    std::string path;         // /tts, /health, etc.
    std::string query_string; // everything after '?'
    std::map<std::string, std::string> headers;
    std::vector<char> body;
    int content_length;       // -1 if not set
};

/// Build a simple JSON error object: {"error": "msg"}
std::string json_error(const std::string & msg);

struct HttpResponse {
    int status_code = 200;
    std::string status_text = "OK";
    std::map<std::string, std::string> headers;
    std::vector<char> body;

    // Convenience: set content-type
    void set_content_type(const std::string & type);

    // Convenience: JSON body from yyjson_doc
    void set_json(yyjson_doc * doc);

    // Convenience: JSON body from a pre-built JSON string
    void set_json_string(const std::string & json_str);
};

// ---------------------------------------------------------------------------
// Request handler signature
//   Returns true to keep connection alive (HTTP keep-alive), false to close
// ---------------------------------------------------------------------------
using RequestHandler = std::function<bool(HttpRequest & req, HttpResponse & res)>;

// ---------------------------------------------------------------------------
// Minimal HTTP/1.1 server (single-threaded accept loop, one thread per conn)
// ---------------------------------------------------------------------------

class HttpServer {
public:
    HttpServer();
    ~HttpServer();

    // Delete copy
    HttpServer(const HttpServer &) = delete;
    HttpServer & operator=(const HttpServer &) = delete;

    // Register a handler for a specific path
    void route(const std::string & path, RequestHandler handler);

    // Register a default handler for unmatched paths (optional)
    void route_default(RequestHandler handler);

    // Start serving (blocking). Call stop() to shut down.
    bool start(const std::string & addr, int port);

    // Signal the server to stop (can be called from another thread)
    void stop();

    // Wait until the server has stopped
    void join();

    // Check if server is running
    bool is_running() const { return _running.load(); }

private:
    void _accept_loop();
    void _handle_connection(int client_fd);
    HttpRequest _parse_request(int fd);
    bool _send_response(int fd, const HttpResponse & res, bool close_conn);

    std::map<std::string, RequestHandler> _routes;
    RequestHandler _default_handler;
    int _server_fd = -1;
    std::string _addr;
    int _port = 0;

    std::atomic<bool> _running{false};
    std::thread _accept_thread;
    std::vector<std::thread> _conn_threads;
    std::mutex _threads_mutex;
};

} // namespace kokopop
