#include "error.h"

namespace kokopop {
namespace {
thread_local std::string g_last_error;
}

void set_error(std::string message) {
    g_last_error = std::move(message);
}

const char * last_error() {
    return g_last_error.c_str();
}

} // namespace kokopop

