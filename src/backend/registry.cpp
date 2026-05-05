#include "backend.h"
#include "metal.h"

#include <memory>
#include <string>

namespace kokopop {

// Factory: create the requested backend.
// KOKOPOP_BACKEND_AUTO  → try Metal, fall back to CPU
// KOKOPOP_BACKEND_CPU   → CPU
// KOKOPOP_BACKEND_METAL → Metal, fail if unavailable
std::unique_ptr<Backend> create_backend(
    [[maybe_unused]] int32_t requested, int32_t n_threads, [[maybe_unused]] std::string & error) {

#ifdef KOKOPOP_HAS_METAL
    if (requested == 0 || requested == 2) {
        auto metal = create_metal_backend(n_threads);
        if (metal) {
            return metal;
        }
        if (requested == 2) {
            error = "Metal backend requested but not available";
            return nullptr;
        }
    }
#endif

    return create_cpu_backend(n_threads);
}

} // namespace kokopop
