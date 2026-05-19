#include "backend.h"
#include "metal.h"
#ifdef KOKOPOP_HAS_CUDA
#include "cuda.h"
#endif
#ifdef KOKOPOP_HAS_VULKAN
#include "vulkan.h"
#endif

#include <memory>
#include <string>

namespace kokopop {

// Factory: create the requested backend.
// KOKOPOP_BACKEND_AUTO  → try CUDA, then Metal, then Vulkan, fall back to CPU
// KOKOPOP_BACKEND_CPU   → CPU
// KOKOPOP_BACKEND_METAL → Metal, fail if unavailable
// KOKOPOP_BACKEND_CUDA  → CUDA, fail if unavailable
// KOKOPOP_BACKEND_VULKAN → Vulkan, fail if unavailable
std::unique_ptr<Backend> create_backend(
    [[maybe_unused]] int32_t requested, int32_t n_threads, [[maybe_unused]] std::string & error) {

#ifdef KOKOPOP_HAS_CUDA
    if (requested == KOKOPOP_BACKEND_AUTO || requested == KOKOPOP_BACKEND_CUDA) {
        auto cuda = create_cuda_backend(n_threads);
        if (cuda) {
            return cuda;
        }
        if (requested == KOKOPOP_BACKEND_CUDA) {
            error = "CUDA backend requested but not available";
            return nullptr;
        }
    }
#else
    if (requested == KOKOPOP_BACKEND_CUDA) {
        error = "CUDA backend requested but not compiled in";
        return nullptr;
    }
#endif

#ifdef KOKOPOP_HAS_METAL
    if (requested == KOKOPOP_BACKEND_AUTO || requested == KOKOPOP_BACKEND_METAL) {
        auto metal = create_metal_backend(n_threads);
        if (metal) {
            return metal;
        }
        if (requested == KOKOPOP_BACKEND_METAL) {
            error = "Metal backend requested but not available";
            return nullptr;
        }
    }
#else
    if (requested == KOKOPOP_BACKEND_METAL) {
        error = "Metal backend requested but not compiled in";
        return nullptr;
    }
#endif

#ifdef KOKOPOP_HAS_VULKAN
    if (requested == KOKOPOP_BACKEND_AUTO || requested == KOKOPOP_BACKEND_VULKAN) {
        auto vulkan = create_vulkan_backend(n_threads);
        if (vulkan) {
            return vulkan;
        }
        if (requested == KOKOPOP_BACKEND_VULKAN) {
            error = "Vulkan backend requested but not available";
            return nullptr;
        }
    }
#else
    if (requested == KOKOPOP_BACKEND_VULKAN) {
        error = "Vulkan backend requested but not compiled in";
        return nullptr;
    }
#endif

    return create_cpu_backend(n_threads);
}

} // namespace kokopop
