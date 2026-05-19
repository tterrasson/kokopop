#pragma once

// Factory for Vulkan backend.
// Only declare when KOKOPOP_HAS_VULKAN is defined; callers check the flag
// before including this header or calling this function.
//
// Returns nullptr if Vulkan is not available at runtime.

#include "backend.h"

namespace kokopop {

#ifdef KOKOPOP_HAS_VULKAN
std::unique_ptr<Backend> create_vulkan_backend(int32_t n_threads);
#else
inline std::unique_ptr<Backend> create_vulkan_backend(int32_t) { return nullptr; }
#endif

} // namespace kokopop
