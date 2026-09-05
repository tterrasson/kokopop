#pragma once

// Factory for WEBGPU backend.
// Only declare when KOKOPOP_HAS_WEBGPU is defined; callers check the flag
// before including this header or calling this function.
//
// Returns nullptr if WEBGPU is not available at runtime (no device, init failed).

#include "backend.h"

namespace kokopop {

#ifdef KOKOPOP_HAS_WEBGPU
std::unique_ptr<Backend> create_webgpu_backend(int32_t n_threads);
#else
inline std::unique_ptr<Backend> create_webgpu_backend(int32_t) { return nullptr; }
#endif

} // namespace kokopop
