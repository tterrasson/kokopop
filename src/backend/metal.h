#pragma once

// Factory for Metal backend.
// Only declare when KOKOPOP_HAS_METAL is defined; callers check the flag
// before including this header or calling this function.
//
// Returns nullptr if Metal is not available at runtime.

#include "backend.h"

namespace kokopop {

#ifdef KOKOPOP_HAS_METAL
std::unique_ptr<Backend> create_metal_backend(int32_t n_threads);
#else
inline std::unique_ptr<Backend> create_metal_backend(int32_t) { return nullptr; }
#endif

} // namespace kokopop
