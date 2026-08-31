#pragma once

// Factory for OpenCL backend.
// Only declare when KOKOPOP_HAS_OPENCL is defined; callers check the flag
// before including this header or calling this function.
//
// Returns nullptr if OpenCL is not available at runtime.

#include "backend.h"

namespace kokopop {

#ifdef KOKOPOP_HAS_OPENCL
std::unique_ptr<Backend> create_opencl_backend(int32_t n_threads);
#else
inline std::unique_ptr<Backend> create_opencl_backend(int32_t) { return nullptr; }
#endif

} // namespace kokopop
