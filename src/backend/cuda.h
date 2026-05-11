#pragma once

// Factory for CUDA backend.
// Only declare when KOKOPOP_HAS_CUDA is defined; callers check the flag
// before including this header or calling this function.
//
// Returns nullptr if CUDA is not available at runtime (no device, init failed).

#include "backend.h"

namespace kokopop {

#ifdef KOKOPOP_HAS_CUDA
std::unique_ptr<Backend> create_cuda_backend(int32_t n_threads);
#else
inline std::unique_ptr<Backend> create_cuda_backend(int32_t) { return nullptr; }
#endif

} // namespace kokopop
