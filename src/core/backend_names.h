#pragma once

// Shared mapping between KOKOPOP_BACKEND_* values and their CLI spellings.
// Lives here so the four command line tools that parse --backend do not each
// carry their own if/else chain (which drifts every time a backend is added).

#include <cstdint>

#include "kokopop.h"

namespace kokopop {

// Parse a --backend value ("cpu", "metal", "cuda", "vulkan", "opencl", "webgpu").
// Returns false and leaves `out` untouched for an unknown name.
bool backend_from_name(const char * name, int32_t & out);

// Lowercase CLI spelling for a KOKOPOP_BACKEND_* value ("auto" for AUTO).
const char * backend_name(int32_t backend);

// Display name for a resolved backend ("CPU", "Metal", "CUDA", ...).
const char * backend_display_name(int32_t backend);

// Pipe-separated list of accepted --backend values, for usage and errors.
const char * backend_name_list();

} // namespace kokopop
