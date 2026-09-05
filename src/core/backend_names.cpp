#include "backend_names.h"

#include <cstring>
#include <string>

namespace kokopop {
namespace {

struct BackendName {
    const char * name;
    const char * display;
    int32_t      value;
};

// AUTO deliberately excluded: it is the default, not a spelling the tools accept.
constexpr BackendName k_backend_names[] = {
    { "cpu",    "CPU",    KOKOPOP_BACKEND_CPU    },
    { "metal",  "Metal",  KOKOPOP_BACKEND_METAL  },
    { "cuda",   "CUDA",   KOKOPOP_BACKEND_CUDA   },
    { "vulkan", "Vulkan", KOKOPOP_BACKEND_VULKAN },
    { "webgpu", "WebGPU", KOKOPOP_BACKEND_WEBGPU },
    { "opencl", "OpenCL", KOKOPOP_BACKEND_OPENCL },
};

} // anonymous namespace

bool backend_from_name(const char * name, int32_t & out) {
    if (name == nullptr) {
        return false;
    }
    for (const BackendName & entry : k_backend_names) {
        if (std::strcmp(name, entry.name) == 0) {
            out = entry.value;
            return true;
        }
    }
    return false;
}

const char * backend_name(int32_t backend) {
    for (const BackendName & entry : k_backend_names) {
        if (entry.value == backend) {
            return entry.name;
        }
    }
    return "auto";
}

const char * backend_display_name(int32_t backend) {
    for (const BackendName & entry : k_backend_names) {
        if (entry.value == backend) {
            return entry.display;
        }
    }
    return "CPU";
}

const char * backend_name_list() {
    static const std::string list = [] {
        std::string joined;
        for (const BackendName & entry : k_backend_names) {
            if (!joined.empty()) {
                joined += '|';
            }
            joined += entry.name;
        }
        return joined;
    }();
    return list.c_str();
}

} // namespace kokopop
