// This translation unit is only compiled when the WEBGPU backend is enabled
// (see CMakeLists.txt). Define the flag defensively so that tooling which
// indexes this file without the target's compile definitions still sees the
// declaration in webgpu.h rather than the inline no-op stub.
#ifndef KOKOPOP_HAS_WEBGPU
#define KOKOPOP_HAS_WEBGPU
#endif

#include "backend.h"
#include "webgpu.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-webgpu.h>

namespace kokopop {
namespace {

class WebGPUBackend final : public SchedBackendBase {
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_t backends_[2] = {nullptr, nullptr};

public:
    explicit WebGPUBackend(int32_t n_threads) {
        // ggml's scheduler expects the CPU backend last: it is used for ops
        // not supported by WEBGPU and for cross-backend fallback/copies.
        sched_op_offload_ = true;
        sched_split_graph_ = true;

        backend_ = ggml_backend_webgpu_init();
        if (backend_ == nullptr) {
            return;
        }

        cpu_backend_ = ggml_backend_cpu_init();
        if (cpu_backend_ != nullptr) {
            ggml_backend_cpu_set_n_threads(cpu_backend_, std::max<int32_t>(1, n_threads));
        }
    }

    ~WebGPUBackend() override {
        free_scheduler();
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }

        if (cpu_backend_ != nullptr) {
            ggml_backend_free(cpu_backend_);
            cpu_backend_ = nullptr;
        }
    }

    bool valid() const {
        return backend_ != nullptr && cpu_backend_ != nullptr;
    }

    ggml_backend_t * sched_backends(size_t & n) override {
        GGML_ASSERT(backend_ != nullptr);
        GGML_ASSERT(cpu_backend_ != nullptr);
        backends_[0] = backend_;
        backends_[1] = cpu_backend_;
        n = 2;
        return backends_;
    }

    ggml_backend_t cpu_fallback_backend() const override {
        return cpu_backend_;
    }

    ggml_backend_buffer_type_t weight_buffer_type() const override {
        GGML_ASSERT(backend_ != nullptr);

        // Weights live in WEBGPU VRAM. Fused LSTM recurrence uses separate
        // host-side w_hh/b_hh caches, so model weights can remain device-side.
        return ggml_backend_get_default_buffer_type(backend_);
    }

    const char * label() const override {
        return "WebGPU (GPU)";
    }

    int32_t type() const override {
        return KOKOPOP_BACKEND_WEBGPU;
    }
};

} // anonymous namespace

std::unique_ptr<Backend> create_webgpu_backend(int32_t n_threads) {
    auto backend = std::make_unique<WebGPUBackend>(n_threads);
    if (!backend->valid()) {
        return nullptr;
    }
    return backend;
}

} // namespace kokopop
