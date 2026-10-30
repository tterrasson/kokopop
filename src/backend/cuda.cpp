// This translation unit is only compiled when the CUDA backend is enabled
// (see CMakeLists.txt). Define the flag defensively so that tooling which
// indexes this file without the target's compile definitions still sees the
// declaration in cuda.h rather than the inline no-op stub.
#ifndef KOKOPOP_HAS_CUDA
#define KOKOPOP_HAS_CUDA
#endif

#include "backend.h"
#include "cuda.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>

namespace kokopop {
namespace {

class CudaBackend final : public SchedBackendBase {
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_t backends_[2] = {nullptr, nullptr};
    int device_id_ = 0;

public:
    explicit CudaBackend(int32_t n_threads) {
        // ggml's scheduler expects the CPU backend last: it is used for ops
        // not supported by CUDA and for cross-backend fallback/copies.
        sched_op_offload_ = true;
        sched_split_graph_ = true;

        const char * env = std::getenv("KOKOPOP_CUDA_DEVICE");
        const int requested = env ? std::atoi(env) : 0;
        const int count = ggml_backend_cuda_get_device_count();
        if (count <= 0) {
            return;
        }

        device_id_ = (requested >= 0 && requested < count) ? requested : 0;

        backend_ = ggml_backend_cuda_init(device_id_);
        if (backend_ == nullptr) {
            return;
        }

        cpu_backend_ = ggml_backend_cpu_init();
        if (cpu_backend_ != nullptr) {
            ggml_backend_cpu_set_n_threads(cpu_backend_, std::max<int32_t>(1, n_threads));
        }
    }

    ~CudaBackend() override {
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

        // Weights live in CUDA VRAM. Fused LSTM recurrence uses separate
        // host-side w_hh/b_hh caches, so model weights can remain device-side.
        return ggml_backend_cuda_buffer_type(device_id_);
    }

    const char * label() const override {
        return "CUDA (GPU)";
    }

    int32_t type() const override {
        return KOKOPOP_BACKEND_CUDA;
    }
};

} // anonymous namespace

std::unique_ptr<Backend> create_cuda_backend(int32_t n_threads) {
    auto backend = std::make_unique<CudaBackend>(n_threads);
    if (!backend->valid()) {
        return nullptr;
    }
    return backend;
}

} // namespace kokopop
