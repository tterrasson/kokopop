#include "backend.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

namespace kokopop {
namespace {

class CpuBackend final : public SchedBackendBase {
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t backends_[1] = {nullptr};

public:
    explicit CpuBackend(int32_t n_threads) {
        // CPU uses ggml's graph allocator directly; recreate per graph so
        // chunk-dependent shapes cannot reuse stale allocation state on x86.
        cache_scheduler_ = false;
        backend_ = ggml_backend_cpu_init();
        if (backend_ != nullptr) {
            ggml_backend_cpu_set_n_threads(backend_, std::max<int32_t>(1, n_threads));
            backends_[0] = backend_;
        }
    }

    ~CpuBackend() override {
        free_scheduler();
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }
    }

    bool valid() const {
        return backend_ != nullptr;
    }

    ggml_backend_t * sched_backends(size_t & n) override {
        GGML_ASSERT(backend_ != nullptr);
        n = 1;
        return backends_;
    }

    ggml_backend_buffer_type_t weight_buffer_type() const override {
        GGML_ASSERT(backend_ != nullptr);
        return ggml_backend_get_default_buffer_type(backend_);
    }
};

} // anonymous namespace

std::unique_ptr<Backend> create_cpu_backend(int32_t n_threads) {
    auto backend = std::make_unique<CpuBackend>(n_threads);
    if (!backend->valid()) {
        return nullptr;
    }
    return backend;
}

} // namespace kokopop
