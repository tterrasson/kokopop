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
#include <cstring>
#include <utility>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>

namespace kokopop {
namespace {

class CudaBackend final : public Backend {
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    size_t sched_capacity_ = 0;
    uint64_t sched_signature_ = 0;
    int device_id_ = 0;
    std::vector<PendingInit> pending_inits_;

    bool ensure_scheduler(ggml_cgraph * graph) {
        GGML_ASSERT(graph != nullptr);
        GGML_ASSERT(backend_ != nullptr);
        GGML_ASSERT(cpu_backend_ != nullptr);

        const size_t required = std::max<size_t>(
            GGML_DEFAULT_GRAPH_SIZE,
            backend_graph_capacity(graph));
        const uint64_t signature = backend_graph_signature(graph);

        if (sched_ != nullptr && (sched_signature_ != signature || sched_capacity_ < required)) {
            ggml_backend_sched_free(sched_);
            sched_ = nullptr;
            sched_capacity_ = 0;
            sched_signature_ = 0;
        }

        if (sched_ != nullptr) {
            return true;
        }

        // ggml's scheduler expects the CPU backend last: it is used for ops not
        // supported by CUDA and for cross-backend fallback/copies.
        ggml_backend_t backends[] = { backend_, cpu_backend_ };

        sched_ = ggml_backend_sched_new(
            backends,
            nullptr,
            2,
            required,
            true,
            true);

        if (sched_ != nullptr) {
            sched_capacity_ = required;
            sched_signature_ = signature;
        }
        return sched_ != nullptr;
    }

    static bool tensor_has_backend_buffer(const ggml_tensor * tensor) {
        return tensor != nullptr &&
               (tensor->buffer != nullptr ||
                (tensor->view_src != nullptr && tensor->view_src->buffer != nullptr));
    }

public:
    explicit CudaBackend(int32_t n_threads) {
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
        if (sched_ != nullptr) {
            ggml_backend_sched_free(sched_);
            sched_ = nullptr;
            sched_capacity_ = 0;
            sched_signature_ = 0;
        }

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

    void sched_reset() override {
        if (sched_ != nullptr) {
            ggml_backend_sched_reset(sched_);
        }
    }

    bool sched_alloc_graph(ggml_cgraph * graph) override {
        if (!ensure_scheduler(graph)) {
            return false;
        }

        return ggml_backend_sched_alloc_graph(sched_, graph);
    }

    void clear_pending_inits() override {
        pending_inits_.clear();
        pending_inits_.reserve(64);
    }

    void queue_tensor_data(ggml_tensor * tensor, const void * data, size_t size) override {
        if (tensor == nullptr || data == nullptr || size == 0) {
            return;
        }

        PendingInit init;
        init.tensor = tensor;
        init.bytes.resize(size);
        std::memcpy(init.bytes.data(), data, size);
        pending_inits_.push_back(std::move(init));
    }

    void queue_zero_tensor(ggml_tensor * tensor) override {
        if (tensor == nullptr) {
            return;
        }

        PendingInit init;
        init.tensor = tensor;
        init.zero = true;
        pending_inits_.push_back(std::move(init));
    }

    void queue_f32_tensor(ggml_tensor * tensor, float value) override {
        if (tensor == nullptr) {
            return;
        }

        GGML_ASSERT(tensor->type == GGML_TYPE_F32 && "queue_f32_tensor expects an F32 tensor");

        const size_t n = static_cast<size_t>(ggml_nelements(tensor));

        PendingInit init;
        init.tensor = tensor;
        init.bytes.resize(n * sizeof(float));

        auto * dst = reinterpret_cast<float *>(init.bytes.data());
        std::fill(dst, dst + n, value);

        pending_inits_.push_back(std::move(init));
    }

    bool apply_pending_inits() override {
        for (const PendingInit & init : pending_inits_) {
            if (init.tensor == nullptr) {
                continue;
            }

            GGML_ASSERT(tensor_has_backend_buffer(init.tensor) &&
                        "pending init tensor buffer not set before apply_pending_inits");

            const size_t nbytes = ggml_nbytes(init.tensor);

            if (init.zero) {
                ggml_backend_tensor_memset(init.tensor, 0, 0, nbytes);
                continue;
            }

            if (!init.bytes.empty()) {
                GGML_ASSERT(init.bytes.size() <= nbytes && "pending init exceeds tensor size");
                ggml_backend_tensor_set(init.tensor, init.bytes.data(), 0, init.bytes.size());
            }
        }

        pending_inits_.clear();
        return true;
    }

    ggml_status compute(ggml_context * ctx, ggml_cgraph * graph) override {
        (void) ctx;
        GGML_ASSERT(sched_ != nullptr && "compute called before sched_alloc_graph");
        GGML_ASSERT(graph != nullptr);

        return ggml_backend_sched_graph_compute(sched_, graph);
    }

    void tensor_set(ggml_tensor * tensor, const void * data, size_t offset, size_t size) override {
        if (tensor == nullptr || data == nullptr || size == 0) {
            return;
        }

        GGML_ASSERT(tensor_has_backend_buffer(tensor) &&
                    "tensor buffer not set before backend tensor_set");
        GGML_ASSERT(offset + size <= ggml_nbytes(tensor) &&
                    "backend tensor_set exceeds tensor size");

        ggml_backend_tensor_set(tensor, data, offset, size);
    }

    void tensor_get(ggml_tensor * tensor, void * data, size_t offset, size_t size) override {
        if (tensor == nullptr || data == nullptr || size == 0) {
            return;
        }

        GGML_ASSERT(tensor_has_backend_buffer(tensor) &&
                    "tensor buffer not set before backend tensor_get");
        GGML_ASSERT(offset + size <= ggml_nbytes(tensor) &&
                    "backend tensor_get exceeds tensor size");

        ggml_backend_tensor_get(tensor, data, offset, size);
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
