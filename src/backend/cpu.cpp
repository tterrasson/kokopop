#include "backend.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

namespace kokopop {
namespace {

class CpuBackend final : public Backend {
    ggml_backend_t backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    std::vector<PendingInit> pending_inits_;

    bool ensure_scheduler(ggml_cgraph * graph) {
        GGML_ASSERT(graph != nullptr);
        GGML_ASSERT(backend_ != nullptr);

        const size_t required = std::max<size_t>(
            GGML_DEFAULT_GRAPH_SIZE,
            backend_graph_capacity(graph));

        // CPU uses ggml's graph allocator directly; recreate per graph so
        // chunk-dependent shapes cannot reuse stale allocation state on x86.
        if (sched_ != nullptr) {
            ggml_backend_sched_free(sched_);
            sched_ = nullptr;
        }

        ggml_backend_t backends[] = { backend_ };
        sched_ = ggml_backend_sched_new(
            backends,
            nullptr,
            1,
            required,
            false,
            true);

        return sched_ != nullptr;
    }

    static bool tensor_has_backend_buffer(const ggml_tensor * tensor) {
        return tensor != nullptr &&
               (tensor->buffer != nullptr ||
                (tensor->view_src != nullptr && tensor->view_src->buffer != nullptr));
    }

public:
    explicit CpuBackend(int32_t n_threads) {
        backend_ = ggml_backend_cpu_init();
        if (backend_ != nullptr) {
            ggml_backend_cpu_set_n_threads(backend_, std::max<int32_t>(1, n_threads));
        }
    }

    ~CpuBackend() override {
        if (sched_ != nullptr) {
            ggml_backend_sched_free(sched_);
            sched_ = nullptr;
        }

        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }
    }

    bool valid() const {
        return backend_ != nullptr;
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
        return ggml_backend_get_default_buffer_type(backend_);
    }

    size_t generation_context_bytes(int64_t total_frames, int64_t n_tokens) const override {
        const size_t frames = static_cast<size_t>(std::max<int64_t>(1, total_frames));
        const size_t tokens = static_cast<size_t>(std::max<int64_t>(1, n_tokens));

        const size_t mask_bytes = frames * tokens * sizeof(float);
        const size_t pred_bytes = 512ULL * tokens * sizeof(float);
        const size_t lstm_work  = frames * 4096ULL * sizeof(float);
        const size_t overhead   = backend_mib(64);

        return mask_bytes + pred_bytes + lstm_work + overhead;
    }

    size_t generator_context_bytes(int64_t decoder_len) const override {
        const size_t dec = static_cast<size_t>(std::max<int64_t>(1, decoder_len));
        const size_t out_frames = dec * 60;

        const size_t decoder_tensor  = 64ULL * dec * sizeof(float);
        const size_t harmonic_tensor = 22ULL * (out_frames + 1) * sizeof(float);
        const size_t post_tensor     = out_frames * sizeof(float);
        const size_t conv_work       = 256ULL * out_frames * sizeof(float) * 3;
        const size_t overhead        = backend_mib(16);

        return decoder_tensor + harmonic_tensor + post_tensor + conv_work + overhead;
    }

    size_t frontend_context_bytes() const override {
        return backend_mib(16);
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
