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

class CudaBackend : public Backend {
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    size_t sched_capacity_ = 0;
    int device_id_ = 0;
    std::vector<PendingInit> pending_inits_;

    bool ensure_scheduler(ggml_cgraph * graph) {
        const size_t required = backend_graph_capacity(graph);
        // Kokoro builds different tensor shapes for each chunk. Recreate the
        // scheduler so ggml's graph allocator never reuses stale shape state.
        if (sched_ != nullptr) {
            ggml_backend_sched_free(sched_);
            sched_ = nullptr;
            sched_capacity_ = 0;
        }
        // ggml's scheduler requires a CPU backend as the last entry — it's
        // used as a fallback for ops not supported on the GPU. CUDA runs
        // the vast majority of nodes; the CPU backend only catches strays.
        ggml_backend_t backends[] = { backend_, cpu_backend_ };
        sched_ = ggml_backend_sched_new(backends, nullptr, 2, required, false, true);
        sched_capacity_ = sched_ == nullptr ? 0 : required;
        return sched_ != nullptr;
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
        if (sched_) {
            ggml_backend_sched_free(sched_);
            sched_ = nullptr;
        }
        if (backend_) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }
        if (cpu_backend_) {
            ggml_backend_free(cpu_backend_);
            cpu_backend_ = nullptr;
        }
    }

    bool valid() const {
        return backend_ != nullptr && cpu_backend_ != nullptr;
    }

    void sched_reset() override {
        if (sched_) {
            ggml_backend_sched_reset(sched_);
        }
    }

    // Pin nodes CUDA cannot execute (custom ops, integer types, etc.) to the
    // CPU backend so their callbacks run on host pointers. The scheduler
    // inserts host<->device copies around CPU islands automatically.
    void pin_cpu_fallbacks(ggml_cgraph * graph) {
        const int n_nodes = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == nullptr) continue;
            if (!ggml_backend_supports_op(backend_, node)) {
                ggml_backend_sched_set_tensor_backend(sched_, node, cpu_backend_);
            }
        }
    }

    bool sched_alloc_graph(ggml_cgraph * graph) override {
        if (!ensure_scheduler(graph)) {
            return false;
        }
        pin_cpu_fallbacks(graph);
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
        if (tensor != nullptr) {
            PendingInit init;
            init.tensor = tensor;
            init.zero = true;
            pending_inits_.push_back(std::move(init));
        }
    }

    void queue_f32_tensor(ggml_tensor * tensor, float value) override {
        if (tensor == nullptr) {
            return;
        }
        const size_t n = static_cast<size_t>(ggml_nelements(tensor));
        PendingInit init;
        init.tensor = tensor;
        init.bytes.resize(n * sizeof(float));
        std::fill(reinterpret_cast<float *>(init.bytes.data()),
                  reinterpret_cast<float *>(init.bytes.data()) + n, value);
        pending_inits_.push_back(std::move(init));
    }

    bool apply_pending_inits() override {
        for (const PendingInit & init : pending_inits_) {
            if (init.tensor == nullptr) {
                continue;
            }
            if (init.zero) {
                ggml_backend_tensor_memset(init.tensor, 0, 0, ggml_nbytes(init.tensor));
            } else if (!init.bytes.empty()) {
                ggml_backend_tensor_set(init.tensor, init.bytes.data(), 0, init.bytes.size());
            }
        }
        pending_inits_.clear();
        return true;
    }

    ggml_status compute(ggml_context * ctx, ggml_cgraph * graph) override {
        (void)ctx;
        return ggml_backend_sched_graph_compute(sched_, graph);
    }

    void tensor_set(ggml_tensor * t, const void * data,
                    size_t offset, size_t size) override {
        if (t != nullptr && data != nullptr && size > 0) {
            GGML_ASSERT((t->buffer != nullptr || (t->view_src != nullptr && t->view_src->buffer != nullptr)) &&
                        "tensor buffer not set before backend tensor_set");
            ggml_backend_tensor_set(t, data, offset, size);
        }
    }

    void tensor_get(ggml_tensor * t, void * data,
                    size_t offset, size_t size) override {
        if (t != nullptr && data != nullptr && size > 0) {
            GGML_ASSERT((t->buffer != nullptr || (t->view_src != nullptr && t->view_src->buffer != nullptr)) &&
                        "tensor buffer not set before backend tensor_get");
            ggml_backend_tensor_get(t, data, offset, size);
        }
    }

    ggml_backend_buffer_type_t weight_buffer_type() const override {
        // Weights must live in host-accessible memory: a few Kokoro custom
        // ops (fused LSTM, Snake1D, vocoder ConvTranspose) capture raw
        // tensor->data pointers at graph build time and dereference them
        // from CPU callbacks. Pinned host memory keeps these pointers
        // dereferenceable while still allowing fast async DMA from the
        // CUDA backend when running matmul/conv on the GPU.
        ggml_backend_buffer_type_t pinned = ggml_backend_cuda_host_buffer_type();
        if (pinned != nullptr) {
            return pinned;
        }
        return ggml_backend_get_default_buffer_type(cpu_backend_);
    }

    const char * label() const override {
        return "CUDA (GPU)";
    }

    size_t generation_context_bytes(int64_t total_frames,
                                    int64_t n_tokens) const override {
        const size_t frames = static_cast<size_t>(std::max<int64_t>(1, total_frames));
        const size_t tokens = static_cast<size_t>(std::max<int64_t>(1, n_tokens));
        const size_t mask_bytes  = frames * tokens * sizeof(float);
        const size_t pred_bytes  = 512ULL * tokens * sizeof(float);
        const size_t lstm_work   = frames * 4096ULL * sizeof(float);
        const size_t overhead    = backend_mib(64);
        return mask_bytes + pred_bytes + lstm_work + overhead;
    }

    size_t generator_context_bytes(int64_t decoder_len) const override {
        const size_t dec        = static_cast<size_t>(std::max<int64_t>(1, decoder_len));
        const size_t out_frames = dec * 60;
        const size_t decoder_tensor  = 64ULL * dec * sizeof(float);
        const size_t harmonic_tensor = 22ULL * (out_frames + 1) * sizeof(float);
        const size_t post_tensor     = 1ULL * out_frames * sizeof(float);
        const size_t conv_work = 256ULL * out_frames * sizeof(float) * 3;
        const size_t overhead  = backend_mib(16);
        return decoder_tensor + harmonic_tensor + post_tensor + conv_work + overhead;
    }

    size_t frontend_context_bytes() const override {
        return backend_mib(16);
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
