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

class CpuBackend : public Backend {
    ggml_backend_t backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    size_t sched_capacity_ = 0;
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
        ggml_backend_t backends[] = { backend_ };
        sched_ = ggml_backend_sched_new(backends, nullptr, 1, required, false, true);
        sched_capacity_ = sched_ == nullptr ? 0 : required;
        return sched_ != nullptr;
    }

public:
    explicit CpuBackend(int32_t n_threads) {
        backend_ = ggml_backend_cpu_init();
        if (backend_ != nullptr) {
            ggml_backend_cpu_set_n_threads(backend_, std::max<int32_t>(1, n_threads));
        }
    }

    ~CpuBackend() override {
        if (sched_) {
            ggml_backend_sched_free(sched_);
            sched_ = nullptr;
        }
        if (backend_) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }
    }

    bool valid() const {
        return backend_ != nullptr;
    }

    void sched_reset() override {
        if (sched_) {
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
        return ggml_backend_get_default_buffer_type(backend_);
    }

    size_t generation_context_bytes(int64_t total_frames,
                                    int64_t n_tokens) const override {
        const size_t frames = static_cast<size_t>(std::max<int64_t>(1, total_frames));
        const size_t tokens = static_cast<size_t>(std::max<int64_t>(1, n_tokens));
        // Dominant tensors: duration_mask [total_frames x n_tokens] F32,
        // duration_pred [512 x n_tokens] F32, LSTM intermediates per step.
        // Bidirectional LSTM: 2 dirs × hidden(512) × 4 gates × n_steps + activations.
        // AdaIN ResBlocks, text_encoder LSTM, conv1d intermediates, graph node metadata.
        // The scheduler keeps all intermediates alive, so estimate must cover peak usage.
        const size_t mask_bytes  = frames * tokens * sizeof(float);
        const size_t pred_bytes  = 512ULL * tokens * sizeof(float);
        // LSTM work: bidirectional gates + hidden/cell states + concat outputs.
        // 4096 floats/frame is sufficient (matching Metal backend's estimate);
        // 8192 was over-provisioned and wasted memory with no measurable benefit.
        const size_t lstm_work   = frames * 4096ULL * sizeof(float);
        const size_t overhead    = backend_mib(64);  // graph nodes, views, ScratchArena
        return mask_bytes + pred_bytes + lstm_work + overhead;
    }

    size_t generator_context_bytes(int64_t decoder_len) const override {
        // decoder_len = number of time steps in the decoder input tensor.
        // Output audio length = decoder_len * 60 (upsampling factor).
        // Dominant tensors:
        //   - decoder input:  [decoder_channels x decoder_len] F32
        //   - harmonic STFT:  [22 x (decoder_len*60 + 1)] F32
        //   - output post:    [1 x decoder_len*60] F32
        //   - intermediate conv activations (upsampling stages)
        const size_t dec        = static_cast<size_t>(std::max<int64_t>(1, decoder_len));
        const size_t out_frames = dec * 60;
        const size_t decoder_tensor  = 64ULL * dec * sizeof(float);
        const size_t harmonic_tensor = 22ULL * (out_frames + 1) * sizeof(float);
        const size_t post_tensor     = 1ULL * out_frames * sizeof(float);
        // Conv intermediates: 3 resblock stages with dilated convs,
        // each channel ~256-512, length grows through upsampling.
        const size_t conv_work = 256ULL * out_frames * sizeof(float) * 3;
        const size_t overhead  = backend_mib(16); // graph nodes, views, ScratchArena base
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
