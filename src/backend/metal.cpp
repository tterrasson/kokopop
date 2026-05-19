#include "metal.h"
#include "metal_lstm.h"
#include "metal_stft.h"
#include "metal_vocoder.h"

#include "core/constants.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-metal.h>

namespace kokopop {
namespace {

static int label_level(const char * label) {
    if (label == nullptr) {
        return 99;
    }
    if (std::strcmp(label, "frontend") == 0) {
        return 0;
    }
    if (std::strcmp(label, "generation") == 0) {
        return 1;
    }
    if (std::strcmp(label, "generator") == 0) {
        return 2;
    }
    return 99;
}

class MetalBackend final : public Backend {
    ggml_backend_t metal_backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    size_t sched_capacity_ = 0;
    uint64_t sched_signature_ = 0;

    MetalLstmKernelState * lstm_kernel_ = nullptr;
    MetalStftState * stft_kernel_ = nullptr;
    MetalVocoderState * vocoder_kernel_ = nullptr;

    const char * active_label_ = nullptr;
    std::vector<PendingInit> pending_inits_;

    // Cached env vars. Read once at construction.
    const char * env_run_only_  = nullptr;   // KOKOPOP_METAL_RUN_ONLY=frontend|generation|generator

    bool env_log_sched_ = false;             // KOKOPOP_METAL_LOG_SCHED=1
    bool env_op_trace_ = false;              // KOKOPOP_METAL_OP_TRACE=1

    bool ensure_scheduler(ggml_cgraph * graph) {
        GGML_ASSERT(graph != nullptr);
        GGML_ASSERT(metal_backend_ != nullptr);
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

        // CPU backend must be last. It handles unsupported ops and fallback copies.
        ggml_backend_t backends[] = { metal_backend_, cpu_backend_ };

        // op_offload=false: ops run on the backend where their inputs live.
        // With weights in Metal shared memory, matmuls and attention naturally
        // stay on Metal. Custom ops (LSTM, vocoder) stay on CPU without forcing
        // extra cross-backend copies.
        sched_ = ggml_backend_sched_new(
            backends,
            nullptr,
            2,
            required,
            false,
            false);

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

    void log_sched_sizes(const char * stage) const {
        if (sched_ == nullptr || !env_log_sched_) {
            return;
        }

        const int n_splits = ggml_backend_sched_get_n_splits(sched_);
        const size_t metal_bytes = ggml_backend_sched_get_buffer_size(sched_, metal_backend_);
        const size_t cpu_bytes = ggml_backend_sched_get_buffer_size(sched_, cpu_backend_);

        std::fprintf(
            stderr,
            "[metal-sched %s/%s] splits=%d metal_buf=%.1f MiB cpu_buf=%.1f MiB\n",
            active_label_ != nullptr ? active_label_ : "?",
            stage,
            n_splits,
            metal_bytes / 1048576.0,
            cpu_bytes / 1048576.0);
    }

    static bool eval_trace_cb(ggml_tensor * tensor, bool ask, void * user_data) {
        if (ask) {
            return true;
        }

        auto * sched = static_cast<ggml_backend_sched_t>(user_data);

        const char * op_name = tensor == nullptr
            ? "?"
            : (tensor->op == GGML_OP_UNARY
                ? ggml_unary_op_name(ggml_get_unary_op(tensor))
                : ggml_op_name(tensor->op));

        ggml_backend_t backend = (sched != nullptr && tensor != nullptr)
            ? ggml_backend_sched_get_tensor_backend(sched, tensor)
            : nullptr;

        const char * backend_name = backend != nullptr ? ggml_backend_name(backend) : "?";

        std::fprintf(
            stderr,
            "[op %-5s] %-20s %-32s shape=[%lld,%lld,%lld,%lld]\n",
            backend_name,
            op_name,
            tensor != nullptr && tensor->name[0] ? tensor->name : "",
            static_cast<long long>(tensor != nullptr ? tensor->ne[0] : 0),
            static_cast<long long>(tensor != nullptr ? tensor->ne[1] : 0),
            static_cast<long long>(tensor != nullptr ? tensor->ne[2] : 0),
            static_cast<long long>(tensor != nullptr ? tensor->ne[3] : 0));

        std::fflush(stderr);
        return true;
    }

public:
    explicit MetalBackend(int32_t n_threads) {
        env_run_only_ = std::getenv("KOKOPOP_METAL_RUN_ONLY");

        env_log_sched_ = std::getenv("KOKOPOP_METAL_LOG_SCHED") != nullptr;
        env_op_trace_ = std::getenv("KOKOPOP_METAL_OP_TRACE") != nullptr;
        cpu_backend_ = ggml_backend_cpu_init();
        if (cpu_backend_ != nullptr) {
            ggml_backend_cpu_set_n_threads(cpu_backend_, std::max<int32_t>(1, n_threads));
        }

        // Hybrid graphs and custom kernels (LSTM, vocoder, STFT) rely on the
        // scheduler splitting ops across backends; ggml Metal fusion can interfere.
        setenv("GGML_METAL_FUSION_DISABLE", "1", 0);

        metal_backend_ = ggml_backend_metal_init();
        if (metal_backend_ == nullptr) {
            if (cpu_backend_ != nullptr) {
                ggml_backend_free(cpu_backend_);
                cpu_backend_ = nullptr;
            }
            return;
        }

        // Metal LSTM kernel enabled by default. Despite the recurrence running
        // on a single SM, it benchmarks ~4% faster end-to-end than the CPU
        // NEON path for this model (H=256, ~6-12 directions/chunk). Disable
        // with KOKOPOP_METAL_LSTM=0 to fall back to CPU LSTM.
        const char * env_lstm = std::getenv("KOKOPOP_METAL_LSTM");
        const bool enable_metal_lstm = (env_lstm == nullptr) ||
                                       (env_lstm[0] != '0' || env_lstm[1] != '\0');
        if (enable_metal_lstm) {
            std::fprintf(stderr, "[metal] creating lstm kernel\n");
            lstm_kernel_ = metal_lstm_create();
        }
        std::fprintf(stderr, "[metal] creating stft kernel\n");
        stft_kernel_    = metal_stft_create(KOKOPOP_STFT_N, KOKOPOP_STFT_HOP);
        std::fprintf(stderr, "[metal] creating vocoder kernel\n");
        vocoder_kernel_ = metal_vocoder_create();
        std::fprintf(stderr, "[metal] backend ready (lstm=%s stft=%s vocoder=%s)\n",
                     lstm_kernel_ ? "ok" : "null",
                     stft_kernel_ ? "ok" : "null",
                     vocoder_kernel_ ? "ok" : "null");
    }

    ~MetalBackend() override {
        if (sched_ != nullptr) {
            ggml_backend_sched_free(sched_);
            sched_ = nullptr;
            sched_capacity_ = 0;
            sched_signature_ = 0;
        }

        if (lstm_kernel_ != nullptr) {
            metal_lstm_destroy(lstm_kernel_);
            lstm_kernel_ = nullptr;
        }

        if (stft_kernel_ != nullptr) {
            metal_stft_destroy(stft_kernel_);
            stft_kernel_ = nullptr;
        }

        if (vocoder_kernel_ != nullptr) {
            metal_vocoder_destroy(vocoder_kernel_);
            vocoder_kernel_ = nullptr;
        }

        if (metal_backend_ != nullptr) {
            ggml_backend_free(metal_backend_);
            metal_backend_ = nullptr;
        }

        if (cpu_backend_ != nullptr) {
            ggml_backend_free(cpu_backend_);
            cpu_backend_ = nullptr;
        }
    }

    bool valid() const {
        return metal_backend_ != nullptr && cpu_backend_ != nullptr;
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

        if (env_op_trace_) {
            ggml_backend_sched_set_eval_callback(sched_, eval_trace_cb, sched_);
        }

        // Do not call ggml_backend_sched_reserve() separately. alloc_graph()
        // handles allocation and split reservation for the current graph.
        const bool ok = ggml_backend_sched_alloc_graph(sched_, graph);
        if (ok) {
            log_sched_sizes("alloc");
        }

        return ok;
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

        if (env_run_only_ != nullptr &&
            label_level(active_label_) > label_level(env_run_only_)) {
            ggml_backend_sched_synchronize(sched_);
            log_sched_sizes("skip-compute");
            return GGML_STATUS_SUCCESS;
        }

        log_sched_sizes("compute-begin");
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

    const char * label() const override {
        return "Metal (GPU)";
    }

    int32_t type() const override {
        return KOKOPOP_BACKEND_METAL;
    }

    void set_active_label(const char * label) override {
        active_label_ = label;
    }

    void set_input_tokens(int n) override {
        (void) n;
    }

    void preload_lstm_whh(const std::string & key, const float * w_hh_f32, int H, int four_H) override {
        if (lstm_kernel_ != nullptr) {
            metal_lstm_preload_whh(lstm_kernel_, key.c_str(), w_hh_f32, H, four_H);
        }
    }

    void * metal_lstm_kernel() const override {
        return lstm_kernel_;
    }

    void * metal_stft_kernel() const override {
        return stft_kernel_;
    }

    void * metal_vocoder_kernel() const override {
        return vocoder_kernel_;
    }

    ggml_backend_buffer_type_t weight_buffer_type() const override {
        GGML_ASSERT(cpu_backend_ != nullptr);
        GGML_ASSERT(metal_backend_ != nullptr);

        // KOKOPOP_METAL_WEIGHTS=1 places weights in the Metal buffer so that
        // matmuls and friends actually run on GPU (ggml-metal scheduler picks
        // the backend of the op's inputs). Off by default: historically the
        // LSTM MAP_CUSTOM2 ops forced CPU↔Metal sync points at every LSTM
        // layer, which made Metal-resident weights a net loss. With Metal
        // LSTM disabled (default since opt A), the frontend LSTMs run on CPU
        // but the splits are still expensive, so leave this opt-in until
        // benchmarked end-to-end.
        if (std::getenv("KOKOPOP_METAL_WEIGHTS") != nullptr) {
            return ggml_backend_get_default_buffer_type(metal_backend_);
        }
        return ggml_backend_get_default_buffer_type(cpu_backend_);
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
        return backend_mib(64);
    }
};

} // anonymous namespace

std::unique_ptr<Backend> create_metal_backend(int32_t n_threads) {
    auto backend = std::make_unique<MetalBackend>(n_threads);
    if (!backend->valid()) {
        return nullptr;
    }
    return backend;
}

} // namespace kokopop
