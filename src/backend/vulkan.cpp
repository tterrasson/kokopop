#include "backend.h"
#include "vulkan.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-vulkan.h>

namespace kokopop {
namespace {

class VulkanBackend final : public Backend {
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    size_t sched_capacity_ = 0;
    uint64_t sched_signature_ = 0;
    int device_id_ = 0;
    bool env_log_sched_ = false;
    bool env_op_trace_ = false;
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

        // CPU backend must be last. It handles unsupported ops and fallback copies.
        ggml_backend_t backends[] = { backend_, cpu_backend_ };

        // parallel=false (5th arg): on ggml-vulkan + MoltenVK, running CPU and
        // Vulkan splits concurrently races on the generator graph (CPU fallback
        // for CONV_TRANSPOSE_1D / PAD_REFLECT_1D bounces 39 MiB tensors back
        // and forth), and a downstream MUL_MAT silently outputs zeros → mute
        // audio for long inputs. Keep this serialised; do not flip back to true.
        sched_ = ggml_backend_sched_new(
            backends,
            nullptr,
            2,
            required,
            false,
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

    void log_sched_sizes(const char * stage) const {
        if (sched_ == nullptr || !env_log_sched_) {
            return;
        }

        const int n_splits = ggml_backend_sched_get_n_splits(sched_);
        const size_t vk_bytes = ggml_backend_sched_get_buffer_size(sched_, backend_);
        const size_t cpu_bytes = ggml_backend_sched_get_buffer_size(sched_, cpu_backend_);

        std::fprintf(
            stderr,
            "[vulkan-sched %s] splits=%d vulkan_buf=%.1f MiB cpu_buf=%.1f MiB\n",
            stage,
            n_splits,
            vk_bytes / 1048576.0,
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
            "[op %-6s] %-20s %-32s shape=[%lld,%lld,%lld,%lld]\n",
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
    explicit VulkanBackend(int32_t n_threads) {
        try {
            env_log_sched_ = std::getenv("KOKOPOP_VULKAN_LOG_SCHED") != nullptr;
            env_op_trace_ = std::getenv("KOKOPOP_VULKAN_OP_TRACE") != nullptr;

            const char * env = std::getenv("KOKOPOP_VULKAN_DEVICE");
            const int requested = env ? std::atoi(env) : 0;
            const int count = ggml_backend_vk_get_device_count();
            if (count <= 0) {
                return;
            }

            device_id_ = (requested >= 0 && requested < count) ? requested : 0;

            if (env_log_sched_) {
                char desc[256] = {};
                ggml_backend_vk_get_device_description(device_id_, desc, sizeof(desc));
                size_t free_mem = 0;
                size_t total_mem = 0;
                ggml_backend_vk_get_device_memory(device_id_, &free_mem, &total_mem);
                std::fprintf(
                    stderr,
                    "[vulkan] device=%d/%d %s free=%.1f MiB total=%.1f MiB\n",
                    device_id_,
                    count,
                    desc[0] ? desc : "?",
                    free_mem / 1048576.0,
                    total_mem / 1048576.0);
            }

            backend_ = ggml_backend_vk_init(static_cast<size_t>(device_id_));
            if (backend_ == nullptr) {
                return;
            }

            cpu_backend_ = ggml_backend_cpu_init();
            if (cpu_backend_ != nullptr) {
                ggml_backend_cpu_set_n_threads(cpu_backend_, std::max<int32_t>(1, n_threads));
            }
        } catch (const std::exception & e) {
            if (env_log_sched_) {
                std::fprintf(stderr, "[vulkan] init failed: %s\n", e.what());
            }
            if (backend_ != nullptr) {
                ggml_backend_free(backend_);
                backend_ = nullptr;
            }
            if (cpu_backend_ != nullptr) {
                ggml_backend_free(cpu_backend_);
                cpu_backend_ = nullptr;
            }
        } catch (...) {
            if (env_log_sched_) {
                std::fprintf(stderr, "[vulkan] init failed: unknown exception\n");
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
    }

    ~VulkanBackend() override {
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

        if (env_op_trace_) {
            ggml_backend_sched_set_eval_callback(sched_, eval_trace_cb, sched_);
        }

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

    ggml_backend_buffer_type_t weight_buffer_type() const override {
        GGML_ASSERT(backend_ != nullptr);
        return ggml_backend_vk_buffer_type(static_cast<size_t>(device_id_));
    }

    const char * label() const override {
        return "Vulkan (GPU)";
    }

    int32_t type() const override {
        return KOKOPOP_BACKEND_VULKAN;
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

std::unique_ptr<Backend> create_vulkan_backend(int32_t n_threads) {
    auto backend = std::make_unique<VulkanBackend>(n_threads);
    if (!backend->valid()) {
        return nullptr;
    }
    return backend;
}

} // namespace kokopop
