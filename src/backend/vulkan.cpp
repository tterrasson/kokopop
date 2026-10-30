#include "backend.h"
#include "vulkan.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-vulkan.h>

namespace kokopop {
namespace {

class VulkanBackend final : public SchedBackendBase {
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_t backends_[2] = {nullptr, nullptr};
    int device_id_ = 0;
    bool env_log_sched_ = false;
    bool env_op_trace_ = false;

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
        // ggml's scheduler expects the CPU backend last: it handles unsupported
        // ops and fallback copies.
        sched_op_offload_ = true;
        sched_split_graph_ = true;

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

    void configure_sched_callbacks() override {
        if (env_op_trace_) {
            ggml_backend_sched_set_eval_callback(sched_, eval_trace_cb, sched_);
        }
    }

    bool sched_alloc_graph(ggml_cgraph * graph) override {
        // Deferred CPU pins are applied by the base class in
        // on_scheduler_ready(), BEFORE sched plans its splits — used to
        // escape MoltenVK fp16 drift on precision-critical ops (e.g. the
        // duration predictor chain).
        const bool ok = SchedBackendBase::sched_alloc_graph(graph);
        if (ok) {
            log_sched_sizes("alloc");
        }
        return ok;
    }

    ggml_status compute(ggml_context * ctx, ggml_cgraph * graph) override {
        log_sched_sizes("compute-begin");
        return SchedBackendBase::compute(ctx, graph);
    }

    ggml_backend_buffer_type_t weight_buffer_type() const override {
        GGML_ASSERT(backend_ != nullptr);
        GGML_ASSERT(cpu_backend_ != nullptr);
        // KOKOPOP_VULKAN_WEIGHTS_CPU=1 places model weights in CPU memory and
        // lets the scheduler stage them to Vulkan as needed. This is much slower
        // (the generator graph goes from ~250 ms to ~5 s on M1/MoltenVK) but
        // works around precision/correctness issues observed with VRAM-resident
        // quantized weights on MoltenVK for long inputs ("deformed voice" on
        // some texts). Default keeps weights in VRAM for full speed.
        if (std::getenv("KOKOPOP_VULKAN_WEIGHTS_CPU") != nullptr) {
            return ggml_backend_get_default_buffer_type(cpu_backend_);
        }
        return ggml_backend_vk_buffer_type(static_cast<size_t>(device_id_));
    }

    const char * label() const override {
        return "Vulkan (GPU)";
    }

    int32_t type() const override {
        return KOKOPOP_BACKEND_VULKAN;
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
