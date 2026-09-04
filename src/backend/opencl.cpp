// This translation unit is only compiled when the OpenCL backend is enabled
// (see CMakeLists.txt). Define the flag defensively so that tooling which
// indexes this file without the target's compile definitions still sees the
// declaration in opencl.h rather than the inline no-op stub.
#ifndef KOKOPOP_HAS_OPENCL
#define KOKOPOP_HAS_OPENCL
#endif

#include "backend.h"
#include "opencl.h"

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
#include <ggml-opencl.h>

namespace kokopop {
namespace {

class OpenCLBackend final : public Backend {
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    size_t sched_capacity_ = 0;
    uint64_t sched_signature_ = 0;
    bool env_log_sched_ = false;
    bool env_op_trace_ = false;
    bool env_pin_ = false;
    std::vector<PendingInit> pending_inits_;
    std::vector<ggml_tensor *> cpu_assignments_;

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

    void log_sched_sizes(const char * stage) const {
        if (sched_ == nullptr || !env_log_sched_) {
            return;
        }

        const int n_splits = ggml_backend_sched_get_n_splits(sched_);
        const size_t cl_bytes = ggml_backend_sched_get_buffer_size(sched_, backend_);
        const size_t cpu_bytes = ggml_backend_sched_get_buffer_size(sched_, cpu_backend_);

        std::fprintf(
            stderr,
            "[opencl-sched %s] splits=%d opencl_buf=%.1f MiB cpu_buf=%.1f MiB\n",
            stage,
            n_splits,
            cl_bytes / 1048576.0,
            cpu_bytes / 1048576.0);
    }

    // ggml-opencl has no device-query helpers of its own; go through the
    // generic registry API for a human-readable device line.
    static void log_device() {
        ggml_backend_reg_t reg = ggml_backend_opencl_reg();
        if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
            std::fprintf(stderr, "[opencl] device=? (no registry entry)\n");
            return;
        }

        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
        if (dev == nullptr) {
            std::fprintf(stderr, "[opencl] device=? (null registry device)\n");
            return;
        }

        const char * desc = ggml_backend_dev_description(dev);
        size_t free_mem = 0;
        size_t total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);

        std::fprintf(
            stderr,
            "[opencl] device=%s free=%.1f MiB total=%.1f MiB\n",
            desc != nullptr ? desc : "?",
            free_mem / 1048576.0,
            total_mem / 1048576.0);
    }

    static bool eval_trace_cb(ggml_tensor * tensor, bool ask, void * user_data) {
        if (ask) {
            return true;
        }

        // Wall time since the previous callback. ggml-sched computes node by
        // node when an eval callback is installed and synchronises before
        // calling it, so this really is the cost of the node just executed --
        // inflated by the per-node sync, but comparable across ops, which is
        // what attribution needs.
        static int64_t last_us = 0;
        const int64_t now_us = ggml_time_us();
        const int64_t dt_us = last_us != 0 ? now_us - last_us : 0;
        last_us = now_us;

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

        // Source shapes matter as much as the destination one: a MUL that is
        // 30x off memory bandwidth is a broadcast pattern the kernel handles
        // badly, and that is only visible in src1's ne.
        char srcs[128] = {};
        int  pos = 0;
        for (int i = 0; tensor != nullptr && i < GGML_MAX_SRC && tensor->src[i] != nullptr; ++i) {
            const ggml_tensor * s = tensor->src[i];
            pos += std::snprintf(srcs + pos, sizeof(srcs) - pos, " s%d=%s[%lld,%lld,%lld,%lld]",
                                 i, ggml_type_name(s->type),
                                 (long long) s->ne[0], (long long) s->ne[1],
                                 (long long) s->ne[2], (long long) s->ne[3]);
            if (pos >= (int) sizeof(srcs) - 1) break;
        }

        std::fprintf(
            stderr,
            "[op %-6s] %8lld us %-20s %-32s shape=[%lld,%lld,%lld,%lld]%s\n",
            backend_name,
            (long long) dt_us,
            op_name,
            tensor != nullptr && tensor->name[0] ? tensor->name : "",
            static_cast<long long>(tensor != nullptr ? tensor->ne[0] : 0),
            static_cast<long long>(tensor != nullptr ? tensor->ne[1] : 0),
            static_cast<long long>(tensor != nullptr ? tensor->ne[2] : 0),
            static_cast<long long>(tensor != nullptr ? tensor->ne[3] : 0),
            srcs);

        std::fflush(stderr);
        return true;
    }

public:
    explicit OpenCLBackend(int32_t n_threads) {
        try {
            env_log_sched_ = std::getenv("KOKOPOP_OPENCL_LOG_SCHED") != nullptr;
            env_op_trace_ = std::getenv("KOKOPOP_OPENCL_OP_TRACE") != nullptr;
            // Off by default here, unlike Vulkan: on an Adreno 630 the pins
            // cost 0.76 s per utterance in extra splits, and running the norms
            // on the GPU instead measured 59.6 dB SNR against the pinned
            // output is inaudible. KOKOPOP_OPENCL_PIN_NORM=1 restores them.
            const char * pin = std::getenv("KOKOPOP_OPENCL_PIN_NORM");
            env_pin_ = pin != nullptr && std::strcmp(pin, "0") != 0;

            // ggml-opencl exposes no device enumeration (no get_device_count /
            // _description / _memory): there is exactly one implicit device and
            // init either succeeds or returns nullptr.
            backend_ = ggml_backend_opencl_init();
            if (backend_ == nullptr) {
                return;
            }

            if (env_log_sched_) {
                log_device();
            }

            cpu_backend_ = ggml_backend_cpu_init();
            if (cpu_backend_ != nullptr) {
                ggml_backend_cpu_set_n_threads(cpu_backend_, std::max<int32_t>(1, n_threads));
            }
        } catch (const std::exception & e) {
            if (env_log_sched_) {
                std::fprintf(stderr, "[opencl] init failed: %s\n", e.what());
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
                std::fprintf(stderr, "[opencl] init failed: unknown exception\n");
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

    ~OpenCLBackend() override {
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

        // Pin deferred tensors to the CPU sub-backend BEFORE sched plans its
        // splits. Used to escape GPU fp16 drift on precision-critical ops
        // (e.g. duration predictor chain) — the same ops that need it on
        // MoltenVK need it on Adreno.
        if (!cpu_assignments_.empty()) {
            size_t verified_cpu = 0;
            for (ggml_tensor * t : cpu_assignments_) {
                if (t != nullptr) {
                    ggml_backend_sched_set_tensor_backend(sched_, t, cpu_backend_);
                    ggml_backend_t got = ggml_backend_sched_get_tensor_backend(sched_, t);
                    if (got == cpu_backend_) {
                        ++verified_cpu;
                    }
                }
            }
            if (env_log_sched_) {
                std::fprintf(stderr, "[opencl-sched] pinned %zu tensors → %zu confirmed on CPU\n",
                             cpu_assignments_.size(), verified_cpu);
                std::fflush(stderr);
            }
        }
        cpu_assignments_.clear();

        const bool ok = ggml_backend_sched_alloc_graph(sched_, graph);
        if (ok) {
            log_sched_sizes("alloc");
        }
        return ok;
    }

    void defer_cpu_assignment(ggml_tensor * tensor) override {
        // Every pinned tensor sitting in the middle of a GPU graph costs the
        // scheduler two splits, i.e. two device<->host round trips. On the
        // Kokoro generator the pins are isolated NORMs interleaved every ~28
        // GPU ops, which is most of the 100+ splits measured on an Adreno 630.
        // Measured on this device: the pinned and unpinned outputs differ by
        // 59.6 dB SNR, so the pinning is off by default and
        // KOKOPOP_OPENCL_PIN_NORM=1 brings it back.
        if (!env_pin_) {
            return;
        }
        if (tensor != nullptr) {
            cpu_assignments_.push_back(tensor);
        }
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

    void preload_lstm_whh(const std::string &, const float * w_hh_f32,
                           int H, int four_H) override {
        ggml_backend_opencl_lstm_preload(backend_, w_hh_f32, H, four_H);
    }

    ggml_backend_buffer_type_t weight_buffer_type() const override {
        GGML_ASSERT(backend_ != nullptr);
        GGML_ASSERT(cpu_backend_ != nullptr);
        // KOKOPOP_OPENCL_WEIGHTS_CPU=1 places model weights in CPU memory and
        // lets the scheduler stage them to the GPU as needed. This is much
        // slower but works around precision/correctness issues observed with
        // device-resident quantized weights on the Vulkan backend for long
        // inputs ("deformed voice" on some texts); keep the same escape hatch
        // here until Adreno is proven clean. Default keeps weights on the GPU.
        if (std::getenv("KOKOPOP_OPENCL_WEIGHTS_CPU") != nullptr) {
            return ggml_backend_get_default_buffer_type(cpu_backend_);
        }
        // Not ggml_backend_opencl_buffer_type(): ggml-opencl.h still declares
        // it, but the implementation is gone (the buffer type is reachable
        // only through the device now), so it fails to link.
        return ggml_backend_get_default_buffer_type(backend_);
    }

    // ggml-opencl implements RELU/NEG/SUB/SCALE but not LEAKY_RELU. In the
    // Kokoro vocoder the leaky_relus sit between convolutions on the largest
    // tensors of the whole pipeline, so letting them fall back to the CPU
    // means copying hundreds of MiB off and back onto the device, dozens of
    // times per utterance.
    bool has_leaky_relu() const override {
        return false;
    }

    bool prefers_direct_conv() const override {
        return true;
    }

    const char * label() const override {
        return "OpenCL (GPU)";
    }

    int32_t type() const override {
        return KOKOPOP_BACKEND_OPENCL;
    }



};

} // anonymous namespace

std::unique_ptr<Backend> create_opencl_backend(int32_t n_threads) {
    auto backend = std::make_unique<OpenCLBackend>(n_threads);
    if (!backend->valid()) {
        return nullptr;
    }
    return backend;
}

} // namespace kokopop
