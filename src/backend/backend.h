#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <ggml.h>
#include <ggml-backend.h>

namespace kokopop {

// Abstract backend interface — zero #ifdef GGML_USE_METAL in inference code.
// Each concrete backend encapsulates its own graph-execution strategy and I/O.
struct Backend {
    virtual ~Backend() = default;

    // ---- Graph lifecycle (called in order by the inference pipeline) ----

    // Prepare the backend for a new graph execution.
    // Reset the scheduler before assigning/allocating a new graph.
    virtual void sched_reset() = 0;

    // Allocate backend resources for the given graph.
    // Returns false on allocation failure.
    // Calls ggml_backend_sched_alloc_graph after ensuring the scheduler is
    // large enough for the graph.
    virtual bool sched_alloc_graph(ggml_cgraph * graph) = 0;

    // ---- Deferred tensor initialization ----

    // Clear all pending tensor inits queued since last clear.
    virtual void clear_pending_inits() = 0;

    // Queue a tensor to be initialized from host bytes on the next apply.
    virtual void queue_tensor_data(ggml_tensor * tensor, const void * data, size_t size) = 0;

    // Queue a tensor to be zero-initialized on the next apply.
    virtual void queue_zero_tensor(ggml_tensor * tensor) = 0;

    // Queue a tensor to be filled with a specific value on the next apply.
    virtual void queue_f32_tensor(ggml_tensor * tensor, float value) = 0;

    // Apply all queued tensor initializations.
    // Must be called after sched_alloc_graph() (when tensor->buffer is valid).
    // Returns false on error.
    virtual bool apply_pending_inits() = 0;

    // ---- Graph execution ----

    // Execute the graph.
    // Execute the graph through the backend scheduler.
    virtual ggml_status compute(ggml_context * ctx, ggml_cgraph * graph) = 0;

    // ---- Tensor I/O ----

    // Write host data into a backend tensor.
    virtual void tensor_set(ggml_tensor * t, const void * data,
                            size_t offset, size_t size) = 0;

    // Read backend tensor data into host buffer.
    virtual void tensor_get(ggml_tensor * t, void * data,
                            size_t offset, size_t size) = 0;

    // Buffer type used for persistent model weights.
    virtual ggml_backend_buffer_type_t weight_buffer_type() const = 0;

    // Human-readable backend name for display purposes.
    virtual const char * label() const { return "CPU"; }

    // Optional: tag the next compute call with a label (e.g. "frontend",
    // "generation", "generator"). Backends may use it for diagnostics or
    // selective compute (KOKOPOP_METAL_RUN_ONLY). Default: no-op.
    virtual void set_active_label(const char * label) { (void)label; }

    // Optional: hint the number of input tokens for the upcoming graph.
    // Backends may use it to decide whether to offload to GPU (Metal only
    // benefits above a minimum batch size). Default: no-op.
    virtual void set_input_tokens(int n_tokens) { (void)n_tokens; }

    // ---- Context sizing ----

    // Memory bytes needed for a generation scratch context.
    virtual size_t generation_context_bytes(int64_t total_frames,
                                            int64_t n_tokens) const = 0;

    // Memory bytes needed for a generator scratch context.
    virtual size_t generator_context_bytes(int64_t decoder_len) const = 0;

    // Memory bytes needed for a frontend scratch context.
    virtual size_t frontend_context_bytes() const = 0;
};

// Shared utilities for backend implementations.
inline size_t backend_mib(size_t value) {
    return value * 1024ull * 1024ull;
}

inline size_t backend_graph_capacity(ggml_cgraph * graph) {
    const size_t need = static_cast<size_t>(std::max(1, ggml_graph_size(graph)));
    return std::max<size_t>(GGML_DEFAULT_GRAPH_SIZE, need + 1024);
}

struct PendingInit {
    ggml_tensor * tensor = nullptr;
    std::vector<uint8_t> bytes;
    bool zero = false;
};

// Factory: creates a CPU backend (always available).
std::unique_ptr<Backend> create_cpu_backend(int32_t n_threads);

// Factory: create the backend for the given request.
// requested: KOKOPOP_BACKEND_AUTO (0), KOKOPOP_BACKEND_CPU (1), KOKOPOP_BACKEND_METAL (2)
std::unique_ptr<Backend> create_backend(
    int32_t requested, int32_t n_threads, std::string & error);

} // namespace kokopop
