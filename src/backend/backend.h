#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <ggml.h>
#include <ggml-backend.h>

#include "kokopop.h"

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

    // Hint the scheduler that `tensor` should be computed on the CPU sub-
    // backend. Used to pin precision-critical ops (e.g. the duration predictor
    // LSTM) to CPU on GPU backends that exhibit fp16 drift (MoltenVK). Default
    // no-op for backends without a CPU fallback. Must be called between graph
    // build and sched_alloc_graph(); applied lazily on next alloc.
    virtual void defer_cpu_assignment(ggml_tensor * tensor) { (void)tensor; }

    /// False when the backend has no LEAKY_RELU kernel, so ggml-sched would
    /// bounce the tensor to the CPU sub-backend and back. Graph builders then
    /// emit the equivalent relu/neg/scale/sub chain instead (see
    /// graph_leaky_relu). Only worth doing where it is true: on the CPU the
    /// decomposition is three extra passes over the tensor for nothing.
    virtual bool has_leaky_relu() const { return true; }

    /// True when a convolution should be emitted as a direct CONV_2D rather
    /// than im2col + mul_mat. im2col materialises [OW*KW, IC] floats: for the
    /// Kokoro vocoder that is ~490 MiB written and read back per convolution,
    /// which measured as 65% of total OpenCL time on an Adreno 630. Backends
    /// with a real direct-convolution kernel should say true; the CPU should
    /// not, its im2col + blocked gemm is faster than a naive direct loop.
    virtual bool prefers_direct_conv() const { return false; }

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

    // Backend enum value stored on Model after AUTO has been resolved.
    virtual int32_t type() const { return KOKOPOP_BACKEND_CPU; }

    // Optional: tag the next compute call with a label (e.g. "frontend",
    // "generation", "generator"). Backends may use it for diagnostics or
    // selective compute (KOKOPOP_METAL_RUN_ONLY). Default: no-op.
    virtual void set_active_label(const char * label) { (void)label; }

    // Optional: hint the number of input tokens for the upcoming graph.
    // Backends may use it to decide whether to offload to GPU (Metal only
    // benefits above a minimum batch size). Default: no-op.
    virtual void set_input_tokens(int n_tokens) { (void)n_tokens; }

    // Pre-load a dequantized LSTM recurrent weight matrix for the fused LSTM
    // kernel.  Called once per LSTM direction at model-load time.
    //   key     : logical tensor name (e.g. "kokopop.text_encoder.lstm.weight_hh_l0")
    //   w_hh_f32: [H, 4*H] F32 (ggml col-major: ne[0]=H is fast dim)
    //   H / four_H: hidden size and 4*hidden
    // Default: no-op (CPU backend runs the recurrence entirely in C++).
    virtual void preload_lstm_whh(const std::string & key,
                                   const float * w_hh_f32, int H, int four_H) {
        (void)key; (void)w_hh_f32; (void)H; (void)four_H;
    }

    // Return the fused LSTM kernel handle cast to void* (null for CPU backend).
    // Used by graph_ops::lstm_direction to fill LstmCustomParams::metal_kernel.
    virtual void * metal_lstm_kernel() const { return nullptr; }

    // Return the Metal STFT kernel handle cast to void* (null for CPU backend).
    // Used by cpu_harmonic_stft to dispatch the DFT on the GPU.
    virtual void * metal_stft_kernel() const { return nullptr; }

    // Return the Metal vocoder kernel handle cast to void*
    // (null for CPU backend). Used by selected custom ggml nodes.
    virtual void * metal_vocoder_kernel() const { return nullptr; }


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

inline uint64_t backend_graph_signature(ggml_cgraph * graph) {
    uint64_t h = 1469598103934665603ull;
    const int n_nodes = graph != nullptr ? ggml_graph_n_nodes(graph) : 0;

    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };

    mix(static_cast<uint64_t>(n_nodes));
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node == nullptr) {
            mix(0);
            continue;
        }

        mix(static_cast<uint64_t>(node->op));
        mix(static_cast<uint64_t>(node->type));
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            mix(static_cast<uint64_t>(node->ne[d]));
        }
    }

    return h;
}

struct PendingInit {
    ggml_tensor * tensor = nullptr;
    std::vector<uint8_t> bytes;
    bool zero = false;
};

struct MetalVocoderConvTransposeParams {
    void * kernel = nullptr;
    const ggml_tensor * bias = nullptr;
    int stride = 1;
    int crop_left = 0;
};

// Params for the Metal generator-resblock custom-op node.
// weights is a type-erased const GeneratorResblockWeights * (cast in graph_ops.cpp).
struct MetalGeneratorResblockParams {
    void       * kernel     = nullptr;  // MetalVocoderState *
    const void * weights    = nullptr;  // const GeneratorResblockWeights *
    int          kernel_size = 3;
    int64_t      style_dim   = 0;
};

// Params for the fused per-stage generator custom-op. One callback runs:
//   leaky_relu(x, 0.1) | conv1d_strided(har, noise_conv) | resblock(noise)
//   | conv_transpose1d(x, up) | [pad_reflect_left1] | add x += x_source
//   | for k in 0..3: branch_k = resblock_k(x); sum(b0,b1,b2)/3
// in a single Metal command buffer per stage.
struct MetalGeneratorStageParams {
    void       * kernel = nullptr;        // MetalVocoderState *

    // har_t graph-input tensor (data pointer valid at callback time).
    const ggml_tensor * har_t = nullptr;

    // Noise conv1d (har → x_source)
    const ggml_tensor * noise_conv_w = nullptr;
    const ggml_tensor * noise_conv_b = nullptr;
    int                 noise_kernel  = 0;
    int                 noise_stride  = 0;
    int                 noise_padding = 0;

    // Upsampling conv_transpose1d
    const ggml_tensor * up_w = nullptr;
    const ggml_tensor * up_b = nullptr;
    int                 up_stride  = 0;
    int                 up_padding = 0;

    // True for stage 1 (left-reflect pad by 1 after convt).
    bool                pad_reflect_left1 = false;

    // Noise resblock (kernel_size = noise_kernel_size).
    const void *        noise_resblock = nullptr;   // GeneratorResblockWeights *
    int                 noise_kernel_size = 0;

    // 3 main resblocks (kernel_size matches KOKOPOP_RESBLOCK_KERNELS).
    const void *        main_resblocks[3] = {nullptr, nullptr, nullptr};  // GeneratorResblockWeights *
    int                 main_kernel_sizes[3] = {3, 7, 11};

    // Style dim (read from any cached gamma weight).
    int64_t             style_dim = 0;
};

// Factory: creates a CPU backend (always available).
std::unique_ptr<Backend> create_cpu_backend(int32_t n_threads);

// Factory: create the backend for the given request.
// requested: KOKOPOP_BACKEND_AUTO (0), KOKOPOP_BACKEND_CPU (1),
//            KOKOPOP_BACKEND_METAL (2), KOKOPOP_BACKEND_CUDA (3),
//            KOKOPOP_BACKEND_VULKAN (4), KOKOPOP_BACKEND_OPENCL (5)
std::unique_ptr<Backend> create_backend(
    int32_t requested, int32_t n_threads, std::string & error);

} // namespace kokopop
