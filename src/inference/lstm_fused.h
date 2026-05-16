#pragma once

#include <ggml.h>
#include <cstdint>

namespace kokopop {

// Parameters for a single fused LSTM direction.
// Instances must remain alive for the duration of ggml graph execution;
// store them in Model::lstm_custom_params (pre-reserved vector).
struct LstmCustomParams {
    const float * w_hh_f32;     // [hidden, 4*hidden] F32, dequantized at model load
    const float * b_hh;         // [4*hidden] F32
    const float * w_hh_rowwise; // [4*hidden, hidden] transposed for SIMD-friendly access
    void        * metal_kernel; // null (CPU) or MetalLstmKernelState* (Metal)
    const char  * whh_key;      // logical tensor name used to look up pre-loaded MTLBuffer
    int64_t       hidden;
    int64_t       n_steps;
    bool          reverse;
};

// ggml_custom2_op_t callback — invoked by ggml's CPU executor during graph compute.
//   dst (= src[0])  : output [hidden, n_steps] — zero-initialised before call
//   pre_gates (= src[1]) : [4*hidden, n_steps] = W_ih*input + b_ih, computed by ggml
//   userdata        : LstmCustomParams*
void lstm_fused_callback(
    ggml_tensor       * dst,
    const ggml_tensor * /*a_unused*/,
    const ggml_tensor * pre_gates,
    int ith, int nth,
    void * userdata);

// Pre-gates matmul callback. Runs `dst = w_ih @ input` on the Metal LSTM
// kernel state. The wih_key must have been preloaded via
// Backend::preload_lstm_wih at model load.
struct LstmPregatesParams {
    void       * metal_kernel; // MetalLstmKernelState *
    const char * wih_key;
    const float * w_ih_f32;
    int          I;
    int          four_H;
    int          n_steps;
};

void lstm_pregates_callback(
    ggml_tensor       * dst,
    const ggml_tensor * /*dst_unused*/,
    const ggml_tensor * input,
    int ith, int nth,
    void * userdata);

} // namespace kokopop
