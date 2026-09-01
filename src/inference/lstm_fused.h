#pragma once

#include <ggml.h>
#include <cstdint>

#ifdef KOKOPOP_HAS_OPENCL
#include <ggml-opencl.h>
#endif

namespace kokopop {

#ifdef KOKOPOP_HAS_OPENCL
// The OpenCL backend reads this block straight out of the custom-op userdata,
// so in an OpenCL build it must be *its* declaration, not a copy of it.
using LstmOpenclParams = ggml_opencl_lstm_params_v1;
inline constexpr uint64_t kLstmOpenclParamsMagic = GGML_OPENCL_LSTM_PARAMS_V1_MAGIC;
#else
// Same ABI, declared locally so that CPU/Metal/CUDA/Vulkan builds do not
// depend on the patched ggml-opencl.h. No backend reads it in that case: only
// the CPU and Metal paths, which are compiled against this declaration.
struct LstmOpenclParams {
    uint64_t      magic;
    const float * w_hh_f32; // ggml [H, 4H]: w[k + H*gate]
    const float * b_hh;     // [4H]
    int64_t       hidden;
    int64_t       n_steps;
    int32_t       reverse;
};
inline constexpr uint64_t kLstmOpenclParamsMagic = UINT64_C(0x4b4f4b4f4c535431);
#endif

// Parameters for a single fused LSTM direction.
// Instances must remain alive for the duration of ggml graph execution;
// store them in Model::lstm_custom_params (pre-reserved vector).
struct LstmCustomParams {
    LstmOpenclParams opencl;
    const float * w_hh_rowwise; // [4*hidden, hidden] transposed for SIMD-friendly access
    void        * metal_kernel; // null (CPU) or MetalLstmKernelState* (Metal)
    const char  * whh_key;      // logical tensor name used to look up pre-loaded MTLBuffer
};

// ggml_custom2_op_t callback — invoked by ggml's CPU executor during graph compute.
//   dst (= src[0])  : output [hidden, n_steps] — fully written by the callback
//   pre_gates (= src[1]) : [4*hidden, n_steps] = W_ih*input + b_ih, computed by ggml
//   userdata        : LstmCustomParams*
void lstm_fused_callback(
    ggml_tensor       * dst,
    const ggml_tensor * /*a_unused*/,
    const ggml_tensor * pre_gates,
    int ith, int nth,
    void * userdata);

} // namespace kokopop
