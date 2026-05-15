#pragma once

// Only available when KOKOPOP_HAS_METAL is defined.
// Implemented in metal_lstm.mm (Objective-C++).

#ifdef KOKOPOP_HAS_METAL

struct MetalLstmKernelState;

// Create / destroy
MetalLstmKernelState * metal_lstm_create();
void                   metal_lstm_destroy(MetalLstmKernelState *);

// Pre-load a dequantized W_hh matrix into a persistent MTLBuffer.
// Call once per LSTM direction at model-load time.
//   key     : unique string identifying the LSTM direction
//   w_hh_f32: [H, 4*H] row-major float (ggml column-major: ne[0]=H is fast dim)
//   H       : hidden size
//   four_H  : 4 * H
void metal_lstm_preload_whh(
    MetalLstmKernelState * state,
    const char * key,
    const float * w_hh_f32,
    int H, int four_H);

// Synchronous: submit + wait in one call. Reuses internal scratch buffers.
//   pre_gates : [4*H, N] float  (ggml col-major)
//   b_hh      : [4*H]   float
//   output    : [H, N]  float   (written on return)
//   H, N      : hidden size, time-steps
//   reverse   : forward (false) or backward (true) pass
void metal_lstm_run(
    MetalLstmKernelState * state,
    const char * whh_key,
    const float * pre_gates,
    const float * b_hh,
    float       * output,
    int H, int N, bool reverse);

// Pre-gates matmul. Preload w_ih (dequantized) once per direction, then call
// the matmul per chunk to compute pre_gates = w_ih @ input on GPU.
//   w_ih shape: [I, 4H] f32, ggml col-major (ne[0]=I fast dim)
void metal_lstm_preload_wih(
    MetalLstmKernelState * state,
    const char * key,
    const float * w_ih_f32,
    int I, int four_H);

// Returns false if the kernel/preload is unavailable; caller then falls
// back to the ggml mul_mat CPU path.
bool metal_lstm_pregates_matmul(
    MetalLstmKernelState * state,
    const char * key,
    const float * input,    // [I, N] f32 col-major
    float       * pre_gates, // [4H, N] f32 col-major
    int I, int four_H, int N);

#endif // KOKOPOP_HAS_METAL
