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

#endif // KOKOPOP_HAS_METAL
