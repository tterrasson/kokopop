#pragma once

// C-compatible interface to the Metal LSTM kernel.
// Only available when KOKOPOP_HAS_METAL is defined.
// Implemented in metal_lstm.mm (Objective-C++).

#ifdef KOKOPOP_HAS_METAL

struct MetalLstmKernelState;

// Opaque handle returned by metal_lstm_submit().
typedef void * MetalLstmHandle;

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

// Async API — submit without waiting. Forward and backward passes of the same
// bidirectional layer are independent and can be submitted together so Metal
// overlaps their execution, cutting syncs from 2 to 1 per layer:
//
//   MetalLstmHandle hf = metal_lstm_submit(s, "fwd", ...);
//   MetalLstmHandle hb = metal_lstm_submit(s, "bwd", ...);
//   metal_lstm_collect(hf);   // one waitUntilCompleted covers both
//   metal_lstm_collect(hb);   // returns immediately if GPU already done
//
// Each handle owns its own MTLBuffers; concurrent calls are safe.
// output pointer must remain valid until metal_lstm_collect() returns.
MetalLstmHandle metal_lstm_submit(
    MetalLstmKernelState * state,
    const char * whh_key,
    const float * pre_gates,
    const float * b_hh,
    float       * output,
    int H, int N, bool reverse);

void metal_lstm_collect(MetalLstmHandle handle);

#endif // KOKOPOP_HAS_METAL
