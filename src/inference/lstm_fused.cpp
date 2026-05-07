#include "lstm_fused.h"
#include <cmath>
#include <cstring>

#ifdef KOKOPOP_HAS_METAL
#include "backend/metal_lstm.h"
#endif

namespace kokopop {

// ---------------------------------------------------------------------------
// CPU tight-loop LSTM recurrence.
// Replaces the per-timestep ggml graph loop: instead of ~18 ggml nodes per
// step (view, add, sigmoid, tanh, set_2d …), we compute the full recurrence
// in a single C++ function call.
//
// Layout (ggml column-major — ne[0] is the fast / contiguous dimension):
//   pre_gates : float[4*H * N]  element (g, t) = pre_gates[g + 4*H*t]
//   w_hh_f32  : float[H * 4*H]  element (k, j) = w_hh[k + H*j]
//   b_hh      : float[4*H]
//   output    : float[H * N]    element (h, t) = output[h + H*t]
// ---------------------------------------------------------------------------
static void cpu_lstm(
    const LstmCustomParams & p,
    const float * pre_gates,
    float       * output)
{
    const int64_t H  = p.hidden;
    const int64_t N  = p.n_steps;
    const int64_t H4 = 4 * H;

    // h, c live on the stack (small: 256 × 2 × 4 = 2 KB)
    float h[256] = {};
    float c[256] = {};

    // Temporary gate accumulator for one timestep
    float gates[1024];  // 4*H, stack

    for (int64_t step = 0; step < N; ++step) {
        const int64_t t = p.reverse ? (N - 1 - step) : step;

        // gates[j] = b_hh[j] + pre_gates[j, t] + (W_hh[:,j] · h)
        const float * pg_t = pre_gates + H4 * t;
        for (int64_t j = 0; j < H4; ++j) {
            float dot = 0.0f;
            const float * col = p.w_hh_f32 + j * H;  // column j of W_hh
            for (int64_t k = 0; k < H; ++k) {
                dot += col[k] * h[k];
            }
            gates[j] = p.b_hh[j] + pg_t[j] + dot;
        }

        // Apply activations and update h, c
        float * out_t = output + H * t;
        for (int64_t i = 0; i < H; ++i) {
            const float i_gate = 1.0f / (1.0f + expf(-gates[i]));
            const float f_gate = 1.0f / (1.0f + expf(-gates[H   + i]));
            const float g_gate = tanhf(gates[2*H + i]);
            const float o_gate = 1.0f / (1.0f + expf(-gates[3*H + i]));

            float c_new = f_gate * c[i] + i_gate * g_gate;
            if (c_new >  50.0f) c_new =  50.0f;
            if (c_new < -50.0f) c_new = -50.0f;
            c[i] = c_new;
            h[i] = o_gate * tanhf(c[i]);
            out_t[i] = h[i];
        }
    }
}

void lstm_fused_callback(
    ggml_tensor       * dst,
    const ggml_tensor * /*a_unused*/,
    const ggml_tensor * pre_gates,
    int   /*ith*/,
    int   /*nth*/,
    void * userdata)
{
    const LstmCustomParams * p = static_cast<const LstmCustomParams *>(userdata);
    const float * pg  = static_cast<const float *>(pre_gates->data);
    float       * out = static_cast<float *>(dst->data);
#ifdef KOKOPOP_HAS_METAL
    if (p->metal_kernel) {
        metal_lstm_run(
            static_cast<MetalLstmKernelState *>(p->metal_kernel),
            p->whh_key,
            pg, p->b_hh, out,
            static_cast<int>(p->hidden),
            static_cast<int>(p->n_steps),
            p->reverse);
        return;
    }
#endif

    cpu_lstm(*p, pg, out);
}

} // namespace kokopop
