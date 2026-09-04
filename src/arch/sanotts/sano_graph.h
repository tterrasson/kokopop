#pragma once

// Graph primitives shared by the sanoTTS stages.
//
// These are deliberately not Kokoro's helpers: those take a `KokoroArch &` and
// carry Metal-vocoder fusion hooks that mean nothing here. What is shared is
// the *convention* — conv1d kernels stored flat as `[IC * K, OC]`, activations
// in `[T, C]` with time in `ne[0]` — so the two sets of helpers read the same
// weight layout.
//
// Two activation layouts appear below and mixing them up is the easiest way to
// get plausible, wrong audio:
//   * `[T, C]` (time in ne[0]) for everything convolutional, because that is
//     what ggml's conv operators consume;
//   * `[C, T]` (channels in ne[0]) inside the vocos trunk, because
//     `ggml_norm` reduces over ne[0] and the LayerNorm there is per frame over
//     channels.

#include <cstdint>
#include <string>

struct ggml_context;
struct ggml_tensor;

namespace kokopop {

struct Backend;
struct ScratchArena;

/// A no-alloc ggml context backed by `arena`. The caller owns the returned
/// context and must `ggml_free` it.
ggml_context * sano_graph_context(ScratchArena & arena, size_t bytes,
                                  const char * label, std::string & error);

/// conv1d with "same" padding semantics applied by the caller.
///
/// `weight` is `[in_ch * kernel, out_ch]`, `x` is `[T, in_ch]`, the result is
/// `[out_length, out_ch]`. Bias is not added here; use `sano_add_channel_bias`.
///
/// The im2col buffer is materialised in F32 rather than ggml_conv_1d's F16:
/// these models are small enough that the extra bandwidth is invisible, and
/// the F32 voices are gated at an error tolerance F16 activations would eat on
/// their own.
ggml_tensor * sano_conv1d(ggml_context * ctx, ggml_tensor * weight,
                          ggml_tensor * x, int64_t in_ch, int kernel,
                          int padding, int dilation);

/// Depthwise conv1d, `weight` `[K, C]`, `x` `[T, C]`, result `[T, C]`.
ggml_tensor * sano_conv1d_dw(ggml_context * ctx, ggml_tensor * weight,
                             ggml_tensor * x, int kernel, int padding);

/// ConvTranspose1d with PyTorch's symmetric `padding`, `weight` `[K, OC, IC]`,
/// `x` `[T, IC]`, result `[(T - 1) * stride - 2 * padding + K, OC]`.
ggml_tensor * sano_conv_transpose1d(ggml_context * ctx, ggml_tensor * weight,
                                    ggml_tensor * x, int stride, int padding);

/// Adds a `[C]` bias to a `[T, C]` activation.
ggml_tensor * sano_add_channel_bias(ggml_context * ctx, ggml_tensor * x,
                                    ggml_tensor * bias);

/// `ggml_leaky_relu`, or its relu decomposition on backends without a kernel
/// for it, so the tensor is not bounced to the CPU sub-backend and back.
ggml_tensor * sano_leaky_relu(ggml_context * ctx, const Backend * backend,
                              ggml_tensor * x, float slope);

/// LayerNorm over `ne[0]`: for a `[C, T]` activation this is the per-frame
/// normalisation over channels the vocos trunk needs.
ggml_tensor * sano_layer_norm(ggml_context * ctx, ggml_tensor * x,
                              ggml_tensor * weight, ggml_tensor * bias);

/// `x + scale * conv2(silu(conv1(x)))` on a `[T, C]` activation.
struct SanoResBlock;
ggml_tensor * sano_res_block(ggml_context * ctx, ggml_tensor * x,
                             const SanoResBlock & block, int64_t channels);

} // namespace kokopop
