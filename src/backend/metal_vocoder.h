#pragma once

#include <ggml.h>

#ifdef KOKOPOP_HAS_METAL

namespace kokopop {

struct MetalVocoderState;

MetalVocoderState * metal_vocoder_create();
void                metal_vocoder_destroy(MetalVocoderState *);

bool metal_vocoder_conv_transpose1d_crop_bias(
    MetalVocoderState * state,
    const ggml_tensor * input,
    const ggml_tensor * weight,
    const ggml_tensor * bias,
    ggml_tensor * output,
    int stride,
    int crop_left);

} // namespace kokopop

#endif // KOKOPOP_HAS_METAL
