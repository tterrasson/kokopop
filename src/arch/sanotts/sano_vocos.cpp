// sanoTTS TinyVocos decoder: mel + deterministic noise -> waveform.
//
// A ConvNeXt1D trunk produces a complex half-spectrum, which an inverse STFT
// turns back into audio. Three things here are not negotiable, because each of
// them fails silently rather than loudly:
//
//   * the noise stream. The decoder is noise-fed; a different draw is a
//     different waveform with no error anywhere. See sano_noise.h.
//   * `ggml_gelu_erf`, not `ggml_gelu`. The latter is the tanh approximation
//     and differs by up to 1e-3 — enough to miss the correlation gate.
//   * the DC and Nyquist bins are zeroed. Upstream skips the exp for them
//     entirely; a magnitude there leaks a constant offset into every frame.
//
// The trunk runs in `[C, T]` layout — channels in `ne[0]` — because
// `ggml_norm` reduces over `ne[0]` and this LayerNorm is per frame over
// channels. Only the two 7-tap convolutions at the stem and the depthwise
// convolution inside each block need `[T, C]`, and they transpose around
// themselves.
//
// Reference: `mcu/src/snt_nano.c` (MIT), commit
// 939d982b9faa54cbcf5d24cc878f5cd514b2646e. See THIRD_PARTY.md.

#include "arch/sanotts/sano_arch.h"
#include "arch/sanotts/sano_graph.h"
#include "arch/sanotts/sano_noise.h"
#include "audio/istft.h"
#include "backend/backend.h"

#include <cmath>
#include <string>
#include <vector>

#include <ggml.h>

namespace kokopop {

namespace {

/// A 7-tap "same" convolution over `[T, C_in]`, returned in `[C_out, T]`.
ggml_tensor * stem_conv(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias,
                        ggml_tensor * x, int64_t in_ch, int kernel) {
    ggml_tensor * out = sano_conv1d(ctx, weight, x, in_ch, kernel, kernel / 2, 1);
    out = sano_add_channel_bias(ctx, out, bias);
    return ggml_cont(ctx, ggml_transpose(ctx, out));
}

/// One ConvNeXt1D block on a `[C, T]` activation.
ggml_tensor * convnext_block(ggml_context * ctx, ggml_tensor * x,
                             const SanoVocosBlock & block, int dw_kernel) {
    // Depthwise is the one operator that wants time in ne[0].
    ggml_tensor * r = ggml_cont(ctx, ggml_transpose(ctx, x));
    r = sano_conv1d_dw(ctx, block.dw_w, r, dw_kernel, dw_kernel / 2);
    r = sano_add_channel_bias(ctx, r, block.dw_b);
    r = ggml_cont(ctx, ggml_transpose(ctx, r));

    r = sano_layer_norm(ctx, r, block.norm_w, block.norm_b);

    ggml_tensor * hidden = ggml_mul_mat(ctx, block.pw0_w, r);
    ggml_mul_mat_set_prec(hidden, GGML_PREC_F32);
    hidden = ggml_gelu_erf(ctx, ggml_add(ctx, hidden, block.pw0_b));

    ggml_tensor * out = ggml_mul_mat(ctx, block.pw1_w, hidden);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    out = ggml_add(ctx, out, block.pw1_b);

    return ggml_add(ctx, x, ggml_mul(ctx, out, block.gamma));
}

/// The zero mask for the DC and Nyquist bins, `[bins, 1]`.
std::vector<float> bin_mask(uint32_t bins) {
    std::vector<float> mask(bins, 1.0f);
    mask.front() = 0.0f;
    mask.back() = 0.0f;
    return mask;
}

/// First-order DC blocker, `H(z) = (1 - z^-1) / (1 - pole * z^-1)`.
///
/// Recurrent, so it stays on the host: there is nothing to parallelise and the
/// state has to be carried in sample order for the result to be defined.
void dc_block(std::vector<float> & audio, float pole) {
    float previous_input = 0.0f;
    float previous_output = 0.0f;
    for (float & sample : audio) {
        const float input = sample;
        const float output = input - previous_input + pole * previous_output;
        previous_input = input;
        previous_output = output;
        sample = output;
    }
}

} // namespace

bool sano_run_vocos(SanoArch & arch, const SanoVoice & voice,
                    const std::vector<float> & mel, int64_t frames,
                    uint64_t noise_seed, std::vector<float> & audio,
                    std::string & error) {
    const SanoVocosWeights & w = voice.vocos;
    const int64_t mels = static_cast<int64_t>(w.mels);
    if (mel.size() != static_cast<size_t>(frames) * static_cast<size_t>(mels)) {
        error = "sanoTTS vocos decoder: mel has the wrong size";
        return false;
    }
    if (frames < 2) {
        error = "sanoTTS vocos decoder: a chunk needs at least two frames";
        return false;
    }

    const IstftPlan * plan = arch.istft_plan(w.n_fft, w.hop);
    if (plan == nullptr) {
        error = "sanoTTS vocos decoder: no iSTFT plan for n_fft="
              + std::to_string(w.n_fft) + " hop=" + std::to_string(w.hop);
        return false;
    }

    // Drawn before the graph is built: it is a graph input, and a seed that
    // cannot be honoured must fail before any allocation.
    std::vector<float> noise;
    if (!sano::seeded_noise(noise_seed, w.noise_ch, static_cast<size_t>(frames),
                            noise, error)) {
        error = "sanoTTS vocos decoder: " + error;
        return false;
    }

    const SanoGraphBudget budget = sano_vocos_budget(voice);
    const size_t bytes = arch.backend->graph_context_bytes(budget.tensors, budget.nodes);
    ggml_context * ctx = sano_graph_context(arch.graph_scratch, bytes, "vocos", error);
    if (ctx == nullptr) {
        return false;
    }

    Backend * backend = arch.backend;
    backend->set_input_tokens(static_cast<int>(frames));
    backend->set_active_label("sanotts_vocos");

    const int64_t bins = static_cast<int64_t>(w.bins);

    ggml_tensor * mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frames, mels);
    ggml_tensor * noise_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frames, w.noise_ch);
    ggml_tensor * mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, bins, 1);
    ggml_set_input(mel_in);
    ggml_set_input(noise_in);
    ggml_set_input(mask_in);

    ggml_tensor * x = ggml_add(
        ctx,
        stem_conv(ctx, w.embed_w, w.embed_b, mel_in, mels, static_cast<int>(w.embed_kernel)),
        stem_conv(ctx, w.noise_w, w.noise_b, noise_in, w.noise_ch,
                  static_cast<int>(w.embed_kernel)));
    x = sano_layer_norm(ctx, x, w.norm_w, w.norm_b);

    for (const auto & block : w.block_weights) {
        x = convnext_block(ctx, x, block, static_cast<int>(w.dw_kernel));
    }

    x = sano_layer_norm(ctx, x, w.final_norm_w, w.final_norm_b);

    ggml_tensor * head = ggml_mul_mat(ctx, w.head_w, x);
    ggml_mul_mat_set_prec(head, GGML_PREC_F32);
    head = ggml_add(ctx, head, w.head_b);   // [2 * bins, frames]

    ggml_tensor * log_magnitude = ggml_cont(
        ctx, ggml_view_2d(ctx, head, bins, frames, head->nb[1], 0));
    ggml_tensor * phase = ggml_cont(
        ctx, ggml_view_2d(ctx, head, bins, frames, head->nb[1],
                          static_cast<size_t>(bins) * head->nb[0]));

    // exp(min(lm, 60)) then min(mag, mag_clip), in that order: the first bound
    // is an overflow guard on the exponent, the second is the model's.
    ggml_tensor * magnitude = ggml_clamp(ctx, log_magnitude, -SANO_LOG_MAG_CLAMP,
                                         SANO_LOG_MAG_CLAMP);
    magnitude = ggml_clamp(ctx, ggml_exp(ctx, magnitude), 0.0f, w.mag_clip);
    magnitude = ggml_mul(ctx, magnitude, mask_in);

    ggml_tensor * real = ggml_cont(ctx, ggml_mul(ctx, magnitude, ggml_cos(ctx, phase)));
    ggml_tensor * imag = ggml_cont(ctx, ggml_mul(ctx, magnitude, ggml_sin(ctx, phase)));
    ggml_set_name(real, "sanotts_spectrum_real");
    ggml_set_name(imag, "sanotts_spectrum_imag");
    ggml_set_output(real);
    ggml_set_output(imag);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, budget.nodes, false);
    ggml_build_forward_expand(graph, real);
    ggml_build_forward_expand(graph, imag);

    backend->sched_reset();
    if (!backend->sched_alloc_graph(graph)) {
        ggml_free(ctx);
        error = "sanoTTS vocos graph allocation failed";
        return false;
    }
    if (!backend->apply_pending_inits()) {
        ggml_free(ctx);
        error = "sanoTTS vocos tensor initialisation failed";
        return false;
    }

    const std::vector<float> mask = bin_mask(w.bins);
    backend->tensor_set(mel_in, mel.data(), 0, ggml_nbytes(mel_in));
    backend->tensor_set(noise_in, noise.data(), 0, ggml_nbytes(noise_in));
    backend->tensor_set(mask_in, mask.data(), 0, ggml_nbytes(mask_in));

    if (backend->compute(ctx, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        error = "sanoTTS vocos graph compute failed";
        return false;
    }

    const size_t spectrum_size = static_cast<size_t>(bins) * static_cast<size_t>(frames);
    std::vector<float> spectrum_real(spectrum_size);
    std::vector<float> spectrum_imag(spectrum_size);
    backend->tensor_get(real, spectrum_real.data(), 0, spectrum_size * sizeof(float));
    backend->tensor_get(imag, spectrum_imag.data(), 0, spectrum_size * sizeof(float));
    ggml_free(ctx);

    ComplexSpectrumView view;
    view.real = spectrum_real.data();
    view.imag = spectrum_imag.data();
    view.bins = static_cast<size_t>(bins);
    view.frames = static_cast<size_t>(frames);
    view.bin_stride = 1;                // ne[0] is the bin axis
    view.frame_stride = bins;

    if (!istft(*plan, arch.istft_workspace, view, audio, error)) {
        return false;
    }

    dc_block(audio, w.dc_pole);
    return true;
}

} // namespace kokopop
