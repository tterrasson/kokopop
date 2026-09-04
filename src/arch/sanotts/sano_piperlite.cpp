// sanoTTS piperlite decoder: latent -> waveform, in one ggml graph.
//
// Three upsampling stages take the 192-channel latent to 24 channels at
// 8 x 8 x 4 = 256 samples per frame, each followed by a residual bank whose
// three branches are averaged. A single 1x1-ish post convolution and a tanh
// produce the waveform; some voices then run a small learned post filter over
// it.
//
// Reference: `pypkg/sanotts/models.py::decoder_forward` (MIT), commit
// 939d982b9faa54cbcf5d24cc878f5cd514b2646e. See THIRD_PARTY.md.
//
// Everything here is in the `[T, C]` layout: time in `ne[0]`, one channel
// contiguous, which is what every ggml convolution consumes.

#include "arch/sanotts/sano_arch.h"
#include "arch/sanotts/sano_graph.h"
#include "backend/backend.h"

#include <string>
#include <vector>

#include <ggml.h>

namespace kokopop {

namespace {

/// One branch: `y2 = y1 + conv2(lrelu(y1)); y1 = x + conv1(lrelu(x))`.
ggml_tensor * residual_branch(ggml_context * ctx, const Backend * backend,
                              ggml_tensor * x, const SanoPiperBranch & branch,
                              int64_t channels) {
    ggml_tensor * t = sano_leaky_relu(ctx, backend, x, 0.1f);
    ggml_tensor * u = sano_conv1d(ctx, branch.conv1_w, t, channels, branch.kernel,
                                  branch.dilation1 * (branch.kernel / 2), branch.dilation1);
    ggml_tensor * y1 = ggml_add(ctx, sano_add_channel_bias(ctx, u, branch.conv1_b), x);

    ggml_tensor * t2 = sano_leaky_relu(ctx, backend, y1, 0.1f);
    ggml_tensor * u2 = sano_conv1d(ctx, branch.conv2_w, t2, channels, branch.kernel,
                                   branch.dilation2 * (branch.kernel / 2), branch.dilation2);
    return ggml_add(ctx, sano_add_channel_bias(ctx, u2, branch.conv2_b), y1);
}

ggml_tensor * residual_bank(ggml_context * ctx, const Backend * backend,
                            ggml_tensor * x, const SanoPiperStage & stage,
                            int64_t channels) {
    ggml_tensor * sum = nullptr;
    for (const uint32_t index : stage.branches) {
        ggml_tensor * branch = residual_branch(ctx, backend, x, stage.bank[index], channels);
        sum = sum == nullptr ? branch : ggml_add(ctx, sum, branch);
    }
    return ggml_scale(ctx, sum, 1.0f / static_cast<float>(stage.branches.size()));
}

/// `tanh(audio + scale * out_conv(r))` with `r` refined by `layers` units.
ggml_tensor * post_filter(ggml_context * ctx, const Backend * backend,
                          ggml_tensor * audio, const SanoPiperliteWeights & w) {
    const int64_t channels = static_cast<int64_t>(w.post_filter_channels);
    const int kernel = static_cast<int>(w.post_filter_kernel);

    ggml_tensor * r = sano_conv1d(ctx, w.post_filter_in_w, audio, 1, kernel, kernel / 2, 1);
    r = sano_add_channel_bias(ctx, r, w.post_filter_in_b);

    for (const auto & unit : w.post_filter_units) {
        ggml_tensor * t = sano_leaky_relu(ctx, backend, r, 0.1f);
        ggml_tensor * u = sano_conv1d(ctx, unit.conv1_w, t, channels, unit.kernel,
                                      unit.dilation1 * (unit.kernel / 2), unit.dilation1);
        u = sano_add_channel_bias(ctx, u, unit.conv1_b);

        ggml_tensor * t2 = sano_leaky_relu(ctx, backend, u, 0.1f);
        ggml_tensor * u2 = sano_conv1d(ctx, unit.conv2_w, t2, channels, unit.kernel,
                                       unit.kernel / 2, 1);
        u2 = sano_add_channel_bias(ctx, u2, unit.conv2_b);

        r = ggml_add(ctx, r, ggml_scale(ctx, u2, unit.scale));
    }

    ggml_tensor * out = sano_conv1d(ctx, w.post_filter_out_w, r, channels, kernel, kernel / 2, 1);
    out = sano_add_channel_bias(ctx, out, w.post_filter_out_b);

    return ggml_tanh(ctx, ggml_add(ctx, audio, ggml_scale(ctx, out, w.post_filter_scale)));
}

} // namespace

bool sano_run_piperlite(SanoArch & arch, const SanoVoice & voice,
                        const std::vector<float> & latent, int64_t frames,
                        std::vector<float> & audio, std::string & error) {
    const SanoPiperliteWeights & w = voice.piperlite;
    const int64_t in_channels = static_cast<int64_t>(voice.ac.out_channels);
    if (latent.size() != static_cast<size_t>(frames) * static_cast<size_t>(in_channels)) {
        error = "sanoTTS piperlite decoder: latent has the wrong size";
        return false;
    }

    const SanoGraphBudget budget = sano_piperlite_budget(voice);
    const size_t bytes = arch.backend->graph_context_bytes(budget.tensors, budget.nodes);
    ggml_context * ctx = sano_graph_context(arch.graph_scratch, bytes, "piperlite", error);
    if (ctx == nullptr) {
        return false;
    }

    Backend * backend = arch.backend;
    backend->set_input_tokens(static_cast<int>(frames));
    backend->set_active_label("sanotts_piperlite");

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frames, in_channels);
    ggml_set_input(input);

    int64_t channels = static_cast<int64_t>(w.channels[0]);
    ggml_tensor * x = sano_conv1d(ctx, w.pre_w, input, in_channels,
                                  static_cast<int>(w.pre_kernel),
                                  static_cast<int>(w.pre_kernel) / 2, 1);
    x = sano_add_channel_bias(ctx, x, w.pre_b);

    for (size_t s = 0; s < w.stages.size(); ++s) {
        const SanoPiperStage & stage = w.stages[s];
        x = sano_leaky_relu(ctx, backend, x, 0.1f);
        x = sano_conv_transpose1d(ctx, stage.up_w, x, stage.up_stride, stage.up_padding);
        x = sano_add_channel_bias(ctx, x, stage.up_b);
        channels = static_cast<int64_t>(w.channels[s + 1]);
        x = residual_bank(ctx, backend, x, stage, channels);
    }

    x = sano_leaky_relu(ctx, backend, x, 0.01f);
    x = sano_conv1d(ctx, w.post_w, x, channels, static_cast<int>(w.post_kernel),
                    static_cast<int>(w.post_kernel) / 2, 1);
    x = sano_add_channel_bias(ctx, x, w.post_b);
    x = ggml_tanh(ctx, x);

    if (w.post_filter_channels > 0) {
        x = post_filter(ctx, backend, x, w);
    }

    x = ggml_cont(ctx, x);
    ggml_set_name(x, "sanotts_piperlite_audio");
    ggml_set_output(x);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, budget.nodes, false);
    ggml_build_forward_expand(graph, x);

    backend->sched_reset();
    if (!backend->sched_alloc_graph(graph)) {
        ggml_free(ctx);
        error = "sanoTTS piperlite graph allocation failed";
        return false;
    }
    if (!backend->apply_pending_inits()) {
        ggml_free(ctx);
        error = "sanoTTS piperlite tensor initialisation failed";
        return false;
    }

    backend->tensor_set(input, latent.data(), 0, ggml_nbytes(input));

    if (backend->compute(ctx, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        error = "sanoTTS piperlite graph compute failed";
        return false;
    }

    audio.resize(static_cast<size_t>(ggml_nelements(x)));
    backend->tensor_get(x, audio.data(), 0, audio.size() * sizeof(float));
    ggml_free(ctx);
    return true;
}

} // namespace kokopop
