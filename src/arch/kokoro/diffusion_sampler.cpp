#include "arch/kokoro/diffusion_sampler.h"

#include "core/constants.h"
#include "arch/kokoro/diffusion_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

#include <ggml.h>

namespace kokopop {
namespace {

struct DiffusionTensor {
    const float * data = nullptr;
    int64_t ne0 = 0;
    int64_t ne1 = 1;
};

bool diffusion_tensor(
    KokoroArch & model,
    const std::string & name,
    DiffusionTensor & out,
    std::string & error) {
    ggml_tensor * t = model.cached_tensor(name);
    if (t == nullptr) {
        error = "missing Kokoro diffusion tensor: " + name;
        return false;
    }

    auto it = model.diffusion_f32.find(name);
    if (it == model.diffusion_f32.end()) {
        std::vector<float> data;
        if (!tensor_to_f32(*model.backend, t, data)) {
            error = "failed to read Kokoro diffusion tensor: " + name;
            return false;
        }
        it = model.diffusion_f32.emplace(name, std::move(data)).first;
    }

    out.data = it->second.data();
    out.ne0 = t->ne[0];
    out.ne1 = t->ne[1];
    return true;
}

float gelu_exact(float x) {
    constexpr float inv_sqrt2 = 0.7071067811865475244f;
    return 0.5f * x * (1.0f + std::erf(x * inv_sqrt2));
}

bool diffusion_linear_rows(
    KokoroArch & model,
    const std::string & prefix,
    const std::vector<float> & input,
    int64_t rows,
    int64_t in_dim,
    std::vector<float> & out,
    int64_t & out_dim,
    std::string & error,
    bool bias_required = true) {
    DiffusionTensor w;
    if (!diffusion_tensor(model, prefix + ".weight", w, error)) return false;
    if (w.ne0 != in_dim) {
        error = "Kokoro diffusion tensor shape mismatch: " + prefix + ".weight";
        return false;
    }
    out_dim = w.ne1;
    out.assign(static_cast<size_t>(rows * out_dim), 0.0f);

    DiffusionTensor b;
    const bool has_bias = model.cached_tensor(prefix + ".bias") != nullptr;
    if (bias_required && !has_bias) {
        error = "missing Kokoro diffusion tensor: " + prefix + ".bias";
        return false;
    }
    if (has_bias && !diffusion_tensor(model, prefix + ".bias", b, error)) return false;

    for (int64_t r = 0; r < rows; ++r) {
        const float * x = input.data() + static_cast<size_t>(r * in_dim);
        float * y = out.data() + static_cast<size_t>(r * out_dim);
        for (int64_t oc = 0; oc < out_dim; ++oc) {
            const float * wr = w.data + static_cast<size_t>(oc * in_dim);
            const float bias = has_bias ? b.data[oc] : 0.0f;
            y[oc] = bias + diffusion_kernel::dot(wr, x, in_dim);
        }
    }
    return true;
}

bool diffusion_layer_norm_rows(
    KokoroArch & model,
    const std::string & prefix,
    const std::vector<float> & input,
    int64_t rows,
    int64_t dim,
    std::vector<float> & out,
    std::string & error) {
    DiffusionTensor gamma;
    DiffusionTensor beta;
    if (!diffusion_tensor(model, prefix + ".weight", gamma, error) ||
        !diffusion_tensor(model, prefix + ".bias", beta, error)) {
        return false;
    }
    if (gamma.ne0 != dim || beta.ne0 != dim) {
        error = "Kokoro diffusion layer norm shape mismatch: " + prefix;
        return false;
    }

    out.resize(input.size());
    for (int64_t r = 0; r < rows; ++r) {
        const float * x = input.data() + static_cast<size_t>(r * dim);
        float * y = out.data() + static_cast<size_t>(r * dim);
        const float mean = diffusion_kernel::sum(x, dim) / static_cast<float>(dim);
        const float var = diffusion_kernel::squared_diff_sum(x, dim, mean) / static_cast<float>(dim);
        const float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        diffusion_kernel::layer_norm_affine(y, x, gamma.data, beta.data, dim, mean, inv_std);
    }
    return true;
}

// AdaLayerNorm (StyleTransformer1d): layer-norm without affine, then modulate
// with gamma/beta produced from the style vector s via a learned fc:
//   h = fc(s); gamma, beta = chunk(h); out = (1 + gamma) * layernorm(x) + beta
// fc lives at `prefix + ".fc"` (weight [2*dim, style_dim], bias [2*dim]).
bool diffusion_ada_layer_norm_rows(
    KokoroArch & model,
    const std::string & prefix,
    const std::vector<float> & input,
    int64_t rows,
    int64_t dim,
    const std::vector<float> & style,
    std::vector<float> & out,
    std::string & error) {
    std::vector<float> h;
    int64_t h_dim = 0;
    if (!diffusion_linear_rows(model, prefix + ".fc", style, 1,
                               static_cast<int64_t>(style.size()), h, h_dim, error)) {
        return false;
    }
    if (h_dim != 2 * dim) {
        error = "Kokoro diffusion AdaLayerNorm shape mismatch: " + prefix;
        return false;
    }
    const float * gamma = h.data();
    const float * beta = h.data() + dim;

    out.resize(input.size());
    for (int64_t r = 0; r < rows; ++r) {
        const float * x = input.data() + static_cast<size_t>(r * dim);
        float * y = out.data() + static_cast<size_t>(r * dim);
        const float mean = diffusion_kernel::sum(x, dim) / static_cast<float>(dim);
        const float var = diffusion_kernel::squared_diff_sum(x, dim, mean) / static_cast<float>(dim);
        const float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        diffusion_kernel::ada_layer_norm(y, x, gamma, beta, dim, mean, inv_std);
    }
    return true;
}

// One transformer block. When *style* is non-null the block is a
// StyleTransformerBlock (AdaLayerNorm conditioned on the style vector);
// otherwise it is a plain TransformerBlock with a static LayerNorm.
bool diffusion_attention_block(
    KokoroArch & model,
    const std::string & prefix,
    std::vector<float> & x,
    int64_t rows,
    int64_t features,
    const std::vector<float> * style,
    std::string & error) {
    std::vector<float> normed_q;
    std::vector<float> normed_kv_storage;
    const std::vector<float> * normed_kv = &normed_q;
    if (style != nullptr) {
        if (!diffusion_ada_layer_norm_rows(model, prefix + ".attention.norm",
                                           x, rows, features, *style, normed_q, error) ||
            !diffusion_ada_layer_norm_rows(model, prefix + ".attention.norm_context",
                                           x, rows, features, *style, normed_kv_storage, error)) {
            return false;
        }
        normed_kv = &normed_kv_storage;
    } else if (!diffusion_layer_norm_rows(model, prefix + ".attention.norm",
                                          x, rows, features, normed_q, error)) {
        return false;
    }

    std::vector<float> q;
    std::vector<float> kv;
    int64_t q_dim = 0;
    int64_t kv_dim = 0;
    if (!diffusion_linear_rows(model, prefix + ".attention.to_q", normed_q, rows, features, q, q_dim, error, false) ||
        !diffusion_linear_rows(model, prefix + ".attention.to_kv", *normed_kv, rows, features, kv, kv_dim, error, false)) {
        return false;
    }
    if ((kv_dim % 2) != 0 || q_dim * 2 != kv_dim || (q_dim % 8) != 0) {
        error = "unsupported Kokoro diffusion attention dimensions";
        return false;
    }

    constexpr int64_t num_heads = 8;
    const int64_t head_dim = q_dim / num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> mid(static_cast<size_t>(rows * q_dim), 0.0f);
    std::vector<float> logits(static_cast<size_t>(rows));
    std::vector<float> probs(static_cast<size_t>(rows));

    for (int64_t h = 0; h < num_heads; ++h) {
        for (int64_t i = 0; i < rows; ++i) {
            float max_logit = -std::numeric_limits<float>::infinity();
            const float * qi = q.data() + static_cast<size_t>(i * q_dim + h * head_dim);
            for (int64_t j = 0; j < rows; ++j) {
                const float * kj = kv.data() + static_cast<size_t>(j * kv_dim + h * head_dim);
                const float v = diffusion_kernel::dot(qi, kj, head_dim) * scale;
                logits[static_cast<size_t>(j)] = v;
                max_logit = std::max(max_logit, v);
            }
            double denom = 0.0;
            for (int64_t j = 0; j < rows; ++j) {
                const double e = std::exp(static_cast<double>(logits[static_cast<size_t>(j)] - max_logit));
                probs[static_cast<size_t>(j)] = static_cast<float>(e);
                denom += e;
            }
            float * out = mid.data() + static_cast<size_t>(i * q_dim + h * head_dim);
            diffusion_kernel::attention_weighted_sum(
                out,
                kv.data() + static_cast<size_t>(q_dim + h * head_dim),
                probs.data(),
                static_cast<float>(1.0 / denom),
                rows,
                kv_dim,
                head_dim);
        }
    }

    std::vector<float> attn_out;
    int64_t attn_out_dim = 0;
    if (!diffusion_linear_rows(model, prefix + ".attention.attention.to_out", mid, rows, q_dim, attn_out, attn_out_dim, error)) {
        return false;
    }
    if (attn_out_dim != features) {
        error = "Kokoro diffusion attention output shape mismatch";
        return false;
    }
    diffusion_kernel::add_inplace(x.data(), attn_out.data(), x.size());

    std::vector<float> ff1;
    std::vector<float> ff2;
    int64_t ff_dim = 0;
    int64_t ff_out_dim = 0;
    if (!diffusion_linear_rows(model, prefix + ".feed_forward.0", x, rows, features, ff1, ff_dim, error)) {
        return false;
    }
    for (float & v : ff1) v = gelu_exact(v);
    if (!diffusion_linear_rows(model, prefix + ".feed_forward.2", ff1, rows, ff_dim, ff2, ff_out_dim, error)) {
        return false;
    }
    if (ff_out_dim != features) {
        error = "Kokoro diffusion feed-forward output shape mismatch";
        return false;
    }
    diffusion_kernel::add_inplace(x.data(), ff2.data(), x.size());
    return true;
}

bool diffusion_transformer_run(
    KokoroArch & model,
    const std::vector<float> & x_style,
    float c_noise,
    const std::vector<float> & embedding,
    int64_t n_tokens,
    int64_t embedding_dim,
    const std::vector<float> & style,
    std::vector<float> & out,
    std::string & error,
    bool use_fixed_embedding) {
    constexpr int64_t style_dim = 256;
    const int64_t features = style_dim + embedding_dim;

    // StyleTransformer1d (multispeaker): AdaLayerNorm blocks conditioned on the
    // style vector + a to_features branch folded into the mapping. Detected by
    // the presence of the per-block AdaLayerNorm fc weights.
    const bool is_style =
        model.cached_tensor("kokopop.diffusion.blocks.0.attention.norm.fc.weight") != nullptr;
    const std::vector<float> * block_style = is_style ? &style : nullptr;
    if (static_cast<int64_t>(x_style.size()) != style_dim ||
        static_cast<int64_t>(embedding.size()) != n_tokens * embedding_dim) {
        error = "invalid Kokoro diffusion transformer inputs";
        return false;
    }

    std::vector<float> fixed;
    const float * embedding_data = embedding.data();
    if (use_fixed_embedding) {
        DiffusionTensor fixed_w;
        if (!diffusion_tensor(model, "kokopop.diffusion.fixed_embedding.embedding.weight", fixed_w, error)) {
            return false;
        }
        if (fixed_w.ne0 != embedding_dim || fixed_w.ne1 < n_tokens) {
            error = "Kokoro diffusion fixed embedding shape mismatch";
            return false;
        }
        fixed.assign(fixed_w.data, fixed_w.data + static_cast<size_t>(n_tokens * embedding_dim));
        embedding_data = fixed.data();
    }

    std::vector<float> time_emb(257);
    time_emb[0] = c_noise;
    DiffusionTensor time_weights;
    if (!diffusion_tensor(model, "kokopop.diffusion.to_time.0.weights", time_weights, error)) {
        return false;
    }
    if (time_weights.ne0 * 2 + 1 != 257) {
        error = "Kokoro diffusion time embedding shape mismatch";
        return false;
    }
    constexpr float two_pi = 6.2831853071795864769f;
    for (int64_t i = 0; i < time_weights.ne0; ++i) {
        const float phase = c_noise * time_weights.data[i] * two_pi;
        time_emb[static_cast<size_t>(1 + i)] = std::sin(phase);
        time_emb[static_cast<size_t>(1 + time_weights.ne0 + i)] = std::cos(phase);
    }

    std::vector<float> mapping;
    int64_t mapping_dim = 0;
    if (!diffusion_linear_rows(model, "kokopop.diffusion.to_time.1", time_emb, 1, 257, mapping, mapping_dim, error)) {
        return false;
    }
    for (float & v : mapping) v = gelu_exact(v);

    // StyleTransformer1d folds the style vector into the mapping via to_features:
    //   mapping = to_mapping( to_time(time) + gelu(to_features(s)) )
    if (is_style) {
        std::vector<float> feat;
        int64_t feat_dim = 0;
        if (!diffusion_linear_rows(model, "kokopop.diffusion.to_features.0",
                                   style, 1, static_cast<int64_t>(style.size()),
                                   feat, feat_dim, error)) {
            return false;
        }
        if (feat_dim != mapping_dim) {
            error = "Kokoro diffusion to_features dimension mismatch";
            return false;
        }
        for (float & v : feat) v = gelu_exact(v);
        for (size_t i = 0; i < mapping.size(); ++i) mapping[i] += feat[i];
    }

    if (model.cached_tensor("kokopop.diffusion.to_mapping.0.weight") != nullptr) {
        std::vector<float> tmp;
        int64_t tmp_dim = 0;
        if (!diffusion_linear_rows(model, "kokopop.diffusion.to_mapping.0", mapping, 1, mapping_dim, tmp, tmp_dim, error)) {
            return false;
        }
        for (float & v : tmp) v = gelu_exact(v);
        if (!diffusion_linear_rows(model, "kokopop.diffusion.to_mapping.2", tmp, 1, tmp_dim, mapping, mapping_dim, error)) {
            return false;
        }
        for (float & v : mapping) v = gelu_exact(v);
    }

    if (mapping_dim != features) {
        error = "Kokoro diffusion mapping dimension mismatch";
        return false;
    }

    std::vector<float> x(static_cast<size_t>(n_tokens * features));
    for (int64_t t = 0; t < n_tokens; ++t) {
        float * row = x.data() + static_cast<size_t>(t * features);
        for (int64_t i = 0; i < style_dim; ++i) row[i] = x_style[static_cast<size_t>(i)];
        std::memcpy(row + style_dim,
                    embedding_data + static_cast<size_t>(t * embedding_dim),
                    static_cast<size_t>(embedding_dim) * sizeof(float));
        // Plain Transformer1d adds the mapping once up front; StyleTransformer1d
        // re-adds it before every block (see run() in Modules/diffusion).
        if (!is_style) {
            diffusion_kernel::add_inplace(row, mapping.data(), static_cast<size_t>(features));
        }
    }

    for (int block = 0; block < 3; ++block) {
        if (is_style) {
            for (int64_t t = 0; t < n_tokens; ++t) {
                float * row = x.data() + static_cast<size_t>(t * features);
                diffusion_kernel::add_inplace(row, mapping.data(), static_cast<size_t>(features));
            }
        }
        if (!diffusion_attention_block(model, "kokopop.diffusion.blocks." + std::to_string(block),
                                       x, n_tokens, features, block_style, error)) {
            return false;
        }
    }

    std::vector<float> mean(static_cast<size_t>(features), 0.0f);
    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * row = x.data() + static_cast<size_t>(t * features);
        diffusion_kernel::add_inplace(mean.data(), row, mean.size());
    }
    for (float & v : mean) v /= static_cast<float>(n_tokens);

    int64_t out_dim = 0;
    if (!diffusion_linear_rows(model, "kokopop.diffusion.to_out.1", mean, 1, features, out, out_dim, error)) {
        return false;
    }
    if (out_dim != style_dim) {
        error = "Kokoro diffusion output dimension mismatch";
        return false;
    }
    return true;
}

std::vector<float> karras_sigmas(int steps) {
    constexpr float sigma_min = 0.0001f;
    constexpr float sigma_max = 3.0f;
    constexpr float rho = 9.0f;
    const float inv_rho = 1.0f / rho;
    std::vector<float> sigmas(static_cast<size_t>(steps + 1), 0.0f);
    for (int i = 0; i < steps; ++i) {
        const float ramp = steps > 1 ? static_cast<float>(i) / static_cast<float>(steps - 1) : 1.0f;
        sigmas[static_cast<size_t>(i)] = std::pow(
            std::pow(sigma_max, inv_rho) +
            ramp * (std::pow(sigma_min, inv_rho) - std::pow(sigma_max, inv_rho)),
            rho);
    }
    return sigmas;
}

bool diffusion_denoise(
    KokoroArch & model,
    const std::vector<float> & x_noisy,
    float sigma,
    const std::vector<float> & embedding,
    int64_t n_tokens,
    int64_t embedding_dim,
    const std::vector<float> & style,
    float embedding_scale,
    std::vector<float> & out,
    std::string & error) {
    const float sigma_data = model.diffusion_sigma_data;
    const float denom = std::sqrt(sigma * sigma + sigma_data * sigma_data);
    const float c_skip = (sigma_data * sigma_data) / (sigma * sigma + sigma_data * sigma_data);
    const float c_out = sigma * sigma_data / denom;
    const float c_in = 1.0f / denom;
    const float c_noise = std::log(sigma) * 0.25f;

    std::vector<float> x_in(x_noisy.size());
    diffusion_kernel::scale(x_in.data(), x_noisy.data(), c_in, x_noisy.size());

    std::vector<float> pred;
    if (!diffusion_transformer_run(model, x_in, c_noise, embedding, n_tokens,
                                   embedding_dim, style, pred, error, false)) {
        return false;
    }
    if (embedding_scale != 1.0f) {
        std::vector<float> masked;
        if (!diffusion_transformer_run(model, x_in, c_noise, embedding, n_tokens,
                                       embedding_dim, style, masked, error, true)) {
            return false;
        }
        diffusion_kernel::classifier_free_guidance(pred.data(), masked.data(), embedding_scale, pred.size());
    }

    out.resize(x_noisy.size());
    diffusion_kernel::denoise_combine(out.data(), x_noisy.data(), pred.data(), c_skip, c_out, x_noisy.size());
    return true;
}

} // namespace

bool apply_diffusion_style_options(
    KokoroArch & model,
    const KokoroDiffusionOptions * options,
    std::vector<float> & style,
    const std::vector<float> & embedding,
    int64_t n_tokens,
    int64_t embedding_dim,
    std::string & error) {
    if (options == nullptr || !options->enabled) {
        return true;
    }
    const float alpha = std::clamp(options->alpha, 0.0f, 1.0f);
    const float beta = std::clamp(options->beta, 0.0f, 1.0f);
    if (options->steps <= 0 || (alpha == 0.0f && beta == 0.0f)) {
        return true;
    }
    if (model.cached_tensor("kokopop.diffusion.to_out.1.weight") == nullptr) {
        error = "diffusion style sampling requested, but this GGUF has no diffusion tensors";
        return false;
    }
    if (style.size() < 256 || embedding.empty() || n_tokens <= 0 || embedding_dim <= 0) {
        error = "invalid Kokoro diffusion style inputs";
        return false;
    }

    // Reference speaker style conditioning the StyleTransformer1d. Snapshot it
    // before the style gets blended with the sampled vector below. Ignored by
    // the plain Transformer1d path.
    const std::vector<float> ref_style(style.begin(), style.begin() + 256);

    const int steps = std::clamp(options->steps, 2, 50);
    std::mt19937 rng(options->seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(256);
    for (float & v : x) v = normal(rng);

    const std::vector<float> sigmas = karras_sigmas(steps);
    diffusion_kernel::scale(x.data(), x.data(), sigmas.front(), x.size());

    for (int i = 0; i < steps - 1; ++i) {
        const float sigma = sigmas[static_cast<size_t>(i)];
        const float sigma_next = sigmas[static_cast<size_t>(i + 1)];
        const float sigma_up = sigma_next == 0.0f ? 0.0f :
            std::sqrt(std::max(0.0f, sigma_next * sigma_next * (sigma * sigma - sigma_next * sigma_next) / (sigma * sigma)));
        const float sigma_down = std::sqrt(std::max(0.0f, sigma_next * sigma_next - sigma_up * sigma_up));
        const float sigma_mid = 0.5f * (sigma + sigma_down);

        std::vector<float> denoised;
        if (!diffusion_denoise(model, x, sigma, embedding, n_tokens, embedding_dim,
                               ref_style, options->embedding_scale, denoised, error)) {
            return false;
        }
        std::vector<float> x_mid(x.size());
        diffusion_kernel::euler_midpoint(x_mid.data(), x.data(), denoised.data(), sigma, sigma_mid, x.size());

        std::vector<float> denoised_mid;
        if (!diffusion_denoise(model, x_mid, sigma_mid, embedding, n_tokens, embedding_dim,
                               ref_style, options->embedding_scale, denoised_mid, error)) {
            return false;
        }
        std::vector<float> noise(x.size());
        for (float & v : noise) v = normal(rng);
        diffusion_kernel::euler_update_with_noise(
            x.data(), x_mid.data(), denoised_mid.data(), noise.data(),
            sigma_mid, sigma_down, sigma, sigma_up, x.size());
    }

    for (float & v : x) v = std::clamp(v, -1.0f, 1.0f);
    diffusion_kernel::blend_style(style.data(), x.data(), alpha, beta, 128);
    return true;
}

} // namespace kokopop
