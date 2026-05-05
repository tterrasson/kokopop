#include "model/model.h"

#include "core/constants.h"
#include "core/utf8.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <thread>
#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>
#include <unordered_map>

namespace kokopop {
namespace {

int find_key(gguf_context * ctx, const char * key) {
    return gguf_find_key(ctx, key);
}

bool get_u32(gguf_context * ctx, const char * key, uint32_t & out) {
    const int idx = find_key(ctx, key);
    if (idx < 0) {
        return false;
    }
    out = gguf_get_val_u32(ctx, idx);
    return true;
}

bool get_bool(gguf_context * ctx, const char * key, bool & out) {
    const int idx = find_key(ctx, key);
    if (idx < 0) {
        return false;
    }
    out = gguf_get_val_bool(ctx, idx);
    return true;
}

bool get_string_array(gguf_context * ctx, const char * key, std::vector<std::string> & out) {
    const int idx = find_key(ctx, key);
    if (idx < 0) {
        return false;
    }
    const int64_t n = gguf_get_arr_n(ctx, idx);
    out.clear();
    out.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const char * s = gguf_get_arr_str(ctx, idx, i);
        out.emplace_back(s ? s : "");
    }
    return true;
}

template <typename Fn>
bool for_each_tensor(gguf_context * meta, ggml_context * weights, Fn fn) {
    const int n = gguf_get_n_tensors(meta);
    for (int i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(meta, i);
        ggml_tensor * tensor = ggml_get_tensor(weights, name);
        if (tensor == nullptr) {
            return false;
        }
        fn(name, tensor);
    }
    return true;
}

bool load_tensor_data_to_backend(
    const std::string & path,
    gguf_context * meta,
    ggml_context * weights,
    Backend & backend,
    std::string & error) {
    std::FILE * file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        error = "failed to reopen GGUF model for tensor data: " + path;
        return false;
    }

    constexpr size_t buf_size = KOKOPOP_IO_BUF_SIZE;
    std::vector<uint8_t> buffer(buf_size);
    const int64_t n_tensors = gguf_get_n_tensors(meta);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(meta, i);
        ggml_tensor * tensor = ggml_get_tensor(weights, name);
        if (tensor == nullptr) {
            std::fclose(file);
            error = std::string("GGUF tensor missing from weight context: ") + (name ? name : "");
            return false;
        }

        const size_t offset = gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, i);
        if (std::fseek(file, static_cast<long>(offset), SEEK_SET) != 0) {
            std::fclose(file);
            error = std::string("failed to seek GGUF tensor data: ") + (name ? name : "");
            return false;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        for (size_t pos = 0; pos < nbytes; pos += buf_size) {
            const size_t chunk = std::min(buf_size, nbytes - pos);
            if (std::fread(buffer.data(), 1, chunk, file) != chunk) {
                std::fclose(file);
                error = std::string("failed to read GGUF tensor data: ") + (name ? name : "");
                return false;
            }
            backend.tensor_set(tensor, buffer.data(), pos, chunk);
        }
    }

    std::fclose(file);
    return true;
}

} // namespace

bool tensor_to_f32(Backend & backend, ggml_tensor * tensor, std::vector<float> & out) {
    if (tensor == nullptr) {
        return false;
    }
    const int64_t n = ggml_nelements(tensor);
    out.resize(static_cast<size_t>(n));

    if (tensor->type == GGML_TYPE_F32) {
        backend.tensor_get(tensor, out.data(), 0, static_cast<size_t>(n) * sizeof(float));
        return true;
    }
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> in(static_cast<size_t>(n));
        backend.tensor_get(tensor, in.data(), 0, in.size() * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<size_t>(i)] = ggml_fp16_to_fp32(in[static_cast<size_t>(i)]);
        }
        return true;
    }

    // Quantized types: use ggml_type_traits->to_float for dequantization
    const struct ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (traits && traits->to_float) {
        const size_t nbytes = ggml_nbytes(tensor);
        std::vector<uint8_t> raw(nbytes);
        backend.tensor_get(tensor, raw.data(), 0, nbytes);
        traits->to_float(raw.data(), out.data(), n);
        return true;
    }
    return false;
}

bool GeneratorResblockWeights::valid() const {
    for (int i = 0; i < 3; ++i) {
        if (!adain1[static_cast<size_t>(i)].valid() ||
            !adain2[static_cast<size_t>(i)].valid() ||
            alpha1[static_cast<size_t>(i)] == nullptr ||
            alpha2[static_cast<size_t>(i)] == nullptr ||
            convs1_w[static_cast<size_t>(i)] == nullptr ||
            convs1_b[static_cast<size_t>(i)] == nullptr ||
            convs2_w[static_cast<size_t>(i)] == nullptr ||
            convs2_b[static_cast<size_t>(i)] == nullptr) {
            return false;
        }
    }
    return true;
}

uint8_t * ScratchArena::data(size_t required) {
    if (bytes.size() < required) {
        bytes.resize(required);
    }
    high_water = std::max(high_water, required);
    return bytes.data();
}

size_t ScratchArena::capacity() const {
    return bytes.size();
}

Model::~Model() {
    // Backend cleanup is handled by std::unique_ptr<Backend>.
    if (weight_buffer != nullptr) {
        ggml_backend_buffer_free(weight_buffer);
        weight_buffer = nullptr;
    }
    if (gguf_ctx != nullptr) {
        gguf_free(gguf_ctx);
        gguf_ctx = nullptr;
    }
    if (weight_ctx != nullptr) {
        ggml_free(weight_ctx);
        weight_ctx = nullptr;
    }
}

bool Model::tokenize_phonemes(const std::string & phonemes, std::vector<uint32_t> & ids, std::string & error) const {
    ids.clear();
    // Two sentinel tokens (0) + one slot per UTF-8 code point is the upper bound.
    ids.reserve(phonemes.size() + 2);
    ids.push_back(0);
    size_t off = 0;
    std::string_view ch;
    while (utf8_next(phonemes, off, ch)) {
        // 5.1 — Direct string_view lookup: no temporary std::string allocation.
        const auto it = token_to_id.find(ch);
        if (it == token_to_id.end()) {
            continue;
        }
        ids.push_back(it->second);
        if (ids.size() > 511) {
            error = "phoneme sequence exceeds Kokoro context length";
            return false;
        }
    }
    if (off != phonemes.size()) {
        error = "phoneme sequence is not valid UTF-8";
        return false;
    }
    ids.push_back(0);
    return true;
}

ggml_tensor * Model::tensor(const std::string & logical_name) const {
    const auto it = tensors.find(logical_name);
    return it == tensors.end() ? nullptr : it->second;
}

void Model::preload_tensor_cache() {
    clear_tensor_cache();
    tensor_cache = tensors;

    // 5.3 — Use string_view prefixes + small local std::string keys to avoid
    // constructing many temporary strings during repeated tensor() lookups.
    // tensor() and cached_tensor() accept string_view, so the suffix
    // concatenation only happens once per call, not inside the map lookup.

    auto cache_adain = [this](const std::string & prefix) {
        AdaIn1dWeights w;
        std::string key;
        key.reserve(prefix.size() + 20);

        key = prefix; key += ".norm.weight";   w.norm_w  = tensor(key);
        key = prefix; key += ".norm.bias";     w.norm_b  = tensor(key);
        key = prefix; key += ".fc.gamma.weight"; w.gamma_w = tensor(key);
        key = prefix; key += ".fc.gamma.bias";  w.gamma_b = tensor(key);
        key = prefix; key += ".fc.beta.weight";  w.beta_w  = tensor(key);
        key = prefix; key += ".fc.beta.bias";    w.beta_b  = tensor(key);
        if (w.valid()) {
            adain_1d_weights.emplace(prefix, w);
        }
    };

    auto cache_generator_resblock = [this, &cache_adain](const std::string & prefix, int kernel_size) {
        GeneratorResblockWeights w;
        std::string key;
        key.reserve(prefix.size() + 30);
        for (int i = 0; i < 3; ++i) {
            const char idx = static_cast<char>('0' + i);

            // Cache adain1 and adain2 sub-blocks
            key = prefix; key += ".adain1."; key += idx;
            cache_adain(key);
            const auto a1 = adain_1d_weights.find(key);
            if (a1 != adain_1d_weights.end()) {
                w.adain1[static_cast<size_t>(i)] = a1->second;
            }

            key = prefix; key += ".adain2."; key += idx;
            cache_adain(key);
            const auto a2 = adain_1d_weights.find(key);
            if (a2 != adain_1d_weights.end()) {
                w.adain2[static_cast<size_t>(i)] = a2->second;
            }

            key = prefix; key += ".alpha1."; key += idx;
            w.alpha1[static_cast<size_t>(i)] = tensor(key);
            key = prefix; key += ".alpha2."; key += idx;
            w.alpha2[static_cast<size_t>(i)] = tensor(key);
            key = prefix; key += ".convs1."; key += idx; key += ".weight";
            w.convs1_w[static_cast<size_t>(i)] = tensor(key);
            key = prefix; key += ".convs1."; key += idx; key += ".bias";
            w.convs1_b[static_cast<size_t>(i)] = tensor(key);
            key = prefix; key += ".convs2."; key += idx; key += ".weight";
            w.convs2_w[static_cast<size_t>(i)] = tensor(key);
            key = prefix; key += ".convs2."; key += idx; key += ".bias";
            w.convs2_b[static_cast<size_t>(i)] = tensor(key);
            const int dilation = KOKOPOP_RESBLOCK_DILATIONS[i];
            w.paddings[static_cast<size_t>(i)] = (kernel_size * dilation - dilation) / 2;
        }
        if (w.valid()) {
            generator_resblock_weights.emplace(prefix, w);
        }
    };

    auto cache_adain_resblk = [this, &cache_adain](const std::string & prefix) {
        std::string key;
        key.reserve(prefix.size() + 10);
        key = prefix; key += ".norm1";
        cache_adain(key);
        const auto n1 = adain_1d_weights.find(key);
        key = prefix; key += ".norm2";
        cache_adain(key);
        const auto n2 = adain_1d_weights.find(key);
        AdainResblk1dWeights w;
        if (n1 != adain_1d_weights.end()) {
            w.norm1 = n1->second;
        }
        if (n2 != adain_1d_weights.end()) {
            w.norm2 = n2->second;
        }
        key = prefix; key += ".conv1.weight";  w.conv1_w   = tensor(key);
        key = prefix; key += ".conv1.bias";    w.conv1_b   = tensor(key);
        key = prefix; key += ".conv2.weight";  w.conv2_w   = tensor(key);
        key = prefix; key += ".conv2.bias";    w.conv2_b   = tensor(key);
        key = prefix; key += ".conv1x1.weight"; w.conv1x1_w = tensor(key);
        if (w.valid()) {
            adain_resblk1d_weights.emplace(prefix, w);
        }
    };

    // Build full prefix strings once, reuse via string_view
    {
        std::string p;
        p.reserve(50);
        for (int stage = 0; stage < 2; ++stage) {
            p = "kokopop.decoder.generator.noise_res.";
            p += static_cast<char>('0' + stage);
            cache_generator_resblock(p, stage == 0 ? 7 : 11);
        }
        for (int block = 0; block < 6; ++block) {
            p = "kokopop.decoder.generator.resblocks.";
            p += static_cast<char>('0' + block);
            cache_generator_resblock(p, KOKOPOP_RESBLOCK_KERNELS[block % 3]);
        }
    }
    cache_adain_resblk("kokopop.decoder.encode");
    {
        std::string p;
        p.reserve(30);
        for (int i = 0; i < 4; ++i) {
            p = "kokopop.decoder.decode.";
            p += static_cast<char>('0' + i);
            cache_adain_resblk(p);
        }
        for (int i = 0; i < 3; ++i) {
            p = "kokopop.predictor.F0.";
            p += static_cast<char>('0' + i);
            cache_adain_resblk(p);
            p = "kokopop.predictor.N.";
            p += static_cast<char>('0' + i);
            cache_adain_resblk(p);
        }
    }
}

ggml_tensor * Model::cached_tensor(const std::string & logical_name) const {
    auto it = tensor_cache.find(logical_name);
    if (it != tensor_cache.end()) {
        return it->second;
    }
    return tensor(logical_name);
}

bool load_model_from_gguf(
    const std::string & path,
    const kokopop_model_options * options,
    std::unique_ptr<Model> & model,
    std::string & error) {
    ggml_context * weight_ctx = nullptr;
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = &weight_ctx;
    gguf_context * meta = gguf_init_from_file(path.c_str(), params);
    if (meta == nullptr || weight_ctx == nullptr) {
        error = "failed to load GGUF model: " + path;
        if (meta != nullptr) {
            gguf_free(meta);
        }
        if (weight_ctx != nullptr) {
            ggml_free(weight_ctx);
        }
        return false;
    }

    std::unique_ptr<Model> m(new Model());
    m->gguf_ctx = meta;
    m->weight_ctx = weight_ctx;
    {
        const int hw = static_cast<int>(std::thread::hardware_concurrency());
        const int def = std::min(4, hw > 0 ? hw : 1);
        m->n_threads = options && options->n_threads > 0 ? options->n_threads : def;
    }
    const int32_t requested_backend = options ? options->backend : KOKOPOP_BACKEND_AUTO;
    if (requested_backend != KOKOPOP_BACKEND_AUTO &&
        requested_backend != KOKOPOP_BACKEND_CPU &&
        requested_backend != KOKOPOP_BACKEND_METAL) {
        error = "invalid kokopop backend option";
        return false;
    }

    uint32_t version = 0;
    if (!get_u32(meta, "kokopop.kokoro.version", version) || version != 4) {
        error = "GGUF is not a kokopop Kokoro v4 model; reconvert it with the current converter";
        return false;
    }
    m->version = version;

    bool mock = false;
    if (get_bool(meta, "kokopop.mock", mock)) {
        m->is_mock = mock;
    }

    uint32_t sample_rate = 24000;
    if (get_u32(meta, "kokopop.sample_rate", sample_rate)) {
        m->sample_rate = static_cast<int32_t>(sample_rate);
    }

    if (!get_string_array(meta, "tokenizer.ggml.tokens", m->vocab)) {
        error = "GGUF is missing tokenizer.ggml.tokens";
        return false;
    }
    for (size_t i = 0; i < m->vocab.size(); ++i) {
        if (!m->vocab[i].empty()) {
            m->token_to_id[m->vocab[i]] = static_cast<uint32_t>(i);
        }
    }

    std::vector<std::string> voice_names;
    get_string_array(meta, "kokopop.voices", voice_names);

    std::vector<std::string> logical_names;
    std::vector<std::string> physical_names;
    std::unordered_map<std::string, std::string> physical_to_logical;
    if (get_string_array(meta, "kokopop.tensor.logical_names", logical_names) &&
        get_string_array(meta, "kokopop.tensor.physical_names", physical_names) &&
        logical_names.size() == physical_names.size()) {
        for (size_t i = 0; i < logical_names.size(); ++i) {
            physical_to_logical[physical_names[i]] = logical_names[i];
            m->tensor_physical_names[logical_names[i]] = physical_names[i];
        }
    }

    bool tensor_failed = false;
    for_each_tensor(meta, weight_ctx, [&](const char * name, ggml_tensor * tensor) {
        std::string logical = name;
        const auto mapped = physical_to_logical.find(logical);
        if (mapped != physical_to_logical.end()) {
            logical = mapped->second;
        }
        // v4: conv1d kernels are stored 2D ([OC, IC*K]) and may be quantized.
        // The runtime decomposes conv1d into im2col + mul_mat to consume them.
        const enum ggml_type t = tensor->type;
        if (t != GGML_TYPE_F32 && t != GGML_TYPE_F16 &&
            t != GGML_TYPE_Q4_K && t != GGML_TYPE_Q5_K &&
            t != GGML_TYPE_Q6_K && t != GGML_TYPE_Q8_0) {
            error = "v4 GGUF unsupported tensor type for "
                + logical + ": type "
                + std::to_string(static_cast<int>(t));
            tensor_failed = true;
            return;
        }
        m->tensors[logical] = tensor;

        constexpr const char * prefix = KOKOPOP_PREFIX_VOICE;
        if (logical.rfind(prefix, 0) == 0) {
            VoiceInfo info;
            info.name = logical.substr(std::strlen(prefix));
            info.rows = tensor->ne[1];
            info.cols = tensor->ne[0];
            m->voices[info.name] = std::move(info);
        }
    });
    if (tensor_failed) {
        if (error.empty()) {
            error = "failed to load GGUF tensors";
        }
        return false;
    }

    // Create the backend via factory
    m->backend = create_backend(requested_backend, m->n_threads, error);
    if (!m->backend) {
        if (error.empty()) {
            error = "failed to create inference backend";
        }
        return false;
    }
    // Record the actual backend type based on what was created,
    // not what was requested (AUTO can resolve to Metal or CPU).
    const char * buft_name = ggml_backend_buft_name(m->backend->weight_buffer_type());
    if (buft_name && std::strncmp(buft_name, "MTL", 3) == 0) {
        m->backend_type = KOKOPOP_BACKEND_METAL;
    } else {
        m->backend_type = KOKOPOP_BACKEND_CPU;
    }

    m->weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        m->weight_ctx,
        m->backend->weight_buffer_type());
    if (m->weight_buffer == nullptr) {
        error = "failed to allocate backend weight buffer";
        return false;
    }
    ggml_backend_buffer_set_usage(m->weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (!load_tensor_data_to_backend(path, meta, weight_ctx, *m->backend, error)) {
        if (error.empty()) {
            error = "failed to load GGUF tensor data into backend";
        }
        return false;
    }

    for (auto & item : m->voices) {
        ggml_tensor * voice_tensor = m->tensor("kokopop.voice." + item.first);
        if (!tensor_to_f32(*m->backend, voice_tensor, item.second.data)) {
            error = "failed to read voice tensor: kokopop.voice." + item.first;
            return false;
        }
    }

    // Pre-populate tensor cache for faster inference
    m->preload_tensor_cache();

    model = std::move(m);
    return true;
}

} // namespace kokopop
