#include "model/model.h"

#include "core/constants.h"
#include "core/file_mapping.h"
#include "model/gguf_util.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>
#include <unordered_map>

namespace kokopop {
namespace {

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
    const int64_t n_tensors    = gguf_get_n_tensors(meta);
    const size_t  data_origin  = gguf_get_data_offset(meta);

    // Fast path: memory-map the GGUF file and hand pointers directly to the
    // backend. Single syscall instead of 1 fopen + N fseeks + N freads;
    // unified memory targets can skip the kernel→userspace copy entirely.
    FileMapping mapping(path);
    if (mapping.ok()) {
        const uint8_t * base       = mapping.data();
        const size_t    total_size = mapping.size();
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(meta, i);
            ggml_tensor * tensor = ggml_get_tensor(weights, name);
            if (tensor == nullptr) {
                error = std::string("GGUF tensor missing from weight context: ") + (name ? name : "");
                return false;
            }
            const size_t offset = data_origin + gguf_get_tensor_offset(meta, i);
            const size_t nbytes = ggml_nbytes(tensor);
            if (offset + nbytes > total_size) {
                error = std::string("GGUF tensor extends past EOF: ") + (name ? name : "");
                return false;
            }
            backend.tensor_set(tensor, base + offset, 0, nbytes);
        }
        // Subsequent file access (if any) will be random, not sequential —
        // switch the kernel hint so it stops pre-fetching.
        mapping.advise_random();
        return true;
    }

    // Fallback: classic fopen/fseek/fread. Used when mmap fails (exotic
    // filesystems, files larger than the address space, etc.).
    std::FILE * file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        error = "failed to reopen GGUF model for tensor data: " + path
              + " (mmap also failed: " + mapping.error() + ")";
        return false;
    }

    constexpr size_t buf_size = KOKOPOP_IO_BUF_SIZE;
    std::vector<uint8_t> buffer(buf_size);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(meta, i);
        ggml_tensor * tensor = ggml_get_tensor(weights, name);
        if (tensor == nullptr) {
            std::fclose(file);
            error = std::string("GGUF tensor missing from weight context: ") + (name ? name : "");
            return false;
        }

        const size_t offset = data_origin + gguf_get_tensor_offset(meta, i);
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

int32_t Model::sample_rate() const {
    if (arch != nullptr) {
        const VoiceDesc * voice = arch->default_voice();
        if (voice != nullptr) {
            return voice->sample_rate;
        }
    }
    return default_sample_rate;
}

int32_t Model::sample_rate(std::string_view voice) const {
    if (!voice.empty() && arch != nullptr) {
        const VoiceDesc * desc = arch->find_voice(voice);
        if (desc != nullptr) {
            return desc->sample_rate;
        }
    }
    return sample_rate();
}

bool Model::tokenize_phonemes(const std::string & phonemes, std::string_view voice,
                              std::vector<uint32_t> & ids, std::string & error) const {
    if (arch == nullptr) {
        error = "model has no architecture loaded";
        return false;
    }
    const VoiceDesc * desc = voice.empty() ? arch->default_voice()
                                           : arch->find_voice(voice);
    if (desc == nullptr) {
        error = voice.empty() ? std::string("model has no voice")
                              : "voice not found in GGUF: " + std::string(voice);
        return false;
    }
    return arch->tokenize(phonemes, *desc, ids, error);
}

bool Model::tokenize_phonemes(const std::string & phonemes,
                              std::vector<uint32_t> & ids, std::string & error) const {
    return tokenize_phonemes(phonemes, std::string_view{}, ids, error);
}

std::string resolve_voice_name(const std::string & requested_voice,
                               const Model & model) {
    if (!requested_voice.empty()) {
        return requested_voice;
    }
    if (model.arch != nullptr) {
        const VoiceDesc * voice = model.arch->default_voice();
        if (voice != nullptr) {
            return voice->name;
        }
    }
    return {};
}

ggml_tensor * Model::tensor(const std::string & logical_name) const {
    const auto it = tensors.find(logical_name);
    return it == tensors.end() ? nullptr : it->second;
}

ggml_tensor * Model::cached_tensor(const std::string & logical_name) const {
    auto it = tensor_cache.find(logical_name);
    if (it != tensor_cache.end()) {
        return it->second;
    }
    return tensor(logical_name);
}

bool make_voice_frontend(Model & model, const std::string & requested_voice,
                         VoiceFrontend & out, std::string & error) {
    if (model.arch == nullptr) {
        error = "model has no architecture loaded";
        return false;
    }
    const std::string name = resolve_voice_name(requested_voice, model);
    const VoiceDesc * desc = model.arch->find_voice(name);
    if (desc == nullptr) {
        error = name.empty() ? std::string("model has no voice")
                             : "voice not found in GGUF: " + name;
        return false;
    }
    out.voice = *desc;

    ModelArch * arch = model.arch.get();
    // `out.voice` is owned by the caller's VoiceFrontend and outlives both
    // closures; capturing a pointer into it avoids re-resolving per fragment.
    const VoiceDesc * voice = &out.voice;
    out.phonemize = [arch, voice](const std::string & text,
                                  std::string & phonemes,
                                  std::string & err) {
        return arch->phonemize(text, *voice, phonemes, err);
    };
    out.tokenize = [arch, voice](const std::string & phonemes,
                                 std::vector<uint32_t> & ids,
                                 std::string & err) {
        return arch->tokenize(phonemes, *voice, ids, err);
    };
    return true;
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
    switch (requested_backend) {
        case KOKOPOP_BACKEND_AUTO:
        case KOKOPOP_BACKEND_CPU:
        case KOKOPOP_BACKEND_METAL:
        case KOKOPOP_BACKEND_CUDA:
        case KOKOPOP_BACKEND_VULKAN:
        case KOKOPOP_BACKEND_OPENCL:
            break;
        default:
            error = "invalid kokopop backend option";
            return false;
    }

    m->arch = create_arch(meta, error);
    if (!m->arch) {
        if (error.empty()) {
            error = "failed to determine the model architecture";
        }
        return false;
    }

    bool mock = false;
    if (gguf_get_bool(meta, "kokopop.mock", mock)) {
        m->is_mock = mock;
    }

    uint32_t sample_rate = 24000;
    if (gguf_get_u32(meta, "kokopop.sample_rate", sample_rate)) {
        m->default_sample_rate = static_cast<int32_t>(sample_rate);
    }

    std::vector<std::string> logical_names;
    std::vector<std::string> physical_names;
    std::unordered_map<std::string, std::string> physical_to_logical;
    if (gguf_get_str_array(meta, "kokopop.tensor.logical_names", logical_names) &&
        gguf_get_str_array(meta, "kokopop.tensor.physical_names", physical_names) &&
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
    });
    if (tensor_failed) {
        if (error.empty()) {
            error = "failed to load GGUF tensors";
        }
        return false;
    }

    // Create the backend via factory. The architecture is passed as a hint
    // because AUTO's answer depends on it and the backend has to exist
    // before any tensor is read.
    m->backend = create_backend(requested_backend, m->n_threads,
                                m->arch->arch(), error);
    if (!m->backend) {
        if (error.empty()) {
            error = "failed to create inference backend";
        }
        return false;
    }
    // Record the actual backend type based on what was created, not what
    // was requested (AUTO can resolve to any compiled backend).
    m->backend_type = m->backend->type();

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

    // Resolve weights, voices, tokenizer and scratch. Everything
    // architecture-specific — including the Kokoro v4 version gate — happens
    // here, not above.
    if (!m->arch->load(*m, error)) {
        if (error.empty()) {
            error = "failed to load model architecture: " + std::string(m->arch->name());
        }
        return false;
    }

    // Lookup index over the architecture's ordered voice table. Names and
    // aliases must be unique across the whole model, otherwise a request would
    // silently resolve to whichever entry landed last.
    for (const VoiceDesc & voice : m->arch->voices()) {
        if (!m->voices.emplace(voice.name, voice).second) {
            error = "duplicate voice name in model: " + voice.name;
            return false;
        }
        for (const std::string & alias : voice.aliases) {
            if (!m->voices.emplace(alias, voice).second) {
                error = "voice alias collides with another voice: " + alias;
                return false;
            }
        }
    }

    model = std::move(m);
    return true;
}

} // namespace kokopop
