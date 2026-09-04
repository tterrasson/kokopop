#include "arch/kokoro/kokoro_arch.h"

#include "arch/kokoro/kokoro.h"
#include "core/constants.h"
#include "core/utf8.h"
#include "model/gguf_util.h"
#include "synthesis/phonemizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

#include <ggml.h>
#include <gguf.h>

namespace kokopop {

// ---------------------------------------------------------------------------
// Voice routing
// ---------------------------------------------------------------------------

std::string espeak_voice_for_kokoro_voice(const std::string & voice) {
    // Kokoro voice names encode their language in the first letter
    // ("af_heart" -> American English). This convention is Kokoro's alone:
    // other architectures carry the espeak voice in their metadata instead.
    const char lang = voice.empty() ? 'a' : voice[0];
    switch (lang) {
    case 'a': return "gmw/en-US";
    case 'b': return "gmw/en";
    case 'e': return "roa/es";
    case 'f': return "roa/fr";
    case 'h': return "inc/hi";
    case 'i': return "roa/it";
    case 'j': return "jpx/ja";
    case 'p': return "roa/pt-BR";
    case 'z': return "sit/cmn";
    default: return "gmw/en-US";
    }
}

bool phonemize_text(const std::string & text, const std::string & kokoro_voice,
                    std::string & phonemes, std::string & error) {
    return phonemize_text(text,
                          espeak_voice_for_kokoro_voice(kokoro_voice),
                          kokoro_voice.empty() ? 'a' : kokoro_voice[0],
                          phonemes, error);
}

// ---------------------------------------------------------------------------
// Graph context sizing
//
// These formulas are Kokoro's: they estimate the tensor data the generation
// and generator graphs stage through their context. The backend contributes
// only its own metadata/margin term.
// ---------------------------------------------------------------------------

namespace {

// Upper bounds on the ggml objects the frontend graph creates. The frontend
// stages no bulk data through its context, so its reservation is metadata
// only and comes entirely from the backend (16 MiB, 64 MiB on Metal — both
// well above the ~1.8 MiB these bounds actually need).
constexpr size_t KOKORO_FRONTEND_MAX_TENSORS = 4096;
constexpr size_t KOKORO_FRONTEND_MAX_NODES   = 8192;

} // namespace

size_t kokoro_frontend_context_bytes(const Backend & backend) {
    return backend.graph_context_bytes(KOKORO_FRONTEND_MAX_TENSORS,
                                       KOKORO_FRONTEND_MAX_NODES);
}

size_t kokoro_generation_context_bytes(const Backend & backend,
                                       int64_t total_frames, int64_t n_tokens) {
    const size_t frames = static_cast<size_t>(std::max<int64_t>(1, total_frames));
    const size_t tokens = static_cast<size_t>(std::max<int64_t>(1, n_tokens));

    const size_t mask_bytes = frames * tokens * sizeof(float);
    const size_t pred_bytes = 512ULL * tokens * sizeof(float);
    const size_t lstm_work  = frames * 4096ULL * sizeof(float);

    // 64 MiB covers the graph metadata (~2 MiB at this graph's size) plus the
    // slack the pipeline has always reserved here. Backend-independent on
    // purpose: the arena is dominated by the data terms above, and making the
    // constant vary per backend would change Kokoro's memory profile for no
    // measured reason.
    (void)backend;
    return mask_bytes + pred_bytes + lstm_work + backend_mib(64);
}

size_t kokoro_generator_context_bytes(const Backend & backend,
                                      int64_t decoder_len) {
    const size_t dec = static_cast<size_t>(std::max<int64_t>(1, decoder_len));
    const size_t out_frames = dec * 60;

    const size_t decoder_tensor  = 64ULL * dec * sizeof(float);
    const size_t harmonic_tensor = 22ULL * (out_frames + 1) * sizeof(float);
    const size_t post_tensor     = out_frames * sizeof(float);
    const size_t conv_work       = 256ULL * out_frames * sizeof(float) * 3;

    // As above: 16 MiB against ~1 MiB of graph metadata, data-dominated arena.
    (void)backend;
    return decoder_tensor + harmonic_tensor + post_tensor + conv_work
         + backend_mib(16);
}

// ---------------------------------------------------------------------------
// KokoroArch
// ---------------------------------------------------------------------------

KokoroArch * kokoro_arch(Model & model) {
    if (model.arch == nullptr || model.arch->arch() != Arch::Kokoro) {
        return nullptr;
    }
    return static_cast<KokoroArch *>(model.arch.get());
}

ggml_tensor * KokoroArch::tensor(const std::string & logical_name) const {
    return base != nullptr ? base->tensor(logical_name) : nullptr;
}

ggml_tensor * KokoroArch::cached_tensor(const std::string & logical_name) const {
    return base != nullptr ? base->cached_tensor(logical_name) : nullptr;
}

void KokoroArch::clear_tensor_cache() {
    if (base != nullptr) {
        base->tensor_cache.clear();
    }
    adain_1d_weights.clear();
    generator_resblock_weights.clear();
    adain_resblk1d_weights.clear();
}

const VoiceDesc * KokoroArch::find_voice(std::string_view name) const {
    for (const auto & voice : voice_descs) {
        if (voice.name == name) {
            return &voice;
        }
    }
    for (const auto & voice : voice_descs) {
        for (const auto & alias : voice.aliases) {
            if (alias == name) {
                return &voice;
            }
        }
    }
    return nullptr;
}

const VoiceDesc * KokoroArch::default_voice() const {
    if (voice_descs.empty()) {
        return nullptr;
    }
    const size_t index = default_voice_index < voice_descs.size() ? default_voice_index : 0;
    return &voice_descs[index];
}

std::string KokoroArch::resolve_voice(const std::string & requested) const {
    if (!requested.empty()) {
        return requested;
    }
    const VoiceDesc * voice = default_voice();
    return voice != nullptr ? voice->name : std::string();
}

bool KokoroArch::phonemize(const std::string & text, const VoiceDesc & voice,
                           std::string & phonemes, std::string & error) const {
    return phonemize_text(text, voice.espeak_voice, voice.normalization_lang,
                          phonemes, error);
}

ChunkConfig KokoroArch::adjust_chunk_config(ChunkConfig cfg,
                                            const VoiceDesc & voice) const {
    // Kokoro's limit is the ALBERT context, identical for every voice, and it
    // is already what the presets are written against.
    (void)voice;
    return cfg;
}

bool KokoroArch::synthesize(Model & base_model,
                            const std::vector<uint32_t> & ids,
                            const VoiceDesc & voice,
                            float speed,
                            const SynthesisExtras & extras,
                            std::vector<float> & audio,
                            std::string & error) {
    // Weight resolution happened in load(); `base_model` is the same object
    // this arch already points at.
    (void)base_model;

    KokoroFrontendProbe probe;
    if (!run_kokoro_frontend_probe(*this, ids, voice.name, probe, error,
                                   extras.kokoro_style_len, &extras.diffusion)) {
        return false;
    }

    KokoroGenerationProbe gen;
    if (!run_kokoro_generation_probe(*this, ids, voice.name, speed, probe, gen,
                                     error, extras.kokoro_style_len)) {
        return false;
    }
    if (gen.audio.empty()) {
        error = "Kokoro generator produced no audio";
        return false;
    }

    audio = std::move(gen.audio);

    // Free the large intermediate probe buffers before returning: gen.asr
    // (~6 MB) and gen.decoder (~3 MB) are dead once the audio is extracted,
    // which matters for peak RAM while streaming.
    gen.asr.clear();
    gen.asr.shrink_to_fit();
    gen.decoder.clear();
    gen.decoder.shrink_to_fit();

    return true;
}

bool KokoroArch::load(Model & base_model, std::string & error) {
    base         = &base_model;
    backend      = base_model.backend.get();
    backend_type = base_model.backend_type;

    gguf_context * meta = base_model.gguf_ctx;

    uint32_t version = 0;
    if (!gguf_get_u32(meta, "kokopop.kokoro.version", version) || version != 4) {
        error = "GGUF is not a kokopop Kokoro v4 model; reconvert it with the current converter";
        return false;
    }
    base_model.version = version;

    float sigma_data = 0.2f;
    if (gguf_get_f32(meta, "kokopop.diffusion.sigma_data", sigma_data) &&
        std::isfinite(sigma_data) && sigma_data > 0.0f) {
        diffusion_sigma_data = sigma_data;
    }

    if (!gguf_get_str_array(meta, "tokenizer.ggml.tokens", vocab)) {
        error = "GGUF is missing tokenizer.ggml.tokens";
        return false;
    }
    for (size_t i = 0; i < vocab.size(); ++i) {
        if (!vocab[i].empty()) {
            token_to_id[vocab[i]] = static_cast<uint32_t>(i);
        }
    }

    // Style matrices: one tensor per voice, named "kokopop.voice.<name>".
    constexpr const char * prefix = KOKOPOP_PREFIX_VOICE;
    const size_t prefix_len = std::strlen(prefix);
    std::vector<std::string> tensor_voices;
    for (const auto & item : base_model.tensors) {
        if (item.first.rfind(prefix, 0) != 0) {
            continue;
        }
        tensor_voices.push_back(item.first.substr(prefix_len));
    }
    // The tensor map is unordered; sort so that a file without a
    // `kokopop.voices` list still yields a stable voice order.
    std::sort(tensor_voices.begin(), tensor_voices.end());

    // `kokopop.voices` carries the converter's order, which is what the
    // default-voice rule below relies on. Anything present as a tensor but
    // missing from the list is appended rather than dropped.
    std::vector<std::string> ordered;
    gguf_get_str_array(meta, "kokopop.voices", ordered);
    std::vector<std::string> names;
    names.reserve(tensor_voices.size());
    for (const auto & name : ordered) {
        if (std::find(tensor_voices.begin(), tensor_voices.end(), name) != tensor_voices.end() &&
            std::find(names.begin(), names.end(), name) == names.end()) {
            names.push_back(name);
        }
    }
    for (const auto & name : tensor_voices) {
        if (std::find(names.begin(), names.end(), name) == names.end()) {
            names.push_back(name);
        }
    }

    // An explicit default voice wins; otherwise the first listed voice does.
    // Both rules are deterministic, which the previous "first entry of an
    // unordered_map" fallback was not.
    //
    // Only the *index* moves: `names` — and therefore `voice_descs`, i.e. what
    // `voices()` returns — keeps the converter's order, which the interface
    // promises and multi-voice enumeration depends on.
    default_voice_index = 0;
    std::string default_name;
    if (gguf_get_str(meta, "kokopop.default_voice", default_name) &&
        !default_name.empty()) {
        const auto it = std::find(names.begin(), names.end(), default_name);
        if (it == names.end()) {
            error = "kokopop.default_voice names a voice absent from the model: "
                  + default_name;
            return false;
        }
        default_voice_index = static_cast<size_t>(it - names.begin());
    }

    const int32_t model_sample_rate = base_model.default_sample_rate;
    voice_descs.reserve(names.size());
    for (const auto & name : names) {
        ggml_tensor * style = base_model.tensor(prefix + name);
        if (style == nullptr) {
            error = "failed to resolve voice tensor: " + std::string(prefix) + name;
            return false;
        }

        VoiceInfo info;
        info.name = name;
        info.rows = style->ne[1];
        info.cols = style->ne[0];
        if (!tensor_to_f32(*backend, style, info.data)) {
            error = "failed to read voice tensor: " + std::string(prefix) + name;
            return false;
        }
        voice_styles[name] = std::move(info);

        VoiceDesc desc;
        desc.name               = name;
        desc.espeak_voice       = espeak_voice_for_kokoro_voice(name);
        desc.normalization_lang = name.empty() ? 'a' : name[0];
        desc.sample_rate        = model_sample_rate;
        desc.length_scale       = 1.0f;
        desc.max_tokens         = 510;  // ALBERT context
        desc.frontend           = FrontendKind::Misaki;
        desc.decoder            = DecoderKind::Kokoro;
        voice_descs.push_back(std::move(desc));
    }

    preload_tensor_cache();
    prereserve_scratch_buffers();
    return true;
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

bool KokoroArch::tokenize(const std::string & phonemes, const VoiceDesc & voice,
                          std::vector<uint32_t> & ids, std::string & error) const {
    // Kokoro's tokenizer is voice-independent: one id per code point of the
    // shared vocabulary, framed by the 0 sentinel.
    (void)voice;
    ids.clear();
    // Two sentinels + one slot per UTF-8 code point (upper bound).
    // Divide by 2 to avoid massive over-allocation for multi-byte scripts.
    ids.reserve(phonemes.size() / 2 + 2);
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

void KokoroArch::preload_tensor_cache() {
    clear_tensor_cache();
    if (base == nullptr) {
        return;
    }
    base->tensor_cache = base->tensors;

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

    // Dequantize all LSTM w_hh tensors (Q5_K → F32) for the fused LSTM kernel
    // and produce a rowwise transposition for SIMD-friendly dot-product access.
    //
    // Phase 1 (parallel): each worker thread reads its assigned tensor from the
    //   immutable weight buffer (tensor_get is a memcpy of read-only data),
    //   dequantizes to F32, and computes the rowwise transpose. Threads write
    //   to disjoint slots in the `results` array — no synchronization needed.
    //
    // Phase 2 (sequential): backend->preload_lstm_whh uploads to GPU on the
    //   Metal backend, which is not necessarily safe to call concurrently; map
    //   inserts (unordered_map) are also not concurrency-safe. Both run serially.
    lstm_w_hh_f32.clear();
    lstm_w_hh_rowwise.clear();

    struct LstmWhhTask {
        std::string         name;
        ggml_tensor       * tensor = nullptr;
        std::vector<float>  f32;
        std::vector<float>  rowwise;
        int                 H      = 0;
        int                 four_H = 0;
        bool                ok     = false;
    };

    std::vector<LstmWhhTask> tasks;
    tasks.reserve(16);
    for (const auto & kv : base->tensors) {
        if (kv.first.find("weight_hh_l0") == std::string::npos) continue;
        if (kv.second == nullptr) continue;
        tasks.push_back({kv.first, kv.second, {}, {}, 0, 0, false});
    }

    auto run_one = [&](LstmWhhTask & task) {
        if (!tensor_to_f32(*backend, task.tensor, task.f32)) return;
        task.H      = static_cast<int>(task.tensor->ne[0]);
        task.four_H = static_cast<int>(task.tensor->ne[1]);
        // Original layout (col-major): f32[k + H*j], k∈[0,H), j∈[0,4*H)
        // Transposed   (row-major):    rowwise[j*H + k], contiguous over k
        task.rowwise.resize(static_cast<size_t>(task.H) * static_cast<size_t>(task.four_H));
        for (int j = 0; j < task.four_H; ++j) {
            const float * src = task.f32.data() + static_cast<size_t>(task.H) * j;
            float       * dst = task.rowwise.data() + static_cast<size_t>(task.H) * j;
            std::memcpy(dst, src, static_cast<size_t>(task.H) * sizeof(float));
        }
        task.ok = true;
    };

    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned n_threads = std::min<unsigned>(
        std::max(1u, hw), std::min<unsigned>(4u, static_cast<unsigned>(tasks.size())));

    if (tasks.size() <= 1 || n_threads <= 1) {
        for (auto & t : tasks) run_one(t);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(n_threads);
        for (unsigned tid = 0; tid < n_threads; ++tid) {
            threads.emplace_back([&, tid] {
                for (size_t i = tid; i < tasks.size(); i += n_threads) {
                    run_one(tasks[i]);
                }
            });
        }
        for (auto & th : threads) th.join();
    }

    for (auto & t : tasks) {
        if (!t.ok) continue;
        backend->preload_lstm_whh(t.name, t.f32.data(), t.H, t.four_H);
        lstm_w_hh_f32.emplace(t.name, std::move(t.f32));
        lstm_w_hh_rowwise.emplace(t.name, std::move(t.rowwise));
    }

    // CPU-side cache of LSTM b_hh tensors. The fused LSTM callback captures
    // a raw host pointer to b_hh data; on backends that place weights in
    // device memory (CUDA VRAM) this cache provides a host-addressable copy.
    lstm_b_hh_f32.clear();
    for (const auto & kv : base->tensors) {
        if (kv.first.find("bias_hh_l0") == std::string::npos) continue;
        if (kv.second == nullptr) continue;
        std::vector<float> data;
        if (tensor_to_f32(*backend, kv.second, data)) {
            lstm_b_hh_f32.emplace(kv.first, std::move(data));
        }
    }

    // LSTM input weights (weight_ih_l0) are consumed via backend mul_mat on
    // every backend; no host-side F32 cache is needed.
}

void KokoroArch::prereserve_scratch_buffers() {
    // Decoder style tensor has a fixed size of 128 floats per inference — set
    // it directly so the resize(128) call inside generation is a no-op.
    tmp_decoder_style_f32.assign(128, 0.0f);

    // Sizing rationale: typical chunks are bounded by target_max_tokens (~180)
    // and a moderate frame-per-token ratio. Reserving (not resizing) here only
    // pre-allocates capacity — vectors stay logically empty so semantics are
    // unchanged. Larger chunks still trigger growth via resize() as before.
    constexpr size_t typ_tokens     = 256;        // covers target_max_tokens with margin
    constexpr size_t typ_frames     = 12000;      // ~60 frames/token × 200 tokens
    constexpr size_t typ_mask       = 1u << 20;   // 1 M floats ≈ 4 MB cap for the mask
    constexpr size_t typ_audio_f32  = typ_frames * 32;  // post/decoder/audio: small multi-MB
    constexpr size_t typ_stft       = 22 * typ_frames;

    tmp_ids_i32.reserve(typ_tokens);
    tmp_pos_i32.reserve(typ_tokens);
    tmp_mask_f32.reserve(typ_mask);
    tmp_post_f32.reserve(typ_audio_f32);
    tmp_decoder_cpu_f32.reserve(typ_audio_f32);
    tmp_audio_f32.reserve(typ_audio_f32);
    tmp_stft_source_f32.reserve(typ_frames * 16);
    tmp_stft_har_f32.reserve(typ_stft);
    // The iSTFT's own overlap-add scratch is O(n_fft) and allocated by the
    // component; what is reserved here is only the complex half-spectrum it
    // reads, 11 bins x frames for Kokoro's n_fft=20.
    tmp_istft_real_f32.reserve(11 * typ_frames);
    tmp_istft_imag_f32.reserve(11 * typ_frames);
}

} // namespace kokopop
