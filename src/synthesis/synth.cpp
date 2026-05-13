#include "synthesis/synth.h"

#include "core/constants.h"
#include "core/utf8.h"
#include "inference/kokoro.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace kokopop {
namespace {

float hash_freq(const std::vector<uint32_t> & ids, const std::string & voice) {
    uint32_t h = 2166136261u;
    for (uint32_t id : ids) {
        h ^= id;
        h *= 16777619u;
    }
    for (unsigned char c : voice) {
        h ^= c;
        h *= 16777619u;
    }
    return 135.0f + static_cast<float>(h % 180u);
}

bool allocate_audio(size_t n, int32_t sr, kokopop_audio & out) {
    out.samples = static_cast<float *>(std::calloc(n, sizeof(float)));
    if (out.samples == nullptr) {
        return false;
    }
    out.n_samples = n;
    out.sample_rate = sr;
    return true;
}

int64_t utf8_codepoint_count(const std::string & text) {
    int64_t count = 0;
    size_t off = 0;
    std::string_view ch;
    while (utf8_next(text, off, ch)) {
        ++count;
    }
    return off == text.size() ? count : static_cast<int64_t>(text.size());
}

bool run_real_synthesis(
    Model & model,
    const std::string & phonemes,
    const std::string & voice,
    float speed,
    std::vector<float> & audio,
    std::string & error) {
    std::vector<uint32_t> ids;
    if (!model.tokenize_phonemes(phonemes, ids, error)) {
        return false;
    }

    KokoroFrontendProbe probe;
    const int64_t style_len = utf8_codepoint_count(phonemes);
    if (!run_kokoro_frontend_probe(model, ids, voice, probe, error, style_len)) {
        return false;
    }
    KokoroGenerationProbe gen;
    if (!run_kokoro_generation_probe(model, ids, voice, speed, probe, gen, error, style_len)) {
        return false;
    }
    if (gen.audio.empty()) {
        error = "Kokoro generator produced no audio";
        return false;
    }

    audio = std::move(gen.audio);

    // Free large intermediate probe buffers before returning.
    // gen.asr (~6 MB) and gen.decoder (~3 MB) are no longer needed
    // after audio extraction, reducing peak RAM during streaming.
    gen.asr.clear();
    gen.asr.shrink_to_fit();
    gen.decoder.clear();
    gen.decoder.shrink_to_fit();

    return true;
}

} // namespace

bool synthesize_phonemes(
    Model & model, const std::string & phonemes,
    const std::string & voice, float speed,
    kokopop_audio & out, std::string & error) {
    if (speed <= 0.0f || speed > 4.0f || !std::isfinite(speed)) {
        error = "speed must be greater than 0 and at most 4";
        return false;
    }
    if (phonemes.empty()) {
        error = "phoneme input is empty";
        return false;
    }
    if (!voice.empty() && !model.voices.empty() && model.voices.find(voice) == model.voices.end()) {
        error = "voice not found in GGUF: " + voice;
        return false;
    }

    if (!model.is_mock) {
        std::vector<float> audio;
        if (!run_real_synthesis(model, phonemes, voice, speed, audio, error)) {
            return false;
        }

        if (!allocate_audio(audio.size(), model.sample_rate, out)) {
            error = "failed to allocate output audio";
            return false;
        }
        std::memcpy(out.samples, audio.data(), audio.size() * sizeof(float));

        return true;
    }

    std::vector<uint32_t> ids;
    if (!model.tokenize_phonemes(phonemes, ids, error)) {
        return false;
    }

    const float duration_s = std::max(0.10f, static_cast<float>(ids.size()) * 0.035f / speed);
    const size_t n = static_cast<size_t>(duration_s * static_cast<float>(model.sample_rate));
    if (!allocate_audio(n, model.sample_rate, out)) {
        error = "failed to allocate output audio";
        return false;
    }

    const float freq = hash_freq(ids, voice);
    const float sr = static_cast<float>(model.sample_rate);
    for (size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / sr;
        const float attack = static_cast<float>(i) / (0.02f * sr);
        const float release = static_cast<float>(n - i) / (0.04f * sr);
        const float env = std::min(1.0f, std::min(attack, release));
        out.samples[i] = 0.08f * env * std::sin(2.0f * M_PI * freq * t);
    }
    return true;
}

} // namespace kokopop
