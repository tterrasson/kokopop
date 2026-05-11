#include "synthesis/synth.h"

#include "core/constants.h"
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

struct AudioHealth {
    double rms = 0.0;
    double clipped_ratio = 0.0;
    float peak = 0.0f;
    bool finite = true;
};

AudioHealth audio_health(const std::vector<float> & audio) {
    AudioHealth health;
    if (audio.empty()) {
        return health;
    }

    double sum_sq = 0.0;
    size_t clipped = 0;
    for (float sample : audio) {
        if (!std::isfinite(sample)) {
            health.finite = false;
            continue;
        }
        const float abs_sample = std::fabs(sample);
        health.peak = std::max(health.peak, abs_sample);
        sum_sq += static_cast<double>(sample) * static_cast<double>(sample);
        if (abs_sample > 0.98f) {
            ++clipped;
        }
    }
    health.rms = std::sqrt(sum_sq / static_cast<double>(audio.size()));
    health.clipped_ratio = static_cast<double>(clipped) / static_cast<double>(audio.size());
    return health;
}

bool is_pathological_audio(const std::vector<float> & audio) {
    if (audio.empty()) {
        return true;
    }
    const AudioHealth health = audio_health(audio);
    if (!health.finite) {
        return true;
    }
    return health.rms > 0.95 && health.peak >= 0.99f && health.clipped_ratio > 0.10;
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
    if (!run_kokoro_frontend_probe(model, ids, voice, probe, error)) {
        return false;
    }
    KokoroGenerationProbe gen;
    if (!run_kokoro_generation_probe(model, ids, voice, speed, probe, gen, error)) {
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
        static const char * kProsodyRetries[] = {
            "",
            " ,",
            " , ,",
            " , , , ,",
            " , , , , , ,",
        };

        std::vector<float> audio;
        std::vector<float> last_audio;
        std::string last_error;
        bool synthesized = false;
        for (const char * suffix : kProsodyRetries) {
            std::string attempt = phonemes;
            attempt += suffix;

            std::string attempt_error;
            audio.clear();
            if (!run_real_synthesis(model, attempt, voice, speed, audio, attempt_error)) {
                last_error = attempt_error;
                continue;
            }

            synthesized = true;
            if (!is_pathological_audio(audio)) {
                break;
            }
            // Audio is pathological — keep it as a fallback before retrying.
            // Moving out of `audio` is safe: the next iteration calls
            // audio.clear() then run_real_synthesis overwrites it.
            last_audio = std::move(audio);
        }

        if (!synthesized) {
            error = last_error.empty() ? "Kokoro generator produced no audio" : last_error;
            return false;
        }

        if ((audio.empty() || is_pathological_audio(audio)) && !last_audio.empty()) {
            audio = std::move(last_audio);
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
