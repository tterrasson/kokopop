#include "audio/audio_encoder.h"

#include "core/wav.h"

#include <algorithm>
#include <limits>

#ifdef KOKOPOP_HAS_OPUS
#include "core/ogg_opus_encoder.h"
#endif

namespace kokopop {

#ifdef KOKOPOP_HAS_OPUS
class AudioEncoder::OggState {
public:
    explicit OggState(int sample_rate) : encoder(sample_rate) {}
    OggOpusEncoder encoder;
};
#endif

bool AudioEncoder::available(EncodedAudioFormat format) {
    if (format == EncodedAudioFormat::OggOpus) {
#ifdef KOKOPOP_HAS_OPUS
        return true;
#else
        return false;
#endif
    }
    return true;
}

const char * AudioEncoder::content_type(EncodedAudioFormat format) {
    switch (format) {
        case EncodedAudioFormat::WavPcm16: return "audio/wav";
        case EncodedAudioFormat::OggOpus: return "audio/ogg";
        case EncodedAudioFormat::PcmF32Le:
        default: return "application/octet-stream";
    }
}

AudioEncoder::AudioEncoder(AudioEncoderOptions options)
    : _options(options) {
#ifdef KOKOPOP_HAS_OPUS
    _prebuffer_target = std::max(0, options.ogg_prebuffer_chunks);
#endif
}

AudioEncoder::~AudioEncoder() {
#ifdef KOKOPOP_HAS_OPUS
    delete _ogg;
#endif
}

bool AudioEncoder::start(std::vector<uint8_t> & out, std::string & error) {
    out.clear();
    if (_started) return true;
    if (_options.sample_rate <= 0) {
        error = "invalid sample rate";
        return false;
    }
    if (!available(_options.format)) {
        error = "Ogg/Opus output is not available in this build";
        return false;
    }
    _started = true;

    if (_options.format == EncodedAudioFormat::OggOpus) {
#ifdef KOKOPOP_HAS_OPUS
        _ogg = new OggState(_options.sample_rate);
        _ogg->encoder.flush_header();
        take_ogg_pending(out);
#endif
    }
    return true;
}

bool AudioEncoder::write_ogg_samples(const float * samples, size_t n_samples, std::string & error) {
    if (n_samples > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error = "too many samples for Ogg/Opus encoder";
        return false;
    }
#ifdef KOKOPOP_HAS_OPUS
    _ogg->encoder.write(samples, static_cast<int>(n_samples));
#else
    (void)samples;
    (void)n_samples;
    error = "Ogg/Opus output is not available in this build";
    return false;
#endif
    return true;
}

bool AudioEncoder::push(const float * samples, size_t n_samples, bool is_final,
                        std::vector<uint8_t> & out, std::string & error) {
    out.clear();
    if (!_started && !start(out, error)) return false;
    if (_finished) {
        error = "encoder is already finished";
        return false;
    }
    if (n_samples > 0 && samples == nullptr) {
        error = "samples is null";
        return false;
    }
    if (n_samples == 0) {
        return true;
    }

    if (_options.format == EncodedAudioFormat::PcmF32Le) {
        const auto * p = reinterpret_cast<const uint8_t *>(samples);
        out.insert(out.end(), p, p + n_samples * sizeof(float));
        return true;
    }

    if (_options.format == EncodedAudioFormat::WavPcm16) {
        _wav_accumulator.insert(_wav_accumulator.end(), samples, samples + n_samples);
        return true;
    }

    if (_options.format == EncodedAudioFormat::OggOpus) {
#ifdef KOKOPOP_HAS_OPUS
        if (_prebuffer_target > 0 && !_prebuffer_released) {
            _prebuffered.emplace_back(samples, samples + n_samples);
            const bool ready =
                static_cast<int>(_prebuffered.size()) >= _prebuffer_target ||
                is_final;
            if (!ready) return true;
            for (const auto & buffered : _prebuffered) {
                if (!write_ogg_samples(buffered.data(), buffered.size(), error)) return false;
            }
            _prebuffered.clear();
            _prebuffer_released = true;
        } else {
            if (!write_ogg_samples(samples, n_samples, error)) return false;
        }
        _ogg->encoder.pull_pages(0);
        take_ogg_pending(out);
        return true;
#else
        error = "Ogg/Opus output is not available in this build";
        return false;
#endif
    }

    error = "unknown audio format";
    return false;
}

bool AudioEncoder::finish(bool success, std::vector<uint8_t> & out, std::string & error) {
    out.clear();
    if (!_started && !start(out, error)) return false;
    if (_finished) return true;
    _finished = true;

    if (!success) return true;

    if (_options.format == EncodedAudioFormat::WavPcm16) {
        kokopop_audio audio{};
        audio.samples = _wav_accumulator.data();
        audio.n_samples = _wav_accumulator.size();
        audio.sample_rate = _options.sample_rate;
        out = wav_bytes(audio);
        return true;
    }

    if (_options.format == EncodedAudioFormat::OggOpus) {
#ifdef KOKOPOP_HAS_OPUS
        if (!_prebuffered.empty()) {
            for (const auto & buffered : _prebuffered) {
                if (!write_ogg_samples(buffered.data(), buffered.size(), error)) return false;
            }
            _prebuffered.clear();
        }
        _ogg->encoder.drain();
        take_ogg_pending(out);
        return true;
#else
        error = "Ogg/Opus output is not available in this build";
        return false;
#endif
    }

    return true;
}

void AudioEncoder::take_ogg_pending(std::vector<uint8_t> & out) {
#ifdef KOKOPOP_HAS_OPUS
    if (!_ogg || !_ogg->encoder.has_pending()) return;
    auto pages = _ogg->encoder.take_pending();
    out.insert(out.end(), pages.begin(), pages.end());
#else
    (void)out;
#endif
}

} // namespace kokopop
