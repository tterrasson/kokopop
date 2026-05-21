#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kokopop {

enum class EncodedAudioFormat {
    PcmF32Le,
    WavPcm16,
    OggOpus
};

struct AudioEncoderOptions {
    EncodedAudioFormat format = EncodedAudioFormat::PcmF32Le;
    int sample_rate = 24000;
    int ogg_prebuffer_chunks = 0;
};

class AudioEncoder {
public:
    static bool available(EncodedAudioFormat format);
    static const char * content_type(EncodedAudioFormat format);

    explicit AudioEncoder(AudioEncoderOptions options);
    ~AudioEncoder();

    AudioEncoder(const AudioEncoder &) = delete;
    AudioEncoder & operator=(const AudioEncoder &) = delete;

    bool start(std::vector<uint8_t> & out, std::string & error);
    bool push(const float * samples, size_t n_samples, bool is_final,
              std::vector<uint8_t> & out, std::string & error);
    bool finish(bool success, std::vector<uint8_t> & out, std::string & error);

private:
    void take_ogg_pending(std::vector<uint8_t> & out);
    bool write_ogg_samples(const float * samples, size_t n_samples, std::string & error);

    AudioEncoderOptions _options;
    bool _started = false;
    bool _finished = false;
    std::vector<float> _wav_accumulator;

#ifdef KOKOPOP_HAS_OPUS
    class OggState;
    OggState * _ogg = nullptr;
    int _prebuffer_target = 0;
    bool _prebuffer_released = false;
    std::vector<std::vector<float>> _prebuffered;
#endif
};

} // namespace kokopop
