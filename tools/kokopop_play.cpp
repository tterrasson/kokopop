#include "playback/playback.h"
#include "playback/playback_stdout.h"
#include "playback/playback_dummy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include "playback/playback_coreaudio.h"
#endif

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s [--format pcm-f32|pcm-s16|wav] [--rate 24000]\n"
        "\n"
        "Reads audio from stdin and plays it directly.\n"
        "\n"
        "On macOS, uses Core Audio for playback.\n"
        "On other platforms, writes PCM to stdout (useful for piping to ffplay).\n"
        "\n"
        "Options:\n"
        "  --format FMT    Input format: pcm-f32 (default), pcm-s16, or wav\n"
        "  --rate RATE     Sample rate (default: 24000)\n"
        "\n"
        "Examples:\n"
        "  cat audio.raw | %s\n",
        argv0, argv0);
}

const char * arg_value(int & i, int argc, char ** argv) {
    if (i + 1 >= argc) return nullptr;
    ++i;
    return argv[i];
}

} // namespace

// ---------------------------------------------------------------------------
// Simple WAV parser (reads header and extracts sample rate + data)
// ---------------------------------------------------------------------------
namespace {

struct WavHeader {
    int sample_rate = 0;
    int bits_per_sample = 16;
    int channels = 1;
    bool valid = false;
};

bool parse_wav_header(const uint8_t * data, size_t size, WavHeader & header) {
    if (size < 44) return false;

    // Check RIFF header
    if (data[0] != 'R' || data[1] != 'I' || data[2] != 'F' || data[3] != 'F') {
        return false;
    }
    if (data[8] != 'W' || data[9] != 'A' || data[10] != 'V' || data[11] != 'E') {
        return false;
    }

    // Find fmt chunk
    size_t pos = 12;
    while (pos + 8 <= size) {
        char chunk_id[4];
        std::memcpy(chunk_id, data + pos, 4);
        uint32_t chunk_size = 0;
        std::memcpy(&chunk_size, data + pos + 4, 4);
        pos += 8;

        if (chunk_id[0] == 'f' && chunk_id[1] == 'm' &&
            chunk_id[2] == 't' && chunk_id[3] == ' ') {
            if (chunk_size < 16) return false;
            header.channels = data[pos];
            header.sample_rate = *(int32_t *)(data + pos + 4);
            header.bits_per_sample = data[pos + 14];
            header.valid = true;
            return true;
        }

        pos += chunk_size;
        // Align to even
        if (chunk_size % 2 != 0) ++pos;
    }
    return false;
}

} // namespace

enum class InputFormat { PcmF32, PcmS16, Wav };

int main(int argc, char ** argv) {
    InputFormat format = InputFormat::PcmF32;
    int sample_rate = 24000;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--format") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (std::strcmp(v, "pcm-f32") == 0) format = InputFormat::PcmF32;
            else if (std::strcmp(v, "pcm-s16") == 0) format = InputFormat::PcmS16;
            else if (std::strcmp(v, "wav") == 0) format = InputFormat::Wav;
            else {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "--rate") == 0) {
            sample_rate = std::stoi(arg_value(i, argc, argv));
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    // Create playback
    kokopop::AudioPlayback * playback = nullptr;

#if defined(__APPLE__)
    playback = new kokopop::CoreAudioPlayback();
#else
    playback = new kokopop::DummyPlayback();
#endif

    if (!playback->start(sample_rate)) {
        std::fprintf(stderr, "Failed to start playback\n");
        kokopop::free_playback(playback);
        return 3;
    }

    // Read from stdin
    constexpr size_t BUF_SIZE = 4096;

    if (format == InputFormat::Wav) {
        // Read entire WAV file from stdin
        std::vector<uint8_t> wav_data;
        uint8_t buf[BUF_SIZE];
        while (!std::feof(stdin)) {
            size_t n = fread(buf, 1, BUF_SIZE, stdin);
            if (n == 0) break;
            wav_data.insert(wav_data.end(), buf, buf + n);
        }

        if (wav_data.size() < 44) {
            std::fprintf(stderr, "WAV file too small\n");
            kokopop::free_playback(playback);
            return 4;
        }

        WavHeader header;
        if (!parse_wav_header(wav_data.data(), wav_data.size(), header)) {
            std::fprintf(stderr, "Failed to parse WAV header\n");
            kokopop::free_playback(playback);
            return 4;
        }

        if (header.sample_rate > 0) {
            // Restart with correct sample rate
            playback->stop();
            playback->start(header.sample_rate);
        }

        // Find data chunk
        size_t pos = 12;
        while (pos + 8 <= wav_data.size()) {
            char chunk_id[4];
            std::memcpy(chunk_id, wav_data.data() + pos, 4);
            uint32_t chunk_size = 0;
            std::memcpy(&chunk_size, wav_data.data() + pos + 4, 4);
            pos += 8;

            if (chunk_id[0] == 'd' && chunk_id[1] == 'a' &&
                chunk_id[2] == 't' && chunk_id[3] == 'a') {
                // Convert samples to float
                std::vector<float> samples(chunk_size / (header.bits_per_sample / 8));
                if (header.bits_per_sample == 16) {
                    for (size_t i = 0; i < samples.size(); ++i) {
                        int16_t s;
                        std::memcpy(&s, wav_data.data() + pos + i * 2, 2);
                        samples[i] = s / 32767.0f;
                    }
                } else if (header.bits_per_sample == 32) {
                    std::memcpy(samples.data(), wav_data.data() + pos, samples.size() * 4);
                }
                playback->write(samples.data(), samples.size());
                break;
            }

            pos += chunk_size;
            if (chunk_size % 2 != 0) ++pos;
        }
    } else {
        // PCM reading
        std::vector<float> buffer;
        buffer.reserve(BUF_SIZE / sizeof(float));

        uint8_t raw[BUF_SIZE];
        while (!std::feof(stdin)) {
            size_t n = fread(raw, 1, BUF_SIZE, stdin);
            if (n == 0) break;

            if (format == InputFormat::PcmF32) {
                if (n % sizeof(float) != 0) continue;
                const float * samples = reinterpret_cast<const float *>(raw);
                playback->write(samples, n / sizeof(float));
            } else if (format == InputFormat::PcmS16) {
                if (n % sizeof(int16_t) != 0) continue;
                const int16_t * s16 = reinterpret_cast<const int16_t *>(raw);
                size_t ns = n / sizeof(int16_t);
                for (size_t i = 0; i < ns; ++i) {
                    buffer.push_back(s16[i] / 32767.0f);
                }
                if (buffer.size() >= 1024) {
                    playback->write(buffer.data(), buffer.size());
                    buffer.clear();
                }
            }
        }

        // Flush remaining buffer
        if (!buffer.empty()) {
            playback->write(buffer.data(), buffer.size());
        }
    }

    // Cleanup
    playback->stop();
    playback->wait();
    kokopop::free_playback(playback);

    return 0;
}
