#include "playback/playback.h"
#include "playback/playback_stdout.h"
#include "playback/playback_dummy.h"
#include "core/cli_args.h"
#include "core/wav.h"

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

} // namespace

enum class InputFormat { PcmF32, PcmS16, Wav };

int main(int argc, char ** argv) {
    InputFormat format = InputFormat::PcmF32;
    int sample_rate = 24000;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--format") == 0) {
            const char * v = kokopop::cli::arg_required(i, argc, argv, "--format", argv[0]);
            if (std::strcmp(v, "pcm-f32") == 0) format = InputFormat::PcmF32;
            else if (std::strcmp(v, "pcm-s16") == 0) format = InputFormat::PcmS16;
            else if (std::strcmp(v, "wav") == 0) format = InputFormat::Wav;
            else {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "--rate") == 0) {
            sample_rate = static_cast<int>(
                kokopop::cli::arg_int(i, argc, argv, "--rate", argv[0], 1000, 384000));
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

        kokopop::WavFileInfo info;
        if (!kokopop::wav_parse(wav_data.data(), wav_data.size(), info)) {
            std::fprintf(stderr, "Failed to parse WAV header (not a supported PCM/float WAV)\n");
            kokopop::free_playback(playback);
            return 4;
        }
        if (info.channels != 1) {
            std::fprintf(stderr, "Only mono WAV input is supported (got %d channels)\n",
                         info.channels);
            kokopop::free_playback(playback);
            return 4;
        }

        if (info.sample_rate != sample_rate) {
            // Restart with the file's sample rate
            playback->stop();
            if (!playback->start(info.sample_rate)) {
                std::fprintf(stderr, "Failed to restart playback at %d Hz\n", info.sample_rate);
                kokopop::free_playback(playback);
                return 3;
            }
            sample_rate = info.sample_rate;
        }

        // Convert samples to float. info.data_size is already clamped to the
        // bytes actually present in the file.
        const int bytes_per_sample = info.bits_per_sample / 8;
        const size_t n = info.data_size / static_cast<size_t>(bytes_per_sample);
        std::vector<float> samples(n);
        const uint8_t * d = wav_data.data() + info.data_offset;
        if (info.bits_per_sample == 16) {
            for (size_t i = 0; i < n; ++i) {
                int16_t s;
                std::memcpy(&s, d + i * 2, 2);
                samples[i] = s / 32768.0f;
            }
        } else if (info.is_float) {
            std::memcpy(samples.data(), d, n * sizeof(float));
        } else {  // 32-bit integer PCM
            for (size_t i = 0; i < n; ++i) {
                int32_t s;
                std::memcpy(&s, d + i * 4, 4);
                samples[i] = s / 2147483648.0f;
            }
        }
        if (!samples.empty()) {
            playback->write(samples.data(), samples.size());
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
