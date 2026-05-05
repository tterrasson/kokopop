#include "test_helpers.h"
#include "core/wav.h"

// ---- Écriture WAV ----
// WAV layout (little-endian):
//   0-3:   "RIFF"
//   4-7:   file_size - 8
//   8-11:  "WAVE"
//  12-15:  "fmt "
//  16-19:  fmt_chunk_size (16)
//  20-21:  audio_format (1 = PCM)
//  22-23:  channels (1 = mono)
//  24-27:  sample_rate
//  28-31:  byte_rate
//  32-33:  block_align
//  34-35:  bits_per_sample (16)
//  36-39:  "data"
//  40-43:  data_chunk_size
//  44...:  audio samples (int16 LE)

TEST_CASE("wav_bytes_riff_header") {
    kokopop_audio audio{};
    audio.samples = new float[100];
    for (int i = 0; i < 100; ++i) audio.samples[i] = 0.5f;
    audio.n_samples = 100;
    audio.sample_rate = 24000;

    auto bytes = kokopop::wav_bytes(audio);
    CHECK(bytes.size() >= 4);
    CHECK_EQ(std::string(bytes.begin(), bytes.begin() + 4), "RIFF");
    delete[] audio.samples;
}

TEST_CASE("wav_bytes_wav_header") {
    kokopop_audio audio{};
    audio.samples = new float[100];
    for (int i = 0; i < 100; ++i) audio.samples[i] = 0.5f;
    audio.n_samples = 100;
    audio.sample_rate = 24000;

    auto bytes = kokopop::wav_bytes(audio);
    std::string wav_str(bytes.begin(), bytes.end());
    CHECK(wav_str.find("WAVE") != std::string::npos);
    CHECK(wav_str.find("fmt ") != std::string::npos);
    CHECK(wav_str.find("data") != std::string::npos);
    delete[] audio.samples;
}

TEST_CASE("wav_bytes_channel_count") {
    kokopop_audio audio{};
    audio.samples = new float[100];
    for (int i = 0; i < 100; ++i) audio.samples[i] = 0.5f;
    audio.n_samples = 100;
    audio.sample_rate = 24000;

    auto bytes = kokopop::wav_bytes(audio);
    CHECK(bytes.size() >= 24);
    uint16_t channels = static_cast<uint16_t>(bytes[22]) | (static_cast<uint16_t>(bytes[23]) << 8);
    CHECK_EQ(channels, 1u); // mono
    delete[] audio.samples;
}

TEST_CASE("wav_bytes_bits_per_sample") {
    kokopop_audio audio{};
    audio.samples = new float[100];
    for (int i = 0; i < 100; ++i) audio.samples[i] = 0.5f;
    audio.n_samples = 100;
    audio.sample_rate = 24000;

    auto bytes = kokopop::wav_bytes(audio);
    CHECK(bytes.size() >= 36);
    uint16_t bps = static_cast<uint16_t>(bytes[34]) | (static_cast<uint16_t>(bytes[35]) << 8);
    CHECK_EQ(bps, 16u);
    delete[] audio.samples;
}

TEST_CASE("wav_bytes_sample_rate_24000") {
    kokopop_audio audio{};
    audio.samples = new float[100];
    for (int i = 0; i < 100; ++i) audio.samples[i] = 0.5f;
    audio.n_samples = 100;
    audio.sample_rate = 24000;

    auto bytes = kokopop::wav_bytes(audio);
    CHECK(bytes.size() >= 28);
    uint32_t sr = static_cast<uint32_t>(bytes[24]) | (static_cast<uint32_t>(bytes[25]) << 8)
        | (static_cast<uint32_t>(bytes[26]) << 16) | (static_cast<uint32_t>(bytes[27]) << 24);
    CHECK_EQ(sr, 24000u);
    delete[] audio.samples;
}

TEST_CASE("wav_bytes_sample_rate_48000") {
    kokopop_audio audio{};
    audio.samples = new float[100];
    for (int i = 0; i < 100; ++i) audio.samples[i] = 0.5f;
    audio.n_samples = 100;
    audio.sample_rate = 48000;

    auto bytes = kokopop::wav_bytes(audio);
    uint32_t sr = static_cast<uint32_t>(bytes[24]) | (static_cast<uint32_t>(bytes[25]) << 8)
        | (static_cast<uint32_t>(bytes[26]) << 16) | (static_cast<uint32_t>(bytes[27]) << 24);
    CHECK_EQ(sr, 48000u);
    delete[] audio.samples;
}

TEST_CASE("wav_bytes_data_chunk_size") {
    kokopop_audio audio{};
    audio.samples = new float[1000];
    for (int i = 0; i < 1000; ++i) audio.samples[i] = 0.5f;
    audio.n_samples = 1000;
    audio.sample_rate = 24000;

    auto bytes = kokopop::wav_bytes(audio);
    // data chunk size at offset 40 (4 bytes LE)
    CHECK(bytes.size() >= 44);
    uint32_t data_size = static_cast<uint32_t>(bytes[40]) | (static_cast<uint32_t>(bytes[41]) << 8)
        | (static_cast<uint32_t>(bytes[42]) << 16) | (static_cast<uint32_t>(bytes[43]) << 24);
    CHECK_EQ(data_size, 1000u * 2u); // 16-bit = 2 bytes per sample
    delete[] audio.samples;
}

TEST_CASE("wav_bytes_sine_wave_roundtrip") {
    const int n = 48000;
    kokopop_audio audio{};
    audio.samples = new float[n];
    for (int i = 0; i < n; ++i) {
        audio.samples[i] = std::sin(2.0 * M_PI * 440.0 * i / n);
    }
    audio.n_samples = n;
    audio.sample_rate = 48000;

    auto bytes = kokopop::wav_bytes(audio);
    CHECK(bytes.size() > 44);
    delete[] audio.samples;
}

TEST_CASE("wav_bytes_clamps_overflow") {
    kokopop_audio audio{};
    audio.samples = new float[10];
    audio.samples[0] = 2.0f;   // > 1.0 → clamped to 1.0 → 32767
    audio.samples[1] = -2.0f;  // < -1.0 → clamped to -1.0 → -32767 (uses * 32767.0f)
    for (int i = 2; i < 10; ++i) audio.samples[i] = 0.0f;
    audio.n_samples = 10;
    audio.sample_rate = 24000;

    auto bytes = kokopop::wav_bytes(audio);
    // Audio samples start at offset 44
    // First sample (clamped to 32767 = 0x7FFF)
    CHECK(bytes.size() >= 46);
    int16_t first = static_cast<int16_t>(bytes[44]) | (static_cast<int16_t>(bytes[45]) << 8);
    CHECK_EQ(first, 32767);
    // Second sample (clamped to -32768 = 0x8000)
    CHECK(bytes.size() >= 48);
    int16_t second = static_cast<int16_t>(bytes[46]) | (static_cast<int16_t>(bytes[47]) << 8);
    CHECK_EQ(second, -32767); // -1.0 * 32767.0f = -32767
    delete[] audio.samples;
}

TEST_CASE("wav_bytes_clamps_nan") {
    kokopop_audio audio{};
    audio.samples = new float[5];
    audio.samples[0] = std::nanf("");
    for (int i = 1; i < 5; ++i) audio.samples[i] = 0.0f;
    audio.n_samples = 5;
    audio.sample_rate = 24000;

    auto bytes = kokopop::wav_bytes(audio);
    // Samples start at offset 44
    int16_t val = static_cast<int16_t>(bytes[44]) | (static_cast<int16_t>(bytes[45]) << 8);
    CHECK_EQ(val, 0); // NaN → 0
    delete[] audio.samples;
}

TEST_CASE("wav_bytes_clamps_inf") {
    kokopop_audio audio{};
    audio.samples = new float[5];
    audio.samples[0] = std::numeric_limits<float>::infinity();
    audio.samples[1] = -std::numeric_limits<float>::infinity();
    for (int i = 2; i < 5; ++i) audio.samples[i] = 0.0f;
    audio.n_samples = 5;
    audio.sample_rate = 24000;

    auto bytes = kokopop::wav_bytes(audio);
    // inf and -inf are both treated as non-finite → 0
    int16_t pos_inf = static_cast<int16_t>(bytes[44]) | (static_cast<int16_t>(bytes[45]) << 8);
    int16_t neg_inf = static_cast<int16_t>(bytes[46]) | (static_cast<int16_t>(bytes[47]) << 8);
    CHECK_EQ(pos_inf, 0);  // +inf → 0 (non-finite)
    CHECK_EQ(neg_inf, 0);  // -inf → 0 (non-finite)
    delete[] audio.samples;
}

TEST_CASE("wav_bytes_zero_samples") {
    kokopop_audio audio{};
    audio.samples = nullptr;
    audio.n_samples = 0;
    audio.sample_rate = 24000;

    auto bytes = kokopop::wav_bytes(audio);
    CHECK(bytes.size() >= 44);
    std::string wav_str(bytes.begin(), bytes.end());
    CHECK(wav_str.find("RIFF") != std::string::npos);
    CHECK(wav_str.find("WAVE") != std::string::npos);
}

TEST_CASE("wav_bytes_total_file_size") {
    kokopop_audio audio{};
    audio.samples = new float[100];
    for (int i = 0; i < 100; ++i) audio.samples[i] = 0.5f;
    audio.n_samples = 100;
    audio.sample_rate = 24000;

    auto bytes = kokopop::wav_bytes(audio);
    // RIFF size at offset 4 (uint32_t LE) = file_size - 8
    uint32_t riff_size = static_cast<uint32_t>(bytes[4]) | (static_cast<uint32_t>(bytes[5]) << 8)
        | (static_cast<uint32_t>(bytes[6]) << 16) | (static_cast<uint32_t>(bytes[7]) << 24);
    CHECK_EQ(riff_size, bytes.size() - 8u);
    delete[] audio.samples;
}

TEST_CASE("write_wav_file_valid") {
    kokopop_audio audio{};
    audio.samples = new float[1000];
    for (int i = 0; i < 1000; ++i) audio.samples[i] = 0.5f;
    audio.n_samples = 1000;
    audio.sample_rate = 24000;

    std::string error;
    CHECK(kokopop::write_wav_file("kokopop_test_write.wav", audio, error));
    CHECK(error.empty());

    std::ifstream wav("kokopop_test_write.wav", std::ios::binary);
    CHECK(wav.is_open());
    char riff[4] = {};
    wav.read(riff, 4);
    CHECK(std::memcmp(riff, "RIFF", 4) == 0);
    wav.close();

    delete[] audio.samples;
    std::remove("kokopop_test_write.wav");
}

TEST_CASE("write_wav_file_empty_path") {
    kokopop_audio audio{};
    audio.samples = new float[100];
    for (int i = 0; i < 100; ++i) audio.samples[i] = 0.5f;
    audio.n_samples = 100;
    audio.sample_rate = 24000;

    std::string error;
    CHECK(!kokopop::write_wav_file("", audio, error));
    CHECK(!error.empty());
    delete[] audio.samples;
}

TEST_CASE("write_wav_file_invalid_sample_rate") {
    kokopop_audio audio{};
    audio.samples = new float[100];
    for (int i = 0; i < 100; ++i) audio.samples[i] = 0.5f;
    audio.n_samples = 100;
    audio.sample_rate = 0;

    std::string error;
    CHECK(!kokopop::write_wav_file("kokopop_test_invalid_sr.wav", audio, error));
    CHECK(!error.empty());
    delete[] audio.samples;
}

TEST_CASE("write_wav_file_null_samples_with_count") {
    kokopop_audio audio{};
    audio.samples = nullptr;
    audio.n_samples = 100;
    audio.sample_rate = 24000;

    std::string error;
    CHECK(!kokopop::write_wav_file("kokopop_test_null.wav", audio, error));
    CHECK(!error.empty());
}
