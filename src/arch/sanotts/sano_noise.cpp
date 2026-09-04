#include "arch/sanotts/sano_noise.h"

#include <cmath>
#include <cstring>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace kokopop::sano {
namespace {

// ---------------------------------------------------------------------------
// SHA-256
// ---------------------------------------------------------------------------

constexpr uint32_t SHA_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

uint32_t rotr32(uint32_t v, int n) {
    return (v >> n) | (v << (32 - n));
}

void sha256_block(uint32_t h[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(block[4 * i]) << 24) |
               (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
               (static_cast<uint32_t>(block[4 * i + 2]) << 8) |
               static_cast<uint32_t>(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t1 = hh + S1 + ch + SHA_K[i] + w[i];
        const uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

// ---------------------------------------------------------------------------
// Box-Muller
// ---------------------------------------------------------------------------

/// ATen's `normal_fill_16` with mean 0 and std 1, in float throughout
/// (aten/src/ATen/native/DistributionTemplates.h). `d` holds 16 uniforms on
/// entry and 16 normals on exit; lanes j and j+8 are the cosine and sine halves
/// of the same pair, and that pairing is part of the contract.
void normal_fill_16(float * d) {
    for (int j = 0; j < 8; ++j) {
        const float u1 = 1.0f - d[j];
        const float u2 = d[j + 8];
        const float radius = std::sqrt(-2.0f * std::log(u1));
        const float theta = static_cast<float>(2.0 * M_PI) * u2;
        d[j] = radius * std::cos(theta);
        d[j + 8] = radius * std::sin(theta);
    }
}

} // namespace

void sha256(const void * data, size_t size, uint8_t digest[32]) {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    const size_t full = size / 64;
    const size_t rem = size % 64;
    for (size_t i = 0; i < full; ++i) {
        sha256_block(h, bytes + i * 64);
    }

    uint8_t tail[128] = {};
    if (rem != 0) {
        std::memcpy(tail, bytes + full * 64, rem);
    }
    tail[rem] = 0x80;
    const size_t tail_len = rem < 56 ? 64u : 128u;
    const uint64_t bits = static_cast<uint64_t>(size) * 8u;
    for (int i = 0; i < 8; ++i) {
        tail[tail_len - 1 - static_cast<size_t>(i)] =
            static_cast<uint8_t>((bits >> (8 * i)) & 0xffu);
    }
    sha256_block(h, tail);
    if (tail_len == 128) {
        sha256_block(h, tail + 64);
    }

    for (int i = 0; i < 8; ++i) {
        digest[4 * i]     = static_cast<uint8_t>(h[i] >> 24);
        digest[4 * i + 1] = static_cast<uint8_t>(h[i] >> 16);
        digest[4 * i + 2] = static_cast<uint8_t>(h[i] >> 8);
        digest[4 * i + 3] = static_cast<uint8_t>(h[i]);
    }
}

uint64_t seed_from_bytes(const void * data, size_t size) {
    uint8_t digest[32];
    sha256(data, size, digest);
    uint64_t seed = 0;
    for (int i = 0; i < 8; ++i) {
        seed = (seed << 8) | digest[i];
    }
    return seed;
}

uint64_t seed_from_text(std::string_view text) {
    return seed_from_bytes(text.data(), text.size());
}

// ---------------------------------------------------------------------------
// MT19937
// ---------------------------------------------------------------------------

namespace {

constexpr int MT_N = 624;
constexpr int MT_M = 397;

uint32_t mt_twist(uint32_t u, uint32_t v) {
    const uint32_t mixed = (u & 0x80000000u) | (v & 0x7fffffffu);
    return (mixed >> 1) ^ ((v & 1u) ? 0x9908b0dfu : 0u);
}

} // namespace

Mt19937::Mt19937(uint64_t seed) {
    // ATen's mt19937_engine::init_with_uint32: the seed is truncated to 32
    // bits, so manual_seed(x) and manual_seed(x + 2^32) give the same stream.
    _state[0] = static_cast<uint32_t>(seed & 0xffffffffu);
    for (int i = 1; i < MT_N; ++i) {
        _state[i] = 1812433253u * (_state[i - 1] ^ (_state[i - 1] >> 30)) +
                    static_cast<uint32_t>(i);
    }
    _left = 1;
    _next = 0;
}

void Mt19937::next_state() {
    uint32_t * s = _state;
    _left = MT_N;
    _next = 0;
    int i = 0;
    for (; i < MT_N - MT_M; ++i) {
        s[i] = s[i + MT_M] ^ mt_twist(s[i], s[i + 1]);
    }
    for (; i < MT_N - 1; ++i) {
        s[i] = s[i + MT_M - MT_N] ^ mt_twist(s[i], s[i + 1]);
    }
    s[MT_N - 1] = s[MT_M - 1] ^ mt_twist(s[MT_N - 1], s[0]);
}

uint32_t Mt19937::next_uint32() {
    if (--_left <= 0) {
        next_state();
    }
    uint32_t y = _state[_next++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

float Mt19937::next_uniform() {
    return static_cast<float>(next_uint32() & 0xffffffu) * (1.0f / 16777216.0f);
}

void uniform_stream(uint64_t seed, size_t n, std::vector<float> & out) {
    Mt19937 gen(seed);
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = gen.next_uniform();
    }
}

bool seeded_noise(uint64_t seed, size_t channels, size_t frames,
                  std::vector<float> & out, std::string & error) {
    if (channels == 0 || frames == 0) {
        error = "sanoTTS noise needs a non-zero channel and frame count";
        return false;
    }
    if (frames > std::numeric_limits<size_t>::max() / channels) {
        error = "sanoTTS noise size overflows";
        return false;
    }
    const size_t size = channels * frames;
    if (size < 16) {
        // torch's normal_ dispatches sizes below 16 to a cached-pair scalar
        // path with a different draw order. Refusing is the honest answer;
        // approximating it would give plausible-looking wrong audio.
        error = "sanoTTS noise needs at least 16 values, got "
              + std::to_string(size);
        return false;
    }

    Mt19937 gen(seed);
    out.resize(size);

    // torch fills the whole tensor with uniforms first, then transforms it in
    // place in batches of 16. The ragged tail is drawn *after* the aligned
    // part and overwrites the last 16 values, which is why it cannot simply be
    // transformed where it lies.
    for (size_t i = 0; i < size; ++i) {
        out[i] = gen.next_uniform();
    }
    for (size_t i = 0; i + 16 <= size; i += 16) {
        normal_fill_16(out.data() + i);
    }
    if (size % 16 != 0) {
        float tail[16];
        for (int j = 0; j < 16; ++j) {
            tail[j] = gen.next_uniform();
        }
        normal_fill_16(tail);
        for (int j = 0; j < 16; ++j) {
            out[size - 16 + static_cast<size_t>(j)] = tail[j];
        }
    }
    return true;
}

uint64_t chunk_seed(uint64_t base_seed, uint32_t chunk_index) {
    uint8_t buffer[12];
    for (int i = 0; i < 8; ++i) {
        buffer[i] = static_cast<uint8_t>(base_seed >> (8 * (7 - i)));
    }
    for (int i = 0; i < 4; ++i) {
        buffer[8 + i] = static_cast<uint8_t>(chunk_index >> (8 * (3 - i)));
    }
    return seed_from_bytes(buffer, sizeof(buffer));
}

uint64_t derive_base_seed(std::string_view provenance, std::string_view voice,
                          const std::vector<uint32_t> & ids) {
    // Length-prefix each part so that ("ab", "c") and ("a", "bc") cannot
    // collide.
    std::vector<uint8_t> buffer;
    buffer.reserve(provenance.size() + voice.size() + ids.size() * 4 + 12);

    const auto push_u32 = [&buffer](uint32_t v) {
        buffer.push_back(static_cast<uint8_t>(v >> 24));
        buffer.push_back(static_cast<uint8_t>(v >> 16));
        buffer.push_back(static_cast<uint8_t>(v >> 8));
        buffer.push_back(static_cast<uint8_t>(v));
    };
    const auto push_bytes = [&buffer, &push_u32](std::string_view s) {
        push_u32(static_cast<uint32_t>(s.size()));
        buffer.insert(buffer.end(), s.begin(), s.end());
    };

    push_bytes(provenance);
    push_bytes(voice);
    push_u32(static_cast<uint32_t>(ids.size()));
    for (uint32_t id : ids) {
        push_u32(id);
    }

    return seed_from_bytes(buffer.data(), buffer.size());
}

} // namespace kokopop::sano
