#pragma once

// Deterministic Gaussian noise for the sanoTTS vocos decoder.
//
// The decoder is noise-fed: four Gaussian channels enter through a learned
// adapter, and upstream draws them with
// `torch.randn(generator=torch.Generator().manual_seed(seed))`. Reproducing
// that draw is not optional — a different noise stream gives different audio,
// silently, with no error anywhere.
//
// So this reproduces the observable contract of ATen's generator:
//   * MT19937, seeded the way `mt19937_engine::init_with_uint32` does (only
//     the low 32 bits of the seed are used);
//   * the 24-bit uniform `(raw & 0xffffff) * 2^-24`, not a 5-bit shift;
//   * `normal_fill_16`'s Box-Muller, in batches of 16 and in its lane order.
//
// The uniform stream is bit-exact against `torch.rand`. The normal values are
// not, and cannot be: `logf`, `cosf` and `sinf` differ by a few ulp between C
// libraries, so the tests hold them to a tolerance and the uniform stream to
// exact equality.
//
// Derived from the MIT-licensed reference implementation in
// Ampixa/sanoTTS `mcu/src/snt_nano.c` (`snt_nano_sha256_seed`,
// `snt_nano_uniform_stream`, `snt_nano_seeded_noise`), commit
// 939d982b9faa54cbcf5d24cc878f5cd514b2646e. See THIRD_PARTY.md.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace kokopop::sano {

/// SHA-256 of `size` bytes at `data`.
void sha256(const void * data, size_t size, uint8_t digest[32]);

/// Upstream's seed derivation: the first 8 bytes of `sha256(text)` read
/// big-endian, i.e. `int.from_bytes(sha256(text).digest()[:8], "big")`.
///
/// The golden fixtures carry this value per row, so tests can inject the exact
/// seed upstream used.
uint64_t seed_from_bytes(const void * data, size_t size);
uint64_t seed_from_text(std::string_view text);

/// ATen's MT19937 engine.
class Mt19937 {
public:
    /// Only the low 32 bits of `seed` take part, matching
    /// `mt19937_engine::init_with_uint32`.
    explicit Mt19937(uint64_t seed);

    uint32_t next_uint32();

    /// `at::uniform_real_distribution<float>`: 24 significant bits, in [0, 1).
    float next_uniform();

private:
    void next_state();

    uint32_t _state[624];
    int _left = 1;
    int _next = 0;
};

/// The first `n` values of `torch.rand()` for `seed`. Bit-exact.
void uniform_stream(uint64_t seed, size_t n, std::vector<float> & out);

/// `torch.randn(channels, frames, generator=manual_seed(seed))`, laid out
/// row-major as `[channels, frames]`.
///
/// Fails when `channels * frames < 16`: torch dispatches those sizes to a
/// different, cached-pair scalar path, and guessing at it would produce noise
/// that looks plausible and is wrong.
bool seeded_noise(uint64_t seed, size_t channels, size_t frames,
                  std::vector<float> & out, std::string & error);

/// Per-chunk seed: `sha256(base_seed_be || chunk_index_be)[:8]`, big-endian.
///
/// Chunks must not share a noise draw — the repetition is audible — but the
/// whole utterance must stay reproducible, so the index is folded in rather
/// than a counter being carried in mutable state.
uint64_t chunk_seed(uint64_t base_seed, uint32_t chunk_index);

/// Base seed to use when the caller supplied none.
///
/// Derived from the model's provenance, the canonical voice name and the
/// utterance's token ids, so that the same request renders identically across
/// runs, processes and machines without the caller having to manage a seed.
uint64_t derive_base_seed(std::string_view provenance, std::string_view voice,
                          const std::vector<uint32_t> & ids);

} // namespace kokopop::sano
