#include "arch/sanotts/sano_noise.h"
#include "sanotts_fixtures.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace sano = kokopop::sano;

namespace {

std::string hex(const uint8_t * bytes, size_t n) {
    static const char * digits = "0123456789abcdef";
    std::string out;
    out.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(digits[bytes[i] >> 4]);
        out.push_back(digits[bytes[i] & 0x0f]);
    }
    return out;
}

std::string sha256_hex(const std::string & text) {
    uint8_t digest[32];
    sano::sha256(text.data(), text.size(), digest);
    return hex(digest, 32);
}

} // namespace

// ---------------------------------------------------------------------------
// SHA-256 — checked against the published FIPS 180-4 vectors, so the seed
// derivation does not depend on any fixture being present.
// ---------------------------------------------------------------------------

TEST_CASE("sanotts_sha256_matches_the_published_vectors") {
    CHECK_EQ(sha256_hex(""),
             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK_EQ(sha256_hex("abc"),
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
             "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

// The message-length encoding has three regimes: one block, one block with the
// length in the last 8 bytes, and two blocks. The row ids upstream hashes are
// short, but the utterance-derived seed is not.
TEST_CASE("sanotts_sha256_handles_every_padding_regime") {
    // 55 bytes: padding and length still fit in one block.
    CHECK_EQ(sha256_hex(std::string(55, 'a')),
             "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
    // 56 bytes: 0x80 fits, the length does not — a second block is required.
    CHECK_EQ(sha256_hex(std::string(56, 'a')),
             "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
    // 64 bytes: exactly one full block plus an all-padding block.
    CHECK_EQ(sha256_hex(std::string(64, 'a')),
             "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
    // 119 bytes: a full block plus a 55-byte remainder — padding and length
    // still share the final block.
    CHECK_EQ(sha256_hex(std::string(119, 'a')),
             "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb");
    // 120 bytes: a full block plus a 56-byte remainder — the length spills
    // into a third block.
    CHECK_EQ(sha256_hex(std::string(120, 'a')),
             "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c");
}

TEST_CASE("sanotts_seed_from_text_reads_eight_bytes_big_endian") {
    // The row ids of the en_us_r227f32 fixture, with the seeds rows.txt
    // carries. These come from `sha256(row_id).digest()[:8]` big-endian, which
    // is what upstream's renderer seeds torch with.
    CHECK_EQ(sano::seed_from_text("000001_i200"), 2236265385529901705ull);
    CHECK_EQ(sano::seed_from_text("000002_i400"), 3701943533470052915ull);
    CHECK_EQ(sano::seed_from_text("000008_i1600"), 6093641443730306605ull);

    // The first 8 digest bytes read big-endian, spelled out once.
    uint8_t digest[32];
    sano::sha256("000001_i200", 11, digest);
    uint64_t manual = 0;
    for (int i = 0; i < 8; ++i) {
        manual = (manual << 8) | digest[i];
    }
    CHECK_EQ(manual, sano::seed_from_text("000001_i200"));
}

// ---------------------------------------------------------------------------
// MT19937 and the 24-bit uniform — bit-exact against torch.rand.
//
// The expected values were produced with torch 2.12.0 (CPU) by
//   g = torch.Generator().manual_seed(SEED); torch.rand(8, generator=g)
// and are exact float32 values, so the comparison is equality, not a
// tolerance: this half of the generator has no libm dependency.
// ---------------------------------------------------------------------------

TEST_CASE("sanotts_uniform_stream_is_bit_exact_against_torch_rand") {
    struct Case {
        uint64_t seed;
        float expected[8];
    };
    const Case cases[] = {
        {0ull, {0.49625658989f, 0.76822179556f, 0.08847743273f, 0.13203048706f,
                0.30742281675f, 0.63407868147f, 0.49009340978f, 0.89644473791f}},
        {1ull, {0.75763159990f, 0.27931088209f, 0.40306925774f, 0.73468446732f,
                0.02928155661f, 0.79985862970f, 0.39713734388f, 0.75437194109f}},
        {42ull, {0.88226926327f, 0.91500395536f, 0.38286375999f, 0.95930564404f,
                 0.39044821262f, 0.60089534521f, 0.25657248497f, 0.79364132881f}},
        // The seed of row 000001_i200 of the golden fixture.
        {2236265385529901705ull,
         {0.64725434780f, 0.97836315632f, 0.82821458578f, 0.73709452152f,
          0.60942178965f, 0.33820986748f, 0.89661461115f, 0.94680511951f}},
    };

    for (const Case & c : cases) {
        std::vector<float> got;
        sano::uniform_stream(c.seed, 8, got);
        REQUIRE_EQ(got.size(), 8u);
        for (int i = 0; i < 8; ++i) {
            // Exact: the uniform is (raw & 0xffffff) * 2^-24, both sides in
            // float32, so any difference is a real divergence.
            CHECK_EQ(got[static_cast<size_t>(i)], c.expected[i]);
        }
    }
}

// A wrong mask (a 5-bit shift, or 32 bits instead of 24) would still look like
// noise, so pin the property directly: every value is a multiple of 2^-24 and
// lies in [0, 1).
TEST_CASE("sanotts_uniform_uses_exactly_24_significant_bits") {
    std::vector<float> values;
    sano::uniform_stream(12345ull, 4096, values);
    for (float v : values) {
        CHECK(v >= 0.0f);
        CHECK(v < 1.0f);
        const float scaled = v * 16777216.0f;
        CHECK_EQ(scaled, std::floor(scaled));
        CHECK(scaled <= 16777215.0f);
    }
}

// Only the low 32 bits of the seed take part (init_with_uint32).
TEST_CASE("sanotts_mt19937_truncates_the_seed_to_32_bits") {
    std::vector<float> low;
    std::vector<float> high;
    sano::uniform_stream(7ull, 16, low);
    sano::uniform_stream(7ull + (1ull << 32), 16, high);
    CHECK_EQ(low, high);
}

// The engine must regenerate its state after 624 draws, not repeat itself.
TEST_CASE("sanotts_mt19937_crosses_its_state_boundary") {
    std::vector<float> values;
    sano::uniform_stream(99ull, 1300, values);
    REQUIRE_EQ(values.size(), 1300u);
    // A stuck twist would make the second block equal the first.
    bool differs = false;
    for (size_t i = 0; i < 624; ++i) {
        if (values[i] != values[i + 624]) {
            differs = true;
            break;
        }
    }
    CHECK(differs);
}

// ---------------------------------------------------------------------------
// Box-Muller
// ---------------------------------------------------------------------------

TEST_CASE("sanotts_seeded_noise_rejects_sizes_torch_draws_differently") {
    std::vector<float> out;
    std::string error;
    // Below 16 values torch takes a cached-pair scalar path with a different
    // draw order; refusing beats approximating.
    CHECK_FALSE(sano::seeded_noise(1ull, 4, 3, out, error));
    CHECK(error.find("at least 16") != std::string::npos);

    error.clear();
    CHECK(sano::seeded_noise(1ull, 4, 4, out, error));
    CHECK(error.empty());
    CHECK_EQ(out.size(), 16u);
}

TEST_CASE("sanotts_seeded_noise_is_reproducible_and_seed_sensitive") {
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    std::string error;
    REQUIRE(sano::seeded_noise(4242ull, 4, 100, a, error));
    REQUIRE(sano::seeded_noise(4242ull, 4, 100, b, error));
    REQUIRE(sano::seeded_noise(4243ull, 4, 100, c, error));
    CHECK_EQ(a, b);
    CHECK(a != c);
}

// The first 16 normals are a pure function of the first 16 uniforms; deriving
// them here independently pins the lane pairing (j with j+8) that upstream's
// normal_fill_16 uses.
TEST_CASE("sanotts_box_muller_pairs_lane_j_with_lane_j_plus_8") {
    constexpr uint64_t seed = 20260904ull;
    std::vector<float> uniforms;
    sano::uniform_stream(seed, 16, uniforms);

    std::vector<float> noise;
    std::string error;
    REQUIRE(sano::seeded_noise(seed, 4, 4, noise, error));
    REQUIRE_EQ(noise.size(), 16u);

    for (int j = 0; j < 8; ++j) {
        const float u1 = 1.0f - uniforms[static_cast<size_t>(j)];
        const float u2 = uniforms[static_cast<size_t>(j + 8)];
        const float radius = std::sqrt(-2.0f * std::log(u1));
        const float theta = static_cast<float>(2.0 * 3.14159265358979323846) * u2;
        CHECK_NEAR(noise[static_cast<size_t>(j)], radius * std::cos(theta), 1e-6);
        CHECK_NEAR(noise[static_cast<size_t>(j + 8)], radius * std::sin(theta), 1e-6);
    }
}

// The ragged tail is drawn *after* the aligned batches and overwrites the last
// 16 values in place, so it reaches back before the last 16-aligned boundary.
// Everything strictly before `size - 16` comes from the aligned batches alone
// and must be independent of the tail.
TEST_CASE("sanotts_seeded_noise_ragged_tail_only_rewrites_the_last_16") {
    std::vector<float> aligned;   // 4 x 8  = 32, no tail
    std::vector<float> ragged;    // 4 x 10 = 40, tail of 8
    std::string error;
    REQUIRE(sano::seeded_noise(5ull, 4, 8, aligned, error));
    REQUIRE(sano::seeded_noise(5ull, 4, 10, ragged, error));
    REQUIRE_EQ(aligned.size(), 32u);
    REQUIRE_EQ(ragged.size(), 40u);

    const size_t untouched = ragged.size() - 16u;  // 24
    for (size_t i = 0; i < untouched; ++i) {
        CHECK_EQ(aligned[i], ragged[i]);
    }
    // And the overlap region really is rewritten, otherwise the assertion
    // above would be vacuous.
    bool rewritten = false;
    for (size_t i = untouched; i < aligned.size(); ++i) {
        if (aligned[i] != ragged[i]) {
            rewritten = true;
            break;
        }
    }
    CHECK(rewritten);
}

TEST_CASE("sanotts_seeded_noise_is_roughly_standard_normal") {
    std::vector<float> noise;
    std::string error;
    REQUIRE(sano::seeded_noise(1234ull, 4, 8192, noise, error));

    double sum = 0.0;
    double sum_sq = 0.0;
    for (float v : noise) {
        CHECK(std::isfinite(v));
        sum += v;
        sum_sq += static_cast<double>(v) * v;
    }
    const double n = static_cast<double>(noise.size());
    const double mean = sum / n;
    const double var = sum_sq / n - mean * mean;
    CHECK(std::fabs(mean) < 0.03);
    CHECK(std::fabs(std::sqrt(var) - 1.0) < 0.03);
}

// ---------------------------------------------------------------------------
// Seed plumbing
// ---------------------------------------------------------------------------

TEST_CASE("sanotts_chunk_seed_separates_chunks_and_stays_reproducible") {
    const uint64_t base = 0xfeedfacecafebeefull;
    CHECK_EQ(sano::chunk_seed(base, 0), sano::chunk_seed(base, 0));
    CHECK(sano::chunk_seed(base, 0) != sano::chunk_seed(base, 1));
    CHECK(sano::chunk_seed(base, 0) != sano::chunk_seed(base + 1, 0));
    // Chunk 0 is not just the base seed passed through: two adjacent
    // utterances would otherwise share their first chunk's noise.
    CHECK(sano::chunk_seed(base, 0) != base);
}

TEST_CASE("sanotts_derive_base_seed_depends_on_every_input") {
    const std::vector<uint32_t> ids{1, 0, 42, 0, 7, 2};
    const uint64_t reference = sano::derive_base_seed("pack@rev", "heart", ids);

    CHECK_EQ(reference, sano::derive_base_seed("pack@rev", "heart", ids));
    CHECK(reference != sano::derive_base_seed("pack@other", "heart", ids));
    CHECK(reference != sano::derive_base_seed("pack@rev", "heartnano", ids));
    CHECK(reference != sano::derive_base_seed("pack@rev", "heart", {1, 0, 42, 0, 7, 3}));
    CHECK(reference != sano::derive_base_seed("pack@rev", "heart", {1, 0, 42, 0, 7}));

    // Length-prefixing: moving a character across the field boundary must not
    // collide.
    CHECK(sano::derive_base_seed("ab", "c", {}) !=
          sano::derive_base_seed("a", "bc", {}));
}

// ---------------------------------------------------------------------------
// Golden fixtures
//
// These files are upstream data, not covered by its MIT scope, so they are
// never redistributed here: the tests read them from a local checkout and skip
// when it is absent. See sanotts_fixtures.h.
// ---------------------------------------------------------------------------

TEST_CASE("sanotts_uniform_stream_matches_e2e_uniform_bin") {
    const std::string path =
        kokopop::test::sanotts_fixture_path("en_us_r227f32", "e2e_uniform.bin");
    if (path.empty()) {
        MESSAGE("skipping: " << kokopop::test::sanotts_fixture_hint());
        return;
    }
    std::vector<float> expected;
    REQUIRE(kokopop::test::read_f32_file(path, expected));
    REQUIRE_EQ(expected.size(), 64u);

    std::vector<float> got;
    sano::uniform_stream(sano::seed_from_text("000001_i200"), expected.size(), got);
    // Bit-exact by contract: this is the half of the generator that has no
    // libm dependency.
    CHECK_EQ(got, expected);
}

TEST_CASE("sanotts_seeded_noise_matches_e2e_noise_bin") {
    const std::string path =
        kokopop::test::sanotts_fixture_path("en_us_r227f32", "e2e_noise.bin");
    if (path.empty()) {
        MESSAGE("skipping: " << kokopop::test::sanotts_fixture_hint());
        return;
    }
    std::vector<float> expected;
    REQUIRE(kokopop::test::read_f32_file(path, expected));
    // [4, T] for row 000001_i200, whose rows.txt entry gives T = 420.
    REQUIRE_EQ(expected.size(), 4u * 420u);

    std::vector<float> got;
    std::string error;
    REQUIRE(sano::seeded_noise(sano::seed_from_text("000001_i200"), 4, 420, got, error));
    REQUIRE_EQ(got.size(), expected.size());

    // Not exact: logf/cosf/sinf differ by a few ulp between C libraries.
    // Upstream measures at most 1.9e-6 on this fixture and holds it to 1e-5.
    double worst = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(got[i] - expected[i])));
    }
    CHECK(worst <= 1e-5);
    MESSAGE("e2e_noise worst absolute delta: " << worst);
}
