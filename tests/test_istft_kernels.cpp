#include "test_helpers.h"
#include "audio/istft_kernels.h"

#include <cmath>
#include <vector>

namespace istft_kernel = kokopop::istft_kernel;

namespace {

/// Deterministic, spread over both signs, and never zero — a kernel that
/// dropped a lane would have to hit the same value by accident.
std::vector<float> kernel_patterned(size_t n, float scale = 1.0f, int phase = 0) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        const int k = static_cast<int>((i * 37 + 11 + phase * 7) % 29) - 14;
        v[i] = scale * (static_cast<float>(k) + 0.5f) / 13.0f;
    }
    return v;
}

void kernel_check_near(const std::vector<float> & actual,
                    const std::vector<float> & expected, double eps) {
    REQUIRE_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        INFO("lane " << i);
        CHECK_NEAR(actual[i], expected[i], eps);
    }
}

// The lengths worth covering: shorter than one vector, exactly one, a tail on
// each width, and enough for the unrolled body.
const std::vector<size_t> KERNEL_LENGTHS = {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 20, 32, 33};

} // namespace

TEST_CASE("istft_kernel_add_inplace_matches_the_scalar_reference") {
    for (size_t n : KERNEL_LENGTHS) {
        const std::vector<float> src = kernel_patterned(n, 0.7f);
        std::vector<float> actual = kernel_patterned(n, -0.3f, 1);
        std::vector<float> expected = actual;

        istft_kernel::add_inplace(actual.data(), src.data(), n);
        istft_kernel::add_inplace_scalar(expected.data(), src.data(), n);
        kernel_check_near(actual, expected, 0.0);
    }
}

TEST_CASE("istft_kernel_accumulate_rotation_matches_the_scalar_reference") {
    for (size_t n : KERNEL_LENGTHS) {
        const std::vector<float> c = kernel_patterned(n, 0.9f);
        const std::vector<float> s = kernel_patterned(n, 0.8f, 2);
        std::vector<float> actual = kernel_patterned(n, 0.2f, 3);
        std::vector<float> expected = actual;

        istft_kernel::accumulate_rotation(actual.data(), 1.5f, c.data(), -0.75f,
                                          s.data(), n);
        istft_kernel::accumulate_rotation_scalar(expected.data(), 1.5f, c.data(),
                                                 -0.75f, s.data(), n);
        // The vector form is free to contract into an FMA, which rounds once
        // where the scalar reference rounds twice.
        kernel_check_near(actual, expected, 1e-6);
    }
}

TEST_CASE("istft_kernel_window_scaled_matches_the_scalar_reference") {
    for (size_t n : KERNEL_LENGTHS) {
        const std::vector<float> src = kernel_patterned(n, 1.3f);
        const std::vector<float> window = kernel_patterned(n, 0.5f, 1);
        std::vector<float> actual(n, 0.0f);
        std::vector<float> expected(n, 0.0f);

        istft_kernel::window_scaled(actual.data(), src.data(), window.data(), 0.05f, n);
        istft_kernel::window_scaled_scalar(expected.data(), src.data(), window.data(), 0.05f, n);
        kernel_check_near(actual, expected, 0.0);
    }
}

TEST_CASE("istft_kernel_interleave_windowed_matches_the_scalar_reference") {
    for (size_t half : KERNEL_LENGTHS) {
        const std::vector<float> zr = kernel_patterned(half, 1.1f);
        const std::vector<float> zi = kernel_patterned(half, 0.9f, 1);
        const std::vector<float> we = kernel_patterned(half, 0.4f, 2);
        const std::vector<float> wo = kernel_patterned(half, 0.6f, 3);
        std::vector<float> actual(2 * half, 0.0f);
        std::vector<float> expected(2 * half, 0.0f);

        istft_kernel::interleave_windowed(actual.data(), zr.data(), zi.data(),
                                          we.data(), wo.data(), half);
        istft_kernel::interleave_windowed_scalar(expected.data(), zr.data(), zi.data(),
                                                 we.data(), wo.data(), half);
        kernel_check_near(actual, expected, 0.0);
    }
}

TEST_CASE("istft_kernel_normalise_matches_the_scalar_reference") {
    constexpr float min_env = 1e-8f;
    for (size_t n : KERNEL_LENGTHS) {
        const std::vector<float> ola = kernel_patterned(n, 2.0f);
        std::vector<float> env = kernel_patterned(n, 1.0f, 1);
        for (size_t i = 0; i < n; ++i) {
            // Positive envelopes, with an uncovered sample every third slot so
            // both sides of the threshold are exercised in the same vector.
            env[i] = std::fabs(env[i]) + 0.25f;
            if (i % 3 == 0) {
                env[i] = 0.0f;
            }
        }
        std::vector<float> actual(n, -1.0f);
        std::vector<float> expected(n, -1.0f);

        istft_kernel::normalise(actual.data(), ola.data(), env.data(), n, min_env);
        istft_kernel::normalise_scalar(expected.data(), ola.data(), env.data(), n, min_env);
        kernel_check_near(actual, expected, 0.0);
        for (size_t i = 0; i < n; i += 3) {
            CHECK(actual[i] == 0.0f);
        }
    }
}

TEST_CASE("istft_kernel_hermitian_split_matches_the_scalar_reference") {
    for (size_t half : KERNEL_LENGTHS) {
        // The split reads bins 0..half, one more than it writes.
        const std::vector<float> re = kernel_patterned(half + 1, 1.0f);
        const std::vector<float> im = kernel_patterned(half + 1, 0.8f, 1);
        const std::vector<float> sc = kernel_patterned(half, 0.99f, 2);
        const std::vector<float> ss = kernel_patterned(half, 0.99f, 3);
        std::vector<float> actual_r(half, 0.0f);
        std::vector<float> actual_i(half, 0.0f);
        std::vector<float> expected_r(half, 0.0f);
        std::vector<float> expected_i(half, 0.0f);

        istft_kernel::hermitian_split(re.data(), im.data(), sc.data(), ss.data(),
                                      0.5f, actual_r.data(), actual_i.data(), half);
        istft_kernel::hermitian_split_scalar(re.data(), im.data(), sc.data(), ss.data(),
                                             0.5f, expected_r.data(), expected_i.data(), half);
        kernel_check_near(actual_r, expected_r, 1e-6);
        kernel_check_near(actual_i, expected_i, 1e-6);
    }
}

TEST_CASE("istft_kernel_fft_stages_match_the_scalar_reference") {
    for (size_t m : {2u, 4u, 8u, 16u, 32u, 64u}) {
        std::vector<float> actual_r = kernel_patterned(m, 1.0f);
        std::vector<float> actual_i = kernel_patterned(m, 0.7f, 1);
        std::vector<float> expected_r = actual_r;
        std::vector<float> expected_i = actual_i;

        istft_kernel::fft_stage_pairs(actual_r.data(), actual_i.data(), m);
        istft_kernel::fft_stage_pairs_scalar(expected_r.data(), expected_i.data(), m);
        kernel_check_near(actual_r, expected_r, 0.0);
        kernel_check_near(actual_i, expected_i, 0.0);

        for (size_t h = 2; h < m; h <<= 1) {
            const std::vector<float> twc = kernel_patterned(h, 0.99f, 2);
            const std::vector<float> tws = kernel_patterned(h, 0.99f, 3);
            istft_kernel::fft_stage(actual_r.data(), actual_i.data(),
                                    twc.data(), tws.data(), m, h);
            istft_kernel::fft_stage_scalar(expected_r.data(), expected_i.data(),
                                           twc.data(), tws.data(), m, h);
            // Errors ride through every stage applied so far, and each stage
            // can double the magnitude.
            kernel_check_near(actual_r, expected_r, 1e-4);
            kernel_check_near(actual_i, expected_i, 1e-4);
        }
    }
}

TEST_CASE("istft_kernel_gather_strided_reads_every_stride") {
    const std::vector<float> src = kernel_patterned(40, 1.0f);
    std::vector<float> got(8, 0.0f);

    istft_kernel::gather_strided(got.data(), src.data() + 3, 5, got.size());
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK(got[i] == src[3 + 5 * i]);
    }

    // A negative stride is what a spectrum stored backwards would hand in.
    istft_kernel::gather_strided(got.data(), src.data() + 39, -3, got.size());
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK(got[i] == src[39 - 3 * i]);
    }
}
