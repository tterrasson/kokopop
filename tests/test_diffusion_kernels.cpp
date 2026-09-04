#include "test_helpers.h"
#include "arch/kokoro/diffusion_kernels.h"

#include <cmath>
#include <vector>

namespace {

std::vector<float> patterned(size_t n, float scale = 1.0f) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        const int k = static_cast<int>((i * 37 + 11) % 29) - 14;
        v[i] = scale * static_cast<float>(k) / 13.0f;
    }
    return v;
}

void check_vec_near(const std::vector<float> & actual, const std::vector<float> & expected, double eps) {
    REQUIRE_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        CHECK_NEAR(actual[i], expected[i], eps);
    }
}

} // namespace

TEST_CASE("diffusion_kernel dot and reductions match scalar references") {
    for (int64_t len : {1, 3, 4, 7, 8, 15, 16, 31, 32, 63}) {
        const std::vector<float> a = patterned(static_cast<size_t>(len), 0.7f);
        const std::vector<float> b = patterned(static_cast<size_t>(len), -0.4f);
        const float mean = kokopop::diffusion_kernel::sum_scalar(a.data(), len) / static_cast<float>(len);

        CHECK_NEAR(kokopop::diffusion_kernel::dot(a.data(), b.data(), len),
                   kokopop::diffusion_kernel::dot_scalar(a.data(), b.data(), len), 1e-5);
        CHECK_NEAR(kokopop::diffusion_kernel::sum(a.data(), len),
                   kokopop::diffusion_kernel::sum_scalar(a.data(), len), 1e-5);
        CHECK_NEAR(kokopop::diffusion_kernel::squared_diff_sum(a.data(), len, mean),
                   kokopop::diffusion_kernel::squared_diff_sum_scalar(a.data(), len, mean), 1e-5);
    }
}

TEST_CASE("diffusion_kernel layer norm helpers match scalar references") {
    constexpr int64_t dim = 33;
    const std::vector<float> x = patterned(static_cast<size_t>(dim), 0.9f);
    const std::vector<float> gamma = patterned(static_cast<size_t>(dim), 0.2f);
    const std::vector<float> beta = patterned(static_cast<size_t>(dim), -0.15f);
    const float mean = kokopop::diffusion_kernel::sum_scalar(x.data(), dim) / static_cast<float>(dim);
    const float var = kokopop::diffusion_kernel::squared_diff_sum_scalar(x.data(), dim, mean) / static_cast<float>(dim);
    const float inv_std = 1.0f / std::sqrt(var + 1e-5f);

    std::vector<float> actual(static_cast<size_t>(dim));
    std::vector<float> expected(static_cast<size_t>(dim));
    kokopop::diffusion_kernel::layer_norm_affine(
        actual.data(), x.data(), gamma.data(), beta.data(), dim, mean, inv_std);
    kokopop::diffusion_kernel::layer_norm_affine_scalar(
        expected.data(), x.data(), gamma.data(), beta.data(), dim, mean, inv_std);
    check_vec_near(actual, expected, 1e-5);

    kokopop::diffusion_kernel::ada_layer_norm(
        actual.data(), x.data(), gamma.data(), beta.data(), dim, mean, inv_std);
    kokopop::diffusion_kernel::ada_layer_norm_scalar(
        expected.data(), x.data(), gamma.data(), beta.data(), dim, mean, inv_std);
    check_vec_near(actual, expected, 1e-5);
}

TEST_CASE("diffusion_kernel attention weighted sum handles head tails") {
    constexpr int64_t rows = 5;
    constexpr int64_t stride = 41;
    constexpr int64_t head_dim = 33;
    std::vector<float> values = patterned(static_cast<size_t>(rows * stride), 0.5f);
    std::vector<float> probs = {0.25f, 0.5f, 0.125f, 0.75f, 0.375f};
    std::vector<float> actual(static_cast<size_t>(head_dim));
    std::vector<float> expected(static_cast<size_t>(head_dim));

    kokopop::diffusion_kernel::attention_weighted_sum(
        actual.data(), values.data() + 3, probs.data(), 0.5f, rows, stride, head_dim);
    kokopop::diffusion_kernel::attention_weighted_sum_scalar(
        expected.data(), values.data() + 3, probs.data(), 0.5f, rows, stride, head_dim);

    check_vec_near(actual, expected, 1e-5);
}

TEST_CASE("diffusion_kernel vector sampling helpers match scalar references") {
    constexpr size_t n = 37;
    const std::vector<float> a = patterned(n, 0.8f);
    const std::vector<float> b = patterned(n, -0.3f);
    const std::vector<float> noise = patterned(n, 0.17f);

    std::vector<float> actual(n);
    std::vector<float> expected(n);
    kokopop::diffusion_kernel::scale(actual.data(), a.data(), 1.25f, n);
    kokopop::diffusion_kernel::scale_scalar(expected.data(), a.data(), 1.25f, n);
    check_vec_near(actual, expected, 1e-6);

    actual = a;
    expected = a;
    kokopop::diffusion_kernel::add_inplace(actual.data(), b.data(), n);
    kokopop::diffusion_kernel::add_inplace_scalar(expected.data(), b.data(), n);
    check_vec_near(actual, expected, 1e-6);

    kokopop::diffusion_kernel::denoise_combine(actual.data(), a.data(), b.data(), 0.35f, -0.65f, n);
    kokopop::diffusion_kernel::denoise_combine_scalar(expected.data(), a.data(), b.data(), 0.35f, -0.65f, n);
    check_vec_near(actual, expected, 1e-6);

    actual = a;
    expected = a;
    kokopop::diffusion_kernel::classifier_free_guidance(actual.data(), b.data(), 1.4f, n);
    kokopop::diffusion_kernel::classifier_free_guidance_scalar(expected.data(), b.data(), 1.4f, n);
    check_vec_near(actual, expected, 1e-6);

    kokopop::diffusion_kernel::euler_midpoint(actual.data(), a.data(), b.data(), 2.0f, 1.2f, n);
    kokopop::diffusion_kernel::euler_midpoint_scalar(expected.data(), a.data(), b.data(), 2.0f, 1.2f, n);
    check_vec_near(actual, expected, 1e-6);

    actual = a;
    expected = a;
    kokopop::diffusion_kernel::euler_update_with_noise(
        actual.data(), b.data(), a.data(), noise.data(), 1.2f, 0.75f, 2.0f, 0.08f, n);
    kokopop::diffusion_kernel::euler_update_with_noise_scalar(
        expected.data(), b.data(), a.data(), noise.data(), 1.2f, 0.75f, 2.0f, 0.08f, n);
    check_vec_near(actual, expected, 1e-6);
}

TEST_CASE("diffusion_kernel style blend matches scalar reference") {
    constexpr size_t half = 128;
    std::vector<float> actual = patterned(half * 2, 0.6f);
    std::vector<float> expected = actual;
    const std::vector<float> sampled = patterned(half * 2, -0.35f);

    kokopop::diffusion_kernel::blend_style(actual.data(), sampled.data(), 0.2f, 0.7f, half);
    kokopop::diffusion_kernel::blend_style_scalar(expected.data(), sampled.data(), 0.2f, 0.7f, half);

    check_vec_near(actual, expected, 1e-6);
}
