#include "test_helpers.h"
#include "inference/lstm_fused.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

float sigmoid_ref(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void lstm_reference(
    const float * pre_gates,
    const float * w_hh,
    const float * b_hh,
    int64_t hidden,
    int64_t n_steps,
    bool reverse,
    std::vector<float> & output) {
    const int64_t h4 = 4 * hidden;
    std::vector<float> h(static_cast<size_t>(hidden), 0.0f);
    std::vector<float> c(static_cast<size_t>(hidden), 0.0f);
    std::vector<float> gates(static_cast<size_t>(h4), 0.0f);
    output.assign(static_cast<size_t>(hidden * n_steps), 0.0f);

    for (int64_t step = 0; step < n_steps; ++step) {
        const int64_t t = reverse ? (n_steps - 1 - step) : step;
        const float * pg_t = pre_gates + h4 * t;
        for (int64_t j = 0; j < h4; ++j) {
            float dot = 0.0f;
            const float * col = w_hh + j * hidden;
            for (int64_t k = 0; k < hidden; ++k) {
                dot += col[k] * h[static_cast<size_t>(k)];
            }
            gates[static_cast<size_t>(j)] = b_hh[j] + pg_t[j] + dot;
        }

        float * out_t = output.data() + hidden * t;
        for (int64_t i = 0; i < hidden; ++i) {
            const float i_gate = sigmoid_ref(gates[static_cast<size_t>(i)]);
            const float f_gate = sigmoid_ref(gates[static_cast<size_t>(hidden + i)]);
            const float g_gate = std::tanh(gates[static_cast<size_t>(2 * hidden + i)]);
            const float o_gate = sigmoid_ref(gates[static_cast<size_t>(3 * hidden + i)]);

            float c_new = f_gate * c[static_cast<size_t>(i)] + i_gate * g_gate;
            c_new = std::clamp(c_new, -50.0f, 50.0f);
            c[static_cast<size_t>(i)] = c_new;
            h[static_cast<size_t>(i)] = o_gate * std::tanh(c_new);
            out_t[i] = h[static_cast<size_t>(i)];
        }
    }
}

void run_lstm_fused_case(int64_t hidden, int64_t n_steps, bool reverse) {
    const int64_t h4 = 4 * hidden;
    std::vector<float> pre_gates(static_cast<size_t>(h4 * n_steps));
    std::vector<float> w_hh(static_cast<size_t>(h4 * hidden));
    std::vector<float> b_hh(static_cast<size_t>(h4));
    std::vector<float> w_rowwise(static_cast<size_t>(h4 * hidden));

    for (size_t i = 0; i < pre_gates.size(); ++i) {
        pre_gates[i] = 0.17f * std::sin(static_cast<float>(i) * 0.37f)
                     + 0.05f * std::cos(static_cast<float>(i) * 0.11f);
    }
    for (size_t i = 0; i < w_hh.size(); ++i) {
        w_hh[i] = 0.08f * std::sin(static_cast<float>(i) * 0.19f);
    }
    for (size_t i = 0; i < b_hh.size(); ++i) {
        b_hh[i] = 0.03f * std::cos(static_cast<float>(i) * 0.23f);
    }
    for (int64_t j = 0; j < h4; ++j) {
        for (int64_t k = 0; k < hidden; ++k) {
            w_rowwise[static_cast<size_t>(j * hidden + k)] =
                w_hh[static_cast<size_t>(k + hidden * j)];
        }
    }

    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * pre = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h4, n_steps);
    ggml_tensor * out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_steps);
    REQUIRE(pre != nullptr);
    REQUIRE(out != nullptr);
    std::memcpy(pre->data, pre_gates.data(), pre_gates.size() * sizeof(float));
    std::fill_n(static_cast<float *>(out->data), hidden * n_steps, -99.0f);

    kokopop::LstmCustomParams op_params{
        w_hh.data(),
        b_hh.data(),
        w_rowwise.data(),
        nullptr,
        nullptr,
        hidden,
        n_steps,
        reverse,
    };

    kokopop::lstm_fused_callback(out, nullptr, pre, 0, 1, &op_params);

    std::vector<float> expected;
    lstm_reference(pre_gates.data(), w_hh.data(), b_hh.data(), hidden, n_steps, reverse, expected);
    const float * actual = static_cast<const float *>(out->data);
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK_NEAR(actual[i], expected[i], 2.0e-5f);
    }

    ggml_free(ctx);
}

} // namespace

TEST_CASE("lstm_fused_matches_scalar_reference_forward") {
    run_lstm_fused_case(33, 9, false);
}

TEST_CASE("lstm_fused_matches_scalar_reference_reverse") {
    run_lstm_fused_case(33, 9, true);
}

TEST_CASE("lstm_pregates_matches_scalar_reference") {
    constexpr int64_t input_dim = 17;
    constexpr int64_t four_hidden = 28;
    constexpr int64_t n_steps = 11;

    std::vector<float> w(static_cast<size_t>(input_dim * four_hidden));
    std::vector<float> x(static_cast<size_t>(input_dim * n_steps));
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = 0.09f * std::sin(static_cast<float>(i) * 0.13f)
             + 0.02f * std::cos(static_cast<float>(i) * 0.29f);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.21f * std::sin(static_cast<float>(i) * 0.07f)
             - 0.04f * std::cos(static_cast<float>(i) * 0.31f);
    }

    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, n_steps);
    ggml_tensor * out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, four_hidden, n_steps);
    REQUIRE(input != nullptr);
    REQUIRE(out != nullptr);
    std::memcpy(input->data, x.data(), x.size() * sizeof(float));
    std::fill_n(static_cast<float *>(out->data), four_hidden * n_steps, -99.0f);

    kokopop::LstmPregatesParams op_params{
        nullptr,
        "test.weight_ih_l0",
        w.data(),
        static_cast<int>(input_dim),
        static_cast<int>(four_hidden),
        static_cast<int>(n_steps),
    };

    kokopop::lstm_pregates_callback(out, nullptr, input, 0, 1, &op_params);

    const float * actual = static_cast<const float *>(out->data);
    for (int64_t t = 0; t < n_steps; ++t) {
        for (int64_t g = 0; g < four_hidden; ++g) {
            float expected = 0.0f;
            for (int64_t i = 0; i < input_dim; ++i) {
                expected += w[static_cast<size_t>(i + input_dim * g)] *
                            x[static_cast<size_t>(i + input_dim * t)];
            }
            CHECK_NEAR(actual[static_cast<size_t>(g + four_hidden * t)], expected, 1.0e-6f);
        }
    }

    ggml_free(ctx);
}
