// Minimal repro for a suspected ggml-cuda MUL_MAT bug observed in Kokoro:
// Q8_0 [K, OC] × F32 [K, N] with large N (e.g. N=7921) produces values that
// diverge from the CPU reference by more than the expected Q8_0 quantization
// noise. Pinning these matmuls on CPU restores correct audio in Kokoro.
//
// Build:
//   cmake --build build --target cuda_mulmat_repro -j
// Run (single GPU):
//   ./build/cuda_mulmat_repro            # default shape: K=1408, OC=128, N=7921
//   ./build/cuda_mulmat_repro 384 128 16 # works: small N
//   ./build/cuda_mulmat_repro 1408 128 7921

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void fill_random_f32(std::vector<float> & out, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto & v : out) v = dist(rng);
}

static double rel_l2(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)a[i] * (double)a[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b, size_t & idx) {
    float worst = 0.0f;
    idx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > worst) { worst = d; idx = i; }
    }
    return worst;
}

static bool run_one(ggml_backend_t backend,
                    ggml_backend_buffer_type_t weights_buft,
                    const std::vector<uint8_t> & q8_0_weight_bytes,
                    int64_t K, int64_t OC, int64_t N,
                    const std::vector<float> & input_f32,
                    std::vector<float> & out) {
    // Build a graph mimicking Kokoro's conv1d (quantized) pattern:
    //   X[F32, K, N] -> view-reshape to [K, N] (no-op for already 2D)
    //   IM2COL-like: emulate via add-zero to introduce a transient buffer
    //   Y = mul_mat(W_q8_0, view) followed by add(Y, Y) to add downstream
    // The simplest form is just one mul_mat. The KOKOPOP_REPRO_CHAIN env
    // var enables additional ops to see if the bug needs a chain.
    const bool chain = std::getenv("KOKOPOP_REPRO_CHAIN") != nullptr;
    const int repeat_n = std::getenv("KOKOPOP_REPRO_REPEAT")
                       ? std::atoi(std::getenv("KOKOPOP_REPRO_REPEAT")) : 0;
    struct ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (32 + repeat_n * 8) + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, OC);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, N);
    ggml_tensor * Zero = nullptr;
    // Optional chain: pass X through a no-op add + view to mimic an upstream
    // CUDA op producing a transient buffer that mul_mat then consumes.
    ggml_tensor * Xin = X;
    if (chain) {
        Zero = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        ggml_set_name(Zero, "Zero");
        Xin = ggml_add(ctx, X, Zero);          // transient buffer
        Xin = ggml_reshape_2d(ctx, Xin, K, N); // view of transient
    }
    ggml_tensor * Y = ggml_mul_mat(ctx, W, Xin);
    if (chain) {
        Y = ggml_scale(ctx, Y, 1.0f); // downstream op consuming Y
    }
    // KOKOPOP_REPRO_REPEAT=N: chain N additional mul_mats reading from a
    // transient compute buffer (mimics Kokoro's series of conv1d in AdaIN
    // ResBlks where scheduler aggressively reuses compute memory).
    if (const char * env_n = std::getenv("KOKOPOP_REPRO_REPEAT")) {
        const int repeat = std::atoi(env_n);
        // Need a path back from Y[OC,N] to something[K,N] for the next matmul.
        // Use ggml_add against X to keep things simple, then project up via
        // another weight. To avoid needing more weights, just reuse W.
        for (int r = 0; r < repeat; ++r) {
            // Pad Y[OC,N] back to [K,N] by repeating along OC dim. Cheapest:
            // ggml_repeat into a [K,N] template.
            ggml_tensor * tmpl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            ggml_tensor * Yk = ggml_repeat(ctx, Y, tmpl);
            Yk = ggml_reshape_2d(ctx, Yk, K, N);
            Y = ggml_mul_mat(ctx, W, Yk);
        }
    }

    ggml_set_name(W, "W");
    ggml_set_name(X, "X");
    ggml_set_name(Y, "Y");

    ggml_backend_buffer_t weights_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, weights_buft);
    if (!weights_buf) { ggml_free(ctx); return false; }

    ggml_backend_tensor_set(W, q8_0_weight_bytes.data(), 0, q8_0_weight_bytes.size());
    ggml_backend_tensor_set(X, input_f32.data(), 0, input_f32.size() * sizeof(float));
    if (Zero != nullptr) {
        std::vector<float> zeros(K * N, 0.0f);
        ggml_backend_tensor_set(Zero, zeros.data(), 0, zeros.size() * sizeof(float));
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(weights_buf);
        ggml_free(ctx);
        return false;
    }

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed: %d\n", (int)st);
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(weights_buf);
        ggml_free(ctx);
        return false;
    }

    out.resize(OC * N);
    ggml_backend_tensor_get(Y, out.data(), 0, out.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(weights_buf);
    ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    int64_t K  = (argc > 1) ? std::atoll(argv[1]) : 1408;
    int64_t OC = (argc > 2) ? std::atoll(argv[2]) : 128;
    int64_t N  = (argc > 3) ? std::atoll(argv[3]) : 7921;

    if (K % 32 != 0) {
        std::fprintf(stderr, "K must be a multiple of 32 for Q8_0 (got %lld)\n", (long long)K);
        return 1;
    }

    std::printf("shape: W[Q8_0, K=%lld, OC=%lld] x X[F32, K=%lld, N=%lld] = Y[F32, OC=%lld, N=%lld]\n",
                (long long)K, (long long)OC, (long long)K, (long long)N, (long long)OC, (long long)N);

    // Generate F32 weight, quantize to Q8_0.
    std::vector<float> w_f32(K * OC);
    fill_random_f32(w_f32, 0xC0FFEEu);

    const size_t qsize = ggml_row_size(GGML_TYPE_Q8_0, K) * OC;
    std::vector<uint8_t> w_q8_0(qsize);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w_f32.data(), w_q8_0.data(),
                        /*start=*/0, /*nrows=*/OC, /*n_per_row=*/K, /*imatrix=*/nullptr);

    std::vector<float> x_f32(K * N);
    fill_random_f32(x_f32, 0xDEADBEEFu);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { std::fprintf(stderr, "cpu init failed\n"); return 1; }

    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    std::vector<float> y_cpu, y_cuda;

    if (!run_one(cpu,  ggml_backend_get_default_buffer_type(cpu),  w_q8_0, K, OC, N, x_f32, y_cpu)) {
        std::fprintf(stderr, "cpu run failed\n"); return 1;
    }
    if (!run_one(cuda, ggml_backend_cuda_buffer_type(0),            w_q8_0, K, OC, N, x_f32, y_cuda)) {
        std::fprintf(stderr, "cuda run failed\n"); return 1;
    }

    size_t worst_idx = 0;
    const float maxabs = max_abs_diff(y_cpu, y_cuda, worst_idx);
    const double rel   = rel_l2(y_cpu, y_cuda);

    std::printf("y_cpu[0..3]   = %.6f %.6f %.6f %.6f\n",
                y_cpu[0], y_cpu[1], y_cpu[2], y_cpu[3]);
    std::printf("y_cuda[0..3]  = %.6f %.6f %.6f %.6f\n",
                y_cuda[0], y_cuda[1], y_cuda[2], y_cuda[3]);
    std::printf("worst abs diff = %.6f at idx %zu (cpu=%.6f cuda=%.6f)\n",
                maxabs, worst_idx, y_cpu[worst_idx], y_cuda[worst_idx]);
    std::printf("relative L2    = %.6e\n", rel);

    // Q8_0 quant noise gives rel L2 around 1e-3 typically; >1e-1 means broken.
    const bool broken = (rel > 5e-2);
    std::printf("verdict: %s\n", broken ? "BROKEN" : "OK (within Q8_0 noise)");

    ggml_backend_free(cpu);
    ggml_backend_free(cuda);
    return broken ? 2 : 0;
}
