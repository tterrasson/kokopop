#include "metal.h"
#include "metal_lstm.h"
#include "metal_stft.h"
#include "core/constants.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-metal.h>

namespace kokopop {

namespace {

static bool is_integer_type(ggml_type type) {
    return type == GGML_TYPE_I8 ||
           type == GGML_TYPE_I16 ||
           type == GGML_TYPE_I32 ||
           type == GGML_TYPE_I64;
}

static size_t graph_capacity(ggml_cgraph * graph) {
    const size_t need = static_cast<size_t>(std::max(1, ggml_graph_size(graph)));
    return std::max<size_t>(GGML_DEFAULT_GRAPH_SIZE, need + 4096);
}

// ---- MetalBackend ---------------------------------------------------------
//
// Placement strategy (bisect 2026-05-02, revised 2026-05-04):
//   * Weights allocated on CPU (weight_buffer_type → cpu_backend_).
//     - CPU-pinned graphs (generation/generator): weights+ops on CPU → 1 split,
//       0 cross-backend copies, zero Metal overhead per inference.
//     - Frontend (Metal ops): scheduler creates CPU→Metal galloc copies before
//       the Metal split (~2 splits). On M1 unified memory (MTLStorageModeShared)
//       these are direct memcpy, not GPU blits.
//     - Eliminates the old `pin_weights_cpu_for_cpu_graph` O(n_nodes) loop and
//       the per-inference Metal→CPU sync that added ~120 ms on generation.
//     - Note: Metal shared/mapped buffer types report is_host=false; CPU backend
//       therefore rejects them and would require copies for CPU graphs.  CPU
//       weight placement is optimal given generation/generator dominates calls.
//   * Scheduler `op_offload=false`: ops naturally follow their weight tensors
//     unless the scheduler has a reason to split.  pin_graph_cpu_fallbacks only
//     forces known CPU fallbacks; it deliberately leaves all other nodes
//     unassigned.  Explicitly pinning every supported frontend op to Metal
//     changes ALBERT numerics enough to skew duration prediction on long inputs.
//   * **NO `ggml_backend_sched_reserve`** — the reserve-then-alloc combo
//     triggers two consecutive `split_graph` calls; tensor copies created
//     in the 1st are freed by the `ggml_free(sched->ctx)` of the 2nd, and
//     the pre-reservation made in the 1st no longer matches the tensors
//     actually allocated in the 2nd. Observed symptom: silent audio despite
//     correct duration. The allocator does its own reserve on the 1st alloc
//     (ggml-backend.cpp:1509-1535), which is sufficient here (~40 MiB total).
//   * **Full generation+generator on CPU by default** — generation: Metal NORM
//     kernel (float4 path) produces corrupted values on certain token patterns
//     (fixed in patches/ggml-metal-norm-scalar.patch) and hybrid Metal/CPU
//     transitions cause ~50 s wall-clock (25 GPU syncs). Generator: ggml Metal
//     CONV_TRANSPOSE_1D dispatches 1 thread/output — GPU watchdog kills it for
//     large dims (fixed in patches/ggml-conv-transpose-1d-simd.patch).
//     Both pinned CPU by default. Override via `KOKOPOP_METAL_FORCE_CPU=none`.
//   * **Adaptive Metal frontend** (2026-05-04) — Metal kernel-launch overhead
//     (~100 us/kernel * N ALBERT kernels) exceeds GPU compute speedup for
//     short inputs. Frontend runs on CPU when n_tokens < KOKOPOP_METAL_MIN_TOKENS
//     (default 100). Set to 0 to always use Metal for frontend.
//   * **Env var caching** — all KOKOPOP_METAL_* env vars are read once at
//     construction and cached as members (env vars do not change at runtime).
//   * **Pin decision cache** — pin assignments for a graph are reused across
//     calls when active_label_ and n_nodes are unchanged, avoiding an O(n_nodes)
//     loop on every inference step.
//
// Environment variables (escape hatches, all optional):
//   KOKOPOP_METAL_LOG_SCHED=1     — log per graph `splits / metal_buf / cpu_buf`.
//   KOKOPOP_METAL_OP_TRACE=1      — log each op at execution time
//                                (with backend, name, shape); useful for
//                                pinpointing where the GPU wedges.
//   KOKOPOP_METAL_ALLOC_ONLY=1    — full compute bypass (memory measurement).
//   KOKOPOP_METAL_RUN_ONLY=<lvl>  — run up to level frontend|generation|
//                                generator inclusive, skip the following.
//   KOKOPOP_METAL_MIN_TOKENS=N    — minimum token count to use Metal for `frontend`
//                                (default 100). Below N, frontend runs on CPU.
//                                Set to 0 to always use Metal for frontend.
//   KOKOPOP_METAL_FORCE_CPU=<lbl> — pin all nodes of named graphs on
//                                CPU (e.g. "generator" or "frontend,generator").
//   KOKOPOP_METAL_PIN_OP=<ops>    — pin by op type everywhere (e.g. "SIN,REPEAT").
//                                Match `ggml_op_name` or `ggml_unary_op_name`.
//   KOKOPOP_METAL_ALLOW_OP=<ops>  — re-allow on Metal ops listed even if
//                                FORCE_CPU/PIN_OP/default_unsafe pin them.
class MetalBackend : public Backend {
    ggml_backend_t metal_backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    MetalLstmKernelState * lstm_kernel_ = nullptr;
    MetalStftState       * stft_kernel_ = nullptr;
    size_t sched_capacity_ = 0;
    const char * active_label_ = nullptr;
    int n_input_tokens_ = 0;
    std::vector<PendingInit> pending_inits_;

    // Cached env vars — read once at construction, never change at runtime.
    const char * env_force_cpu_  = nullptr;  // KOKOPOP_METAL_FORCE_CPU
    const char * env_pin_op_     = nullptr;  // KOKOPOP_METAL_PIN_OP
    const char * env_allow_op_   = nullptr;  // KOKOPOP_METAL_ALLOW_OP
    const char * env_run_only_   = nullptr;  // KOKOPOP_METAL_RUN_ONLY
    int          env_min_tokens_ = 100;      // KOKOPOP_METAL_MIN_TOKENS
    bool         env_log_sched_  = false;    // KOKOPOP_METAL_LOG_SCHED
    bool         env_op_trace_   = false;    // KOKOPOP_METAL_OP_TRACE
    bool         env_alloc_only_ = false;    // KOKOPOP_METAL_ALLOC_ONLY

    // Pin decision cache — valid when active_label_ == pin_cache_label_
    // and the graph has pin_cache_n_nodes_ nodes.
    std::string              pin_cache_label_;
    int                      pin_cache_n_nodes_ = -1;
    std::vector<ggml_backend_t> pin_cache_;

    // Levels for KOKOPOP_METAL_RUN_ONLY: we compute a graph iff its level
    // is ≤ the requested one. Unknown → infinite level → skip.
    static int label_level(const char * s) {
        if (s == nullptr) return 99;
        if (std::strcmp(s, "frontend")   == 0) return 0;
        if (std::strcmp(s, "generation") == 0) return 1;
        if (std::strcmp(s, "generator")  == 0) return 2;
        return 99;
    }

    bool ensure_scheduler(ggml_cgraph * graph) {
        const size_t required = graph_capacity(graph);
        // Kokoro builds different tensor shapes for each chunk. Recreate the
        // scheduler so ggml's graph allocator never reuses stale shape state.
        if (sched_ != nullptr) {
            ggml_backend_sched_free(sched_);
            sched_ = nullptr;
            sched_capacity_ = 0;
        }
        ggml_backend_t backends[] = { metal_backend_, cpu_backend_ };
        sched_ = ggml_backend_sched_new(backends, nullptr, 2, required, false, false);
        sched_capacity_ = sched_ == nullptr ? 0 : required;
        return sched_ != nullptr;
    }

    // Exact match of an element in a case-sensitive comma-separated list.
    // Ex: list_contains("a,bc", "bc") = true.
    static bool list_contains(const char * csv, const char * needle) {
        if (csv == nullptr || needle == nullptr) return false;
        const size_t want_len = std::strlen(needle);
        const char * cursor = csv;
        while (*cursor) {
            const char * comma = std::strchr(cursor, ',');
            const size_t span = comma ? static_cast<size_t>(comma - cursor) : std::strlen(cursor);
            if (span == want_len && std::strncmp(cursor, needle, span) == 0) {
                return true;
            }
            if (!comma) break;
            cursor = comma + 1;
        }
        return false;
    }

    bool force_cpu_for_active_graph() const {
        if (active_label_ == nullptr) return false;
        // Default: `generation` AND `generator` on CPU (conservative).
        //   The two original blocking bugs are now patched:
        //   - patches/ggml-metal-norm-scalar.patch: disables the float4 NORM
        //     kernel that corrupted values for certain token patterns in `generation`.
        //   - patches/ggml-conv-transpose-1d-simd.patch: CONV_TRANSPOSE_1D now
        //     uses 32 threads/group + simd_sum, avoiding GPU watchdog on the large
        //     generator dims.
        //   CPU default is kept until full-Metal generation/generator paths are
        //   validated end-to-end.  Override: KOKOPOP_METAL_FORCE_CPU=none.
        //   `frontend` short inputs: Metal kernel-launch overhead (~100 µs per
        //   kernel × N ALBERT kernels) exceeds GPU compute speedup for small token
        //   counts. Threshold = env_min_tokens_ (default 100).
        if (env_force_cpu_ == nullptr) {
            if (std::strcmp(active_label_, "generator") == 0
                || std::strcmp(active_label_, "generation") == 0) {
                return true;
            }
            if (std::strcmp(active_label_, "frontend") == 0 && n_input_tokens_ > 0) {
                return n_input_tokens_ < env_min_tokens_;
            }
            return false;
        }

        if (std::strcmp(env_force_cpu_, "none") == 0) {
            return false;
        }

        return list_contains(env_force_cpu_, active_label_);
    }

    static bool node_op_in_list(ggml_tensor * node, const char * csv) {
        if (csv == nullptr || node == nullptr) return false;
        if (list_contains(csv, ggml_op_name(node->op))) return true;
        if (node->op == GGML_OP_UNARY) {
            return list_contains(csv, ggml_unary_op_name(ggml_get_unary_op(node)));
        }
        return false;
    }

    // Ops Metal claims to support but which produce bad values or
    // wedge the GPU. All identified by bisection (PIN_OP / OP_TRACE) on 2026-05-02.
    // NORM and CONV_TRANSPOSE_1D were previously unsafe; both are fixed by
    // patches/ggml-metal-norm-scalar.patch and patches/ggml-conv-transpose-1d-simd.patch.
    bool is_default_unsafe_op(ggml_tensor * node) const {
        (void)node;
        return false;
    }

    void pin_graph_cpu_fallbacks(ggml_cgraph * graph) {
        // hard_pin       : Metal cannot execute the op (integer type, or
        //                  ggml_backend_supports_op=false). No override.
        // default_unsafe : (currently none) reserved for ops with verified GPU bugs.
        //                  NORM and CONV_TRANSPOSE_1D were here; both patched out.
        //                  Override via ALLOW_OP — at user's own risk.
        // soft_pin       : user-requested pin via FORCE_CPU or PIN_OP env var.
        //                  Override via ALLOW_OP.
        //
        // Leave all other nodes unassigned so ggml's scheduler keeps the same
        // placement decisions it used before the pin-decision cache.  Forcing
        // every supported frontend op to Metal changes ALBERT numerics enough
        // to skew duration prediction on long inputs.
        //
        // Pin cache: when active_label_ and n_nodes are unchanged the per-node
        // CPU pins are deterministic, so we replay the cached vector instead
        // of re-running the O(n_nodes) classification loop.
        const int n_nodes = ggml_graph_n_nodes(graph);
        const bool cache_hit = (pin_cache_n_nodes_ == n_nodes)
                            && (active_label_ != nullptr)
                            && (pin_cache_label_ == active_label_);

        if (cache_hit) {
            for (int i = 0; i < n_nodes; ++i) {
                ggml_tensor * node = ggml_graph_node(graph, i);
                if (node != nullptr && pin_cache_[i] != nullptr) {
                    ggml_backend_sched_set_tensor_backend(sched_, node, pin_cache_[i]);
                }
            }
            return;
        }

        const bool force_cpu = force_cpu_for_active_graph();
        pin_cache_.resize(n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == nullptr) {
                pin_cache_[i] = nullptr;
                continue;
            }
            const bool hard_pin       = is_integer_type(node->type)
                                     || !ggml_backend_supports_op(metal_backend_, node);
            const bool default_unsafe = is_default_unsafe_op(node);
            const bool soft_pin       = force_cpu || node_op_in_list(node, env_pin_op_);
            const bool allowed        = node_op_in_list(node, env_allow_op_);
            ggml_backend_t backend;
            if (hard_pin || ((default_unsafe || soft_pin) && !allowed)) {
                backend = cpu_backend_;
            } else {
                backend = nullptr;
            }
            pin_cache_[i] = backend;
            if (backend != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched_, node, backend);
            }
        }
        pin_cache_label_   = active_label_ ? active_label_ : "";
        pin_cache_n_nodes_ = n_nodes;
    }

    void log_sched_sizes(const char * stage) {
        if (sched_ == nullptr || !env_log_sched_) {
            return;
        }
        const int n_splits = ggml_backend_sched_get_n_splits(sched_);
        const size_t metal_bytes = ggml_backend_sched_get_buffer_size(sched_, metal_backend_);
        const size_t cpu_bytes   = ggml_backend_sched_get_buffer_size(sched_, cpu_backend_);
        std::fprintf(stderr,
            "[metal-sched %s/%s] splits=%d metal_buf=%.1f MiB cpu_buf=%.1f MiB cap=%zu\n",
            active_label_ ? active_label_ : "?", stage, n_splits,
            metal_bytes / 1048576.0, cpu_bytes / 1048576.0, sched_capacity_);
    }

    static bool eval_trace_cb(ggml_tensor * t, bool ask, void * user_data) {
        // ask=true  : scheduler asks if we want the callback (yes).
        // ask=false : just before tensor execution — we log + flush to
        //             capture the current op even if the GPU freezes right after.
        auto * sched = static_cast<ggml_backend_sched_t>(user_data);
        if (ask) return true;
        const char * label = (t && t->op == GGML_OP_UNARY)
            ? ggml_unary_op_name(ggml_get_unary_op(t))
            : (t ? ggml_op_name(t->op) : "?");
        ggml_backend_t b = (sched && t)
            ? ggml_backend_sched_get_tensor_backend(sched, t)
            : nullptr;
        const char * be = b ? ggml_backend_name(b) : "?";
        std::fprintf(stderr, "[op %-5s] %-20s %-32s shape=[%lld,%lld,%lld,%lld]\n",
            be, label, t && t->name[0] ? t->name : "",
            (long long)(t ? t->ne[0] : 0), (long long)(t ? t->ne[1] : 0),
            (long long)(t ? t->ne[2] : 0), (long long)(t ? t->ne[3] : 0));
        std::fflush(stderr);
        return true;
    }

public:
    explicit MetalBackend(int32_t n_threads) {
        // Cache all KOKOPOP_METAL_* env vars once — they do not change at runtime.
        env_force_cpu_  = std::getenv("KOKOPOP_METAL_FORCE_CPU");
        env_pin_op_     = std::getenv("KOKOPOP_METAL_PIN_OP");
        env_allow_op_   = std::getenv("KOKOPOP_METAL_ALLOW_OP");
        env_run_only_   = std::getenv("KOKOPOP_METAL_RUN_ONLY");
        env_log_sched_  = std::getenv("KOKOPOP_METAL_LOG_SCHED")  != nullptr;
        env_op_trace_   = std::getenv("KOKOPOP_METAL_OP_TRACE")   != nullptr;
        env_alloc_only_ = std::getenv("KOKOPOP_METAL_ALLOC_ONLY") != nullptr;
        if (const char * s = std::getenv("KOKOPOP_METAL_MIN_TOKENS")) {
            if (s[0]) env_min_tokens_ = std::atoi(s);
        }

        cpu_backend_ = ggml_backend_cpu_init();
        if (cpu_backend_ != nullptr) {
            ggml_backend_cpu_set_n_threads(cpu_backend_, std::max<int32_t>(1, n_threads));
        }
        // Disable Metal op fusion by default — the fused kernel
        // `kernel_norm_mul_add` produces corrupted values for certain
        // token patterns (e.g. comma in "J'aime la galette, savez-vous
        // comment ?" → unintelligible audio). Bisection 2026-05-02:
        // PIN_OP=NORM fixes it, isolating the bug to the LayerNorm op
        // executed by the fused kernel. ggml-metal reads the variable
        // once at init via getenv (see ggml-metal-context.m). We set it
        // if not already defined; users can force fusion with
        // KOKOPOP_METAL_ENABLE_FUSION=1 (at their own risk).
        if (std::getenv("KOKOPOP_METAL_ENABLE_FUSION") == nullptr) {
            setenv("GGML_METAL_FUSION_DISABLE", "1", 0);
        }
        metal_backend_ = ggml_backend_metal_init();
        if (metal_backend_ == nullptr) {
            if (cpu_backend_) {
                ggml_backend_free(cpu_backend_);
                cpu_backend_ = nullptr;
            }
        } else {
            lstm_kernel_ = metal_lstm_create();
            stft_kernel_ = metal_stft_create(KOKOPOP_STFT_N, KOKOPOP_STFT_HOP);
        }
    }

    ~MetalBackend() override {
        if (sched_) {
            ggml_backend_sched_free(sched_);
            sched_ = nullptr;
        }
        if (lstm_kernel_) {
            metal_lstm_destroy(lstm_kernel_);
            lstm_kernel_ = nullptr;
        }
        if (stft_kernel_) {
            metal_stft_destroy(stft_kernel_);
            stft_kernel_ = nullptr;
        }
        if (metal_backend_) {
            ggml_backend_free(metal_backend_);
            metal_backend_ = nullptr;
        }
        if (cpu_backend_) {
            ggml_backend_free(cpu_backend_);
            cpu_backend_ = nullptr;
        }
    }

    bool valid() const {
        return metal_backend_ != nullptr && cpu_backend_ != nullptr;
    }

    void sched_reset() override {
        if (sched_) {
            ggml_backend_sched_reset(sched_);
        }
    }

    bool sched_alloc_graph(ggml_cgraph * graph) override {
        if (!ensure_scheduler(graph)) {
            return false;
        }
        if (env_op_trace_) {
            ggml_backend_sched_set_eval_callback(sched_, eval_trace_cb, sched_);
        }
        pin_graph_cpu_fallbacks(graph);
        // Do NOT call ggml_backend_sched_reserve here: reserve() calls
        // ggml_backend_sched_reset() at the end, which clears all backend IDs
        // back to -1 — erasing the Metal/CPU assignments just made above.
        // alloc_graph → alloc_splits handles reservation internally when it
        // detects backend_ids_changed=1 (first Metal run), using the IDs that
        // split_graph preserved from pin_graph_cpu_fallbacks.
        const bool ok = ggml_backend_sched_alloc_graph(sched_, graph);
        if (ok) {
            log_sched_sizes("alloc");
        }
        return ok;
    }

    void clear_pending_inits() override {
        pending_inits_.clear();
    }

    void queue_tensor_data(ggml_tensor * tensor, const void * data, size_t size) override {
        if (tensor == nullptr || data == nullptr || size == 0) {
            return;
        }
        PendingInit init;
        init.tensor = tensor;
        init.bytes.resize(size);
        std::memcpy(init.bytes.data(), data, size);
        pending_inits_.push_back(std::move(init));
    }

    void queue_zero_tensor(ggml_tensor * tensor) override {
        if (tensor != nullptr) {
            PendingInit init;
            init.tensor = tensor;
            init.zero = true;
            pending_inits_.push_back(std::move(init));
        }
    }

    void queue_f32_tensor(ggml_tensor * tensor, float value) override {
        if (tensor == nullptr) {
            return;
        }
        std::vector<float> data(static_cast<size_t>(ggml_nelements(tensor)), value);
        queue_tensor_data(tensor, data.data(), data.size() * sizeof(float));
    }

    bool apply_pending_inits() override {
        for (const PendingInit & init : pending_inits_) {
            if (init.tensor == nullptr) {
                continue;
            }
            if (init.zero) {
                ggml_backend_tensor_memset(init.tensor, 0, 0, ggml_nbytes(init.tensor));
            } else if (!init.bytes.empty()) {
                ggml_backend_tensor_set(init.tensor, init.bytes.data(), 0, init.bytes.size());
            }
        }
        pending_inits_.clear();
        return true;
    }

    ggml_status compute(ggml_context * ctx, ggml_cgraph * graph) override {
        (void)ctx;
        // Global compute bypass: KOKOPOP_METAL_ALLOC_ONLY=1 skips all graphs.
        if (env_alloc_only_) {
            ggml_backend_sched_synchronize(sched_);
            log_sched_sizes("alloc-only");
            (void)graph;
            return GGML_STATUS_SUCCESS;
        }
        // Selective bypass: KOKOPOP_METAL_RUN_ONLY={frontend,generation,generator}
        // runs only graphs at level ≤ the requested one. Allows bisection of
        // which graph wedges the GPU without rerunning the full inference.
        if (env_run_only_ != nullptr) {
            if (label_level(active_label_) > label_level(env_run_only_)) {
                ggml_backend_sched_synchronize(sched_);
                log_sched_sizes("skip-compute");
                (void)graph;
                return GGML_STATUS_SUCCESS;
            }
        }
        log_sched_sizes("compute-begin");
        return ggml_backend_sched_graph_compute(sched_, graph);
    }

    void tensor_set(ggml_tensor * t, const void * data,
                    size_t offset, size_t size) override {
        if (t != nullptr && data != nullptr && size > 0) {
            GGML_ASSERT((t->buffer != nullptr || (t->view_src != nullptr && t->view_src->buffer != nullptr)) &&
                        "tensor buffer not set before backend tensor_set");
            ggml_backend_tensor_set(t, data, offset, size);
        }
    }

    void tensor_get(ggml_tensor * t, void * data,
                    size_t offset, size_t size) override {
        if (t != nullptr && data != nullptr && size > 0) {
            GGML_ASSERT((t->buffer != nullptr || (t->view_src != nullptr && t->view_src->buffer != nullptr)) &&
                        "tensor buffer not set before backend tensor_get");
            ggml_backend_tensor_get(t, data, offset, size);
        }
    }

    const char * label() const override { return "Metal (GPU)"; }

    void set_active_label(const char * label) override {
        if (active_label_ != label) {
            active_label_ = label;
            // Invalidate pin cache — decisions depend on active_label_.
            pin_cache_n_nodes_ = -1;
        }
    }

    void set_input_tokens(int n) override {
        n_input_tokens_ = n;
    }

    void preload_lstm_whh(const std::string & key,
                          const float * w_hh_f32, int H, int four_H) override {
        if (lstm_kernel_) {
            metal_lstm_preload_whh(lstm_kernel_, key.c_str(), w_hh_f32, H, four_H);
        }
    }

    void * metal_lstm_kernel() const override {
        return lstm_kernel_;
    }

    void * metal_stft_kernel() const override {
        return stft_kernel_;
    }

    ggml_backend_buffer_type_t weight_buffer_type() const override {
        return ggml_backend_get_default_buffer_type(cpu_backend_);
    }

    size_t generation_context_bytes(int64_t total_frames, int64_t n_tokens) const override {
        const size_t frames = static_cast<size_t>(std::max<int64_t>(1, total_frames));
        const size_t tokens = static_cast<size_t>(std::max<int64_t>(1, n_tokens));
        const size_t mask_bytes = frames * tokens * sizeof(float);
        const size_t pred_bytes = 512ULL * tokens * sizeof(float);
        const size_t lstm_work  = frames * 4096ULL * sizeof(float);
        const size_t overhead   = backend_mib(64);
        return mask_bytes + pred_bytes + lstm_work + overhead;
    }

    size_t generator_context_bytes(int64_t decoder_len) const override {
        const size_t dec        = static_cast<size_t>(std::max<int64_t>(1, decoder_len));
        const size_t out_frames = dec * 60;
        const size_t decoder_tensor  = 64ULL * dec * sizeof(float);
        const size_t harmonic_tensor = 22ULL * (out_frames + 1) * sizeof(float);
        const size_t post_tensor     = 1ULL * out_frames * sizeof(float);
        const size_t conv_work = 256ULL * out_frames * sizeof(float) * 3;
        const size_t overhead  = backend_mib(16);
        return decoder_tensor + harmonic_tensor + post_tensor + conv_work + overhead;
    }

    size_t frontend_context_bytes() const override {
        return backend_mib(64);
    }
};

} // anonymous namespace

std::unique_ptr<Backend> create_metal_backend(int32_t n_threads) {
    auto backend = std::make_unique<MetalBackend>(n_threads);
    if (!backend->valid()) {
        return nullptr;
    }
    return backend;
}

} // namespace kokopop
