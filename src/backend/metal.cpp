#include "metal.h"

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
//   * Scheduler `op_offload=false`: with CPU weights, op_offload=true would
//     place all CPU-sourced ops on CPU, defeating Metal for the frontend.
//   * **NO `ggml_backend_sched_reserve`** — the reserve-then-alloc combo
//     triggers two consecutive `split_graph` calls; tensor copies created
//     in the 1st are freed by the `ggml_free(sched->ctx)` of the 2nd, and
//     the pre-reservation made in the 1st no longer matches the tensors
//     actually allocated in the 2nd. Observed symptom: silent audio despite
//     correct duration. The allocator does its own reserve on the 1st alloc
//     (ggml-backend.cpp:1509-1535), which is sufficient here (~40 MiB total).
//   * **Full generation+generator on CPU by default** — generation: Metal NORM
//     kernel produces corrupted values on certain token patterns and hybrid
//     Metal/CPU transitions cause ~50 s wall-clock (25 GPU syncs). Generator:
//     hybrid Metal/CPU (CONV_TRANSPOSE_1D + rest Metal) corrupts activations.
//     Both pinned CPU by default. Override via `KOKOPOP_METAL_FORCE_CPU=none`.
//   * **Adaptive Metal frontend** (2026-05-04) — Metal kernel-launch overhead
//     (~100 us/kernel * N ALBERT kernels) exceeds GPU compute speedup for
//     short inputs. Frontend runs on CPU when n_tokens < KOKOPOP_METAL_MIN_TOKENS
//     (default 100). Set to 0 to always use Metal for frontend.
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
    size_t sched_capacity_ = 0;
    const char * active_label_ = nullptr;
    int n_input_tokens_ = 0;
    std::vector<PendingInit> pending_inits_;

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
        if (sched_ != nullptr && sched_capacity_ >= required) {
            return true;
        }
        if (sched_ != nullptr) {
            ggml_backend_sched_free(sched_);
            sched_ = nullptr;
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
        // Default: `generation` AND `generator` on CPU. Reasons:
        //   1. `generator`: in hybrid mode (CONV_TRANSPOSE_1D pinned +
        //      rest Metal), activation Metal↔CPU transitions corrupt
        //      values (audio "mush" on Bonjour).
        //   2. `generation`: hybrid Metal with pinned NORM explodes wall-clock
        //      (50+ s vs 2 s pure CPU, ~25 GPU sync splits). Additionally
        //      the Metal NORM kernel produces incorrect values for some
        //      token patterns (unintelligible audio on comma phrases).
        //      Pinning the full `generation` removes both problems at once.
        //   3. `frontend` short inputs: Metal kernel-launch overhead (~100 µs
        //      per kernel × N kernels ALBERT) exceeds GPU compute speedup for
        //      small token counts. Default threshold = 20 tokens, override via
        //      KOKOPOP_METAL_MIN_TOKENS=N (0 = always Metal, large = always CPU).
        // What stays on Metal: `frontend` when n_tokens >= KOKOPOP_METAL_MIN_TOKENS.
        // Override: KOKOPOP_METAL_FORCE_CPU=none|<labels>.
        const char * forced = std::getenv("KOKOPOP_METAL_FORCE_CPU");
        if (forced == nullptr) {
            if (std::strcmp(active_label_, "generator") == 0
                || std::strcmp(active_label_, "generation") == 0) {
                return true;
            }
            if (std::strcmp(active_label_, "frontend") == 0 && n_input_tokens_ > 0) {
                const char * s = std::getenv("KOKOPOP_METAL_MIN_TOKENS");
                const int min_tok = (s && s[0]) ? std::atoi(s) : 100;
                return n_input_tokens_ < min_tok;
            }
            return false;
        }

        if (std::strcmp(forced, "none") == 0) {
            return false;
        }

        return list_contains(forced, active_label_);
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
    bool is_default_unsafe_op(ggml_tensor * node) const {
        if (node == nullptr) return false;
        switch (node->op) {
            case GGML_OP_CONV_TRANSPOSE_1D:
                // Wedges the GPU (watchdog) on large dims (e.g. [430, 256]
                // in the generator). At shape [43, 512] in `generation` it
                // passes, but pinning everywhere avoids Metal↔CPU transitions
                // that corrupt activations on certain sequences.
                return active_label_ != nullptr
                    && std::strcmp(active_label_, "generator") == 0;
            case GGML_OP_NORM:
                // Metal LayerNorm produces corrupted values on certain input
                // patterns in `generation` (e.g. phrases with commas →
                // unintelligible audio). Bisection: PIN_OP=NORM fixes it;
                // GGML_METAL_FUSION_DISABLE alone is not enough (bug also in
                // the non-fused kernel). We pin in `generation` (and for
                // consistency, everywhere except `frontend` where ALBERT 12
                // layers use LayerNorm heavily — bug not observed there, and
                // pinning would create ~50 Metal↔CPU transitions = terrible perf).
                return active_label_ != nullptr
                    && std::strcmp(active_label_, "frontend") != 0;
            default:
                return false;
        }
    }

    void pin_graph_cpu_fallbacks(ggml_cgraph * graph) {
        // hard_pin       : Metal cannot execute the op (integer, or
        //                  ggml_backend_supports_op=false). No override.
        // default_unsafe : op that wedges in practice (CONV_TRANSPOSE_1D).
        //                  Override via ALLOW_OP — at the risk of wedging.
        // soft_pin       : user-requested pin via FORCE_CPU/PIN_OP.
        //                  Override via ALLOW_OP.
        const bool force_cpu = force_cpu_for_active_graph();
        const char * pin_op  = std::getenv("KOKOPOP_METAL_PIN_OP");
        const char * allow_op = std::getenv("KOKOPOP_METAL_ALLOW_OP");
        const int n_nodes = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == nullptr) {
                continue;
            }
            const bool hard_pin       = is_integer_type(node->type)
                                     || !ggml_backend_supports_op(metal_backend_, node);
            const bool default_unsafe = is_default_unsafe_op(node);
            const bool soft_pin       = force_cpu || node_op_in_list(node, pin_op);
            const bool allowed        = node_op_in_list(node, allow_op);
            if (hard_pin || ((default_unsafe || soft_pin) && !allowed)) {
                ggml_backend_sched_set_tensor_backend(sched_, node, cpu_backend_);
            }
        }
    }

    void log_sched_sizes(const char * stage) {
        if (sched_ == nullptr || std::getenv("KOKOPOP_METAL_LOG_SCHED") == nullptr) {
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
        }
    }

    ~MetalBackend() override {
        if (sched_) {
            ggml_backend_sched_free(sched_);
            sched_ = nullptr;
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
        if (std::getenv("KOKOPOP_METAL_OP_TRACE") != nullptr) {
            ggml_backend_sched_set_eval_callback(sched_, eval_trace_cb, sched_);
        }
        pin_graph_cpu_fallbacks(graph);
        // Reserve AFTER pin_graph_cpu_fallbacks so that the split_graph
        // inside reserve sees the same node→backend assignments as the
        // split_graph inside alloc_graph below.  Calling reserve before
        // pinning creates stale copy tensors that get invalidated by the
        // 2nd split_graph (silent audio symptom).
        ggml_backend_sched_reserve(sched_, graph);
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
        if (std::getenv("KOKOPOP_METAL_ALLOC_ONLY") != nullptr) {
            ggml_backend_sched_synchronize(sched_);
            log_sched_sizes("alloc-only");
            (void)graph;
            return GGML_STATUS_SUCCESS;
        }
        // Selective bypass: KOKOPOP_METAL_RUN_ONLY={frontend,generation,generator}
        // runs only graphs at level ≤ the requested one. Allows bisection of
        // which graph wedges the GPU without rerunning the full inference.
        if (const char * run_only = std::getenv("KOKOPOP_METAL_RUN_ONLY")) {
            if (label_level(active_label_) > label_level(run_only)) {
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
        active_label_ = label;
    }

    void set_input_tokens(int n) override {
        n_input_tokens_ = n;
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
