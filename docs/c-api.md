# Kokopop C API

The public C API lives in `include/kokopop.h`. All functions return `KOKOPOP_OK`
on success; on failure, call `kokopop_last_error()` for a human-readable error.

## One-shot synthesis

```c
#include "kokopop.h"

kokopop_model *model = NULL;
kokopop_audio audio = {0};

kokopop_model_load("models/kokoro.gguf", NULL, &model);
kokopop_synthesize_text(model, "Hello, world!", "af_heart", 1.0f, &audio);
kokopop_write_wav("output.wav", &audio);

kokopop_audio_free(&audio);
kokopop_model_free(model);
```

Use `kokopop_synthesize_phonemes()` when your input is already phonemized.
`kokopop_audio` owns its `samples` pointer and must be released with
`kokopop_audio_free()`.

## Backend selection

`kokopop_model_options.backend` picks the inference backend. `KOKOPOP_BACKEND_AUTO`
(the default) tries CUDA, Metal, Vulkan and OpenCL in that order and falls back to
the CPU; every other value is a hard request that fails if the backend is not
compiled in or not available at runtime.

| Value | Meaning |
|---|---|
| `KOKOPOP_BACKEND_AUTO` | Resolve at load time (default) |
| `KOKOPOP_BACKEND_CPU` | CPU |
| `KOKOPOP_BACKEND_METAL` | Metal (macOS) |
| `KOKOPOP_BACKEND_CUDA` | CUDA (NVIDIA) |
| `KOKOPOP_BACKEND_VULKAN` | Vulkan |
| `KOKOPOP_BACKEND_OPENCL` | OpenCL (Adreno / Android) |

`kokopop_model_backend()` reports which one a loaded model actually got, so a
caller that passed `AUTO` can tell whether it ended up on the GPU:

```c
kokopop_model_options model_opts = {0};
model_opts.backend = KOKOPOP_BACKEND_AUTO;

kokopop_model *model = NULL;
kokopop_model_load("models/kokoro.gguf", &model_opts, &model);

if (kokopop_model_backend(model) == KOKOPOP_BACKEND_CPU) {
    /* No GPU backend was available; expect a lower real-time factor. */
}
```

It never returns `KOKOPOP_BACKEND_AUTO`, and returns `KOKOPOP_BACKEND_CPU` for a
`NULL` model.

## Chunked pull synthesis

Use `kokopop_synthesis` when you want progressive audio chunks and explicit
back-pressure from your own event loop.

```c
kokopop_synthesis_options opts = {0};
opts.voice = "af_heart";
opts.speed = 1.0f;
opts.mode = KOKOPOP_SYNTH_ADAPTATIVE;
opts.first_chunk_target_tokens = 64; /* optional */
opts.enable_diffusion = 0;           /* default: stable voice style */

kokopop_synthesis *synth = NULL;
kokopop_synthesis_create(model, &opts, &synth);

kokopop_synthesis_push_text(synth, "First fragment. ");
kokopop_synthesis_push_text(synth, "Second fragment.");
kokopop_synthesis_finish_input(synth);

for (;;) {
    kokopop_audio_chunk *chunks = NULL;
    size_t n_chunks = 0;
    int rc = kokopop_synthesis_next(synth, 2, &chunks, &n_chunks);
    if (rc != KOKOPOP_OK || n_chunks == 0) break;

    for (size_t i = 0; i < n_chunks; ++i) {
        /* chunks[i].samples is float32 PCM at chunks[i].sample_rate */
    }
    int done = n_chunks > 0 && chunks[n_chunks - 1].is_final;
    kokopop_audio_chunks_free(chunks, n_chunks);
    if (done) break;
}

kokopop_synthesis_free(synth);
```

`KOKOPOP_SYNTH_ADAPTATIVE` targets low time-to-first-audio and adjusts chunk
size as generation progresses. `KOKOPOP_SYNTH_LONG_FORM` uses larger chunks for
more stable long-form prosody. The token and pause fields in
`kokopop_synthesis_options` are optional overrides; leave them as zero to use
the mode preset. `trim_silence` uses `0` for preset, `1` for enabled, and `-1`
for disabled.

### Diffusion style sampling

Diffusion style sampling is opt-in through `enable_diffusion` and the
`diffusion_*` fields on `kokopop_synthesis_options`.

```c
kokopop_synthesis_options opts = {0};
opts.voice = "af_heart";
opts.speed = 1.0f;
opts.mode = KOKOPOP_SYNTH_ADAPTATIVE;

opts.enable_diffusion = 1;
opts.diffusion_seed = 1234;
opts.diffusion_steps = 5;              /* default when <= 0 */
opts.diffusion_alpha = 0.1f;           /* default when 0 */
opts.diffusion_beta = 0.5f;            /* default when 0 */
opts.diffusion_embedding_scale = 1.0f; /* default when 0 */
```

Leave `enable_diffusion` at `0` for the stable voice style path and source
compatibility with existing callers. When enabled, the model must contain
`kokopop.diffusion.*` tensors emitted by `tools/convert_kokoro_to_gguf.py`.
Builds without a C++ diffusion sampler fail the request with a clear
`KOKOPOP_ERROR_INFERENCE` error instead of silently changing the generated
audio. One-shot `kokopop_synthesize_text()` and
`kokopop_synthesize_phonemes()` intentionally keep the stable path; use
`kokopop_synthesis` when you need diffusion options.

For v1, push all text before generation. After `kokopop_synthesis_finish_input()`
and the first `kokopop_synthesis_next()`, further `push_text()` calls fail.

## Streaming audio encoding

`kokopop_audio_encoder` converts float32 samples to a streamable byte format.

```c
kokopop_encoder_options enc_opts = {0};
enc_opts.format = KOKOPOP_AUDIO_WAV_PCM16;
enc_opts.sample_rate = 24000;

kokopop_audio_encoder *enc = NULL;
kokopop_audio_encoder_create(&enc_opts, &enc);

kokopop_bytes bytes = {0};
kokopop_audio_encoder_start(enc, &bytes);
kokopop_bytes_free(&bytes);

kokopop_audio_encoder_push(enc, audio.samples, audio.n_samples, 1, &bytes);
/* PCM and OGG may return bytes here; WAV returns the full file at finish. */
kokopop_bytes_free(&bytes);

kokopop_audio_encoder_finish(enc, 1, &bytes);
/* bytes.data contains the final WAV bytes for WAV mode. */
kokopop_bytes_free(&bytes);
kokopop_audio_encoder_free(enc);
```

Formats:

| Format | Behavior |
|---|---|
| `KOKOPOP_AUDIO_PCM_F32LE` | Emits raw little-endian float32 PCM on each push. |
| `KOKOPOP_AUDIO_WAV_PCM16` | Accumulates samples and emits a complete WAV file at finish. |
| `KOKOPOP_AUDIO_OGG_OPUS` | Emits Ogg/Opus header/pages progressively. Requires `KOKOPOP_ENABLE_OPUS` dependencies at build time. |

If Ogg/Opus is unavailable, `kokopop_audio_encoder_create()` returns an error
and `kokopop_last_error()` explains that the format is disabled.

## Ownership and Threading

Free every `kokopop_audio` with `kokopop_audio_free()`, every chunk array with
`kokopop_audio_chunks_free()`, and every `kokopop_bytes` with
`kokopop_bytes_free()`.

A `kokopop_synthesis` session is not thread-safe. The underlying model should
not run multiple inference calls concurrently unless the caller serializes them
externally, as the HTTP scheduler does internally.
