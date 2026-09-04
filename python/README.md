# kokopop for Python

Native Python bindings for running Kokoro GGUF models with the Kokopop runtime.

## Installation

```bash
pip install kokopop
```

The package is distributed as source code. `pip install` compiles the
native C++ runtime on your machine. Prerequisites are a C++17 compiler and
CMake 3.24 or newer.

### Selecting a GPU backend

By default, the build is **CPU-only**. No backend is detected automatically,
even on platforms where the SDK is present, because the chosen backend
determines the runtime dependencies of the resulting installation. Opt in
explicitly with an environment variable at install time:

```bash
KOKOPOP_ENABLE_METAL=ON  pip install kokopop   # macOS (requires Xcode CLT)
KOKOPOP_ENABLE_CUDA=ON   pip install kokopop   # NVIDIA (requires CUDA toolkit)
KOKOPOP_ENABLE_VULKAN=ON pip install kokopop   # cross-platform (requires Vulkan SDK)
```

Available environment variables:

| Variable | Default | Notes |
|---|---|---|
| `KOKOPOP_ENABLE_METAL` | `OFF` | macOS Metal backend. Apple platforms only. |
| `KOKOPOP_ENABLE_CUDA` | `OFF` | NVIDIA CUDA backend. Requires the CUDA toolkit. |
| `KOKOPOP_ENABLE_VULKAN` | `OFF` | Vulkan backend. Requires the Vulkan SDK and SPIR-V headers. |
| `KOKOPOP_ENABLE_OPUS` | `ON` | Build Ogg/Opus encoder if `opus`, `ogg`, `libopusenc` are found. |

When switching backends, force a clean rebuild:

```bash
pip install --no-cache-dir --force-reinstall kokopop
```

For development builds from a local checkout, see
[BUILDING.md](BUILDING.md).

## Quick start

```python
import kokopop

model = kokopop.Model("../models/kokoro-md.gguf", backend="cpu")
audio = model.synthesize("Hello!", voice="af_heart")
audio.write_wav("hello.wav")
```

## One-shot synthesis

```python
import kokopop

model = kokopop.Model("../models/kokoro-md.gguf", backend=kokopop.Backend.CPU)

audio = model.synthesize("Hello, world!", voice="af_heart", speed=1.0)
audio.write_wav("output.wav")

phoneme_audio = model.synthesize_phonemes("abc", voice="af_heart")
wav_bytes = phoneme_audio.to_wav_bytes()
```

`Audio` objects expose float32 PCM samples through Python's buffer protocol, so
`memoryview(audio)` can be passed to code that accepts contiguous float32 audio.

## Chunked pull synthesis

Use `SynthesisSession` when you want progressive audio chunks and explicit
back-pressure from your own loop.

```python
import kokopop

model = kokopop.Model("../models/kokoro-md.gguf", backend="cpu")

with kokopop.SynthesisSession(
    model,
    voice="af_heart",
    speed=1.0,
    mode=kokopop.Mode.ADAPTATIVE,
    first_chunk_target_tokens=64,
    enable_diffusion=False,
) as synth:
    synth.push_text("First fragment. ")
    synth.push_text("Second fragment.")
    synth.finish_input()

    while True:
        chunks = synth.next(max_chunks=2)
        if not chunks:
            break

        for chunk in chunks:
            pcm = memoryview(chunk)
            # Write, play, or encode float32 PCM at chunk.sample_rate.

        if chunks[-1].is_final:
            break
```

For the common case where all input text is known up front, `Model.stream()`
returns a lazy iterator over `AudioChunk` objects:

```python
for chunk in model.stream(
    "First fragment. Second fragment.",
    voice="af_heart",
    mode="long_form",
    max_chunks=2,
):
    pcm = memoryview(chunk)
```

Diffusion style options are accepted by `SynthesisSession` and `Model.stream()`
using the same names as the C API (`enable_diffusion`, `diffusion_seed`,
`diffusion_steps`, `diffusion_alpha`, `diffusion_beta`,
`diffusion_embedding_scale`). Leave `enable_diffusion=False` to use the stable
default voice style path.

## Streaming audio encoding

`AudioEncoder` converts float32 PCM samples to streamable bytes.

```python
import kokopop

encoder = kokopop.AudioEncoder(kokopop.AudioFormat.WAV_PCM16, sample_rate=24000)
with encoder:
    header = encoder.start()
    body = encoder.push(memoryview(audio), is_final=True)
    final_wav = header + body + encoder.finish()
```

Formats:

| Format | Behavior |
|---|---|
| `kokopop.AudioFormat.PCM_F32LE` | Emits raw little-endian float32 PCM on each push. |
| `kokopop.AudioFormat.WAV_PCM16` | Accumulates samples and emits a complete WAV file at finish. |
| `kokopop.AudioFormat.OGG_OPUS` | Emits Ogg/Opus pages progressively when Ogg/Opus support is enabled. |
