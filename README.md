# kokopop: Standalone neural TTS runtime (Kokoro, sanoTTS)

A standalone C++ library and toolkit for running neural text-to-speech models in
GGUF format, with no Python dependencies. Two architectures are supported:
[Kokoro](https://github.com/hexgrad/kokoro) (82M, 24 kHz) and
[sanoTTS](https://github.com/Ampixa/sanoTTS) (0.3M to 1.5M, 22.05 or 24 kHz).

## Features

- **Two model architectures** in one runtime and one API, see [Supported models](#supported-models)
- **Zero dependencies beyond libespeak-ng and ggml:** no Python or heavy ML frameworks
- **Inference backends**:
  - **CPU** with configurable thread count
  - **Metal GPU** (macOS)
  - **CUDA** (Linux/Windows) on NVIDIA GPUs
  - **Vulkan** (Linux/Windows/macOS via MoltenVK)
  - **OpenCL** (Android/Adreno, where Vulkan is unavailable)
- **Streaming API** for real-time audio generation
- **Chunked synthesis** for long-form text processing
- **WAV, PCM (float32/s16), and Ogg/Opus audio output**
- **Full C & C++ API:** usable from C, C++, Rust, Go, and other languages via FFI
- **Multi-voice GGUF** with per-voice sample rates

📊 See [Benchmarks](#benchmarks)

## Supported models

One GGUF carries one architecture and one or more voices. `kokopop_probe`
prints what a file contains; the C API exposes the same through
`kokopop_model_arch()`, `kokopop_model_voice_count()` and
`kokopop_model_voice_sample_rate()`.

| Architecture | Voices | Decoder | Rate | Size | Frontend |
|---|---|---|---|--:|---|
| `kokoro-82m` | any subset of the Kokoro voice packs | ISTFTNet | 24000 Hz | ~157 MiB (F16, all voices in one file) | misaki via espeak-ng |
| `sanotts` / `heart` | 1 | Vocos (F32) | 24000 Hz | 8.9 MiB | misaki via espeak-ng |
| `sanotts` / `heartnano` | 1 | TinyVocos (int8 → F32) | 24000 Hz | 1.4 MiB | misaki via espeak-ng |
| `sanotts` / `amy`, `kristin`, and equivalent Piperlite packs | 1 each | Piperlite | 22050 Hz | 2.7 – 3.1 MiB | Piper (NFD, interleaved PAD) |

A single sanoTTS GGUF may hold several voices at **mixed sample rates** — the
four above come to 15.6 MiB together. Everything downstream of the voice
resolution (session, WAV writer, HTTP encoders) uses that voice's rate, not the
model's.

Not supported: the sanoTTS `chinese` voice (acoustic adapter), the `robot` / R7
MCU variants, Piperlite decoders with `res_layers != 1`, an output adapter or
`pre_tanh_repair`.

## Quick Start

### Prerequisites

- CMake 3.24+
- pkg-config
- C++17 / C11 compiler (GCC 11+, Clang 14+, MSVC 2019+)
- [libespeak-ng](https://github.com/espeak-ng/espeak-ng) (phonemization)

### Build

Install `espeak-ng` (required for phonemization):

**Debian / Ubuntu:**
```bash
sudo apt install libespeak-ng-dev
```

**macOS (Homebrew):**
```bash
brew install espeak-ng
```

Then build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Install and use from another CMake project

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /usr/local
```

The install ships `kokopop.h`, the library, the CLI tools and a CMake package:

```cmake
find_package(kokopop REQUIRED)
target_link_libraries(my_app PRIVATE kokopop::kokopop)
```

This works from a shared or a static build; a static one adds ggml, yyjson,
espeak-ng and the C++ runtime to your link line for you. Pass
`-DKOKOPOP_INSTALL=OFF` to suppress the install rules when embedding kokopop as
a subdirectory of another project.

### Build with Metal GPU support (macOS)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKOKOPOP_ENABLE_METAL=ON
cmake --build build
```

### Build with CUDA support

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKOKOPOP_ENABLE_CUDA=ON
cmake --build build
```

### Build with Vulkan support

On Linux (Debian/Ubuntu), install the Vulkan SDK plus SPIR-V headers so CMake can find Vulkan and `glslc`:

```bash
sudo apt install libvulkan-dev vulkan-tools glslang-tools glslc spirv-headers
```

On macOS, install the Vulkan SDK plus SPIR-V headers so CMake can find Vulkan, `glslc`, MoltenVK, and `spirv/unified1/spirv.hpp`:

```bash
brew install vulkan-sdk vulkan-headers spirv-headers
```

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKOKOPOP_ENABLE_VULKAN=ON
cmake --build build
```

### Build with OpenCL support

The OpenCL backend targets Qualcomm Adreno GPUs on Android, where Vulkan is not
an option: `ggml-vulkan` hard-requires `VK_KHR_16bit_storage`, which the Adreno
6xx driver does not expose.

ggml's OpenCL backend needs the Khronos headers (`CL/cl.h`) and an ICD loader.
kokopop deliberately does not call `find_package(OpenCL)` itself. ggml does,
and cross-compiling embedders pre-seed the result as cache variables, following
the same pattern as `ESPEAK_NG_LIBRARY` for Android builds:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKOKOPOP_ENABLE_OPENCL=ON \
  -DOpenCL_INCLUDE_DIR=/path/to/OpenCL-Headers \
  -DOpenCL_LIBRARY=/path/to/libOpenCL.so
cmake --build build
```

Kernels are embedded in the library (`GGML_OPENCL_EMBED_KERNELS=ON`), so there
are no `.cl` files to ship alongside it. If kernel compilation fails at runtime
on an older driver, set `-DKOKOPOP_OPENCL_TARGET_VERSION=200`; if the optimized
Adreno matmul path misbehaves, rebuild ggml with
`-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF` to fall back to the generic kernels.

### HTTP Server Dependencies (Ogg/Opus)

The HTTP server (`kokopop_stream --http`) supports Ogg/Opus audio streaming. Install the required libraries:

**Debian / Ubuntu:**
```bash
sudo apt install libopus-dev libogg-dev libopusenc-dev
```

**macOS (Homebrew):**
```bash
brew install libogg opus libopusenc
```

If these libraries are not found, CMake will automatically disable Ogg/Opus output and continue with PCM and WAV support.

### Export a Kokoro model to GGUF format

A pre-converted GGUF model is available at **[tterrasson/Kokoro-GGUF](https://huggingface.co/tterrasson/Kokoro-GGUF/tree/main)**.

To convert the model yourself, install [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv run python tools/convert_kokoro_to_gguf.py \
  --output models/kokoro.gguf \
  --voices af_heart,ff_siwis,zf_xiaoxiao,im_nicola
```

> **Note:** The Kokoro PyTorch model is automatically downloaded from [Hugging Face](https://huggingface.co/hexgrad/Kokoro-82M) on first run.

The converter emits a single F16 model (~157 MiB). Earlier tier-based
quantization (`kokoro-md` / `kokoro-lg`) was removed: any K-quant on the
acoustic path produces small per-element errors that compound through AdaIN
+ Snake1D, and any K-quant on ALBERT compounds through 12 transformer
layers. Both eventually destabilise the duration head or saturate the vocoder
on some inputs. F16 stays bit-stable across all backends.

#### Diffusion style tensors

The converter records diffusion metadata (`kokopop.diffusion.*`) and exports
diffusion UNet tensors when the source Kokoro package exposes an available
diffusion module. Models without that module continue to load and synthesize
normally; the default runtime path always uses the deterministic voice style
embedded in `kokopop.voice.*`.

Diffusion style sampling is opt-in through the chunked C API
(`kokopop_synthesis_options.enable_diffusion`) and the matching Python
streaming options. Leave it disabled for existing behavior. If diffusion is
enabled with a model that has no diffusion tensors, or with a build that does
not include a C++ sampler for them, synthesis fails with an explicit error
instead of silently changing voice style or perturbing standard generation.

### Export sanoTTS voices to GGUF format

The converter downloads the voice packs from an immutable Hugging Face commit
into `~/.cache/sanotts/`:

```bash
uv run python tools/convert_sanotts_to_gguf.py \
  --output models/sanotts-en.gguf \
  --voices heart,heartnano,amy,kristin
```

One voice per file works the same way, and `--dry-run` prints the tensor plan,
its shapes and its sizes without writing anything:

```bash
uv run python tools/convert_sanotts_to_gguf.py \
  --output /dev/null --voices heart --dry-run
```

The two input families are packaged very differently. A Piperlite pack is
self-describing — its `manifest.json` gives every tensor's name, shape, dtype,
offset and SHA-256, so the conversion is a validated copy. A Vocos pack is two
nameless byte blobs plus a generated C header of offsets, so the converter
reconstructs the layout from the declared dimensions and checks each
reconstructed size against the next offset, the blob totals and the declared
parameter count. A checkpoint whose operators are not covered is refused, not
guessed at.

> **Note:** `AUTO` backend selection resolves to **CPU** for sanoTTS. The
> models are two orders of magnitude smaller than Kokoro's, so dispatch
> overhead dominates and a GPU wins nothing, see [Benchmarks](#benchmarks).
> `--backend metal` (or `cuda`, `vulkan`, `opencl`) still forces a GPU.

### Usage

Synthesize text to a WAV file:

```bash
./build/kokopop_say \
  --model models/kokoro.gguf \
  --voice af_heart \
  --text "Hello world." \
  --out hello.wav
```

For embedding, see the [C API guide](docs/c-api.md), including one-shot
synthesis, backend selection, pull-based chunked synthesis, and streaming audio
encoders.

Generate audio from phonemes:

```bash
./build/kokopop_say \
  --model models/kokoro.gguf \
  --voice af_heart \
  --phonemes "həˈloʊ wɜrld" \
  --out hello.wav
```

Adjust generation speed:

```bash
./build/kokopop_say \
  --model models/kokoro.gguf \
  --voice af_heart \
  --text "Hello John, how are you today?" \
  --speed 1.5 \
  --out fast.wav
```

The same tool reads a sanoTTS GGUF. `--voice` selects from the file, and the
WAV is written at *that voice's* rate — 24000 Hz here, 22050 Hz for `amy` from
the very same file:

```bash
./build/kokopop_say \
  --model models/sanotts-en.gguf \
  --voice heart \
  --text "Hello world." \
  --out hello.wav
```

sanoTTS decoders are driven by deterministic noise. Two runs of the same text
already give the same audio; `--seed` pins the stream explicitly, which is what
makes a rendering reproducible across voices and versions. It applies to
sanoTTS models only, and warns rather than changing anything on a Kokoro one:

```bash
./build/kokopop_say \
  --model models/sanotts-en.gguf \
  --voice heart \
  --text "Hello world." \
  --seed 1234 \
  --out hello.wav
```

Inspect what a GGUF contains, and what each stage produces for one utterance:

```bash
./build/kokopop_probe \
  --model models/sanotts-en.gguf \
  --voice heart \
  --text "Hello world."
```

### Streaming mode

The `kokopop_stream` tool supports two operating modes: **STDIO** (default) and **HTTP server** (asynchronous and event-driven).

#### STDIO mode (default)

Reads JSON commands from stdin and streams raw audio (float32) to stdout:

```bash
# Feed commands via stdin
echo '{"text": "Hello world", "flush": true}' | \
  ./build/kokopop_stream \
    --model models/kokoro.gguf \
    --voice af_heart \
    --mode long_form

# Save full output to WAV (accumulates all chunks, writes WAV on exit)
echo '{"text": "Hello world", "flush": true}' | \
  ./build/kokopop_stream \
    --model models/kokoro.gguf \
    --voice af_heart \
    --out output.wav
```

JSON protocol (one command per line):

| Command | Description |
|---|---|
| `{"text": "..."}` | Accumulate text |
| `{"text": "...", "flush": true}` | Add text and trigger generation |
| `{"flush": true}` | Generate all accumulated text |
| `{"stop": true}` | Stop streaming |

#### HTTP server mode (asynchronous)

Start an asynchronous, event-driven HTTP server for TTS synthesis. The server
uses `poll()` for non-blocking I/O and a `SynthesisScheduler` to interleave
chunks from concurrent requests in round-robin order:

```bash
./build/kokopop_stream --model models/kokoro.gguf --http --port 8080

# Unload model after 10 minutes of inactivity (saves memory, reloads on next request)
./build/kokopop_stream --model models/kokoro.gguf --http --port 8080 --idle-unload 10

# Bind to all interfaces, use a different port
./build/kokopop_stream --model models/kokoro.gguf --http --port 9000 --bind 0.0.0.0

# Force a specific backend (cpu, metal, cuda, vulkan, opencl)
./build/kokopop_stream --model models/kokoro.gguf --http --backend cuda

# Set default speed and mode for all requests
./build/kokopop_stream --model models/kokoro.gguf --http --speed 1.2 --mode long_form

# Use a specific voice as the default for all requests
./build/kokopop_stream --model models/kokoro.gguf --http --voice af_heart

# Set thread count
./build/kokopop_stream --model models/kokoro.gguf --http --threads 8
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--http` | not set | Run in HTTP server mode (asynchronous and event-driven) |
| `--port N` | `8080` | HTTP server port |
| `--bind ADDR` | `127.0.0.1` | HTTP server bind address (use `0.0.0.0` for all interfaces) |
| `--idle-unload N` | disabled | Unload model after N minutes of inactivity; reload on next request (saves memory) |
| `--backend` | `auto` | Inference backend: `cpu`, `metal`, `cuda`, `vulkan`, or `opencl` |
| `--threads N` | `min(4, hw_concurrency)` | Number of inference threads (affects model loading; scheduler worker is single-threaded) |
| `--speed FLOAT` | `1.0` | Default synthesis speed for HTTP requests |
| `--mode MODE` | `adaptative` | Default synthesis mode: `adaptative` or `long_form` |
| `--voice NAME` | not set | Default voice for HTTP requests (overrides the per-request `voice` field) |

> **Note:** In HTTP server mode, `--voice` sets the server-wide default voice. Individual requests can override it by including a `voice` field in the JSON payload. Without `--http`, `--voice` is required.

**Architecture:**

- Single-threaded event loop using `poll()` with non-blocking sockets
- Max **64 concurrent connections** (configurable via `MAX_CONNECTIONS`)
- Max **16 MiB request body** (`MAX_BODY_SIZE`)
- Max **64 KiB headers** (`MAX_HEADER_SIZE`)
- **30-second idle timeout** for connections not fully receiving headers/body
- **256 KiB write buffer high-water mark** for back-pressure on slow clients
- Round-robin chunk interleaving across concurrent requests via `SynthesisScheduler`

**Available endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/tts` | `POST` | Synthesize text to audio: PCM float32, Ogg/Opus, or complete WAV |
| `/health` | `GET` | Server health check. Returns `{"status":"ready","sample_rate":24000}` or `{"status":"unloaded"}` if the model was unloaded due to inactivity |
| `/voices` | `GET` | List all voices embedded in the GGUF model. Returns `{"voices":[{"name":"..."}, ...]}` |

**Example requests:**

```bash
# Stream raw PCM float32 (default), using chunked transfer encoding
curl -X POST http://localhost:8080/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world", "voice": "ff_siwis", "speed": 1.0}' \
  -o output.raw

# Convert PCM to WAV with ffmpeg:
ffmpeg -f f32le -ar 24000 -ac 1 -i output.raw output.wav

# Request a complete WAV file directly (buffered server-side)
curl -X POST http://localhost:8080/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world", "voice": "ff_siwis", "format": "wav"}' \
  -o output.wav

# Stream Ogg/Opus (requires Ogg/Opus libraries at build time)
curl -X POST http://localhost:8080/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world", "voice": "af_heart", "format": "ogg"}' \
  -o output.ogg

# Health check
curl http://localhost:8080/health

# List voices, in file order, each with its own sample rate
curl http://localhost:8080/voices
# {"voices":[{"name":"heart","sample_rate":24000},{"name":"amy","sample_rate":22050}]}
```

`GET /health` reports the architecture alongside the default voice's rate:
`{"status":"ready","arch":"sanotts","sample_rate":24000}`. On a pack that mixes
rates, the per-voice value from `/voices` is the one a client must use to
interpret a raw PCM stream.

**POST /tts request body fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | **required** | Text to synthesize (max 100,000 characters) |
| `voice` | string | CLI `--voice` | Voice name (required if not set via CLI `--voice`) |
| `speed` | float | CLI `--speed` (1.0) | Synthesis speed multiplier |
| `mode` | string | CLI `--mode` (`adaptative`) | `adaptative` (fast TTFB, dynamic chunk sizing) or `long_form` (larger chunks, better prosody) |
| `format` | string | `pcm` | Output format: `pcm` (raw float32 stream), `wav` (complete WAV file), or `ogg` (Ogg/Opus stream) |
| `prebuffer_chunks` | int | `0` | For `ogg` format: number of server-side Ogg synthesis chunks to buffer before starting playback |
| `first_chunk_target_tokens` | int | preset default | Override the first-chunk token target to reduce TTFB (e.g., `50` for faster initial audio) |
| `noise_seed` | int | derived from the voice | sanoTTS only: pin the decoder's deterministic noise. Must be a non-negative integer; `0` is a valid seed, not "unset" |

**Response formats:**

- **PCM** (`format: pcm`): Raw float32 big-endian or native-endian PCM stream with `Transfer-Encoding: chunked` and `Content-Type: audio/pcm-f32le`
- **WAV** (`format: wav`): Complete WAV file returned as a single response (server accumulates all chunks)
- **Ogg/Opus** (`format: ogg`): Ogg/Opus stream with `Transfer-Encoding: chunked` and `Content-Type: audio/ogg`

**Error responses:**

| HTTP Status | Meaning |
|---|---|
| `400 Bad Request` | Invalid JSON, missing fields, unknown mode/format |
| `405 Method Not Allowed` | Wrong HTTP method on `/tts` (POST required) |
| `413 Payload Too Large` | Text exceeds 100,000 characters |
| `501 Not Implemented` | Ogg/Opus not available in this build |
| `503 Service Unavailable` | Model unloaded (idle timeout) and reload failed; or scheduler not configured |

#### Python client (`tools/tts_client.py`)

A minimal Python client is provided. It requires no third-party packages.

```bash
# Start the server first
./build/kokopop_stream --model models/kokoro.gguf --http --port 8080

# Stable Ogg/Opus playback for longer text (requires ffplay)
uv run python tools/tts_client.py \
  --file examples/english.txt \
  --format ogg \
  --mode long_form \
  --prebuffer-chunks 2 \
  --voice af_bella | ffplay -i pipe:0

# Lower-latency adaptive playback. The server sizes chunks from live timings.
uv run python tools/tts_client.py \
  --file examples/english.txt \
  --format ogg \
  --prebuffer-chunks 2 \
  --voice af_bella | ffplay -i pipe:0

uv run python tools/tts_client.py \
  --file examples/mandarin.txt \
  --format ogg \
  --voice zm_yunjian | ffplay -i pipe:0

# Stream PCM and save as WAV (client-side conversion)
uv run python tools/tts_client.py "Hello world" --out hello.wav

# Receive a complete WAV file from the server
uv run python tools/tts_client.py "Hello world" --format wav --out hello.wav

# Override voice and speed
uv run python tools/tts_client.py "Hello faster world" --speed 1.2 --format wav --out hello.wav
```

All options:

| Option | Default | Description |
|---|---|---|
| `--url` | `http://127.0.0.1:8080/tts` | Server URL |
| `--voice` | server default | Voice name |
| `--speed` | `1.0` | Speed multiplier |
| `--mode` | `adaptative` | `adaptative` or `long_form` |
| `--format` | `pcm` | `pcm` (float32 stream), `wav` (complete file), or `ogg` (Ogg/Opus stream) |
| `--out` | not set | Output file. If omitted, raw bytes go to stdout |
| `--prebuffer-chunks` | `0` | Server-side Ogg synthesis chunks to buffer before playback |

## Docker

A multi-stage [Dockerfile](Dockerfile) is provided for building and running kokopop without local dependencies.
The default image is CPU-only. A separate CUDA target builds against NVIDIA CUDA and uses
`nvidia/cuda:13.2.1-runtime-ubuntu24.04` as its runtime image.

> **⚠️ Performance note:** The pre-built Docker images use conservative SIMD flags. If your host
> CPU lacks **AVX2** (x86_64/amd64) or **NEON** (ARM64), inference will fall back to slower
> scalar code and may be noticeably slower. For best performance on older or embedded hardware,
> build the image locally so CMake can detect and enable the appropriate SIMD instructions.

### Pre-built image (CPU)

A ready-to-use CPU image is published on Docker Hub as [`tterrasson/kokopop-cpu`](https://hub.docker.com/r/tterrasson/kokopop-cpu):

```bash
# Pull the image
docker pull tterrasson/kokopop-cpu

# Run the HTTP server (default voices: af_heart, ff_siwis, zf_xiaoxiao, im_nicola)
docker run --cpus 4 -p 8080:8080 tterrasson/kokopop-cpu

# Override voice, port, or threads at runtime
docker run --cpus 4 -p 9000:9000 tterrasson/kokopop-cpu \
  --model models/kokoro.gguf \
  --http \
  --voice bf_emma \
  --bind 0.0.0.0 \
  --port 9000

# Quick one-shot synthesis
echo '{"text": "Hello from Docker!", "flush": true}' | \
  docker run -i --rm tterrasson/kokopop-cpu \
    --model models/kokoro.gguf \
    --voice af_heart \
    --mode long_form > output.raw

# Convert to WAV
ffmpeg -f f32le -ar 24000 -ac 1 -i output.raw output.wav
```

### Build

```bash
# Build the default CPU image with default voices
# (af_heart, ff_siwis, zf_xiaoxiao, im_nicola)
docker build --format docker -t kokopop-cpu .

# Build the CPU image with custom voices
docker build --format docker \
  --build-arg VOICES="af_heart,bf_emma,zf_xiaoxiao" \
  -t kokopop-cpu .

# Build the CUDA image
docker build --format docker --target runtime-cuda -t kokopop-cuda .

# Build the CUDA image with custom voices
docker build --format docker \
  --target runtime-cuda \
  --build-arg VOICES="af_heart,bf_emma,zf_xiaoxiao" \
  -t kokopop-cuda .
```

### Run the HTTP server

```bash
# CPU, capped at 4 CPUs. kokopop_stream auto-selects
# min(4, hardware_concurrency()) threads unless --threads is passed.
docker run --cpus 4 -p 8080:8080 kokopop-cpu

# CUDA with Docker
docker run --gpus all -p 8080:8080 kokopop-cuda

# CUDA with Podman using NVIDIA CDI
podman run --device nvidia.com/gpu=all -p 8080:8080 kokopop-cuda
```

To override kokopop's inference thread count explicitly, pass `--threads N`
after the image name together with the full command arguments.

### Override voice, port, mode, or threads at runtime

```bash
docker run --cpus 4 -p 9000:9000 kokopop-cpu \
  --model models/kokoro.gguf \
  --http \
  --voice bf_emma \
  --bind 0.0.0.0 \
  --port 9000 \
  --mode long_form \
  --threads 4
```

### Quick synthesis (one-shot)

Override the default HTTP server command to perform one-shot synthesis with the stream tool:

```bash
echo '{"text": "Hello from Docker!", "flush": true}' | \
  docker run -i --rm kokopop-cpu \
    --model models/kokoro.gguf \
    --voice af_heart \
    --mode long_form > output.raw

# Convert to WAV
ffmpeg -f f32le -ar 24000 -ac 1 -i output.raw output.wav
```

## Library Integration

Link against `libkokopop` in your CMake project:

```cmake
target_link_libraries(myapp PRIVATE kokopop)
target_include_directories(myapp PRIVATE ${CMAKE_SOURCE_DIR}/include)
```

### C API example

All functions return `KOKOPOP_OK` (0) on success. On failure, call
`kokopop_last_error()` for a human-readable description.

**One-shot synthesis**

```c
#include "kokopop.h"
#include <stdio.h>

/* Optional: configure backend and thread count */
kokopop_model_options model_opts = {0};
model_opts.backend    = KOKOPOP_BACKEND_AUTO; /* CPU, Metal, CUDA, Vulkan, or OpenCL */
model_opts.n_threads  = 4;

kokopop_model *model = NULL;
if (kokopop_model_load("models/kokoro.gguf", &model_opts, &model) != KOKOPOP_OK) {
    fprintf(stderr, "load failed: %s\n", kokopop_last_error());
    return 1;
}

/* Synthesize text → WAV */
kokopop_audio audio = {0};
if (kokopop_synthesize_text(model, "Hello, world!", "af_heart", 1.0f, &audio) != KOKOPOP_OK) {
    fprintf(stderr, "synthesis failed: %s\n", kokopop_last_error());
}
kokopop_write_wav("output.wav", &audio);

/* Query the model sample rate */
int sample_rate = kokopop_model_sample_rate(model); /* typically 24000 */

/* Query the backend AUTO actually resolved to (KOKOPOP_BACKEND_*) */
int32_t resolved_backend = kokopop_model_backend(model);

kokopop_audio_free(&audio);
kokopop_model_free(model);
```

Use `kokopop_synthesize_phonemes()` when your input is already phonemized:

```c
kokopop_synthesize_phonemes(model, "həˈloʊ wɜrld", "af_heart", 1.0f, &audio);
```

**Chunked pull synthesis**

Use `kokopop_synthesis` when you want progressive audio chunks and
back-pressure from your own event loop:

```c
kokopop_synthesis_options opts = {0};
opts.voice  = "af_heart";
opts.speed  = 1.0f;
opts.mode   = KOKOPOP_SYNTH_ADAPTATIVE; /* or KOKOPOP_SYNTH_LONG_FORM */
opts.first_chunk_target_tokens = 50;   /* optional: reduce TTFB */

kokopop_synthesis *synth = NULL;
kokopop_synthesis_create(model, &opts, &synth);

kokopop_synthesis_push_text(synth, "First fragment. ");
kokopop_synthesis_push_text(synth, "Second fragment.");
kokopop_synthesis_finish_input(synth);

for (;;) {
    kokopop_audio_chunk *chunks = NULL;
    size_t n = 0;
    if (kokopop_synthesis_next(synth, 2, &chunks, &n) != KOKOPOP_OK || n == 0) break;

    for (size_t i = 0; i < n; ++i) {
        /* chunks[i].samples: float32 PCM at chunks[i].sample_rate */
        /* chunks[i].chunk_index, chunks[i].is_final */
    }
    int done = chunks[n - 1].is_final;
    kokopop_audio_chunks_free(chunks, n);
    if (done) break;
}

kokopop_synthesis_free(synth);
```

**Streaming audio encoding**

Convert float32 samples to a streamable byte format (PCM, WAV, or Ogg/Opus):

```c
kokopop_encoder_options enc_opts = {0};
enc_opts.format      = KOKOPOP_AUDIO_WAV_PCM16; /* PCM_F32LE, WAV_PCM16, OGG_OPUS */
enc_opts.sample_rate = 24000;

kokopop_audio_encoder *enc = NULL;
kokopop_audio_encoder_create(&enc_opts, &enc);

kokopop_bytes bytes = {0};
kokopop_audio_encoder_start(enc, &bytes); /* WAV header / Ogg stream header */
kokopop_bytes_free(&bytes);

/* Push each chunk. PCM and Ogg emit bytes immediately; WAV buffers until finish. */
kokopop_audio_encoder_push(enc, audio.samples, audio.n_samples, /*is_final=*/1, &bytes);
kokopop_bytes_free(&bytes);

kokopop_audio_encoder_finish(enc, /*success=*/1, &bytes);
/* bytes.data is the complete WAV file in WAV mode */
kokopop_bytes_free(&bytes);
kokopop_audio_encoder_free(enc);
```

## Available Voices

For a complete list of all available voices, see [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).

Common voice names include:

| Voice          | Description                         |
|----------------|-------------------------------------|
| `af_heart`     | American English, female            |
| `am_adam`      | American English, male              |
| `bf_emma`      | British English, female             |
| `bm_george`    | British English, male               |
| `zf_xiaoxiao`  | Mandarin Chinese, female            |
| `zm_yunxi`     | Mandarin Chinese, male              |
| `ef_dora`      | Spanish, female                     |
| `em_alex`      | Spanish, male                       |
| `ff_siwis`     | French, female                      |
| `if_sara`      | Italian, female                     |
| `im_nicola`    | Italian, male                       |
| `pf_dora`      | Brazilian Portuguese, female        |
| `pm_alex`      | Brazilian Portuguese, male          |

## Supported Languages

Kokopop supports the following languages:

- **English** (American & British)
- **Mandarin Chinese**
- **Spanish**
- **French**
- **Italian**
- **Brazilian Portuguese**

*Japanese and Hindi are not yet supported.*

## CMake Options

| Option                | Default | Description                      |
|-----------------------|---------|----------------------------------|
| `KOKOPOP_BUILD_TESTS` | `OFF`   | Build unit tests                 |
| `KOKOPOP_BUILD_TOOLS` | `ON`    | Build CLI tools                  |
| `KOKOPOP_ENABLE_METAL`| `OFF`   | Enable Metal GPU backend (macOS) |
| `KOKOPOP_ENABLE_CUDA` | `OFF`   | Enable CUDA backend (NVIDIA GPUs) |
| `KOKOPOP_ENABLE_VULKAN` | `OFF` | Enable Vulkan GPU backend |
| `KOKOPOP_VULKAN_VALIDATE` | `OFF` | Enable Vulkan validation layers in ggml |
| `KOKOPOP_VULKAN_DEBUG` | `OFF` | Enable Vulkan debug output in ggml |
| `KOKOPOP_VULKAN_MEMORY_DEBUG` | `OFF` | Enable Vulkan memory debug output in ggml |
| `KOKOPOP_VULKAN_SHADER_DEBUG_INFO` | `OFF` | Build Vulkan shaders with debug info |
| `KOKOPOP_ENABLE_OPENCL` | `OFF` | Enable OpenCL GPU backend (Adreno / Android) |
| `KOKOPOP_OPENCL_PROFILING` | `OFF` | Enable OpenCL profiling in ggml (adds CPU overhead) |
| `KOKOPOP_OPENCL_TARGET_VERSION` | `300` | OpenCL version ggml targets (try `200` on older Adreno drivers) |
| `KOKOPOP_BUILD_BENCH` | `OFF`   | Build benchmarks (requires a model) |

## Testing

```bash
ctest --test-dir build --output-on-failure
```

## Benchmarks

Build and run benchmarks (requires a GGUF model):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKOKOPOP_BUILD_BENCH=ON
cmake --build build
./build/kokopop_bench --model models/kokoro.gguf
./build/kokopop_bench --model models/sanotts-en.gguf
```

`kokopop_bench` reads the architecture from the file and picks that
architecture's preset voice, text and repeat count. A sanoTTS voice is
50 to 500 times smaller than Kokoro, and Kokoro's reference chunk sizes do not
carry over.

### sanoTTS: CPU vs Metal

**Hardware:** MacBook Pro M1, **Threads:** 4, ~63 tokens, 8 iterations,
minimum of the per-iteration wall time for one full utterance.

| Voice | Decoder | Audio | CPU | Metal | Peak RSS (CPU / Metal) |
|---|---|--:|--:|--:|--:|
| `heart` | Vocos | 3.77 s | 16.1 ms | 16.0 ms | 74 MB / 224 MB |
| `heartnano` | TinyVocos | 3.62 s | 5.2 ms | 5.2 ms | 57 MB / 207 MB |
| `amy` | Piperlite | 4.21 s | 157.0 ms | 157.2 ms | 432 MB / 583 MB |

Metal buys nothing and costs ~150 MB of resident memory, which is why `AUTO`
resolves to CPU for sanoTTS. On 4 CPU threads, `heartnano` runs at
~700x real time, `heart` at ~230x, and the Piperlite decoder, which does far
more work per frame, at ~27x.

### Runtime Benchmarks (`kokopop_rt`)

Run `kokopop_rt` to get a detailed per-chunk real-time factor breakdown:

```bash
./build/kokopop_rt \
  --model models/kokoro.gguf \
  --voice af_heart \
  --backend cpu \
  --threads 4 \
  --seed 1234 2>/dev/null
```

**Hardware:** MacBook Air M5, **Backend:** CPU, **Threads:** 8

```
~/w/n/kokopop ❯❯❯ ./build/kokopop_rt \
  --model models/sanotts-en.gguf \
  --voice heart \
  --backend cpu \
  --threads 8 \
  --seed 1234 2>/dev/null

  Backend:     CPU
  Voice:       heart
  Threads:     8
  Sample Rate: 24000 Hz
  Sentences:   10 → 15 chunk(s) (936 tokens)

  Model load time:    8.2 ms
  Prepare time:      15.7 ms (chunking + phonemization)

  ┌──────┬────────┬────────────┬──────────┬──────────┬──────────┐
  │ Chunk│ Tokens │  Gen Time  │ Duration │    RT    │ Samples  │
  ├──────┼────────┼────────────┼──────────┼──────────┼──────────┤
  │     1│       4│      4.8ms │     0.49s│   101.79x│     11712│
  │     2│      66│     16.4ms │     4.22s│   256.55x│    101248│
  │     3│      47│     10.8ms │     2.88s│   266.53x│     69040│
  │     4│      65│     17.9ms │     4.77s│   266.37x│    114536│
  │     5│      69│     16.8ms │     4.39s│   261.94x│    105344│
  │     6│      68│     15.9ms │     4.33s│   272.67x│    103808│
  │     7│      70│     20.1ms │     4.70s│   234.00x│    112768│
  │     8│      71│     17.3ms │     4.74s│   274.49x│    113792│
  │     9│      64│     14.5ms │     3.99s│   275.57x│     95872│
  │    10│      70│     17.2ms │     4.91s│   285.08x│    117936│
  │    11│      51│     11.7ms │     3.23s│   276.36x│     77488│
  │    12│      74│     17.7ms │     4.91s│   276.87x│    117888│
  │    13│      67│     16.8ms │     4.02s│   239.59x│     96384│
  │    14│      78│     16.6ms │     4.73s│   285.44x│    113584│
  │    15│      72│     14.8ms │     3.98s│   268.95x│     95608│
  └──────┴────────┴────────────┴──────────┴──────────┴──────────┘

  Total Generation:   229.2 ms
  Total Audio:        60.29 s  (1447008 samples @ 24000 Hz)
  Overall RT:        263.02x
  TTFB warm-start:      4.8 ms (1st chunk inference)
  TTFB cold-start:     28.7 ms (load + prepare + 1st chunk)
  → 263.0x faster than real-time
```

**Hardware:** MacBook Pro M1, **Backend:** CPU, **Threads:** 4

```
  Backend:     CPU
  Voice:       af_heart
  Threads:     4
  Sample Rate: 24000 Hz
  Sentences:   10 → 14 chunk(s) (923 tokens)

  Model load time:   54.3 ms
  Prepare time:      13.7 ms (chunking + phonemization)

  ┌──────┬────────┬────────────┬──────────┬──────────┬──────────┐
  │ Chunk│ Tokens │  Gen Time  │ Duration │    RT    │ Samples  │
  ├──────┼────────┼────────────┼──────────┼──────────┼──────────┤
  │     1│       4│    497.1ms │     0.98s│     1.97x│     23520│
  │     2│      77│   2038.4ms │     4.24s│     2.08x│    101640│
  │     3│      54│   1392.5ms │     3.04s│     2.18x│     72960│
  │     4│      67│   2051.9ms │     4.54s│     2.21x│    108960│
  │     5│      77│   2166.2ms │     4.58s│     2.12x│    110040│
  │     6│      74│   2050.4ms │     4.29s│     2.09x│    102840│
  │     7│      77│   2082.7ms │     4.41s│     2.12x│    105840│
  │     8│      67│   1746.9ms │     3.92s│     2.24x│     93960│
  │     9│      76│   2259.0ms │     4.71s│     2.09x│    113040│
  │    10│      66│   1676.6ms │     3.67s│     2.19x│     87960│
  │    11│      53│   1581.0ms │     3.34s│     2.11x│     80160│
  │    12│      70│   2031.6ms │     4.54s│     2.23x│    108960│
  │    13│      89│   2599.2ms │     5.59s│     2.15x│    134160│
  │    14│      72│   1947.4ms │     4.17s│     2.14x│    100080│
  └──────┴────────┴────────────┴──────────┴──────────┴──────────┘

  Total Generation:  26120.8 ms
  Total Audio:        56.00 s  (1344120 samples @ 24000 Hz)
  Overall RT:         2.14x
  TTFB warm-start:    497.1 ms (1st chunk inference)
  TTFB cold-start:    565.1 ms (load + prepare + 1st chunk)
  → 2.1x faster than real-time
```

**Hardware:** MacBook Pro M1, **Backend:** Vulkan, **Threads:** 4

```
  Backend:     Vulkan
  Voice:       af_heart
  Threads:     4
  Sample Rate: 24000 Hz
  Sentences:   10 → 14 chunk(s) (923 tokens)

  Model load time:  256.8 ms
  Prepare time:      14.0 ms (chunking + phonemization)

  ┌──────┬────────┬────────────┬──────────┬──────────┬──────────┐
  │ Chunk│ Tokens │  Gen Time  │ Duration │    RT    │ Samples  │
  ├──────┼────────┼────────────┼──────────┼──────────┼──────────┤
  │     1│       4│    319.7ms │     0.95s│     2.99x│     22920│
  │     2│      77│    833.7ms │     4.24s│     5.08x│    101640│
  │     3│      54│    585.6ms │     3.09s│     5.28x│     74160│
  │     4│      67│    798.1ms │     4.46s│     5.59x│    107160│
  │     5│      77│    896.6ms │     4.56s│     5.09x│    109440│
  │     6│      74│    822.0ms │     4.29s│     5.21x│    102840│
  │     7│      77│    830.7ms │     4.43s│     5.34x│    106440│
  │     8│      67│    710.0ms │     3.89s│     5.48x│     93360│
  │     9│      76│    888.0ms │     4.74s│     5.33x│    113640│
  │    10│      66│    698.0ms │     3.67s│     5.25x│     87960│
  │    11│      53│    647.9ms │     3.31s│     5.12x│     79560│
  │    12│      70│    812.9ms │     4.57s│     5.62x│    109560│
  │    13│      89│    977.6ms │     5.59s│     5.72x│    134160│
  │    14│      72│    780.0ms │     4.14s│     5.31x│     99480│
  └──────┴────────┴────────────┴──────────┴──────────┴──────────┘

  Total Generation:  10600.8 ms
  Total Audio:        55.93 s  (1342320 samples @ 24000 Hz)
  Overall RT:         5.28x
  TTFB warm-start:    319.7 ms (1st chunk inference)
  TTFB cold-start:    590.5 ms (load + prepare + 1st chunk)
  → 5.3x faster than real-time
```

**Hardware:** AMD Ryzen 9 7950X 16-Core Processor, **Backend:** CPU, **Threads:** 16

```
  Backend:     CPU
  Voice:       af_heart
  Threads:     16
  Sample Rate: 24000 Hz
  Sentences:   10 → 15 chunk(s) (941 tokens)

  Model load time:   22.4 ms
  Prepare time:       7.5 ms (chunking + phonemization)

  ┌──────┬────────┬────────────┬──────────┬──────────┬──────────┐
  │ Chunk│ Tokens │  Gen Time  │ Duration │    RT    │ Samples  │
  ├──────┼────────┼────────────┼──────────┼──────────┼──────────┤
  │     1│       4│    221.4ms │     0.91s│     4.09x│     21720│
  │     2│      77│    907.5ms │     4.16s│     4.58x│     99840│
  │     3│      54│    605.7ms │     2.99s│     4.94x│     71760│
  │     4│      67│    863.6ms │     4.37s│     5.05x│    104760│
  │     5│      75│    927.1ms │     4.31s│     4.65x│    103440│
  │     6│      76│    877.7ms │     4.24s│     4.83x│    101640│
  │     7│      55│    657.2ms │     3.34s│     5.08x│     80160│
  │     8│      76│   1002.9ms │     4.68s│     4.67x│    112440│
  │     9│      80│    884.0ms │     4.21s│     4.76x│    101040│
  │    10│      36│    430.9ms │     2.04s│     4.73x│     48960│
  │    11│      63│    793.2ms │     4.07s│     5.12x│     97560│
  │    12│      77│    876.2ms │     4.13s│     4.72x│     99240│
  │    13│      67│    770.7ms │     3.84s│     4.98x│     92160│
  │    14│      62│    758.7ms │     3.67s│     4.83x│     87960│
  │    15│      72│    871.1ms │     4.14s│     4.76x│     99480│
  └──────┴────────┴────────────┴──────────┴──────────┴──────────┘

  Total Generation:  11447.8 ms
  Total Audio:        55.09 s  (1322160 samples @ 24000 Hz)
  Overall RT:         4.81x
  TTFB warm-start:    221.4 ms (1st chunk inference)
  TTFB cold-start:    251.3 ms (load + prepare + 1st chunk)
  → 4.8x faster than real-time
```

**Hardware:** NVIDIA GeForce RTX 4090, **Backend:** CUDA, **Threads:** 16

```
  Backend:     CUDA
  Voice:       af_heart
  Threads:     16
  Sample Rate: 24000 Hz
  Sentences:   10 → 15 chunk(s) (941 tokens)

  Model load time:  171.2 ms
  Prepare time:       7.5 ms (chunking + phonemization)

  ┌──────┬────────┬────────────┬──────────┬──────────┬──────────┐
  │ Chunk│ Tokens │  Gen Time  │ Duration │    RT    │ Samples  │
  ├──────┼────────┼────────────┼──────────┼──────────┼──────────┤
  │     1│       4│    116.4ms │     0.91s│     7.77x│     21720│
  │     2│      77│    248.2ms │     4.13s│    16.66x│     99240│
  │     3│      54│    164.9ms │     2.99s│    18.13x│     71760│
  │     4│      67│    221.2ms │     4.37s│    19.74x│    104760│
  │     5│      75│    234.9ms │     4.31s│    18.35x│    103440│
  │     6│      76│    218.1ms │     4.24s│    19.42x│    101640│
  │     7│      55│    162.1ms │     3.34s│    20.60x│     80160│
  │     8│      76│    246.0ms │     4.68s│    19.05x│    112440│
  │     9│      80│    223.8ms │     4.21s│    18.81x│    101040│
  │    10│      36│    110.0ms │     2.04s│    18.54x│     48960│
  │    11│      63│    197.8ms │     4.07s│    20.56x│     97560│
  │    12│      77│    223.0ms │     4.13s│    18.54x│     99240│
  │    13│      67│    191.9ms │     3.84s│    20.01x│     92160│
  │    14│      62│    190.7ms │     3.67s│    19.22x│     87960│
  │    15│      72│    224.2ms │     4.14s│    18.49x│     99480│
  └──────┴────────┴────────────┴──────────┴──────────┴──────────┘

  Total Generation:  2973.1 ms
  Total Audio:        55.06 s  (1321560 samples @ 24000 Hz)
  Overall RT:        18.52x
  TTFB warm-start:    116.4 ms (1st chunk inference)
  TTFB cold-start:    295.2 ms (load + prepare + 1st chunk)
  → 18.5x faster than real-time
```

**Hardware:** NVIDIA GeForce RTX 4090, **Backend:** Vulkan, **Threads:** 16

```
  Backend:     Vulkan
  Voice:       af_heart
  Threads:     16
  Sample Rate: 24000 Hz
  Sentences:   10 → 15 chunk(s) (941 tokens)

  Model load time:  157.8 ms
  Prepare time:       6.8 ms (chunking + phonemization)

  ┌──────┬────────┬────────────┬──────────┬──────────┬──────────┐
  │ Chunk│ Tokens │  Gen Time  │ Duration │    RT    │ Samples  │
  ├──────┼────────┼────────────┼──────────┼──────────┼──────────┤
  │     1│       4│     70.8ms │     0.91s│    12.78x│     21720│
  │     2│      77│    202.9ms │     4.16s│    20.51x│     99840│
  │     3│      54│    140.1ms │     2.99s│    21.35x│     71760│
  │     4│      67│    186.4ms │     4.37s│    23.42x│    104760│
  │     5│      75│    212.6ms │     4.31s│    20.27x│    103440│
  │     6│      76│    198.0ms │     4.24s│    21.39x│    101640│
  │     7│      55│    152.3ms │     3.34s│    21.93x│     80160│
  │     8│      76│    209.7ms │     4.68s│    22.34x│    112440│
  │     9│      80│    198.9ms │     4.21s│    21.17x│    101040│
  │    10│      36│    103.7ms │     2.04s│    19.67x│     48960│
  │    11│      63│    174.4ms │     4.07s│    23.31x│     97560│
  │    12│      77│    201.0ms │     4.13s│    20.57x│     99240│
  │    13│      67│    177.2ms │     3.84s│    21.67x│     92160│
  │    14│      62│    170.8ms │     3.67s│    21.46x│     87960│
  │    15│      72│    200.4ms │     4.14s│    20.69x│     99480│
  └──────┴────────┴────────────┴──────────┴──────────┴──────────┘

  Total Generation:  2599.2 ms
  Total Audio:        55.09 s  (1322160 samples @ 24000 Hz)
  Overall RT:        21.20x
  TTFB warm-start:     70.8 ms (1st chunk inference)
  TTFB cold-start:    235.5 ms (load + prepare + 1st chunk)
  → 21.2x faster than real-time
```

## License

Licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Kokoro](https://github.com/hexgrad/kokoro): Text-to-speech model
- [sanoTTS](https://github.com/Ampixa/sanoTTS) by Ampixa: compact text-to-speech models and runtime
- [ggml](https://github.com/ggerganov/ggml): Tensor library for ML inference
- [espeak-ng](https://github.com/espeak-ng/espeak-ng): Speech synthesis engine for phonemization

See [THIRD_PARTY.md](THIRD_PARTY.md) for the licence notices these require.
