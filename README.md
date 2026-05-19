# kokopop — Standalone Kokoro GGML Runtime

A standalone C++ library and toolkit for running [Kokoro](https://github.com/hexgrad/kokoro) text-to-speech models in GGUF format, with no Python dependency.

## Features

- **Zero dependencies beyond libespeak-ng and ggml** — no Python, no heavy ML frameworks
- **CPU inference** with configurable thread count
- **Metal GPU backend** (macOS) for accelerated inference
- **CUDA backend** (Linux/Windows) for accelerated inference on NVIDIA GPUs
- **Vulkan backend** (Linux/Windows/macOS via MoltenVK) for portable GPU acceleration
- **Streaming API** for real-time audio generation
- **Chunked synthesis** for long-form text processing
- **WAV output** and optional direct playback (Core Audio on macOS)
- **Full C & C++ API** — usable from C, C++, Rust, Go, and other languages via FFI

📊 See [Benchmarks](#benchmarks)

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

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKOKOPOP_ENABLE_VULKAN=ON
cmake --build build
```

On macOS, install the Vulkan SDK plus SPIR-V headers so CMake can find Vulkan, `glslc`, MoltenVK, and `spirv/unified1/spirv.hpp`:

```bash
brew install vulkan-sdk vulkan-headers spirv-headers
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
  -DKOKOPOP_ENABLE_VULKAN=ON \
  -DKOKOPOP_VULKAN_VALIDATE=ON
cmake --build build
```

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

### Export Kokoro model to gguf format

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

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
layers — both eventually destabilise the duration head or saturate the
vocoder on some inputs. F16 stays bit-stable across all backends.

### Usage

Synthesize text to a WAV file:

```bash
./build/kokopop_say \
  --model models/kokoro.gguf \
  --voice af_heart \
  --text "Hello world." \
  --out hello.wav
```

Synthesize and play directly (macOS):

```bash
./build/kokopop_say \
  --model models/kokoro.gguf \
  --voice af_heart \
  --text "Hello world." \
  --play
```

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

### Audio playback

The `kokopop_play` tool reads raw audio from stdin and plays it via Core Audio (macOS) or outputs PCM (other platforms):

```bash
# Pipe audio from kokopop_say directly to playback
./build/kokopop_say \
  --model models/kokoro.gguf \
  --voice af_heart \
  --text "Hello, world!" \
  --play
```

See `kokopop_play --help` for format options (pcm-f32, pcm-s16, wav).

### Streaming mode

The `kokopop_stream` tool supports two operating modes: **STDIO** (default) and **HTTP server** (async, event-driven).

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

#### HTTP server mode (async)

Start an async, event-driven HTTP server for TTS synthesis. Uses `poll()` for non-blocking I/O with a `SynthesisScheduler` for round-robin chunk interleaving across concurrent requests:

```bash
./build/kokopop_stream --model models/kokoro.gguf --http --port 8080

# Unload model after 10 minutes of inactivity (saves memory, reloads on next request)
./build/kokopop_stream --model models/kokoro.gguf --http --port 8080 --idle-unload 10
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--http` | — | Run in HTTP server mode |
| `--port N` | `8080` | Server port |
| `--bind ADDR` | `127.0.0.1` | Bind address |
| `--idle-unload N` | disabled | Unload model after N minutes of inactivity, reload on next request |

Available endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/tts` | `POST` | Synthesize text to audio — PCM float32 stream, Ogg/Opus stream, or complete WAV file |
| `/health` | `GET` | Server health check |
| `/voices` | `GET` | List voices embedded in the GGUF model |

Example requests:

```bash
# Stream raw PCM float32 (default) — chunked transfer encoding
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

# Stream Ogg/Opus
curl -X POST http://localhost:8080/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world", "voice": "af_heart", "format": "ogg"}' \
  -o output.ogg
```

Request body fields:

| Field | Type | Description |
|---|---|---|
| `text` | string | **Required** — text to synthesize |
| `voice` | string | Override default voice (e.g., `ff_siwis`) |
| `speed` | float | Synthesis speed (default: from CLI `--speed`) |
| `mode` | string | `adaptative` (default) or `long_form` |
| `format` | string | `pcm` (default) — raw float32 stream; `wav` — complete WAV file; `ogg` — Ogg/Opus stream |

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
| `--out` | — | Output file; if omitted, raw bytes go to stdout |
| `--prebuffer-chunks` | `0` | Server-side Ogg synthesis chunks to buffer before playback |

## Docker

A multi-stage [Dockerfile](Dockerfile) is provided for building and running kokopop without local dependencies.
The default image is CPU-only. A separate CUDA target builds against NVIDIA CUDA and uses
`nvidia/cuda:13.2.1-runtime-ubuntu24.04` for the runtime image.

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

Override the default HTTP server command to use `kokopop_say`-like synthesis via the stream tool:

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

```c
#include "kokopop.h"

kokopop_model *model = NULL;
kokopop_audio audio = {};

kokopop_model_load("models/kokoro.gguf", NULL, &model);
kokopop_synthesize_text(model, "Hello, world!", "ff_siwis", 1.0f, &audio);
kokopop_write_wav("output.wav", &audio);

kokopop_audio_free(&audio);
kokopop_model_free(model);
```

## Project Structure

```
include/     — Public headers (kokopop.h)
src/         — Source code
  core/      — Error handling, string utilities, WAV I/O
  backend/   — CPU, Metal, CUDA, and Vulkan backend implementations
  inference/ — Kokoro graph operations and audio utilities
  synthesis/ — Phonemizer, text chunking, G2P (zh_g2p, pinyin), and main synthesis pipeline
  audio/     — Audio post-processing
  streaming/ — Streaming generation support
  playback/  — Audio playback (stdout, Core Audio on macOS)
tools/       — CLI tools
  kokopop_say    — Synthesize text/phonemes to WAV or play directly
  kokopop_stream — STDIO streaming (stdin → stdout) and async HTTP server
  kokopop_play   — Play raw audio from stdin
tests/       — Unit and integration tests
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
| `KOKOPOP_BUILD_TESTS` | `ON`    | Build unit tests                 |
| `KOKOPOP_BUILD_TOOLS` | `ON`    | Build CLI tools                  |
| `KOKOPOP_ENABLE_METAL`| `OFF`   | Enable Metal GPU backend (macOS) |
| `KOKOPOP_ENABLE_CUDA` | `OFF`   | Enable CUDA backend (NVIDIA GPUs) |
| `KOKOPOP_ENABLE_VULKAN` | `OFF` | Enable Vulkan GPU backend |
| `KOKOPOP_VULKAN_VALIDATE` | `OFF` | Enable Vulkan validation layers in ggml |
| `KOKOPOP_VULKAN_DEBUG` | `OFF` | Enable Vulkan debug output in ggml |
| `KOKOPOP_VULKAN_MEMORY_DEBUG` | `OFF` | Enable Vulkan memory debug output in ggml |
| `KOKOPOP_VULKAN_SHADER_DEBUG_INFO` | `OFF` | Build Vulkan shaders with debug info |
| `KOKOPOP_BUILD_BENCH` | `OFF`   | Build benchmarks (requires model)|

## Testing

```bash
ctest --test-dir build --output-on-failure
```

## Benchmarks

Build and run benchmarks (requires a Kokoro GGUF model):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKOKOPOP_BUILD_BENCH=ON
cmake --build build
./build/kokopop_bench --model models/kokoro.gguf
```

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
  Sentences:   10 → 9 chunk(s) (852 tokens)

  Model load time:   31.3 ms
  Prepare time:       3.5 ms (chunking + phonemization)

  ┌──────┬────────┬────────────┬──────────┬──────────┬──────────┐
  │ Chunk│ Tokens │  Gen Time  │ Duration │    RT    │ Samples  │
  ├──────┼────────┼────────────┼──────────┼──────────┼──────────┤
  │     1│     178│   3263.4ms │    11.76s│     3.60x│    282240│
  │     2│      24│    506.8ms │     2.22s│     4.38x│     53280│
  │     3│     125│   2093.9ms │     7.79s│     3.72x│    186960│
  │     4│      36│    586.5ms │     2.52s│     4.29x│     60403│
  │     5│     226│   4149.1ms │    15.06s│     3.63x│    361560│
  │     6│      39│    665.0ms │     2.66s│     4.00x│     63840│
  │     7│      90│   1500.8ms │     5.95s│     3.96x│    142680│
  │     8│      62│    960.2ms │     3.77s│     3.92x│     90360│
  │     9│      72│   1095.9ms │     4.17s│     3.81x│    100080│
  └──────┴────────┴────────────┴──────────┴──────────┴──────────┘

  Total Generation:  14821.4 ms
  Total Audio:        55.89 s  (1341403 samples @ 24000 Hz)
  Overall RT:         3.77x
  TTFB warm-start:   3263.4 ms (1st chunk inference)
  TTFB cold-start:   3298.2 ms (load + prepare + 1st chunk)
  → 3.8x faster than real-time
```

**Hardware:** NVIDIA GeForce RTX 4090, **Backend:** CUDA, **Threads:** 16

```
  Backend:     CUDA
  Voice:       af_heart
  Threads:     16
  Sample Rate: 24000 Hz
  Sentences:   10 → 9 chunk(s) (852 tokens)

  Model load time:  156.5 ms
  Prepare time:       3.5 ms (chunking + phonemization)

  ┌──────┬────────┬────────────┬──────────┬──────────┬──────────┐
  │ Chunk│ Tokens │  Gen Time  │ Duration │    RT    │ Samples  │
  ├──────┼────────┼────────────┼──────────┼──────────┼──────────┤
  │     1│     178│    856.1ms │    11.69s│    13.65x│    280440│
  │     2│      24│     93.8ms │     2.04s│    21.76x│     48960│
  │     3│     125│    440.0ms │     7.79s│    17.70x│    186960│
  │     4│      36│    121.3ms │     2.44s│    20.11x│     58560│
  │     5│     226│   1055.1ms │    14.99s│    14.21x│    359760│
  │     6│      39│    129.4ms │     2.49s│    19.24x│     59760│
  │     7│      90│    310.6ms │     5.74s│    18.48x│    137760│
  │     8│      62│    204.8ms │     3.81s│    18.63x│     91560│
  │     9│      72│    229.0ms │     4.14s│    18.10x│     99480│
  └──────┴────────┴────────────┴──────────┴──────────┴──────────┘

  Total Generation:  3440.1 ms
  Total Audio:        55.14 s  (1323240 samples @ 24000 Hz)
  Overall RT:        16.03x
  TTFB warm-start:    856.1 ms (1st chunk inference)
  TTFB cold-start:   1016.2 ms (load + prepare + 1st chunk)
  → 16.0x faster than real-time
```

## License

Licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Kokoro](https://github.com/hexgrad/kokoro) — Text-to-speech model
- [ggml](https://github.com/ggerganov/ggml) — Tensor library for ML inference
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) — Speech synthesis engine for phonemization
