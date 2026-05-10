# kokopop — Standalone Kokoro GGML Runtime

A standalone C++ library and toolkit for running [Kokoro](https://github.com/hexgrad/kokoro) text-to-speech models in GGUF format, with no Python dependency.

## Features

- **Zero dependencies beyond libespeak-ng and ggml** — no Python, no heavy ML frameworks
- **CPU inference** with configurable thread count
- **Metal GPU backend** (macOS) for accelerated inference (experimental)
- **Streaming API** for real-time audio generation
- **Chunked synthesis** for long-form text processing
- **WAV output** and optional direct playback (Core Audio on macOS)
- **Full C & C++ API** — usable from C, C++, Rust, Go, and other languages via FFI

## Quick Start

### Prerequisites

- CMake 3.24+
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
  --voices af_heart,ff_siwis,zf_xiaoxiao,im_nicola \
  --tier kokoro-md
```

> **Note:** The Kokoro PyTorch model is automatically downloaded from [Hugging Face](https://huggingface.co/hexgrad/Kokoro-82M) on first run.

Two tiers are available:

| Tier        | Description                          |
|-------------|--------------------------------------|
| `kokoro-md` | Balanced (Q5_K/Q6_K, ~default)       |
| `kokoro-lg` | Quality first (Q6_K/Q8_0)            |

### Usage

Synthesize text to a WAV file:

```bash
./build/kokopop_say \
  --model models/kokoro.gguf \
  --voice af_heart \
  --text "Hello, world!" \
  --out hello.wav
```

Synthesize and play directly (macOS):

```bash
./build/kokopop_say \
  --model models/kokoro.gguf \
  --voice af_heart \
  --text "Hello, world!" \
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
  --text "Hello, world!" \
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
./build/kokopop_stream \
  --model models/kokoro.gguf \
  --voice af_heart \
  --http \
  --port 8080
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--http` | — | Run in HTTP server mode |
| `--port N` | `8080` | Server port |
| `--bind ADDR` | `127.0.0.1` | Bind address |

Available endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/tts` | `POST` | Synthesize text to audio — PCM float32 stream or complete WAV file |
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
```

Request body fields:

| Field | Type | Description |
|---|---|---|
| `text` | string | **Required** — text to synthesize |
| `voice` | string | Override default voice (e.g., `ff_siwis`) |
| `speed` | float | Synthesis speed (default: from CLI `--speed`) |
| `mode` | string | `interactive` (default) or `long_form` |
| `format` | string | `pcm` (default) — raw float32 stream; `wav` — complete WAV file; `ogg` — Ogg/Opus stream |

#### Python client (`tools/tts_client.py`)

A minimal Python client is provided. It requires no third-party packages.

```bash
# Start the server first
./build/kokopop_stream \
  --model models/kokoro.gguf \
  --voice af_heart --http --port 8080

# Stream ogg/opus and play in real time (requires ffplay)
uv run python tools/tts_client.py \
  --file examples/english.txt \
  --format ogg \
  --prebuffer-chunks 3 \
  --chunk-target-min 20 \
  --chunk-target-max 50 \
  --voice af_bella | ffplay -i pipe:0

uv run python tools/tts_client.py \
  --file examples/mandarin.txt \
  --format ogg \
  --prebuffer-ms 2000 \
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
| `--mode` | `interactive` | `interactive` or `long_form` |
| `--format` | `pcm` | `pcm` (float32 stream), `wav` (complete file), or `ogg` (Ogg/Opus stream) |
| `--out` | — | Output file; if omitted, raw bytes go to stdout |
| `--prebuffer-ms` | `3000` | Milliseconds of audio to buffer before playback starts (Ogg mode) |
| `--prebuffer-chunks` | `0` | Number of TTS synthesis chunks to buffer before playback (Ogg mode; overrides `--prebuffer-ms` when > 0) |
| `--prebuffer-gap-ms` | `250` | Idle gap in ms used to detect boundaries between TTS chunks |
| `--chunk-target-min` | preset | Minimum tokens per synthesis chunk |
| `--chunk-target-max` | preset | Target maximum tokens per synthesis chunk |
| `--chunk-soft-max` | preset | Soft token limit before forced split |
| `--chunk-hard-max` | preset | Hard token limit per chunk |
| `--chunk-first-max` | preset | Token limit for the first chunk (lower = faster TTFB) |
| `--chunk-sentence-pause` | preset | Pause between sentences in ms |
| `--chunk-paragraph-pause` | preset | Pause between paragraphs in ms |
| `--chunk-crossfade` | preset | Crossfade between chunks in ms |

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
  backend/   — CPU and Metal backend implementations
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
  --model models/kokoro-md.gguf \
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
  Sentences:   10 → 10 chunk(s) (947 tokens)

  Prepare time:       7.6 ms (chunking + phonemization)

  ┌──────┬────────┬────────────┬──────────┬──────────┬──────────┐
  │ Chunk│ Tokens │  Gen Time  │ Duration │    RT    │ Samples  │
  ├──────┼────────┼────────────┼──────────┼──────────┼──────────┤
  │     1│     183│   5813.0ms │    11.66s│     2.01x│    279840│
  │     2│      41│   1328.0ms │     2.79s│     2.10x│     66960│
  │     3│     180│   5788.3ms │    11.49s│     1.99x│    275760│
  │     4│      21│    886.0ms │     1.81s│     2.05x│     43560│
  │     5│     123│   3807.0ms │     7.69s│     2.02x│    184560│
  │     6│      76│   2284.3ms │     4.74s│     2.08x│    113760│
  │     7│      95│   2966.0ms │     6.07s│     2.04x│    145560│
  │     8│      92│   2784.8ms │     5.71s│     2.05x│    137160│
  │     9│      63│   1810.5ms │     3.79s│     2.09x│     90960│
  │    10│      73│   2046.5ms │     4.22s│     2.06x│    101280│
  └──────┴────────┴────────────┴──────────┴──────────┴──────────┘

  Total Generation:  29514.3 ms
  Total Audio:        59.98 s  (1439400 samples @ 24000 Hz)
  Overall RT:         2.03x
  TTFB (1st chunk):  5813.0 ms
  → 2.0x faster than real-time
```

**Hardware:** AMD Ryzen 9 7950X 16-Core Processor, **Backend:** CPU, **Threads:** 16

```
  Backend:     CPU
  Voice:       af_heart
  Threads:     16
  Sample Rate: 24000 Hz
  Sentences:   10 → 11 chunk(s) (876 tokens)

  Prepare time:       4.2 ms (chunking + phonemization)

  ┌──────┬────────┬────────────┬──────────┬──────────┬──────────┐
  │ Chunk│ Tokens │  Gen Time  │ Duration │    RT    │ Samples  │
  ├──────┼────────┼────────────┼──────────┼──────────┼──────────┤
  │     1│      85│   1273.4ms │     5.61s│     4.41x│    134640│
  │     2│      25│    428.4ms │     2.06s│     4.82x│     49560│
  │     3│     113│   1655.3ms │     7.26s│     4.39x│    174360│
  │     4│      86│   1167.3ms │     5.21s│     4.47x│    125160│
  │     5│     123│   1761.8ms │     7.64s│     4.34x│    183360│
  │     6│      76│   1084.3ms │     4.74s│     4.37x│    113760│
  │     7│     100│   1382.9ms │     6.19s│     4.48x│    148560│
  │     8│      40│    577.4ms │     2.59s│     4.49x│     62160│
  │     9│      92│   1319.6ms │     5.69s│     4.31x│    136560│
  │    10│      63│    864.2ms │     3.84s│     4.44x│     92160│
  │    11│      73│    944.3ms │     4.14s│     4.39x│     99480│
  └──────┴────────┴────────────┴──────────┴──────────┴──────────┘

  Total Generation:  12458.9 ms
  Total Audio:        54.99 s  (1319760 samples @ 24000 Hz)
  Overall RT:         4.41x
  TTFB (1st chunk):  1273.4 ms
  → 4.4x faster than real-time
```

## License

Licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Kokoro](https://github.com/hexgrad/kokoro) — Text-to-speech model
- [ggml](https://github.com/ggerganov/ggml) — Tensor library for ML inference
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) — Speech synthesis engine for phonemization
