# kokopop — Standalone Kokoro GGML Runtime

A standalone C++ library and toolkit for running [Kokoro](https://github.com/hexgrad/kokoro) text-to-speech models in GGUF format, with no Python dependency.

## Features

- **Zero dependencies beyond libespeak-ng and ggml** — no Python, no heavy ML frameworks
- **CPU inference** with configurable thread count
- **Metal GPU backend** (macOS) for accelerated inference
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

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Build with Metal GPU support (macOS)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKOKOPOP_ENABLE_METAL=ON
cmake --build build
```

### Export Kokoro model to gguf format

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv run python tools/convert_kokoro_to_gguf.py \
  --output models/kokoro.gguf \
  --voices af_heart,ff_siwis,zf_xiaoni,im_nicola \
  --tier kokoro-md
```

Three tiers are available:

| Tier        | Description                          |
|-------------|--------------------------------------|
| `kokoro-sm` | Small / aggressive (Q4_K majority)   |
| `kokoro-md` | Balanced (Q5_K/Q6_K, ~default)       |
| `kokoro-lg` | Quality first (Q6_K/Q8_0)            |

### Usage

Synthesize text to a WAV file:

```bash
./kokopop_say \
  --model models/kokoro.gguf \
  --voice af_heart \
  --text "Hello, world!" \
  --out hello.wav
```

Synthesize and play directly (macOS):

```bash
./kokopop_say \
  --model models/kokoro.gguf \
  --voice af_heart \
  --text "Hello, world!" \
  --play
```

Generate audio from phonemes:

```bash
./kokopop_say \
  --model models/kokoro.gguf \
  --voice af_heart \
  --phonemes "həˈloʊ wɜrld" \
  --out hello.wav
```

Adjust generation speed:

```bash
./kokopop_say \
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
./kokopop_say \
  --model models/kokoro.gguf \
  --voice af_heart \
  --text "Hello, world!" \
  --play
```

See `kokopop_play --help` for format options (pcm-f32, pcm-s16, wav).

### Streaming mode

The `kokopop_stdio_stream` tool reads JSON commands from stdin and streams TTS audio to stdout:

```bash
# Start interactive mode
./kokopop_stdio_stream \
  --model models/kokoro.gguf \
  --voice af_heart \
  --mode interactive \
  --play

# Feed commands via stdin
echo '{"text": "Hello from piped input"}' | \
  ./kokopop_stdio_stream \
    --model models/kokoro.gguf \
    --voice af_heart \
    --mode long_form
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
  backend/   — CPU and Metal backend implementations
  inference/ — Kokoro graph operations and audio utilities
  synthesis/ — Phonemizer, text chunking, G2P (zh_g2p, pinyin), and main synthesis pipeline
  audio/     — Audio post-processing
  streaming/ — Streaming generation support
  playback/  — Audio playback (stdout, Core Audio on macOS)
tools/       — CLI tools
  kokopop_say          — Synthesize text/phonemes to WAV or play directly
  kokopop_stdio_stream — JSON-streamed TTS (stdin → stdout)
  kokopop_play         — Play raw audio from stdin
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

## License

Licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Kokoro](https://github.com/hexgrad/kokoro) — Text-to-speech model
- [ggml](https://github.com/ggerganov/ggml) — Tensor library for ML inference
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) — Speech synthesis engine for phonemization
