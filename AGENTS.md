# kokopop — Agent Cheat Sheet

Standalone C++ runtime for Kokoro text-to-speech models in GGUF format.

## Tech Stack

- **C++17** / **C11** — core library and CLI tools
- **CMake 3.24+** — build system (fetches `ggml` and `doctest` as submodules)

## Project Structure

```
include/          Public header (kokopop.h — C/C++ API)
src/
  core/           Error handling, UTF-8, string replace, WAV I/O
  backend/        CPU + Metal GPU inference backends
  model/          GGUF model loading
  inference/      Kokoro graph ops, audio utilities
  synthesis/      Phonemizer, text chunking/splitting, G2P (zh_g2p), main synth pipeline
  audio/          Audio post-processing
  streaming/      Streaming generation support
  playback/       Stdout + Core Audio (macOS) playback
tools/
  kokopop_say              Synthesize text → WAV or play directly
  kokopop_stdio_stream     JSON-streamed TTS (stdin → stdout)
  kokopop_play             Play raw audio from stdin
  convert_kokoro_to_gguf.py  Export Kokoro PyTorch model → GGUF
  export_pinyin_dict.py      Generate pinyin dictionary for zh_g2p
tests/            Unit & integration tests (doctest)
models/           Pre-exported GGUF model files
build/            CMake output directory
```

## Build

```bash
# Default (CPU-only)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# With Metal GPU (macOS)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKOKOPOP_ENABLE_METAL=ON
cmake --build build
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `KOKOPOP_BUILD_TESTS` | `ON` | Build unit tests |
| `KOKOPOP_BUILD_TOOLS` | `ON` | Build CLI tools |
| `KOKOPOP_ENABLE_METAL` | `OFF` | Metal GPU backend (macOS) |
| `KOKOPOP_BUILD_BENCH` | `OFF` | Build benchmarks |

## Run

### CLI Tools (after building)

```bash
# Synthesize text to WAV
./build/kokopop_say --model models/kokoro-md.gguf --voice af_heart --text "Hello!" --out hello.wav

# Streaming (JSON on stdin → audio on stdout)
echo '{"text": "Hello!"}' | ./build/kokopop_stdio_stream --model models/kokoro-md.gguf --voice af_heart --mode long_form
```

### Tests

```bash
ctest --test-dir build --output-on-failure
```

### Benchmarks

```bash
./build/kokopop_bench --model models/kokoro-md.gguf
```

## Python / uv

All Python work uses **uv**. Never run bare `python` commands.

```bash
# Convert a Kokoro model to GGUF format
uv run python tools/convert_kokoro_to_gguf.py \
  --output models/kokoro.gguf \
  --voices af_heart,ff_siwis,zf_xiaoni,im_nicola \
  --tier kokoro-md
```