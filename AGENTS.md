Standalone C++ runtime for neural text-to-speech models in GGUF format. Two
architectures: Kokoro (82M) and sanoTTS (0.3M to 1.5M, Piperlite and Vocos
decoders, mixed sample rates in one file).

## Tech Stack

- **C++17** / **C11** — core library and CLI tools
- **CMake 3.24+** — build system (fetches `ggml` and `doctest`)

## Project Structure

```
include/          Public header (kokopop.h — C/C++ API)
src/
  core/           Error handling, UTF-8, string replace, WAV I/O
  backend/        CPU + Metal / CUDA / Vulkan / OpenCL GPU inference backends
  model/          GGUF model loading
  arch/kokoro/    Kokoro graph ops, audio utilities, KokoroArch
  arch/sanotts/   sanoTTS frontend, decoders, tokenizer, deterministic noise
  synthesis/      Phonemizer, text chunking/splitting, G2P (zh_g2p), main synth pipeline
  audio/          Audio post-processing
  streaming/      Streaming generation support
  playback/       Stdout + Core Audio (macOS) playback
tools/
  kokopop_say              Synthesize text → WAV or play directly
  kokopop_stream           STDIO streaming (stdin → stdout) + async HTTP server
  kokopop_play             Play raw audio from stdin
  kokopop_probe            Inspect a GGUF and dump each inference stage
  kokopop_rt               Per-chunk real-time factor breakdown
  convert_kokoro_to_gguf.py  Export Kokoro PyTorch model → GGUF
  convert_sanotts_to_gguf.py Export sanoTTS voice packs → multi-voice GGUF
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
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `KOKOPOP_BUILD_TESTS` | `OFF` | Build unit tests |
| `KOKOPOP_BUILD_TOOLS` | `OFF` | Build CLI tools |
| `KOKOPOP_ENABLE_METAL` | `OFF` | Metal GPU backend (macOS) |
| `KOKOPOP_ENABLE_CUDA` | `OFF` | CUDA backend (NVIDIA GPUs) |
| `KOKOPOP_ENABLE_VULKAN` | `OFF` | Vulkan GPU backend (requires SPIR-V headers) |
| `KOKOPOP_ENABLE_OPENCL` | `OFF` | OpenCL GPU backend (Adreno / Android) |
| `KOKOPOP_OPENCL_PROFILING` | `OFF` | OpenCL profiling in ggml (debug only) |
| `KOKOPOP_OPENCL_TARGET_VERSION` | `300` | OpenCL version ggml targets (`200` on older Adreno) |
| `KOKOPOP_BUILD_BENCH` | `OFF` | Build benchmarks |
| `KOKOPOP_INSTALL` | top-level | Generate install/export rules (`find_package(kokopop)` → `kokopop::kokopop`) |

## Run

### CLI Tools (after building)

```bash
# Synthesize text to WAV
./build/kokopop_say --model models/kokoro-md.gguf --voice af_heart --text "Hello!" --out hello.wav

# Same, with a sanoTTS voice. The WAV gets that voice's rate, which a
# multi-voice pack does not share across voices.
./build/kokopop_say --model models/sanotts-en.gguf --voice heart --text "Hello!" --out hello.wav

# Inspect a GGUF: arch, voices, frontend, decoder, rates, durations
./build/kokopop_probe --model models/sanotts-en.gguf --voice heart --text "Hello!"

# STDIO streaming (JSON on stdin → audio on stdout)
echo '{"text": "Hello!"}' | ./build/kokopop_stream --model models/kokoro-md.gguf --voice af_heart --mode long_form

# HTTP server mode (async)
./build/kokopop_stream --model models/kokoro-md.gguf --voice af_heart --http --port 8080

# /tts — stream raw PCM float32 (default)
curl -X POST http://localhost:8080/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello!", "voice": "af_heart"}' -o out.raw

# /tts — receive a complete WAV file
curl -X POST http://localhost:8080/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello!", "voice": "af_heart", "format": "wav"}' -o out.wav
```

### Tests

```bash
ctest --test-dir build --output-on-failure
```

### Benchmarks

```bash
./build/kokopop_bench --model models/kokoro.gguf
./build/kokopop_bench --model models/sanotts-en.gguf
```

The benchmark reads the architecture from the file and uses that
architecture's preset for voice, text and repeat count.

## Python / uv

All Python work uses **uv**

```bash
# Convert a Kokoro model to GGUF format
uv run python tools/convert_kokoro_to_gguf.py \
  --output models/kokoro.gguf \
  --voices af_heart,ff_siwis,zf_xiaoni,im_nicola

# Convert sanoTTS voices to a single multi-voice GGUF (mixed sample rates)
uv run python tools/convert_sanotts_to_gguf.py \
  --output models/sanotts-en.gguf \
  --voices heart,heartnano,amy,kristin
```