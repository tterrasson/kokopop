# Building kokopop Python

The Python package is isolated in this directory and builds the C/C++ runtime
from the parent repository.

## Native Build

```bash
uv build ./python
uv pip install ./python
```

For development:

```bash
uv run --project python pytest
```

The default Python package build is CPU-only for the Kokopop runtime. GGML may
still compile platform helper code internally, but `KOKOPOP_ENABLE_METAL`,
`KOKOPOP_ENABLE_CUDA`, and `KOKOPOP_ENABLE_VULKAN` are off unless passed
explicitly.

No backend is auto-detected, even on platforms where the SDK is present (e.g.
Metal on macOS, Vulkan SDK on Linux). This is intentional: the chosen backend
determines runtime dependencies of the resulting wheel, so it must be an
explicit decision by whoever runs the build — either via
`--config-setting=cmake.define.KOKOPOP_ENABLE_<BACKEND>=ON` or via the matching
environment variable (see below).

## Build Options

Pass CMake options through `--config-setting=cmake.define.<NAME>=<VALUE>`:

```bash
uv build ./python \
  --config-setting=cmake.define.KOKOPOP_ENABLE_METAL=ON
```

Common options:

| Option | Default | Notes |
|---|---:|---|
| `KOKOPOP_ENABLE_METAL` | `OFF` | macOS Metal backend. Requires Apple platforms and Xcode SDKs. |
| `KOKOPOP_ENABLE_CUDA` | `OFF` | NVIDIA CUDA backend. Requires CUDA toolkit/compiler support. |
| `KOKOPOP_ENABLE_VULKAN` | `OFF` | Vulkan backend. Requires Vulkan SDK plus SPIR-V headers. |
| `KOKOPOP_ENABLE_OPUS` | `ON` | Ogg/Opus encoder if `opus`, `ogg`, and `libopusenc` are found. |
| `CMAKE_BUILD_TYPE` | `Release` | Use `Debug`, `RelWithDebInfo`, or `Release`. |

Examples:

```bash
# macOS Metal wheel
uv build ./python \
  --config-setting=cmake.define.KOKOPOP_ENABLE_METAL=ON

# Vulkan wheel
uv build ./python \
  --config-setting=cmake.define.KOKOPOP_ENABLE_VULKAN=ON

# CUDA wheel
uv build ./python \
  --config-setting=cmake.define.KOKOPOP_ENABLE_CUDA=ON

# Disable Ogg/Opus support
uv build ./python \
  --config-setting=cmake.define.KOKOPOP_ENABLE_OPUS=OFF
```

## macOS Architectures

Build for the host architecture:

```bash
uv build ./python
```

Build for Apple Silicon explicitly:

```bash
ARCHFLAGS="-arch arm64" uv build ./python \
  --config-setting=cmake.define.CMAKE_OSX_ARCHITECTURES=arm64
```

Build for Intel macOS on an Intel-capable toolchain:

```bash
ARCHFLAGS="-arch x86_64" uv build ./python \
  --config-setting=cmake.define.CMAKE_OSX_ARCHITECTURES=x86_64
```

Build a universal2 wheel:

```bash
ARCHFLAGS="-arch arm64 -arch x86_64" uv build ./python \
  --config-setting=cmake.define.CMAKE_OSX_ARCHITECTURES="arm64;x86_64"
```

Notes:

- Metal is only available on macOS builds.
- CUDA is not supported on current macOS toolchains.
- Universal2 builds require all linked libraries to be available for both
  `arm64` and `x86_64`.

## Linux Architectures

Native Linux build:

```bash
uv build ./python
```

For manylinux/auditwheel production wheels, build inside a manylinux container
for each target architecture. For example, on x86_64:

```bash
python -m pip install build uv
uv build ./python
auditwheel repair python/dist/*.whl
```

For aarch64, use an aarch64 runner/container or QEMU-enabled manylinux image.
Set a CMake toolchain only when cross-compiling:

```bash
uv build ./python \
  --config-setting=cmake.define.CMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake \
  --config-setting=cmake.define.CMAKE_SYSTEM_PROCESSOR=aarch64
```

Vulkan builds need SDK headers and libraries available to the target sysroot.
CUDA builds need a matching CUDA toolkit for the target architecture.

## Windows

Build with the active MSVC environment:

```powershell
uv build .\python
```

CUDA builds require the CUDA toolkit and an MSVC version supported by that CUDA
release:

```powershell
uv build .\python --config-setting=cmake.define.KOKOPOP_ENABLE_CUDA=ON
```

The exported C API already marks symbols correctly for Windows shared builds;
the Python extension links the runtime through CMake.

## Installing From Source

```bash
pip install ./python
```

With backend options, either via `--config-settings`:

```bash
pip install ./python \
  --config-settings=cmake.define.KOKOPOP_ENABLE_VULKAN=ON
```

…or via environment variables (handy for `pip install kokopop` from PyPI, where
passing config settings is awkward):

```bash
KOKOPOP_ENABLE_CUDA=ON pip install kokopop
KOKOPOP_ENABLE_METAL=ON pip install kokopop
KOKOPOP_ENABLE_VULKAN=ON pip install kokopop
```

Supported variables: `KOKOPOP_ENABLE_METAL`, `KOKOPOP_ENABLE_CUDA`,
`KOKOPOP_ENABLE_VULKAN`, `KOKOPOP_ENABLE_OPUS`.

Prerequisites on the user's machine: a C++17 compiler, CMake ≥ 3.24, and the
backend SDK (CUDA toolkit, Vulkan SDK, or Xcode for Metal).

Use a clean environment (`pip install --no-cache-dir --force-reinstall`) when
switching architecture or GPU backend flags so the wheel is rebuilt from
scratch.
