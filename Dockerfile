# ──────────────────────────────────────────────
# Stage 1 – CPU build
# ──────────────────────────────────────────────
FROM debian:stable-slim AS builder-cpu

# Build dependencies (README: CMake 3.24+, C++17, espeak-ng, opus/ogg for HTTP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    ca-certificates \
    python3 \
    espeak-ng \
    libespeak-ng-dev \
    libogg-dev \
    libopus-dev \
    libopusfile-dev \
    libopusenc-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy project source
COPY . .

# Compile
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DKOKOPOP_BUILD_TESTS=OFF \
    -DKOKOPOP_BUILD_BENCH=OFF \
    && cmake --build build --parallel "$(nproc)"

# Bundle runtime artifacts produced by the CMake build. Depending on CMake
# cache/options, kokopop may be linked against libkokopop.so instead of a
# static archive, so keep generated shared libraries with the executable.
RUN mkdir -p /app/runtime/bin /app/runtime/lib \
    && cp /app/build/kokopop_stream /app/runtime/bin/ \
    && find /app/build -type f -name '*.so*' -exec cp -a {} /app/runtime/lib/ \;

# ──────────────────────────────────────────────
# Stage 1b – CUDA build
# ──────────────────────────────────────────────
FROM nvidia/cuda:13.2.1-devel-ubuntu24.04 AS builder-cuda

# Build dependencies (README: CMake 3.24+, C++17, espeak-ng, opus/ogg for HTTP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    ca-certificates \
    python3 \
    espeak-ng \
    libespeak-ng-dev \
    libogg-dev \
    libopus-dev \
    libopusfile-dev \
    libopusenc-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project source
COPY . .

# Compile with CUDA enabled. CMake forwards this to ggml via GGML_CUDA.
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DKOKOPOP_BUILD_TESTS=OFF \
    -DKOKOPOP_BUILD_BENCH=OFF \
    -DKOKOPOP_ENABLE_CUDA=ON \
    && cmake --build build --parallel "$(nproc)"

# Bundle runtime artifacts produced by the CMake build.
RUN mkdir -p /app/runtime/bin /app/runtime/lib \
    && cp /app/build/kokopop_stream /app/runtime/bin/ \
    && find /app/build -type f -name '*.so*' -exec cp -a {} /app/runtime/lib/ \;

# ──────────────────────────────────────────────
# Stage 2 – Export model
# ──────────────────────────────────────────────
FROM builder-cpu AS model-export

# Voices to embed — override at build time:
#   docker build --build-arg VOICES="af_heart,bf_emma,zf_xiaoxiao" .
ARG VOICES="af_heart,ff_siwis,zf_xiaoxiao,im_nicola"
ARG TIER="kokoro-md"

RUN uv sync --no-dev && \
    uv run python3 tools/convert_kokoro_to_gguf.py \
    --output models/kokoro.gguf \
    --voices "${VOICES}" \
    --tier "${TIER}" \
    && rm -rf .venv /root/.cache/uv

# ──────────────────────────────────────────────
# Stage 3 – CPU runtime
# ──────────────────────────────────────────────
FROM debian:stable-slim AS runtime-cpu

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    espeak-ng \
    libgomp1 \
    libogg0 \
    libopus0 \
    libopusfile0 \
    libopusenc0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the full CMake build tree for runtime/debug artifacts, plus register any
# generated shared libraries in the dynamic linker path.
COPY --from=builder-cpu /app/build/ /app/build/
COPY --from=builder-cpu /app/runtime/bin/kokopop_stream /app/kokopop_stream
COPY --from=builder-cpu /app/runtime/lib/ /usr/local/lib/
RUN ldconfig

# Copy model (built with selected voices)
COPY --from=model-export /app/models/kokoro.gguf /app/models/kokoro.gguf

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/kokopop_stream"]
CMD [ \
    "--model", "models/kokoro.gguf", \
    "--http", \
    "--port", "8080", \
    "--mode", "interactive", \
    "--voice", "af_heart" \
]

# ──────────────────────────────────────────────
# Stage 4 – CUDA runtime
# ──────────────────────────────────────────────
FROM nvidia/cuda:13.2.1-runtime-ubuntu24.04 AS runtime-cuda

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    espeak-ng \
    libgomp1 \
    libogg0 \
    libopus0 \
    libopusfile0 \
    libopusenc0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the full CMake build tree for runtime/debug artifacts, plus register any
# generated shared libraries in the dynamic linker path.
COPY --from=builder-cuda /app/build/ /app/build/
COPY --from=builder-cuda /app/runtime/bin/kokopop_stream /app/kokopop_stream
COPY --from=builder-cuda /app/runtime/lib/ /usr/local/lib/
RUN ldconfig

# Copy model (built with selected voices)
COPY --from=model-export /app/models/kokoro.gguf /app/models/kokoro.gguf

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/kokopop_stream"]
CMD [ \
    "--model", "models/kokoro.gguf", \
    "--http", \
    "--port", "8080", \
    "--mode", "adaptative", \
    "--voice", "af_heart" \
]

# Default image: CPU.
FROM runtime-cpu AS runtime
