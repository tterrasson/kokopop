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
#
# Default set — all languages except Japanese and Hindi:
#   American English – Female: af_heart af_alloy af_aoede af_bella af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky
#   American English – Male:   am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa
#   British English – Female:  bf_alice bf_emma bf_isabella bf_lily
#   British English – Male:    bm_daniel bm_fable bm_george bm_lewis
#   Mandarin Chinese – Female: zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi
#   Mandarin Chinese – Male:   zm_yunjian zm_yunxi zm_yunxia zm_yunyang
#   Spanish – Female:          ef_dora
#   Spanish – Male:            em_alex em_santa
#   French – Female:           ff_siwis
#   Italian – Female:          if_sara
#   Italian – Male:            im_nicola
#   Brazilian Portuguese – F:  pf_dora
#   Brazilian Portuguese – M:  pm_alex pm_santa
ARG VOICES="af_heart,af_alloy,af_aoede,af_bella,af_jessica,af_kore,af_nicole,af_nova,af_river,af_sarah,af_sky,am_adam,am_echo,am_eric,am_fenrir,am_liam,am_michael,am_onyx,am_puck,am_santa,bf_alice,bf_emma,bf_isabella,bf_lily,bm_daniel,bm_fable,bm_george,bm_lewis,zf_xiaobei,zf_xiaoni,zf_xiaoxiao,zf_xiaoyi,zm_yunjian,zm_yunxi,zm_yunxia,zm_yunyang,ef_dora,em_alex,em_santa,ff_siwis,if_sara,im_nicola,pf_dora,pm_alex,pm_santa"

RUN uv sync --no-dev && \
    uv run python3 tools/convert_kokoro_to_gguf.py \
    --output models/kokoro.gguf \
    --voices "${VOICES}" \
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
    "--bind", "0.0.0.0", \
    "--port", "8080", \
    "--mode", "adaptative", \
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
    "--bind", "0.0.0.0", \
    "--port", "8080", \
    "--mode", "adaptative", \
    "--voice", "af_heart" \
]

# Default image: CPU.
FROM runtime-cpu AS runtime
