# ──────────────────────────────────────────────
# Stage 1 – Build
# ──────────────────────────────────────────────
FROM debian:stable-slim AS builder

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

# ──────────────────────────────────────────────
# Stage 2 – Export model
# ──────────────────────────────────────────────
FROM builder AS model-export

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
# Stage 3 – Runtime
# ──────────────────────────────────────────────
FROM debian:stable-slim AS runtime

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    espeak-ng \
    libogg0 \
    libopus0 \
    libopusfile0 \
    libopusenc0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary
COPY --from=builder /app/build/kokopop_stream /app/kokopop_stream

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