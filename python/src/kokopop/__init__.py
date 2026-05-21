from __future__ import annotations

from ._native import (
    Audio,
    AudioChunk,
    AudioEncoder,
    KokopopError,
    Model,
    SynthesisSession,
)


class Backend:
    AUTO = "auto"
    CPU = "cpu"
    METAL = "metal"
    CUDA = "cuda"
    VULKAN = "vulkan"


class Mode:
    ADAPTATIVE = "adaptative"
    LONG_FORM = "long_form"


class AudioFormat:
    PCM_F32LE = "pcm_f32le"
    WAV_PCM16 = "wav"
    OGG_OPUS = "ogg_opus"


__all__ = [
    "Audio",
    "AudioChunk",
    "AudioEncoder",
    "AudioFormat",
    "Backend",
    "KokopopError",
    "Mode",
    "Model",
    "SynthesisSession",
]
