from collections.abc import Iterator
from types import TracebackType
from typing import Any, TypeVar

_SynthesisSessionT = TypeVar("_SynthesisSessionT", bound="SynthesisSession")
_AudioEncoderT = TypeVar("_AudioEncoderT", bound="AudioEncoder")

class KokopopError(RuntimeError):
    code: int

class Audio:
    sample_rate: int
    n_samples: int
    def to_wav_bytes(self) -> bytes: ...
    def write_wav(self, path: str) -> None: ...

class AudioChunk(Audio):
    chunk_index: int
    is_final: bool

class Model:
    sample_rate: int
    def __init__(self, path: str, *, n_threads: int = 0, backend: str = "auto") -> None: ...
    def synthesize(self, text: str, *, voice: str = "", speed: float = 1.0) -> Audio: ...
    def synthesize_phonemes(self, phonemes: str, *, voice: str = "", speed: float = 1.0) -> Audio: ...
    def stream(
        self,
        text: str,
        *,
        voice: str = "",
        speed: float = 1.0,
        mode: str = "adaptative",
        max_chunks: int = 1,
        **chunk_options: Any,
    ) -> Iterator[AudioChunk]: ...

class SynthesisSession:
    def __init__(self, model: Model, *, voice: str = "", speed: float = 1.0, mode: str = "adaptative", **chunk_options: Any) -> None: ...
    def __enter__(self: _SynthesisSessionT) -> _SynthesisSessionT: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def push_text(self, text: str) -> None: ...
    def finish_input(self) -> None: ...
    def next(self, max_chunks: int = 1) -> list[AudioChunk]: ...
    def close(self) -> None: ...

class AudioEncoder:
    def __init__(self, format: str = "pcm_f32le", *, sample_rate: int = 24000, ogg_prebuffer_chunks: int = 0) -> None: ...
    def __enter__(self: _AudioEncoderT) -> _AudioEncoderT: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def start(self) -> bytes: ...
    def push(self, samples: object, *, is_final: bool = False) -> bytes: ...
    def finish(self, success: bool = True) -> bytes: ...
    def close(self) -> None: ...

class Backend:
    AUTO: str
    CPU: str
    METAL: str
    CUDA: str
    VULKAN: str

class Mode:
    ADAPTATIVE: str
    LONG_FORM: str

class AudioFormat:
    PCM_F32LE: str
    WAV_PCM16: str
    OGG_OPUS: str
