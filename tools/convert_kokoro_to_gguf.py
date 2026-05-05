#!/usr/bin/env python3
"""
Convert hexgrad Kokoro PyTorch weights to the kokopop GGUF layout.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from kokoro import KModel

# ─────────────────────────────────────────────────────────────────────────────
# Suppress third-party warnings we cannot fix (kokoro package)
# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    message="dropout.*num_layers",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*weight_norm.*deprecated.*",
    category=FutureWarning,
)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy ML dependencies (loaded on demand)
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Model defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_REPO = "hexgrad/Kokoro-82M"
DEFAULT_VOICES = "af_heart,ff_siwis"
MODEL_NAME = "kokoro-v1_0.pth"

# ─────────────────────────────────────────────────────────────────────────────
# GGUF type constants
# ─────────────────────────────────────────────────────────────────────────────

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# ─────────────────────────────────────────────────────────────────────────────
# GGML type constants
# ─────────────────────────────────────────────────────────────────────────────

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14

KOKOPOP_GGUF_VERSION = 3

# ─────────────────────────────────────────────────────────────────────────────
# Quantization mapping tables
# ─────────────────────────────────────────────────────────────────────────────

_QUANT_UNIFORM_MAP: dict[str, int] = {
    "f16": GGML_TYPE_F16,
    "f32": GGML_TYPE_F32,
    "q5_k": GGML_TYPE_Q5_K,
    "q6_k": GGML_TYPE_Q6_K,
    "q8_0": GGML_TYPE_Q8_0,
}

# Block size for each GGML quant type (from ggml_type_traits)
_GGML_BLOCK_SIZES: dict[int, int] = {
    GGML_TYPE_Q5_K: 256,
    GGML_TYPE_Q6_K: 256,
    GGML_TYPE_Q8_0: 32,
    GGML_TYPE_F16: 1,
    GGML_TYPE_F32: 1,
}

# Patterns indicating a Conv1d weight that must stay F16 at runtime
_CONV1D_PATTERNS: tuple[str, ...] = (
    ".convs1.",
    ".convs2.",
    "noise_convs.",
    "conv_post.weight",
    ".conv1.weight",
    ".conv2.weight",
    ".conv1x1.weight",
    ".cnn.",
    "F0_conv.weight",
    "N_conv.weight",
    "asr_res.0.weight",
)


# ─────────────────────────────────────────────────────────────────────────────
# Alignment & utility helpers
# ─────────────────────────────────────────────────────────────────────────────


def align(value: int, alignment: int) -> int:
    """Return *value* rounded up to the next multiple of *alignment*."""
    return (value + alignment - 1) // alignment * alignment


def as_f32(value) -> np.ndarray:
    """Convert a tensor-like object to a contiguous F32 NumPy array."""
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().to(torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)


def weight_norm_value(module) -> np.ndarray:
    """Apply learned weight normalization to a module's weight."""
    weight_g = getattr(module, "weight_g")
    weight_v = getattr(module, "weight_v")
    return as_f32(torch._weight_norm(weight_v, weight_g, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Tensor type detection
# ─────────────────────────────────────────────────────────────────────────────


def is_conv1d_weight(name: str) -> bool:
    """Return True if *name* refers to a Conv1d weight that must stay F16."""
    if not name.endswith(".weight"):
        return False
    if ".ups." in name or ".pool.weight" in name:
        return False
    return any(pat in name for pat in _CONV1D_PATTERNS)


def runtime_tensor_type(name: str, data: np.ndarray, requested_type: int) -> int:
    """Select the runtime GGML type for a tensor.

    In GGUF v2+, Conv1d weights are always stored as F16 regardless of
    the requested type.
    """
    if KOKOPOP_GGUF_VERSION < 2:
        return requested_type
    if is_conv1d_weight(name):
        return GGML_TYPE_F16
    return GGML_TYPE_F32


def _is_type_aligned(ggml_type: int, innermost_dim: int) -> bool:
    """Check if *ggml_type* can represent a row of *innermost_dim* elements."""
    blck = _GGML_BLOCK_SIZES.get(ggml_type, 1)
    if blck <= 1:
        return True
    return innermost_dim % blck == 0


def _downgrade_type(ggml_type: int, innermost_dim: int) -> int:
    """Downgrade *ggml_type* if it cannot handle *innermost_dim*.

    Fallback chain:
        Q5_K / Q6_K → Q8_0 → F16
        Q8_0        → F16
    """
    if _is_type_aligned(ggml_type, innermost_dim):
        return ggml_type

    if ggml_type in (GGML_TYPE_Q5_K, GGML_TYPE_Q6_K):
        if _is_type_aligned(GGML_TYPE_Q8_0, innermost_dim):
            return GGML_TYPE_Q8_0
    if ggml_type == GGML_TYPE_Q8_0:
        pass
    return GGML_TYPE_F16


# ─────────────────────────────────────────────────────────────────────────────
# Mixed-quantization type resolution
# ─────────────────────────────────────────────────────────────────────────────


def get_tensor_quant_type(logical_name: str, ndim: int) -> int:
    """Determine the target GGML type for a tensor in mixed quantization.

    Returns the quantization type based on the tensor's logical name and
    dimensionality, following the mixed-quantization scheme defined in
    QUANTIZATION.md.
    """
    # 1D tensors (biases, norm params, Snake alpha) → always F32
    if ndim <= 1:
        return GGML_TYPE_F32

    # Voice embeddings → always F16
    if logical_name.startswith("kokopop.voice."):
        return GGML_TYPE_F16

    # Embedding lookup tables → F32
    if (".embeddings." in logical_name or ".embedding.weight" in logical_name) and \
       logical_name.endswith(".weight"):
        return GGML_TYPE_F32

    # Snake activation alpha params → F32 (binary ops)
    if ".alpha" in logical_name and ".weight" not in logical_name:
        return GGML_TYPE_F32

    # Conv1d weights → always F16 (required by runtime)
    if is_conv1d_weight(logical_name):
        return GGML_TYPE_F16

    # AdaIn FC gamma/beta weights in predictor F0/N → Q6_K
    if (".fc.gamma.weight" in logical_name or ".fc.beta.weight" in logical_name) and \
       (logical_name.startswith("kokopop.predictor.F0.") or
        logical_name.startswith("kokopop.predictor.N.")):
        return GGML_TYPE_Q6_K

    # Vocoder conv_post → F16
    if "conv_post" in logical_name:
        return GGML_TYPE_F16

    # Harmonic source merge → F16
    if "m_source" in logical_name:
        return GGML_TYPE_F16

    # Vocoder generator upsampling → Q8_0
    if logical_name.startswith("kokopop.decoder.generator.ups."):
        return GGML_TYPE_Q8_0

    # Vocoder residual blocks → Q8_0
    if ".generator.resblocks." in logical_name or ".generator.noise_res." in logical_name:
        return GGML_TYPE_Q8_0

    # ALBERT / BERT → Q5_K (majority), Q6_K for FFN output projection
    if logical_name.startswith("kokopop.albert.") or \
       logical_name.startswith("kokopop.bert_encoder."):
        if "ffn_output" in logical_name:
            return GGML_TYPE_Q6_K
        return GGML_TYPE_Q5_K

    # Text encoder → Q5_K
    if logical_name.startswith("kokopop.text_encoder."):
        return GGML_TYPE_Q5_K

    # Predictor → Q5_K
    if logical_name.startswith("kokopop.predictor."):
        return GGML_TYPE_Q5_K

    # Decoder (encode + decode) → Q5_K
    if logical_name.startswith("kokopop.decoder."):
        return GGML_TYPE_Q5_K

    # Fallback
    return GGML_TYPE_F16


def resolve_tensor_type(writer, name: str, data: np.ndarray, base_ftype: int) -> int:
    """Resolve final GGML type for a tensor, respecting quantization scheme.

    Also enforces block-size alignment: GGML K-quant types require the
    innermost dimension to be a multiple of the block size.  If the
    preferred type can't represent the tensor shape, the type is
    downgraded (Q5_K/Q6_K → Q8_0 → F16).
    """
    scheme = writer.quant_scheme

    if scheme == "none":
        return runtime_tensor_type(name, data, base_ftype)

    if scheme == "mixed":
        target = get_tensor_quant_type(name, data.ndim)
    else:
        # Uniform quantization
        if data.ndim <= 1:
            return GGML_TYPE_F32
        if name.startswith("kokopop.voice."):
            return GGML_TYPE_F16
        if is_conv1d_weight(name):
            return GGML_TYPE_F16
        target = _QUANT_UNIFORM_MAP.get(scheme, GGML_TYPE_F16)

    # Enforce block-size alignment
    if data.ndim > 1:
        innermost = data.shape[-1]
        target = _downgrade_type(target, innermost)

    return target


# ─────────────────────────────────────────────────────────────────────────────
# libggml ctypes loading (for quantization)
# ─────────────────────────────────────────────────────────────────────────────

_ggml_lib = None


def _setup_ggml_ctypes(lib) -> None:
    """Configure ctypes signatures for ggml quantization API."""

    class GgmlTypeTraits(ctypes.Structure):
        _fields_ = [
            ("type_name", ctypes.c_char_p),
            ("blck_size", ctypes.c_int64),
            ("blck_size_interleave", ctypes.c_int64),
            ("type_size", ctypes.c_size_t),
            ("is_quantized", ctypes.c_bool),
            ("to_float", ctypes.c_void_p),
            ("from_float_ref", ctypes.c_void_p),
        ]

    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    lib.ggml_get_type_traits.restype = ctypes.POINTER(GgmlTypeTraits)

    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,       # type
        ctypes.c_void_p,    # src (float*)
        ctypes.c_void_p,    # dst (void*)
        ctypes.c_int64,     # start
        ctypes.c_int64,     # nrows
        ctypes.c_int64,     # n_per_row
        ctypes.c_void_p,    # imatrix (float* or NULL)
    ]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t


def _load_ggml_lib():
    """Load libggml via ctypes for quantization support.

    Returns the loaded library or *None* if it cannot be found.
    """
    global _ggml_lib
    if _ggml_lib is not None:
        return _ggml_lib

    candidates = [
        "build/_deps/ggml-build/src/libggml.dylib",
        "build/_deps/ggml-build/src/libggml.so",
        "build/_deps/ggml-build/src/ibggml.so.0",
    ]
    base = Path(__file__).resolve().parent.parent
    for candidate in candidates:
        path = base / candidate
        if path.exists():
            try:
                _ggml_lib = ctypes.CDLL(str(path))
                _setup_ggml_ctypes(_ggml_lib)
                return _ggml_lib
            except OSError:
                continue

    found = ctypes.util.find_library("ggml")
    if found:
        try:
            _ggml_lib = ctypes.CDLL(found)
            _setup_ggml_ctypes(_ggml_lib)
            return _ggml_lib
        except OSError:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Tensor data class
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TensorEntry:
    """A single tensor ready to be serialized into the GGUF blob."""

    name: str
    data: np.ndarray
    ggml_type: int
    offset: int = 0

    @property
    def dims(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.data.shape[::-1])

    def encoded(self) -> bytes:
        """Encode tensor data in the appropriate bytes format."""
        if self.ggml_type == GGML_TYPE_F16:
            return np.asarray(self.data, dtype=np.float16).tobytes(order="C")
        if self.ggml_type == GGML_TYPE_F32:
            return np.asarray(self.data, dtype=np.float32).tobytes(order="C")
        return self._encode_quantized()

    def _encode_quantized(self) -> bytes:
        """Quantize F32 data to the target GGML type via libggml.

        GGML K-quant types (Q5_K, Q6_K) require the innermost dimension
        (ne[0] in GGML) to be a multiple of the block size (256).
        Data is quantized row-by-row to match GGML's memory layout.
        """
        libggml = _load_ggml_lib()
        if libggml is None:
            logger.warning(
                "libggml not found, writing %s as F16 instead of type %s",
                self.name, self.ggml_type,
            )
            return np.asarray(self.data, dtype=np.float16).tobytes(order="C")

        traits = libggml.ggml_get_type_traits(self.ggml_type)
        blck_size = traits.contents.blck_size
        type_size = traits.contents.type_size

        # Innermost dimension = GGML ne[0] (last axis in numpy)
        n_per_row = self.data.shape[-1]
        nrows = int(self.data.size // n_per_row)

        if n_per_row % blck_size != 0:
            raise ValueError(
                f"{self.name}: innermost dim {n_per_row} not multiple of "
                f"block size {blck_size} for type {self.ggml_type}. "
                f"resolve_tensor_type should have downgraded the type."
            )

        n_elements = int(self.data.size)
        quant_size = (n_elements // blck_size) * type_size
        quant_buf = ctypes.create_string_buffer(quant_size)

        flat = self.data.flatten(order="C")
        src_arr = (ctypes.c_float * n_elements)(*flat)

        libggml.ggml_quantize_chunk(
            self.ggml_type,
            ctypes.cast(src_arr, ctypes.c_void_p),
            quant_buf,
            0,        # start
            nrows,
            n_per_row,
            None,     # imatrix
        )
        return quant_buf.raw


# ─────────────────────────────────────────────────────────────────────────────
# GGUF writer
# ─────────────────────────────────────────────────────────────────────────────


class GGUFWriter:
    """Incremental builder that serialises tensors and KV pairs to a GGUF file."""

    def __init__(self, path: Path, alignment: int = 32) -> None:
        self.path = path
        self.alignment = alignment
        self.kv: list[tuple[str, int, object]] = []
        self.tensors: list[TensorEntry] = []
        self.logical_names: list[str] = []
        self.physical_names: list[str] = []
        self.used_tensor_names: set[str] = set()
        self.quant_scheme: str = "none"

    # -- metadata helpers ---------------------------------------------------

    def add_u32(self, key: str, value: int) -> None:
        self.kv.append((key, GGUF_TYPE_UINT32, int(value)))

    def add_bool(self, key: str, value: bool) -> None:
        self.kv.append((key, GGUF_TYPE_BOOL, bool(value)))

    def add_string(self, key: str, value: str) -> None:
        self.kv.append((key, GGUF_TYPE_STRING, value))

    def add_string_array(self, key: str, values: Iterable[str]) -> None:
        self.kv.append((key, GGUF_TYPE_ARRAY, list(values)))

    # -- tensor helpers -----------------------------------------------------

    def add_tensor(self, name: str, data, ggml_type: int) -> None:
        """Queue a tensor for serialization with automatic type resolution."""
        arr = as_f32(data)
        final_type = resolve_tensor_type(self, name, arr, ggml_type)
        physical = self._physical_tensor_name(name)
        self.tensors.append(
            TensorEntry(
                name=physical,
                data=np.ascontiguousarray(arr),
                ggml_type=final_type,
            )
        )

    def _physical_tensor_name(self, logical: str) -> str:
        """Assign a physical name, using BLAKE2s hash for collisions."""
        encoded_len = len(logical.encode("utf-8"))
        if encoded_len < 64 and logical not in self.used_tensor_names:
            self.used_tensor_names.add(logical)
            if logical.startswith("kokopop.voice."):
                return logical
            self.logical_names.append(logical)
            self.physical_names.append(logical)
            return logical

        digest = hashlib.blake2s(logical.encode("utf-8"), digest_size=12).hexdigest()
        physical = "kt." + digest
        if physical in self.used_tensor_names:
            raise ValueError(f"tensor name hash collision for {logical}")
        self.used_tensor_names.add(physical)
        self.logical_names.append(logical)
        self.physical_names.append(physical)
        return physical

    # -- serialization internals --------------------------------------------

    @staticmethod
    def _string(value: str) -> bytes:
        data = value.encode("utf-8")
        return struct.pack("<Q", len(data)) + data

    def _kv_bytes(self) -> bytes:
        out = bytearray()
        for key, typ, value in self.kv:
            out += self._string(key)
            out += struct.pack("<I", typ)
            if typ == GGUF_TYPE_UINT32:
                out += struct.pack("<I", value)
            elif typ == GGUF_TYPE_BOOL:
                out += b"\x01" if value else b"\x00"
            elif typ == GGUF_TYPE_STRING:
                out += self._string(value)
            elif typ == GGUF_TYPE_ARRAY:
                out += struct.pack("<I", GGUF_TYPE_STRING)
                out += struct.pack("<Q", len(value))
                for item in (value if isinstance(value, list) else [value]):
                    out += self._string(item)
            else:
                raise ValueError(f"unsupported GGUF kv type {typ}")
        return bytes(out)

    def _tensor_info_bytes(self) -> bytes:
        out = bytearray()
        offset = 0
        for tensor in self.tensors:
            offset = align(offset, self.alignment)
            tensor.offset = offset
            payload = tensor.encoded()
            out += self._string(tensor.name)
            out += struct.pack("<I", len(tensor.dims))
            for dim in tensor.dims:
                out += struct.pack("<Q", dim)
            out += struct.pack("<I", tensor.ggml_type)
            out += struct.pack("<Q", tensor.offset)
            offset += len(payload)
        return bytes(out)

    # -- main write ---------------------------------------------------------

    def write(self) -> None:
        self.add_u32("general.alignment", self.alignment)
        self.add_string_array("kokopop.tensor.logical_names", self.logical_names)
        self.add_string_array("kokopop.tensor.physical_names", self.physical_names)

        header = b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.kv))
        kv = self._kv_bytes()
        infos = self._tensor_info_bytes()

        blob = bytearray(header + kv + infos)
        blob += b"\x00" * (align(len(blob), self.alignment) - len(blob))
        data_start = len(blob)

        for tensor in self.tensors:
            want = data_start + tensor.offset
            if len(blob) < want:
                blob += b"\x00" * (want - len(blob))
            blob += tensor.encoded()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(blob)


# ─────────────────────────────────────────────────────────────────────────────
# State-dict conversion helpers
# ─────────────────────────────────────────────────────────────────────────────


def split_lstm(name: str, tensor, writer: GGUFWriter, ftype: int) -> None:
    """Split an LSTM weight/bias tensor into 4 equal chunks."""
    data = as_f32(tensor)
    chunks = np.split(data, 4, axis=0)
    for idx, chunk in enumerate(chunks):
        writer.add_tensor(f"{name}.{idx}", chunk, ftype)


def add_adain_fc(prefix: str, weight, bias, writer: GGUFWriter, ftype: int) -> None:
    """Decompose AdaIn FC weight/bias into gamma and beta branches."""
    w0, w1 = np.split(as_f32(weight), 2, axis=0)
    b0, b1 = np.split(as_f32(bias), 2, axis=0)
    writer.add_tensor(prefix + ".gamma.weight", w0, ftype)
    writer.add_tensor(prefix + ".gamma.bias", b0, GGML_TYPE_F32)
    writer.add_tensor(prefix + ".beta.weight", w1, ftype)
    writer.add_tensor(prefix + ".beta.bias", b1, GGML_TYPE_F32)


def add_state_dict(
    prefix: str,
    state: dict[str, object],
    writer: GGUFWriter,
    ftype: int,
) -> None:
    """Process a plain state dict and add tensors to *writer*."""
    for name, tensor in sorted(state.items()):
        if name.endswith("weight_v"):
            continue
        key = f"{prefix}.{name}"
        if "lstm" in name and ("weight_" in name or "bias_" in name):
            split_lstm(key, tensor, writer, ftype)
            continue
        writer.add_tensor(
            key, tensor, ftype if tensor.ndim > 1 else GGML_TYPE_F32
        )


def add_regularized_modules(
    prefix: str, module, writer: GGUFWriter, ftype: int
) -> None:
    """Process a Module with weight-norm and AdaIn layers."""
    modules = dict(module.named_modules())
    for name, param in sorted(module.named_parameters()):
        if name.endswith("weight_v"):
            continue
        if name.endswith("weight_g"):
            base = name[: -len(".weight_g")]
            wn_module = modules[base]
            writer.add_tensor(
                f"{prefix}.{base}.weight",
                weight_norm_value(wn_module),
                ftype,
            )
            continue
        if name.endswith(".fc.weight"):
            base = name[: -len(".fc.weight")]
            add_adain_fc(
                f"{prefix}.{base}.fc",
                param,
                dict(module.named_parameters())[f"{base}.fc.bias"],
                writer,
                ftype,
            )
            continue
        if name.endswith(".fc.bias"):
            continue
        if "lstm" in name and ("weight_" in name or "bias_" in name):
            split_lstm(f"{prefix}.{name}", param, writer, ftype)
            continue
        writer.add_tensor(
            f"{prefix}.{name}", param, ftype if param.ndim > 1 else GGML_TYPE_F32
        )


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary builder
# ─────────────────────────────────────────────────────────────────────────────


def build_vocab(config: dict) -> list[str]:
    """Build the vocabulary list from the model config."""
    reverse = {int(v): k for k, v in config["vocab"].items()}
    max_id = max(reverse)
    return [""] + [reverse.get(i, "") for i in range(1, max_id + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def convert(args: argparse.Namespace) -> None:
    ftype = GGML_TYPE_F16 if args.ftype == "f16" else GGML_TYPE_F32
    config_path = hf_hub_download(repo_id=args.repo_id, filename="config.json")
    model_path = hf_hub_download(repo_id=args.repo_id, filename=MODEL_NAME)
    config = json.loads(Path(config_path).read_text())

    model = KModel(
        repo_id=args.repo_id,
        config=config_path,
        model=model_path,
    ).eval()
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]

    writer = GGUFWriter(Path(args.output))
    writer.quant_scheme = args.quant

    # Metadata
    writer.add_u32("kokopop.kokoro.version", KOKOPOP_GGUF_VERSION)
    writer.add_bool("kokopop.mock", False)
    writer.add_u32("kokopop.sample_rate", 24000)
    writer.add_string("kokopop.arch", "kokoro-82m")
    writer.add_string("kokopop.tensor_layout", "runtime-v2")
    writer.add_string("kokopop.source_repo", args.repo_id)
    writer.add_string("kokopop.quantization", args.quant)
    writer.add_string_array("tokenizer.ggml.tokens", build_vocab(config))
    writer.add_string_array("kokopop.voices", voices)

    # Model dimensions
    writer.add_u32("kokopop.context_length", int(config["plbert"]["max_position_embeddings"]))
    writer.add_u32("kokopop.hidden_dim", int(config["hidden_dim"]))
    writer.add_u32("kokopop.style_dim", int(config["style_dim"]))
    writer.add_u32("kokopop.max_dur", int(config["max_dur"]))
    writer.add_u32("kokopop.n_mels", int(config["n_mels"]))

    # Tensors
    add_state_dict("kokopop.albert", model.bert.state_dict(), writer, ftype)
    writer.add_tensor("kokopop.bert_encoder.weight", model.bert_encoder.weight, ftype)
    writer.add_tensor("kokopop.bert_encoder.bias", model.bert_encoder.bias, GGML_TYPE_F32)
    add_regularized_modules("kokopop.predictor", model.predictor, writer, ftype)
    add_regularized_modules("kokopop.text_encoder", model.text_encoder, writer, ftype)
    add_regularized_modules("kokopop.decoder", model.decoder, writer, ftype)

    for voice in voices:
        voice_path = hf_hub_download(
            repo_id=args.repo_id, filename=f"voices/{voice}.pt"
        )
        pack = torch.load(voice_path, weights_only=True).squeeze(1)
        writer.add_tensor(f"kokopop.voice.{voice}", pack, ftype)

    writer.write()
    print(f"wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Kokoro PyTorch weights to kokopop GGUF format.",
    )
    parser.add_argument("--output", required=True, help="output GGUF path")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--voices", default=DEFAULT_VOICES)
    parser.add_argument("--ftype", choices=("f16", "f32"), default="f16")
    parser.add_argument(
        "--quant", default="f16",
        choices=("none", "f16", "f32", "q5_k", "q6_k", "q8_0", "mixed"),
        help="Quantization scheme for the output GGUF",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
