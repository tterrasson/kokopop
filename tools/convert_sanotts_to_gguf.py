#!/usr/bin/env python3
"""Convert sanoTTS voices to a kokopop GGUF.

Two input families, packaged very differently:

  piperlite  a self-describing voice pack: manifest.json (format
             "roota.raw-fp16.v1") + weights.fp16.bin + piper-phoneme-config.json.
             Every tensor's name, shape, dtype, offset and SHA-256 is in the
             manifest, so the conversion is a validated copy.

  vocos      two flat, nameless byte blobs (front/decoder) plus a *generated C
             header* of offsets, `nano_q8_meta.h`. Nothing in the blob says
             where a tensor starts: the header is the only description, so the
             converter reconstructs the layout from the dimensions and checks
             every reconstructed size against the next offset and the totals.

One GGUF can carry several voices, at mixed sample rates. The metadata and
tensor-name schema this writes, along with the validation rules, are deliberately
the upstream ones: sanoTTS refuses checkpoints
whose operators it has not verified rather than guessing at a layout, and that
is the right call.

Derived from the MIT-licensed parts of Ampixa/sanoTTS
(`pypkg/sanotts/{models,voicepack,frontend}.py`, `mcu/models/*/nano_q8_meta.h`),
read at revision 939d982b9faa54cbcf5d24cc878f5cd514b2646e. Weights and manifests
are downloaded from an immutable commit too (`--artifact-revision`), and the two
revisions are recorded separately in the GGUF. See THIRD_PARTY.md.

Usage:

    uv run python tools/convert_sanotts_to_gguf.py \\
        --output models/sanotts-en.gguf --voices amy,kristin,heart

    uv run python tools/convert_sanotts_to_gguf.py \\
        --output models/sanotts-amy.gguf \\
        --voice amy --voice-dir ~/.cache/sanotts/amy-en-1p46m

    uv run python tools/convert_sanotts_to_gguf.py \\
        --output /dev/null --voices heart --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import struct
import sys
import unicodedata
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("convert_sanotts")

SANOTTS_REPO = "ampixa/sanoTTS"

# Two revisions, recorded separately because they move independently.
#
# CODE is the revision this converter was read from and validated against: it
# says which upstream layout rules and phoneme contracts the code implements.
# ARTIFACT is the revision the weights and manifests are downloaded from. They
# happen to be the same commit today, but conflating them was a real bug —
# downloading from `main` while stamping a fixed revision into the provenance
# produces a different GGUF under the same recorded identity.
SANOTTS_CODE_REVISION = "939d982b9faa54cbcf5d24cc878f5cd514b2646e"
SANOTTS_DEFAULT_ARTIFACT_REVISION = SANOTTS_CODE_REVISION

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def check_immutable_revision(revision: str) -> str:
    """Refuse anything a branch could move under us."""
    if not _COMMIT_RE.match(revision):
        raise ConversionError(
            f"artifact revision {revision!r} is not an immutable commit sha; pass the "
            "40-character hex commit a Hugging Face branch currently points at, so "
            "the recorded provenance keeps matching the bytes that were converted"
        )
    return revision


def hf_url(relative: str, revision: str) -> str:
    return f"https://huggingface.co/{SANOTTS_REPO}/resolve/{revision}/{relative}"

# Same location the upstream Python package caches into, so a user who has
# already run `sanotts` pays nothing here.
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sanotts"

SANOTTS_GGUF_VERSION = 1

# ---------------------------------------------------------------------------
# GGUF writing
#
# Deliberately separate from convert_kokoro_to_gguf.py's writer: this one needs
# u32/i32 arrays and no quantization, that one needs K-quants and no arrays.
# Merging them would mean one writer carrying both sets of concerns.
# ---------------------------------------------------------------------------

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

GGML_TYPE_NAMES = {GGML_TYPE_F32: "F32", GGML_TYPE_F16: "F16"}


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class ConversionError(RuntimeError):
    """A validation failure. Always fatal: a warning here becomes bad audio."""


@dataclass
class Tensor:
    name: str
    data: np.ndarray
    ggml_type: int
    offset: int = 0

    @property
    def dims(self) -> tuple[int, ...]:
        # GGUF stores dimensions innermost-first; numpy is outermost-first.
        return tuple(int(v) for v in self.data.shape[::-1])

    @property
    def nbytes(self) -> int:
        return int(self.data.size) * (4 if self.ggml_type == GGML_TYPE_F32 else 2)

    def encoded(self) -> bytes:
        dtype = np.float32 if self.ggml_type == GGML_TYPE_F32 else np.float16
        return np.ascontiguousarray(self.data, dtype=dtype).tobytes(order="C")


class GgufWriter:
    def __init__(self, path: Path, alignment: int = 32) -> None:
        self.path = path
        self.alignment = alignment
        self.kv: list[tuple[str, int, Any]] = []
        self.tensors: list[Tensor] = []
        self.logical_names: list[str] = []
        self.physical_names: list[str] = []
        self._used: set[str] = set()

    # -- metadata ----------------------------------------------------------

    def add_u32(self, key: str, value: int) -> None:
        if not 0 <= int(value) < 2**32:
            raise ConversionError(f"{key}: {value} does not fit in u32")
        self.kv.append((key, GGUF_TYPE_UINT32, int(value)))

    def add_i32(self, key: str, value: int) -> None:
        self.kv.append((key, GGUF_TYPE_INT32, int(value)))

    def add_f32(self, key: str, value: float) -> None:
        self.kv.append((key, GGUF_TYPE_FLOAT32, float(value)))

    def add_bool(self, key: str, value: bool) -> None:
        self.kv.append((key, GGUF_TYPE_BOOL, bool(value)))

    def add_string(self, key: str, value: str) -> None:
        self.kv.append((key, GGUF_TYPE_STRING, str(value)))

    def add_string_array(self, key: str, values: Iterable[str]) -> None:
        self.kv.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, list(values))))

    def add_u32_array(self, key: str, values: Iterable[int]) -> None:
        items = [int(v) for v in values]
        for v in items:
            if not 0 <= v < 2**32:
                raise ConversionError(f"{key}: {v} does not fit in u32")
        self.kv.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_UINT32, items)))

    # -- tensors -----------------------------------------------------------

    def add_tensor(self, logical: str, data: np.ndarray, ggml_type: int) -> None:
        arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
        if arr.size == 0:
            raise ConversionError(f"{logical}: empty tensor")
        if not np.all(np.isfinite(arr)):
            raise ConversionError(f"{logical}: contains NaN or infinity")
        if ggml_type == GGML_TYPE_F16:
            peak = float(np.max(np.abs(arr)))
            if peak > 65504.0:
                raise ConversionError(
                    f"{logical}: peak magnitude {peak:g} overflows F16; "
                    "this tensor must be stored as F32"
                )
        physical = self._physical_name(logical)
        self.tensors.append(Tensor(name=physical, data=arr, ggml_type=ggml_type))

    def _physical_name(self, logical: str) -> str:
        """GGUF tensor names are capped at 64 bytes; hash the long ones.

        The runtime resolves logical names through
        `kokopop.tensor.{logical,physical}_names`, the same mechanism the
        Kokoro converter uses.
        """
        if len(logical.encode("utf-8")) < 64 and logical not in self._used:
            self._used.add(logical)
            self.logical_names.append(logical)
            self.physical_names.append(logical)
            return logical
        digest = hashlib.blake2s(logical.encode("utf-8"), digest_size=12).hexdigest()
        physical = "st." + digest
        if physical in self._used:
            raise ConversionError(f"tensor name hash collision for {logical}")
        self._used.add(physical)
        self.logical_names.append(logical)
        self.physical_names.append(physical)
        return physical

    # -- serialisation -----------------------------------------------------

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
            elif typ == GGUF_TYPE_INT32:
                out += struct.pack("<i", value)
            elif typ == GGUF_TYPE_FLOAT32:
                out += struct.pack("<f", value)
            elif typ == GGUF_TYPE_BOOL:
                out += b"\x01" if value else b"\x00"
            elif typ == GGUF_TYPE_STRING:
                out += self._string(value)
            elif typ == GGUF_TYPE_ARRAY:
                item_type, items = value
                out += struct.pack("<I", item_type)
                out += struct.pack("<Q", len(items))
                for item in items:
                    if item_type == GGUF_TYPE_STRING:
                        out += self._string(item)
                    elif item_type == GGUF_TYPE_UINT32:
                        out += struct.pack("<I", item)
                    else:
                        raise ConversionError(f"unsupported array item type {item_type}")
            else:
                raise ConversionError(f"unsupported GGUF kv type {typ}")
        return bytes(out)

    def _tensor_info_bytes(self) -> bytes:
        out = bytearray()
        offset = 0
        for tensor in self.tensors:
            offset = align_up(offset, self.alignment)
            tensor.offset = offset
            out += self._string(tensor.name)
            out += struct.pack("<I", len(tensor.dims))
            for dim in tensor.dims:
                out += struct.pack("<Q", dim)
            out += struct.pack("<I", tensor.ggml_type)
            out += struct.pack("<Q", tensor.offset)
            offset += tensor.nbytes
        return bytes(out)

    def total_tensor_bytes(self) -> int:
        offset = 0
        for tensor in self.tensors:
            offset = align_up(offset, self.alignment) + tensor.nbytes
        return offset

    def write(self) -> None:
        self.add_u32("general.alignment", self.alignment)
        self.add_string_array("kokopop.tensor.logical_names", self.logical_names)
        self.add_string_array("kokopop.tensor.physical_names", self.physical_names)

        header = b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.kv))
        kv = self._kv_bytes()
        infos = self._tensor_info_bytes()

        blob = bytearray(header + kv + infos)
        blob += b"\x00" * (align_up(len(blob), self.alignment) - len(blob))
        data_start = len(blob)
        for tensor in self.tensors:
            want = data_start + tensor.offset
            if len(blob) < want:
                blob += b"\x00" * (want - len(blob))
            blob += tensor.encoded()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(blob)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def artifact_cache(cache_dir: Path, revision: str) -> Path:
    """Where artifacts of one revision live.

    The revision is part of the path: a blob cached from another revision is a
    different file, and silently reusing it is what would make the recorded
    provenance a lie.
    """
    return cache_dir / "hf" / revision


def fetch(relative: str, cache_dir: Path, revision: str) -> Path:
    """Download `relative` at `revision` from the sanoTTS repo, cached on disk."""
    dest = artifact_cache(cache_dir, revision) / relative
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = hf_url(relative, revision)
    logger.info("downloading %s", url)
    try:
        with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310
            payload = response.read()
    except OSError as exc:
        raise ConversionError(
            f"could not download {url}: {exc}. Point --voice-dir at a local copy instead."
        ) from exc
    dest.write_bytes(payload)
    return dest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The NFD table
#
# The Piper tokenizer decomposes its phoneme string to NFD before mapping code
# points to ids. kokopop's runtime has no Unicode dependency and cannot infer
# which precomposed characters exist, so the full canonical-decomposition table
# is generated here, once per model, from Python's Unicode database.
# ---------------------------------------------------------------------------


@dataclass
class NfdTable:
    unicode_version: str
    codepoints: list[int]
    offsets: list[int]
    values: list[int]
    # Code points with a non-zero canonical combining class, and their class.
    # NFD is decomposition *plus* canonical ordering, and the ordering needs
    # these: without them a sequence of two combining marks could come out in
    # the wrong order and map to different ids.
    ccc_codepoints: list[int] = field(default_factory=list)
    ccc_classes: list[int] = field(default_factory=list)

    @property
    def entries(self) -> int:
        return len(self.codepoints)


def build_nfd_table() -> NfdTable:
    """Every code point whose full canonical decomposition is not itself.

    Stored flattened: `codepoints[i]` decomposes to
    `values[offsets[i] : offsets[i + 1]]`. `offsets` has one extra trailing
    entry so the runtime needs no special case for the last row.
    """
    codepoints: list[int] = []
    offsets: list[int] = [0]
    values: list[int] = []
    ccc_codepoints: list[int] = []
    ccc_classes: list[int] = []
    for cp in range(0x110000):
        # Surrogates are not valid scalar values and normalize() rejects them.
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        combining = unicodedata.combining(ch)
        if combining:
            ccc_codepoints.append(cp)
            ccc_classes.append(combining)
        decomposed = unicodedata.normalize("NFD", ch)
        if decomposed == ch:
            continue
        codepoints.append(cp)
        values.extend(ord(c) for c in decomposed)
        offsets.append(len(values))
    return NfdTable(
        unicode_version=unicodedata.unidata_version,
        codepoints=codepoints,
        offsets=offsets,
        values=values,
        ccc_codepoints=ccc_codepoints,
        ccc_classes=ccc_classes,
    )


# ---------------------------------------------------------------------------
# Voice registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PiperliteVoice:
    name: str
    package: str
    espeak_voice: str  # filled from piper-phoneme-config.json


@dataclass(frozen=True)
class VocosVoice:
    name: str
    directory: str  # path within the HF repo
    front_blob: str
    dec_blob: str
    int8: bool


# Mirrors pypkg/sanotts/tables/voices.json plus the two vocos voices, which
# that table does not list because they use a different runtime.
PIPERLITE_PACKAGES = {
    "amy": "amy-en-1p46m",
    "amy-1p1m": "amy-en-1p1m",
    "amy-1p8m": "amy-en-1p8m",
    "hfc": "hfc-en-1p8m",
    "kristin": "kristin-en-1p4m",
    "vi": "vi-vais1000-1p46m",
    "id": "id-newstts-1p46m",
}

VOCOS_VOICES = {
    "heart": VocosVoice("heart", "heart", "front_f32.bin", "model_f32.bin", int8=False),
    "heartnano": VocosVoice(
        "heartnano", "heartnano", "front_q8.bin", "model_q8.bin", int8=True
    ),
}

# `heart-nano` is accepted on the command line; the GGUF only ever carries the
# canonical `heartnano`, so a model never has two names for one voice.
VOICE_ALIASES = {"heart-nano": "heartnano"}

# espeak-ng voice ids, needed because the vocos packs do not record one (their
# frontend is misaki's, always en-US).
VOCOS_ESPEAK_VOICE = "gmw/en-US"
VOCOS_NORMALIZATION_LANG = "a"

# Piper packs record a short espeak name ("en-us"); the runtime selects voices
# by espeak's own identifiers, so map the ones the shipped packs use.
ESPEAK_NAME_MAP = {
    "en-us": "gmw/en-US",
    "en-gb": "gmw/en",
    "en": "gmw/en-US",
    "vi": "aav/vi",
    "id": "poz/id",
    "hi": "inc/hi",
    "ne": "inc/ne",
    "zh": "sit/cmn",
    "cmn": "sit/cmn",
}


def resolve_espeak_voice(name: str) -> str:
    if name in ESPEAK_NAME_MAP:
        return ESPEAK_NAME_MAP[name]
    if "/" in name:
        # Already an espeak path such as "gmw/en-US".
        return name
    raise ConversionError(
        f"unknown espeak voice {name!r}; add it to ESPEAK_NAME_MAP with the "
        "espeak-ng identifier the runtime should select"
    )


# ---------------------------------------------------------------------------
# The piperlite path
# ---------------------------------------------------------------------------

CONV1D_3D_SUFFIXES = (
    ".net.0.weight",
    ".net.2.weight",
    ".conv1.weight",
    ".conv2.weight",
    "pre.weight",
    "post.weight",
    "in_conv.weight",
    "out_conv.weight",
)


@dataclass
class VoicePack:
    name: str
    directory: Path
    manifest: dict[str, Any]
    weights: bytes
    phoneme_config: dict[str, Any]
    phoneme_config_name: str = ""
    phoneme_config_sha256: str = ""

    def component(self, which: str) -> dict[str, Any]:
        comp = self.manifest.get("components", {}).get(which)
        if comp is None:
            raise ConversionError(f"{self.name}: manifest has no {which!r} component")
        return comp

    def tensors(self, which: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for entry in self.component(which)["tensors"]:
            name = entry["name"]
            shape = tuple(int(v) for v in entry["shape"])
            offset = int(entry["offset_bytes"])
            nbytes = int(entry["nbytes"])
            raw = self.weights[offset : offset + nbytes]
            if len(raw) != nbytes:
                raise ConversionError(
                    f"{self.name}.{which}.{name}: truncated weights blob "
                    f"(wanted {nbytes} bytes at {offset}, got {len(raw)})"
                )
            expected_sha = entry.get("sha256")
            if expected_sha and hashlib.sha256(raw).hexdigest() != expected_sha:
                raise ConversionError(
                    f"{self.name}.{which}.{name}: sha256 mismatch against the manifest"
                )
            dtype = entry["dtype"]
            if dtype == "float16":
                array = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            elif dtype == "float32":
                array = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            else:
                raise ConversionError(
                    f"{self.name}.{which}.{name}: unsupported dtype {dtype!r}"
                )
            if array.size != int(np.prod(shape)):
                raise ConversionError(
                    f"{self.name}.{which}.{name}: {array.size} values do not fill shape {shape}"
                )
            out[name] = array.reshape(shape)
        return out


def load_voice_pack(name: str, directory: Path) -> VoicePack:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ConversionError(f"{directory}: missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    fmt = manifest.get("format")
    if fmt != "roota.raw-fp16.v1":
        raise ConversionError(
            f"{directory}: unsupported manifest format {fmt!r}, expected 'roota.raw-fp16.v1'"
        )

    weights_path = directory / manifest["weights_file"]
    if not weights_path.is_file():
        raise ConversionError(f"{directory}: missing weights file {weights_path.name}")
    weights = weights_path.read_bytes()

    expected_size = int(manifest.get("weights_size_bytes", -1))
    if expected_size >= 0 and len(weights) != expected_size:
        raise ConversionError(
            f"{weights_path}: size {len(weights)} != manifest weights_size_bytes {expected_size}"
        )
    expected_sha = manifest.get("weights_sha256")
    if expected_sha:
        actual = hashlib.sha256(weights).hexdigest()
        if actual != expected_sha:
            raise ConversionError(
                f"{weights_path}: sha256 mismatch (manifest={expected_sha}, actual={actual}); "
                "the voice pack is corrupt or was tampered with"
            )

    included = (
        manifest.get("frontend", {}).get("included_config") or "piper-phoneme-config.json"
    )
    config_path = directory / included
    if not config_path.is_file():
        raise ConversionError(f"{directory}: missing phoneme config {included}")
    phoneme_config = json.loads(config_path.read_text(encoding="utf-8"))

    return VoicePack(
        name=name,
        directory=directory,
        manifest=manifest,
        weights=weights,
        phoneme_config=phoneme_config,
        phoneme_config_name=included,
        # The phoneme config is a conversion input like the weights are: it
        # fixes the espeak voice and the whole id table, so a GGUF that does not
        # record its hash cannot be audited back to what produced it.
        phoneme_config_sha256=sha256_file(config_path),
    )


@dataclass
class PhonemeTable:
    espeak_voice: str
    symbols: list[str]
    ids: list[int]
    pad_id: int
    bos_id: int
    eos_id: int


def parse_piper_phoneme_config(name: str, config: dict[str, Any]) -> PhonemeTable:
    """Validate a Piper `phoneme_id_map`, with upstream's exact requirements.

    Single-code-point keys, single-id values, and the `_`/`^`/`$` framing ids
    Piper hardcodes. Anything else means the tokenizer contract differs from
    the one implemented in the runtime.
    """
    phoneme_type = config.get("phoneme_type", "espeak")
    if phoneme_type not in (None, "espeak"):
        raise ConversionError(f"{name}: unsupported phoneme_type={phoneme_type!r}")

    espeak_voice = (config.get("espeak") or {}).get("voice")
    if not espeak_voice:
        raise ConversionError(f"{name}: phoneme config is missing espeak.voice")

    raw_map = config.get("phoneme_id_map")
    if not isinstance(raw_map, dict) or not raw_map:
        raise ConversionError(f"{name}: phoneme config has no phoneme_id_map")

    symbols: list[str] = []
    ids: list[int] = []
    seen_ids: set[int] = set()
    for key, value in raw_map.items():
        if len(key) != 1:
            raise ConversionError(f"{name}: multi-code-point map key {key!r}")
        if not isinstance(value, list) or len(value) != 1:
            raise ConversionError(f"{name}: multi-id map value {key!r} -> {value!r}")
        pid = int(value[0])
        if pid < 0:
            raise ConversionError(f"{name}: negative phoneme id for {key!r}")
        if pid in seen_ids:
            raise ConversionError(
                f"{name}: phoneme id {pid} is used by more than one symbol"
            )
        seen_ids.add(pid)
        symbols.append(key)
        ids.append(pid)

    for symbol, want in (("_", 0), ("^", 1), ("$", 2)):
        got = raw_map.get(symbol)
        got_id = int(got[0]) if isinstance(got, list) and got else None
        if got_id != want:
            raise ConversionError(
                f"{name}: framing symbol {symbol!r} maps to {got_id}, expected {want}"
            )

    return PhonemeTable(
        espeak_voice=str(espeak_voice),
        symbols=symbols,
        ids=ids,
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )


@dataclass
class VoiceTensor:
    """One tensor to emit, already in kokopop's on-disk layout."""

    suffix: str  # logical name under the voice prefix
    data: np.ndarray
    f32: bool = False


@dataclass
class VoiceBuild:
    """Everything one voice contributes to the GGUF."""

    name: str
    aliases: list[str]
    sample_rate: int
    length_scale: float
    espeak_voice: str
    normalization_lang: str
    frontend: str  # "piper" | "misaki"
    decoder: str  # "piperlite" | "vocos"
    max_tokens: int
    token_symbols: list[str]
    token_ids: list[int]
    bos_id: int
    eos_id: int
    pad_id: int  # -1 when the tokenizer interleaves nothing
    fallback_id: int  # -1 when an out-of-vocab id must be an error
    scalar_meta: dict[str, int] = field(default_factory=dict)
    float_meta: dict[str, float] = field(default_factory=dict)
    array_meta: dict[str, list[int]] = field(default_factory=dict)
    tensors: list[VoiceTensor] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)

    # Parameter count the source declares, or 0 when it declares none.
    declared_parameters: int = 0

    def element_count(self) -> int:
        return int(sum(t.data.size for t in self.tensors))


def flatten_conv1d(name: str, array: np.ndarray) -> np.ndarray:
    """[OC, IC, K] -> [OC, IC*K], kokopop's v4 kernel layout.

    The runtime consumes conv1d through im2col + mul_mat (or a direct CONV_2D
    where the backend prefers it), both of which want the kernel flattened.
    """
    if array.ndim != 3:
        raise ConversionError(f"{name}: expected a 3D conv1d kernel, got {array.shape}")
    oc, ic, k = array.shape
    return array.reshape(oc, ic * k)


def build_piperlite_voice(
    name: str, pack: VoicePack, aliases: Sequence[str]
) -> VoiceBuild:
    manifest = pack.manifest
    dur_cfg = pack.component("duration")["config"]
    ac_cfg = pack.component("acoustic")["config"]
    dec_cfg = pack.component("decoder")["config"]

    # -- refuse what the runtime does not implement -------------------------
    if str(dur_cfg.get("architecture")) != "duration_conv":
        raise ConversionError(
            f"{name}: duration architecture {dur_cfg.get('architecture')!r} is not supported "
            "(only 'duration_conv')"
        )
    if str(ac_cfg.get("architecture")) != "token_context":
        raise ConversionError(
            f"{name}: acoustic architecture {ac_cfg.get('architecture')!r} is not supported "
            "(only 'token_context')"
        )
    if str(dec_cfg.get("variant")) != "piperlite":
        raise ConversionError(
            f"{name}: decoder variant {dec_cfg.get('variant')!r} is not supported "
            "(only 'piperlite')"
        )
    if str(dec_cfg.get("activation") or "leaky_relu") != "leaky_relu":
        raise ConversionError(
            f"{name}: decoder activation {dec_cfg.get('activation')!r} is not supported "
            "(only 'leaky_relu')"
        )
    res_layers = int(dec_cfg.get("res_layers") or 1)
    if res_layers != 1:
        raise ConversionError(f"{name}: only res_layers=1 is supported, got {res_layers}")
    if float(dec_cfg.get("pre_tanh_repair_channels") or 0) > 0:
        raise ConversionError(
            f"{name}: pre_tanh_repair is not implemented (no shipped voice uses it)"
        )

    dur_tensors = pack.tensors("duration")
    ac_tensors = pack.tensors("acoustic")
    dec_tensors = pack.tensors("decoder")

    adapter_keys = sorted(k for k in ac_tensors if "adapter" in k)
    if adapter_keys:
        raise ConversionError(
            f"{name}: the acoustic checkpoint has an output adapter ({adapter_keys}); "
            "only the adapter-free token_context path is implemented"
        )

    for stage in range(3):
        branches = dec_cfg.get(f"stage{stage}_branches")
        if branches is not None and list(branches) != [0, 1, 2]:
            raise ConversionError(
                f"{name}: decoder stage {stage} uses branches {list(branches)}; "
                "only the full [0, 1, 2] bank is implemented"
            )

    table = parse_piper_phoneme_config(name, pack.phoneme_config)

    dur_vocab = int(dur_cfg["vocab_size"])
    dur_hidden = int(dur_cfg["hidden"])
    dur_depth = int(dur_cfg["depth"])
    dur_kernel = int(dur_cfg.get("kernel_size") or 5)
    ac_vocab = int(ac_cfg["vocab_size"])
    ac_hidden = int(ac_cfg["hidden"])
    ac_depth = int(ac_cfg["depth"])
    ac_token_depth = int(ac_cfg["token_depth"])
    ac_kernel = int(ac_cfg.get("kernel_size") or 5)
    ac_out = int(ac_cfg["out_channels"])
    channels = [int(c) for c in dec_cfg["channels"][:4]]

    build = VoiceBuild(
        name=name,
        aliases=list(aliases),
        sample_rate=int(manifest["sample_rate"]),
        length_scale=float(
            manifest.get("inference", {}).get("duration_length_scale", 1.0)
        ),
        espeak_voice=resolve_espeak_voice(table.espeak_voice),
        # The Piper frontend applies no misaki normalisation; the field is
        # carried anyway so every voice has one well-defined value.
        normalization_lang="a",
        frontend="piper",
        decoder="piperlite",
        max_tokens=int(dur_cfg["max_tokens"]),
        token_symbols=table.symbols,
        token_ids=table.ids,
        bos_id=table.bos_id,
        eos_id=table.eos_id,
        pad_id=table.pad_id,
        # A phoneme id from the (larger, shared) code-point table can exceed a
        # component's trained vocab. Upstream remaps those to schwa; the id is
        # taken from the voice's own table rather than hardcoded.
        fallback_id=_schwa_fallback_id(table),
    )

    build.scalar_meta.update(
        {
            "dur.vocab": dur_vocab,
            "dur.hidden": dur_hidden,
            "dur.depth": dur_depth,
            "dur.kernel": dur_kernel,
            "dur.max_tokens": int(dur_cfg["max_tokens"]),
            "dur.max_duration": int(dur_cfg["max_duration"]),
            "ac.vocab": ac_vocab,
            "ac.hidden": ac_hidden,
            "ac.token_depth": ac_token_depth,
            "ac.depth": ac_depth,
            "ac.kernel": ac_kernel,
            "ac.out_channels": ac_out,
            "dec.post_filter_channels": int(dec_cfg.get("post_filter_channels") or 0),
            "dec.post_filter_layers": int(dec_cfg.get("post_filter_layers") or 0),
            "dec.post_filter_kernel": int(dec_cfg.get("post_filter_kernel") or 9),
        }
    )
    build.float_meta["dec.post_filter_scale"] = float(
        dec_cfg.get("post_filter_scale") or 0.0
    )
    build.array_meta["dec.channels"] = channels
    for stage in range(3):
        build.array_meta[f"dec.branches.{stage}"] = [0, 1, 2]

    # -- duration ----------------------------------------------------------
    def take(tensors: dict[str, np.ndarray], key: str, where: str) -> np.ndarray:
        if key not in tensors:
            raise ConversionError(f"{name}: {where} is missing tensor {key!r}")
        return tensors[key]

    def expect(array: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
        if tuple(array.shape) != shape:
            raise ConversionError(
                f"{name}: {label} has shape {tuple(array.shape)}, expected {shape}"
            )
        return array

    add = build.tensors.append

    add(
        VoiceTensor(
            "dur.embedding.weight",
            expect(
                take(dur_tensors, "embedding.weight", "duration"),
                (dur_vocab, dur_hidden),
                "dur.embedding.weight",
            ),
            f32=True,
        )
    )
    add(
        VoiceTensor(
            "dur.input_proj.weight",
            flatten_conv1d(
                "dur.input_proj.weight",
                expect(
                    take(dur_tensors, "input_proj.weight", "duration"),
                    (dur_hidden, dur_hidden + 3, 1),
                    "dur.input_proj.weight",
                ),
            ),
        )
    )
    add(
        VoiceTensor(
            "dur.input_proj.bias",
            expect(
                take(dur_tensors, "input_proj.bias", "duration"),
                (dur_hidden,),
                "dur.input_proj.bias",
            ),
            f32=True,
        )
    )
    _add_residual_blocks(
        build, name, dur_tensors, "blocks", "dur.blocks", dur_depth, dur_hidden, dur_kernel
    )
    # dur.output stays F32 and its matmul runs on CPU: round(exp(x)) is a step
    # function, so a fp16 backend difference here changes the audio *length*
    # and makes cross-backend parity untestable.
    add(
        VoiceTensor(
            "dur.output.weight",
            flatten_conv1d(
                "dur.output.weight",
                expect(
                    take(dur_tensors, "output.weight", "duration"),
                    (1, dur_hidden, 1),
                    "dur.output.weight",
                ),
            ),
            f32=True,
        )
    )
    add(
        VoiceTensor(
            "dur.output.bias",
            expect(take(dur_tensors, "output.bias", "duration"), (1,), "dur.output.bias"),
            f32=True,
        )
    )

    # -- acoustic ----------------------------------------------------------
    add(
        VoiceTensor(
            "ac.embedding.weight",
            expect(
                take(ac_tensors, "embedding.weight", "acoustic"),
                (ac_vocab, ac_hidden),
                "ac.embedding.weight",
            ),
            f32=True,
        )
    )
    for src, dst, extra in (
        ("token_input_proj", "ac.token_input_proj", 2),
        ("frame_input_proj", "ac.frame_input_proj", 3),
    ):
        add(
            VoiceTensor(
                f"{dst}.weight",
                flatten_conv1d(
                    f"{dst}.weight",
                    expect(
                        take(ac_tensors, f"{src}.weight", "acoustic"),
                        (ac_hidden, ac_hidden + extra, 1),
                        f"{dst}.weight",
                    ),
                ),
            )
        )
        add(
            VoiceTensor(
                f"{dst}.bias",
                expect(
                    take(ac_tensors, f"{src}.bias", "acoustic"),
                    (ac_hidden,),
                    f"{dst}.bias",
                ),
                f32=True,
            )
        )
    _add_residual_blocks(
        build, name, ac_tensors, "token_blocks", "ac.token_blocks",
        ac_token_depth, ac_hidden, ac_kernel,
    )
    _add_residual_blocks(
        build, name, ac_tensors, "frame_blocks", "ac.frame_blocks",
        ac_depth, ac_hidden, ac_kernel,
    )
    add(
        VoiceTensor(
            "ac.output.weight",
            flatten_conv1d(
                "ac.output.weight",
                expect(
                    take(ac_tensors, "output.weight", "acoustic"),
                    (ac_out, ac_hidden, 1),
                    "ac.output.weight",
                ),
            ),
        )
    )
    add(
        VoiceTensor(
            "ac.output.bias",
            expect(take(ac_tensors, "output.bias", "acoustic"), (ac_out,), "ac.output.bias"),
            f32=True,
        )
    )

    # -- decoder -----------------------------------------------------------
    c0, c1, c2, c3 = channels
    pre = take(dec_tensors, "pre.weight", "decoder")
    if pre.ndim != 3 or pre.shape[0] != c0 or pre.shape[1] != ac_out:
        raise ConversionError(
            f"{name}: dec.pre.weight has shape {tuple(pre.shape)}, expected "
            f"({c0}, {ac_out}, K)"
        )
    add(VoiceTensor("dec.pre.weight", flatten_conv1d("dec.pre.weight", pre)))
    add(
        VoiceTensor(
            "dec.pre.bias",
            expect(take(dec_tensors, "pre.bias", "decoder"), (c0,), "dec.pre.bias"),
            f32=True,
        )
    )
    build.scalar_meta["dec.pre_kernel"] = int(pre.shape[2])

    stage_specs = [
        (c0, c1, 16, "up0", "res0.0"),
        (c1, c2, 16, "up1", "res1.0"),
        (c2, c3, 8, "up2", "res2.0"),
    ]
    bank_kernels = (3, 5, 7)
    for stage, (in_c, out_c, up_k, up_name, bank) in enumerate(stage_specs):
        up_w = expect(
            take(dec_tensors, f"{up_name}.weight", "decoder"),
            (in_c, out_c, up_k),
            f"dec.up{stage}.weight",
        )
        # ConvTranspose1d keeps its [IC, OC, K] shape: ggml_conv_transpose_1d
        # wants ne = [K, OC, IC], which is exactly this reversed.
        add(VoiceTensor(f"dec.up{stage}.weight", up_w))
        add(
            VoiceTensor(
                f"dec.up{stage}.bias",
                expect(
                    take(dec_tensors, f"{up_name}.bias", "decoder"),
                    (out_c,),
                    f"dec.up{stage}.bias",
                ),
                f32=True,
            )
        )
        for branch in range(3):
            k = bank_kernels[branch]
            for conv in (1, 2):
                w = expect(
                    take(dec_tensors, f"{bank}.blocks.{branch}.conv{conv}.weight", "decoder"),
                    (out_c, out_c, k),
                    f"dec.res{stage}.blocks.{branch}.conv{conv}.weight",
                )
                add(
                    VoiceTensor(
                        f"dec.res{stage}.blocks.{branch}.conv{conv}.weight",
                        flatten_conv1d("conv", w),
                    )
                )
                add(
                    VoiceTensor(
                        f"dec.res{stage}.blocks.{branch}.conv{conv}.bias",
                        expect(
                            take(
                                dec_tensors,
                                f"{bank}.blocks.{branch}.conv{conv}.bias",
                                "decoder",
                            ),
                            (out_c,),
                            "bias",
                        ),
                        f32=True,
                    )
                )

    post = take(dec_tensors, "post.weight", "decoder")
    if post.ndim != 3 or post.shape[0] != 1 or post.shape[1] != c3:
        raise ConversionError(
            f"{name}: dec.post.weight has shape {tuple(post.shape)}, expected (1, {c3}, K)"
        )
    add(VoiceTensor("dec.post.weight", flatten_conv1d("dec.post.weight", post)))
    add(
        VoiceTensor(
            "dec.post.bias",
            expect(take(dec_tensors, "post.bias", "decoder"), (1,), "dec.post.bias"),
            f32=True,
        )
    )
    build.scalar_meta["dec.post_kernel"] = int(post.shape[2])

    _add_post_filter(build, name, dec_tensors, dec_cfg)

    build.sources = {
        "package": str(manifest.get("package_name") or name),
        "weights_sha256": str(manifest.get("weights_sha256") or ""),
        "phoneme_config": pack.phoneme_config_name,
        "phoneme_config_sha256": pack.phoneme_config_sha256,
        "total_parameters": str(manifest.get("total_parameters") or ""),
    }
    build.declared_parameters = int(manifest.get("total_parameters") or 0)
    check_parameter_count(build)
    return build


def _schwa_fallback_id(table: PhonemeTable) -> int:
    """Upstream's out-of-vocab fallback: schwa, looked up in this voice's table.

    Upstream hardcodes 59 because that is schwa's id in the Piper tables it
    ships. Reading it from the table keeps the behaviour identical for those
    voices without baking in a number that is the em dash in the vocos
    vocabulary.
    """
    for symbol, pid in zip(table.symbols, table.ids):
        if symbol == "ə":  # LATIN SMALL LETTER SCHWA
            return int(pid)
    return 0


def _add_residual_blocks(
    build: VoiceBuild,
    voice: str,
    tensors: dict[str, np.ndarray],
    src_prefix: str,
    dst_prefix: str,
    depth: int,
    channels: int,
    kernel: int,
) -> None:
    for i in range(depth):
        scale_key = f"{src_prefix}.{i}.scale"
        if scale_key not in tensors:
            raise ConversionError(f"{voice}: missing tensor {scale_key!r}")
        scale = tensors[scale_key]
        if scale.size != 1:
            raise ConversionError(
                f"{voice}: {scale_key} has {scale.size} values, expected 1"
            )
        build.tensors.append(
            VoiceTensor(f"{dst_prefix}.{i}.scale", scale.reshape(1), f32=True)
        )
        for src, dst in (("net.0", "net0"), ("net.2", "net2")):
            w_key = f"{src_prefix}.{i}.{src}.weight"
            b_key = f"{src_prefix}.{i}.{src}.bias"
            if w_key not in tensors or b_key not in tensors:
                raise ConversionError(f"{voice}: missing {w_key!r} or {b_key!r}")
            w = tensors[w_key]
            if tuple(w.shape) != (channels, channels, kernel):
                raise ConversionError(
                    f"{voice}: {w_key} has shape {tuple(w.shape)}, expected "
                    f"({channels}, {channels}, {kernel})"
                )
            build.tensors.append(
                VoiceTensor(f"{dst_prefix}.{i}.{dst}.weight", flatten_conv1d(w_key, w))
            )
            b = tensors[b_key]
            if tuple(b.shape) != (channels,):
                raise ConversionError(
                    f"{voice}: {b_key} has shape {tuple(b.shape)}, expected ({channels},)"
                )
            build.tensors.append(
                VoiceTensor(f"{dst_prefix}.{i}.{dst}.bias", b, f32=True)
            )


def _post_filter_unit_kernel(
    voice: str, tensors: dict[str, np.ndarray], layers: int, channels: int
) -> int:
    """Kernel size of the post-filter units, read from the weights themselves.

    Every unit conv must agree, and it must be odd: the runtime emits them as
    "same"-padded convolutions, which is only well defined for an odd kernel.
    """
    kernels: set[int] = set()
    for layer in range(layers):
        for conv in (1, 2):
            key = f"post_filter.units.{layer}.conv{conv}.weight"
            if key not in tensors:
                raise ConversionError(
                    f"{voice}: the manifest enables the post-filter but {key!r} is missing"
                )
            array = tensors[key]
            if array.ndim != 3 or array.shape[0] != channels or array.shape[1] != channels:
                raise ConversionError(
                    f"{voice}: {key} has shape {tuple(array.shape)}, expected "
                    f"({channels}, {channels}, K)"
                )
            kernels.add(int(array.shape[2]))
    if len(kernels) != 1:
        raise ConversionError(
            f"{voice}: the post-filter units use mixed kernel sizes {sorted(kernels)}; "
            "the runtime emits one kernel per unit stack"
        )
    kernel = kernels.pop()
    if kernel % 2 == 0:
        raise ConversionError(
            f"{voice}: post-filter unit kernel {kernel} must be odd for 'same' padding"
        )
    return kernel


def _add_post_filter(
    build: VoiceBuild,
    voice: str,
    tensors: dict[str, np.ndarray],
    config: dict[str, Any],
) -> None:
    """The optional Piperlite post-filter (active for Kristin).

    When the manifest declares it, every tensor must be present with the exact
    expected shape; when it declares zero channels, none of them may be.
    """
    channels = int(config.get("post_filter_channels") or 0)
    layers = int(config.get("post_filter_layers") or 0)
    kernel = int(config.get("post_filter_kernel") or 9)
    present = sorted(k for k in tensors if k.startswith("post_filter."))

    if channels <= 0:
        if present:
            raise ConversionError(
                f"{voice}: post_filter_channels is 0 but the checkpoint carries "
                f"{present}; the manifest and the weights disagree"
            )
        return
    if layers <= 0:
        raise ConversionError(
            f"{voice}: post_filter_channels={channels} with post_filter_layers={layers}"
        )

    def need(key: str, shape: tuple[int, ...]) -> np.ndarray:
        if key not in tensors:
            raise ConversionError(
                f"{voice}: the manifest enables the post-filter but {key!r} is missing"
            )
        array = tensors[key]
        if tuple(array.shape) != shape:
            raise ConversionError(
                f"{voice}: {key} has shape {tuple(array.shape)}, expected {shape}"
            )
        return array

    # `post_filter_kernel` describes the in/out convs only. The per-unit convs
    # carry their own, smaller kernel (3 on Kristin) and upstream reads it from
    # the weight tensor rather than from the config, so it is recovered here and
    # written as its own metadata key instead of being assumed.
    unit_kernel = _post_filter_unit_kernel(voice, tensors, layers, channels)
    build.scalar_meta["dec.post_filter_unit_kernel"] = unit_kernel

    add = build.tensors.append
    add(
        VoiceTensor(
            "dec.post_filter.in_conv.weight",
            flatten_conv1d(
                "in_conv", need("post_filter.in_conv.weight", (channels, 1, kernel))
            ),
        )
    )
    add(
        VoiceTensor(
            "dec.post_filter.in_conv.bias",
            need("post_filter.in_conv.bias", (channels,)),
            f32=True,
        )
    )
    for layer in range(layers):
        add(
            VoiceTensor(
                f"dec.post_filter.units.{layer}.scale",
                need(f"post_filter.units.{layer}.scale", (1,)).reshape(1),
                f32=True,
            )
        )
        for conv in (1, 2):
            add(
                VoiceTensor(
                    f"dec.post_filter.units.{layer}.conv{conv}.weight",
                    flatten_conv1d(
                        "unit",
                        need(
                            f"post_filter.units.{layer}.conv{conv}.weight",
                            (channels, channels, unit_kernel),
                        ),
                    ),
                )
            )
            add(
                VoiceTensor(
                    f"dec.post_filter.units.{layer}.conv{conv}.bias",
                    need(f"post_filter.units.{layer}.conv{conv}.bias", (channels,)),
                    f32=True,
                )
            )
    add(
        VoiceTensor(
            "dec.post_filter.out_conv.weight",
            flatten_conv1d(
                "out_conv", need("post_filter.out_conv.weight", (1, channels, kernel))
            ),
        )
    )
    add(
        VoiceTensor(
            "dec.post_filter.out_conv.bias",
            need("post_filter.out_conv.bias", (1,)),
            f32=True,
        )
    )

    unexpected = [
        k
        for k in present
        if k
        not in {
            "post_filter.in_conv.weight",
            "post_filter.in_conv.bias",
            "post_filter.out_conv.weight",
            "post_filter.out_conv.bias",
        }
        and not re.fullmatch(r"post_filter\.units\.\d+\.(scale|conv[12]\.(weight|bias))", k)
    ]
    if unexpected:
        raise ConversionError(f"{voice}: unrecognised post-filter tensors {unexpected}")


# ---------------------------------------------------------------------------
# The vocos path
# ---------------------------------------------------------------------------

DEFINE_RE = re.compile(r"^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(-?\d+)\s*$")


def parse_nano_meta(path: Path) -> dict[str, int]:
    """Parse `nano_q8_meta.h`. The grammar is `#define NAME INTEGER` and nothing else."""
    out: dict[str, int] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("/*", "*", "//", "#pragma")):
            continue
        match = DEFINE_RE.match(line)
        if match is None:
            raise ConversionError(
                f"{path}:{line_no}: cannot parse {line.strip()!r}; the generated header's "
                "grammar changed and the reconstructed layout can no longer be trusted"
            )
        out[match.group(1)] = int(match.group(2))
    if not out:
        raise ConversionError(f"{path}: no #define found")
    return out


@dataclass
class BlobArray:
    """One array inside a flat blob, as reconstructed from the header."""

    offset: int
    rows: int  # 1 for a vector
    row_len: int  # padded length (n16) for weights, element count for vectors
    used: int  # real elements per row, before the n16 padding
    kind: str  # "w" | "f32"


class BlobReader:
    """Reads the reconstructed arrays out of one blob, checking as it goes.

    Every array's end is checked against the next array's offset and the whole
    blob against `NANO_*_BYTES`. Upstream's exporter is the only description of
    this layout, so a change there has to surface as a conversion error, not as
    corrupted audio.
    """

    def __init__(self, name: str, blob: bytes, total_key: int, int8: bool) -> None:
        self.name = name
        self.blob = blob
        self.int8 = int8
        self.elem_size = 1 if int8 else 4
        if len(blob) != total_key:
            raise ConversionError(
                f"{name}: blob is {len(blob)} bytes, the header says {total_key}"
            )
        self._claimed: list[tuple[int, int, str]] = []

    def _claim(self, offset: int, nbytes: int, label: str) -> bytes:
        if offset < 0 or offset + nbytes > len(self.blob):
            raise ConversionError(
                f"{self.name}.{label}: [{offset}, {offset + nbytes}) is outside the "
                f"{len(self.blob)}-byte blob"
            )
        self._claimed.append((offset, offset + nbytes, label))
        return self.blob[offset : offset + nbytes]

    def weights(self, offset: int, rows: int, row_len: int, used: int, label: str) -> np.ndarray:
        """A weight matrix stored as `rows` rows of `row_len` elements.

        `row_len` is the export's 16-element-padded stride; only the first
        `used` elements of each row are real, the rest are zeros that exist so
        the MCU's dot product can run unrolled.
        """
        if used > row_len:
            raise ConversionError(
                f"{self.name}.{label}: {used} used elements exceed the {row_len} padded stride"
            )
        raw = self._claim(offset, rows * row_len * self.elem_size, label)
        dtype = np.int8 if self.int8 else "<f4"
        flat = np.frombuffer(raw, dtype=dtype).reshape(rows, row_len)
        padding = flat[:, used:]
        if padding.size and np.any(padding != 0):
            raise ConversionError(
                f"{self.name}.{label}: the row padding is not zero; the export layout "
                "is not what this converter reconstructs"
            )
        return flat[:, :used].astype(np.float32)

    def f32(self, offset: int, count: int, label: str) -> np.ndarray:
        raw = self._claim(offset, count * 4, label)
        return np.frombuffer(raw, dtype="<f4").astype(np.float32)

    def check_coverage(self) -> None:
        """Every claimed range must end where the next one starts, modulo the
        16-byte alignment the exporter pads to, and the last must reach the end.
        """
        ordered = sorted(self._claimed)
        cursor = 0
        for start, end, label in ordered:
            if start < cursor:
                raise ConversionError(
                    f"{self.name}.{label}: overlaps the previous array "
                    f"(starts at {start}, previous ended at {cursor})"
                )
            if start - cursor >= 16:
                raise ConversionError(
                    f"{self.name}.{label}: {start - cursor} unclaimed bytes before it; "
                    "the reconstruction missed an array"
                )
            gap = self.blob[cursor:start]
            if gap and any(byte != 0 for byte in gap):
                raise ConversionError(
                    f"{self.name}.{label}: the alignment gap before it is not zero"
                )
            cursor = end
        if align_up(cursor, 16) < len(self.blob):
            raise ConversionError(
                f"{self.name}: {len(self.blob) - cursor} trailing bytes were never claimed"
            )


def n16(value: int) -> int:
    return align_up(value, 16)


def build_vocos_voice(
    name: str,
    aliases: Sequence[str],
    meta_header: Path,
    meta_json: Path,
    front_blob: Path,
    dec_blob: Path,
    vocabulary: dict[str, int],
) -> VoiceBuild:
    meta = parse_nano_meta(meta_header)
    info = json.loads(meta_json.read_text(encoding="utf-8"))

    def need(key: str) -> int:
        if key not in meta:
            raise ConversionError(f"{meta_header}: missing #define {key}")
        return meta[key]

    if need("NANO_FORMAT_VERSION") != 1:
        raise ConversionError(
            f"{name}: NANO_FORMAT_VERSION={meta['NANO_FORMAT_VERSION']}, only 1 is supported"
        )
    if need("NANO_NORM_TYPE") != 0:
        raise ConversionError(
            f"{name}: NANO_NORM_TYPE={meta['NANO_NORM_TYPE']} selects DyT normalisation, "
            "which the runtime deliberately refuses rather than silently treating as LayerNorm"
        )
    if need("NANO_ACT_TYPE") != 0:
        raise ConversionError(
            f"{name}: NANO_ACT_TYPE={meta['NANO_ACT_TYPE']} selects ReLU, "
            "which the runtime deliberately refuses rather than silently using GELU"
        )

    # NANO_WEIGHT_FORMAT is only emitted by the float export; its absence means
    # the int8 rows the MCU runs.
    weight_format = meta.get("NANO_WEIGHT_FORMAT", 0)
    if weight_format not in (0, 1):
        raise ConversionError(f"{name}: unknown NANO_WEIGHT_FORMAT={weight_format}")
    int8 = weight_format == 0

    declared_weights = str(info.get("weights") or "")
    if declared_weights and (declared_weights == "int8") != int8:
        raise ConversionError(
            f"{name}: meta.json says weights={declared_weights!r} but the header's "
            f"NANO_WEIGHT_FORMAT={weight_format} says otherwise"
        )

    for blob, key in ((front_blob, "front_sha256"), (dec_blob, "dec_sha256")):
        expected = info.get(key)
        if not expected:
            continue
        actual = sha256_file(blob)
        if actual != expected:
            raise ConversionError(
                f"{blob}: sha256 mismatch (meta.json={expected}, actual={actual})"
            )

    vocab = need("NANO_VOCAB")
    dur_hidden = need("NANO_DUR_HIDDEN")
    dur_depth = need("NANO_DUR_DEPTH")
    dur_kernel = need("NANO_DUR_KERNEL")
    ac_hidden = need("NANO_AC_HIDDEN")
    ac_token_depth = need("NANO_AC_TOKEN_DEPTH")
    ac_depth = need("NANO_AC_DEPTH")
    ac_kernel = need("NANO_AC_KERNEL")
    mels = need("NANO_MELS")
    dim = need("NANO_DIM")
    blocks = need("NANO_BLOCKS")
    pw_hidden = need("NANO_PW_HIDDEN")
    dw_kernel = need("NANO_DW_KERNEL")
    embed_kernel = need("NANO_EMBED_KERNEL")
    noise_ch = need("NANO_NOISE_CH")
    n_fft = need("NANO_N_FFT")
    hop = need("NANO_HOP")
    bins = need("NANO_BINS")
    head_out = need("NANO_HEAD_OUT")

    if bins != n_fft // 2 + 1:
        raise ConversionError(f"{name}: NANO_BINS={bins} != n_fft/2+1 = {n_fft // 2 + 1}")
    if head_out != 2 * bins:
        raise ConversionError(f"{name}: NANO_HEAD_OUT={head_out} != 2*bins = {2 * bins}")
    if not 0 < hop <= n_fft:
        raise ConversionError(f"{name}: hop={hop} is outside (0, n_fft={n_fft}]")
    for label, k in (("dw_kernel", dw_kernel), ("embed_kernel", embed_kernel),
                     ("dur_kernel", dur_kernel), ("ac_kernel", ac_kernel)):
        if k % 2 == 0:
            raise ConversionError(f"{name}: {label}={k} must be odd for 'same' padding")
    if len(vocabulary) != vocab:
        raise ConversionError(
            f"{name}: the supplied vocabulary has {len(vocabulary)} symbols, "
            f"the model was trained with NANO_VOCAB={vocab}"
        )

    front = BlobReader(
        f"{name}/front", front_blob.read_bytes(), need("NANO_FRONT_BYTES"), int8
    )
    dec = BlobReader(f"{name}/dec", dec_blob.read_bytes(), need("NANO_DEC_BYTES"), int8)

    build = VoiceBuild(
        name=name,
        aliases=list(aliases),
        sample_rate=int(info.get("sample_rate") or 24000),
        # The vocos packs carry no length scale; the duration policy applies
        # 1.0 and `speed` divides it at run time.
        length_scale=1.0,
        espeak_voice=VOCOS_ESPEAK_VOICE,
        normalization_lang=VOCOS_NORMALIZATION_LANG,
        frontend="misaki",
        decoder="vocos",
        max_tokens=need("NANO_DUR_MAX_TOKENS"),
        token_symbols=[],
        token_ids=[],
        bos_id=1,
        eos_id=2,
        # The misaki frontend frames [BOS] ids [EOS] with nothing interleaved,
        # and has no schwa fallback: 59 is the em dash in this vocabulary.
        pad_id=-1,
        fallback_id=-1,
    )

    specials = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    for symbol, want in specials.items():
        if vocabulary.get(symbol) != want:
            raise ConversionError(
                f"{name}: vocabulary requires {symbol!r} -> {want}, got {vocabulary.get(symbol)}"
            )
    ids = sorted(vocabulary.values())
    if ids != list(range(len(ids))):
        raise ConversionError(
            f"{name}: vocabulary ids must be unique and contiguous from zero"
        )
    # The special symbols are framing, not phonemes: they are written so the
    # runtime can validate the table, and skipped when mapping input.
    for symbol, pid in sorted(vocabulary.items(), key=lambda kv: kv[1]):
        build.token_symbols.append(symbol)
        build.token_ids.append(int(pid))

    build.scalar_meta.update(
        {
            "dur.vocab": vocab,
            "dur.hidden": dur_hidden,
            "dur.depth": dur_depth,
            "dur.kernel": dur_kernel,
            "dur.max_tokens": need("NANO_DUR_MAX_TOKENS"),
            "dur.max_duration": need("NANO_DUR_MAX_DURATION"),
            "ac.vocab": vocab,
            "ac.hidden": ac_hidden,
            "ac.token_depth": ac_token_depth,
            "ac.depth": ac_depth,
            "ac.kernel": ac_kernel,
            "ac.out_channels": mels,
            "dec.dim": dim,
            "dec.blocks": blocks,
            "dec.pw_hidden": pw_hidden,
            "dec.dw_kernel": dw_kernel,
            "dec.embed_kernel": embed_kernel,
            "dec.noise_ch": noise_ch,
            "dec.mels": mels,
            "dec.n_fft": n_fft,
            "dec.hop": hop,
            "dec.bins": bins,
            "dec.norm_type": 0,
            "dec.act_type": 0,
        }
    )
    build.float_meta.update({"dec.dc_pole": 0.9973, "dec.mag_clip": 100.0})

    add = build.tensors.append

    def quantized(
        reader: BlobReader,
        w_off: int,
        scale_off: int,
        bias_off: int,
        rows: int,
        used: int,
        padded: int,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One weight matrix with its per-output-row scale and its bias.

        Not a ggml quant: the scale is per output channel, Q8_0's is per block
        of 32. The first implementation therefore dequantizes here and emits
        F32, which isolates the runtime's activation quantisation from any
        extra F16 rounding.
        """
        w = reader.weights(w_off, rows, padded, used, label + ".w")
        scale = reader.f32(scale_off, n16(rows * 4) // 4, label + ".scale")
        bias = reader.f32(bias_off, n16(rows * 4) // 4, label + ".bias")
        if np.any(scale[rows:] != 0.0) or np.any(bias[rows:] != 0.0):
            raise ConversionError(f"{label}: the scale/bias alignment padding is not zero")
        scale = scale[:rows]
        bias = bias[:rows]
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ConversionError(f"{label}: non-positive or non-finite row scale")
        return w * scale[:, None], bias

    def scalar_f32(reader: BlobReader, offset: int, label: str) -> np.ndarray:
        """A single f32 padded to the exporter's 16-byte grid."""
        values = reader.f32(offset, 4, label)
        if np.any(values[1:] != 0.0):
            raise ConversionError(f"{label}: padding after the scalar is not zero")
        return values[:1]

    # -- front blob: duration then acoustic --------------------------------
    add(
        VoiceTensor(
            "dur.embedding.weight",
            front.f32(need("NOFF_DUR_EMB_F32"), vocab * dur_hidden, "dur.embedding")
            .reshape(vocab, dur_hidden),
            f32=True,
        )
    )
    w, b = quantized(
        front, need("NOFF_DUR_PROJ_W8"), need("NOFF_DUR_PROJ_SCALE"),
        need("NOFF_DUR_PROJ_BIAS"), dur_hidden, dur_hidden + 3,
        need("NANO_DUR_PROJ_N16"), "dur.input_proj",
    )
    add(VoiceTensor("dur.input_proj.weight", w, f32=True))
    add(VoiceTensor("dur.input_proj.bias", b, f32=True))

    for i in range(dur_depth):
        for src, dst in (("C0", "net0"), ("C1", "net2")):
            w, b = quantized(
                front,
                need(f"NOFF_DUR_B{i}_{src}_W8"),
                need(f"NOFF_DUR_B{i}_{src}_SCALE"),
                need(f"NOFF_DUR_B{i}_{src}_BIAS"),
                dur_hidden,
                dur_hidden * dur_kernel,
                need(f"NANO_DUR_B0_{src}_N16"),
                f"dur.blocks.{i}.{dst}",
            )
            add(VoiceTensor(f"dur.blocks.{i}.{dst}.weight", w, f32=True))
            add(VoiceTensor(f"dur.blocks.{i}.{dst}.bias", b, f32=True))
        add(
            VoiceTensor(
                f"dur.blocks.{i}.scale",
                scalar_f32(front, need(f"NOFF_DUR_B{i}_SCALE_F32"), f"dur.blocks.{i}.scale"),
                f32=True,
            )
        )

    w, b = quantized(
        front, need("NOFF_DUR_OUT_W8"), need("NOFF_DUR_OUT_SCALE"),
        need("NOFF_DUR_OUT_BIAS"), 1, dur_hidden, need("NANO_DUR_OUT_N16"), "dur.output",
    )
    add(VoiceTensor("dur.output.weight", w, f32=True))
    add(VoiceTensor("dur.output.bias", b, f32=True))

    add(
        VoiceTensor(
            "ac.embedding.weight",
            front.f32(need("NOFF_AC_EMB_F32"), vocab * ac_hidden, "ac.embedding")
            .reshape(vocab, ac_hidden),
            f32=True,
        )
    )
    for src, dst, extra in (("TPROJ", "ac.token_input_proj", 2),
                            ("FPROJ", "ac.frame_input_proj", 3)):
        w, b = quantized(
            front, need(f"NOFF_AC_{src}_W8"), need(f"NOFF_AC_{src}_SCALE"),
            need(f"NOFF_AC_{src}_BIAS"), ac_hidden, ac_hidden + extra,
            need(f"NANO_AC_{src}_N16"), dst,
        )
        add(VoiceTensor(f"{dst}.weight", w, f32=True))
        add(VoiceTensor(f"{dst}.bias", b, f32=True))

    for stage, depth, dst_prefix in (("T", ac_token_depth, "ac.token_blocks"),
                                     ("F", ac_depth, "ac.frame_blocks")):
        for i in range(depth):
            for src, dst in (("C0", "net0"), ("C1", "net2")):
                w, b = quantized(
                    front,
                    need(f"NOFF_AC_{stage}B{i}_{src}_W8"),
                    need(f"NOFF_AC_{stage}B{i}_{src}_SCALE"),
                    need(f"NOFF_AC_{stage}B{i}_{src}_BIAS"),
                    ac_hidden,
                    ac_hidden * ac_kernel,
                    need(f"NANO_AC_{stage}B0_{src}_N16"),
                    f"{dst_prefix}.{i}.{dst}",
                )
                add(VoiceTensor(f"{dst_prefix}.{i}.{dst}.weight", w, f32=True))
                add(VoiceTensor(f"{dst_prefix}.{i}.{dst}.bias", b, f32=True))
            add(
                VoiceTensor(
                    f"{dst_prefix}.{i}.scale",
                    scalar_f32(
                        front,
                        need(f"NOFF_AC_{stage}B{i}_SCALE_F32"),
                        f"{dst_prefix}.{i}.scale",
                    ),
                    f32=True,
                )
            )

    w, b = quantized(
        front, need("NOFF_AC_OUT_W8"), need("NOFF_AC_OUT_SCALE"), need("NOFF_AC_OUT_BIAS"),
        mels, ac_hidden, need("NANO_AC_OUT_N16"), "ac.output",
    )
    add(VoiceTensor("ac.output.weight", w, f32=True))
    add(VoiceTensor("ac.output.bias", b, f32=True))
    front.check_coverage()

    # -- decoder blob ------------------------------------------------------
    w, b = quantized(
        dec, need("DOFF_EMBED_W8"), need("DOFF_EMBED_SCALE"), need("DOFF_EMBED_BIAS"),
        dim, mels * embed_kernel, need("NANO_EMBED_N16"), "dec.embed",
    )
    add(VoiceTensor("dec.embed.weight", w, f32=True))
    add(VoiceTensor("dec.embed.bias", b, f32=True))

    w, b = quantized(
        dec, need("DOFF_NOISE_W8"), need("DOFF_NOISE_SCALE"), need("DOFF_NOISE_BIAS"),
        dim, noise_ch * embed_kernel, need("NANO_NOISE_N16"), "dec.noise",
    )
    add(VoiceTensor("dec.noise.weight", w, f32=True))
    add(VoiceTensor("dec.noise.bias", b, f32=True))

    add(VoiceTensor("dec.norm.weight", dec.f32(need("DOFF_NORM_W_F32"), dim, "dec.norm.w"), f32=True))
    add(VoiceTensor("dec.norm.bias", dec.f32(need("DOFF_NORM_B_F32"), dim, "dec.norm.b"), f32=True))

    for i in range(blocks):
        add(
            VoiceTensor(
                f"dec.blocks.{i}.dw.weight",
                dec.f32(need(f"DOFF_B{i}_DW_W_F32"), dim * dw_kernel, f"b{i}.dw.w")
                .reshape(dim, dw_kernel),
                f32=True,
            )
        )
        add(
            VoiceTensor(
                f"dec.blocks.{i}.dw.bias",
                dec.f32(need(f"DOFF_B{i}_DW_B_F32"), dim, f"b{i}.dw.b"),
                f32=True,
            )
        )
        add(
            VoiceTensor(
                f"dec.blocks.{i}.norm.weight",
                dec.f32(need(f"DOFF_B{i}_NORM_W_F32"), dim, f"b{i}.norm.w"),
                f32=True,
            )
        )
        add(
            VoiceTensor(
                f"dec.blocks.{i}.norm.bias",
                dec.f32(need(f"DOFF_B{i}_NORM_B_F32"), dim, f"b{i}.norm.b"),
                f32=True,
            )
        )
        w, b = quantized(
            dec, need(f"DOFF_B{i}_PW0_W8"), need(f"DOFF_B{i}_PW0_SCALE"),
            need(f"DOFF_B{i}_PW0_BIAS"), pw_hidden, dim, need(f"NANO_B{i}_PW0_N16"),
            f"dec.blocks.{i}.pw0",
        )
        add(VoiceTensor(f"dec.blocks.{i}.pw0.weight", w, f32=True))
        add(VoiceTensor(f"dec.blocks.{i}.pw0.bias", b, f32=True))
        w, b = quantized(
            dec, need(f"DOFF_B{i}_PW1_W8"), need(f"DOFF_B{i}_PW1_SCALE"),
            need(f"DOFF_B{i}_PW1_BIAS"), dim, pw_hidden, need(f"NANO_B{i}_PW1_N16"),
            f"dec.blocks.{i}.pw1",
        )
        add(VoiceTensor(f"dec.blocks.{i}.pw1.weight", w, f32=True))
        add(VoiceTensor(f"dec.blocks.{i}.pw1.bias", b, f32=True))
        # LayerScale stays F32: values near 1e-6 are subnormal in F16 and some
        # backends flush them to zero.
        add(
            VoiceTensor(
                f"dec.blocks.{i}.gamma",
                dec.f32(need(f"DOFF_B{i}_GAMMA_F32"), dim, f"b{i}.gamma"),
                f32=True,
            )
        )

    add(
        VoiceTensor(
            "dec.final_norm.weight",
            dec.f32(need("DOFF_FNORM_W_F32"), dim, "fnorm.w"),
            f32=True,
        )
    )
    add(
        VoiceTensor(
            "dec.final_norm.bias",
            dec.f32(need("DOFF_FNORM_B_F32"), dim, "fnorm.b"),
            f32=True,
        )
    )
    w, b = quantized(
        dec, need("DOFF_HEAD_W8"), need("DOFF_HEAD_SCALE"), need("DOFF_HEAD_BIAS"),
        head_out, dim, need("NANO_HEAD_N16"), "dec.head",
    )
    add(VoiceTensor("dec.head.weight", w, f32=True))
    add(VoiceTensor("dec.head.bias", b, f32=True))
    dec.check_coverage()

    build.declared_parameters = int(info.get("params") or 0)
    check_parameter_count(build)
    build.sources = {
        "lineage": str(info.get("lineage") or ""),
        "weights": "int8-dequantized" if int8 else "f32",
        "front_sha256": str(info.get("front_sha256") or ""),
        "dec_sha256": str(info.get("dec_sha256") or ""),
        # The vocos layout is reconstructed from nano_q8_meta.h, so that header
        # is as much an input as the blobs are.
        "offsets_header_sha256": str(info.get("offsets_header_sha256") or ""),
        "parameters": str(info.get("params") or ""),
    }
    return build


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def emit(
    writer: GgufWriter,
    builds: Sequence[VoiceBuild],
    default_voice: str,
    nfd: NfdTable,
    artifact_revision: str,
) -> None:
    names = [b.name for b in builds]
    writer.add_string("kokopop.arch", "sanotts")
    writer.add_u32("kokopop.sanotts.version", SANOTTS_GGUF_VERSION)
    # Provenance. The two revisions are separate keys, not one blended string:
    # the code revision says which upstream rules this converter implements, the
    # artifact revision says which bytes it read. Only the second one identifies
    # the weights, and it is immutable, so `<repo>@<artifact_revision>` resolves
    # to exactly the files that produced this GGUF.
    writer.add_string("kokopop.sanotts.source.repo", SANOTTS_REPO)
    writer.add_string("kokopop.sanotts.source.code_revision", SANOTTS_CODE_REVISION)
    writer.add_string("kokopop.sanotts.source.artifact_revision", artifact_revision)
    writer.add_string(
        "kokopop.sanotts.source",
        f"{SANOTTS_REPO} code@{SANOTTS_CODE_REVISION} "
        f"artifacts@{artifact_revision} voices={','.join(names)}",
    )
    writer.add_string_array("kokopop.voices", names)
    writer.add_string("kokopop.default_voice", default_voice)

    default = next(b for b in builds if b.name == default_voice)
    # Documented as "the default voice's rate"; per-voice rates are below.
    writer.add_u32("kokopop.sample_rate", default.sample_rate)

    writer.add_string("kokopop.sanotts.unicode_version", nfd.unicode_version)
    writer.add_u32_array("kokopop.sanotts.nfd_codepoints", nfd.codepoints)
    writer.add_u32_array("kokopop.sanotts.nfd_offsets", nfd.offsets)
    writer.add_u32_array("kokopop.sanotts.nfd_values", nfd.values)
    writer.add_u32_array("kokopop.sanotts.nfd_ccc_codepoints", nfd.ccc_codepoints)
    writer.add_u32_array("kokopop.sanotts.nfd_ccc_classes", nfd.ccc_classes)

    for index, build in enumerate(builds):
        prefix = f"kokopop.sanotts.voice.{index}."
        writer.add_string(prefix + "name", build.name)
        writer.add_string_array(prefix + "aliases", build.aliases)
        writer.add_u32(prefix + "sample_rate", build.sample_rate)
        writer.add_f32(prefix + "length_scale", build.length_scale)
        writer.add_string(prefix + "espeak_voice", build.espeak_voice)
        writer.add_string(prefix + "normalization_lang", build.normalization_lang)
        writer.add_string(prefix + "frontend", build.frontend)
        writer.add_string(prefix + "decoder", build.decoder)
        writer.add_u32(prefix + "max_tokens", build.max_tokens)
        writer.add_string_array(prefix + "token_symbols", build.token_symbols)
        writer.add_u32_array(prefix + "token_ids", build.token_ids)
        writer.add_u32(prefix + "bos_id", build.bos_id)
        writer.add_u32(prefix + "eos_id", build.eos_id)
        writer.add_i32(prefix + "pad_id", build.pad_id)
        writer.add_i32(prefix + "fallback_id", build.fallback_id)
        for key, value in sorted(build.scalar_meta.items()):
            writer.add_u32(prefix + key, value)
        for key, value in sorted(build.float_meta.items()):
            writer.add_f32(prefix + key, value)
        for key, value in sorted(build.array_meta.items()):
            writer.add_u32_array(prefix + key, value)
        for key, value in sorted(build.sources.items()):
            writer.add_string(prefix + "source." + key, value)

        for tensor in build.tensors:
            writer.add_tensor(
                prefix + tensor.suffix,
                tensor.data,
                GGML_TYPE_F32 if tensor.f32 else GGML_TYPE_F16,
            )


def check_parameter_count(build: VoiceBuild) -> None:
    """Compare the emitted element count against the source's declared total.

    This is the single strongest validation in the converter, and it is why
    the vocos path can be trusted at all: that layout is reconstructed from a
    generated C header, and a missed tensor, a wrong dimension or padding left
    in a row all change this number. It matches to the element for every
    shipped voice.
    """
    if build.declared_parameters <= 0:
        logger.warning(
            "%s: the source declares no parameter count; the reconstructed layout "
            "cannot be cross-checked", build.name,
        )
        return
    actual = build.element_count()
    if actual != build.declared_parameters:
        raise ConversionError(
            f"{build.name}: reconstructed {actual} parameters, the source declares "
            f"{build.declared_parameters} (difference {actual - build.declared_parameters:+d}); "
            "a tensor is missing, mis-shaped, or still carries export padding"
        )
    logger.info("%s: %d parameters, matching the source", build.name, actual)


def check_unique(builds: Sequence[VoiceBuild]) -> None:
    seen: dict[str, str] = {}
    for build in builds:
        for label in [build.name, *build.aliases]:
            if label in seen:
                raise ConversionError(
                    f"voice name/alias {label!r} is claimed by both {seen[label]!r} "
                    f"and {build.name!r}"
                )
            seen[label] = build.name


def describe(builds: Sequence[VoiceBuild], writer: GgufWriter) -> str:
    lines = []
    for build in builds:
        lines.append(
            f"{build.name}  {build.decoder}/{build.frontend}  {build.sample_rate} Hz  "
            f"length_scale={build.length_scale:g}  max_tokens={build.max_tokens}  "
            f"vocab={len(build.token_symbols)}"
        )
        for tensor in build.tensors:
            shape = "x".join(str(d) for d in tensor.data.shape)
            kind = "F32" if tensor.f32 else "F16"
            nbytes = tensor.data.size * (4 if tensor.f32 else 2)
            lines.append(f"    {tensor.suffix:<48} [{shape}] {kind} {nbytes:>10} B")
    total = writer.total_tensor_bytes()
    lines.append(f"total tensor bytes: {total} ({total / 1024 / 1024:.2f} MiB)")
    return "\n".join(lines) + "\n"


def load_vocos_vocabulary(path: Path | None) -> dict[str, int]:
    if path is None:
        path = Path(__file__).with_name("sanotts_vocos_vocab.json")
    if not path.is_file():
        raise ConversionError(
            f"missing vocos vocabulary table: {path}. Pass --vocos-vocabulary with a JSON "
            "object mapping each symbol to its id."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    table = data.get("vocabulary", data)
    if not isinstance(table, dict) or not table:
        raise ConversionError(f"{path}: expected a non-empty symbol -> id object")
    out: dict[str, int] = {}
    for symbol, pid in table.items():
        if not isinstance(pid, int) or isinstance(pid, bool):
            raise ConversionError(f"{path}: {symbol!r} maps to a non-integer id")
        out[str(symbol)] = int(pid)
    return out


def resolve_requested(voices: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for raw in voices:
        name = VOICE_ALIASES.get(raw, raw)
        if name in resolved:
            continue
        if name not in PIPERLITE_PACKAGES and name not in VOCOS_VOICES:
            known = ", ".join(sorted({*PIPERLITE_PACKAGES, *VOCOS_VOICES, *VOICE_ALIASES}))
            raise ConversionError(f"unknown voice {raw!r}; known voices: {known}")
        resolved.append(name)
    return resolved


def build_voice(
    name: str,
    args: argparse.Namespace,
    cache_dir: Path,
    vocabulary_path: Path | None,
    revision: str = SANOTTS_DEFAULT_ARTIFACT_REVISION,
) -> VoiceBuild:
    aliases = [alias for alias, target in VOICE_ALIASES.items() if target == name]
    if name in VOCOS_VOICES:
        voice = VOCOS_VOICES[name]
        if args.voice_dir:
            directory = Path(args.voice_dir).expanduser().resolve()
            paths = {
                "meta_header": directory / "nano_q8_meta.h",
                "meta_json": directory / "meta.json",
                "front": directory / voice.front_blob,
                "dec": directory / voice.dec_blob,
            }
            for label, path in paths.items():
                if not path.is_file():
                    raise ConversionError(f"--voice-dir {directory}: missing {path.name}")
        else:
            paths = {
                "meta_header": fetch(f"{voice.directory}/nano_q8_meta.h", cache_dir, revision),
                "meta_json": fetch(f"{voice.directory}/meta.json", cache_dir, revision),
                "front": fetch(f"{voice.directory}/{voice.front_blob}", cache_dir, revision),
                "dec": fetch(f"{voice.directory}/{voice.dec_blob}", cache_dir, revision),
            }
        return build_vocos_voice(
            name,
            aliases,
            paths["meta_header"],
            paths["meta_json"],
            paths["front"],
            paths["dec"],
            load_vocos_vocabulary(vocabulary_path),
        )

    package = PIPERLITE_PACKAGES[name]
    if args.voice_dir:
        directory = Path(args.voice_dir).expanduser().resolve()
    else:
        directory = artifact_cache(cache_dir, revision) / package
        for filename in ("manifest.json", "piper-phoneme-config.json", "weights.fp16.bin"):
            fetch(f"{package}/{filename}", cache_dir, revision)
    return build_piperlite_voice(name, load_voice_pack(name, directory), aliases)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, type=Path, help="output GGUF path")
    parser.add_argument(
        "--voices", default="", help="comma-separated voice names (heart, amy, ...)"
    )
    parser.add_argument(
        "--voice", action="append", default=[], help="a voice name; repeatable"
    )
    parser.add_argument(
        "--voice-dir",
        help="local directory for the single requested voice, instead of downloading",
    )
    parser.add_argument(
        "--default-voice",
        help="canonical name of the default voice (defaults to the first one)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help=f"download cache (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--artifact-revision", default=SANOTTS_DEFAULT_ARTIFACT_REVISION,
        help="immutable Hugging Face commit sha to download weights and manifests "
             f"from (default: {SANOTTS_DEFAULT_ARTIFACT_REVISION})",
    )
    parser.add_argument(
        "--vocos-vocabulary", type=Path,
        help="JSON symbol -> id table for the vocos voices "
             "(default: tools/sanotts_vocos_vocab.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the tensor plan, sizes and types without writing anything",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    requested = [v.strip() for v in args.voices.split(",") if v.strip()] + list(args.voice)
    if not requested:
        parser.error("at least one voice is required (--voices or --voice)")

    names = resolve_requested(requested)
    if args.voice_dir and len(names) != 1:
        parser.error("--voice-dir applies to a single voice")

    try:
        cache_dir = Path(args.cache_dir).expanduser()
        revision = check_immutable_revision(args.artifact_revision)
        builds = [
            build_voice(name, args, cache_dir, args.vocos_vocabulary, revision)
            for name in names
        ]
        check_unique(builds)

        default_voice = args.default_voice or builds[0].name
        if default_voice not in {b.name for b in builds}:
            raise ConversionError(
                f"--default-voice {default_voice!r} is not among the converted voices"
            )

        nfd = build_nfd_table()
        logger.info(
            "NFD table: %d code points, %d decomposition values, Unicode %s",
            nfd.entries, len(nfd.values), nfd.unicode_version,
        )

        writer = GgufWriter(args.output)
        # `--voice-dir` bypasses the download, so the artifact revision no
        # longer describes where the bytes came from; say so rather than stamp
        # a commit the files may not be from.
        artifact_revision = "local" if args.voice_dir else revision
        emit(writer, builds, default_voice, nfd, artifact_revision)

        print(describe(builds, writer))
        if args.dry_run:
            logger.info("--dry-run: nothing written")
            return 0

        writer.write()
        logger.info(
            "wrote %s (%d tensors, %d KV pairs, %.2f MiB)",
            args.output, len(writer.tensors), len(writer.kv),
            args.output.stat().st_size / 1024 / 1024,
        )
    except ConversionError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
