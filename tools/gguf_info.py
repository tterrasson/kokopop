#!/usr/bin/env python3
"""Inspect a kokopop GGUF file: dump its metadata and list bundled voices.

Pure-stdlib GGUF v3 reader (no dependency on the `gguf` pip package, since
kokopop's converters emit files with a hand-rolled writer — see
convert_kokoro_to_gguf.py / convert_sanotts_to_gguf.py).

Usage:
    python3 tools/gguf_info.py model.gguf
    python3 tools/gguf_info.py model.gguf --tensors
    python3 tools/gguf_info.py model.gguf --json
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

GGUF_MAGIC = b"GGUF"

T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FORMATS = {
    T_UINT8: "<B", T_INT8: "<b",
    T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i",
    T_FLOAT32: "<f",
    T_UINT64: "<Q", T_INT64: "<q",
    T_FLOAT64: "<d",
}

# name, block_size, type_size (bytes/block) — from ggml_type_traits.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}

# kv arrays that are internal plumbing / bulk vocab rather than human-facing
# metadata — collapsed to a count unless --full is passed.
_BULK_ARRAY_KEYS = {
    "kokopop.tensor.logical_names",
    "kokopop.tensor.physical_names",
    "tokenizer.ggml.tokens",
}

_SANOTTS_VOICE_RE = re.compile(r"^kokopop\.sanotts\.voice\.(\d+)\.(.+)$")


class GGUFParseError(ValueError):
    pass


class Tensor:
    __slots__ = ("name", "dims", "ggml_type", "offset", "size")

    def __init__(self, name: str, dims: tuple[int, ...], ggml_type: int, offset: int):
        self.name = name
        self.dims = dims
        self.ggml_type = ggml_type
        self.offset = offset
        self.size = 0

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.ggml_type, (f"unknown({self.ggml_type})", 1, 1))[0]

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        chunk = self.data[self.pos:self.pos + n]
        if len(chunk) != n:
            raise GGUFParseError("unexpected end of file")
        self.pos += n
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype == T_STRING:
            return self.string()
        if vtype == T_BOOL:
            return self.read(1) != b"\x00"
        if vtype == T_ARRAY:
            item_type = self.u32()
            count = self.u64()
            return [self.value(item_type) for _ in range(count)]
        fmt = _SCALAR_FORMATS.get(vtype)
        if fmt is None:
            raise GGUFParseError(f"unsupported GGUF value type {vtype}")
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]


class GGUFFile:
    def __init__(self, path: Path):
        self.path = path
        data = path.read_bytes()
        self.file_size = len(data)
        r = _Reader(data)

        magic = r.read(4)
        if magic != GGUF_MAGIC:
            raise GGUFParseError(f"{path}: not a GGUF file (magic={magic!r})")
        self.version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()

        self.kv: dict[str, object] = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            self.kv[key] = r.value(vtype)

        self.tensors: list[Tensor] = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            dims = tuple(r.u64() for _ in range(n_dims))
            ggml_type = r.u32()
            offset = r.u64()
            self.tensors.append(Tensor(name, dims, ggml_type, offset))

        alignment = int(self.kv.get("general.alignment", 32))
        self.data_start = ((r.pos + alignment - 1) // alignment) * alignment
        self._compute_tensor_sizes()

    def _compute_tensor_sizes(self) -> None:
        ordered = sorted(self.tensors, key=lambda t: t.offset)
        for i, tensor in enumerate(ordered):
            type_info = GGML_TYPES.get(tensor.ggml_type)
            if type_info is not None:
                _, blck_size, type_size = type_info
                tensor.size = (tensor.n_elements // blck_size) * type_size
                continue
            next_offset = (
                ordered[i + 1].offset
                if i + 1 < len(ordered)
                else self.file_size - self.data_start
            )
            tensor.size = max(0, next_offset - tensor.offset)

    def tensor(self, name: str) -> Tensor | None:
        for t in self.tensors:
            if t.name == name:
                return t
        return None


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GiB"


def format_kv_value(key: str, value: object, full: bool) -> str:
    if isinstance(value, list):
        if not full and (key in _BULK_ARRAY_KEYS or len(value) > 16):
            preview = ", ".join(str(v) for v in value[:5])
            return f"[{len(value)} items] {preview}, ..."
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def print_metadata(gg: GGUFFile, full: bool) -> None:
    print(f"# {gg.path}")
    print(f"gguf version : {gg.version}")
    print(f"file size    : {human_size(gg.file_size)} ({gg.file_size} bytes)")
    print(f"tensors      : {len(gg.tensors)}")
    print(f"kv pairs     : {len(gg.kv)}")
    print()
    print("## Metadata")
    for key in sorted(gg.kv):
        print(f"  {key} = {format_kv_value(key, gg.kv[key], full)}")


def collect_voices(gg: GGUFFile) -> list[dict]:
    arch = gg.kv.get("kokopop.arch", "unknown")
    names = gg.kv.get("kokopop.voices")
    default_voice = gg.kv.get("kokopop.default_voice")

    if arch == "sanotts":
        by_index: dict[int, dict[str, object]] = {}
        for key, value in gg.kv.items():
            m = _SANOTTS_VOICE_RE.match(key)
            if not m:
                continue
            index, field = int(m.group(1)), m.group(2)
            by_index.setdefault(index, {})[field] = value
        voices = []
        for index in sorted(by_index):
            fields = by_index[index]
            name = fields.get("name", f"<voice {index}>")
            tensors = [t for t in gg.tensors if t.name.startswith(f"kokopop.sanotts.voice.{index}.")]
            voices.append({
                "name": name,
                "default": name == default_voice,
                "aliases": fields.get("aliases", []),
                "sample_rate": fields.get("sample_rate"),
                "espeak_voice": fields.get("espeak_voice"),
                "normalization_lang": fields.get("normalization_lang"),
                "frontend": fields.get("frontend"),
                "decoder": fields.get("decoder"),
                "tensor_count": len(tensors),
                "tensor_bytes": sum(t.size for t in tensors),
            })
        return voices

    voice_names = names if isinstance(names, list) else []
    if not voice_names:
        prefix = "kokopop.voice."
        voice_names = sorted(
            t.name[len(prefix):] for t in gg.tensors if t.name.startswith(prefix)
        )

    voices = []
    for name in voice_names:
        tensor = gg.tensor(f"kokopop.voice.{name}")
        voices.append({
            "name": name,
            "default": name == default_voice,
            "shape": list(tensor.dims) if tensor else None,
            "dtype": tensor.type_name if tensor else None,
            "tensor_bytes": tensor.size if tensor else 0,
        })
    return voices


def print_voices(gg: GGUFFile) -> None:
    voices = collect_voices(gg)
    print()
    print(f"## Voices ({len(voices)})")
    if not voices:
        print("  (none found)")
        return
    for v in voices:
        marker = " [default]" if v.get("default") else ""
        details = []
        if v.get("shape") is not None:
            details.append(f"shape={v['shape']}")
        if v.get("dtype"):
            details.append(f"dtype={v['dtype']}")
        if v.get("sample_rate") is not None:
            details.append(f"sample_rate={v['sample_rate']}")
        if v.get("espeak_voice"):
            details.append(f"espeak={v['espeak_voice']}")
        if v.get("aliases"):
            details.append(f"aliases={v['aliases']}")
        if v.get("tensor_count") is not None:
            details.append(f"tensors={v['tensor_count']}")
        details.append(f"size={human_size(v.get('tensor_bytes', 0))}")
        print(f"  - {v['name']}{marker}: {', '.join(details)}")


def print_tensors(gg: GGUFFile) -> None:
    print()
    print(f"## Tensors ({len(gg.tensors)})")
    by_type: dict[str, int] = {}
    for t in sorted(gg.tensors, key=lambda t: t.offset):
        print(f"  {t.name}  dims={list(t.dims)}  type={t.type_name}  size={human_size(t.size)}")
        by_type[t.type_name] = by_type.get(t.type_name, 0) + 1
    print()
    print("## Tensor type breakdown")
    for type_name, count in sorted(by_type.items()):
        print(f"  {type_name}: {count}")


def to_json(gg: GGUFFile) -> dict:
    return {
        "path": str(gg.path),
        "gguf_version": gg.version,
        "file_size": gg.file_size,
        "metadata": gg.kv,
        "voices": collect_voices(gg),
        "tensors": [
            {"name": t.name, "dims": list(t.dims), "type": t.type_name, "size": t.size}
            for t in sorted(gg.tensors, key=lambda t: t.offset)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf_path", type=Path, help="path to the .gguf file")
    parser.add_argument("--tensors", action="store_true", help="list every tensor")
    parser.add_argument("--full", action="store_true", help="don't truncate large metadata arrays")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        gg = GGUFFile(args.gguf_path)
    except (GGUFParseError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(to_json(gg), indent=2, ensure_ascii=False))
        return 0

    print_metadata(gg, args.full)
    print_voices(gg)
    if args.tensors:
        print_tensors(gg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
