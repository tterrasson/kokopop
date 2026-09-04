"""Tests for tools/convert_sanotts_to_gguf.py.

Two halves:

* self-contained tests over synthetic inputs, which run everywhere. They cover
  the GGUF writer, the NFD table, and every validation that is supposed to refuse
  an input rather than guess at a layout;
* tests against the real voice packs, which are skipped unless a local copy is
  present. sanoTTS's weights are not covered by its MIT scope, so nothing is
  redistributed here (see THIRD_PARTY.md). Set KOKOPOP_SANOTTS_CACHE, or let
  the converter's own ~/.cache/sanotts default apply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import convert_sanotts_to_gguf as conv  # noqa: E402


# ---------------------------------------------------------------------------
# A minimal GGUF reader, so the tests check the file rather than the writer's
# own bookkeeping.
# ---------------------------------------------------------------------------

_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_gguf(path: Path) -> tuple[dict[str, object], dict[str, tuple[tuple[int, ...], int]]]:
    blob = path.read_bytes()
    assert blob[:4] == b"GGUF"
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", blob, 4)
    assert version == 3
    offset = 24

    def read_string() -> str:
        nonlocal offset
        (length,) = struct.unpack_from("<Q", blob, offset)
        offset += 8
        value = blob[offset : offset + length].decode("utf-8")
        offset += length
        return value

    def read_scalar(typ: int) -> object:
        nonlocal offset
        if typ == 8:
            return read_string()
        size = _SCALAR_SIZES[typ]
        raw = blob[offset : offset + size]
        offset += size
        if typ == 4:
            return struct.unpack("<I", raw)[0]
        if typ == 5:
            return struct.unpack("<i", raw)[0]
        if typ == 6:
            return struct.unpack("<f", raw)[0]
        if typ == 7:
            return raw[0] != 0
        return raw

    kv: dict[str, object] = {}
    for _ in range(n_kv):
        key = read_string()
        (typ,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        if typ == 9:
            (item_type,) = struct.unpack_from("<I", blob, offset)
            offset += 4
            (count,) = struct.unpack_from("<Q", blob, offset)
            offset += 8
            kv[key] = [read_scalar(item_type) for _ in range(count)]
        else:
            kv[key] = read_scalar(typ)

    tensors: dict[str, tuple[tuple[int, ...], int]] = {}
    for _ in range(n_tensors):
        name = read_string()
        (n_dims,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        dims = struct.unpack_from(f"<{n_dims}Q", blob, offset)
        offset += 8 * n_dims
        (ggml_type,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        offset += 8  # data offset
        tensors[name] = (tuple(dims), ggml_type)
    return kv, tensors


def logical_tensors(kv, tensors) -> dict[str, tuple[tuple[int, ...], int]]:
    """Map logical tensor names back onto the physical entries."""
    logical = kv["kokopop.tensor.logical_names"]
    physical = kv["kokopop.tensor.physical_names"]
    assert len(logical) == len(physical)
    return {lg: tensors[ph] for lg, ph in zip(logical, physical)}


# ---------------------------------------------------------------------------
# The GGUF writer
# ---------------------------------------------------------------------------


def test_writer_round_trips_every_kv_type(tmp_path: Path) -> None:
    writer = conv.GgufWriter(tmp_path / "out.gguf")
    writer.add_u32("a.u32", 4242)
    writer.add_i32("a.i32", -7)
    writer.add_f32("a.f32", 1.5)
    writer.add_bool("a.bool", True)
    writer.add_string("a.str", "héllo")
    writer.add_string_array("a.strs", ["x", "ɑ", ""])
    writer.add_u32_array("a.u32s", [0, 1, 4294967295])
    writer.add_tensor("t.f32", np.arange(6, dtype=np.float32).reshape(2, 3), conv.GGML_TYPE_F32)
    writer.add_tensor("t.f16", np.ones((4,), dtype=np.float32), conv.GGML_TYPE_F16)
    writer.write()

    kv, tensors = read_gguf(tmp_path / "out.gguf")
    assert kv["a.u32"] == 4242
    assert kv["a.i32"] == -7
    assert kv["a.f32"] == pytest.approx(1.5)
    assert kv["a.bool"] is True
    assert kv["a.str"] == "héllo"
    assert kv["a.strs"] == ["x", "ɑ", ""]
    assert kv["a.u32s"] == [0, 1, 4294967295]

    by_logical = logical_tensors(kv, tensors)
    # GGUF stores dimensions innermost-first, so a numpy [2, 3] becomes (3, 2).
    assert by_logical["t.f32"] == ((3, 2), conv.GGML_TYPE_F32)
    assert by_logical["t.f16"] == ((4,), conv.GGML_TYPE_F16)


def test_writer_hashes_names_over_the_gguf_limit(tmp_path: Path) -> None:
    long_name = "kokopop.sanotts.voice.0." + "x" * 80
    writer = conv.GgufWriter(tmp_path / "out.gguf")
    writer.add_tensor(long_name, np.ones((2,), dtype=np.float32), conv.GGML_TYPE_F32)
    writer.write()
    kv, tensors = read_gguf(tmp_path / "out.gguf")
    (physical,) = tensors
    assert physical.startswith("st.")
    assert len(physical.encode()) < 64
    assert logical_tensors(kv, tensors)[long_name][0] == (2,)


def test_writer_rejects_unrepresentable_tensors(tmp_path: Path) -> None:
    writer = conv.GgufWriter(tmp_path / "out.gguf")
    with pytest.raises(conv.ConversionError, match="NaN or infinity"):
        writer.add_tensor("t", np.array([1.0, np.nan], dtype=np.float32), conv.GGML_TYPE_F32)
    with pytest.raises(conv.ConversionError, match="overflows F16"):
        writer.add_tensor("t", np.array([1e6], dtype=np.float32), conv.GGML_TYPE_F16)
    with pytest.raises(conv.ConversionError, match="empty tensor"):
        writer.add_tensor("t", np.zeros((0,), dtype=np.float32), conv.GGML_TYPE_F32)


# ---------------------------------------------------------------------------
# The NFD table
# ---------------------------------------------------------------------------


def test_nfd_table_matches_pythons_unicode_database() -> None:
    table = conv.build_nfd_table()
    assert table.unicode_version == unicodedata.unidata_version
    assert len(table.offsets) == table.entries + 1
    assert table.offsets[0] == 0
    assert table.offsets[-1] == len(table.values)
    assert table.entries > 1000  # Latin, Greek, Cyrillic, Hangul, ...

    # Every listed code point decomposes exactly as the table says.
    for index, cp in enumerate(table.codepoints):
        start, end = table.offsets[index], table.offsets[index + 1]
        got = "".join(chr(v) for v in table.values[start:end])
        assert got == unicodedata.normalize("NFD", chr(cp)), hex(cp)

    # And nothing that decomposes is missing: the table is the full set.
    listed = set(table.codepoints)
    for cp in range(0x3000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        decomposes = unicodedata.normalize("NFD", ch) != ch
        assert decomposes == (cp in listed), hex(cp)

    # Spot-checks that matter for IPA input.
    index = table.codepoints.index(0x00E9)  # é
    start, end = table.offsets[index], table.offsets[index + 1]
    assert table.values[start:end] == [0x0065, 0x0301]

    # NFD is decomposition plus canonical ordering, so the combining classes
    # travel with the table.
    assert len(table.ccc_codepoints) == len(table.ccc_classes)
    ccc = dict(zip(table.ccc_codepoints, table.ccc_classes))
    assert ccc[0x0301] == 230  # COMBINING ACUTE ACCENT
    assert ccc[0x0329] == 220  # COMBINING VERTICAL LINE BELOW (syllabic)
    assert 0x0061 not in ccc   # 'a' has class 0 and is not listed
    for cp, cls in ccc.items():
        assert unicodedata.combining(chr(cp)) == cls
        assert cls > 0


# ---------------------------------------------------------------------------
# Piper phoneme table validation
# ---------------------------------------------------------------------------


def _piper_config(**overrides) -> dict:
    id_map = {"_": [0], "^": [1], "$": [2], "ə": [59], "t": [32]}
    config = {"phoneme_type": "espeak", "espeak": {"voice": "en-us"}, "phoneme_id_map": id_map}
    config.update(overrides)
    return config


def test_piper_phoneme_table_accepts_a_well_formed_config() -> None:
    table = conv.parse_piper_phoneme_config("v", _piper_config())
    assert table.espeak_voice == "en-us"
    assert (table.pad_id, table.bos_id, table.eos_id) == (0, 1, 2)
    assert dict(zip(table.symbols, table.ids))["ə"] == 59
    assert conv._schwa_fallback_id(table) == 59


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"phoneme_type": "text"}, "unsupported phoneme_type"),
        ({"espeak": {}}, "missing espeak.voice"),
        ({"phoneme_id_map": {}}, "no phoneme_id_map"),
        # Multi-code-point keys mean a different tokenizer than the one
        # implemented: piper maps one code point to one id.
        ({"phoneme_id_map": {"_": [0], "^": [1], "$": [2], "aɪ": [9]}},
         "multi-code-point map key"),
        ({"phoneme_id_map": {"_": [0], "^": [1], "$": [2], "t": [9, 10]}},
         "multi-id map value"),
        # Framing ids are hardcoded in the runtime's tokenizer.
        ({"phoneme_id_map": {"_": [3], "^": [1], "$": [2]}}, "framing symbol"),
        ({"phoneme_id_map": {"_": [0], "^": [1], "$": [2], "t": [0]}},
         "used by more than one symbol"),
    ],
)
def test_piper_phoneme_table_refuses_a_different_contract(overrides, message) -> None:
    with pytest.raises(conv.ConversionError, match=message):
        conv.parse_piper_phoneme_config("v", _piper_config(**overrides))


def test_schwa_fallback_falls_back_to_zero_without_schwa() -> None:
    config = _piper_config(phoneme_id_map={"_": [0], "^": [1], "$": [2], "t": [32]})
    table = conv.parse_piper_phoneme_config("v", config)
    assert conv._schwa_fallback_id(table) == 0


# ---------------------------------------------------------------------------
# nano_q8_meta.h parsing
# ---------------------------------------------------------------------------


def test_nano_meta_parses_defines_and_ignores_comments(tmp_path: Path) -> None:
    path = tmp_path / "nano_q8_meta.h"
    path.write_text(
        "/* generated -- do not edit */\n"
        "#pragma once\n"
        "\n"
        "#define NANO_VOCAB 62\n"
        "#define NANO_HOP 256\n"
        "/* byte offsets */\n"
        "#define NOFF_DUR_EMB_F32 0\n",
        encoding="utf-8",
    )
    meta = conv.parse_nano_meta(path)
    assert meta == {"NANO_VOCAB": 62, "NANO_HOP": 256, "NOFF_DUR_EMB_F32": 0}


def test_nano_meta_refuses_an_unexpected_line(tmp_path: Path) -> None:
    # The header is the only description of the blob layout. If its grammar
    # changes, the reconstruction has to fail loudly rather than skip a line.
    path = tmp_path / "nano_q8_meta.h"
    path.write_text("#define NANO_VOCAB 62\n#define NANO_MAGIC (1u << 4)\n", encoding="utf-8")
    with pytest.raises(conv.ConversionError, match="grammar changed"):
        conv.parse_nano_meta(path)


def test_nano_meta_refuses_an_empty_header(tmp_path: Path) -> None:
    path = tmp_path / "nano_q8_meta.h"
    path.write_text("/* nothing here */\n#pragma once\n", encoding="utf-8")
    with pytest.raises(conv.ConversionError, match="no #define"):
        conv.parse_nano_meta(path)


# ---------------------------------------------------------------------------
# Blob reconstruction
# ---------------------------------------------------------------------------


def _blob_with_padded_rows(rows: int, used: int, padded: int, pad_value: float = 0.0) -> bytes:
    matrix = np.zeros((rows, padded), dtype=np.float32)
    matrix[:, :used] = np.arange(rows * used, dtype=np.float32).reshape(rows, used)
    matrix[:, used:] = pad_value
    return matrix.tobytes()


def test_blob_reader_strips_the_n16_row_padding() -> None:
    rows, used, padded = 3, 5, 16
    blob = _blob_with_padded_rows(rows, used, padded)
    scale = np.ones((4,), dtype=np.float32)  # 3 rows padded to a 16-byte grid
    scale[rows:] = 0.0
    blob += scale.tobytes()
    reader = conv.BlobReader("t", blob, len(blob), int8=False)
    weights = reader.weights(0, rows, padded, used, "w")
    assert weights.shape == (rows, used)
    assert weights[0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_blob_reader_refuses_non_zero_row_padding() -> None:
    blob = _blob_with_padded_rows(2, 5, 16, pad_value=0.25)
    reader = conv.BlobReader("t", blob, len(blob), int8=False)
    with pytest.raises(conv.ConversionError, match="row padding is not zero"):
        reader.weights(0, 2, 16, 5, "w")


def test_blob_reader_refuses_a_size_mismatch() -> None:
    with pytest.raises(conv.ConversionError, match="the header says"):
        conv.BlobReader("t", b"\x00" * 16, 32, int8=False)


def test_blob_reader_refuses_reads_past_the_end() -> None:
    reader = conv.BlobReader("t", b"\x00" * 64, 64, int8=False)
    with pytest.raises(conv.ConversionError, match="outside the"):
        reader.weights(32, 4, 16, 16, "w")


def test_blob_reader_coverage_catches_an_unclaimed_array() -> None:
    # Two 64-byte arrays; claim only the first.
    reader = conv.BlobReader("t", b"\x00" * 128, 128, int8=False)
    reader.f32(0, 16, "a")
    with pytest.raises(conv.ConversionError, match="never claimed"):
        reader.check_coverage()


def test_blob_reader_coverage_catches_overlap() -> None:
    reader = conv.BlobReader("t", b"\x00" * 128, 128, int8=False)
    reader.f32(0, 32, "a")
    reader.f32(64, 16, "b")
    with pytest.raises(conv.ConversionError, match="overlaps"):
        reader.check_coverage()


def test_blob_reader_int8_rows_are_one_byte_each() -> None:
    rows, used, padded = 2, 3, 16
    raw = np.zeros((rows, padded), dtype=np.int8)
    raw[:, :used] = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int8)
    reader = conv.BlobReader("t", raw.tobytes(), raw.nbytes, int8=True)
    weights = reader.weights(0, rows, padded, used, "w")
    assert weights.tolist() == [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]


# ---------------------------------------------------------------------------
# Voice name resolution
# ---------------------------------------------------------------------------


def test_heart_nano_is_an_input_alias_for_heartnano() -> None:
    assert conv.resolve_requested(["heart-nano"]) == ["heartnano"]
    # Asking for both spellings converts one voice, not two.
    assert conv.resolve_requested(["heartnano", "heart-nano"]) == ["heartnano"]


def test_unknown_voice_lists_the_known_ones() -> None:
    with pytest.raises(conv.ConversionError, match="known voices"):
        conv.resolve_requested(["nope"])


def test_check_unique_rejects_a_name_alias_collision() -> None:
    def build(name: str, aliases: list[str]) -> conv.VoiceBuild:
        return conv.VoiceBuild(
            name=name, aliases=aliases, sample_rate=24000, length_scale=1.0,
            espeak_voice="gmw/en-US", normalization_lang="a", frontend="misaki",
            decoder="vocos", max_tokens=207, token_symbols=[], token_ids=[],
            bos_id=1, eos_id=2, pad_id=-1, fallback_id=-1,
        )

    with pytest.raises(conv.ConversionError, match="claimed by both"):
        conv.check_unique([build("a", ["shared"]), build("b", ["shared"])])
    with pytest.raises(conv.ConversionError, match="claimed by both"):
        conv.check_unique([build("a", []), build("b", ["a"])])


def test_espeak_voice_mapping_is_explicit() -> None:
    assert conv.resolve_espeak_voice("en-us") == "gmw/en-US"
    assert conv.resolve_espeak_voice("aav/vi") == "aav/vi"
    with pytest.raises(conv.ConversionError, match="unknown espeak voice"):
        conv.resolve_espeak_voice("kl")


# ---------------------------------------------------------------------------
# Real voice packs (skipped without a local copy)
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    env = os.environ.get("KOKOPOP_SANOTTS_CACHE")
    return Path(env).expanduser() if env else conv.DEFAULT_CACHE_DIR


def _artifacts() -> Path:
    return conv.artifact_cache(_cache_dir(), conv.SANOTTS_DEFAULT_ARTIFACT_REVISION)


def _has_vocos(name: str) -> bool:
    voice = conv.VOCOS_VOICES[name]
    base = _artifacts() / voice.directory
    return all(
        (base / f).is_file()
        for f in ("nano_q8_meta.h", "meta.json", voice.front_blob, voice.dec_blob)
    )


def _has_piperlite(name: str) -> bool:
    base = _artifacts() / conv.PIPERLITE_PACKAGES[name]
    return all(
        (base / f).is_file()
        for f in ("manifest.json", "piper-phoneme-config.json", "weights.fp16.bin")
    )


def _build(name: str) -> conv.VoiceBuild:
    return conv.build_voice(name, argparse.Namespace(voice_dir=None), _cache_dir(), None)


SKIP_HINT = (
    "sanoTTS voice data not cached; run tools/convert_sanotts_to_gguf.py once, or set "
    "KOKOPOP_SANOTTS_CACHE"
)


@pytest.mark.parametrize("name", ["heart", "heartnano"])
def test_vocos_reconstruction_matches_the_declared_parameter_count(name: str) -> None:
    if not _has_vocos(name):
        pytest.skip(SKIP_HINT)
    build = _build(name)
    # The layout is reconstructed from a generated C header; the element count
    # matching the source's own total to the unit is what makes it trustworthy.
    assert build.element_count() == build.declared_parameters
    assert build.decoder == "vocos"
    assert build.frontend == "misaki"
    assert build.pad_id == -1
    # 59 is the em dash in this vocabulary, not schwa: no fallback is allowed.
    assert build.fallback_id == -1
    assert len(build.token_symbols) == 62
    assert build.scalar_meta["dec.bins"] == build.scalar_meta["dec.n_fft"] // 2 + 1


@pytest.mark.parametrize("name", ["amy", "kristin"])
def test_piperlite_reconstruction_matches_the_declared_parameter_count(name: str) -> None:
    if not _has_piperlite(name):
        pytest.skip(SKIP_HINT)
    build = _build(name)
    assert build.element_count() == build.declared_parameters
    assert build.decoder == "piperlite"
    assert build.frontend == "piper"
    assert build.pad_id == 0
    assert build.bos_id == 1 and build.eos_id == 2
    assert build.sample_rate == 22050


def test_kristin_carries_the_post_filter_and_amy_does_not() -> None:
    if not (_has_piperlite("amy") and _has_piperlite("kristin")):
        pytest.skip(SKIP_HINT)
    amy = _build("amy")
    kristin = _build("kristin")

    assert amy.scalar_meta["dec.post_filter_channels"] == 0
    assert not [t for t in amy.tensors if "post_filter" in t.suffix]

    assert kristin.scalar_meta["dec.post_filter_channels"] > 0
    assert kristin.scalar_meta["dec.post_filter_layers"] == 1
    # `post_filter_kernel` describes the in/out convs; the units carry their own.
    assert kristin.scalar_meta["dec.post_filter_kernel"] == 9
    assert kristin.scalar_meta["dec.post_filter_unit_kernel"] == 3
    suffixes = {t.suffix for t in kristin.tensors}
    assert "dec.post_filter.in_conv.weight" in suffixes
    assert "dec.post_filter.out_conv.weight" in suffixes
    assert "dec.post_filter.units.0.conv1.weight" in suffixes


def test_mixed_rate_multi_voice_gguf(tmp_path: Path) -> None:
    if not (_has_piperlite("amy") and _has_vocos("heart")):
        pytest.skip(SKIP_HINT)
    output = tmp_path / "mixed.gguf"
    rc = conv.main(
        ["--output", str(output), "--voices", "amy,heart", "--default-voice", "heart"]
    )
    assert rc == 0

    kv, tensors = read_gguf(output)
    assert kv["kokopop.arch"] == "sanotts"
    assert kv["kokopop.sanotts.version"] == conv.SANOTTS_GGUF_VERSION
    # Code and artifact revisions are recorded separately, and the artifact one
    # is the immutable commit the bytes were actually read from.
    assert kv["kokopop.sanotts.source.repo"] == conv.SANOTTS_REPO
    assert kv["kokopop.sanotts.source.code_revision"] == conv.SANOTTS_CODE_REVISION
    assert (
        kv["kokopop.sanotts.source.artifact_revision"]
        == conv.SANOTTS_DEFAULT_ARTIFACT_REVISION
    )
    assert conv.SANOTTS_CODE_REVISION in kv["kokopop.sanotts.source"]
    assert kv["kokopop.voices"] == ["amy", "heart"]
    assert kv["kokopop.default_voice"] == "heart"

    # Two rates in one file, and the global key documents the default voice's.
    assert kv["kokopop.sanotts.voice.0.sample_rate"] == 22050
    assert kv["kokopop.sanotts.voice.1.sample_rate"] == 24000
    assert kv["kokopop.sample_rate"] == 24000

    # Per-voice index and name agree, and each voice declares its frontend.
    assert kv["kokopop.sanotts.voice.0.name"] == "amy"
    assert kv["kokopop.sanotts.voice.1.name"] == "heart"
    assert kv["kokopop.sanotts.voice.0.frontend"] == "piper"
    assert kv["kokopop.sanotts.voice.1.frontend"] == "misaki"
    assert kv["kokopop.sanotts.voice.0.decoder"] == "piperlite"
    assert kv["kokopop.sanotts.voice.1.decoder"] == "vocos"
    assert kv["kokopop.sanotts.voice.1.aliases"] == []

    # The token table arrays are the same length for each voice.
    for index in (0, 1):
        prefix = f"kokopop.sanotts.voice.{index}."
        assert len(kv[prefix + "token_symbols"]) == len(kv[prefix + "token_ids"])

    # The NFD table is global, written once, and self-consistent.
    assert kv["kokopop.sanotts.unicode_version"] == unicodedata.unidata_version
    codepoints = kv["kokopop.sanotts.nfd_codepoints"]
    offsets = kv["kokopop.sanotts.nfd_offsets"]
    values = kv["kokopop.sanotts.nfd_values"]
    assert len(offsets) == len(codepoints) + 1
    assert offsets[-1] == len(values)

    # Tensor shapes follow the manifest dimensions.
    by_logical = logical_tensors(kv, tensors)
    amy_prefix = "kokopop.sanotts.voice.0."
    dur_hidden = kv[amy_prefix + "dur.hidden"]
    dur_vocab = kv[amy_prefix + "dur.vocab"]
    dur_kernel = kv[amy_prefix + "dur.kernel"]
    # Embedding is [vocab, hidden] in numpy, so ne = (hidden, vocab).
    assert by_logical[amy_prefix + "dur.embedding.weight"][0] == (dur_hidden, dur_vocab)
    # conv1d kernels are flattened to [OC, IC*K].
    assert by_logical[amy_prefix + "dur.blocks.0.net0.weight"][0] == (
        dur_hidden * dur_kernel, dur_hidden,
    )
    # ConvTranspose1d keeps three dimensions: ne = (K, OC, IC).
    channels = kv[amy_prefix + "dec.channels"]
    assert by_logical[amy_prefix + "dec.up0.weight"][0] == (16, channels[1], channels[0])

    # The duration head is F32 on purpose: round(exp(x)) is a step function and
    # a fp16 difference there changes the audio length.
    assert by_logical[amy_prefix + "dur.output.weight"][1] == conv.GGML_TYPE_F32
    heart_prefix = "kokopop.sanotts.voice.1."
    for suffix in ("dec.blocks.0.gamma", "dec.norm.weight", "dec.head.bias"):
        assert by_logical[heart_prefix + suffix][1] == conv.GGML_TYPE_F32


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    if not _has_vocos("heart"):
        pytest.skip(SKIP_HINT)
    output = tmp_path / "nope.gguf"
    assert conv.main(["--output", str(output), "--voices", "heart", "--dry-run"]) == 0
    assert not output.exists()


def test_vocos_refuses_a_tampered_blob(tmp_path: Path) -> None:
    if not _has_vocos("heartnano"):
        pytest.skip(SKIP_HINT)
    voice = conv.VOCOS_VOICES["heartnano"]
    source = _artifacts() / voice.directory
    for name in ("nano_q8_meta.h", "meta.json", voice.front_blob, voice.dec_blob):
        (tmp_path / name).write_bytes((source / name).read_bytes())

    blob = tmp_path / voice.dec_blob
    data = bytearray(blob.read_bytes())
    data[len(data) // 2] ^= 0xFF
    blob.write_bytes(bytes(data))

    with pytest.raises(conv.ConversionError, match="sha256 mismatch"):
        conv.build_vocos_voice(
            "heartnano", [], tmp_path / "nano_q8_meta.h", tmp_path / "meta.json",
            tmp_path / voice.front_blob, blob, conv.load_vocos_vocabulary(None),
        )


@pytest.mark.parametrize(
    ("define", "value", "message"),
    [
        ("NANO_NORM_TYPE", 1, "DyT"),
        ("NANO_ACT_TYPE", 1, "ReLU"),
        ("NANO_FORMAT_VERSION", 2, "only 1 is supported"),
        ("NANO_BINS", 512, "!= n_fft/2"),
    ],
)
def test_vocos_refuses_unsupported_operators(tmp_path: Path, define, value, message) -> None:
    if not _has_vocos("heartnano"):
        pytest.skip(SKIP_HINT)
    voice = conv.VOCOS_VOICES["heartnano"]
    source = _artifacts() / voice.directory
    for name in ("meta.json", voice.front_blob, voice.dec_blob):
        (tmp_path / name).write_bytes((source / name).read_bytes())

    header = (source / "nano_q8_meta.h").read_text(encoding="utf-8")
    patched = []
    for line in header.splitlines():
        if line.startswith(f"#define {define} "):
            line = f"#define {define} {value}"
        patched.append(line)
    (tmp_path / "nano_q8_meta.h").write_text("\n".join(patched) + "\n", encoding="utf-8")

    with pytest.raises(conv.ConversionError, match=message):
        conv.build_vocos_voice(
            "heartnano", [], tmp_path / "nano_q8_meta.h", tmp_path / "meta.json",
            tmp_path / voice.front_blob, tmp_path / voice.dec_blob,
            conv.load_vocos_vocabulary(None),
        )


def test_piperlite_refuses_an_acoustic_adapter(tmp_path: Path) -> None:
    if not _has_piperlite("amy"):
        pytest.skip(SKIP_HINT)
    source = _artifacts() / conv.PIPERLITE_PACKAGES["amy"]
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    weights = (source / "weights.fp16.bin").read_bytes()

    # Rename one acoustic tensor so it looks like an output adapter. The runtime
    # only implements the adapter-free path; guessing at the layout of one that
    # is not implemented is exactly what must not happen.
    acoustic = manifest["components"]["acoustic"]["tensors"]
    acoustic[0]["name"] = "adapter." + acoustic[0]["name"]

    (tmp_path / "weights.fp16.bin").write_bytes(weights)
    manifest["weights_sha256"] = hashlib.sha256(weights).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "piper-phoneme-config.json").write_bytes(
        (source / "piper-phoneme-config.json").read_bytes()
    )

    pack = conv.load_voice_pack("amy", tmp_path)
    with pytest.raises(conv.ConversionError, match="output adapter"):
        conv.build_piperlite_voice("amy", pack, [])


def test_piperlite_refuses_a_corrupt_weights_blob(tmp_path: Path) -> None:
    if not _has_piperlite("amy"):
        pytest.skip(SKIP_HINT)
    source = _artifacts() / conv.PIPERLITE_PACKAGES["amy"]
    for name in ("manifest.json", "piper-phoneme-config.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    data = bytearray((source / "weights.fp16.bin").read_bytes())
    data[1000] ^= 0xFF
    (tmp_path / "weights.fp16.bin").write_bytes(bytes(data))

    with pytest.raises(conv.ConversionError, match="sha256 mismatch"):
        conv.load_voice_pack("amy", tmp_path)


def test_piperlite_refuses_an_unknown_manifest_format(tmp_path: Path) -> None:
    if not _has_piperlite("amy"):
        pytest.skip(SKIP_HINT)
    source = _artifacts() / conv.PIPERLITE_PACKAGES["amy"]
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["format"] = "roota.raw-fp16.v2"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "weights.fp16.bin").write_bytes((source / "weights.fp16.bin").read_bytes())
    (tmp_path / "piper-phoneme-config.json").write_bytes(
        (source / "piper-phoneme-config.json").read_bytes()
    )
    with pytest.raises(conv.ConversionError, match="unsupported manifest format"):
        conv.load_voice_pack("amy", tmp_path)


# ---------------------------------------------------------------------------
# The vocos vocabulary table
# ---------------------------------------------------------------------------


def test_bundled_vocos_vocabulary_is_well_formed() -> None:
    vocab = conv.load_vocos_vocabulary(None)
    assert len(vocab) == 62
    assert sorted(vocab.values()) == list(range(62))
    assert vocab["<pad>"] == 0 and vocab["<bos>"] == 1 and vocab["<eos>"] == 2
    # 59 is the em dash here, which is why the misaki path has no schwa
    # fallback: remapping an out-of-vocab id to 59 would insert a dash.
    assert vocab["—"] == 59
    assert vocab["ə"] == 42


def test_vocos_vocabulary_decodes_the_golden_ids_to_plausible_ipa() -> None:
    """The table is the model's frozen input interface, so it is validated the
    only way that is meaningful without the checkpoint: decode the golden
    fixture's ids and check the result is well-formed misaki IPA.
    """
    root = os.environ.get("KOKOPOP_SANOTTS_FIXTURES") or str(
        Path.home() / ".cache/sanotts/upstream/mcu/test/fixtures"
    )
    ids_path = Path(root) / "en_us_r227f32" / "r00_ids.bin"
    if not ids_path.is_file():
        pytest.skip("sanoTTS golden fixtures not found; set KOKOPOP_SANOTTS_FIXTURES")

    vocab = conv.load_vocos_vocabulary(None)
    inverse = {v: k for k, v in vocab.items()}
    ids = np.fromfile(ids_path, dtype=np.int32).tolist()

    assert ids[0] == vocab["<bos>"]
    assert ids[-1] == vocab["<eos>"]
    assert all(0 <= i < 62 for i in ids)
    # No PAD interleaving on this path.
    assert vocab["<pad>"] not in ids

    decoded = "".join(inverse[i] for i in ids[1:-1])
    # The row is an English sentence; the decode must be readable IPA and not
    # a permutation of it.
    assert decoded.startswith("ði əkˈustɪk stˈudᵊnt")
    assert decoded.endswith("klˈɪpt.")
    # Every decoded symbol is a phoneme, punctuation or a stress mark — never a
    # framing token.
    assert not any(symbol in decoded for symbol in ("<pad>", "<bos>", "<eos>"))
