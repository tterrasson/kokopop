#!/usr/bin/env python3
"""Generate piperlite parity fixtures from the upstream numpy reference.

The vocos voices ship golden fixtures upstream (`mcu/test/fixtures/**`); the
piperlite voices do not. This script fills that gap by running
`pypkg/sanotts/models.py` — the reference forward pass the shipped voices are
gated against — over a versioned corpus of token id sequences, and writing the
same `(ids, durs, latent, audio)` layout the C++ tests already read.

Nothing produced here is committed: the weight packs it reads are not
redistributable, so the fixtures inherit that. `tests/test_sanotts_decoder.cpp`
skips cleanly when the directory is absent.

    uv run tools/gen_sanotts_fixtures.py --voice amy --voice kristin

Reads the voice packs from the converter's own HF cache and the reference
implementation from a checkout of the pinned upstream revision. Both paths are
overridable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from convert_sanotts_to_gguf import (  # noqa: E402
    SANOTTS_CODE_REVISION,
    VOICE_ALIASES,
    ConversionError,
    load_voice_pack,
)

logger = logging.getLogger("sanotts.fixtures")

DEFAULT_UPSTREAM = Path.home() / ".cache" / "sanotts" / "upstream"
DEFAULT_PACKS = Path.home() / ".cache" / "sanotts" / "hf"
DEFAULT_OUT = Path.home() / ".cache" / "sanotts" / "piperlite-fixtures"

# The corpus is token ids, not text: the point of these fixtures is to gate the
# duration model, the acoustic model and the decoder, and feeding them ids
# removes espeak-ng's version from the comparison entirely. G2P parity has its
# own gate (tests/data/sanotts_g2p_corpus.json).
#
# Produced by kokopop's own tokenizer, so the sequences are exactly what the
# runtime will see: BOS, PAD-interleaved phonemes, EOS.
CORPUS_VERSION = 1


def load_corpus(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("version", 0)) != CORPUS_VERSION:
        raise ConversionError(f"{path}: expected corpus version {CORPUS_VERSION}")
    return list(data["rows"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", action="append", default=[], help="voice name, repeatable")
    parser.add_argument("--corpus", type=Path,
                        default=Path(__file__).parent.parent / "tests" / "data" / "sanotts_piperlite_corpus.json")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM,
                        help=f"checkout of Ampixa/sanoTTS at {SANOTTS_CODE_REVISION}")
    parser.add_argument("--packs", type=Path, default=DEFAULT_PACKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sys.path.insert(0, str(args.upstream / "pypkg"))
    try:
        from sanotts import models  # noqa: PLC0415
    except ImportError as exc:
        raise ConversionError(
            f"cannot import the reference implementation from {args.upstream / 'pypkg'}: {exc}"
        ) from exc

    corpus = load_corpus(args.corpus)
    voices = [VOICE_ALIASES.get(v, v) for v in (args.voice or ["amy", "kristin"])]

    for voice in voices:
        rows = [row for row in corpus if row["voice"] == voice]
        if not rows:
            logger.warning("%s: no corpus row, skipped", voice)
            continue

        package = rows[0]["package"]
        pack = load_voice_pack(package, args.packs / package)
        length_scale = float(pack.manifest.get("inference", {}).get("duration_length_scale", 1.0))

        directory = args.out / voice
        directory.mkdir(parents=True, exist_ok=True)

        for index, row in enumerate(rows):
            ids = np.asarray(row["ids"], dtype=np.int64)
            durations = models.duration_forward(
                pack.tensors("duration"), pack.component("duration")["config"],
                ids, length_scale=length_scale,
            )
            latent = models.acoustic_forward(
                pack.tensors("acoustic"), pack.component("acoustic")["config"],
                ids, durations,
            )
            audio = models.decoder_forward(
                pack.tensors("decoder"), pack.component("decoder")["config"], latent,
            )

            stem = directory / f"r{index:02d}"
            ids.astype("<i4").tofile(f"{stem}_ids.bin")
            durations.astype("<i4").tofile(f"{stem}_durs.bin")
            # [out_channels, frames] in numpy; the runtime reads a channel's
            # time series contiguously, which is the same order.
            latent.astype("<f4").tofile(f"{stem}_latent.bin")
            audio.astype("<f4").tofile(f"{stem}_audio.bin")
            logger.info("%s r%02d: %d tokens, %d frames, %d samples",
                        voice, index, ids.size, int(durations.sum()), audio.size)

        (directory / "rows.txt").write_text(
            "".join(
                f"r{index:02d} {len(row['ids'])} {row['text']}\n"
                for index, row in enumerate(rows)
            ),
            encoding="utf-8",
        )
        (directory / "meta.json").write_text(
            json.dumps({
                "voice": voice,
                "package": package,
                "length_scale": length_scale,
                "code_revision": SANOTTS_CODE_REVISION,
                "corpus_version": CORPUS_VERSION,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("%s: wrote %d rows to %s", voice, len(rows), directory)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConversionError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
