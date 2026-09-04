#!/usr/bin/env python3
"""Generate the sanoTTS G2P parity corpus.

Grapheme-to-phoneme divergence is the top risk in this project: wrong phoneme
ids produce confident, wrong audio with no error anywhere.
So the gate is exact equality of *ids* against the reference implementation, on
a pinned corpus and a pinned espeak-ng, and this script is what produces the
reference side of it.

The two contracts have different references, and both are runnable:

  piper    `pypkg/sanotts/frontend.py` from Ampixa/sanoTTS (MIT): phonemizer
           with `preserve_punctuation`, Piper's punctuation set, `with_stress`,
           `tie=False`, `language_switch="remove-flags"`, then `rstrip()`, NFD,
           and Piper's `[BOS, PAD] + (id, PAD)* + [EOS]` framing.

  misaki   phonemizer with `tie='^'` and the default punctuation set, then
           misaki's own `EspeakFallback` rewrite tail (misaki is Apache-2.0, so
           its table is read from the package rather than transcribed), then
           `[BOS] + ids + [EOS]` over the 62-symbol vocabulary.

Neither reference is vendored: this script imports them, and the corpus it
writes is kokopop's own output.

    uv run --with phonemizer-fork --with espeakng-loader --with misaki \\
        python tools/gen_sanotts_g2p_corpus.py

Add `--use-system-espeak` to run against the system espeak-ng instead of the
one espeakng-loader bundles. The two agree on the shipped corpus; the flag is
there so a divergence can be attributed rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import convert_sanotts_to_gguf as conv  # noqa: E402

OUTPUT = REPO_ROOT / "tests" / "data" / "sanotts_g2p_corpus.json"

PIPER_PUNCTUATION = "!'(),-.:;?\""

# The corpus. Chosen to exercise what actually breaks a G2P port rather than to
# be large: clause and sentence punctuation, quotes and parentheses, numbers and
# ordinals, abbreviations, hyphenation, casing, apostrophes, the syllabic
# consonants misaki rewrites, the diphthongs the tie disambiguates, non-ASCII
# input, and the degenerate inputs (empty-ish, punctuation only).
ENGLISH_CORPUS = [
    # Plain sentences.
    "Hello world.",
    "The acoustic student should speak clearly without sounding rushed or clipped.",
    "The teacher model needs stable timing without sounding rushed or clipped.",
    "The assistant should avoid metallic noise without sounding rushed or clipped.",
    "The teacher model must handle unfamiliar wording without sounding rushed or clipped.",
    "The acoustic student needs natural pauses without sounding rushed or clipped.",
    "The assistant needs reliable pronunciation without sounding rushed or clipped.",
    # Two sentences: the second clause gets its own espeak context.
    "The assistant should avoid metallic noise. Variant two.",
    "The teacher model must handle unfamiliar wording. Variant three.",
    # Internal punctuation.
    "Please read item eighty eight with a calm pause, then continue with the next phrase.",
    "Please read item one hundred twenty with a calm pause, then continue with the next phrase.",
    "First, second, third; then finally: done.",
    "Wait -- what happened?",
    "Really?! Are you sure?",
    # Syllabic consonants, which misaki rewrites to a superscript schwa.
    "The model handles a little bottle of metal.",
    "A sudden button, a hidden kitten, a rotten mitten.",
    # Diphthongs and affricates: the tie is what makes these unambiguous.
    "The choice of voice made the boys rejoice.",
    "How now brown cow, out about the house.",
    "Judge Jones changed the ledger.",
    "Watch the cheap chair, teacher.",
    "Go home, so slow, no hope.",
    "The rain in Spain came late today.",
    # Rhotics and the ɜː family.
    "Her early bird heard the word first.",
    "A better letter for the doctor's daughter.",
    # Numbers, ordinals, symbols.
    "Chapter 3, verse 16.",
    "It costs 42 dollars and 7 cents.",
    "The 1st, 2nd and 3rd attempts failed.",
    "100% of 250 users, at 3.5 percent.",
    # Casing, apostrophes, hyphens.
    "DON'T PANIC.",
    "It's the cat's pyjamas, isn't it?",
    "A well-known, half-baked, state-of-the-art idea.",
    "iPhone, eBay and NASA walk into a bar.",
    # Quotes and brackets.
    'He said "hello" and left.',
    "She replied (quietly) and smiled.",
    "“Curly quotes” and — an em dash — too.",
    # Abbreviations and initialisms.
    "Dr. Smith visited Mr. Jones at 5 p.m.",
    "The U.S.A. and the U.K. signed it.",
    # Non-ASCII input that still has to phonemize.
    "Café naïve résumé.",
    "The Straße was crowded.",
    # Whitespace and degenerate inputs.
    "  leading and trailing spaces  ",
    "double  spaces   inside",
    "one",
    "a",
    # Punctuation only: the reference has a dedicated branch for it.
    "...",
    "?!",
    ",",
]

# Voices to cover. Each entry is (kokopop voice name, espeak voice id).
# The non-English piperlite packs use the same code path with a different
# espeak voice, which is exactly the thing worth covering.
NON_ENGLISH_CORPUS = {
    "vi": [
        "Xin chào thế giối.",
        "Tôi muốn đọc một câu dài hơn, rồi dừng lại.",
        "Một, hai, ba.",
    ],
    "id": [
        "Halo dunia.",
        "Saya ingin membaca kalimat yang lebih panjang, lalu berhenti.",
        "Satu, dua, tiga.",
    ],
}


def make_backend(espeak_voice: str, tie: str | None, punctuation: str | None):
    from phonemizer.backend import EspeakBackend

    kwargs = {
        "preserve_punctuation": True,
        "with_stress": True,
        "language_switch": "remove-flags",
    }
    if tie is not None:
        kwargs["tie"] = tie
    else:
        kwargs["tie"] = False
    if punctuation is not None:
        kwargs["punctuation_marks"] = punctuation

    # Older packs (kristin) record a bare "en", which current espeak-ng only
    # exposes as a secondary tag. Upstream's frontend.py has the same shim; the
    # regional variant it lands on is the one the corpus records, so kokopop
    # selects the same voice rather than a different accent.
    candidates = [espeak_voice]
    if "-" not in espeak_voice and "/" not in espeak_voice:
        candidates += [f"{espeak_voice}-us", f"{espeak_voice}-gb"]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return EspeakBackend(candidate, **kwargs)
        except Exception as exc:  # noqa: BLE001 - probing voice-name spellings
            last_error = exc
    raise RuntimeError(
        f"espeak-ng has no voice matching {espeak_voice!r} (tried {candidates})"
    ) from last_error


def configure_espeak(use_system: bool) -> str:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    if not use_system:
        import espeakng_loader

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    version = EspeakWrapper().version
    return ".".join(str(v) for v in version)


def phonemize_piper(backend, text: str) -> str:
    out = backend.phonemize([text], strip=False, separator=None)
    if not out:
        return ""
    # phonemizer appends a separator space that Piper's own clause-based bridge
    # does not emit at true end of input.
    return out[0].rstrip()


def phonemize_misaki(backend, text: str) -> str:
    from misaki.espeak import EspeakFallback

    out = backend.phonemize([text], strip=False, separator=None)
    if not out:
        return ""
    ps = out[0].strip()
    for old, new in EspeakFallback.E2M:
        ps = ps.replace(old, new)
    ps = re.sub(r"(\S)̩", r"ᵊ\1", ps).replace(chr(809), "")
    # american (british=False)
    ps = ps.replace("o^ʊ", "O")
    ps = ps.replace("ɜːɹ", "ɜɹ")
    ps = ps.replace("ɜː", "ɜɹ")
    ps = ps.replace("ɪə", "iə")
    ps = ps.replace("ː", "")
    ps = ps.replace("o", "ɔ")
    ps = ps.replace("ɾ", "T").replace("ʔ", "t")
    return ps.replace("^", "")


def piper_ids(phonemes: str, id_map: dict[str, int]) -> list[int]:
    """Piper's framing, exactly as pypkg/sanotts/frontend.py does it."""
    ids = [1, 0]  # BOS, PAD
    for ch in unicodedata.normalize("NFD", phonemes):
        pid = id_map.get(ch)
        if pid is None:
            continue
        ids.append(pid)
        ids.append(0)
    ids.append(2)  # EOS
    return ids


def misaki_ids(phonemes: str, vocabulary: dict[str, int]) -> list[int]:
    specials = {"<pad>", "<bos>", "<eos>"}
    ids = [vocabulary["<bos>"]]
    for ch in phonemes:
        if ch in vocabulary and ch not in specials:
            ids.append(vocabulary[ch])
    ids.append(vocabulary["<eos>"])
    return ids


def build_corpus_nfd_subset(entries: list[dict[str, object]]) -> dict[str, object]:
    """The canonical-decomposition rows this corpus actually exercises.

    The full table is ~4500 rows; carrying all of it in a JSON fixture would be
    100 kB of noise. `build_nfd_table()` is validated against Python's Unicode
    database in python/tests/test_sanotts_converter.py, so what matters here is
    that the C++ decomposition path sees the rows the corpus depends on — plus a
    few deliberately awkward extras.
    """
    full = conv.build_nfd_table()
    by_cp = {cp: i for i, cp in enumerate(full.codepoints)}
    ccc = dict(zip(full.ccc_codepoints, full.ccc_classes))

    wanted: set[int] = set()
    for entry in entries:
        for field_name in ("text", "phonemes"):
            for ch in str(entry[field_name]):
                wanted.add(ord(ch))
    # Extras: a two-mark sequence that needs canonical reordering, a Hangul
    # syllable (three-value row), and a recursive decomposition.
    wanted.update({0x00E9, 0x00E7, 0x01D5, 0x1E17, 0xAC01, 0x0407, 0x1FA7, 0x0104})

    codepoints: list[int] = []
    offsets: list[int] = [0]
    values: list[int] = []
    for cp in sorted(wanted):
        index = by_cp.get(cp)
        if index is None:
            continue
        start, end = full.offsets[index], full.offsets[index + 1]
        codepoints.append(cp)
        values.extend(full.values[start:end])
        offsets.append(len(values))

    # The combining-class table is *not* subsetted. Canonical ordering needs the
    # class of every mark it meets, and a mark whose class is missing is
    # silently treated as a starter -- which reorders nothing and yields
    # different ids. It is ~900 rows; the decomposition rows are what was large.
    ccc_codepoints = sorted(ccc)
    return {
        "_comment": "Subset of the canonical-decomposition table this corpus exercises.",
        "codepoints": codepoints,
        "offsets": offsets,
        "values": values,
        "ccc_codepoints": ccc_codepoints,
        "ccc_classes": [ccc[cp] for cp in ccc_codepoints],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--use-system-espeak", action="store_true",
        help="use the system espeak-ng instead of the espeakng-loader bundle",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=conv.DEFAULT_CACHE_DIR,
        help="voice-pack cache, for the Piper phoneme tables",
    )
    args = parser.parse_args(argv)

    espeak_version = configure_espeak(args.use_system_espeak)
    print(f"espeak-ng {espeak_version}", file=sys.stderr)

    entries: list[dict[str, object]] = []

    # -- misaki / vocos ----------------------------------------------------
    vocabulary = conv.load_vocos_vocabulary(None)
    misaki_backend = make_backend("en-us", tie="^", punctuation=None)
    for text in ENGLISH_CORPUS:
        phonemes = phonemize_misaki(misaki_backend, text)
        if not phonemes:
            continue
        entries.append(
            {
                "frontend": "misaki",
                "voice": "heart",
                "espeak_voice": "gmw/en-US",
                "text": text,
                "phonemes": phonemes,
                "ids": misaki_ids(phonemes, vocabulary),
            }
        )

    # -- piper / piperlite -------------------------------------------------
    voice_texts = {"amy": ENGLISH_CORPUS, "kristin": ENGLISH_CORPUS}
    voice_texts.update(NON_ENGLISH_CORPUS)
    for voice, texts in voice_texts.items():
        package = conv.PIPERLITE_PACKAGES[voice]
        directory = Path(args.cache_dir).expanduser() / "hf" / package
        config_path = directory / "piper-phoneme-config.json"
        if not config_path.is_file():
            print(f"skipping {voice}: {config_path} not found", file=sys.stderr)
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        table = conv.parse_piper_phoneme_config(voice, config)
        id_map = dict(zip(table.symbols, table.ids))
        espeak_voice = conv.resolve_espeak_voice(table.espeak_voice)
        backend = make_backend(table.espeak_voice, tie=None,
                               punctuation=PIPER_PUNCTUATION)
        for text in texts:
            phonemes = phonemize_piper(backend, text)
            if not phonemes:
                continue
            entries.append(
                {
                    "frontend": "piper",
                    "voice": voice,
                    "espeak_voice": espeak_voice,
                    "text": text,
                    "phonemes": phonemes,
                    "ids": piper_ids(phonemes, id_map),
                }
            )

    # -- the tables the runtime needs to reproduce these ids ---------------
    #
    # Shipping them with the corpus keeps the C++ gate self-contained: it needs
    # no converted GGUF and no voice pack. The GGUF's copies are produced by the
    # same functions, and the C++ test cross-checks them when a model is
    # present.
    token_tables: dict[str, object] = {}
    for voice, entry_voice in (("heart", "misaki"), ("heartnano", "misaki")):
        token_tables[voice] = {
            "frontend": entry_voice,
            "symbols": [s for s, _ in sorted(vocabulary.items(), key=lambda kv: kv[1])],
            "ids": sorted(vocabulary.values()),
            "bos_id": vocabulary["<bos>"],
            "eos_id": vocabulary["<eos>"],
            "pad_id": -1,
            "fallback_id": -1,
            "special_symbols": ["<pad>", "<bos>", "<eos>"],
        }
    for voice in ("amy", "kristin", "vi", "id"):
        package = conv.PIPERLITE_PACKAGES[voice]
        config_path = Path(args.cache_dir).expanduser() / "hf" / package / "piper-phoneme-config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        table = conv.parse_piper_phoneme_config(voice, config)
        token_tables[voice] = {
            "frontend": "piper",
            "symbols": table.symbols,
            "ids": table.ids,
            "bos_id": table.bos_id,
            "eos_id": table.eos_id,
            "pad_id": table.pad_id,
            "fallback_id": conv._schwa_fallback_id(table),
            "special_symbols": ["_", "^", "$"],
        }

    nfd = build_corpus_nfd_subset(entries)

    document = {
        "_comment": [
            "Generated by tools/gen_sanotts_g2p_corpus.py. Do not edit by hand.",
            "The reference side of the G2P parity gate: kokopop must",
            "reproduce these phoneme strings and id sequences exactly.",
            "Produced from phonemizer-fork + misaki + the pinned espeak-ng below; a",
            "change in any of those has to be re-generated deliberately, not absorbed",
            "by loosening the gate.",
        ],
        "espeak_ng_version": espeak_version,
        "espeak_source": "system" if args.use_system_espeak else "espeakng-loader",
        "sanotts_revision": conv.SANOTTS_REVISION,
        "unicode_version": unicodedata.unidata_version,
        "nfd": nfd,
        "token_tables": token_tables,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for entry in entries:
        counts[str(entry["frontend"])] = counts.get(str(entry["frontend"]), 0) + 1
    print(f"wrote {args.output} ({len(entries)} entries: {counts})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
