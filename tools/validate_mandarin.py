#!/usr/bin/env python3
"""
Validate Mandarin pronunciation by round-tripping text → TTS audio → Whisper transcription.

Usage:
    uv run python tools/validate_mandarin.py "你好，世界"
    uv run python tools/validate_mandarin.py --voice zf_xiaoni "今天天气不错"
    uv run python tools/validate_mandarin.py --model models/kokoro-big.gguf "我喜欢编程"
"""

import argparse
import subprocess
import sys
import tempfile
import unicodedata
from functools import partial
from pathlib import Path

from opencc import OpenCC


class C:
    """ANSI colour helpers."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def colour(text: str, fg: str, bold: bool = False) -> str:
    prefix = C.BOLD + fg if bold else fg
    return f"{prefix}{text}{C.RESET}"


green = partial(colour, fg=C.GREEN, bold=True)
red = partial(colour, fg=C.RED, bold=True)
yellow = partial(colour, fg=C.YELLOW)
cyan = partial(colour, fg=C.CYAN, bold=True)
dim = partial(colour, fg=C.DIM)


# Force traditional → simplified conversion so comparisons are consistent
# regardless of which character variant Whisper happens to emit.
CC_TW2SP = OpenCC("tw2sp")

MANDARIN_PUNCTUATION = set(
    "，。！？、；："
    + '"' * 2
    + "''（）【】《》〈〉〔〕｛｝＃＄％＆＊＋，－．／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
    + ".,;:!?()[]{}<>"  # ASCII punctuation also stripped
)


def normalise(text: str) -> str:
    """Normalise Mandarin text for comparison: convert to simplified Chinese, strip punctuation, whitespace, full-width chars."""
    text = unicodedata.normalize("NFC", text)
    text = CC_TW2SP.convert(text)  # guarantee simplified Chinese
    return "".join(
        ch for ch in text if ch not in MANDARIN_PUNCTUATION and not ch.isspace()
    )


def levenshtein(a: str, b: str) -> int:
    """Classic DP Levenshtein distance."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "models" / "kokoro.gguf")
DEFAULT_VOICE = "zf_xiaobei"


def generate_audio(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    voice: str = DEFAULT_VOICE,
    output: Path,
    kokopop_say_bin: str = "./build/kokopop_say",
) -> None:
    """Run kokopop_say to synthesise *text* into a WAV file at *output*."""
    cmd = [
        kokopop_say_bin,
        "--model",
        model,
        "--voice",
        voice,
        "--text",
        text,
        "--out",
        str(output),
    ]
    print(cyan("  cmd: "), " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(red(f"\n  kokopop_say failed (exit {result.returncode})"))
        if result.stderr:
            print(dim(f"  stderr:\n{result.stderr}"))
        sys.exit(result.returncode)
    print(green(f"  ✓ Audio written → {output}"))


def transcribe(audio_path: Path, *, model_name: str = "medium") -> dict:
    """Load Whisper and transcribe *audio_path* in Mandarin."""
    print(cyan(f"\n🎙  Loading Whisper ({model_name}) …"))
    import whisper

    model = whisper.load_model(model_name)
    print(green("  ✓ Model loaded"))

    print(cyan(f"  Transcribing → {audio_path}"))
    result = model.transcribe(str(audio_path), language="zh", verbose=False)
    return result


def report(original: str, transcription: dict) -> bool:
    """Print a colourful comparison report. Returns True on match."""
    text = transcription["text"].strip()
    orig_norm = normalise(original)
    text_norm = normalise(text)

    exact = text_norm == orig_norm
    distance = levenshtein(orig_norm, text_norm)
    similarity = 1.0 - (distance / max(len(orig_norm), len(text_norm), 1))

    width = 64
    print()
    print("=" * width)
    print(cyan("  📊 Mandarin Pronunciation Validation Report"))
    print("=" * width)

    print(f"\n  {colour('Original text:', C.BOLD)}")
    print(f"    {original}")

    print(f"\n  {colour('Whisper transcription:', C.BOLD)}")
    print(f"    {text}")

    print(f"\n  {colour('Normalised comparison:', C.BOLD)}")
    print(f"    Original:      {orig_norm}")
    print(f"    Transcribed:   {text_norm}")

    print(f"\n  {colour('Metrics:', C.BOLD)}")
    print(f"    Characters (original):    {len(orig_norm)}")
    print(f"    Characters (transcribed): {len(text_norm)}")
    print(f"    Levenshtein distance:     {distance}")
    print(f"    Similarity:               {similarity:.1%}")

    # Timing info if available
    segments = transcription.get("segments", [])
    if segments:
        print(f"\n  {colour('Segments:', C.BOLD)}")
        for i, seg in enumerate(segments):
            start = f"{seg['start']:.2f}s"
            end = f"{seg['end']:.2f}s"
            seg_text = seg.get("text", "").strip()
            print(f"    [{i}] {dim(start)} → {dim(end)}  {seg_text}")

    print()
    print("-" * width)
    if exact:
        print(green("  ✅ PERFECT MATCH — pronunciation validated!"))
        print("-" * width)
        return True
    elif similarity >= 0.8:
        print(
            yellow(f"  ⚠️  CLOSE MATCH ({similarity:.0%}) — minor differences detected")
        )
        print("-" * width)
        show_diff(orig_norm, text_norm)
        return False
    else:
        print(
            red(
                f"  ❌ MISMATCH ({similarity:.0%}) — pronunciation differs significantly"
            )
        )
        print("-" * width)
        show_diff(orig_norm, text_norm)
        return False


def show_diff(a: str, b: str) -> None:
    """Show a character-level diff between two normalised strings."""
    max_len = max(len(a), len(b))
    a_padded = a.ljust(max_len)
    b_padded = b.ljust(max_len)
    diff_line = []
    for ca, cb in zip(a_padded, b_padded):
        if ca == cb:
            diff_line.append(" ")
        elif ca == "":
            diff_line.append(red("^"))
        elif cb == "":
            diff_line.append(red("↓"))
        else:
            diff_line.append(yellow("↑↓"))
    print()
    print(f"    Expected:   {a}")
    print(f"    Got:        {b}")
    print(f"    {'':16s}{''.join(diff_line)}")
    print(f"    {'':16s}{''.join('─' if ch == ' ' else ch for ch in diff_line)}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Mandarin pronunciation via TTS + Whisper round-trip.",
    )
    parser.add_argument("text", help="Mandarin text to validate")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to GGUF model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Voice name (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--whisper-model",
        default="medium",
        help="Whisper model size: tiny, base, small, medium, large (default: medium)",
    )
    parser.add_argument(
        "--bin",
        default="./build/kokopop_say",
        help="Path to kokopop_say binary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(cyan("\n🇨🇳  Mandarin Pronunciation Validator"))
    print(cyan(f"  Text: {args.text}"))
    print()

    print(cyan("🔊 Step 1: Generating TTS audio …"))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        generate_audio(
            args.text,
            model=args.model,
            voice=args.voice,
            output=wav_path,
            kokopop_say_bin=args.bin,
        )

        print(cyan("\n🎙  Step 2: Transcribing with Whisper …"))
        result = transcribe(wav_path, model_name=args.whisper_model)

        print(cyan("\n📊 Step 3: Comparing …"))
        matched = report(args.text, result)
        sys.exit(0 if matched else 1)
    finally:
        if wav_path.exists():
            wav_path.unlink()


if __name__ == "__main__":
    main()
