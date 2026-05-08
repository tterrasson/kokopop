#!/usr/bin/env python3
"""
Minimal Python client for the kokopop HTTP TTS server.

Usage:
    # Stream PCM float32 and play via ffplay
    uv run python tools/tts_client.py "Hello world" | ffplay -f f32le -ar 24000 -ac 1 -

    # Read text from a file
    uv run python tools/tts_client.py --file story.txt | ffplay -f f32le -ar 24000 -ac 1 -

    # Stream PCM and save as WAV
    uv run python tools/tts_client.py "Hello world" --out hello.wav

    # Receive a complete WAV file directly
    uv run python tools/tts_client.py "Hello world" --format wav --out hello.wav
"""

from __future__ import annotations

import argparse
import struct
import sys
import urllib.request
import json
from pathlib import Path


def _pcm_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw float32 PCM bytes in a WAV container (IEEE float, mono)."""
    num_channels = 1
    bits_per_sample = 32
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 3, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )
    return header + pcm


def stream_pcm(
    text: str,
    voice: str | None,
    speed: float,
    mode: str,
    url: str,
) -> bytes:
    payload = {"text": text, "speed": speed, "mode": mode}
    if voice:
        payload["voice"] = voice

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    chunks: list[bytes] = []
    with urllib.request.urlopen(req) as resp:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if sys.stdout.buffer.isatty() is False:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()

    return b"".join(chunks)


def fetch_wav(
    text: str,
    voice: str | None,
    speed: float,
    mode: str,
    url: str,
) -> bytes:
    payload = {"text": text, "speed": speed, "mode": mode, "format": "wav"}
    if voice:
        payload["voice"] = voice

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        return resp.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="kokopop TTS HTTP client")
    parser.add_argument("text", nargs="?", default=None, help="Text to synthesize")
    parser.add_argument("--file", "-f", default=None, metavar="PATH",
                        help="Read text from a file (alternative to positional text)")
    parser.add_argument("--url", default="http://127.0.0.1:8080/tts",
                        help="Server URL (default: http://127.0.0.1:8080/tts)")
    parser.add_argument("--voice", default=None, help="Voice name (e.g. ff_siwis)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier")
    parser.add_argument("--mode", choices=["interactive", "long_form"],
                        default="interactive", help="Synthesis mode")
    parser.add_argument("--format", choices=["pcm", "wav"], default="pcm",
                        dest="fmt", help="Output format: pcm (float32) or wav")
    parser.add_argument("--out", default=None,
                        help="Output file (.wav). If omitted and format=pcm, "
                             "raw PCM is written to stdout.")
    args = parser.parse_args()

    # Resolve text: positional arg takes precedence, then --file, then error
    if args.text is not None:
        text = args.text
    elif args.file is not None:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        parser.error("the following arguments are required: text or --file")

    if args.fmt == "wav":
        data = fetch_wav(text, args.voice, args.speed, args.mode, args.url)
        if args.out:
            Path(args.out).write_bytes(data)
            print(f"Saved {len(data):,} bytes → {args.out}", file=sys.stderr)
        else:
            sys.stdout.buffer.write(data)
    else:
        # PCM mode: stream to stdout or save as WAV
        pcm = stream_pcm(text, args.voice, args.speed, args.mode, args.url)
        if args.out:
            wav = _pcm_to_wav(pcm)
            Path(args.out).write_bytes(wav)
            print(f"Saved {len(wav):,} bytes → {args.out}", file=sys.stderr)
        elif not sys.stdout.buffer.isatty():
            # Already streamed to stdout in stream_pcm
            pass
        else:
            print("No --out specified and stdout is a TTY. Use --out or pipe to ffplay.",
                  file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
