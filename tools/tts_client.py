#!/usr/bin/env python3
"""
Minimal Python client for the kokopop HTTP TTS server.

Usage:
    uv run python tools/tts_client.py "Hello world" | ffplay -f f32le -ar 24000 -ac 1 -
    uv run python tools/tts_client.py "Hello world" --format ogg | ffplay -i pipe:0
    uv run python tools/tts_client.py "Hello world" --format wav --out hello.wav
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request
from pathlib import Path
from typing import BinaryIO, Callable


STREAM_READ_SIZE = 16 * 1024


def _read_stream(resp: BinaryIO, emit_stdout: bool) -> bytes:
    chunks: list[bytes] = []
    read = getattr(resp, "read1", resp.read)
    while True:
        chunk = read(STREAM_READ_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
        if emit_stdout:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    return b"".join(chunks)


def _pcm_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
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


def _build_payload(
    text: str,
    voice: str | None,
    speed: float,
    mode: str,
    fmt: str,
    prebuffer_chunks: int,
    first_chunk_tokens: int | None,
    diffusion: bool,
) -> dict:
    payload = {"text": text, "speed": speed, "mode": mode, "format": fmt}
    if voice:
        payload["voice"] = voice
    if fmt == "ogg" and prebuffer_chunks > 0:
        payload["prebuffer_chunks"] = prebuffer_chunks
    if first_chunk_tokens is not None:
        payload["first_chunk_target_tokens"] = first_chunk_tokens
    if diffusion:
        payload["diffusion"] = True

    return payload


def _send_tts(url: str, payload: dict, read_fn: Callable[[BinaryIO], bytes]) -> bytes:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return read_fn(resp)


def fetch_tts(
    text: str,
    voice: str | None,
    speed: float,
    mode: str,
    fmt: str,
    url: str,
    emit_stdout: bool,
    prebuffer_chunks: int,
    first_chunk_tokens: int | None = None,
    diffusion: bool = False,
) -> bytes:
    payload = _build_payload(text, voice, speed, mode, fmt, prebuffer_chunks,
                             first_chunk_tokens, diffusion)
    return _send_tts(url, payload, lambda resp: _read_stream(resp, emit_stdout))


def main() -> None:
    parser = argparse.ArgumentParser(description="kokopop TTS HTTP client")
    parser.add_argument("text", nargs="?", default=None, help="Text to synthesize")
    parser.add_argument("--file", "-f", default=None, metavar="PATH",
                        help="Read text from a file (alternative to positional text)")
    parser.add_argument("--url", default="http://127.0.0.1:8080/tts",
                        help="Server URL (default: http://127.0.0.1:8080/tts)")
    parser.add_argument("--voice", default=None, help="Voice name (e.g. ff_siwis)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier")
    parser.add_argument("--mode", choices=["adaptative", "long_form"],
                        default="adaptative", help="Synthesis mode")
    parser.add_argument("--format", choices=["pcm", "wav", "ogg"], default="pcm",
                        dest="fmt", help="Output format: pcm, wav, or ogg")
    parser.add_argument("--prebuffer-chunks", type=int, default=0,
                        help="Server-side Ogg synthesis chunks to buffer before playback")
    parser.add_argument("--first-chunk-tokens", type=int, default=None,
                        help="Target max tokens for the first audio chunk (adaptive mode only)")
    parser.add_argument("--diffusion", action="store_true",
                        help="Enable diffusion style sampling (requires a GGUF with diffusion tensors)")
    parser.add_argument("--out", default=None,
                        help="Output file. If omitted, streaming formats write to stdout.")
    args = parser.parse_args()

    if args.text is not None:
        text = args.text
    elif args.file is not None:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        parser.error("the following arguments are required: text or --file")

    emit_stdout = args.out is None and not sys.stdout.buffer.isatty()
    data = fetch_tts(
        text, args.voice, args.speed, args.mode, args.fmt, args.url,
        emit_stdout, max(0, args.prebuffer_chunks), args.first_chunk_tokens,
        args.diffusion)

    if args.out:
        output = _pcm_to_wav(data) if args.fmt == "pcm" else data
        Path(args.out).write_bytes(output)
        print(f"Saved {len(output):,} bytes -> {args.out}", file=sys.stderr)
    elif not emit_stdout:
        print("No --out specified and stdout is a TTY. Use --out or pipe to ffplay.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
