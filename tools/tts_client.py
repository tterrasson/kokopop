#!/usr/bin/env python3
"""
Minimal Python client for the kokopop HTTP TTS server.

Usage:
    # Stream PCM float32 and play via ffplay
    uv run python tools/tts_client.py "Hello world" | ffplay -f f32le -ar 24000 -ac 1 -

    # Stream Ogg/Opus and play via ffplay (prebuffers by default)
    uv run python tools/tts_client.py "Hello world" --format ogg | ffplay -i pipe:0

    # Read text from a file
    uv run python tools/tts_client.py --file story.txt | ffplay -f f32le -ar 24000 -ac 1 -

    # Stream PCM and save as WAV
    uv run python tools/tts_client.py "Hello world" --out hello.wav

    # Receive a complete WAV file directly
    uv run python tools/tts_client.py "Hello world" --format wav --out hello.wav

    # Save Ogg/Opus to file
    uv run python tools/tts_client.py "Hello world" --format ogg --out hello.ogg
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import urllib.request
import json
from pathlib import Path
from typing import BinaryIO


STREAM_READ_SIZE = 1024
OGG_OPUS_GRANULE_RATE = 48000
DEFAULT_OGG_PREBUFFER_MS = 5000
DEFAULT_OGG_BURST_GAP_MS = 500


def _read_stream(resp: BinaryIO, emit_stdout: bool) -> bytes:
    """Read a streaming HTTP response, optionally forwarding bytes to stdout."""
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


def _read_ogg_stream_by_duration(resp: BinaryIO, emit_stdout: bool,
                                 prebuffer_ms: int) -> bytes:
    """Read Ogg/Opus while optionally buffering enough audio before playback."""
    if not emit_stdout or prebuffer_ms <= 0:
        return _read_stream(resp, emit_stdout)

    chunks: list[bytes] = []
    pending = bytearray()
    parse_pos = 0
    emitted = False
    target_granule = prebuffer_ms * OGG_OPUS_GRANULE_RATE // 1000
    read = getattr(resp, "read1", resp.read)

    while True:
        chunk = read(STREAM_READ_SIZE)
        if not chunk:
            break
        chunks.append(chunk)

        if emitted:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            continue

        pending.extend(chunk)
        while parse_pos + 27 <= len(pending):
            if pending[parse_pos:parse_pos + 4] != b"OggS":
                sync = pending.find(b"OggS", parse_pos + 1)
                if sync < 0:
                    parse_pos = max(0, len(pending) - 3)
                    break
                parse_pos = sync
                continue

            segment_count = pending[parse_pos + 26]
            lacing_start = parse_pos + 27
            lacing_end = lacing_start + segment_count
            if lacing_end > len(pending):
                break

            body_size = sum(pending[lacing_start:lacing_end])
            page_end = lacing_end + body_size
            if page_end > len(pending):
                break

            granule = struct.unpack_from("<q", pending, parse_pos + 6)[0]
            parse_pos = page_end
            if granule >= target_granule:
                sys.stdout.buffer.write(pending)
                sys.stdout.buffer.flush()
                pending.clear()
                parse_pos = 0
                emitted = True
                break

    if pending:
        sys.stdout.buffer.write(pending)
        sys.stdout.buffer.flush()

    return b"".join(chunks)


def _read_ogg_stream_by_second_burst(resp: BinaryIO, emit_stdout: bool,
                                     burst_gap_ms: int) -> bytes:
    """Buffer until audio resumes after the first TTS chunk burst."""
    if not emit_stdout:
        return _read_stream(resp, emit_stdout)

    chunks: list[bytes] = []
    pending = bytearray()
    parse_pos = 0
    emitted = False
    saw_audio = False
    last_read_at: float | None = None
    gap_seconds = max(0, burst_gap_ms) / 1000.0
    read = getattr(resp, "read1", resp.read)

    while True:
        chunk = read(STREAM_READ_SIZE)
        now = time.monotonic()
        if not chunk:
            break
        chunks.append(chunk)

        if emitted:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            last_read_at = now
            continue

        starts_new_burst = (
            saw_audio
            and last_read_at is not None
            and now - last_read_at >= gap_seconds
        )
        last_read_at = now

        pending.extend(chunk)
        chunk_has_audio = False
        while parse_pos + 27 <= len(pending):
            if pending[parse_pos:parse_pos + 4] != b"OggS":
                sync = pending.find(b"OggS", parse_pos + 1)
                if sync < 0:
                    parse_pos = max(0, len(pending) - 3)
                    break
                parse_pos = sync
                continue

            segment_count = pending[parse_pos + 26]
            lacing_start = parse_pos + 27
            lacing_end = lacing_start + segment_count
            if lacing_end > len(pending):
                break

            body_size = sum(pending[lacing_start:lacing_end])
            page_end = lacing_end + body_size
            if page_end > len(pending):
                break

            granule = struct.unpack_from("<q", pending, parse_pos + 6)[0]
            parse_pos = page_end
            if granule > 0:
                saw_audio = True
                chunk_has_audio = True

        if starts_new_burst and chunk_has_audio:
            sys.stdout.buffer.write(pending)
            sys.stdout.buffer.flush()
            pending.clear()
            parse_pos = 0
            emitted = True

    if pending:
        sys.stdout.buffer.write(pending)
        sys.stdout.buffer.flush()

    return b"".join(chunks)


def _read_ogg_stream(resp: BinaryIO, emit_stdout: bool, prebuffer_mode: str,
                     prebuffer_ms: int, burst_gap_ms: int) -> bytes:
    if prebuffer_mode == "second-chunk":
        return _read_ogg_stream_by_second_burst(resp, emit_stdout, burst_gap_ms)
    return _read_ogg_stream_by_duration(resp, emit_stdout, prebuffer_ms)


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
    emit_stdout: bool,
) -> bytes:
    payload = {"text": text, "speed": speed, "mode": mode, "format": "pcm"}
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
        return _read_stream(resp, emit_stdout)


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


def stream_ogg(
    text: str,
    voice: str | None,
    speed: float,
    mode: str,
    url: str,
    emit_stdout: bool,
    prebuffer_mode: str,
    prebuffer_ms: int,
    burst_gap_ms: int,
) -> bytes:
    payload = {"text": text, "speed": speed, "mode": mode, "format": "ogg"}
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
        return _read_ogg_stream(resp, emit_stdout, prebuffer_mode,
                                prebuffer_ms, burst_gap_ms)


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
    parser.add_argument("--format", choices=["pcm", "wav", "ogg"], default="pcm",
                        dest="fmt", help="Output format: pcm (float32), wav, or ogg (Opus)")
    parser.add_argument("--prebuffer-mode", choices=["duration", "second-chunk"],
                        default="duration",
                        help="Ogg/Opus startup buffering strategy (default: duration)")
    parser.add_argument("--prebuffer-ms", type=int, default=DEFAULT_OGG_PREBUFFER_MS,
                        help="Ogg/Opus audio to buffer before writing to stdout "
                             f"(default: {DEFAULT_OGG_PREBUFFER_MS}; use 0 for immediate)")
    parser.add_argument("--prebuffer-gap-ms", type=int, default=DEFAULT_OGG_BURST_GAP_MS,
                        help="Idle gap used to detect the next TTS chunk in "
                             "second-chunk mode "
                             f"(default: {DEFAULT_OGG_BURST_GAP_MS})")
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
    elif args.fmt == "ogg":
        emit_stdout = args.out is None and not sys.stdout.buffer.isatty()
        ogg = stream_ogg(text, args.voice, args.speed, args.mode, args.url,
                         emit_stdout, args.prebuffer_mode,
                         args.prebuffer_ms, args.prebuffer_gap_ms)
        if args.out:
            Path(args.out).write_bytes(ogg)
            print(f"Saved {len(ogg):,} bytes → {args.out}", file=sys.stderr)
        elif not sys.stdout.buffer.isatty():
            # Already streamed to stdout in stream_ogg
            pass
        else:
            print("No --out specified and stdout is a TTY. Use --out or pipe to ffplay.",
                  file=sys.stderr)
            sys.exit(1)
    else:
        # PCM mode: stream to stdout or save as WAV
        emit_stdout = args.out is None and not sys.stdout.buffer.isatty()
        pcm = stream_pcm(text, args.voice, args.speed, args.mode, args.url, emit_stdout)
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
