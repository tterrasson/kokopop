#!/usr/bin/env python3
"""
Minimal Python client for the kokopop HTTP TTS server.

Usage:
    # Stream PCM float32 and play via ffplay
    uv run python tools/tts_client.py "Hello world" | ffplay -f f32le -ar 24000 -ac 1 -

    # Stream Ogg/Opus and play via ffplay (prebuffers 2 synthesis chunks before playback)
    uv run python tools/tts_client.py "Hello world" --format ogg --prebuffer-chunks 2 | ffplay -i pipe:0

    # Read text from a file
    uv run python tools/tts_client.py --file story.txt | ffplay -f f32le -ar 24000 -ac 1 -

    # Stream PCM and save as WAV
    uv run python tools/tts_client.py "Hello world" --out hello.wav

    # Receive a complete WAV file directly
    uv run python tools/tts_client.py "Hello world" --format wav --out hello.wav

    # Save Ogg/Opus to file
    uv run python tools/tts_client.py "Hello world" --format ogg --out hello.ogg

    # Override stable chunking params (patches the mode preset)
    uv run python tools/tts_client.py "Hello world" \\
        --chunk-target-max 80 --chunk-first-max 40 \\
        --chunk-crossfade 15 --chunk-sentence-pause 100 \\
        --out hello.wav
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
DEFAULT_OGG_PREBUFFER_MS = 3000
DEFAULT_OGG_BURST_GAP_MS = 250


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


def _read_ogg_stream_by_chunk_count(resp: BinaryIO, emit_stdout: bool,
                                    prebuffer_chunks: int,
                                    burst_gap_ms: int) -> bytes:
    """Buffer until N TTS synthesis chunks have been received, then emit."""
    if not emit_stdout or prebuffer_chunks <= 0:
        return _read_stream(resp, emit_stdout)

    chunks: list[bytes] = []
    pending = bytearray()
    parse_pos = 0
    emitted = False
    bursts_seen = 0
    in_burst = False
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

        # Detect gap between bursts (= boundary between two TTS chunks)
        gap_detected = (
            in_burst
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
                chunk_has_audio = True

        if chunk_has_audio:
            if gap_detected or not in_burst:
                bursts_seen += 1
            in_burst = True
        elif gap_detected:
            in_burst = False

        if bursts_seen >= prebuffer_chunks:
            sys.stdout.buffer.write(pending)
            sys.stdout.buffer.flush()
            pending.clear()
            parse_pos = 0
            emitted = True

    if pending:
        sys.stdout.buffer.write(pending)
        sys.stdout.buffer.flush()

    return b"".join(chunks)


def _read_ogg_stream(resp: BinaryIO, emit_stdout: bool,
                     prebuffer_ms: int, burst_gap_ms: int,
                     prebuffer_chunks: int) -> bytes:
    if prebuffer_chunks > 0:
        return _read_ogg_stream_by_chunk_count(resp, emit_stdout,
                                               prebuffer_chunks, burst_gap_ms)
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


def _build_payload(
    text: str,
    voice: str | None,
    speed: float,
    mode: str,
    fmt: str,
    chunking: dict | None,
) -> dict:
    payload = {"text": text, "speed": speed, "mode": mode, "format": fmt}
    if voice:
        payload["voice"] = voice
    if chunking:
        payload["chunking"] = chunking
    return payload


def _send_tts(url: str, payload: dict, read_fn) -> bytes:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return read_fn(resp)


def stream_pcm(
    text: str,
    voice: str | None,
    speed: float,
    mode: str,
    url: str,
    emit_stdout: bool,
    chunking: dict | None,
) -> bytes:
    payload = _build_payload(text, voice, speed, mode, "pcm", chunking)
    buf: list[bytes] = []
    def read_fn(resp):
        out = _read_stream(resp, emit_stdout)
        buf.append(out)
        return out
    _send_tts(url, payload, read_fn)
    return buf[0] if buf else b""


def fetch_wav(
    text: str,
    voice: str | None,
    speed: float,
    mode: str,
    url: str,
    chunking: dict | None,
) -> bytes:
    payload = _build_payload(text, voice, speed, mode, "wav", chunking)
    def read_fn(resp):
        return resp.read()
    return _send_tts(url, payload, read_fn)


def stream_ogg(
    text: str,
    voice: str | None,
    speed: float,
    mode: str,
    url: str,
    emit_stdout: bool,
    prebuffer_ms: int,
    burst_gap_ms: int,
    prebuffer_chunks: int,
    chunking: dict | None,
) -> bytes:
    payload = _build_payload(text, voice, speed, mode, "ogg", chunking)
    if prebuffer_chunks > 0:
        payload["prebuffer_chunks"] = prebuffer_chunks
    def read_fn(resp):
        if prebuffer_chunks > 0:
            return _read_stream(resp, emit_stdout)
        return _read_ogg_stream(resp, emit_stdout,
                                prebuffer_ms, burst_gap_ms, prebuffer_chunks)
    return _send_tts(url, payload, read_fn)


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
    parser.add_argument("--prebuffer-ms", type=int, default=DEFAULT_OGG_PREBUFFER_MS,
                        help="Ogg/Opus audio to buffer before writing to stdout "
                             f"(default: {DEFAULT_OGG_PREBUFFER_MS}; use 0 for immediate)")
    parser.add_argument("--prebuffer-chunks", type=int, default=0,
                        help="Buffer this many TTS synthesis chunks before playback "
                             "(overrides --prebuffer-ms when > 0; e.g. --prebuffer-chunks 2)")
    parser.add_argument("--prebuffer-gap-ms", type=int, default=DEFAULT_OGG_BURST_GAP_MS,
                        help="Idle gap in ms used to detect boundaries between TTS chunks "
                             f"(default: {DEFAULT_OGG_BURST_GAP_MS})")

    # -- Chunking overrides (sent as "chunking" JSON object) ----
    chunk_grp = parser.add_argument_group("chunking overrides")
    chunk_grp.add_argument("--chunk-target-min", type=int, default=None,
                           help="Minimum tokens per chunk (default: preset)")
    chunk_grp.add_argument("--chunk-target-max", type=int, default=None,
                           help="Target max tokens per chunk (default: preset)")
    chunk_grp.add_argument("--chunk-soft-max", type=int, default=None,
                           help="Soft max tokens before forced split (default: preset)")
    chunk_grp.add_argument("--chunk-hard-max", type=int, default=None,
                           help="Hard max tokens per chunk (default: preset)")
    chunk_grp.add_argument("--chunk-first-max", type=int, default=None,
                           help="First-chunk target max for low TTFB (default: preset)")
    chunk_grp.add_argument("--chunk-sentence-pause", type=int, default=None,
                           help="Sentence pause in ms (default: preset)")
    chunk_grp.add_argument("--chunk-paragraph-pause", type=int, default=None,
                           help="Paragraph pause in ms (default: preset)")
    chunk_grp.add_argument("--chunk-crossfade", type=int, default=None,
                           help="Crossfade between chunks in ms (default: preset)")
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

    # Build chunking override dict (only non-None fields)
    _chunking = {}
    field_map = [
        ("chunk_target_min", "target_min_tokens"),
        ("chunk_target_max", "target_max_tokens"),
        ("chunk_soft_max", "soft_max_tokens"),
        ("chunk_hard_max", "hard_max_tokens"),
        ("chunk_first_max", "first_chunk_target_max_tokens"),
        ("chunk_sentence_pause", "sentence_pause_ms"),
        ("chunk_paragraph_pause", "paragraph_pause_ms"),
        ("chunk_crossfade", "crossfade_ms"),
    ]
    for cli_name, json_name in field_map:
        val = getattr(args, cli_name)
        if val is not None:
            _chunking[json_name] = val
    chunking = _chunking if _chunking else None

    if args.fmt == "wav":
        data = fetch_wav(text, args.voice, args.speed, args.mode, args.url, chunking)
        if args.out:
            Path(args.out).write_bytes(data)
            print(f"Saved {len(data):,} bytes → {args.out}", file=sys.stderr)
        else:
            sys.stdout.buffer.write(data)
    elif args.fmt == "ogg":
        emit_stdout = args.out is None and not sys.stdout.buffer.isatty()
        ogg = stream_ogg(text, args.voice, args.speed, args.mode, args.url,
                         emit_stdout, args.prebuffer_ms, args.prebuffer_gap_ms,
                         args.prebuffer_chunks, chunking)
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
        pcm = stream_pcm(text, args.voice, args.speed, args.mode, args.url,
                         emit_stdout, chunking)
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
