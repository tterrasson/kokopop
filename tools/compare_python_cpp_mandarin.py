#!/usr/bin/env python3
"""
Compare Mandarin synthesis quality between:
1. Python Kokoro reference (`kokoro` package in `.venv`)
2. Local C++ runtime (`./build/kokopop_say`)

This is a debugging tool for separating:
- Mandarin frontend / G2P issues
- GGUF export / runtime inference issues
- Base model limitations

Examples:
    uv run python tools/compare_python_cpp_mandarin.py \
      --voice zf_xiaoni \
      "这座古老城市的街道两旁种满了高大的法国梧桐，给人一种宁静的感觉。"

    uv run python tools/compare_python_cpp_mandarin.py \
      --voice zf_xiaoni \
      --mode both \
      "团队合作是项目成功的关键，每一位成员的贡献都不可或缺。"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
from kokoro import KPipeline

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.validate_mandarin import levenshtein, normalise, transcribe


def similarity(original: str, transcription: str) -> float:
    a = normalise(original)
    b = normalise(transcription)
    dist = levenshtein(a, b)
    return 1.0 - (dist / max(len(a), len(b), 1))


def audio_stats(audio: np.ndarray) -> tuple[int, float, float, float]:
    return (
        int(audio.shape[0]),
        float(np.sqrt(np.mean(audio * audio))),
        float(np.max(np.abs(audio))),
        float(np.mean(audio)),
    )


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 24000) -> None:
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm16.tobytes())


def synth_python(text: str, voice: str) -> tuple[str, np.ndarray]:
    pipeline = KPipeline(lang_code="z")
    parts: list[np.ndarray] = []
    last_ps = ""
    for _gs, ps, audio in pipeline(text, voice=voice):
        last_ps = ps
        parts.append(np.asarray(audio, dtype=np.float32))
    if not parts:
        raise RuntimeError("Python Kokoro returned no audio")
    return last_ps, np.concatenate(parts)


def synth_python_with_durations(text: str, voice: str) -> tuple[str, np.ndarray, list[int]]:
    pipeline = KPipeline(lang_code="z")
    pack = pipeline.load_voice(voice).to(pipeline.model.device)
    parts: list[np.ndarray] = []
    all_durations: list[int] = []
    last_ps = ""
    for _gs, ps, _audio in pipeline(text, voice=voice):
        last_ps = ps
        output = KPipeline.infer(pipeline.model, ps, pack, 1.0)
        parts.append(np.asarray(output.audio, dtype=np.float32))
        if output.pred_dur is None:
            raise RuntimeError("Python Kokoro returned no pred_dur")
        all_durations.extend(int(x) for x in output.pred_dur.tolist())
    if not parts:
        raise RuntimeError("Python Kokoro returned no audio")
    return last_ps, np.concatenate(parts), all_durations


def probe_python_from_phonemes(phonemes: str, voice: str) -> dict[str, Any]:
    pipeline = KPipeline(lang_code="z")
    model = pipeline.model
    assert model is not None
    pack = pipeline.load_voice(voice).to(model.device)

    input_ids = [x for x in (model.vocab.get(ch) for ch in phonemes) if x is not None]
    input_tensor = torch.LongTensor([[0, *input_ids, 0]]).to(model.device)
    ref_s = pack[len(phonemes) - 1].to(model.device)
    if ref_s.ndim == 1:
        ref_s = ref_s.unsqueeze(0)

    input_lengths = torch.full((1,), input_tensor.shape[-1], device=input_tensor.device, dtype=torch.long)
    text_mask = torch.arange(input_lengths.max(), device=input_tensor.device).unsqueeze(0).expand(1, -1)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(model.device)

    with torch.no_grad():
        bert_dur = model.bert(input_tensor, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        prosody_style = ref_s[:, 128:]
        decoder_style = ref_s[:, :128]

        d = model.predictor.text_encoder(d_en, prosody_style, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze(0)

        indices = torch.repeat_interleave(torch.arange(input_tensor.shape[1], device=model.device), pred_dur)
        pred_aln_trg = torch.zeros((input_tensor.shape[1], indices.shape[0]), device=model.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=model.device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ pred_aln_trg
        f0_pred, n_pred = model.predictor.F0Ntrain(en, prosody_style)
        t_en = model.text_encoder(input_tensor, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

        f0_conv = model.decoder.F0_conv(f0_pred.unsqueeze(1))
        n_conv = model.decoder.N_conv(n_pred.unsqueeze(1))
        decoder_cur = torch.cat([asr, f0_conv, n_conv], axis=1)
        decoder_cur = model.decoder.encode(decoder_cur, decoder_style)
        asr_res = model.decoder.asr_res(asr)
        use_residual = True
        for block in model.decoder.decode:
            if use_residual:
                decoder_cur = torch.cat([decoder_cur, asr_res, f0_conv, n_conv], axis=1)
            decoder_cur = block(decoder_cur, decoder_style)
            if block.upsample_type != "none":
                use_residual = False
        audio = model.decoder.generator(decoder_cur, decoder_style, f0_pred).squeeze().cpu().numpy().astype(np.float32)

    def tstats(tensor: torch.Tensor) -> tuple[float, float]:
        arr = tensor.detach().float().cpu().numpy()
        return float(arr.mean()), float(np.sqrt(np.mean(arr * arr)))

    f0_mean, f0_rms = tstats(f0_pred)
    noise_mean, noise_rms = tstats(n_pred)
    asr_mean, asr_rms = tstats(asr)
    decoder_mean, decoder_rms = tstats(decoder_cur)

    return {
        "durations": [int(x) for x in pred_dur.detach().cpu().tolist()],
        "f0_mean": f0_mean,
        "f0_rms": f0_rms,
        "noise_mean": noise_mean,
        "noise_rms": noise_rms,
        "asr_mean": asr_mean,
        "asr_rms": asr_rms,
        "decoder_mean": decoder_mean,
        "decoder_rms": decoder_rms,
        "audio": audio,
    }


def synth_cpp_from_text(text: str, voice: str, model: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = Path(tmp.name)
    try:
        cmd = [
            "./build/kokopop_say",
            "--model", model,
            "--voice", voice,
            "--text", text,
            "--out", str(wav),
        ]
        subprocess.run(cmd, check=True)
        with wave.open(str(wav), "rb") as f:
            frames = f.readframes(f.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        return pcm
    finally:
        wav.unlink(missing_ok=True)


def synth_cpp_from_phonemes(phonemes: str, voice: str, model: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = Path(tmp.name)
    try:
        cmd = [
            "./build/kokopop_say",
            "--model", model,
            "--voice", voice,
            "--phonemes", phonemes,
            "--out", str(wav),
        ]
        subprocess.run(cmd, check=True)
        with wave.open(str(wav), "rb") as f:
            frames = f.readframes(f.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        return pcm
    finally:
        wav.unlink(missing_ok=True)


def transcribe_audio(audio: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = Path(tmp.name)
    try:
        write_wav(wav, audio)
        result = transcribe(wav)
        return result["text"].strip()
    finally:
        wav.unlink(missing_ok=True)


def run_cpp_probe(text: str | None, phonemes: str | None, voice: str, model: str) -> dict[str, Any]:
    cmd = [
        "./build/kokopop_probe",
        "--model", model,
        "--voice", voice,
    ]
    if text is not None:
        cmd.extend(["--text", text])
    else:
        cmd.extend(["--phonemes", phonemes or ""])
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data: dict[str, Any] = {}
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"frontend_durations"}:
            data[key] = [float(x) for x in value.split(",") if x]
        elif key in {"rounded_durations"}:
            data[key] = [int(x) for x in value.split(",") if x]
        elif key in {"token_count", "frontend_hidden_dim", "generation_frames", "rounded_frames", "audio_samples"}:
            data[key] = int(value)
        elif key in {
            "f0_mean", "f0_rms",
            "noise_mean", "noise_rms",
            "asr_mean", "asr_rms",
            "decoder_mean", "decoder_rms",
            "audio_mean", "audio_rms", "audio_peak",
        }:
            data[key] = float(value)
        else:
            data[key] = value
    return data


def print_duration_report(label: str, durations: list[int]) -> None:
    print(f"{label}_durations:")
    print(f"  count: {len(durations)}")
    print(f"  sum: {sum(durations)}")
    if durations:
        print(f"  first10: {durations[:10]}")


def compare_duration_lists(reference: list[int], candidate: list[int], label: str) -> None:
    same_len = len(reference) == len(candidate)
    compared = min(len(reference), len(candidate))
    mismatches = 0
    max_abs_diff = 0
    examples: list[tuple[int, int, int]] = []
    for idx in range(compared):
        diff = abs(reference[idx] - candidate[idx])
        if diff:
            mismatches += 1
            max_abs_diff = max(max_abs_diff, diff)
            if len(examples) < 8:
                examples.append((idx, reference[idx], candidate[idx]))
    print(f"{label}_vs_python_durations:")
    print(f"  same_len: {same_len}")
    print(f"  compared: {compared}")
    print(f"  mismatches: {mismatches}")
    print(f"  max_abs_diff: {max_abs_diff}")
    print(f"  frame_sum_delta: {sum(candidate) - sum(reference)}")
    if examples:
        print(f"  examples: {examples}")


def print_probe_stats(label: str, probe: dict[str, Any]) -> None:
    print(f"{label}_probe_stats:")
    for key in ("f0_mean", "f0_rms", "noise_mean", "noise_rms", "asr_mean", "asr_rms", "decoder_mean", "decoder_rms"):
        print(f"  {key}: {probe[key]:.6f}")


def compare_probe_stats(reference: dict[str, Any], candidate: dict[str, Any], label: str) -> None:
    print(f"{label}_vs_python_probe_stats:")
    for key in ("f0_mean", "f0_rms", "noise_mean", "noise_rms", "asr_mean", "asr_rms", "decoder_mean", "decoder_rms"):
        delta = candidate[key] - reference[key]
        print(f"  {key}_delta: {delta:.6f}")


def print_report(label: str, text: str, audio: np.ndarray, transcription: str) -> None:
    sim = similarity(text, transcription)
    n, rms, peak, mean = audio_stats(audio)
    print(f"{label}:")
    print(f"  similarity: {sim:.3f}")
    print(f"  transcription: {transcription}")
    print(f"  samples: {n}")
    print(f"  rms: {rms:.6f}")
    print(f"  peak: {peak:.6f}")
    print(f"  mean: {mean:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--voice", default="zf_xiaoni")
    parser.add_argument("--model", default="models/kokoro-md.gguf")
    parser.add_argument(
        "--mode",
        choices=("text", "phonemes", "both"),
        default="both",
        help="Compare C++ from text, C++ from Python phonemes, or both",
    )
    parser.add_argument(
        "--include-probes",
        action="store_true",
        help="Also compare internal predicted durations between Python and C++",
    )
    args = parser.parse_args()

    print(f"text: {args.text}")
    if args.include_probes:
        py_phonemes, py_audio, py_durations = synth_python_with_durations(args.text, args.voice)
        py_probe = probe_python_from_phonemes(py_phonemes, args.voice)
    else:
        py_phonemes, py_audio = synth_python(args.text, args.voice)
        py_durations = []
        py_probe = {}
    print(f"python phonemes: {py_phonemes}")
    py_tr = transcribe_audio(py_audio)
    print_report("python", args.text, py_audio, py_tr)
    if args.include_probes:
        print_duration_report("python", py_durations)
        print_probe_stats("python", py_probe)

    if args.mode in ("text", "both"):
        cpp_audio = synth_cpp_from_text(args.text, args.voice, args.model)
        cpp_tr = transcribe_audio(cpp_audio)
        print_report("cpp_text", args.text, cpp_audio, cpp_tr)
        if args.include_probes:
            cpp_text_probe = run_cpp_probe(args.text, None, args.voice, args.model)
            print_duration_report("cpp_text", cpp_text_probe["rounded_durations"])
            compare_duration_lists(py_durations, cpp_text_probe["rounded_durations"], "cpp_text")
            print_probe_stats("cpp_text", cpp_text_probe)
            compare_probe_stats(py_probe, cpp_text_probe, "cpp_text")

    if args.mode in ("phonemes", "both"):
        cpp_py_audio = synth_cpp_from_phonemes(py_phonemes, args.voice, args.model)
        cpp_py_tr = transcribe_audio(cpp_py_audio)
        print_report("cpp_python_phonemes", args.text, cpp_py_audio, cpp_py_tr)
        if args.include_probes:
            cpp_ps_probe = run_cpp_probe(None, py_phonemes, args.voice, args.model)
            print_duration_report("cpp_python_phonemes", cpp_ps_probe["rounded_durations"])
            compare_duration_lists(py_durations, cpp_ps_probe["rounded_durations"], "cpp_python_phonemes")
            print_probe_stats("cpp_python_phonemes", cpp_ps_probe)
            compare_probe_stats(py_probe, cpp_ps_probe, "cpp_python_phonemes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
