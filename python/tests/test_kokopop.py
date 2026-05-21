from __future__ import annotations

import array
import struct

import kokopop
import pytest


def _put_u32(out: bytearray, value: int) -> None:
    out.extend(struct.pack("<I", value))


def _put_u64(out: bytearray, value: int) -> None:
    out.extend(struct.pack("<Q", value))


def _put_f32(out: bytearray, value: float) -> None:
    out.extend(struct.pack("<f", value))


def _put_string(out: bytearray, value: str) -> None:
    raw = value.encode()
    _put_u64(out, len(raw))
    out.extend(raw)


def _put_kv_u32(out: bytearray, key: str, value: int) -> None:
    _put_string(out, key)
    _put_u32(out, 4)
    _put_u32(out, value)


def _put_kv_bool(out: bytearray, key: str, value: bool) -> None:
    _put_string(out, key)
    _put_u32(out, 7)
    out.append(1 if value else 0)


def _put_kv_str_array(out: bytearray, key: str, values: list[str]) -> None:
    _put_string(out, key)
    _put_u32(out, 9)
    _put_u32(out, 8)
    _put_u64(out, len(values))
    for value in values:
        _put_string(out, value)


def _align_to(out: bytearray, alignment: int) -> None:
    while len(out) % alignment:
        out.append(0)


@pytest.fixture()
def mock_gguf(tmp_path):
    path = tmp_path / "kokopop_mock.gguf"
    data = bytearray(b"GGUF")
    _put_u32(data, 3)
    _put_u64(data, 1)
    _put_u64(data, 6)

    _put_kv_u32(data, "general.alignment", 32)
    _put_kv_u32(data, "kokopop.kokoro.version", 4)
    _put_kv_bool(data, "kokopop.mock", True)
    _put_kv_u32(data, "kokopop.sample_rate", 24000)
    _put_kv_str_array(data, "tokenizer.ggml.tokens", ["", "a", "b", "c", " ", "ɑ", "ɔ", "ʃ"])
    _put_kv_str_array(data, "kokopop.voices", ["af_heart"])

    _put_string(data, "kokopop.voice.af_heart")
    _put_u32(data, 2)
    _put_u64(data, 4)
    _put_u64(data, 2)
    _put_u32(data, 0)
    _put_u64(data, 0)

    _align_to(data, 32)
    for i in range(8):
        _put_f32(data, i / 8.0)

    path.write_bytes(data)
    return path


def test_import_surface():
    assert kokopop.Backend.CPU == "cpu"
    assert kokopop.Mode.ADAPTATIVE == "adaptative"
    assert kokopop.AudioFormat.WAV_PCM16 == "wav"


def test_model_and_phoneme_synthesis(mock_gguf):
    model = kokopop.Model(str(mock_gguf), n_threads=1, backend="cpu")
    assert model.sample_rate == 24000

    audio = model.synthesize_phonemes("abc", voice="af_heart")
    assert audio.sample_rate == 24000
    assert audio.n_samples > 1000

    view = memoryview(audio)
    assert view.format == "f"
    assert view.shape == (audio.n_samples,)
    assert abs(view[0]) < 1.0


def test_wav_helpers(mock_gguf, tmp_path):
    model = kokopop.Model(str(mock_gguf), backend="cpu")
    audio = model.synthesize_phonemes("abc", voice="af_heart")

    wav = audio.to_wav_bytes()
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"

    out = tmp_path / "out.wav"
    audio.write_wav(str(out))
    assert out.read_bytes()[:4] == b"RIFF"


def test_errors(mock_gguf, tmp_path):
    with pytest.raises(kokopop.KokopopError):
        kokopop.Model(str(tmp_path / "missing.gguf"), backend="cpu")

    model = kokopop.Model(str(mock_gguf), backend="cpu")
    with pytest.raises(ValueError):
        model.synthesize_phonemes("", voice="af_heart")
    with pytest.raises(kokopop.KokopopError):
        model.synthesize_phonemes("abc", voice="missing")


def test_streaming_session(mock_gguf):
    model = kokopop.Model(str(mock_gguf), backend="cpu")
    session = kokopop.SynthesisSession(
        model,
        voice="af_heart",
        mode="long_form",
        target_min_tokens=1,
        target_max_tokens=12,
        soft_max_tokens=16,
        hard_max_tokens=32,
    )
    session.push_text("Alpha sentence. Beta sentence.")
    session.finish_input()
    chunks = session.next(max_chunks=4)
    assert chunks
    assert chunks[0].sample_rate == 24000
    assert chunks[0].n_samples > 0


def test_model_stream_iterator(mock_gguf):
    model = kokopop.Model(str(mock_gguf), backend="cpu")
    chunks = list(
        model.stream(
            "Alpha sentence. Beta sentence.",
            voice="af_heart",
            mode="long_form",
            target_min_tokens=1,
            target_max_tokens=12,
            soft_max_tokens=16,
            hard_max_tokens=32,
        )
    )
    assert chunks
    assert any(chunk.is_final for chunk in chunks)


def test_audio_encoder_pcm_and_wav():
    samples = array.array("f", [0.0, 0.25, -0.25, 1.0])

    pcm = kokopop.AudioEncoder("pcm_f32le", sample_rate=24000)
    out = pcm.push(samples, is_final=True)
    assert len(out) == len(samples) * 4

    wav = kokopop.AudioEncoder("wav", sample_rate=24000)
    assert wav.push(samples, is_final=True) == b""
    encoded = wav.finish()
    assert encoded[:4] == b"RIFF"
