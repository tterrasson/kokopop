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
    _put_kv_str_array(
        data, "tokenizer.ggml.tokens", ["", "a", "b", "c", " ", "ɑ", "ɔ", "ʃ"]
    )
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


def test_model_text_and_phoneme_synthesis(mock_gguf):
    model = kokopop.Model(str(mock_gguf), n_threads=1, backend="cpu")
    assert model.sample_rate == 24000

    text_audio = model.synthesize("Alpha sentence.", voice="af_heart")
    assert isinstance(text_audio, kokopop.Audio)
    assert text_audio.sample_rate == 24000
    assert text_audio.n_samples > 1000

    audio = model.synthesize_phonemes("abc", voice="af_heart")
    assert isinstance(audio, kokopop.Audio)
    assert audio.sample_rate == 24000
    assert audio.n_samples > 1000

    view = memoryview(audio)
    assert view.format == "f"
    assert view.shape == (audio.n_samples,)
    assert abs(view[0]) < 1.0


def test_model_arch_and_voice_table(mock_gguf):
    model = kokopop.Model(str(mock_gguf), n_threads=1, backend="cpu")
    assert model.arch == "kokoro-82m"
    assert model.voices == ("af_heart",)
    assert model.voice_sample_rate("af_heart") == 24000
    with pytest.raises(KeyError):
        model.voice_sample_rate("no_such_voice")


def test_streaming_session_accepts_noise_seed(mock_gguf):
    # Kokoro ignores the seed, but it must be accepted and reach the C API
    # rather than being rejected as an unknown keyword.
    model = kokopop.Model(str(mock_gguf), n_threads=1, backend="cpu")
    with kokopop.SynthesisSession(
        model, voice="af_heart", mode="long_form", noise_seed=0
    ) as session:
        session.push_text("Alpha sentence.")
        session.finish_input()
        assert session.next(1)

    with pytest.raises(TypeError):
        kokopop.SynthesisSession(model, voice="af_heart", not_an_option=1)


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
    with kokopop.SynthesisSession(
        model,
        voice="af_heart",
        mode="long_form",
        target_min_tokens=1,
        target_max_tokens=12,
        soft_max_tokens=16,
        hard_max_tokens=32,
    ) as session:
        session.push_text("Alpha sentence. Beta sentence.")
        session.finish_input()
        chunks = session.next(max_chunks=4)
        with pytest.raises(ValueError):
            session.push_text("too late")

    assert chunks
    assert chunks[0].sample_rate == 24000
    assert chunks[0].n_samples > 0

    with pytest.raises(ValueError):
        session.next()


def test_streaming_session_accepts_disabled_diffusion_options(mock_gguf):
    model = kokopop.Model(str(mock_gguf), backend="cpu")
    with kokopop.SynthesisSession(
        model,
        voice="af_heart",
        mode="long_form",
        enable_diffusion=False,
        diffusion_seed=1234,
        diffusion_steps=7,
        diffusion_alpha=0.2,
        diffusion_beta=0.6,
        diffusion_embedding_scale=1.5,
    ) as session:
        session.push_text("Alpha sentence.")
        session.finish_input()
        chunks = session.next()

    assert chunks
    assert chunks[0].n_samples > 0


def test_model_stream_iterator(mock_gguf):
    model = kokopop.Model(str(mock_gguf), backend="cpu")
    stream = model.stream(
        "Alpha sentence. Beta sentence.",
        voice="af_heart",
        mode="long-form",
        max_chunks=0,
        target_min_tokens=1,
        target_max_tokens=12,
        soft_max_tokens=16,
        hard_max_tokens=32,
    )
    assert iter(stream) is stream
    assert hasattr(stream, "close")

    first = next(stream)
    assert isinstance(first, kokopop.AudioChunk)

    chunks = [first, *list(stream)]
    assert chunks
    assert any(chunk.is_final for chunk in chunks)


def test_model_stream_accepts_disabled_diffusion_options(mock_gguf):
    model = kokopop.Model(str(mock_gguf), backend="cpu")
    chunks = list(
        model.stream(
            "Alpha sentence.",
            voice="af_heart",
            enable_diffusion=False,
            diffusion_seed=1234,
            diffusion_steps=7,
            diffusion_alpha=0.2,
            diffusion_beta=0.6,
            diffusion_embedding_scale=1.5,
        )
    )
    assert chunks
    assert any(chunk.is_final for chunk in chunks)


def test_audio_encoder_pcm_and_wav():
    samples = array.array("f", [0.0, 0.25, -0.25, 1.0])

    with kokopop.AudioEncoder("pcm_f32le", sample_rate=24000) as pcm:
        out = pcm.push(samples, is_final=True)
    assert len(out) == len(samples) * 4
    with pytest.raises(ValueError):
        pcm.push(samples)

    with kokopop.AudioEncoder("wav", sample_rate=24000) as wav:
        assert wav.start() == b""
        assert wav.push(samples, is_final=True) == b""
        encoded = wav.finish()
    assert encoded[:4] == b"RIFF"


def test_audio_encoder_wav_roundtrip_via_audio_to_wav_bytes(mock_gguf):
    # The streaming encoder's WAV output should match the one-shot helper
    # for the same samples (both PCM16, 24kHz, mono).
    model = kokopop.Model(str(mock_gguf), backend="cpu")
    audio = model.synthesize_phonemes("abc", voice="af_heart")

    one_shot = audio.to_wav_bytes()
    assert one_shot[:4] == b"RIFF"
    assert one_shot[8:12] == b"WAVE"

    samples = array.array("f", memoryview(audio))
    encoder = kokopop.AudioEncoder("wav", sample_rate=audio.sample_rate)
    streamed = encoder.start() + encoder.push(samples, is_final=True) + encoder.finish()
    encoder.close()

    assert streamed[:4] == b"RIFF"
    assert streamed[8:12] == b"WAVE"
    # Both encodings should produce identical PCM16 audio of equal length.
    assert len(streamed) == len(one_shot)


def test_audio_encoder_ogg_opus_or_skip():
    samples = array.array("f", [0.1, -0.1] * 4800)  # 0.4s @ 24kHz
    try:
        encoder = kokopop.AudioEncoder("ogg_opus", sample_rate=24000)
    except kokopop.KokopopError:
        pytest.skip("build without libopusenc")
    stream = encoder.start() + encoder.push(samples, is_final=True) + encoder.finish()
    encoder.close()
    assert b"OggS" in stream


def test_audio_encoder_use_after_close():
    samples = array.array("f", [0.0, 0.25])
    enc = kokopop.AudioEncoder("pcm_f32le", sample_rate=24000)
    enc.close()
    with pytest.raises(ValueError):
        enc.push(samples)
    with pytest.raises(ValueError):
        enc.finish()
    # close() is idempotent.
    enc.close()


def test_stream_iterator_use_after_close(mock_gguf):
    model = kokopop.Model(str(mock_gguf), backend="cpu")
    stream = model.stream("Alpha sentence.", voice="af_heart")
    stream.close()
    with pytest.raises((ValueError, StopIteration)):
        next(stream)
    # close() is idempotent.
    stream.close()
