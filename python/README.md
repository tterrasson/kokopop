# kokopop Python

Native Python bindings for the Kokopop Kokoro GGUF runtime.

```bash
uv pip install ./python
```

```python
import kokopop

model = kokopop.Model("../models/kokoro-md.gguf", backend="cpu")
audio = model.synthesize("Hello!", voice="af_heart")
audio.write_wav("hello.wav")
```

## One-shot synthesis

```python
import kokopop

model = kokopop.Model("../models/kokoro-md.gguf", backend=kokopop.Backend.CPU)

audio = model.synthesize("Hello, world!", voice="af_heart", speed=1.0)
audio.write_wav("output.wav")

phoneme_audio = model.synthesize_phonemes("abc", voice="af_heart")
wav_bytes = phoneme_audio.to_wav_bytes()
```

`Audio` objects expose float32 PCM samples through Python's buffer protocol, so
`memoryview(audio)` can be passed to code that accepts contiguous float32 audio.

## Chunked pull synthesis

Use `SynthesisSession` when you want progressive audio chunks and explicit
back-pressure from your own loop.

```python
import kokopop

model = kokopop.Model("../models/kokoro-md.gguf", backend="cpu")

with kokopop.SynthesisSession(
    model,
    voice="af_heart",
    speed=1.0,
    mode=kokopop.Mode.ADAPTATIVE,
    first_chunk_target_tokens=64,
) as synth:
    synth.push_text("First fragment. ")
    synth.push_text("Second fragment.")
    synth.finish_input()

    while True:
        chunks = synth.next(max_chunks=2)
        if not chunks:
            break

        for chunk in chunks:
            pcm = memoryview(chunk)
            # Write, play, or encode float32 PCM at chunk.sample_rate.

        if chunks[-1].is_final:
            break
```

For the common case where all input text is known up front, `Model.stream()`
returns a lazy iterator over `AudioChunk` objects:

```python
for chunk in model.stream(
    "First fragment. Second fragment.",
    voice="af_heart",
    mode="long_form",
    max_chunks=2,
):
    pcm = memoryview(chunk)
```

## Streaming audio encoding

`AudioEncoder` converts float32 PCM samples to streamable bytes.

```python
import kokopop

encoder = kokopop.AudioEncoder(kokopop.AudioFormat.WAV_PCM16, sample_rate=24000)
with encoder:
    header = encoder.start()
    body = encoder.push(memoryview(audio), is_final=True)
    final_wav = header + body + encoder.finish()
```

Formats:

| Format | Behavior |
|---|---|
| `kokopop.AudioFormat.PCM_F32LE` | Emits raw little-endian float32 PCM on each push. |
| `kokopop.AudioFormat.WAV_PCM16` | Accumulates samples and emits a complete WAV file at finish. |
| `kokopop.AudioFormat.OGG_OPUS` | Emits Ogg/Opus pages progressively when Ogg/Opus support is enabled. |
