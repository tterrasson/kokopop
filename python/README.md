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
