# Third-party notices

kokopop is MIT licensed. This file records the third-party work it derives from
and the notices those licences require.

## Ampixa/sanoTTS — MIT

Upstream: <https://github.com/Ampixa/sanoTTS>
Pinned revision: `939d982b9faa54cbcf5d24cc878f5cd514b2646e`

The sanoTTS repository as a whole is GPL-3.0-or-later, because its
grapheme-to-phoneme layer embeds espeak-ng. Its `LICENSE.MIT` carves out the
inference runtime and bindings under MIT, listing them exhaustively. kokopop's
sanoTTS support derives only from files on that MIT list:

| Upstream file | Used for | kokopop file |
|---|---|---|
| `mcu/src/snt_nano.c` | SHA-256 seed derivation, ATen MT19937, the 24-bit uniform, `normal_fill_16` Box-Muller | [src/arch/sanotts/sano_noise.cpp](src/arch/sanotts/sano_noise.cpp) |
| `mcu/include/snt_nano.h` | TinyVocos runtime contract | [src/arch/sanotts/](src/arch/sanotts/) |
| `mcu/models/*/nano_q8_meta.h` | generated blob offsets and dimensions | [tools/convert_sanotts_to_gguf.py](tools/convert_sanotts_to_gguf.py) |
| `pypkg/sanotts/models.py` | numpy forward pass of `duration_conv`, `token_context`, `piperlite` | [src/arch/sanotts/](src/arch/sanotts/), [tools/convert_sanotts_to_gguf.py](tools/convert_sanotts_to_gguf.py) |
| `pypkg/sanotts/voicepack.py` | voice-pack manifest validation | [tools/convert_sanotts_to_gguf.py](tools/convert_sanotts_to_gguf.py) |

Nothing in kokopop derives from `tools/`, `web/`, `npmpkg/` or any espeak-ng
port in that repository — all of which stay GPL-3.0-or-later.

### Notice

```
MIT License

Copyright (c) 2026 Ampixa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Data and weights are *not* covered

`LICENSE.MIT` covers code. It does not place `mcu/test/fixtures/**`, the weight
blobs, the voice packs or the rendered PCM under MIT. kokopop therefore
redistributes none of them:

- the golden-fixture tests read them from a local checkout and skip when there
  is none — see [tests/sanotts_fixtures.h](tests/sanotts_fixtures.h) and the
  `KOKOPOP_SANOTTS_FIXTURES` environment variable;
- the converter downloads voice packs into `~/.cache/sanotts/`, from an
  immutable commit, and records where they came from and what they hashed to:
  `kokopop.sanotts.source.repo`, `.code_revision` (the upstream revision the
  converter implements) and `.artifact_revision` (the commit the weights and
  manifests were read at) globally, and per voice
  `kokopop.sanotts.voice.<i>.source.*` — the weight blobs' SHA-256, plus the
  `piper-phoneme-config.json` hash for piperlite voices and the
  `nano_q8_meta.h` hash for vocos ones, both being conversion inputs that
  decide the layout and the id table.

  No licence is recorded: none of the manifests declares one. `<repo>@<artifact
  revision>` is what makes the terms auditable, since it resolves to the exact
  files that were converted.

`web/trellis_frontend.js`, which documents the 62-symbol vocos tokenizer, is
outside the MIT list. kokopop's tokenizer is an independent implementation
validated against the fixtures' `r*_ids.bin`, not a port of that file.

## ggml — MIT

Upstream: <https://github.com/ggml-org/ggml>, pinned in
[CMakeLists.txt](CMakeLists.txt). Fetched at build time, MIT licensed.

## espeak-ng — GPL-3.0-or-later

kokopop links libespeak-ng dynamically as a system library and ships no part of
it. A distributor that bundles espeak-ng with a kokopop binary takes on GPL-3.0
obligations for that combined distribution.

## Other build-time dependencies

| Project | Licence | Role |
|---|---|---|
| [doctest](https://github.com/doctest/doctest) | MIT | unit tests only |
| [yyjson](https://github.com/ibireme/yyjson) | MIT | JSON in the streaming tool |
| libopus / libogg / libopusenc | BSD-3-Clause | optional Ogg/Opus output |
