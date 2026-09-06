# kokopop for the browser

Build from the repository root using the Emscripten SDK:

```sh
source /path/to/emsdk/emsdk_env.sh
emcmake cmake -S . -B build-web -DCMAKE_BUILD_TYPE=Release \
  -DKOKOPOP_ENABLE_WEBGPU=ON '-DKOKOPOP_WEB_LANGUAGES=en;fr'
cmake --build build-web --target kokopop_web --parallel 4
```

CMake builds eSpeak NG from a pinned revision, including its language data with
native host tools. Prerequisites: CMake, Git, a host C/C++ compiler, Python (use
`uv` to provision it) and Emscripten. No system eSpeak library is used for wasm.
For CPU-only browsers build a separate package with `KOKOPOP_ENABLE_WEBGPU=OFF`.

The complete package is `build-web/web/dist/` (`@kokopop/web`). A single runtime
supports both Kokoro and sanoTTS, using the same core library as native builds.
The directory contains `index.js`, TypeScript declarations, Worker modules,
`kokopop-runtime.js`, `kokopop-runtime.wasm`, `kokopop-runtime.data`, and licenses.
Keep these files together. Install with `npm install /path/to/dist`, or run
`npm pack` in that directory. Nothing is published automatically.

`KOKOPOP_WEB_LANGUAGES` defaults to `all`. Set a quoted semicolon-separated list
of eSpeak dictionary codes, such as `en` or `en;fr`. Use `en` for both US and UK
English, not `en-us`; codes correspond to the generated `*_dict` filenames.
Unknown codes fail the build. The filtered data retains common phoneme tables,
voice variants, and language definitions for included dictionaries. English is
always retained for initialization/fallback, and dictionary language-switch
references add their dependencies transitively. Select every language your model voices need; omitted languages
are not downloaded on demand. Reconfigure and rebuild to change the selection.

Language filtering affects only Web builds. Native builds retain both
architectures and their existing eSpeak setup.

## Direct browser usage

Copy the package directory to your site's `/kokopop/` and serve it over HTTP
(localhost) or HTTPS. Models are downloaded separately; none is bundled.

```js
import { Kokopop } from '/kokopop/index.js';

const tts = await Kokopop.create({
  model: '/models/sanotts-heartnano.gguf',
  backend: 'cpu', // 'webgpu' explicitly requests GPU; 'auto' uses native policy
});
console.log(tts.voices, tts.backend);

const { samples, sampleRate } = await tts.synthesize('Hello from the browser!', {
  voice: 'heartnano',
});

// In a click handler, to satisfy browser autoplay policy:
const context = new AudioContext();
await context.resume();

const buffer = context.createBuffer(1, samples.length, sampleRate);
buffer.copyToChannel(samples, 0);

const source = context.createBufferSource();
source.buffer = buffer;
source.connect(context.destination);
source.start();

await tts.dispose();
```

## Vite / other bundlers

```ts
import { Kokopop } from '@kokopop/web';

const tts = await Kokopop.create({
  model: '/models/kokoro.gguf',
  backend: 'webgpu',
  workerUrl: '/kokopop/worker.js',
  assetsUrl: '/kokopop/',
});
```

Copy all package files into `public/kokopop/`. Explicit `workerUrl` avoids
bundler-specific handling of Emscripten-generated code; `assetsUrl` locates
`.wasm` and `.data` after deployment. Serve `.wasm` as `application/wasm` and
`.js` as JavaScript. Cross-origin model/assets hosts must allow CORS.

Inference runs in a dedicated Worker, one CPU thread, with SIMD and growable
WASM memory (2 GiB limit). SharedArrayBuffer/COOP/COEP are not required. Model
input buffers are copied, so the caller's buffers remain usable. Synthesis
returns mono PCM at the selected voice's sample rate, including mixed-rate
sanoTTS packs. Concurrent calls are serialized. `dispose()` waits for queued
work; terminating the page cancels the Worker. `signal` cancels initial loading.

The WebGPU build requires WebAssembly JSPI as well as WebGPU in a secure
context. Select the CPU-only package for browsers without JSPI: checking
`navigator.gpu` alone is insufficient. Unsupported GGML operations fall back
to CPU. Explicit `webgpu` errors if unavailable; `auto` retains the native
policy (sanoTTS prefers CPU). GPU coverage and performance depend on the model,
GGML, and browser/device. This API returns complete utterances, not streaming
chunks. eSpeak NG is GPL-3.0-or-later; its license is included in the package.
Review dependency and model licenses when distributing your application.

## Smoke tests

Run the actual WASM on the CPU using an SDK Node with JSPI support for WebGPU
builds (add `--experimental-wasm-jspi` if required by your Node version):

```sh
node web/test-espeak-data.mjs build-web
node web/smoke.mjs build-web/web/dist models/sanotts-en.gguf sanotts
node web/smoke.mjs build-web/web/dist models/kokoro-fr.gguf kokoro-82m
```

The test checks audio for every voice, invalid input, and disposal.
`web/test.html` additionally exercises browser Workers and WebGPU with sanoTTS
models.
