// Run from the repository root: node web/smoke.mjs build-web-cpu/web/dist models/sanotts-en.gguf
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';

const [dist = 'build-web-cpu/web/dist', model = 'models/sanotts-en.gguf', architecture = 'sanotts'] = process.argv.slice(2);
const url = (name) => pathToFileURL(resolve(dist, name)).href;

function encodeWav(samples, sampleRate) {
  const out = Buffer.alloc(44 + samples.length * 2);
  out.write('RIFF');
  out.writeUInt32LE(out.length - 8, 4);
  out.write('WAVEfmt ', 8);
  out.writeUInt32LE(16, 16);
  out.writeUInt16LE(1, 20);
  out.writeUInt16LE(1, 22);
  out.writeUInt32LE(sampleRate, 24);
  out.writeUInt32LE(sampleRate * 2, 28);
  out.writeUInt16LE(2, 32);
  out.writeUInt16LE(16, 34);
  out.write('data', 36);
  out.writeUInt32LE(samples.length * 2, 40);
  samples.forEach((x, i) => out.writeInt16LE(Math.round(Math.max(-1, Math.min(1, x)) * 32767), 44 + i * 2));
  return out;
}

async function testVoice(runtime, voice) {
  const start = performance.now();
  const audio = await runtime.synthesize(voice.name.startsWith('ff_') ? 'Bonjour, comment allez-vous ?' : 'Hello from the browser.', { voice: voice.name });

  assert.equal(audio.sampleRate, voice.sampleRate);
  assert.ok(audio.samples.length > 1000);
  assert.ok(audio.samples.every(Number.isFinite));

  const rms = Math.sqrt(audio.samples.reduce((sum, x) => sum + x * x, 0) / audio.samples.length);
  assert.ok(rms > 1e-5, 'audio must not be silent');
  console.log(JSON.stringify({
    voice: voice.name,
    samples: audio.samples.length,
    sampleRate: audio.sampleRate,
    rms,
    elapsedMs: Math.round(performance.now() - start),
  }));

  if (process.env.KOKOPOP_SMOKE_AUDIO_DIR) {
    const wav = encodeWav(audio.samples, audio.sampleRate);
    writeFileSync(resolve(process.env.KOKOPOP_SMOKE_AUDIO_DIR, `wasm-${voice.name}.wav`), wav);
  }
}

async function main() {
  const packageInfo = JSON.parse(readFileSync(resolve(dist, 'package.json'), 'utf8'));
  assert.equal(packageInfo.name, '@kokopop/web');
  const { default: createModule } = await import(url('kokopop-runtime.js'));
  const { Runtime } = await import(url('runtime.js'));

  const data = readFileSync(resolve(dist, 'kokopop-runtime.data'));
  const module = await createModule({
    wasmBinary: readFileSync(resolve(dist, 'kokopop-runtime.wasm')),
    getPreloadedPackage: () => data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
  });
  const runtime = new Runtime(module);

  const info = await runtime.load(readFileSync(model), 1);
  assert.equal(info.architecture, architecture);
  assert.equal(info.backend, 'cpu');
  assert.ok(info.voices.length > 0);
  console.log(JSON.stringify(info));

  for (const voice of info.voices) {
    await testVoice(runtime, voice);
  }

  await assert.rejects(runtime.synthesize('hello', { speed: 0 }), /speed/);
  await assert.rejects(runtime.synthesize('hello', { voice: 'not-a-voice' }));
  await runtime.dispose();
  await runtime.dispose();
  await assert.rejects(runtime.synthesize('hello'), /No model/);

  console.log('WASM smoke passed');
}

await main();
