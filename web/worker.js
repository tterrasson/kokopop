import createModule from './kokopop-runtime.js';
import { webgpu } from './build-info.js';
import { Runtime } from './runtime.js';

let runtime;
let queue = Promise.resolve();

const BACKEND_VALUES = { auto: 0, cpu: 1, webgpu: 6 };

async function resolveBackend(requested) {
  if (webgpu && typeof WebAssembly.Suspending !== 'function') {
    throw new Error('This WebGPU package requires WebAssembly JSPI; use the CPU-only package in this browser');
  }
  if (!webgpu && requested === 'webgpu') {
    throw new Error('WebGPU was not compiled into this package');
  }
  if (!webgpu) return 'cpu';

  let backend = requested;
  if (backend !== 'cpu') {
    const adapter = navigator.gpu ? await navigator.gpu.requestAdapter().catch(() => null) : null;
    // GGML requires shader-f16 and asserts on adapter/device creation failure.
    const available = adapter?.features.has('shader-f16');
    if (!available && backend === 'webgpu') {
      throw new Error('No WebGPU adapter with shader-f16 support is available');
    }
    if (!available) backend = 'cpu';
  }
  return backend;
}

async function load(bytes, options) {
  const backend = await resolveBackend(options.backend);
  const module = await createModule({
    locateFile: (name) => new URL(name, options.assetsUrl || import.meta.url).href,
  });
  runtime = new Runtime(module);
  return runtime.load(bytes, BACKEND_VALUES[backend]);
}

async function handle(method, args) {
  switch (method) {
    case 'load':
      return load(...args);
    case 'synthesize':
      return runtime.synthesize(...args);
    case 'dispose':
      await runtime?.dispose();
      return undefined;
    default:
      throw new Error(`Unknown operation: ${method}`);
  }
}

self.onmessage = ({ data: { id, method, args } }) => {
  queue = queue.then(async () => {
    try {
      const result = await handle(method, args);
      self.postMessage({ id, result }, result?.samples ? [result.samples.buffer] : []);
    } catch (error) {
      self.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
    }
  });
};
