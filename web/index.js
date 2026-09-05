async function readModelBytes(model, signal) {
  if (typeof model === 'string' || model instanceof URL) {
    const response = await fetch(model, { signal });
    if (!response.ok) {
      throw new Error(`Model download failed: HTTP ${response.status}`);
    }
    return response.arrayBuffer();
  }

  if (model instanceof ArrayBuffer) {
    return model.slice(0);
  }

  if (model instanceof Uint8Array) {
    return model.slice().buffer;
  }

  throw new TypeError('model must be a URL, ArrayBuffer or Uint8Array');
}

export class Kokopop {
  #worker;
  #pending = new Map();
  #id = 0;
  #closed = false;
  #closing;

  constructor(worker) {
    this.#worker = worker;

    worker.onmessage = ({ data: { id, result, error } }) => {
      const pending = this.#pending.get(id);
      if (!pending) {
        return;
      }

      this.#pending.delete(id);
      if (error) {
        pending.reject(new Error(error));
      } else {
        pending.resolve(result);
      }
    };

    worker.onerror = (event) => this.#terminate(new Error(event.message || 'kokopop Worker failed'));
    worker.onmessageerror = () => this.#terminate(new Error('Invalid kokopop Worker response'));
  }

  #terminate(error) {
    this.#closed = true;
    this.#worker.terminate();

    for (const { reject } of this.#pending.values()) {
      reject(error);
    }
    this.#pending.clear();
  }

  #request(method, args = [], transfer = []) {
    if (this.#closed) {
      return Promise.reject(new Error('Kokopop is disposed'));
    }

    const id = ++this.#id;

    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      try {
        this.#worker.postMessage({ id, method, args }, transfer);
      } catch (error) {
        this.#pending.delete(id);
        reject(error);
      }
    });
  }

  static async create({ model, backend = 'auto', assetsUrl, workerUrl, signal } = {}) {
    if (!['auto', 'cpu', 'webgpu'].includes(backend)) {
      throw new TypeError('Unknown backend');
    }

    const bytes = await readModelBytes(model, signal);
    signal?.throwIfAborted();

    const worker = workerUrl
      ? new Worker(workerUrl, { type: 'module' })
      : new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

    const instance = new Kokopop(worker);

    const abort = () => instance.#terminate(signal.reason || new Error('Model loading aborted'));
    signal?.addEventListener('abort', abort, { once: true });

    try {
      const info = await instance.#request(
        'load',
        [
          bytes,
          {
            backend,
            assetsUrl: assetsUrl ? new URL(assetsUrl, globalThis.location.href).href : undefined,
          },
        ],
        [bytes],
      );

      instance.voices = Object.freeze(info.voices.map(Object.freeze));
      instance.backend = info.backend;
      instance.architecture = info.architecture;

      return instance;
    } catch (error) {
      instance.#terminate(error);
      throw error;
    } finally {
      signal?.removeEventListener('abort', abort);
    }
  }

  synthesize(text, options = {}) {
    if (this.#closing) {
      return Promise.reject(new Error('Kokopop is disposing'));
    }
    return this.#request('synthesize', [text, options]);
  }

  dispose() {
    if (this.#closing) {
      return this.#closing;
    }
    if (this.#closed) {
      return Promise.resolve();
    }

    this.#closing = this.#request('dispose').finally(() => this.#terminate(new Error('Kokopop is disposed')));
    return this.#closing;
  }
}
