// Internal worker engine; calls are serialized by worker.js (JSPI is reentrant).
export class Runtime {
  constructor(module) {
    this.module = module;
    this.model = 0;
  }

  call(name, type, types = [], args = [], async = false) {
    return this.module.ccall(`kp_${name}`, type, types, args, { async });
  }

  error() {
    return new Error(this.call('error', 'string') || 'kokopop operation failed');
  }

  async load(bytes, backend) {
    if (this.model) throw new Error('A model is already loaded');

    const path = '/model.gguf';
    this.module.FS.writeFile(path, new Uint8Array(bytes));

    try {
      this.model = await this.call('load', 'number', ['string', 'number'], [path, backend], true);
      if (!this.model) throw this.error();

      const voices = [];
      const count = this.call('voice_count', 'number', ['number'], [this.model]);
      for (let i = 0; i < count; i++) {
        const name = this.call('voice_name', 'string', ['number', 'number'], [this.model, i]);
        const sampleRate = this.call('voice_rate', 'number', ['number', 'string'], [this.model, name]);
        voices.push({ name, sampleRate });
      }

      return {
        voices,
        backend: this.call('backend', 'number', ['number'], [this.model]) === 6 ? 'webgpu' : 'cpu',
        architecture: this.call('arch', 'string', ['number'], [this.model]),
      };
    } finally {
      this.module.FS.unlink(path);
    }
  }

  async synthesize(text, { voice = '', speed = 1, phonemes = false } = {}) {
    if (!this.model) throw new Error('No model loaded');
    if (typeof text !== 'string' || !text.trim() || text.includes('\0')) {
      throw new TypeError('text must be a nonempty string without NUL');
    }
    if (typeof voice !== 'string' || voice.includes('\0')) {
      throw new TypeError('voice must be a string without NUL');
    }
    if (!Number.isFinite(speed) || speed <= 0) {
      throw new RangeError('speed must be positive and finite');
    }

    const audio = await this.call(
      'synthesize',
      'number',
      ['number', 'string', 'string', 'number', 'number'],
      [this.model, text, voice, speed, Number(phonemes)],
      true,
    );
    if (!audio) throw this.error();

    try {
      const start = this.call('samples', 'number', ['number'], [audio]) / 4;
      const length = this.call('length', 'number', ['number'], [audio]);
      const sampleRate = this.call('rate', 'number', ['number'], [audio]);
      // Copy before freeing native audio; never expose a view into growing WASM memory.
      return { samples: this.module.HEAPF32.slice(start, start + length), sampleRate };
    } finally {
      this.call('audio_free', null, ['number'], [audio]);
    }
  }

  async dispose() {
    if (this.model) {
      await this.call('free', null, ['number'], [this.model], true);
      this.model = 0;
    }
  }
}
