export type Backend = 'auto' | 'cpu' | 'webgpu';

export interface Voice {
  readonly name: string;
  readonly sampleRate: number;
}

export interface Audio {
  samples: Float32Array;
  sampleRate: number;
}

export interface CreateOptions {
  model: string | URL | ArrayBuffer | Uint8Array;
  backend?: Backend;
  /** Absolute or page-relative directory URL ending in / for .wasm and .data assets. */
  assetsUrl?: string | URL;
  /** Optional URL of the module Worker (for public-directory deployments). */
  workerUrl?: string | URL;
  /** Cancel the model download or loading operation. */
  signal?: AbortSignal;
}

export interface SynthesisOptions {
  voice?: string;
  speed?: number;
  /** Input is already phonemized, using the selected model's phoneme convention. */
  phonemes?: boolean;
}

export class Kokopop {
  private constructor();
  static create(options: CreateOptions): Promise<Kokopop>;

  readonly voices: readonly Voice[];
  readonly backend: Exclude<Backend, 'auto'>;
  readonly architecture: string;

  /** Calls are queued. PCM is mono, independently owned and safe after dispose(). */
  synthesize(text: string, options?: SynthesisOptions): Promise<Audio>;

  /** Waits for queued synthesis, releases model and terminates the Worker. */
  dispose(): Promise<void>;
}
