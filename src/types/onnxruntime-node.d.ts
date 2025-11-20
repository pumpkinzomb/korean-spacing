declare module 'onnxruntime-node' {
  export interface SessionOptions {
    executionProviders?: string[];
  }

  export interface RunOptions {
    logId?: string;
    logSeverityLevel?: number;
  }

  export class Tensor<T extends number | bigint = number> {
    constructor(type: string, data: T[] | BigInt64Array | Float32Array | Int32Array, dims: number[]);
    readonly data: Float32Array | BigInt64Array | Int32Array;
    readonly dims: number[];
  }

  export class InferenceSession {
    static create(modelPath: string, options?: SessionOptions): Promise<InferenceSession>;
    run(
      feeds: Record<string, Tensor>,
      options?: RunOptions
    ): Promise<Record<string, Tensor<Float32Array | BigInt64Array | Int32Array>>>;
  }
}

