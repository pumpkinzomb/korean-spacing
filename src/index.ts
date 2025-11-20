import { InferenceSession, Tensor } from "onnxruntime-node";
import path from "node:path";
import fs from "node:fs";

const MAX_LENGTH = 512;
const VOCAB_SIZE = 65536;
const PAD_TOKEN_ID = 0;
const UNK_TOKEN_ID = VOCAB_SIZE - 1;

const SPACE_LABEL = 1;
const NEWLINE_LABEL = 2;

function charToId(char: string): number {
  const codePoint = char.codePointAt(0) ?? 0;
  const shifted = codePoint + 1; // 0은 PAD 전용
  if (shifted >= VOCAB_SIZE) {
    return UNK_TOKEN_ID;
  }
  return shifted;
}

export class KoreanSpacing {
  private session: InferenceSession | null = null;
  private modelPath: string;

  constructor(modelPath?: string) {
    this.modelPath =
      modelPath || path.join(__dirname, "../models/korean_spacing.onnx");
  }

  async load(): Promise<void> {
    if (!fs.existsSync(this.modelPath)) {
      throw new Error(`Model file not found: ${this.modelPath}`);
    }

    this.session = await InferenceSession.create(this.modelPath, {
      executionProviders: ["cpu"],
    });
  }

  private tokenize(text: string): {
    inputIds: number[];
    attentionMask: number[];
    length: number;
  } {
    const chars = Array.from(text);
    const inputIds: number[] = [];
    const attentionMask: number[] = [];

    for (let i = 0; i < Math.min(chars.length, MAX_LENGTH); i++) {
      inputIds.push(charToId(chars[i]));
      attentionMask.push(1);
    }

    const sequenceLength = inputIds.length;

    while (inputIds.length < MAX_LENGTH) {
      inputIds.push(PAD_TOKEN_ID);
      attentionMask.push(0);
    }

    return {
      inputIds,
      attentionMask,
      length: sequenceLength,
    };
  }

  async correct(text: string): Promise<string> {
    if (!this.session) {
      await this.load();
    }

    if (!text || text.trim().length === 0) {
      return text;
    }

    const { inputIds, attentionMask, length } = this.tokenize(text);

    const inputIdsTensor = new Tensor(
      "int64",
      new BigInt64Array(inputIds.map(BigInt)),
      [1, inputIds.length]
    );
    const attentionMaskTensor = new Tensor(
      "int64",
      new BigInt64Array(attentionMask.map(BigInt)),
      [1, attentionMask.length]
    );

    const feeds = {
      input_ids: inputIdsTensor,
      attention_mask: attentionMaskTensor,
    };

    const results = await this.session!.run(feeds);
    const predictions = results.logits || results.output;

    if (!predictions) {
      throw new Error("Model output not found");
    }

    const outputData = predictions.data as Float32Array;
    const outputShape = predictions.dims;
    const numLabels = outputShape[outputShape.length - 1];

    let result = "";
    const chars = Array.from(text).slice(0, length);

    for (let i = 0; i < chars.length; i++) {
      const startIdx = i * numLabels;
      const probs = outputData.slice(startIdx, startIdx + numLabels);
      let maxProbIdx = 0;
      let maxProb = Number.NEGATIVE_INFINITY;
      for (let j = 0; j < probs.length; j++) {
        if (probs[j] > maxProb) {
          maxProb = probs[j];
          maxProbIdx = j;
        }
      }

      result += chars[i];

      if (maxProbIdx === SPACE_LABEL) {
        result += " ";
      } else if (maxProbIdx === NEWLINE_LABEL) {
        result += "\n";
      }
    }

    return result;
  }

  async correctBatch(texts: string[]): Promise<string[]> {
    const results: string[] = [];
    for (const text of texts) {
      results.push(await this.correct(text));
    }
    return results;
  }
}

export default KoreanSpacing;
