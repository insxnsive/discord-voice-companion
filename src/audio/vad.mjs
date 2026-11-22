// Voice activity detection and barge-in tracking. The RMS detector is the
// dependency-free default; Silero runs through an injected ONNX runtime so the
// unit suite never loads a native model.

import { pcm16MonoToFloat32 } from "./pcm.mjs";

const SILERO_STATE_SHAPE = [2, 1, 128];
const SILERO_STATE_VALUES = 2 * 128;

/** Root-mean-square amplitude of float32 samples, in [0, 1]. Empty -> 0. */
export function computeRms(float32Samples) {
  if (!float32Samples || float32Samples.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < float32Samples.length; i += 1) {
    sum += float32Samples[i] * float32Samples[i];
  }
  return Math.sqrt(sum / float32Samples.length);
}

/**
 * Level-based detector. Loud frames report 0.9; quieter frames fall off
 * smoothly so the shared speechThreshold still sits near the configured level.
 */
export function createRmsSpeechDetector({ threshold = 0.018 } = {}) {
  if (!(threshold > 0)) {
    throw new RangeError("threshold must be greater than 0");
  }
  return function detect(float32Samples) {
    const rms = computeRms(float32Samples);
    if (rms >= threshold) return 0.9;
    return 0.9 * (rms / threshold) ** 6;
  };
}

/**
 * Silero VAD through onnxruntime-node. The session keeps the recurrent state
 * tensor between calls; reset() clears it at utterance boundaries.
 */
export async function createSileroDetector({ modelPath, ort, sampleRate = 16000 } = {}) {
  if (!modelPath) {
    throw new Error("createSileroDetector requires the path to a Silero VAD ONNX model");
  }
  const runtime = ort ?? (await import("onnxruntime-node"));
  const session = await runtime.InferenceSession.create(modelPath, {
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
  });

  const sr = new runtime.Tensor("int64", BigInt64Array.from([BigInt(sampleRate)]), []);
  let state = new runtime.Tensor("float32", new Float32Array(SILERO_STATE_VALUES), SILERO_STATE_SHAPE);

  async function detect(float32Samples) {
    const samples =
      float32Samples instanceof Float32Array ? float32Samples : Float32Array.from(float32Samples);
    const input = new runtime.Tensor("float32", samples, [1, samples.length]);
    const results = await session.run({ input, state, sr });

    const probability = results.output ?? results[session.outputNames?.[0]] ?? Object.values(results)[0];
    const nextState = results.stateN ?? Object.values(results)[1];
    if (nextState) state = nextState;

    const value = probability?.data?.[0];
    return typeof value === "number" ? value : Number(value ?? 0);
  }

  detect.reset = () => {
    state = new runtime.Tensor("float32", new Float32Array(SILERO_STATE_VALUES), SILERO_STATE_SHAPE);
  };

  return detect;
}

function copyPcm(input) {
  return Buffer.from(input);
}

/**
 * Per-speaker utterance state machine: VAD, noise-blip filtering, hangover
 * merging, and barge-in detection while the bot is talking. One detector
 * instance is kept per speaker so recurrent state never crosses speakers.
 */
export class PerSpeakerVad {
  constructor({
    botUserId = null,
    ignoredUserIds = [],
    detectorFactory = () => createRmsSpeechDetector(),
    speechThreshold = 0.5,
    minSpeechMs = 250,
    hangoverMs = 240,
    bargeInMs = 180,
    maxUtteranceMs = 15000,
    frameMs = 20,
    isBotSpeaking = () => false,
    onSpeechStart,
    onSpeechEnd,
    onBargeIn,
    onNoiseDrop,
  } = {}) {
    this.botUserId = botUserId;
    this.ignoredUserIds = new Set(ignoredUserIds);
    this.detectorFactory = detectorFactory;
    this.speechThreshold = speechThreshold;
    this.minSpeechMs = minSpeechMs;
    this.hangoverMs = hangoverMs;
    this.bargeInMs = bargeInMs;
    this.maxUtteranceMs = maxUtteranceMs;
    this.frameMs = frameMs;
    this.isBotSpeaking = isBotSpeaking;
    this.onSpeechStart = onSpeechStart;
    this.onSpeechEnd = onSpeechEnd;
    this.onBargeIn = onBargeIn;
    this.onNoiseDrop = onNoiseDrop;
    this.speakers = new Map();
  }

  /** Replace the ignored-user set without touching active speaker state. */
  setIgnoredUsers(ids = []) {
    this.ignoredUserIds = new Set(ids);
  }

  async pushFrame({
    userId,
    displayName = "Unknown",
    pcm16kMono,
    timestampMs = Date.now(),
  } = {}) {
    if (!userId || userId === this.botUserId || this.ignoredUserIds.has(userId)) {
      return { ignored: true };
    }

    let state = this.speakers.get(userId);
    if (!state) {
      state = {
        userId,
        displayName,
        detector: null,
        active: false,
        speechMs: 0,
        silenceMs: 0,
        chunks: [],
        bargeInFired: false,
        startedAt: null,
      };
      this.speakers.set(userId, state);
    }
    state.displayName = displayName ?? "Unknown";

    if (!state.detector) {
      state.detector = await this.detectorFactory(userId);
    }

    const samples = pcm16MonoToFloat32(pcm16kMono);
    const probability = await state.detector(samples);
    const isSpeech = probability >= this.speechThreshold;

    if (isSpeech) {
      state.speechMs += this.frameMs;
      state.silenceMs = 0;
      state.chunks.push(copyPcm(pcm16kMono));

      if (!state.active) {
        state.active = true;
        state.startedAt = timestampMs;
        this.onSpeechStart?.({
          userId,
          displayName: state.displayName,
          timestampMs,
        });
      }

      if (!state.bargeInFired && state.speechMs >= this.bargeInMs && this.isBotSpeaking()) {
        state.bargeInFired = true;
        this.onBargeIn?.({
          userId,
          displayName: state.displayName,
          timestampMs,
        });
      }

      if (state.speechMs >= this.maxUtteranceMs) {
        await this.#finish(state, timestampMs);
      }
    } else if (state.active) {
      state.silenceMs += this.frameMs;
      if (state.silenceMs <= this.hangoverMs) {
        state.chunks.push(copyPcm(pcm16kMono));
      }
      if (state.silenceMs >= this.hangoverMs) {
        await this.#finish(state, timestampMs);
      }
    }

    return { ignored: false, speech: isSpeech, probability };
  }

  async #finish(state, timestampMs) {
    const chunks = state.chunks;
    const speechMs = state.speechMs;
    const { userId, displayName } = state;

    state.active = false;
    state.speechMs = 0;
    state.silenceMs = 0;
    state.chunks = [];
    state.bargeInFired = false;
    state.startedAt = null;
    state.detector?.reset?.();

    if (speechMs < this.minSpeechMs) {
      this.onNoiseDrop?.({ userId, displayName, speechMs });
      return;
    }

    await this.onSpeechEnd?.({
      userId,
      displayName,
      pcm16kMono: Buffer.concat(chunks),
      durationMs: speechMs,
      endedAt: timestampMs,
    });
  }

  /** Drop one speaker's buffered utterance and detector state. */
  resetSpeaker(userId) {
    const state = this.speakers.get(userId);
    state?.detector?.reset?.();
    this.speakers.delete(userId);
  }

  /** Drop every speaker's buffered utterance and detector state. */
  clear() {
    for (const state of this.speakers.values()) {
      state.detector?.reset?.();
    }
    this.speakers.clear();
  }
}
