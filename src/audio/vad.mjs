// Voice activity detection and barge-in tracking. The RMS detector is the
// dependency-free default; Silero runs through an injected ONNX runtime so the
// unit suite never loads a native model.

import { VAD_SAMPLE_RATE, pcm16MonoToFloat32 } from "./pcm.mjs";

const SILERO_SAMPLE_RATE = 16000;
const SILERO_WINDOW_SAMPLES = 512;
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

function assertModelPath(modelPath) {
  if (typeof modelPath !== "string" || modelPath.trim().length === 0) {
    throw new TypeError("createSileroDetector requires a non-empty modelPath string");
  }
  // Control characters or a null byte in a path only ever hide something.
  if (/[\u0000-\u001f\u007f]/.test(modelPath)) {
    throw new RangeError("modelPath must not contain null bytes or control characters");
  }
}

function toFiniteFloat32(input) {
  if (input === null || input === undefined || typeof input.length !== "number") {
    throw new TypeError("samples must be a Float32Array or array-like of numbers");
  }
  const samples = input instanceof Float32Array ? input : Float32Array.from(input);
  for (let i = 0; i < samples.length; i += 1) {
    if (!Number.isFinite(samples[i])) {
      throw new TypeError("samples must contain only finite numbers");
    }
  }
  return samples;
}

function assertStateTensor(tensor) {
  const dims = tensor?.dims;
  const values = tensor?.data?.length ?? 0;
  const shapeOk =
    Array.isArray(dims) &&
    dims.length === 3 &&
    dims[0] === 2 &&
    dims[1] === 1 &&
    dims[2] === 128;
  if (!shapeOk || values !== SILERO_STATE_VALUES) {
    throw new Error(
      `Silero returned an unexpected state tensor (expected [2, 1, 128] with ${SILERO_STATE_VALUES} floats)`,
    );
  }
  return tensor;
}

/**
 * Silero VAD through onnxruntime-node. Silero v5 only scores 512-sample 16kHz
 * windows, so arbitrary chunk sizes are rebuffered here. The session keeps the
 * recurrent state tensor between windows; reset() clears both buffers.
 */
export async function createSileroDetector({ modelPath, ort, sampleRate = 16000 } = {}) {
  if (sampleRate !== SILERO_SAMPLE_RATE) {
    throw new RangeError(
      `Silero VAD expects a ${SILERO_SAMPLE_RATE} Hz sample rate, got ${sampleRate}`,
    );
  }
  assertModelPath(modelPath);

  const runtime = ort ?? (await import("onnxruntime-node"));
  const session = await runtime.InferenceSession.create(modelPath, {
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
  });

  const sr = new runtime.Tensor("int64", BigInt64Array.from([BigInt(sampleRate)]), []);
  let state = new runtime.Tensor(
    "float32",
    new Float32Array(SILERO_STATE_VALUES),
    SILERO_STATE_SHAPE,
  );
  let pending = new Float32Array(0);
  let lastProbability = 0;

  async function runWindow(windowSamples) {
    const input = new runtime.Tensor("float32", windowSamples, [1, windowSamples.length]);
    const results = await session.run({ input, state, sr });

    const probability = results.output ?? results[session.outputNames?.[0]] ?? Object.values(results)[0];
    const nextState = results.stateN ?? Object.values(results)[1];
    if (nextState) state = assertStateTensor(nextState);

    const value = probability?.data?.[0];
    return typeof value === "number" ? value : Number(value ?? 0);
  }

  async function detect(float32Samples) {
    const samples = toFiniteFloat32(float32Samples);

    if (samples.length > 0) {
      const merged = new Float32Array(pending.length + samples.length);
      merged.set(pending, 0);
      merged.set(samples, pending.length);

      let offset = 0;
      let maxProbability = null;
      while (merged.length - offset >= SILERO_WINDOW_SAMPLES) {
        const windowSamples = merged.subarray(offset, offset + SILERO_WINDOW_SAMPLES);
        const probability = await runWindow(windowSamples);
        maxProbability = maxProbability === null ? probability : Math.max(maxProbability, probability);
        offset += SILERO_WINDOW_SAMPLES;
      }

      pending =
        offset === merged.length
          ? new Float32Array(0)
          : Float32Array.from(merged.subarray(offset));
      if (maxProbability !== null) lastProbability = maxProbability;
    }

    return lastProbability;
  }

  detect.reset = () => {
    state = new runtime.Tensor(
      "float32",
      new Float32Array(SILERO_STATE_VALUES),
      SILERO_STATE_SHAPE,
    );
    pending = new Float32Array(0);
    lastProbability = 0;
  };

  return detect;
}

function copyPcm(input) {
  return Buffer.from(input);
}

/**
 * Per-speaker utterance state machine: VAD, noise-blip filtering, hangover
 * merging, and barge-in detection while the bot is talking. Frames are
 * validated, per-speaker work is serialized through a promise chain, speaker
 * state is bounded, and every await re-checks that the speaker was not muted
 * or reset in the meantime.
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
    expectedFrameBytes = (VAD_SAMPLE_RATE / 1000) * frameMs * 2,
    maxFrameBytes = 6400,
    maxUtteranceBytes = 512000,
    maxSpeakers = 32,
    idleSpeakerTtlMs = 60000,
    maxPendingFrames = 8,
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
    this.expectedFrameBytes = expectedFrameBytes;
    this.maxFrameBytes = maxFrameBytes;
    this.maxUtteranceBytes = maxUtteranceBytes;
    this.maxSpeakers = maxSpeakers;
    this.idleSpeakerTtlMs = idleSpeakerTtlMs;
    this.maxPendingFrames = maxPendingFrames;
    this.isBotSpeaking = isBotSpeaking;
    this.onSpeechStart = onSpeechStart;
    this.onSpeechEnd = onSpeechEnd;
    this.onBargeIn = onBargeIn;
    this.onNoiseDrop = onNoiseDrop;
    this.speakers = new Map();
  }

  /** Replace the ignored-user set and drop any newly muted speaker's state. */
  setIgnoredUsers(ids = []) {
    this.ignoredUserIds = new Set(ids);
    for (const userId of [...this.speakers.keys()]) {
      if (this.ignoredUserIds.has(userId)) this.#dropSpeaker(userId);
    }
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

    this.#validateFrame(pcm16kMono);
    this.#pruneIdle(timestampMs);

    let state = this.speakers.get(userId);
    if (!state) {
      state = this.#createState(userId, displayName, timestampMs);
      if (!this.#admitSpeaker(state, timestampMs)) {
        return { ignored: true, dropped: "max_speakers" };
      }
    }

    state.displayName = displayName ?? "Unknown";
    state.lastSeenAt = timestampMs;

    if (state.pendingCount >= this.maxPendingFrames) {
      return { ignored: true, dropped: "backpressure" };
    }

    state.pendingCount += 1;
    const task = state.queue.then(() => this.#processFrame(state, pcm16kMono, timestampMs));
    state.queue = task.then(
      () => undefined,
      () => undefined,
    );

    try {
      return await task;
    } finally {
      state.pendingCount -= 1;
    }
  }

  #validateFrame(frame) {
    if (!Buffer.isBuffer(frame) && !(frame instanceof Uint8Array)) {
      throw new TypeError("pcm16kMono must be a Buffer or Uint8Array");
    }
    const size = frame.byteLength;
    if (size < 2 || size > this.maxFrameBytes) {
      throw new RangeError(
        `pcm16kMono must be 2..${this.maxFrameBytes} bytes ` +
          `(expected ${this.expectedFrameBytes}), got ${size}`,
      );
    }
    if (size % 2 !== 0) {
      throw new RangeError(`pcm16kMono must contain whole 16-bit samples, got ${size} bytes`);
    }
  }

  #createState(userId, displayName, timestampMs) {
    return {
      userId,
      displayName,
      detector: null,
      active: false,
      speechMs: 0,
      silenceMs: 0,
      chunks: [],
      bufferedBytes: 0,
      bargeInFired: false,
      startedAt: null,
      lastSeenAt: timestampMs,
      epoch: 0,
      queue: Promise.resolve(),
      pendingCount: 0,
    };
  }

  #admitSpeaker(state, timestampMs) {
    if (this.speakers.size < this.maxSpeakers) {
      this.speakers.set(state.userId, state);
      return true;
    }

    this.#pruneIdle(timestampMs);
    if (this.speakers.size < this.maxSpeakers) {
      this.speakers.set(state.userId, state);
      return true;
    }

    let oldestId = null;
    let oldestSeen = Infinity;
    for (const [userId, candidate] of this.speakers) {
      if (!candidate.active && candidate.lastSeenAt < oldestSeen) {
        oldestSeen = candidate.lastSeenAt;
        oldestId = userId;
      }
    }
    if (oldestId === null) return false;

    this.#dropSpeaker(oldestId);
    this.speakers.set(state.userId, state);
    return true;
  }

  #pruneIdle(nowMs) {
    for (const [userId, state] of [...this.speakers]) {
      if (!state.active && nowMs - state.lastSeenAt >= this.idleSpeakerTtlMs) {
        this.#dropSpeaker(userId);
      }
    }
  }

  #dropSpeaker(userId) {
    const state = this.speakers.get(userId);
    if (!state) return;
    state.epoch += 1;
    state.detector?.reset?.();
    this.speakers.delete(userId);
  }

  #isCurrent(state, epoch) {
    return (
      this.speakers.get(state.userId) === state &&
      state.epoch === epoch &&
      !this.ignoredUserIds.has(state.userId) &&
      state.userId !== this.botUserId
    );
  }

  async #processFrame(state, pcm16kMono, timestampMs) {
    const { userId } = state;
    const startEpoch = state.epoch;

    if (!state.detector) {
      const detector = await this.detectorFactory(userId);
      if (!this.#isCurrent(state, startEpoch)) {
        return { ignored: true, dropped: "reset" };
      }
      state.detector = detector;
    }

    const samples = pcm16MonoToFloat32(pcm16kMono);
    const probability = await state.detector(samples);
    if (!this.#isCurrent(state, startEpoch)) {
      return { ignored: true, dropped: "reset" };
    }

    const isSpeech = probability >= this.speechThreshold;

    if (isSpeech) {
      state.speechMs += this.frameMs;
      state.silenceMs = 0;
      this.#appendChunk(state, pcm16kMono);

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

      if (
        state.bufferedBytes >= this.maxUtteranceBytes ||
        state.speechMs >= this.maxUtteranceMs
      ) {
        await this.#finish(state, timestampMs);
      }
    } else if (state.active) {
      state.silenceMs += this.frameMs;
      if (state.silenceMs <= this.hangoverMs) {
        this.#appendChunk(state, pcm16kMono);
      }
      if (state.silenceMs >= this.hangoverMs || state.bufferedBytes >= this.maxUtteranceBytes) {
        await this.#finish(state, timestampMs);
      }
    }

    return { ignored: false, speech: isSpeech, probability };
  }

  #appendChunk(state, frame) {
    state.chunks.push(copyPcm(frame));
    state.bufferedBytes += frame.byteLength;
  }

  async #finish(state, timestampMs) {
    const chunks = state.chunks;
    const speechMs = state.speechMs;
    const { userId, displayName } = state;

    state.active = false;
    state.speechMs = 0;
    state.silenceMs = 0;
    state.chunks = [];
    state.bufferedBytes = 0;
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
    this.#dropSpeaker(userId);
  }

  /** Drop every speaker's buffered utterance and detector state. */
  clear() {
    for (const state of this.speakers.values()) {
      state.epoch += 1;
      state.detector?.reset?.();
    }
    this.speakers.clear();
  }
}
