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
    maxInFlightFrames = 64,
    detectorTimeoutMs = 5000,
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
    if (
      !Number.isSafeInteger(maxInFlightFrames) ||
      maxInFlightFrames < 1 ||
      maxInFlightFrames > 256
    ) {
      throw new RangeError("maxInFlightFrames must be an integer between 1 and 256");
    }
    this.maxInFlightFrames = maxInFlightFrames;
    if (
      !Number.isSafeInteger(detectorTimeoutMs) ||
      detectorTimeoutMs < 1 ||
      detectorTimeoutMs > 30000
    ) {
      throw new RangeError("detectorTimeoutMs must be an integer between 1 and 30000");
    }
    this.detectorTimeoutMs = detectorTimeoutMs;
    this.inFlightCount = 0;
    this.isBotSpeaking = isBotSpeaking;
    this.onSpeechStart = onSpeechStart;
    this.onSpeechEnd = onSpeechEnd;
    this.onBargeIn = onBargeIn;
    this.onNoiseDrop = onNoiseDrop;
    this.speakers = new Map();
    this.quarantinedUsers = new Map();
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

    if (this.quarantinedUsers.has(userId)) {
      return { ignored: true, dropped: "detector_quarantined" };
    }
    if (this.inFlightCount + this.quarantinedUsers.size >= this.maxInFlightFrames) {
      return { ignored: true, dropped: "global_backpressure" };
    }

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
    if (this.inFlightCount >= this.maxInFlightFrames) {
      return { ignored: true, dropped: "global_backpressure" };
    }

    state.pendingCount += 1;
    this.inFlightCount += 1;
    const task = state.queue.then(() => this.#processFrame(state, pcm16kMono, timestampMs));
    state.queue = task.then(
      () => undefined,
      () => undefined,
    );

    let outcome;
    try {
      outcome = await task;
    } finally {
      state.pendingCount -= 1;
      this.inFlightCount -= 1;
    }
    // Release the ingress slot before external callbacks so slow STT or LLM
    // work downstream can never starve the VAD.
    if (typeof outcome?.post === "function") await outcome.post();
    return outcome?.result;
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
      inFlightDetector: null,
    };
  }

  #admitSpeaker(state, timestampMs) {
    if (this.speakers.size + this.quarantinedUsers.size < this.maxSpeakers) {
      this.speakers.set(state.userId, state);
      return true;
    }

    this.#pruneIdle(timestampMs);
    if (this.speakers.size + this.quarantinedUsers.size < this.maxSpeakers) {
      this.speakers.set(state.userId, state);
      return true;
    }

    let oldestId = null;
    let oldestSeen = Infinity;
    for (const [userId, candidate] of this.speakers) {
      // Busy speakers have in-flight detector work whose result would be
      // wasted if their state vanished mid-await.
      if (candidate.active || candidate.pendingCount > 0) continue;
      if (candidate.lastSeenAt < oldestSeen) {
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
      // Never touch work that is queued or running; an idle entry (even one
      // still flagged active because its connection dropped without silence)
      // can be expired once it has been silent past the TTL.
      if (state.pendingCount > 0) continue;
      if (nowMs - state.lastSeenAt >= this.idleSpeakerTtlMs) {
        this.#dropSpeaker(userId);
      }
    }
  }

  #dropSpeaker(userId) {
    const state = this.speakers.get(userId);
    if (!state) return;
    // Keep tracking any still-running detector work before the entry vanishes.
    if (state.inFlightDetector) {
      this.quarantinedUsers.set(userId, state.inFlightDetector);
    }
    // Delete before reset so a detector reset that calls back into the VAD
    // cannot recurse through this speaker entry.
    this.speakers.delete(userId);
    state.epoch += 1;
    state.detector?.reset?.();
  }

  #isCurrent(userId, state, epoch) {
    return (
      this.speakers.get(userId) === state &&
      state.epoch === epoch &&
      !this.ignoredUserIds.has(userId) &&
      userId !== this.botUserId
    );
  }

  async #runDetector(state, task) {
    const pending = Promise.resolve().then(task);
    state.inFlightDetector = pending;
    pending
      .finally(() => {
        if (state.inFlightDetector === pending) state.inFlightDetector = null;
        if (this.quarantinedUsers.get(state.userId) === pending) {
          this.quarantinedUsers.delete(state.userId);
        }
      })
      .catch(() => {});
    let timer;
    const timeout = new Promise((resolve, reject) => {
      timer = setTimeout(() => {
        reject(Object.assign(new Error("VAD detector timed out"), { code: "DETECTOR_TIMEOUT" }));
      }, this.detectorTimeoutMs);
    });
    try {
      return await Promise.race([pending, timeout]);
    } catch (error) {
      if (error?.code === "DETECTOR_TIMEOUT") {
        this.#quarantineDetector(state, pending);
      }
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  #quarantineDetector(state, pending) {
    state.active = false;
    state.chunks = [];
    state.bufferedBytes = 0;
    state.speechMs = 0;
    state.silenceMs = 0;
    state.bargeInFired = false;
    state.startedAt = null;
    state.detector = null;
    state.epoch += 1;
    this.quarantinedUsers.set(state.userId, pending);
    pending
      .finally(() => {
        if (this.quarantinedUsers.get(state.userId) === pending) {
          this.quarantinedUsers.delete(state.userId);
        }
      })
      .catch(() => {});
  }

  async #processFrame(state, pcm16kMono, timestampMs) {
    const { userId } = state;
    const startEpoch = state.epoch;

    // A frame can be queued and then dropped by resetSpeaker, clear, or a mute
    // before its turn comes. Bail out before creating a detector.
    if (!this.#isCurrent(userId, state, startEpoch)) {
      return { result: { ignored: true, dropped: "reset" } };
    }
    if (this.quarantinedUsers.has(userId)) {
      return { result: { ignored: true, dropped: "detector_quarantined" } };
    }

    if (!state.detector) {
      const detector = await this.#runDetector(state, () => this.detectorFactory(userId));
      if (!this.#isCurrent(userId, state, startEpoch)) {
        return { result: { ignored: true, dropped: "reset" } };
      }
      state.detector = detector;
    }

    if (!this.#isCurrent(userId, state, startEpoch)) {
      return { result: { ignored: true, dropped: "reset" } };
    }

    const samples = pcm16MonoToFloat32(pcm16kMono);
    const probability = await this.#runDetector(state, () => state.detector(samples));
    if (!this.#isCurrent(userId, state, startEpoch)) {
      return { result: { ignored: true, dropped: "reset" } };
    }

    const isSpeech = probability >= this.speechThreshold;
    let post;

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
        if (!this.#isCurrent(userId, state, startEpoch)) {
          return { result: { ignored: true, dropped: "reset" } };
        }
      }

      if (!state.bargeInFired && state.speechMs >= this.bargeInMs) {
        const botSpeaking = this.isBotSpeaking();
        if (!this.#isCurrent(userId, state, startEpoch)) {
          return { result: { ignored: true, dropped: "reset" } };
        }
        if (botSpeaking) {
          state.bargeInFired = true;
          this.onBargeIn?.({
            userId,
            displayName: state.displayName,
            timestampMs,
          });
          if (!this.#isCurrent(userId, state, startEpoch)) {
            return { result: { ignored: true, dropped: "reset" } };
          }
        }
      }

      if (
        state.bufferedBytes >= this.maxUtteranceBytes ||
        state.speechMs >= this.maxUtteranceMs
      ) {
        post = this.#finish(state, timestampMs);
      }
    } else if (state.active) {
      state.silenceMs += this.frameMs;
      if (state.silenceMs <= this.hangoverMs) {
        this.#appendChunk(state, pcm16kMono);
      }
      if (state.silenceMs >= this.hangoverMs || state.bufferedBytes >= this.maxUtteranceBytes) {
        post = this.#finish(state, timestampMs);
      }
    }

    return {
      result: { ignored: false, speech: isSpeech, probability },
      post,
    };
  }

  #appendChunk(state, frame) {
    state.chunks.push(copyPcm(frame));
    state.bufferedBytes += frame.byteLength;
  }

  #finish(state, timestampMs) {
    const chunks = state.chunks;
    const speechMs = state.speechMs;
    const { userId, displayName } = state;
    const epoch = state.epoch;

    state.active = false;
    state.speechMs = 0;
    state.silenceMs = 0;
    state.chunks = [];
    state.bufferedBytes = 0;
    state.bargeInFired = false;
    state.startedAt = null;
    state.detector?.reset?.();

    if (speechMs < this.minSpeechMs) {
      return async () => {
        if (!this.#isCurrent(userId, state, epoch)) return;
        this.onNoiseDrop?.({ userId, displayName, speechMs });
      };
    }

    return async () => {
      if (!this.#isCurrent(userId, state, epoch)) return;
      await this.onSpeechEnd?.({
        userId,
        displayName,
        pcm16kMono: Buffer.concat(chunks),
        durationMs: speechMs,
        endedAt: timestampMs,
      });
    };
  }

  /** Drop one speaker's buffered utterance and detector state. */
  resetSpeaker(userId) {
    this.#dropSpeaker(userId);
  }

  /** Drop every speaker's buffered utterance and detector state. */
  clear() {
    for (const state of this.speakers.values()) {
      if (state.inFlightDetector) {
        this.quarantinedUsers.set(state.userId, state.inFlightDetector);
      }
      state.epoch += 1;
      state.detector?.reset?.();
    }
    this.speakers.clear();
  }
}
