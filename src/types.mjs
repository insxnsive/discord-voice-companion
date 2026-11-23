// Shared contracts for provider adapters, reaction cues, and live latency
// accounting. Everything here is dependency-free so unit tests run on plain
// Node 22 without native audio packages.

export const PROVIDERS = Object.freeze({
  llm: Object.freeze([
    "opencode",
    "deepseek",
    "openai",
    "openrouter",
    "groq",
    "ollama",
    "custom",
  ]),
  stt: Object.freeze(["deepgram", "elevenlabs", "openai", "groq", "faster-whisper"]),
  tts: Object.freeze(["elevenlabs", "openai", "openai-compatible"]),
});

export const REACTION_MOODS = Object.freeze([
  "subtle_breath",
  "curious_hmm",
  "light_chuckle",
  "playful_scoff",
]);

export const FILLER_MODES = Object.freeze(["adaptive", "subtle", "off"]);

export const CHIME_MODES = Object.freeze(["smart", "always", "mention"]);

// Timestamp stages a turn can report. `bargeInCancelled` is a flag rather than
// a timestamp, but mark() accepts it so callers have one code path.
const TIMESTAMP_STAGES = Object.freeze([
  "vadEndAt",
  "sttDoneAt",
  "gateDoneAt",
  "llmFirstClauseAt",
  "ttsFirstByteAt",
  "opusFirstFrameAt",
  "fillerFiredAt",
]);

const STAGES = new Set([...TIMESTAMP_STAGES, "bargeInCancelled"]);

const METRIC_FIELDS = Object.freeze([
  "sttMs",
  "gateMs",
  "llmFirstClauseMs",
  "ttsFirstByteMs",
  "opusEncodeMs",
  "totalLiveMs",
  "perceivedMs",
]);

/**
 * Verify that an adapter object implements every method its slot requires.
 * Fails loudly at wiring time instead of mid-call in a voice session.
 */
export function assertAdapterContract(adapter, requiredMethods, label = "adapter") {
  if (!Array.isArray(requiredMethods)) {
    throw new TypeError(`${label} requiredMethods must be an array of method names`);
  }
  if (adapter === null || typeof adapter !== "object") {
    throw new TypeError(
      `${label} must be a non-null object (required: ${requiredMethods.join(", ")})`,
    );
  }
  const missing = requiredMethods.filter((name) => typeof adapter[name] !== "function");
  if (missing.length > 0) {
    throw new TypeError(`${label} is missing required method(s): ${missing.join(", ")}`);
  }
}

const roundMs = (from, to) => {
  if (from === null || from === undefined || to === null || to === undefined) return null;
  return Math.round(to - from);
};

/**
 * Per-turn latency ledger. Each turn keeps the six stage durations plus the
 * total live and perceived figures, and finished turns roll into a bounded
 * history for `/latency` averages.
 */
export class LatencyTracker {
  constructor({ historyLimit = 50, now = () => performance.now() } = {}) {
    this.historyLimit =
      Number.isInteger(historyLimit) && historyLimit >= 0 ? historyLimit : 50;
    this.now = typeof now === "function" ? now : () => performance.now();
    this.history = [];
    this.activeTurns = new Map();
  }

  startTurn(turnId, t0 = this.now()) {
    const turn = {
      turnId,
      startedAt: t0,
      vadEndAt: t0,
      sttDoneAt: null,
      gateDoneAt: null,
      llmFirstClauseAt: null,
      ttsFirstByteAt: null,
      opusFirstFrameAt: null,
      fillerFiredAt: null,
      bargeInCancelled: false,
    };
    this.activeTurns.set(turnId, turn);
    return turn;
  }

  mark(turnId, stage, at) {
    if (!STAGES.has(stage)) {
      throw new RangeError(`Unknown latency stage: ${stage}`);
    }
    const turn = this.activeTurns.get(turnId);
    if (!turn) {
      throw new Error(`Unknown turn: ${turnId}`);
    }
    if (stage === "bargeInCancelled") {
      turn.bargeInCancelled = Boolean(at ?? true);
      return turn;
    }
    turn[stage] = at === undefined ? this.now() : at;
    return turn;
  }

  finishTurn(turnId) {
    const turn = this.activeTurns.get(turnId);
    if (!turn) {
      throw new Error(`Unknown turn: ${turnId}`);
    }

    const firstAudioAt = turn.opusFirstFrameAt ?? turn.ttsFirstByteAt;
    const totalLiveMs = roundMs(turn.vadEndAt, firstAudioAt);
    const fillerMasked =
      turn.fillerFiredAt !== null &&
      turn.fillerFiredAt !== undefined &&
      firstAudioAt !== null &&
      firstAudioAt !== undefined &&
      turn.fillerFiredAt < firstAudioAt;

    const record = {
      turnId,
      sttMs: roundMs(turn.vadEndAt, turn.sttDoneAt),
      gateMs: roundMs(turn.sttDoneAt, turn.gateDoneAt),
      llmFirstClauseMs: roundMs(turn.gateDoneAt, turn.llmFirstClauseAt),
      ttsFirstByteMs: roundMs(turn.llmFirstClauseAt, turn.ttsFirstByteAt),
      opusEncodeMs: roundMs(turn.ttsFirstByteAt, turn.opusFirstFrameAt),
      totalLiveMs,
      perceivedMs: fillerMasked ? roundMs(turn.vadEndAt, turn.fillerFiredAt) : totalLiveMs,
      bargeInCancelled: turn.bargeInCancelled,
    };

    this.history.push(record);
    const overflow = this.history.length - this.historyLimit;
    if (overflow > 0) this.history.splice(0, overflow);
    this.activeTurns.delete(turnId);

    return record;
  }

  getLast() {
    return this.history.length > 0 ? this.history[this.history.length - 1] : null;
  }

  getAverage() {
    if (this.history.length === 0) return null;
    const average = { turns: this.history.length };
    for (const field of METRIC_FIELDS) {
      const values = this.history
        .map((record) => record[field])
        .filter((value) => typeof value === "number" && Number.isFinite(value));
      average[field] =
        values.length === 0
          ? null
          : Math.round(values.reduce((sum, value) => sum + value, 0) / values.length);
    }
    return average;
  }

  formatSummary(record = this.getLast()) {
    if (!record) return "No turns recorded yet.";
    const ms = (value) => (typeof value === "number" ? `${value}ms` : "n/a");
    return (
      `Live ${ms(record.totalLiveMs)} (perceived ${ms(record.perceivedMs)})` +
      ` | STT ${ms(record.sttMs)}` +
      ` | Gate ${ms(record.gateMs)}` +
      ` | LLM ${ms(record.llmFirstClauseMs)}` +
      ` | TTS ${ms(record.ttsFirstByteMs)}` +
      ` | Opus ${ms(record.opusEncodeMs)}`
    );
  }
}

export const TIMING_STOP_REASONS = Object.freeze([
  "barge_in_same_speaker",
  "newer_turn",
  "deadline",
  "audio_cap",
  "frame_cap",
  "session_reset",
  "provider_error",
  "player_error",
  "normal_drain",
  "unknown",
]);

export const TIMING_GATE_REASONS = Object.freeze([
  "reply",
  "no_reply",
  "timeout",
  "error",
  "unknown",
]);

export const TIMING_PLAYER_STATES = Object.freeze(["playing", "idle", "unknown"]);

export const TIMING_IDLE_OBSERVATIONS = Object.freeze([
  "output_open_empty_queue",
  "normal",
  "unknown",
]);

const TIMING_NUMBER_FIELDS = new Set([
  "lastVoicedToEndMs",
  "endToCallbackMs",
  "sttMs",
  "gateMs",
  "firstContentTokenMs",
  "firstClauseMs",
  "firstPcmMs",
  "firstAcceptedFrameMs",
  "queueDepth",
  "sinceLastQueuedFrameMs",
]);

const TIMING_BOOLEAN_FIELDS = new Set(["writableEnded", "frameBackpressure"]);
const TIMING_ENUM_FIELDS = new Map([
  ["gateReason", TIMING_GATE_REASONS],
  ["stopReason", TIMING_STOP_REASONS],
  ["playerState", TIMING_PLAYER_STATES],
  ["idleObservation", TIMING_IDLE_OBSERVATIONS],
]);

const MAX_TIMING_MS = 600_000;

function boundedTimingNumber(value, maxMs) {
  if (value === null || value === undefined) return null;
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError("timing values must be non-negative finite numbers");
  }
  return Math.min(Math.round(value), maxMs);
}

function boundedTimingDelta(from, to, maxMs) {
  if (!Number.isFinite(from) || !Number.isFinite(to)) return null;
  return Math.min(Math.max(0, Math.round(to - from)), maxMs);
}

/**
 * Bounded, session-local timing and stop trace. Records carry only local turn
 * and resource ordinals plus bounded numbers and allowed enums; names, IDs,
 * transcript text, audio and raw provider errors never enter this model.
 */
export class TurnTimingTrace {
  constructor({ limit = 8, maxMs = MAX_TIMING_MS } = {}) {
    this.limit = Number.isInteger(limit) && limit > 0 ? Math.min(limit, 32) : 8;
    this.maxMs = Number.isInteger(maxMs) && maxMs > 0 ? maxMs : MAX_TIMING_MS;
    this.records = [];
    this.resourceSeq = 0;
  }

  startTurn(turn, { lastVoicedAt = null, endedAt = null, callbackAt = null } = {}) {
    const record = {
      turn,
      resource: null,
      lastVoicedToEndMs: null,
      endToCallbackMs: null,
      sttMs: null,
      gateMs: null,
      gateReason: null,
      firstContentTokenMs: null,
      firstClauseMs: null,
      firstPcmMs: null,
      firstAcceptedFrameMs: null,
      playerState: null,
      queueDepth: null,
      writableEnded: null,
      sinceLastQueuedFrameMs: null,
      frameBackpressure: null,
      idleObservation: null,
      stopReason: null,
    };
    if (endedAt !== null && lastVoicedAt !== null) {
      record.lastVoicedToEndMs = boundedTimingDelta(lastVoicedAt, endedAt, this.maxMs);
    }
    if (endedAt !== null && callbackAt !== null) {
      record.endToCallbackMs = boundedTimingDelta(endedAt, callbackAt, this.maxMs);
    }
    this.records.push(record);
    const overflow = this.records.length - this.limit;
    if (overflow > 0) this.records.splice(0, overflow);
    return record;
  }

  #find(turn) {
    for (let index = this.records.length - 1; index >= 0; index -= 1) {
      if (this.records[index].turn === turn) return this.records[index];
    }
    return null;
  }

  mark(turn, field, value) {
    const record = this.#find(turn);
    if (!record) return null;
    if (TIMING_NUMBER_FIELDS.has(field)) {
      record[field] = boundedTimingNumber(value, this.maxMs);
      return record;
    }
    if (TIMING_BOOLEAN_FIELDS.has(field)) {
      if (typeof value !== "boolean") throw new RangeError(`${field} must be a boolean`);
      record[field] = value;
      return record;
    }
    throw new RangeError(`Unknown timing field: ${field}`);
  }

  setReason(turn, field, value, { onlyIfUnset = false } = {}) {
    const allowed = TIMING_ENUM_FIELDS.get(field);
    if (!allowed) throw new RangeError(`Unknown timing reason field: ${field}`);
    if (!allowed.includes(value)) throw new RangeError(`Invalid ${field}: ${value}`);
    const record = this.#find(turn);
    if (!record) return null;
    if (onlyIfUnset && record[field]) return record;
    record[field] = value;
    return record;
  }

  setGateReason(turn, value, options) {
    return this.setReason(turn, "gateReason", value, options);
  }

  setStopReason(turn, value, options) {
    return this.setReason(turn, "stopReason", value, options);
  }

  notePlayer(turn, { playerState, queueDepth, writableEnded, sinceLastQueuedFrameMs, idleObservation }) {
    const record = this.#find(turn);
    if (!record) return null;
    if (playerState !== undefined) this.setReason(turn, "playerState", playerState);
    if (idleObservation !== undefined) this.setReason(turn, "idleObservation", idleObservation);
    if (queueDepth !== undefined) this.mark(turn, "queueDepth", queueDepth);
    if (writableEnded !== undefined) this.mark(turn, "writableEnded", writableEnded);
    if (sinceLastQueuedFrameMs !== undefined) {
      this.mark(turn, "sinceLastQueuedFrameMs", sinceLastQueuedFrameMs);
    }
    return record;
  }

  assignResource(turn) {
    const record = this.#find(turn);
    if (!record) return null;
    this.resourceSeq += 1;
    record.resource = this.resourceSeq;
    return this.resourceSeq;
  }

  getLatest() {
    return this.records.length > 0 ? this.records[this.records.length - 1] : null;
  }

  reset() {
    this.records = [];
    this.resourceSeq = 0;
  }

  formatSummary(record = this.getLatest()) {
    if (!record) return "No timing trace yet.";
    const ms = (value) => (typeof value === "number" ? `${value}ms` : "n/a");
    const parts = [`turn ${record.turn}`];
    if (record.resource !== null) parts.push(`resource ${record.resource}`);
    if (record.stopReason) parts.push(`stop ${record.stopReason}`);
    else parts.push("stop unknown");
    if (record.lastVoicedToEndMs !== null) parts.push(`voice to end ${ms(record.lastVoicedToEndMs)}`);
    if (record.endToCallbackMs !== null) parts.push(`end to callback ${ms(record.endToCallbackMs)}`);
    if (record.sttMs !== null) parts.push(`STT ${ms(record.sttMs)}`);
    if (record.gateMs !== null) {
      parts.push(`gate ${ms(record.gateMs)}${record.gateReason ? ` (${record.gateReason})` : ""}`);
    }
    if (record.firstContentTokenMs !== null) parts.push(`token ${ms(record.firstContentTokenMs)}`);
    if (record.firstClauseMs !== null) parts.push(`clause ${ms(record.firstClauseMs)}`);
    if (record.firstPcmMs !== null) parts.push(`PCM ${ms(record.firstPcmMs)}`);
    if (record.firstAcceptedFrameMs !== null) {
      parts.push(`frame ${ms(record.firstAcceptedFrameMs)}`);
    }
    if (record.frameBackpressure === true) parts.push("write backpressure");
    if (record.playerState) parts.push(`player ${record.playerState}`);
    if (record.queueDepth !== null) parts.push(`queue ${record.queueDepth}`);
    if (record.idleObservation === "output_open_empty_queue") {
      parts.push("idle with output open and queue empty");
    } else if (record.idleObservation === "normal") {
      parts.push("idle drained");
    }
    return parts.join(" | ");
  }
}
