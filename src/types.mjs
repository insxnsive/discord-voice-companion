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
