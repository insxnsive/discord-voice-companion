import test from "node:test";
import assert from "node:assert/strict";

import * as typesModule from "../src/types.mjs";
import {
  CHIME_MODES,
  FILLER_MODES,
  LatencyTracker,
  PROVIDERS,
  REACTION_MOODS,
  assertAdapterContract,
} from "../src/types.mjs";

test("provider and mode tables expose every supported value", () => {
  assert.deepEqual(PROVIDERS.llm, [
    "opencode",
    "deepseek",
    "openai",
    "openrouter",
    "groq",
    "ollama",
    "custom",
  ]);
  assert.deepEqual(PROVIDERS.stt, [
    "deepgram",
    "elevenlabs",
    "openai",
    "groq",
    "faster-whisper",
  ]);
  assert.deepEqual(PROVIDERS.tts, ["elevenlabs", "openai", "openai-compatible"]);
  assert.deepEqual(REACTION_MOODS, [
    "subtle_breath",
    "curious_hmm",
    "light_chuckle",
    "playful_scoff",
  ]);
  assert.deepEqual(FILLER_MODES, ["adaptive", "subtle", "off"]);
  assert.deepEqual(CHIME_MODES, ["smart", "always", "mention"]);
});

test("provider and mode tables are frozen", () => {
  assert.ok(Object.isFrozen(PROVIDERS));
  assert.ok(Object.isFrozen(PROVIDERS.llm));
  assert.ok(Object.isFrozen(PROVIDERS.stt));
  assert.ok(Object.isFrozen(PROVIDERS.tts));
  assert.ok(Object.isFrozen(REACTION_MOODS));
  assert.ok(Object.isFrozen(FILLER_MODES));
  assert.ok(Object.isFrozen(CHIME_MODES));
});

test("assertAdapterContract accepts a complete adapter", () => {
  const adapter = { connect() {}, speak() {}, close() {} };
  assert.doesNotThrow(() => assertAdapterContract(adapter, ["connect", "speak", "close"], "tts"));
});

test("assertAdapterContract names the label and every missing method", () => {
  assert.throws(
    () => assertAdapterContract({ connect() {} }, ["connect", "speak", "close"], "voice adapter"),
    (error) => {
      assert.ok(error instanceof TypeError);
      assert.match(error.message, /voice adapter/);
      assert.match(error.message, /speak/);
      assert.match(error.message, /close/);
      return true;
    },
  );
});

test("assertAdapterContract rejects a null adapter", () => {
  assert.throws(
    () => assertAdapterContract(null, ["speak", "close"], "voice adapter"),
    (error) => {
      assert.ok(error instanceof TypeError);
      assert.match(error.message, /voice adapter/);
      assert.match(error.message, /speak/);
      assert.match(error.message, /close/);
      return true;
    },
  );
});

test("LatencyTracker computes exact stage milliseconds without a filler", () => {
  const tracker = new LatencyTracker();
  tracker.startTurn("turn-1", 1000);
  tracker.mark("turn-1", "sttDoneAt", 1078);
  tracker.mark("turn-1", "gateDoneAt", 1120);
  tracker.mark("turn-1", "llmFirstClauseAt", 1315);
  tracker.mark("turn-1", "ttsFirstByteAt", 1610);
  tracker.mark("turn-1", "opusFirstFrameAt", 1612);

  const metrics = tracker.finishTurn("turn-1");

  assert.equal(metrics.sttMs, 78);
  assert.equal(metrics.gateMs, 42);
  assert.equal(metrics.llmFirstClauseMs, 195);
  assert.equal(metrics.ttsFirstByteMs, 295);
  assert.equal(metrics.opusEncodeMs, 2);
  assert.equal(metrics.totalLiveMs, 612);
  assert.equal(metrics.perceivedMs, 612);
});

test("LatencyTracker reports perceived latency when a filler masks dead air", () => {
  const tracker = new LatencyTracker();
  tracker.startTurn("turn-2", 2000);
  tracker.mark("turn-2", "sttDoneAt", 2100);
  tracker.mark("turn-2", "gateDoneAt", 2140);
  tracker.mark("turn-2", "llmFirstClauseAt", 2300);
  tracker.mark("turn-2", "fillerFiredAt", 2185);
  tracker.mark("turn-2", "ttsFirstByteAt", 2600);
  tracker.mark("turn-2", "opusFirstFrameAt", 2602);

  const metrics = tracker.finishTurn("turn-2");

  assert.equal(metrics.sttMs, 100);
  assert.equal(metrics.gateMs, 40);
  assert.equal(metrics.llmFirstClauseMs, 160);
  assert.equal(metrics.ttsFirstByteMs, 300);
  assert.equal(metrics.opusEncodeMs, 2);
  assert.equal(metrics.totalLiveMs, 602);
  assert.equal(metrics.perceivedMs, 185);

  assert.equal(
    tracker.formatSummary(),
    "Live 602ms (perceived 185ms) | STT 100ms | Gate 40ms | LLM 160ms | TTS 300ms | Opus 2ms",
  );
});

test("LatencyTracker keeps a bounded history of the newest turns", () => {
  const tracker = new LatencyTracker({ historyLimit: 2 });
  const turns = [
    ["turn-a", 0, 100],
    ["turn-b", 1000, 1200],
    ["turn-c", 2000, 2300],
  ];

  for (const [turnId, start, opusFrame] of turns) {
    tracker.startTurn(turnId, start);
    tracker.mark(turnId, "sttDoneAt", start + 10);
    tracker.mark(turnId, "gateDoneAt", start + 20);
    tracker.mark(turnId, "llmFirstClauseAt", start + 50);
    tracker.mark(turnId, "ttsFirstByteAt", start + 90);
    tracker.mark(turnId, "opusFirstFrameAt", opusFrame);
    tracker.finishTurn(turnId);
  }

  assert.equal(tracker.history.length, 2);
  assert.equal(tracker.getLast().turnId, "turn-c");
  assert.equal(tracker.getLast().totalLiveMs, 300);

  const average = tracker.getAverage();
  assert.equal(average.turns, 2);
  assert.equal(average.totalLiveMs, 250);
  assert.equal(average.perceivedMs, 250);
});

test("LatencyTracker reports empty state for averages and summaries", () => {
  const tracker = new LatencyTracker();

  assert.equal(tracker.getLast(), null);
  assert.equal(tracker.getAverage(), null);
  assert.equal(tracker.formatSummary(), "No turns recorded yet.");
});

test("LatencyTracker marks unknown stages as a RangeError and leaves gaps null", () => {
  const tracker = new LatencyTracker();
  tracker.startTurn("turn-d", 3000);

  assert.throws(() => tracker.mark("turn-d", "sttDone", 3100), RangeError);
  assert.throws(() => tracker.mark("turn-d", "nope", 3100), RangeError);

  const metrics = tracker.finishTurn("turn-d");
  assert.equal(metrics.sttMs, null);
  assert.equal(metrics.totalLiveMs, null);
});

test("timing trace stays bounded, clamps numbers, and stores only allowed fields", () => {
  const trace = new typesModule.TurnTimingTrace({ limit: 2 });
  trace.startTurn(1, { lastVoicedAt: 100, endedAt: 340, callbackAt: 342 });
  trace.mark(1, "sttMs", 120);
  trace.mark(1, "gateMs", 9_999_999);
  trace.mark(1, "queueDepth", 0);
  trace.mark(1, "writableEnded", false);
  trace.setGateReason(1, "gate_yes");
  trace.notePlayer(1, {
    playerState: "idle",
    queueDepth: 0,
    writableEnded: false,
    sinceLastQueuedFrameMs: 12,
    idleObservation: "output_open_empty_queue",
  });
  trace.assignResource(1);
  trace.setStopReason(1, "normal_drain");

  const first = trace.getLatest();
  assert.equal(first.turn, 1);
  assert.equal(first.resource, 1);
  assert.equal(first.lastVoicedToEndMs, 240);
  assert.equal(first.endToCallbackMs, 2);
  assert.equal(first.sttMs, 120);
  assert.equal(first.gateMs, 600_000);
  assert.equal(first.gateReason, "gate_yes");
  assert.equal(first.playerState, "idle");
  assert.equal(first.idleObservation, "output_open_empty_queue");
  assert.equal(first.stopReason, "normal_drain");

  assert.throws(() => trace.mark(1, "nickname", "Ana"), RangeError);
  assert.throws(() => trace.mark(1, "sttMs", -1), RangeError);
  assert.throws(() => trace.setStopReason(1, "because"), RangeError);

  trace.startTurn(2);
  trace.startTurn(3);
  assert.equal(trace.records.length, 2);
  assert.equal(trace.getLatest().turn, 3);
  assert.match(trace.formatSummary(), /turn 3/);
  assert.equal(trace.formatSummary().includes("Ana"), false);
  assert.equal(trace.formatSummary().includes("guild"), false);

  const keys = Object.keys(first).sort();
  assert.deepEqual(keys, [
    "endToCallbackMs",
    "fillerScheduleMs",
    "firstAcceptedFrameMs",
    "firstClauseMs",
    "firstContentTokenMs",
    "firstPcmMs",
    "frameBackpressure",
    "gateMs",
    "gateReason",
    "idleAtMs",
    "idleObservation",
    "lastVoicedToEndMs",
    "playerState",
    "playingMs",
    "queueDepth",
    "resource",
    "sinceLastQueuedFrameMs",
    "stopReason",
    "sttMs",
    "transcoderMs",
    "turn",
    "writableEnded",
  ]);
  trace.reset();
  assert.equal(trace.getLatest(), null);
  assert.equal(trace.formatSummary(), "No timing trace yet.");
});

test("timing trace snapshots cannot mutate the stored record", () => {
  const trace = new typesModule.TurnTimingTrace();
  trace.startTurn(1);
  const snapshot = trace.getLatest();
  snapshot.turn = 99;
  snapshot.sttMs = 123;
  snapshot.stopReason = "normal_drain";

  const next = trace.getLatest();
  assert.equal(next.turn, 1);
  assert.equal(next.sttMs, null);
  assert.equal(next.stopReason, null);
});
