import test from "node:test";
import assert from "node:assert/strict";

import {
  REQUIRED_GATEWAY_INTENTS,
  REQUIRED_VOICE_PERMISSIONS,
  createVoiceOrchestrator,
  verifyDavePrerequisites,
} from "../src/discord/session.mjs";

import {
  SLASH_COMMANDS,
  createCommandHandler,
  formatStatusReply,
} from "../src/discord/commands.mjs";

import { LatencyTracker } from "../src/types.mjs";

const PCM16K = Buffer.alloc(640, 4);
const PCM24K = Buffer.alloc(960, 6);

function tick(ms = 0) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function abortError() {
  return Object.assign(new Error("aborted"), { name: "AbortError" });
}

function makeStt(transcript = "hello there") {
  return {
    calls: 0,
    lastSignal: null,
    name: () => "mock-stt",
    close() {},
    async transcribe(pcm, { signal } = {}) {
      this.calls += 1;
      this.lastSignal = signal;
      return transcript;
    },
  };
}

function makeLlm({
  gate = { shouldReply: true },
  tokens = ["Hi ", "there."],
  blockGate = false,
  blockStream = false,
} = {}) {
  const state = {
    gateCalls: 0,
    streamCalls: 0,
    releaseGate: null,
    streamSignal: null,
    gateArgs: [],
    messagesLog: [],
  };
  let release = null;
  const gatePromise = new Promise((resolve) => {
    release = resolve;
  });

  return {
    state,
    releaseGate(value = gate) {
      release(value);
    },
    async shouldChimeIn(args) {
      state.gateCalls += 1;
      state.gateArgs.push(args);
      if (blockGate) return await gatePromise;
      return gate;
    },
    async streamReply({ messages, onToken, signal }) {
      state.streamCalls += 1;
      state.streamSignal = signal;
      state.messagesLog.push(Array.isArray(messages) ? [...messages] : []);
      if (blockStream) {
        await new Promise((resolve, reject) => {
          if (signal?.aborted) {
            reject(abortError());
            return;
          }
          signal?.addEventListener("abort", () => reject(abortError()), { once: true });
        });
        return { fullText: "" };
      }
      for (const token of tokens) {
        if (signal?.aborted) throw abortError();
        onToken?.(token);
      }
      return { fullText: tokens.join("") };
    },
  };
}

function makeTts() {
  const state = { calls: [], signals: [] };
  return {
    state,
    name: () => "mock-tts",
    close() {},
    async synthesizeClause({ text, signal, onPcmChunk }) {
      state.calls.push(text);
      state.signals.push(signal);
      if (signal?.aborted) throw abortError();
      onPcmChunk?.(PCM24K);
      return { firstByteAt: 1, byteLength: PCM24K.length };
    },
  };
}

function makeVad() {
  const captured = {};
  const pushes = [];
  const factory = (callbacks = {}) => {
    captured.callbacks = callbacks;
    return {
      pushes,
      async pushFrame(frame) {
        pushes.push(frame);
        return { ignored: false, speech: true, probability: 0.9 };
      },
      resetSpeaker() {},
      clear() {},
      setIgnoredUsers() {},
      speakers: new Map(),
    };
  };
  return { factory, captured, pushes };
}

function makeOpus() {
  return {
    backend: "opusscript",
    encode(frame) {
      return Buffer.from([0xaa, frame.length & 0xff]);
    },
    decode() {
      return Buffer.alloc(3840, 2);
    },
  };
}

function makePlayerSink() {
  const frames = [];
  return {
    frames,
    stopped: 0,
    cleared: 0,
    playing: false,
    playOpusFrame(packet) {
      frames.push(Buffer.from(packet));
    },
    stop() {
      this.stopped += 1;
      this.playing = false;
    },
    clear() {
      this.cleared += 1;
    },
    isPlaying() {
      return this.playing;
    },
  };
}

function makeReactionPool(clip = { id: "clip-1", mood: "curious_hmm", frames48kOpus: [], pcm24k: Buffer.alloc(480, 3) }) {
  return {
    picks: [],
    pick(transcript, options) {
      this.picks.push({ transcript, options });
      return clip;
    },
  };
}

function makeFillerScheduler() {
  const state = { cancelled: false, fired: false, onFire: null, delayMs: null };
  const impl = ({ delayMs, onFire }) => {
    state.delayMs = delayMs;
    state.onFire = onFire;
    return {
      cancel() {
        state.cancelled = true;
      },
      get fired() {
        return state.fired;
      },
    };
  };
  return { impl, state };
}

function makeClauseSplitter() {
  return ({ onClause }) => ({
    push(delta) {
      if (!delta) return [];
      onClause(delta);
      return [delta];
    },
    flush() {
      return [];
    },
  });
}

function buildOrchestrator(overrides = {}) {
  const stt = overrides.sttAdapter ?? makeStt();
  const llm = overrides.llmClient ?? makeLlm();
  const tts = overrides.ttsAdapter ?? makeTts();
  const vad = makeVad();
  const filler = makeFillerScheduler();
  const playerSink = makePlayerSink();
  const reactionPool = makeReactionPool();
  let clock = 0;

  const orchestrator = createVoiceOrchestrator({
    config: {
      reactions: { mode: "adaptive", fillerDelayMs: 165 },
      chime: { mode: "smart" },
      discord: { token: "super-secret-token" },
      llm: { apiKey: "super-secret-key" },
    },
    botUserId: "bot-1",
    sttAdapter: stt,
    llmClient: llm,
    ttsAdapter: tts,
    reactionPool,
    opusTranscoder: makeOpus(),
    vadFactory: vad.factory,
    createClauseSplitterImpl: makeClauseSplitter(),
    scheduleReactionFillerImpl: filler.impl,
    playerSink,
    now: () => (clock += 10),
    verifyDaveImpl: async () => ({
      nodeVersion: "22.12.0",
      daveReady: true,
      opusBackend: "opusscript",
    }),
    ...overrides,
  });

  return { orchestrator, stt, llm, tts, vad, filler, playerSink, reactionPool };
}

test("DAVE prerequisites verify the Node floor, packages, and Opus backend", async () => {
  assert.deepEqual(REQUIRED_GATEWAY_INTENTS, ["Guilds", "GuildVoiceStates"]);
  assert.deepEqual(REQUIRED_VOICE_PERMISSIONS, ["Connect", "Speak"]);

  await assert.rejects(
    () => verifyDavePrerequisites({ nodeVersion: "22.11.9", loadModule: async () => ({}) }),
    /22\.12\.0/,
  );

  await assert.rejects(
    () =>
      verifyDavePrerequisites({
        nodeVersion: "22.12.0",
        loadModule: async (id) => {
          if (id === "@snazzah/davey") throw new Error("missing");
          return {};
        },
      }),
    /@snazzah\/davey/,
  );

  const native = await verifyDavePrerequisites({
    nodeVersion: "22.12.0",
    loadModule: async () => ({}),
  });
  assert.equal(native.daveReady, true);
  assert.equal(native.opusBackend, "native");

  const js = await verifyDavePrerequisites({
    nodeVersion: "23.1.0",
    loadModule: async (id) => {
      if (id === "@discordjs/opus") throw new Error("no native");
      return {};
    },
  });
  assert.equal(js.daveReady, true);
  assert.equal(js.opusBackend, "opusscript");
});

test("orchestrator ignores bot and muted users before VAD", async () => {
  const { orchestrator, vad } = buildOrchestrator();
  await orchestrator.start();
  await orchestrator.joinChannel({ guildId: "guild-1", channelId: "voice-1" });

  await orchestrator.handleOpusPacket({ userId: "bot-1", opusPacket: Buffer.alloc(8) });
  await orchestrator.handleOpusPacket({ userId: "u1", isBot: true, opusPacket: Buffer.alloc(8) });
  assert.equal(vad.pushes.length, 0);

  orchestrator.muteUser("u2", true);
  await orchestrator.handleOpusPacket({ userId: "u2", opusPacket: Buffer.alloc(8) });
  assert.equal(vad.pushes.length, 0);

  await orchestrator.handleOpusPacket({ userId: "u3", opusPacket: Buffer.alloc(8) });
  assert.equal(vad.pushes.length, 1);
  assert.equal(vad.pushes[0].userId, "u3");
});

test("full turn: STT, filler, gate, streaming LLM, TTS, Opus frames, and latency", async () => {
  const llm = makeLlm({ blockGate: true });
  const { orchestrator, stt, tts, vad, filler, playerSink, reactionPool } = buildOrchestrator({
    llmClient: llm,
  });
  await orchestrator.start();

  const turn = vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 0,
  });

  await tick(5);
  assert.equal(stt.calls, 1);
  assert.ok(filler.state.onFire, "reaction filler should be scheduled before the gate resolves");
  assert.equal(filler.state.delayMs, 165);
  filler.state.onFire();
  await tick(5);
  assert.ok(playerSink.frames.length >= 1, "filler clip should emit Opus frames");
  assert.equal(reactionPool.picks.length, 1);

  llm.releaseGate({ shouldReply: true });
  await turn;

  assert.ok(tts.state.calls.length >= 1);
  assert.ok(playerSink.frames.length >= 2);

  const status = orchestrator.getStatus();
  assert.equal(status.active, true);
  assert.equal(status.daveReady, true);
  assert.equal(status.chimeMode, "smart");
  assert.equal(status.fillerMode, "adaptive");

  const metrics = status.latencyLast;
  assert.ok(metrics);
  assert.ok(metrics.sttMs >= 0);
  assert.ok(metrics.gateMs >= 0);
  assert.ok(metrics.llmFirstClauseMs >= 0);
  assert.ok(metrics.totalLiveMs >= metrics.perceivedMs);
  assert.ok(metrics.perceivedMs < metrics.totalLiveMs, "filler must mask dead air");
});

test("fast replies cancel the pending reaction filler", async () => {
  const { orchestrator, vad, filler, tts } = buildOrchestrator();
  await orchestrator.start();

  await vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 0,
  });

  assert.equal(filler.state.cancelled, true);
  assert.ok(tts.state.calls.length >= 1);
});

test("barge-in aborts the in-flight reply and cancels the filler", async () => {
  const llm = makeLlm({ blockStream: true });
  const { orchestrator, vad, filler, tts } = buildOrchestrator({ llmClient: llm });
  await orchestrator.start();

  const turn = vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 0,
  });

  await tick(10);
  assert.equal(llm.state.streamCalls, 1);
  assert.ok(llm.state.streamSignal);

  vad.captured.callbacks.onBargeIn({ userId: "u1", displayName: "Ana", timestampMs: 120 });
  await turn;

  assert.equal(llm.state.streamSignal.aborted, true);
  assert.equal(filler.state.cancelled, true);
  assert.equal(tts.state.calls.length, 0);
});

test("a NO chime gate suppresses the reply without error", async () => {
  const llm = makeLlm({ gate: { shouldReply: false } });
  const { orchestrator, vad, filler, tts, playerSink } = buildOrchestrator({ llmClient: llm });
  await orchestrator.start();

  await vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 0,
  });

  assert.equal(llm.state.streamCalls, 0);
  assert.equal(tts.state.calls.length, 0);
  assert.equal(playerSink.frames.length, 0);
  assert.equal(filler.state.cancelled, true);
});

test("getStatus redacts every secret", async () => {
  const { orchestrator } = buildOrchestrator();
  await orchestrator.start();
  const status = orchestrator.getStatus();
  const serialized = JSON.stringify(status);
  assert.equal(serialized.includes("super-secret-token"), false);
  assert.equal(serialized.includes("super-secret-key"), false);
  assert.equal(serialized.includes("[REDACTED]"), true);
});

test("mute and mode controls validate their inputs", async () => {
  const { orchestrator } = buildOrchestrator();
  await orchestrator.start();

  assert.throws(() => orchestrator.setChimeMode("shout"), RangeError);
  assert.throws(() => orchestrator.setFillerMode("loud"), RangeError);

  orchestrator.setChimeMode("always");
  orchestrator.setFillerMode("subtle");
  orchestrator.muteUser("u9", true);
  assert.deepEqual(orchestrator.getStatus().mutedUsers, ["u9"]);
  orchestrator.muteUser("u9", false);
  assert.deepEqual(orchestrator.getStatus().mutedUsers, []);
});

test("close is idempotent and stops further work", async () => {
  const { orchestrator, vad } = buildOrchestrator();
  await orchestrator.start();
  await orchestrator.joinChannel({ guildId: "g1", channelId: "c1" });
  assert.equal(orchestrator.getStatus().channelId, "c1");

  await orchestrator.close();
  await orchestrator.close();
  assert.equal(orchestrator.getStatus().active, false);
  assert.equal(orchestrator.getStatus().guildId, null);
  assert.ok(vad.pushes.length === 0);
});

test("slash commands expose the expected shape", () => {
  const names = SLASH_COMMANDS.map((command) => command.name);
  assert.deepEqual(names, ["join", "leave", "mode", "status", "mute-user"]);

  const mode = SLASH_COMMANDS.find((command) => command.name === "mode");
  const chime = mode.options.find((option) => option.name === "chime");
  assert.deepEqual(chime.choices, ["smart", "always", "mention"]);

  const filler = mode.options.find((option) => option.name === "filler");
  assert.deepEqual(filler.choices, ["adaptive", "subtle", "off"]);
});

test("command handler routes join, leave, mode, status, and mute-user", async () => {
  const calls = [];
  const status = {
    active: true,
    guildId: "g1",
    channelId: "c1",
    daveReady: true,
    chimeMode: "smart",
    fillerMode: "adaptive",
    mutedUsers: [],
    config: {},
    latencySummary: "No turns recorded yet.",
    latencyLast: null,
  };
  const orchestrator = {
    async joinChannel(args) {
      calls.push(["join", args]);
    },
    async leaveChannel(args) {
      calls.push(["leave", args]);
    },
    setChimeMode(mode) {
      calls.push(["chime", mode]);
    },
    setFillerMode(mode) {
      calls.push(["filler", mode]);
    },
    muteUser(userId, muted) {
      calls.push(["mute", userId, muted]);
    },
    getStatus() {
      return status;
    },
  };

  const handler = createCommandHandler({ orchestrator });
  const replies = [];

  const interaction = (commandName, options = {}, extra = {}) => ({
    commandName,
    options: {
      get(name) {
        return options[name] === undefined ? undefined : { value: options[name] };
      },
      getString(name) {
        return options[name];
      },
      getBoolean(name) {
        return options[name];
      },
    },
    member: { voice: { channelId: "c1" }, permissions: { has: () => true }, ...(extra.member ?? {}) },
    memberPermissions: { has: () => true },
    guild: { voiceAdapterCreator: "creator-1" },
    user: { id: "requester" },
    guildId: "g1",
    async reply(payload) {
      replies.push(payload);
      return payload;
    },
  });

  await handler.handleInteraction(interaction("join"));
  assert.deepEqual(calls[0], [
    "join",
    { guildId: "g1", channelId: "c1", adapterCreator: "creator-1" },
  ]);

  await handler.handleInteraction(interaction("leave"));
  assert.deepEqual(calls[1], ["leave", { guildId: "g1" }]);

  await handler.handleInteraction(interaction("mode", { chime: "always" }));
  assert.deepEqual(calls[2], ["chime", "always"]);

  await handler.handleInteraction(interaction("mode", { filler: "subtle" }));
  assert.deepEqual(calls[3], ["filler", "subtle"]);

  await handler.handleInteraction(interaction("mode", { chime: "shout" }));
  assert.match(replies.at(-1).content, /invalid|not|unsupported/i);
  assert.equal(replies.at(-1).ephemeral, true);

  await handler.handleInteraction(interaction("mute-user", { user: "u5", muted: false }));
  assert.deepEqual(calls.at(-1), ["mute", "u5", false]);

  await handler.handleInteraction(interaction("status"));
  const lastReply = replies.at(-1);
  assert.equal(lastReply.ephemeral, true);
  assert.match(lastReply.content, /smart/);
  assert.match(lastReply.content, /adaptive/);
});

test("command handler rejects missing voice, permissions, and unknown commands safely", async () => {
  const calls = [];
  const orchestrator = {
    joinChannel: async () => calls.push("join"),
    leaveChannel: async () => calls.push("leave"),
    setChimeMode: () => calls.push("chime"),
    setFillerMode: () => calls.push("filler"),
    muteUser: () => calls.push("mute"),
    getStatus: () => ({ active: false, latencySummary: "No turns recorded yet." }),
  };
  const handler = createCommandHandler({ orchestrator });
  const replies = [];

  const base = {
    commandName: "join",
    guildId: "g1",
    options: { get: () => undefined },
    member: { voice: { channelId: null }, permissions: { has: () => false } },
    user: { id: "u" },
    async reply(payload) {
      replies.push(payload);
      return payload;
    },
  };

  await handler.handleInteraction({ ...base, commandName: "join" });
  assert.match(replies.at(-1).content, /voice/i);
  assert.equal(calls.includes("join"), false);

  await handler.handleInteraction({ ...base, commandName: "mode", options: { get: () => ({ value: "off" }) } });
  assert.match(replies.at(-1).content, /permission/i);

  await handler.handleInteraction({ ...base, commandName: "explode" });
  assert.match(replies.at(-1).content, /unknown|not recognized|invalid/i);

  const throwing = createCommandHandler({
    orchestrator: {
      ...orchestrator,
      getStatus() {
        throw new Error("token=super-secret");
      },
    },
  });
  await throwing.handleInteraction({ ...base, commandName: "status" });
  assert.equal(replies.at(-1).content.includes("super-secret"), false);
});

test("formatStatusReply is concise and secret free", () => {
  const text = formatStatusReply({
    active: true,
    channelId: "c1",
    chimeMode: "smart",
    fillerMode: "adaptive",
    daveReady: true,
    latencySummary: "Live 612ms (perceived 185ms) | STT 78ms",
    config: { discord: { token: "secret" } },
  });
  assert.match(text, /smart/);
  assert.match(text, /adaptive/);
  assert.match(text, /612ms/);
  assert.equal(text.includes("secret"), false);
});

test("joinChannel validates ids, rejects a second guild, and resets session state", async () => {
  const llm = makeLlm();
  const { orchestrator, vad, playerSink } = buildOrchestrator({ llmClient: llm });
  await orchestrator.start();

  await assert.rejects(() => orchestrator.joinChannel({ channelId: "c1" }), /guildId/);
  await assert.rejects(() => orchestrator.joinChannel({ guildId: "g1" }), /channelId/);

  await orchestrator.joinChannel({ guildId: "g1", channelId: "c1" });
  await assert.rejects(
    () => orchestrator.joinChannel({ guildId: "g2", channelId: "c2" }),
    /another guild/i,
  );

  await vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 0,
  });

  await orchestrator.joinChannel({ guildId: "g1", channelId: "c2" });
  assert.equal(orchestrator.getStatus().channelId, "c2");
  assert.ok(playerSink.stopped >= 1);
  assert.ok(playerSink.cleared >= 1);
});

test("opus packets require an active matching session", async () => {
  const { orchestrator, vad } = buildOrchestrator();
  await orchestrator.start();

  const inactive = await orchestrator.handleOpusPacket({
    userId: "u1",
    opusPacket: Buffer.alloc(8),
  });
  assert.deepEqual(inactive, { ignored: true, dropped: "inactive_session" });

  await orchestrator.joinChannel({ guildId: "g1", channelId: "c1" });

  const wrongGuild = await orchestrator.handleOpusPacket({
    guildId: "g2",
    userId: "u1",
    opusPacket: Buffer.alloc(8),
  });
  assert.equal(wrongGuild.dropped, "wrong_guild");

  const wrongChannel = await orchestrator.handleOpusPacket({
    channelId: "c2",
    userId: "u1",
    opusPacket: Buffer.alloc(8),
  });
  assert.equal(wrongChannel.dropped, "wrong_guild");

  const accepted = await orchestrator.handleOpusPacket({
    guildId: "g1",
    channelId: "c1",
    userId: "u1",
    opusPacket: Buffer.alloc(8),
  });
  assert.notEqual(accepted.dropped, "wrong_guild");
  assert.equal(vad.pushes.length, 1);
});

test("leaveChannel rejects another guild and clears the binding", async () => {
  const { orchestrator } = buildOrchestrator();
  await orchestrator.start();
  await orchestrator.joinChannel({ guildId: "g1", channelId: "c1" });

  await assert.rejects(() => orchestrator.leaveChannel({ guildId: "g2" }), /another guild/i);
  await orchestrator.leaveChannel({ guildId: "g1" });
  assert.equal(orchestrator.getStatus().guildId, null);
  assert.equal(orchestrator.getStatus().channelId, null);
});

test("barge-in stops playback and cancels the filler immediately", async () => {
  const llm = makeLlm({ blockStream: true });
  const { orchestrator, vad, filler, playerSink } = buildOrchestrator({ llmClient: llm });
  await orchestrator.start();
  await orchestrator.joinChannel({ guildId: "g1", channelId: "c1" });

  const turn = vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 0,
  });

  await tick(10);
  vad.captured.callbacks.onBargeIn({ userId: "u1", timestampMs: 120 });

  assert.equal(filler.state.cancelled, true);
  assert.ok(playerSink.stopped >= 1);
  assert.ok(playerSink.cleared >= 1);

  await turn;
});

test("the first LLM clause cancels the filler before the first TTS call", async () => {
  const observations = [];
  const tts = makeTts();
  const original = tts.synthesizeClause.bind(tts);
  let fillerRef;
  tts.synthesizeClause = async (args) => {
    observations.push(fillerRef.state.cancelled);
    return original(args);
  };

  const built = buildOrchestrator({ ttsAdapter: tts });
  fillerRef = built.filler;
  await built.orchestrator.start();
  await built.orchestrator.joinChannel({ guildId: "g1", channelId: "c1" });

  await built.vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 0,
  });

  assert.ok(observations.length >= 1);
  assert.equal(observations[0], true);
});

test("Whisper hallucinations never reach the chime gate", async () => {
  const stt = makeStt("Thank you.");
  const { orchestrator, vad, llm } = buildOrchestrator({ sttAdapter: stt });
  await orchestrator.start();
  await orchestrator.joinChannel({ guildId: "g1", channelId: "c1" });

  await vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 0,
  });

  assert.equal(llm.state.gateCalls, 0);
});

test("conversation history respects maxHistoryTurns", async () => {
  const llm = makeLlm();
  const { orchestrator, vad } = buildOrchestrator({ llmClient: llm, maxHistoryTurns: 1 });
  await orchestrator.start();
  await orchestrator.joinChannel({ guildId: "g1", channelId: "c1" });

  await vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 0,
  });
  await vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 100,
  });

  assert.equal(llm.state.messagesLog.length, 2);
  assert.ok(llm.state.messagesLog[1].length <= 2);
});

test("chime gate receives mode, speaker, timings, and lastBotReplyAt", async () => {
  const llm = makeLlm();
  const { orchestrator, vad } = buildOrchestrator({ llmClient: llm });
  await orchestrator.start();
  await orchestrator.joinChannel({ guildId: "g1", channelId: "c1" });
  orchestrator.setChimeMode("always");

  await vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 0,
  });
  await vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 100,
  });

  assert.equal(llm.state.gateArgs[0].chimeMode, "always");
  assert.equal(llm.state.gateArgs[0].speakerName, "Ana");
  assert.equal(llm.state.gateArgs[0].lastBotReplyAt, 0);
  assert.ok(llm.state.gateArgs[1].lastBotReplyAt > 0);
  assert.ok("followUpWindowMs" in llm.state.gateArgs[0]);
  assert.ok("gateTimeoutMs" in llm.state.gateArgs[0]);
  assert.ok(llm.state.gateArgs[0].signal instanceof AbortSignal);
});

test("createVoiceOrchestrator constructs without an opus transcoder", () => {
  assert.doesNotThrow(() =>
    createVoiceOrchestrator({
      config: {},
      reactionPool: makeReactionPool(),
      latencyTracker: new LatencyTracker(),
      llmClient: makeLlm(),
      ttsAdapter: makeTts(),
      sttAdapter: makeStt(),
    }),
  );
});

test("joinChannel forwards the voice adapter and voice deps", async () => {
  const voiceCalls = [];
  const { orchestrator } = buildOrchestrator({
    voiceDeps: {
      joinVoiceChannel(options) {
        voiceCalls.push(options);
        return { destroy() {} };
      },
    },
  });
  await orchestrator.start();
  await orchestrator.joinChannel({
    guildId: "g1",
    channelId: "c1",
    adapterCreator: "creator-1",
  });

  assert.equal(voiceCalls.length, 1);
  assert.equal(voiceCalls[0].guildId, "g1");
  assert.equal(voiceCalls[0].channelId, "c1");
  assert.equal(voiceCalls[0].adapterCreator, "creator-1");
});

test("muting the active speaker aborts the in-flight turn", async () => {
  const llm = makeLlm({ blockStream: true });
  const { orchestrator, vad, filler, playerSink } = buildOrchestrator({ llmClient: llm });
  await orchestrator.start();
  await orchestrator.joinChannel({ guildId: "g1", channelId: "c1" });

  const turn = vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 0,
  });

  await tick(10);
  orchestrator.muteUser("u1", true);
  await turn;

  assert.equal(llm.state.streamSignal.aborted, true);
  assert.equal(filler.state.cancelled, true);
  assert.ok(playerSink.stopped >= 1);
});

test("a TTS rejection during an active stream does not leak", async () => {
  const tts = makeTts();
  tts.synthesizeClause = async () => {
    throw new Error("tts down");
  };
  const { orchestrator, vad } = buildOrchestrator({ ttsAdapter: tts });
  await orchestrator.start();
  await orchestrator.joinChannel({ guildId: "g1", channelId: "c1" });

  await vad.captured.callbacks.onSpeechEnd({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: PCM16K,
    durationMs: 300,
    endedAt: 0,
  });

  assert.equal(orchestrator.getStatus().active, true);
});

test("commands reject a missing guild and any cross-guild request", async () => {
  const replies = [];
  const handler = createCommandHandler({
    orchestrator: {
      getStatus: () => ({ guildId: "g1", active: true, latencySummary: "No turns recorded yet." }),
      joinChannel: async () => {},
      leaveChannel: async () => {},
      setChimeMode: () => {},
      setFillerMode: () => {},
      muteUser: () => {},
    },
  });

  const base = {
    commandName: "status",
    options: { get: () => undefined },
    member: { voice: { channelId: "c1" } },
    memberPermissions: { has: () => true },
    async reply(payload) {
      replies.push(payload);
      return payload;
    },
  };

  await handler.handleInteraction({ ...base, guildId: undefined });
  assert.match(replies.at(-1).content, /server|guild/i);

  await handler.handleInteraction({ ...base, guildId: "g2" });
  assert.match(replies.at(-1).content, /server|guild/i);
});

test("mute-user denies when memberPermissions are absent or false", async () => {
  const calls = [];
  const handler = createCommandHandler({
    orchestrator: {
      getStatus: () => ({ guildId: null }),
      muteUser: (userId, muted) => calls.push([userId, muted]),
    },
  });
  const replies = [];

  const base = {
    commandName: "mute-user",
    guildId: "g1",
    options: { get: (name) => ({ value: name === "user" ? "u1" : true }) },
    member: { voice: { channelId: "c1" } },
    async reply(payload) {
      replies.push(payload);
      return payload;
    },
  };

  await handler.handleInteraction({ ...base });
  assert.match(replies.at(-1).content, /permission/i);
  assert.equal(calls.length, 0);

  await handler.handleInteraction({ ...base, memberPermissions: { has: () => false } });
  assert.match(replies.at(-1).content, /permission/i);
  assert.equal(calls.length, 0);
});
