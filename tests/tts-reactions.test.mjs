import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";

import { loadConfig } from "../src/config/config.mjs";
import {
  createElevenLabsHttpStreamAdapter,
  createElevenLabsWsAdapter,
  createOpenAiTtsAdapter,
  createTtsAdapter,
  stripWavHeaderIfPresent,
} from "../src/tts/adapters.mjs";
import {
  REACTION_CLIP_PROMPTS,
  ReactionPool,
  classifyTranscriptMood,
  scheduleReactionFiller,
} from "../src/tts/reactions.mjs";

function makeMemoryFs(files = {}) {
  return {
    existsSync(target) {
      return Object.prototype.hasOwnProperty.call(files, target);
    },
    readFileSync(target) {
      if (!Object.prototype.hasOwnProperty.call(files, target)) {
        const err = new Error(`ENOENT: ${target}`);
        err.code = "ENOENT";
        throw err;
      }
      return files[target];
    },
  };
}

function makeFakeWsFactory({ chunks = [], errorMsg = null } = {}) {
  const sentMessages = [];
  const urls = [];
  let closed = false;

  class FakeWebSocket {
    constructor(url, options) {
      this.url = url;
      this.options = options;
      this.readyState = 0;
      this._listeners = new Map();
      urls.push({ url, options });
      queueMicrotask(() => {
        if (closed) return;
        this.readyState = 1;
        this._emit("open");
      });
    }

    on(event, fn) {
      if (!this._listeners.has(event)) this._listeners.set(event, []);
      this._listeners.get(event).push(fn);
    }

    removeListener(event, fn) {
      const list = this._listeners.get(event);
      if (!list) return;
      const idx = list.indexOf(fn);
      if (idx >= 0) list.splice(idx, 1);
    }

    _emit(event, ...args) {
      for (const fn of [...(this._listeners.get(event) ?? [])]) {
        fn(...args);
      }
    }

    send(payload) {
      const parsed = JSON.parse(payload);
      sentMessages.push(parsed);
      if (parsed.text === "") {
        queueMicrotask(() => {
          if (closed) return;
          if (errorMsg) {
            this._emit("message", JSON.stringify({ error: errorMsg }));
            return;
          }
          for (const chunk of chunks) {
            if (closed) return;
            this._emit("message", JSON.stringify({ audio: chunk.toString("base64") }));
          }
          if (!closed) {
            this._emit("message", JSON.stringify({ isFinal: true }));
          }
        });
      }
    }

    close() {
      if (closed) return;
      closed = true;
      this.readyState = 3;
      this._emit("close", 1000);
    }
  }

  return { FakeWebSocket, sentMessages, urls };
}

test("createTtsAdapter selects WS flash by default, HTTP v4_turbo when expressiveTags=true, and rejects insecure baseUrl", () => {
  const { FakeWebSocket } = makeFakeWsFactory();
  const fastCfg = loadConfig({
    env: {
      TTS_PROVIDER: "elevenlabs",
      ELEVENLABS_API_KEY: "el-key",
      ELEVENLABS_VOICE_ID: "voice-1",
      ELEVENLABS_EXPRESSIVE_TAGS: "false",
    },
    fsImpl: makeMemoryFs(),
  });
  const wsAdapter = createTtsAdapter(fastCfg, { WebSocketImpl: FakeWebSocket });
  assert.equal(wsAdapter.name(), "elevenlabs-ws");

  const expressiveCfg = loadConfig({
    env: {
      TTS_PROVIDER: "elevenlabs",
      ELEVENLABS_API_KEY: "el-key",
      ELEVENLABS_VOICE_ID: "voice-1",
      ELEVENLABS_EXPRESSIVE_TAGS: "true",
    },
    fsImpl: makeMemoryFs(),
  });
  const httpAdapter = createTtsAdapter(expressiveCfg);
  assert.equal(httpAdapter.name(), "elevenlabs-http-stream");

  assert.throws(
    () =>
      createTtsAdapter({
        tts: {
          provider: "openai-compatible",
          apiKey: "k",
          baseUrl: "http://remote.example.com/v1",
          voiceId: "Capri",
          model: "capri-tts",
          expressiveTags: false,
        },
      }),
    RangeError,
  );
});

test("createElevenLabsWsAdapter sends BOS/clause/EOS, decodes PCM, enforces maxAudioBytes, and aborts cleanly", async () => {
  const pcm1 = Buffer.from([1, 2, 3, 4]);
  const pcm2 = Buffer.from([5, 6, 7, 8]);
  const { FakeWebSocket, sentMessages, urls } = makeFakeWsFactory({ chunks: [pcm1, pcm2] });

  const adapter = createElevenLabsWsAdapter({
    apiKey: "el-key",
    voiceId: "voice-capri",
    modelId: "eleven_flash_v2_5",
    WebSocketImpl: FakeWebSocket,
  });
  await adapter.warmup();

  const received = [];
  const res = await adapter.synthesizeClause({
    text: "Hey Blue",
    onPcmChunk: (chunk) => received.push(chunk),
  });

  assert.match(urls[0].url, /voice-capri\/stream-input\?model_id=eleven_flash_v2_5&output_format=pcm_24000/);
  assert.equal(urls[0].options.headers["xi-api-key"], "el-key");
  assert.equal(sentMessages[0].text, " ");
  assert.equal(sentMessages[1].text, "Hey Blue ");
  assert.equal(sentMessages[1].try_trigger_generation, true);
  assert.equal(sentMessages[2].text, "");
  assert.equal(Buffer.concat(received).equals(Buffer.concat([pcm1, pcm2])), true);
  assert.equal(res.byteLength, 8);
  assert.equal(typeof res.firstByteAt, "number");
  adapter.close();

  const bigFactory = makeFakeWsFactory({ chunks: [Buffer.alloc(1024)] });
  const capped = createElevenLabsWsAdapter({
    apiKey: "el-key",
    voiceId: "voice-capri",
    maxAudioBytes: 512,
    WebSocketImpl: bigFactory.FakeWebSocket,
  });
  await assert.rejects(() => capped.synthesizeClause({ text: "Too big" }), /size limit/i);

  const ac = new AbortController();
  const slowFactory = makeFakeWsFactory({ chunks: [Buffer.alloc(16)] });
  const abortAdapter = createElevenLabsWsAdapter({
    apiKey: "el-key",
    voiceId: "voice-capri",
    WebSocketImpl: slowFactory.FakeWebSocket,
  });
  const pending = abortAdapter.synthesizeClause({ text: "Abort me", signal: ac.signal });
  ac.abort();
  await assert.rejects(pending, { name: "AbortError" });
});

test("createElevenLabsHttpStreamAdapter streams chunks, sets redirect error, hides error body, and enforces maxAudioBytes", async () => {
  const chunkA = Buffer.from([10, 20, 30, 40]);
  const chunkB = Buffer.from([50, 60, 70, 80]);
  let seenInit = null;

  const adapter = createElevenLabsHttpStreamAdapter({
    apiKey: "el-key",
    voiceId: "voice-capri",
    modelId: "eleven_v4_turbo",
    fetchImpl: async (_url, init) => {
      seenInit = init;
      const stream = new ReadableStream({
        start(c) {
          c.enqueue(chunkA);
          c.enqueue(chunkB);
          c.close();
        },
      });
      return new Response(stream, { status: 200 });
    },
  });

  const got = [];
  const res = await adapter.synthesizeClause({
    text: "[laughs] Hey!",
    onPcmChunk: (chunk) => got.push(chunk),
  });
  assert.equal(seenInit.redirect, "error");
  assert.equal(JSON.parse(seenInit.body).voice_settings.style, undefined);
  assert.equal(got.length, 2);
  assert.equal(res.byteLength, 8);

  let cancelledErrBody = false;
  const errAdapter = createElevenLabsHttpStreamAdapter({
    apiKey: "el-key",
    voiceId: "voice-capri",
    fetchImpl: async () => {
      const stream = new ReadableStream({
        start(c) {
          c.enqueue(new TextEncoder().encode("secret-provider-error"));
        },
        cancel() {
          cancelledErrBody = true;
        },
      });
      return new Response(stream, { status: 429 });
    },
  });
  await assert.rejects(
    () => errAdapter.synthesizeClause({ text: "test" }),
    (err) => {
      assert.equal(err.message, "TTS request failed (HTTP 429)");
      assert.doesNotMatch(err.message, /secret-provider-error/);
      return true;
    },
  );
  assert.equal(cancelledErrBody, true);

  let cancelledBig = false;
  const bigAdapter = createElevenLabsHttpStreamAdapter({
    apiKey: "el-key",
    voiceId: "voice-capri",
    maxAudioBytes: 64,
    fetchImpl: async () => {
      const stream = new ReadableStream({
        pull(c) {
          c.enqueue(Buffer.alloc(128));
        },
        cancel() {
          cancelledBig = true;
        },
      });
      return new Response(stream, { status: 200 });
    },
  });
  await assert.rejects(() => bigAdapter.synthesizeClause({ text: "huge" }), /size limit/i);
  assert.equal(cancelledBig, true);
});

test("createOpenAiTtsAdapter strips RIFF/WAVE header via stripWavHeaderIfPresent", async () => {
  const pcm = Buffer.from([1, 2, 3, 4, 5, 6]);
  const wavHeader = Buffer.alloc(44);
  wavHeader.write("RIFF", 0, "ascii");
  wavHeader.writeUInt32LE(36 + pcm.length, 4);
  wavHeader.write("WAVE", 8, "ascii");
  wavHeader.write("fmt ", 12, "ascii");
  wavHeader.writeUInt32LE(16, 16);
  wavHeader.writeUInt16LE(1, 20);
  wavHeader.writeUInt16LE(1, 22);
  wavHeader.writeUInt32LE(24000, 24);
  wavHeader.writeUInt32LE(48000, 28);
  wavHeader.writeUInt16LE(2, 32);
  wavHeader.writeUInt16LE(16, 34);
  wavHeader.write("data", 36, "ascii");
  wavHeader.writeUInt32LE(pcm.length, 40);
  const fullWav = Buffer.concat([wavHeader, pcm]);

  assert.equal(stripWavHeaderIfPresent(fullWav).equals(pcm), true);
  assert.equal(stripWavHeaderIfPresent(pcm).equals(pcm), true);

  const adapter = createOpenAiTtsAdapter({
    apiKey: "oa-key",
    baseUrl: "http://127.0.0.1:18765/v1",
    model: "capri-tts",
    voice: "Capri",
    fetchImpl: async () =>
      new Response(new ReadableStream({
        start(c) {
          c.enqueue(fullWav);
          c.close();
        },
      }), { status: 200 }),
  });

  const got = [];
  const res = await adapter.synthesizeClause({
    text: "Hello",
    onPcmChunk: (chunk) => got.push(chunk),
  });
  assert.equal(Buffer.concat(got).equals(pcm), true);
  assert.equal(res.byteLength, pcm.length);
});

test("ReactionPool enforces 10-turn no-repeat, modes, and safe directory loading", () => {
  assert.equal(REACTION_CLIP_PROMPTS.length, 14);
  assert.equal(classifyTranscriptMood("Wait, why did that happen?"), "curious_hmm");
  assert.equal(classifyTranscriptMood("Haha that was crazy!"), "light_chuckle");
  assert.equal(classifyTranscriptMood("Bruh seriously"), "playful_scoff");
  assert.equal(classifyTranscriptMood("I walked to the kitchen."), "subtle_breath");

  const pool = new ReactionPool({ cooldownTurns: 10 });
  for (const prompt of REACTION_CLIP_PROMPTS) {
    pool.registerClip({ id: prompt.id, mood: prompt.mood, pcm24k: Buffer.alloc(16) });
  }
  assert.equal(pool.size, 14);
  assert.equal(pool.pick("Hello?", { mode: "off" }), null);

  const subtlePick = pool.pick("Haha lol!", { mode: "subtle" });
  assert.equal(subtlePick.mood, "subtle_breath");

  const seen10 = [];
  for (let i = 0; i < 10; i++) {
    const picked = pool.pick("Why?", { mode: "adaptive", rng: () => 0 });
    assert.ok(picked);
    assert.equal(seen10.includes(picked.id), false);
    seen10.push(picked.id);
  }

  const fakeDir = "reactions-cache";
  const entries = [
    "subtle_breath_01.pcm",
    "curious_hmm_01.wav",
    "light_chuckle_01_symlink.pcm",
    "../evil_clip.pcm",
    "light_chuckle_huge.pcm",
  ];
  const wav = Buffer.concat([Buffer.from("RIFF\x2c\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\xc0\x5d\x00\x00\x80\xbb\x00\x00\x02\x00\x10\x00data\x04\x00\x00\x00", "latin1"), Buffer.from([9, 8, 7, 6])]);
  const dirFs = {
    existsSync: (p) => p === fakeDir,
    readdirSync: () => entries,
    lstatSync: (p) => {
      const name = path.basename(p);
      if (name === "symlink_clip.pcm" || name === "light_chuckle_01_symlink.pcm") {
        return { isFile: () => false, size: 10 };
      }
      if (name === "light_chuckle_huge.pcm") {
        return { isFile: () => true, size: 999999 };
      }
      return { isFile: () => true, size: 64 };
    },
    readFileSync: (p) => (p.endsWith(".wav") ? wav : Buffer.from([1, 2, 3, 4])),
  };
  const diskPool = new ReactionPool();
  const loaded = diskPool.loadFromDirectory(fakeDir, { fsImpl: dirFs, maxFileBytes: 1024 });
  assert.equal(loaded, 2);
});

test("scheduleReactionFiller fires at 165ms when slow and cancels cleanly before firing", () => {
  let scheduledFn = null;
  let scheduledDelay = null;
  let clearedId = null;
  let fireCount = 0;

  const handle1 = scheduleReactionFiller({
    delayMs: 165,
    onFire: () => {
      fireCount += 1;
    },
    setTimer: (fn, ms) => {
      scheduledFn = fn;
      scheduledDelay = ms;
      return 101;
    },
    clearTimer: (id) => {
      clearedId = id;
    },
  });
  assert.equal(scheduledDelay, 165);
  assert.equal(handle1.fired, false);
  scheduledFn();
  assert.equal(fireCount, 1);
  assert.equal(handle1.fired, true);
  handle1.cancel();
  assert.equal(clearedId, null);

  let cancelledFireCount = 0;
  let cancelledFn = null;
  const handle2 = scheduleReactionFiller({
    delayMs: 165,
    onFire: () => {
      cancelledFireCount += 1;
    },
    setTimer: (fn) => {
      cancelledFn = fn;
      return 202;
    },
    clearTimer: (id) => {
      clearedId = id;
    },
  });
  handle2.cancel();
  assert.equal(clearedId, 202);
  cancelledFn();
  assert.equal(cancelledFireCount, 0);
  assert.equal(handle2.fired, false);
});
