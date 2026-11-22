import test from "node:test";
import assert from "node:assert/strict";
import { getEventListeners } from "node:events";
import path from "node:path";

import { loadConfig } from "../src/config/config.mjs";
import { createCompanionApp } from "../src/index.mjs";
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
  const wav = Buffer.concat([Buffer.from("RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\xc0\x5d\x00\x00\x80\xbb\x00\x00\x02\x00\x10\x00data\x04\x00\x00\x00", "latin1"), Buffer.from([9, 8, 7, 6])]);
  const dirFs = {
    existsSync: (p) => p === fakeDir,
    readdirSync: () => entries,
    lstatSync: (p) => {
      if (p === fakeDir) {
        return { isSymbolicLink: () => false, isDirectory: () => true, isFile: () => false, size: 0 };
      }
      const name = path.basename(p);
      if (name === "symlink_clip.pcm" || name === "light_chuckle_01_symlink.pcm") {
        return { isSymbolicLink: () => true, isDirectory: () => false, isFile: () => false, size: 10 };
      }
      if (name === "light_chuckle_huge.pcm") {
        return { isSymbolicLink: () => false, isDirectory: () => false, isFile: () => true, size: 999999 };
      }
      return { isSymbolicLink: () => false, isDirectory: () => false, isFile: () => true, size: 64 };
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

test("WS abort during CONNECTING absorbs async error after close without unhandled crash", async () => {
  const { EventEmitter } = await import("node:events");
  let constructed = false;
  class ConnectingAbortWs extends EventEmitter {
    constructor() {
      super();
      constructed = true;
      this.readyState = 0;
    }
    send() {}
    close() {
      this.readyState = 2;
      queueMicrotask(() => {
        this.emit("error", new Error("WebSocket was closed before the connection was established"));
        this.readyState = 3;
        this.emit("close", 1006);
      });
    }
  }

  const adapter = createElevenLabsWsAdapter({
    apiKey: "el-test-key",
    voiceId: "voice_capri_01",
    WebSocketImpl: ConnectingAbortWs,
  });

  const controller = new AbortController();
  const promise = adapter.synthesizeClause({
    text: "Hold up",
    signal: controller.signal,
    onPcmChunk: () => {},
  });
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(constructed, true);
  controller.abort();

  await assert.rejects(promise, (err) => err?.name === "AbortError");
  await new Promise((r) => setTimeout(r, 10));
});

test("WS forwards maxPayload, enforces pre-decode size caps, validates maxAudioBytes, and sanitizes errors", async () => {
  const { EventEmitter } = await import("node:events");
  let capturedOptions = null;

  class InspectingWs extends EventEmitter {
    constructor(_url, options) {
      super();
      capturedOptions = options;
      this.readyState = 0;
      queueMicrotask(() => {
        this.readyState = 1;
        this.emit("open");
      });
    }
    send(payload) {
      const parsed = JSON.parse(payload);
      if (parsed.text === " ") return;
      if (parsed.text.startsWith("secret-error")) {
        queueMicrotask(() => {
          this.emit("message", JSON.stringify({ error: "secret-api-key-leak" }));
        });
      } else if (parsed.text.startsWith("huge-raw")) {
        queueMicrotask(() => {
          const pad = "A".repeat(4 * 1024 * 1024 + 64);
          this.emit("message", JSON.stringify({ audio: pad }));
        });
      } else if (parsed.text.startsWith("huge-audio-field")) {
        queueMicrotask(() => {
          const pad = "A".repeat(200);
          this.emit("message", JSON.stringify({ audio: pad }));
        });
      } else if (parsed.text === "") {
        queueMicrotask(() => {
          this.emit("message", JSON.stringify({ isFinal: true }));
        });
      }
    }
    close() {
      this.readyState = 3;
      this.emit("close", 1000);
    }
  }

  assert.throws(
    () =>
      createElevenLabsWsAdapter({
        apiKey: "el-test-key",
        voiceId: "voice_capri_01",
        maxAudioBytes: 32 * 1024 * 1024,
        WebSocketImpl: InspectingWs,
      }),
    RangeError,
  );

  const errAdapter = createElevenLabsWsAdapter({
    apiKey: "el-test-key",
    voiceId: "voice_capri_01",
    maxAudioBytes: 512,
    WebSocketImpl: InspectingWs,
  });

  await assert.rejects(
    () => errAdapter.synthesizeClause({ text: "secret-error", onPcmChunk: () => {} }),
    (err) => err instanceof Error && !err.message.includes("secret-api-key-leak"),
  );
  assert.equal(capturedOptions.maxPayload, 512);

  await assert.rejects(
    () => errAdapter.synthesizeClause({ text: "huge-raw", onPcmChunk: () => {} }),
    /limit|size|payload/i,
  );

  const tightAdapter = createElevenLabsWsAdapter({
    apiKey: "el-test-key",
    voiceId: "voice_capri_01",
    maxAudioBytes: 64,
    WebSocketImpl: InspectingWs,
  });
  await assert.rejects(
    () => tightAdapter.synthesizeClause({ text: "huge-audio-field", onPcmChunk: () => {} }),
    /limit|size/i,
  );
});

test("WS partial PCM before socket close or adapter.close() rejects without isFinal", async () => {
  const { EventEmitter } = await import("node:events");
  const partialBase64 = Buffer.from([1, 2, 3, 4]).toString("base64");

  class PrematureCloseWs extends EventEmitter {
    constructor() {
      super();
      this.readyState = 0;
      queueMicrotask(() => {
        this.readyState = 1;
        this.emit("open");
      });
    }
    send(payload) {
      const parsed = JSON.parse(payload);
      if (parsed.text === "") {
        queueMicrotask(() => {
          this.emit("message", JSON.stringify({ audio: partialBase64 }));
          queueMicrotask(() => {
            this.readyState = 3;
            this.emit("close", 1000);
          });
        });
      }
    }
    close() {
      this.readyState = 3;
    }
  }

  const wsAdapter1 = createElevenLabsWsAdapter({
    apiKey: "el-test-key",
    voiceId: "voice_capri_01",
    WebSocketImpl: PrematureCloseWs,
  });
  await assert.rejects(
    () => wsAdapter1.synthesizeClause({ text: "cut mid sentence", onPcmChunk: () => {} }),
    /before completion|closed/i,
  );

  class HangAfterPartialWs extends EventEmitter {
    constructor() {
      super();
      this.readyState = 0;
      queueMicrotask(() => {
        this.readyState = 1;
        this.emit("open");
      });
    }
    send(payload) {
      const parsed = JSON.parse(payload);
      if (parsed.text === "") {
        queueMicrotask(() => {
          this.emit("message", JSON.stringify({ audio: partialBase64 }));
        });
      }
    }
    close() {
      this.readyState = 3;
      queueMicrotask(() => this.emit("close", 1000));
    }
  }

  const wsAdapter2 = createElevenLabsWsAdapter({
    apiKey: "el-test-key",
    voiceId: "voice_capri_01",
    WebSocketImpl: HangAfterPartialWs,
  });
  const chunks = [];
  const pending = wsAdapter2.synthesizeClause({
    text: "close mid stream",
    onPcmChunk: (buf) => chunks.push(buf),
  });
  await new Promise((r) => setTimeout(r, 15));
  assert.equal(chunks.length, 1);
  wsAdapter2.close();
  await assert.rejects(pending, /closed|abort/i);
});

test("HTTP streaming delivers chunks incrementally, aligns odd PCM splits, and rejects trailing 1 byte", async () => {
  let sawFirstBeforeEof = false;
  let releaseSecond = null;
  const secondGate = new Promise((resolve) => {
    releaseSecond = resolve;
  });

  const incrementalBody = new ReadableStream({
    async start(controller) {
      controller.enqueue(new Uint8Array([1]));
      await new Promise((r) => setTimeout(r, 5));
      controller.enqueue(new Uint8Array([2, 3, 4]));
      await secondGate;
      controller.enqueue(new Uint8Array([5, 6]));
      controller.close();
    },
  });

  const emitted = [];
  const httpAdapter = createElevenLabsHttpStreamAdapter({
    apiKey: "el-test-key",
    voiceId: "voice_capri_01",
    fetchImpl: async () => new Response(incrementalBody, { status: 200 }),
  });

  const synthPromise = httpAdapter.synthesizeClause({
    text: "Stream me",
    onPcmChunk: (chunk) => {
      emitted.push(Buffer.from(chunk));
      if (emitted.length === 1) {
        sawFirstBeforeEof = true;
        releaseSecond();
      }
    },
  });

  const meta = await synthPromise;
  assert.equal(sawFirstBeforeEof, true);
  assert.equal(meta.byteLength, 6);
  for (const chunk of emitted) {
    assert.equal(chunk.length % 2, 0);
    assert.ok(chunk.length >= 2);
  }
  assert.deepEqual(Buffer.concat(emitted), Buffer.from([1, 2, 3, 4, 5, 6]));

  const oddEofBody = new ReadableStream({
    start(controller) {
      controller.enqueue(new Uint8Array([1, 2, 3]));
      controller.close();
    },
  });
  const oddAdapter = createElevenLabsHttpStreamAdapter({
    apiKey: "el-test-key",
    voiceId: "voice_capri_01",
    fetchImpl: async () => new Response(oddEofBody, { status: 200 }),
  });
  await assert.rejects(
    () => oddAdapter.synthesizeClause({ text: "odd", onPcmChunk: () => {} }),
    /PCM|odd|alignment|frame/i,
  );
});

test("OpenAI TTS strips split extended WAV headers and rejects non-PCM or malformed WAV", async () => {
  const pcm = Buffer.from([10, 20, 30, 40, 50, 60]);
  const junkData = Buffer.from("EXTRAJUNK!"); // 10 bytes even
  const junkChunk = Buffer.concat([
    Buffer.from("JUNK", "ascii"),
    Buffer.from([junkData.length, 0, 0, 0]),
    junkData,
  ]);
  const fmtChunk = Buffer.from("fmt \x10\x00\x00\x00\x01\x00\x01\x00\xc0\x5d\x00\x00\x80\xbb\x00\x00\x02\x00\x10\x00", "latin1");
  const dataHeader = Buffer.from("data\x06\x00\x00\x00", "latin1");
  const riffSize = 4 + fmtChunk.length + junkChunk.length + 8 + pcm.length;
  const riffHeader = Buffer.alloc(12);
  riffHeader.write("RIFF", 0, "ascii");
  riffHeader.writeUInt32LE(riffSize, 4);
  riffHeader.write("WAVE", 8, "ascii");
  const extendedWav = Buffer.concat([riffHeader, fmtChunk, junkChunk, dataHeader, pcm]);

  const splitAt = 16;
  const splitBody = new ReadableStream({
    start(controller) {
      controller.enqueue(extendedWav.subarray(0, splitAt));
      controller.enqueue(extendedWav.subarray(splitAt, splitAt + 25));
      controller.enqueue(extendedWav.subarray(splitAt + 25));
      controller.close();
    },
  });

  const chunks = [];
  const oaAdapter = createOpenAiTtsAdapter({
    apiKey: "sk-test-key",
    fetchImpl: async () => new Response(splitBody, { status: 200 }),
  });
  const res = await oaAdapter.synthesizeClause({
    text: "Split wav",
    onPcmChunk: (buf) => chunks.push(Buffer.from(buf)),
  });
  assert.equal(res.byteLength, pcm.length);
  assert.deepEqual(Buffer.concat(chunks), pcm);

  const floatFmt = Buffer.from("fmt \x10\x00\x00\x00\x03\x00\x01\x00\xc0\x5d\x00\x00\x80\xbb\x00\x00\x04\x00\x20\x00", "latin1");
  const badWav = Buffer.concat([riffHeader, floatFmt, dataHeader, pcm]);
  const badAdapter = createOpenAiTtsAdapter({
    apiKey: "sk-test-key",
    fetchImpl: async () => new Response(badWav, { status: 200 }),
  });
  await assert.rejects(
    () => badAdapter.synthesizeClause({ text: "bad wav", onPcmChunk: () => {} }),
    /WAV|PCM|format/i,
  );
  assert.throws(() => stripWavHeaderIfPresent(badWav), /WAV|PCM|format/i);
});

test("ReactionPool uses global cross-mood LRU on cooldown exhaustion, enforces 8 MiB pool cap, and rejects symlinked root dirs", () => {
  const pool = new ReactionPool({ cooldownTurns: 10 });
  pool.registerClip({ id: "subtle_1", mood: "subtle_breath", pcm24k: Buffer.from([1, 2]) });
  pool.registerClip({ id: "hmm_1", mood: "curious_hmm", pcm24k: Buffer.from([3, 4]) });
  pool.registerClip({ id: "chuckle_1", mood: "light_chuckle", pcm24k: Buffer.from([5, 6]) });

  const first = pool.pick("Why?", { mode: "adaptive", rng: () => 0 });
  const second = pool.pick("Haha!", { mode: "adaptive", rng: () => 0 });
  const third = pool.pick("Ok.", { mode: "adaptive", rng: () => 0 });
  assert.equal(first.id, "hmm_1");
  assert.equal(second.id, "chuckle_1");
  assert.equal(third.id, "subtle_1");

  // Every clip is now inside recentIds (oldest is hmm_1).
  // Even asking for "Ok." (subtle_breath) must pick the global LRU clip (hmm_1), not subtle_1 again.
  const fourth = pool.pick("Ok.", { mode: "adaptive", rng: () => 0 });
  assert.equal(fourth.id, "hmm_1");

  // 8 MiB aggregate byte cap
  const cappedPool = new ReactionPool({ maxClips: 32 });
  const chunk500k = Buffer.alloc(500 * 1024);
  for (let i = 0; i < 16; i++) {
    assert.equal(
      cappedPool.registerClip({
        id: `subtle_big_${i}`,
        mood: "subtle_breath",
        pcm24k: chunk500k,
      }),
      true,
    );
  }
  assert.throws(
    () =>
      cappedPool.registerClip({
        id: "subtle_overflow",
        mood: "subtle_breath",
        pcm24k: chunk500k,
      }),
    /8|size|limit|capacity|bytes/i,
  );

  // Reject symlinked root directory
  const symlinkRootPool = new ReactionPool();
  assert.throws(
    () =>
      symlinkRootPool.loadFromDirectory("symlink-root", {
        fsImpl: {
          existsSync: () => true,
          lstatSync: () => ({
            isSymbolicLink: () => true,
            isDirectory: () => false,
            isFile: () => false,
          }),
          readdirSync: () => [],
        },
      }),
    /symlink/i,
  );

  // Descriptor O_NOFOLLOW path rejects symlinks/non-files before read
  let closedFds = 0;
  const fdPool = new ReactionPool();
  const fdLoaded = fdPool.loadFromDirectory("safe-root", {
    fsImpl: {
      constants: { O_RDONLY: 0, O_NOFOLLOW: 128 },
      existsSync: () => true,
      lstatSync: (p) => {
        if (p === "safe-root") {
          return { isSymbolicLink: () => false, isDirectory: () => true, isFile: () => false, ino: 100 };
        }
        return {
          isSymbolicLink: () => false,
          isDirectory: () => false,
          isFile: () => true,
          ino: p.includes("curious_hmm_01") ? 11 : 10,
        };
      },
      readdirSync: () => ["subtle_breath_01.pcm", "curious_hmm_01.pcm"],
      openSync: (p) => (p.includes("curious_hmm_01") ? 11 : 10),
      fstatSync: (fd) =>
        fd === 11
          ? { isSymbolicLink: () => false, isFile: () => false, size: 4, ino: 11 }
          : { isSymbolicLink: () => false, isFile: () => true, size: 4, ino: 10 },
      readSync: (_fd, target) => {
        Buffer.from([1, 2, 3, 4]).copy(target);
        return 4;
      },
      closeSync: () => {
        closedFds += 1;
      },
    },
  });
  assert.equal(fdLoaded, 1);
  assert.equal(closedFds, 2);
});

test("1-second (48000-byte PCM) 24kHz WAV passes stripWavHeaderIfPresent and createOpenAiTtsAdapter", async () => {
  const pcm = Buffer.alloc(48000);
  for (let i = 0; i < pcm.length; i += 2) {
    pcm.writeInt16LE((i % 32768) - 16384, i);
  }
  const trailingJunk = Buffer.from("LIST\x04\x00\x00\x00INFO", "latin1");
  const header = Buffer.alloc(44);
  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVE", 8, "ascii");
  header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(24000, 24);
  header.writeUInt32LE(48000, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36, "ascii");
  header.writeUInt32LE(pcm.length, 40);

  const tailHeader = Buffer.from(header);
  tailHeader.writeUInt32LE(36 + pcm.length + trailingJunk.length, 4);
  const wavWithTail = Buffer.concat([tailHeader, pcm, trailingJunk]);
  const stripped = stripWavHeaderIfPresent(wavWithTail);
  assert.equal(stripped.length, 48000);
  assert.equal(stripped.equals(pcm), true);

  const fullWav = Buffer.concat([header, pcm]);
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(fullWav.subarray(0, 20000));
      controller.enqueue(fullWav.subarray(20000));
      controller.close();
    },
  });

  const chunks = [];
  const oaAdapter = createOpenAiTtsAdapter({
    apiKey: "sk-test-key",
    fetchImpl: async () => new Response(stream, { status: 200 }),
  });
  const res = await oaAdapter.synthesizeClause({
    text: "One full second of audio",
    onPcmChunk: (chunk) => chunks.push(Buffer.from(chunk)),
  });
  assert.equal(res.byteLength, 48000);
  assert.equal(Buffer.concat(chunks).equals(pcm), true);
});

test("WS aligns odd base64 PCM chunks, routes callback/send errors, bounds warmup timeout, and allows >4 MiB by default", async () => {
  const { EventEmitter } = await import("node:events");
  const oddChunks = [Buffer.from([1]), Buffer.from([2, 3, 4]), Buffer.from([5, 6])];

  class OddChunkWs extends EventEmitter {
    constructor() {
      super();
      this.readyState = 0;
      queueMicrotask(() => {
        this.readyState = 1;
        this.emit("open");
      });
    }
    send(payload) {
      const parsed = JSON.parse(payload);
      if (parsed.text === "") {
        queueMicrotask(() => {
          for (const c of oddChunks) {
            this.emit("message", JSON.stringify({ audio: c.toString("base64") }));
          }
          this.emit("message", JSON.stringify({ isFinal: true }));
        });
      }
    }
    close() {
      this.readyState = 3;
      this.emit("close", 1000);
    }
  }

  const emitted = [];
  const wsAdapter = createElevenLabsWsAdapter({
    apiKey: "el-test-key",
    voiceId: "voice_capri_01",
    WebSocketImpl: OddChunkWs,
  });
  const meta = await wsAdapter.synthesizeClause({
    text: "Odd splits",
    onPcmChunk: (buf) => emitted.push(Buffer.from(buf)),
  });
  assert.equal(meta.byteLength, 6);
  assert.deepEqual(emitted, [Buffer.from([1, 2, 3, 4]), Buffer.from([5, 6])]);

  // Callback throw must reject synthesizeClause instead of escaping as an unhandled exception
  const cbFailAdapter = createElevenLabsWsAdapter({
    apiKey: "el-test-key",
    voiceId: "voice_capri_01",
    WebSocketImpl: OddChunkWs,
  });
  await assert.rejects(
    () =>
      cbFailAdapter.synthesizeClause({
        text: "Throw in callback",
        onPcmChunk: () => {
          throw new Error("downstream playback broke");
        },
      }),
    /downstream playback broke/,
  );

  // ws.send throw must reject synthesizeClause cleanly
  class SendThrowWs extends EventEmitter {
    constructor() {
      super();
      this.readyState = 1;
    }
    send() {
      throw new Error("socket send failed");
    }
    close() {
      this.readyState = 3;
    }
  }
  const sendFailAdapter = createElevenLabsWsAdapter({
    apiKey: "el-test-key",
    voiceId: "voice_capri_01",
    WebSocketImpl: SendThrowWs,
  });
  await assert.rejects(
    () => sendFailAdapter.synthesizeClause({ text: "Fail send", onPcmChunk: () => {} }),
    /socket send failed/,
  );

  // Stalled warmup socket must reject under timeoutMs
  class HangWarmupWs extends EventEmitter {
    constructor() {
      super();
      this.readyState = 0;
    }
    close() {
      this.readyState = 3;
    }
  }
  const warmupTimeoutAdapter = createElevenLabsWsAdapter({
    apiKey: "el-test-key",
    voiceId: "voice_capri_01",
    timeoutMs: 20,
    WebSocketImpl: HangWarmupWs,
  });
  await assert.rejects(() => warmupTimeoutAdapter.warmup(), /timed out|abort/i);

  // Default maxAudioBytes is 4 MiB (rejects 5 MiB), while explicit override up to 16 MiB succeeds
  const fiveMiB = Buffer.alloc(5 * 1024 * 1024);
  const httpDefaultRejected = createElevenLabsHttpStreamAdapter({
    apiKey: "el-key",
    voiceId: "voice_capri_01",
    fetchImpl: async () =>
      new Response(
        new ReadableStream({
          start(c) {
            c.enqueue(fiveMiB);
            c.close();
          },
        }),
        { status: 200 },
      ),
  });
  await assert.rejects(
    () => httpDefaultRejected.synthesizeClause({ text: "Too long for default", onPcmChunk: () => {} }),
    /size limit/i,
  );

  const httpExplicitOverride = createElevenLabsHttpStreamAdapter({
    apiKey: "el-key",
    voiceId: "voice_capri_01",
    maxAudioBytes: 8 * 1024 * 1024,
    fetchImpl: async () =>
      new Response(
        new ReadableStream({
          start(c) {
            c.enqueue(fiveMiB);
            c.close();
          },
        }),
        { status: 200 },
      ),
  });
  const bigMeta = await httpExplicitOverride.synthesizeClause({
    text: "Long story with 8 MiB limit",
    onPcmChunk: () => {},
  });
  assert.equal(bigMeta.byteLength, 5 * 1024 * 1024);
});

test("ReactionPool skips symlinks before openSync when O_NOFOLLOW is undefined and stores immutable cloned clips", () => {
  const openedPaths = [];
  let closedCount = 0;
  const pool = new ReactionPool();
  const loaded = pool.loadFromDirectory("win-root", {
    fsImpl: {
      constants: { O_RDONLY: 0 },
      existsSync: () => true,
      lstatSync: (p) => {
        if (p === "win-root") {
          return { isSymbolicLink: () => false, isDirectory: () => true, isFile: () => false, ino: 1 };
        }
        if (p.includes("curious_hmm_symlink")) {
          return { isSymbolicLink: () => true, isDirectory: () => false, isFile: () => false, ino: 2 };
        }
        return { isSymbolicLink: () => false, isDirectory: () => false, isFile: () => true, ino: 3 };
      },
      readdirSync: () => ["subtle_breath_ok.pcm", "curious_hmm_symlink.pcm"],
      openSync: (p) => {
        openedPaths.push(p);
        return 42;
      },
      fstatSync: () => ({ isSymbolicLink: () => false, isFile: () => true, size: 4, ino: 3 }),
      readSync: (_fd, target) => {
        Buffer.from([1, 2, 3, 4]).copy(target);
        return 4;
      },
      closeSync: () => {
        closedCount += 1;
      },
    },
  });
  assert.equal(loaded, 1);
  assert.equal(closedCount, 1);
  assert.equal(openedPaths.length, 1);
  assert.equal(openedPaths[0].includes("curious_hmm_symlink"), false);

  // Registered clip must clone and freeze buffers so caller mutations cannot corrupt or inflate storage
  const rawPcm = Buffer.from([10, 20, 30, 40]);
  const rawFrames = [Buffer.from([1, 2])];
  pool.registerClip({
    id: "playful_scoff_mut",
    mood: "playful_scoff",
    pcm24k: rawPcm,
    frames48kOpus: rawFrames,
  });
  rawPcm[0] = 99;
  rawFrames[0][0] = 88;
  const stored = pool.pick("Seriously?", { mode: "adaptive", rng: () => 0 });
  assert.equal(stored.id, "playful_scoff_mut");
  assert.equal(stored.pcm24k[0], 10);
  assert.equal(stored.frames48kOpus[0][0], 1);
  assert.equal(Object.isFrozen(stored), true);
  assert.equal(Object.isFrozen(stored.frames48kOpus), true);
});

test("OpenAI TTS clips streamed WAV at declared data chunk extent and rejects truncated WAV data", async () => {
  const pcm = Buffer.alloc(48000);
  for (let i = 0; i < pcm.length; i += 2) {
    pcm.writeInt16LE(((i * 7) % 32768) - 16384, i);
  }
  const trailingList = Buffer.from("LIST\x04\x00\x00\x00INFO", "latin1");
  assert.equal(trailingList.length, 12);

  const header = Buffer.alloc(44);
  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + pcm.length + trailingList.length, 4);
  header.write("WAVE", 8, "ascii");
  header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(24000, 24);
  header.writeUInt32LE(48000, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36, "ascii");
  header.writeUInt32LE(pcm.length, 40);

  const wavWithTrailingList = Buffer.concat([header, pcm, trailingList]);
  const streamedChunks = [];
  const oaAdapter = createOpenAiTtsAdapter({
    apiKey: "sk-test-key",
    fetchImpl: async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(wavWithTrailingList.subarray(0, 20000));
            controller.enqueue(wavWithTrailingList.subarray(20000, 48040));
            controller.enqueue(wavWithTrailingList.subarray(48040));
            controller.close();
          },
        }),
        { status: 200 },
      ),
  });
  const res = await oaAdapter.synthesizeClause({
    text: "Bounded data extent",
    onPcmChunk: (buf) => streamedChunks.push(Buffer.from(buf)),
  });
  assert.equal(res.byteLength, 48000);
  assert.equal(Buffer.concat(streamedChunks).equals(pcm), true);

  const truncatedWav = Buffer.concat([header, pcm.subarray(0, 1000)]);
  assert.throws(() => stripWavHeaderIfPresent(truncatedWav), /Truncated WAV/i);

  const truncAdapter = createOpenAiTtsAdapter({
    apiKey: "sk-test-key",
    fetchImpl: async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(truncatedWav.subarray(0, 500));
            controller.enqueue(truncatedWav.subarray(500));
            controller.close();
          },
        }),
        { status: 200 },
      ),
  });
  await assert.rejects(
    () => truncAdapter.synthesizeClause({ text: "Truncated audio", onPcmChunk: () => {} }),
    /Truncated WAV/i,
  );
});

test("ReactionPool.loadFromDirectory enforces trustedRootDir containment, realpath entry bounds, and dev/ino match", () => {
  const escapedPool = new ReactionPool();
  assert.throws(
    () =>
      escapedPool.loadFromDirectory("/untrusted/cache", {
        trustedRootDir: "/trusted/root",
        fsImpl: {
          existsSync: () => true,
          lstatSync: () => ({
            isSymbolicLink: () => false,
            isDirectory: () => true,
            isFile: () => false,
          }),
          realpathSync: (p) => path.posix.resolve(p),
          readdirSync: () => [],
        },
      }),
    RangeError,
  );

  // Entry whose realpathSync resolves outside resolvedDir must be skipped cleanly
  const entryEscapePool = new ReactionPool();
  const loadedEscapedEntry = entryEscapePool.loadFromDirectory("/trusted/root/cache", {
    trustedRootDir: "/trusted/root",
    fsImpl: {
      constants: { O_RDONLY: 0 },
      existsSync: () => true,
      lstatSync: (p) => {
        if (p.endsWith("cache")) {
          return { isSymbolicLink: () => false, isDirectory: () => true, isFile: () => false, dev: 1, ino: 10 };
        }
        return { isSymbolicLink: () => false, isDirectory: () => false, isFile: () => true, dev: 1, ino: 20 };
      },
      realpathSync: (p) => {
        if (p.includes("curious_hmm_escape")) return "/elsewhere/curious_hmm_escape.pcm";
        return p;
      },
      readdirSync: () => ["subtle_breath_ok.pcm", "curious_hmm_escape.pcm"],
      openSync: () => 5,
      fstatSync: () => ({ isSymbolicLink: () => false, isFile: () => true, size: 4, dev: 1, ino: 20 }),
      readSync: (_fd, target) => {
        Buffer.from([1, 2, 3, 4]).copy(target);
        return 4;
      },
      closeSync: () => {},
    },
  });
  assert.equal(loadedEscapedEntry, 1);

  // Entry whose entryLstat dev/ino disagrees with fstatSync(fd) dev/ino must be skipped cleanly
  const devMismatchPool = new ReactionPool();
  const loadedMismatched = devMismatchPool.loadFromDirectory("/trusted/root/cache", {
    trustedRootDir: "/trusted/root",
    fsImpl: {
      constants: { O_RDONLY: 0 },
      existsSync: () => true,
      lstatSync: (p) => {
        if (p.endsWith("cache")) {
          return { isSymbolicLink: () => false, isDirectory: () => true, isFile: () => false, dev: 1, ino: 10 };
        }
        return { isSymbolicLink: () => false, isDirectory: () => false, isFile: () => true, dev: 1, ino: 20 };
      },
      realpathSync: (p) => p,
      readdirSync: () => ["subtle_breath_swapped.pcm"],
      openSync: () => 6,
      fstatSync: () => ({ isSymbolicLink: () => false, isFile: () => true, size: 4, dev: 2, ino: 20 }),
      readSync: (_fd, target) => {
        Buffer.from([1, 2, 3, 4]).copy(target);
        return 4;
      },
      closeSync: () => {},
    },
  });
  assert.equal(loadedMismatched, 0);
});

function buildCustomWavBuffer({
  audioFormat = 1,
  numChannels = 1,
  sampleRate = 24000,
  bitsPerSample = 16,
  blockAlign = numChannels * (bitsPerSample / 8),
  junkChunkSizes = [],
  includeData = true,
  pcmBytes = Buffer.from([1, 2, 3, 4]),
} = {}) {
  const chunks = [];
  const fmt = Buffer.alloc(24);
  fmt.write("fmt ", 0, 4, "ascii");
  fmt.writeUInt32LE(16, 4);
  fmt.writeUInt16LE(audioFormat, 8);
  fmt.writeUInt16LE(numChannels, 10);
  fmt.writeUInt32LE(sampleRate, 12);
  fmt.writeUInt32LE(sampleRate * blockAlign, 16);
  fmt.writeUInt16LE(blockAlign, 20);
  fmt.writeUInt16LE(bitsPerSample, 22);
  chunks.push(fmt);

  for (const size of junkChunkSizes) {
    const padded = size + (size % 2);
    const junk = Buffer.alloc(8 + padded);
    junk.write("JUNK", 0, 4, "ascii");
    junk.writeUInt32LE(size, 4);
    chunks.push(junk);
  }

  if (includeData) {
    const dataHeader = Buffer.alloc(8);
    dataHeader.write("data", 0, 4, "ascii");
    dataHeader.writeUInt32LE(pcmBytes.length, 4);
    chunks.push(dataHeader, pcmBytes);
  }

  const body = Buffer.concat(chunks);
  const riff = Buffer.alloc(12);
  riff.write("RIFF", 0, 4, "ascii");
  riff.writeUInt32LE(4 + body.length, 4);
  riff.write("WAVE", 8, 4, "ascii");
  return Buffer.concat([riff, body]);
}

test("ElevenLabs WS adapter close() aborts in-flight warmup and rejects subsequent synthesis", async () => {
  const sockets = [];
  class PendingWebSocket {
    constructor(url) {
      this.url = url;
      this.readyState = 0;
      this.closeCalls = 0;
      this.listeners = new Map();
      sockets.push(this);
    }
    on(event, fn) {
      const list = this.listeners.get(event) ?? [];
      list.push(fn);
      this.listeners.set(event, list);
    }
    removeListener(event, fn) {
      const list = this.listeners.get(event) ?? [];
      this.listeners.set(
        event,
        list.filter((entry) => entry !== fn),
      );
    }
    close() {
      this.closeCalls += 1;
      this.readyState = 3;
    }
    send() {}
  }

  const adapter = createElevenLabsWsAdapter({
    apiKey: "el-key",
    voiceId: "voice-1",
    timeoutMs: 500,
    WebSocketImpl: PendingWebSocket,
  });

  const w1 = adapter.warmup();
  const w2 = adapter.warmup();
  const settled = Promise.allSettled([w1, w2]);
  await new Promise((resolve) => setImmediate(resolve));
  assert.ok(sockets.length >= 1, "warmup must open a socket");

  adapter.close();
  const allClosedImmediately = sockets.every((socket) => socket.closeCalls >= 1);
  const results = await settled;
  assert.ok(allClosedImmediately, "every pending warmup socket must be closed immediately on close()");
  assert.equal(results.length, 2);
  for (const result of results) {
    assert.equal(result.status, "rejected");
    assert.match(String(result.reason?.message ?? result.reason), /aborted|closed/i);
  }

  await assert.rejects(adapter.warmup(), /closed/i);
  await assert.rejects(adapter.synthesizeClause({ text: "hello" }), /closed/i);
  const countAfterClose = sockets.length;
  assert.ok(countAfterClose >= 1);
  await assert.rejects(adapter.warmup(), /closed/i);
  assert.equal(sockets.length, countAfterClose, "closed adapter must not open new sockets");
});

test("ElevenLabs WS overlapping warmup() calls retain exactly one socket and close any surplus before close()", async () => {
  const sockets = [];
  class OpeningWebSocket {
    constructor() {
      this.readyState = 0;
      this.closeCalls = 0;
      this.handlers = new Map();
      sockets.push(this);
      queueMicrotask(() => {
        this.readyState = 1;
        this.emit("open");
      });
    }
    on(event, fn) {
      const list = this.handlers.get(event) ?? [];
      list.push(fn);
      this.handlers.set(event, list);
    }
    removeListener(event, fn) {
      const list = this.handlers.get(event) ?? [];
      this.handlers.set(
        event,
        list.filter((entry) => entry !== fn),
      );
    }
    emit(event, ...args) {
      for (const fn of [...(this.handlers.get(event) ?? [])]) fn(...args);
    }
    send() {}
    close() {
      this.closeCalls += 1;
      this.readyState = 3;
    }
  }

  const adapter = createElevenLabsWsAdapter({
    apiKey: "el-key",
    voiceId: "voice-1",
    WebSocketImpl: OpeningWebSocket,
  });

  const w1 = adapter.warmup();
  const w2 = adapter.warmup();
  await Promise.all([w1, w2]);

  const live = sockets.filter((socket) => socket.readyState === 1);
  const surplus = sockets.filter((socket) => socket.readyState !== 1);
  assert.equal(live.length, 1, "exactly one warmup socket must be retained");
  assert.equal(
    surplus.length,
    sockets.length - 1,
    "every surplus warmup socket must be closed before close() is called",
  );
  assert.ok(
    surplus.every((socket) => socket.closeCalls >= 1),
    "surplus sockets must record a close call",
  );

  adapter.close();
  assert.ok(
    sockets.every((socket) => socket.readyState === 3),
    "every opened socket must be closed after adapter close()",
  );
});

test("ElevenLabs WS concurrent warmup() clears all deadline timers on close() and cleans external AbortSignal listeners on open and abort", async () => {
  const originalSetTimeout = globalThis.setTimeout;
  const originalClearTimeout = globalThis.clearTimeout;
  const activeTimers = new Set();

  globalThis.setTimeout = (...args) => {
    const handle = originalSetTimeout(...args);
    activeTimers.add(handle);
    return handle;
  };
  globalThis.clearTimeout = (handle) => {
    activeTimers.delete(handle);
    return originalClearTimeout(handle);
  };

  const originalAny = AbortSignal.any;
  try {
    const combinedSignals = [];
    AbortSignal.any = function captureAny(signals) {
      const combined = originalAny.call(AbortSignal, signals);
      combinedSignals.push(combined);
      return combined;
    };

    const pendingSockets = [];
    class HangingWebSocket {
      constructor() {
        this.readyState = 0;
        this.closeCalls = 0;
      }
      on() {}
      removeListener() {}
      close() {
        this.closeCalls += 1;
        this.readyState = 3;
      }
    }

    const adapter1 = createElevenLabsWsAdapter({
      apiKey: "el-key",
      voiceId: "voice-1",
      timeoutMs: 250,
      WebSocketImpl: class extends HangingWebSocket {
        constructor() {
          super();
          pendingSockets.push(this);
        }
      },
    });

    const p1 = adapter1.warmup();
    const p2 = adapter1.warmup();
    const settled = Promise.allSettled([p1, p2]);
    await new Promise((resolve) => setImmediate(resolve));
    assert.ok(pendingSockets.length >= 1, "warmup must open a socket");

    adapter1.close();
    const allClosedImmediately = pendingSockets.every((s) => s.closeCalls >= 1);
    const results = await settled;
    assert.ok(allClosedImmediately, "every pending warmup socket must be closed immediately on close()");
    assert.ok(results.every((result) => result.status === "rejected"));
    assert.equal(activeTimers.size, 0, "all warmup deadline timers must be cleared on close()");

    // Verify external AbortSignal listeners are removed on both normal open+complete and mid-stream abort
    class ControllableWebSocket {
      constructor() {
        this.readyState = 0;
        this.handlers = new Map();
      }
      on(event, fn) {
        const list = this.handlers.get(event) ?? [];
        list.push(fn);
        this.handlers.set(event, list);
        if (event === "open") {
          queueMicrotask(() => {
            this.readyState = 1;
            this.emit("open");
          });
        }
      }
      removeListener(event, fn) {
        const list = this.handlers.get(event) ?? [];
        this.handlers.set(event, list.filter((entry) => entry !== fn));
      }
      emit(event, ...args) {
        for (const fn of [...(this.handlers.get(event) ?? [])]) {
          fn(...args);
        }
      }
      send(raw) {
        const parsed = JSON.parse(raw);
        if (parsed.text === "") {
          queueMicrotask(() => {
            this.emit("message", JSON.stringify({ audio: Buffer.from([1, 2]).toString("base64"), isFinal: true }));
          });
        }
      }
      close() {
        this.readyState = 3;
      }
    }

    const openAc = new AbortController();
    const openAdapter = createElevenLabsWsAdapter({
      apiKey: "el-key",
      voiceId: "voice-1",
      timeoutMs: 60000,
      WebSocketImpl: ControllableWebSocket,
    });
    await openAdapter.synthesizeClause({ text: "hi", signal: openAc.signal, onPcmChunk: () => {} });
    assert.equal(getEventListeners(openAc.signal, "abort").length, 0);
    assert.ok(combinedSignals.length > 0, "synthesizeClause must build a combined abort signal");
    for (const combined of combinedSignals) {
      assert.equal(
        getEventListeners(combined, "abort").length,
        0,
        "no abort listener may remain on the combined signal after a normal open and completion",
      );
    }
    assert.equal(activeTimers.size, 0);

    combinedSignals.length = 0;
    const abortAc = new AbortController();
    const abortAdapter = createElevenLabsWsAdapter({
      apiKey: "el-key",
      voiceId: "voice-1",
      timeoutMs: 60000,
      WebSocketImpl: HangingWebSocket,
    });
    const synthPromise = abortAdapter.synthesizeClause({ text: "bye", signal: abortAc.signal });
    await Promise.resolve();
    abortAc.abort(new Error("user interrupted"));
    await assert.rejects(synthPromise, /interrupted|aborted/i);
    assert.equal(getEventListeners(abortAc.signal, "abort").length, 0);
    assert.ok(combinedSignals.length > 0, "aborted synthesizeClause must build a combined abort signal");
    for (const combined of combinedSignals) {
      assert.equal(
        getEventListeners(combined, "abort").length,
        0,
        "no abort listener may remain on the combined signal after a mid-stream abort",
      );
    }
    assert.equal(activeTimers.size, 0);
  } finally {
    AbortSignal.any = originalAny;
    globalThis.setTimeout = originalSetTimeout;
    globalThis.clearTimeout = originalClearTimeout;
  }
});

test("WAV header parsers reject stereo, non-24kHz, and >4096-byte single chunks exceeding scan cap", async () => {
  const stereoWav = buildCustomWavBuffer({ numChannels: 2, sampleRate: 24000 });
  const stereoMonoAlignWav = buildCustomWavBuffer({
    numChannels: 2,
    sampleRate: 24000,
    bitsPerSample: 16,
    blockAlign: 2,
  });
  const wav48k = buildCustomWavBuffer({ numChannels: 1, sampleRate: 48000 });
  const wav16k = buildCustomWavBuffer({ numChannels: 1, sampleRate: 16000 });

  assert.throws(() => stripWavHeaderIfPresent(stereoWav), /Unsupported/i);
  assert.throws(() => stripWavHeaderIfPresent(stereoMonoAlignWav), /Unsupported/i);
  assert.throws(() => stripWavHeaderIfPresent(wav48k), /Unsupported/i);
  assert.throws(() => stripWavHeaderIfPresent(wav16k), /Unsupported/i);
  assert.deepEqual(
    stripWavHeaderIfPresent(wav16k, { sampleRate: 16000 }),
    Buffer.from([1, 2, 3, 4]),
  );

  // Single large chunk (> 4096 bytes) whose data chunk sits past MAX_WAV_SCAN_BYTES
  const oversizedHeaderWav = buildCustomWavBuffer({
    junkChunkSizes: Array.from({ length: 36 }, () => 128),
  });
  assert.ok(oversizedHeaderWav.length > 4096);

  for (const badWav of [stereoWav, stereoMonoAlignWav, wav48k, oversizedHeaderWav]) {
    const adapter = createOpenAiTtsAdapter({
      apiKey: "oa-key",
      fetchImpl: async () =>
        new Response(new ReadableStream({
          start(controller) {
            controller.enqueue(badWav);
            controller.close();
          },
        }), { status: 200 }),
    });
    await assert.rejects(
      adapter.synthesizeClause({ text: "test", onPcmChunk: () => {} }),
      /Unsupported|exceeded maximum scan size/i,
    );
  }
});

function makeNoCloneUint8Array(byteLength) {
  const view = new Uint8Array(byteLength);
  Object.defineProperty(view, "buffer", {
    get() {
      throw new Error("cloned before limit check");
    },
  });
  return view;
}

test("ReactionPool.registerClip validates Buffer/Uint8Array types and enforces file, aggregate, and maxClips limits before cloning", () => {
  const pool = new ReactionPool();
  for (const badPcm of [{ type: "Buffer", data: [1, 2, 3] }, "raw-string", 12345]) {
    assert.throws(
      () => pool.registerClip({ id: "subtle_breath_bad", mood: "subtle_breath", pcm24k: badPcm }),
      TypeError,
    );
  }
  for (const badFrame of [{ type: "Buffer", data: [1, 2] }, "opus-str", [1, 2]]) {
    assert.throws(
      () =>
        pool.registerClip({
          id: "subtle_breath_bad_frame",
          mood: "subtle_breath",
          frames48kOpus: [badFrame],
        }),
      TypeError,
    );
  }
  assert.equal(pool.size, 0);

  const oversizePcm = makeNoCloneUint8Array(600 * 1024);
  assert.throws(
    () => pool.registerClip({ id: "subtle_breath_big", mood: "subtle_breath", pcm24k: oversizePcm }),
    RangeError,
  );

  const tinyPool = new ReactionPool({ maxFileBytes: 512 * 1024, maxTotalBytes: 256 * 1024 });
  assert.equal(
    tinyPool.registerClip({
      id: "subtle_breath_ok",
      mood: "subtle_breath",
      pcm24k: Buffer.alloc(150 * 1024),
    }),
    true,
  );
  const aggregateOverPcm = makeNoCloneUint8Array(200 * 1024);
  assert.throws(
    () =>
      tinyPool.registerClip({
        id: "subtle_breath_over",
        mood: "subtle_breath",
        pcm24k: aggregateOverPcm,
      }),
    RangeError,
  );
  assert.equal(tinyPool.size, 1);

  const opusOverPool = new ReactionPool({ maxFileBytes: 128 * 1024, maxTotalBytes: 200 * 1024 });
  assert.equal(
    opusOverPool.registerClip({
      id: "subtle_breath_1",
      mood: "subtle_breath",
      frames48kOpus: [Buffer.alloc(100 * 1024)],
    }),
    true,
  );
  assert.throws(
    () =>
      opusOverPool.registerClip({
        id: "subtle_breath_opus_over",
        mood: "subtle_breath",
        pcm24k: Buffer.alloc(0),
        frames48kOpus: [Buffer.alloc(60 * 1024), Buffer.alloc(60 * 1024)],
      }),
    RangeError,
  );
  assert.equal(opusOverPool.size, 1);

  const cappedPool = new ReactionPool({ maxClips: 1 });
  cappedPool.registerClip({ id: "subtle_breath_1", mood: "subtle_breath", pcm24k: Buffer.alloc(4) });
  const uncopiedFrame = makeNoCloneUint8Array(4);
  assert.equal(
    cappedPool.registerClip({
      id: "subtle_breath_2",
      mood: "subtle_breath",
      frames48kOpus: [uncopiedFrame],
    }),
    false,
  );
});

function makeWalkFs({
  dirs = {},
  files = {},
  real = {},
  entries = {},
  lstatHook = null,
  onReaddir = null,
  onClose = null,
} = {}) {
  const calls = { mkdir: [], lstat: [], arrayBuffer: 0 };
  const dirMap = new Map(Object.entries(dirs));
  const enoent = (target) => {
    const err = new Error(`ENOENT: ${target}`);
    err.code = "ENOENT";
    return err;
  };
  const fileIno = (target) => 5000 + Object.keys(files).indexOf(target);
  const dirIno = (target, overrides) =>
    overrides.ino ?? 1000 + [...dirMap.keys()].indexOf(target);
  const dirStat = (target, overrides) => ({
    isSymbolicLink: () => overrides.symlink === true,
    isDirectory: () => true,
    isFile: () => false,
    dev: 1,
    ino: dirIno(target, overrides),
    fileAttributes: overrides.fileAttributes,
    mode: overrides.mode,
  });
  const fileStat = (target) => ({
    isSymbolicLink: () => false,
    isDirectory: () => false,
    isFile: () => true,
    size: files[target].length,
    dev: 1,
    ino: fileIno(target),
  });
  const fds = new Map();
  let nextFd = 10;
  const fsImpl = {
    constants: { O_RDONLY: 0 },
    existsSync: (target) => dirMap.has(target) || Object.hasOwn(files, target),
    lstatSync(target) {
      calls.lstat.push(target);
      if (lstatHook) {
        const hit = lstatHook(target, calls.lstat.filter((entry) => entry === target).length);
        if (hit) return hit;
      }
      if (dirMap.has(target)) return dirStat(target, dirMap.get(target));
      if (Object.hasOwn(files, target)) return fileStat(target);
      throw enoent(target);
    },
    realpathSync(target) {
      if (!dirMap.has(target) && !Object.hasOwn(files, target)) throw enoent(target);
      return real[target] ?? target;
    },
    readdirSync(target) {
      onReaddir?.(target);
      return entries[target] ?? [];
    },
    openSync(target) {
      const fd = nextFd++;
      fds.set(fd, target);
      return fd;
    },
    fstatSync: (fd) => fileStat(fds.get(fd)),
    readSync(fd, target, offset, length) {
      const source = files[fds.get(fd)];
      source.copy(target, offset, 0, Math.min(length, source.length));
      return Math.min(length, source.length);
    },
    closeSync() {
      onClose?.();
    },
    mkdirSync(target, options) {
      calls.mkdir.push({ path: target, mode: options?.mode });
      dirMap.set(target, {});
    },
  };
  return { fsImpl, calls };
}

test("C5 cache walk: loadFromDirectory rejects a symlink or reparse-point ancestor even when it resolves inside the trusted root", () => {
  const base = path.resolve("/trusted/root");
  const mid = path.join(base, "mid");
  const cache = path.join(mid, "cache");
  const realCache = path.join(base, "real", "cache");
  const clip = path.join(realCache, "subtle_breath_ok.pcm");
  const cases = [
    { label: "symlink ancestor", overrides: { symlink: true } },
    { label: "reparse-point ancestor", overrides: { fileAttributes: 0x400 } },
  ];
  for (const { label, overrides } of cases) {
    const { fsImpl } = makeWalkFs({
      dirs: { [base]: {}, [mid]: overrides, [cache]: {}, [realCache]: {} },
      files: { [clip]: Buffer.from([1, 2, 3, 4]) },
      // realpath returns the path unchanged, so only the per-component lstat walk can see
      // the link; the older leaf realpath comparison has nothing to flag.
      entries: { [cache]: ["subtle_breath_ok.pcm"] },
    });
    const pool = new ReactionPool();
    assert.throws(
      () => pool.loadFromDirectory(cache, { fsImpl, trustedRootDir: base }),
      /symlink or reparse point/,
      label,
    );
    assert.equal(pool.size, 0, label);
  }
});

test("C5 cache walk: loadFromDirectory aborts when an ancestor is swapped after the walk", () => {
  const base = path.resolve("/trusted/root");
  const mid = path.join(base, "mid");
  const cache = path.join(mid, "cache");
  const clip = path.join(cache, "subtle_breath_ok.pcm");
  const swaps = [
    { label: "inode changed", swapped: { ino: 999 } },
    { label: "became symlink", swapped: { symlink: true } },
  ];
  for (const { label, swapped } of swaps) {
    const { fsImpl } = makeWalkFs({
      dirs: { [base]: {}, [mid]: {}, [cache]: {} },
      files: { [clip]: Buffer.from([1, 2, 3, 4]) },
      entries: { [cache]: ["subtle_breath_ok.pcm"] },
      lstatHook: (target, count) =>
        target === mid && count > 1
          ? {
              isSymbolicLink: () => swapped.symlink === true,
              isDirectory: () => true,
              isFile: () => false,
              dev: 1,
              ino: swapped.ino ?? 1001,
            }
          : null,
    });
    const pool = new ReactionPool();
    assert.throws(
      () => pool.loadFromDirectory(cache, { fsImpl, trustedRootDir: base }),
      RangeError,
      label,
    );
    assert.equal(pool.size, 0, label);
  }
});

test("C5 cache walk: createDir builds missing components with mode 0o700 only after verifying existing ancestors", () => {
  const base = path.resolve("/trusted/root");
  const mid = path.join(base, "mid");
  const cache = path.join(mid, "cache");

  const missing = makeWalkFs({ dirs: { [base]: {} } });
  assert.equal(
    new ReactionPool().loadFromDirectory(cache, {
      fsImpl: missing.fsImpl,
      trustedRootDir: base,
      createDir: true,
    }),
    0,
  );
  assert.deepEqual(missing.calls.mkdir, [
    { path: mid, mode: 0o700 },
    { path: cache, mode: 0o700 },
  ]);

  const readOnly = makeWalkFs({ dirs: { [base]: {} } });
  assert.equal(
    new ReactionPool().loadFromDirectory(cache, { fsImpl: readOnly.fsImpl, trustedRootDir: base }),
    0,
  );
  assert.deepEqual(readOnly.calls.mkdir, []);

  const linked = makeWalkFs({ dirs: { [base]: {}, [mid]: { symlink: true } } });
  assert.throws(
    () =>
      new ReactionPool().loadFromDirectory(cache, {
        fsImpl: linked.fsImpl,
        trustedRootDir: base,
        createDir: true,
      }),
    RangeError,
  );
  assert.deepEqual(linked.calls.mkdir, []);
});

test("C5 declared length: registerClip rejects a value whose declared byteLength disagrees with its extent", () => {
  class LyingBytes extends Uint8Array {
    get byteLength() {
      return 4;
    }
  }
  const pool = new ReactionPool({ maxFileBytes: 16, maxTotalBytes: 64 });
  assert.throws(
    () => pool.registerClip({ id: "subtle_breath_lie", mood: "subtle_breath", pcm24k: new LyingBytes(64) }),
    RangeError,
  );
  assert.throws(
    () =>
      pool.registerClip({
        id: "subtle_breath_lie_frame",
        mood: "subtle_breath",
        frames48kOpus: [new LyingBytes(64)],
      }),
    RangeError,
  );
  // A lie that stays under every cap is only caught by comparing the clone with the declared
  // length, so this case isolates the post-copy measurement from the cap checks.
  assert.throws(
    () => pool.registerClip({ id: "subtle_breath_small_lie", mood: "subtle_breath", pcm24k: new LyingBytes(8) }),
    /declared byte length does not match/,
  );
  assert.throws(
    () =>
      pool.registerClip({
        id: "subtle_breath_small_lie_frame",
        mood: "subtle_breath",
        frames48kOpus: [new LyingBytes(8)],
      }),
    /declared byte length does not match/,
  );
  assert.equal(pool.size, 0);
  assert.equal(pool.totalBytes, 0);
});

test("C5 outer RIFF length: stripWavHeaderIfPresent rejects a declared container size that disagrees with the buffer", () => {
  const good = buildCustomWavBuffer();
  assert.equal(stripWavHeaderIfPresent(good).length, 4);

  const tooBig = Buffer.from(good);
  tooBig.writeUInt32LE(good.length - 8 + 100, 4);
  assert.throws(() => stripWavHeaderIfPresent(tooBig), /RIFF/i);

  const tooSmall = Buffer.from(good);
  tooSmall.writeUInt32LE(good.length - 8 - 2, 4);
  assert.throws(() => stripWavHeaderIfPresent(tooSmall), /RIFF/i);

  const trailing = Buffer.concat([good, Buffer.from([9, 9, 9, 9])]);
  assert.throws(() => stripWavHeaderIfPresent(trailing), /RIFF/i);

  const sentinel = Buffer.from(good);
  sentinel.writeUInt32LE(0xffffffff, 4);
  assert.equal(stripWavHeaderIfPresent(sentinel).length, 4);
});

test("C5 outer RIFF length: OpenAI streaming adapter rejects a stream whose total size disagrees with the RIFF header", async () => {
  const run = async (mutate) => {
    const wav = mutate(buildCustomWavBuffer());
    const adapter = createOpenAiTtsAdapter({
      apiKey: "k",
      baseUrl: "https://tts.example.com/v1",
      fetchImpl: async () => new Response(wav, { status: 200 }),
    });
    return adapter.synthesizeClause({ text: "riff", onPcmChunk: () => {} });
  };
  const ok = await run((wav) => wav);
  assert.equal(ok.byteLength, 4);
  await assert.rejects(
    run((wav) => {
      wav.writeUInt32LE(wav.length - 8 + 64, 4);
      return wav;
    }),
    /RIFF/i,
  );
  await assert.rejects(run((wav) => Buffer.concat([wav, Buffer.from([7, 7])])), /RIFF/i);
  const sentinel = await run((wav) => {
    wav.writeUInt32LE(0xffffffff, 4);
    return wav;
  });
  assert.equal(sentinel.byteLength, 4);
});

test("C5 injected non-streaming responses: any body that cannot be read incrementally is rejected before it is read", async () => {
  const makeResponse = ({ contentLength, bytes }) => {
    const state = { reads: 0 };
    const headers = new Headers();
    if (contentLength !== undefined) headers.set("content-length", String(contentLength));
    return {
      state,
      response: {
        ok: true,
        status: 200,
        body: null,
        headers,
        async arrayBuffer() {
          state.reads += 1;
          return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        },
      },
    };
  };
  const synth = async (fixture, factory) => {
    const adapter = factory(async () => fixture.response);
    return adapter.synthesizeClause({ text: "inject", onPcmChunk: () => {} });
  };
  const factories = [
    (fetchImpl) => createElevenLabsHttpStreamAdapter({ apiKey: "k", voiceId: "v1", maxAudioBytes: 1024, fetchImpl }),
    (fetchImpl) =>
      createOpenAiTtsAdapter({
        apiKey: "k",
        baseUrl: "https://tts.example.com/v1",
        maxAudioBytes: 1024,
        fetchImpl,
      }),
  ];
  for (const factory of factories) {
    const oversized = makeResponse({ contentLength: 10 * 1024 * 1024, bytes: Buffer.alloc(4) });
    await assert.rejects(synth(oversized, factory), /size|stream/i);
    assert.equal(oversized.state.reads, 0, "oversized declared length must be rejected before reading");

    const unknown = makeResponse({ bytes: Buffer.alloc(4) });
    await assert.rejects(synth(unknown, factory), /size|stream/i);
    assert.equal(unknown.state.reads, 0, "unknown length must be rejected before reading");

    // A declared length is attacker-controlled metadata: even a small, honest-looking value
    // cannot bound arrayBuffer(), so a body that is not a stream is never buffered.
    const lying = makeResponse({ contentLength: 4, bytes: Buffer.alloc(2048) });
    await assert.rejects(synth(lying, factory), /stream/i);
    assert.equal(lying.state.reads, 0, "a declared length must not unlock arrayBuffer()");

    const honest = makeResponse({ contentLength: 4, bytes: Buffer.from([1, 2, 3, 4]) });
    await assert.rejects(synth(honest, factory), /stream/i);
    assert.equal(honest.state.reads, 0);
  }
});

test("C5 rework: without createDir a missing intermediate ancestor ends the load instead of being skipped", () => {
  const base = path.resolve("/trusted/root");
  const mid = path.join(base, "mid");
  const cache = path.join(mid, "cache");
  const clip = path.join(cache, "subtle_breath_ok.pcm");
  // The leaf exists while its parent does not: the shape left by a concurrent create or
  // replace. Every later check would run against an unverified component.
  const { fsImpl, calls } = makeWalkFs({
    dirs: { [base]: {}, [cache]: {} },
    files: { [clip]: Buffer.from([1, 2, 3, 4]) },
    entries: { [cache]: ["subtle_breath_ok.pcm"] },
  });
  const pool = new ReactionPool();
  assert.equal(pool.loadFromDirectory(cache, { fsImpl, trustedRootDir: base }), 0);
  assert.equal(pool.size, 0);
  assert.deepEqual(calls.mkdir, []);
});

test("C5 rework: the ancestor chain is verified before the directory is listed", () => {
  const base = path.resolve("/trusted/root");
  const mid = path.join(base, "mid");
  const cache = path.join(mid, "cache");
  const clip = path.join(cache, "subtle_breath_ok.pcm");
  const swaps = [
    { label: "inode changed", swapped: { ino: 999 } },
    { label: "became symlink", swapped: { symlink: true } },
  ];
  for (const { label, swapped } of swaps) {
    let cacheLstats = 0;
    let listed = false;
    const { fsImpl } = makeWalkFs({
      dirs: { [base]: {}, [mid]: {}, [cache]: {} },
      files: { [clip]: Buffer.from([1, 2, 3, 4]) },
      entries: { [cache]: ["subtle_breath_ok.pcm"] },
      // The ancestor is swapped after the walk and restored before the listing returns, so
      // only a check that runs before readdirSync can see it.
      lstatHook: (target) => {
        if (target === cache) cacheLstats += 1;
        if (target === mid && cacheLstats >= 2 && !listed) {
          return {
            isSymbolicLink: () => swapped.symlink === true,
            isDirectory: () => true,
            isFile: () => false,
            dev: 1,
            ino: swapped.ino ?? 1001,
          };
        }
        return null;
      },
      onReaddir: () => {
        listed = true;
      },
    });
    const pool = new ReactionPool();
    assert.throws(
      () => pool.loadFromDirectory(cache, { fsImpl, trustedRootDir: base }),
      /changed during load/,
      label,
    );
    assert.equal(pool.size, 0, label);
  }
});

test("C5 rework: an ancestor swapped while the directory is listed is detected right after the listing", () => {
  const base = path.resolve("/trusted/root");
  const mid = path.join(base, "mid");
  const cache = path.join(mid, "cache");
  const clip = path.join(cache, "subtle_breath_ok.pcm");
  const swaps = [
    { label: "inode changed", swapped: { ino: 999 } },
    { label: "became symlink", swapped: { symlink: true } },
  ];
  for (const { label, swapped } of swaps) {
    let listed = false;
    const { fsImpl } = makeWalkFs({
      dirs: { [base]: {}, [mid]: {}, [cache]: {} },
      files: { [clip]: Buffer.from([1, 2, 3, 4]) },
      // No entry matches the clip pattern, so no per-entry check runs: only the check right
      // after the listing can notice the swap.
      entries: { [cache]: ["notes.txt"] },
      lstatHook: (target) =>
        target === mid && listed
          ? {
              isSymbolicLink: () => swapped.symlink === true,
              isDirectory: () => true,
              isFile: () => false,
              dev: 1,
              ino: swapped.ino ?? 1001,
            }
          : null,
      onReaddir: () => {
        listed = true;
      },
    });
    const pool = new ReactionPool();
    assert.throws(
      () => pool.loadFromDirectory(cache, { fsImpl, trustedRootDir: base }),
      /changed during load/,
      label,
    );
    assert.equal(pool.size, 0, label);
  }
});

test("C5 rework: createCompanionApp creates the private cache directory through the verified walk", async () => {
  const base = path.resolve("/trusted/root");
  const envPath = path.join(base, ".env");
  const envText = [
    "DISCORD_BOT_TOKEN=bot-token",
    "DISCORD_APP_ID=app-1",
    "LLM_PROVIDER=deepseek",
    "LLM_API_KEY=llm-k",
    "STT_PROVIDER=deepgram",
    "DEEPGRAM_API_KEY=dg-k",
    "TTS_PROVIDER=elevenlabs",
    "ELEVENLABS_API_KEY=el-k",
    "ELEVENLABS_VOICE_ID=v-1",
    "REACTION_CACHE_DIR=.cache/reactions",
  ].join("\n");
  const { fsImpl, calls } = makeWalkFs({
    dirs: { [base]: {} },
    files: { [envPath]: Buffer.from(envText) },
  });
  fsImpl.readFileSync = (target) => {
    if (target !== envPath) {
      const err = new Error(`ENOENT: ${target}`);
      err.code = "ENOENT";
      throw err;
    }
    return envText;
  };
  const app = await createCompanionApp({
    env: {},
    envPath,
    configPath: path.join(base, "config.json"),
    soulPath: path.join(base, "SOUL.md"),
    trustedRootDir: base,
    fsImpl,
    createDiscordClientImpl: () => ({ login: async () => "ok", destroy() {} }),
    createLlmClientImpl: () => ({ streamReply: async () => {}, shouldChimeIn: async () => false }),
    createTtsAdapterImpl: () => ({
      name: () => "elevenlabs-ws",
      async warmup() {},
      async synthesizeClause() {},
      close() {},
    }),
    createSttAdapterImpl: () => ({ name: () => "deepgram", async transcribe() {}, close() {} }),
    createVoiceOrchestratorImpl: () => ({ async start() {}, close() {} }),
    createCommandHandlerImpl: () => ({ commands: [], async handleInteraction() {} }),
  });
  assert.deepEqual(calls.mkdir, [
    { path: path.join(base, ".cache"), mode: 0o700 },
    { path: path.join(base, ".cache", "reactions"), mode: 0o700 },
  ]);
  await app.stop();
});

test("C5 rework: registerClip rejects an oversized container from its intrinsic size before any copy is attempted", () => {
  let lengthReads = 0;
  class LyingLength extends Uint8Array {
    get length() {
      lengthReads += 1;
      return 4;
    }
    get byteLength() {
      return 4;
    }
  }
  const pool = new ReactionPool({ maxFileBytes: 16, maxTotalBytes: 64 });
  assert.throws(
    () => pool.registerClip({ id: "subtle_breath_big", mood: "subtle_breath", pcm24k: new LyingLength(64) }),
    /maximum file byte limit/,
  );
  assert.throws(
    () =>
      pool.registerClip({
        id: "subtle_breath_big_frame",
        mood: "subtle_breath",
        frames48kOpus: [new LyingLength(64)],
      }),
    /maximum file byte limit/,
  );
  assert.equal(lengthReads, 0, "the copy path must not run for an over-cap container");
  assert.equal(pool.size, 0);
  assert.equal(pool.totalBytes, 0);
});

test("C5 rework: the streaming OpenAI adapter rejects a RIFF overrun in the first chunk before emitting any audio", async () => {
  const good = buildCustomWavBuffer();
  const overrun = Buffer.concat([good, Buffer.from([5, 6, 7, 8])]);
  const emitted = [];
  const adapter = createOpenAiTtsAdapter({
    apiKey: "k",
    baseUrl: "https://tts.example.com/v1",
    fetchImpl: async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(overrun);
            controller.close();
          },
        }),
        { status: 200 },
      ),
  });
  await assert.rejects(
    adapter.synthesizeClause({ text: "overrun", onPcmChunk: (chunk) => emitted.push(Buffer.from(chunk)) }),
    /RIFF/i,
  );
  assert.equal(emitted.length, 0, "no PCM may reach the caller once the overrun is already known");
});

test("C5 rework: an ancestor swapped between clip reads aborts the remaining load", () => {
  const base = path.resolve("/trusted/root");
  const mid = path.join(base, "mid");
  const cache = path.join(mid, "cache");
  const first = path.join(cache, "subtle_breath_a.pcm");
  const second = path.join(cache, "subtle_breath_b.pcm");
  let swapped = false;
  const { fsImpl } = makeWalkFs({
    dirs: { [base]: {}, [mid]: {}, [cache]: {} },
    files: { [first]: Buffer.from([1, 2, 3, 4]), [second]: Buffer.from([5, 6, 7, 8]) },
    entries: { [cache]: ["subtle_breath_a.pcm", "subtle_breath_b.pcm"] },
    lstatHook: (target) =>
      target === mid && swapped
        ? { isSymbolicLink: () => true, isDirectory: () => true, isFile: () => false, dev: 1, ino: 1001 }
        : null,
    onClose: () => {
      swapped = true;
    },
  });
  const pool = new ReactionPool();
  assert.throws(
    () => pool.loadFromDirectory(cache, { fsImpl, trustedRootDir: base }),
    /changed during load/,
  );
  assert.equal(pool.size, 0, "a clip read after the swap must not be registered");
});

const INTRINSIC_VIEW_BYTE_LENGTH = Object.getOwnPropertyDescriptor(
  Object.getPrototypeOf(Uint8Array.prototype),
  "byteLength",
).get;

test("C5 rework 2: a length-tracking view that grows when its declared length is read cannot retain more than the cap", () => {
  const backing = new ArrayBuffer(4, { maxByteLength: 16 });
  let reads = 0;
  class GrowingView extends Uint8Array {
    get byteLength() {
      reads += 1;
      backing.resize(8);
      return Reflect.apply(INTRINSIC_VIEW_BYTE_LENGTH, this, []);
    }
  }
  const pool = new ReactionPool({ maxFileBytes: 4, maxTotalBytes: 16 });
  assert.throws(
    () => pool.registerClip({ id: "subtle_breath_grow", mood: "subtle_breath", pcm24k: new GrowingView(backing) }),
    RangeError,
  );
  assert.equal(reads, 1);
  assert.equal(pool.size, 0);
  assert.equal(pool.totalBytes, 0);

  const frameBacking = new ArrayBuffer(4, { maxByteLength: 16 });
  let frameReads = 0;
  class GrowingFrame extends Uint8Array {
    get byteLength() {
      frameReads += 1;
      frameBacking.resize(8);
      return Reflect.apply(INTRINSIC_VIEW_BYTE_LENGTH, this, []);
    }
  }
  assert.throws(
    () =>
      pool.registerClip({
        id: "subtle_breath_grow_frame",
        mood: "subtle_breath",
        frames48kOpus: [new GrowingFrame(frameBacking)],
      }),
    RangeError,
  );
  assert.equal(frameReads, 1);
  assert.equal(pool.size, 0);
  assert.equal(pool.totalBytes, 0);
});

test("C5 rework 2: the copy uses the extent saved at snapshot time, so a growth during the declared read is never copied", () => {
  const backing = new ArrayBuffer(4, { maxByteLength: 16 });
  new Uint8Array(backing).set([1, 2, 3, 4]);
  let reads = 0;
  class GrowingQuietly extends Uint8Array {
    get byteLength() {
      reads += 1;
      backing.resize(8);
      return 4;
    }
  }
  const pool = new ReactionPool({ maxFileBytes: 16, maxTotalBytes: 64 });
  assert.equal(
    pool.registerClip({ id: "subtle_breath_quiet", mood: "subtle_breath", pcm24k: new GrowingQuietly(backing) }),
    true,
  );
  assert.equal(reads, 1, "the declared length is read once");
  assert.equal(backing.byteLength, 8, "the buffer really did grow");
  const retained = pool.clips.get("subtle_breath_quiet").pcm24k;
  assert.deepEqual([...retained], [1, 2, 3, 4]);
  assert.equal(pool.totalBytes, 4);
});

test("C5 rework 2: the frames list is read once as a fixed snapshot", () => {
  const frames = [Buffer.from([1, 2]), Buffer.from([3, 4])];
  let lengthReads = 0;
  const watched = new Proxy(frames, {
    get(target, key, receiver) {
      if (key === "length") lengthReads += 1;
      return Reflect.get(target, key, receiver);
    },
  });
  const pool = new ReactionPool({ maxFileBytes: 16, maxTotalBytes: 64 });
  assert.equal(
    pool.registerClip({ id: "subtle_breath_snap", mood: "subtle_breath", frames48kOpus: watched }),
    true,
  );
  assert.equal(lengthReads, 1, "the list length must be read once, then a fixed snapshot is used");
  assert.equal(pool.totalBytes, 4);
});

test("C5 rework 3: a view that shrinks and regrows its backing store is rejected before any copy and its getter runs once", () => {
  const backing = new ArrayBuffer(8, { maxByteLength: 16 });
  let reads = 0;
  class ShrinkRegrow extends Uint8Array {
    get byteLength() {
      reads += 1;
      // First read shrinks the store, so a pinned fixed view goes out of bounds and reports
      // length 0. A second read would regrow it to 8 and let a copy see 8 bytes.
      if (reads === 1) backing.resize(0);
      if (reads === 2) backing.resize(8);
      return Reflect.apply(INTRINSIC_VIEW_BYTE_LENGTH, this, []);
    }
  }
  const pool = new ReactionPool({ maxFileBytes: 4, maxTotalBytes: 16 });
  assert.throws(
    () => pool.registerClip({ id: "subtle_breath_shrink", mood: "subtle_breath", pcm24k: new ShrinkRegrow(backing, 0, 8) }),
    /maximum file byte limit/,
  );
  assert.equal(reads, 1, "caller metadata is read once, before the checks, never while cloning");
  assert.equal(pool.size, 0);
  assert.equal(pool.totalBytes, 0);

  reads = 0;
  backing.resize(8);
  assert.throws(
    () =>
      pool.registerClip({
        id: "subtle_breath_shrink_frame",
        mood: "subtle_breath",
        frames48kOpus: [new ShrinkRegrow(backing, 0, 8)],
      }),
    /maximum file byte limit/,
  );
  assert.equal(reads, 1);
  assert.equal(pool.size, 0);
});

test("C5 rework 3: declared byteLength is captured once, so a value that reports different lengths cannot retarget the copy", () => {
  class ShiftingBytes extends Uint8Array {
    reads = 0;
    get byteLength() {
      this.reads += 1;
      return this.reads === 1 ? 2 : 8;
    }
  }
  const pool = new ReactionPool({ maxFileBytes: 16, maxTotalBytes: 10 });
  assert.equal(
    pool.registerClip({ id: "subtle_breath_kept", mood: "subtle_breath", pcm24k: Buffer.alloc(4, 1) }),
    true,
  );
  const shifting = new ShiftingBytes(8);
  assert.throws(
    () => pool.registerClip({ id: "subtle_breath_shift", mood: "subtle_breath", pcm24k: shifting }),
    /declared byte length does not match/,
  );
  assert.equal(shifting.reads, 1, "the getter must not run again while cloning");
  assert.equal(pool.size, 1);
  assert.equal(pool.totalBytes, 4);
});

test("C5 rework 3: a backing store that shrinks after the extent is saved fails closed instead of retaining a short clone", () => {
  const backing = new ArrayBuffer(8, { maxByteLength: 16 });
  new Uint8Array(backing).set([1, 2, 3, 4, 5, 6, 7, 8]);
  class Shrinker extends Uint8Array {
    get byteLength() {
      // Runs after the first container's extent was measured and saved.
      backing.resize(0);
      return 0;
    }
  }
  const pool = new ReactionPool({ maxFileBytes: 16, maxTotalBytes: 64 });
  assert.throws(
    () =>
      pool.registerClip({
        id: "subtle_breath_shrunk",
        mood: "subtle_breath",
        frames48kOpus: [new Uint8Array(backing, 0, 4), new Shrinker(new ArrayBuffer(0))],
      }),
    RangeError,
  );
  assert.equal(pool.size, 0);
  assert.equal(pool.totalBytes, 0);
});
