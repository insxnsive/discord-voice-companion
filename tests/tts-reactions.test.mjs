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
  const trailingJunk = Buffer.from("INFO");
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

  const wavWithTail = Buffer.concat([header, pcm, trailingJunk]);
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
