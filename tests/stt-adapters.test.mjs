import test from "node:test";
import assert from "node:assert/strict";

import { assertAdapterContract } from "../src/types.mjs";
import {
  DEFAULT_MAX_PCM_BYTES,
  DEFAULT_MAX_RESPONSE_BYTES,
  createDeepgramSttAdapter,
  createFasterWhisperAdapter,
  createRestWhisperAdapter,
  createScribeSttAdapter,
  createSttAdapter,
  encodeWav16kMono,
  isWhisperHallucination,
} from "../src/stt/adapters.mjs";

const PCM = Buffer.alloc(640, 7);
const CONTRACT = ["name", "transcribe", "close"];

async function tick() {
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
}

function jsonResponse(payload, init = {}) {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "content-type": "application/json" },
    ...init,
  });
}

class MockWebSocket {
  static autoOpen = true;
  static instances = [];

  static reset() {
    MockWebSocket.instances = [];
    MockWebSocket.autoOpen = true;
  }

  static last() {
    return MockWebSocket.instances[MockWebSocket.instances.length - 1];
  }

  constructor(url, options = {}) {
    this.url = String(url);
    this.options = options;
    this.sent = [];
    this.listeners = new Map();
    this.closed = false;
    MockWebSocket.instances.push(this);
    if (MockWebSocket.autoOpen) setImmediate(() => this.emit("open"));
  }

  on(event, handler) {
    const list = this.listeners.get(event) ?? [];
    list.push(handler);
    this.listeners.set(event, list);
    return this;
  }

  once(event, handler) {
    const wrapper = (...args) => {
      this.off(event, wrapper);
      handler(...args);
    };
    return this.on(event, wrapper);
  }

  off(event, handler) {
    const list = this.listeners.get(event) ?? [];
    this.listeners.set(
      event,
      list.filter((entry) => entry !== handler),
    );
    return this;
  }

  addEventListener(event, handler) {
    return this.on(event, handler);
  }

  removeEventListener(event, handler) {
    return this.off(event, handler);
  }

  send(data) {
    this.sent.push(data);
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    setImmediate(() => {
      this.emit("error", new Error("WebSocket was closed before the connection was established"));
      this.emit("close", 1006, "");
    });
  }

  emit(event, ...args) {
    const list = this.listeners.get(event) ?? [];
    if (event === "error" && list.length === 0) {
      throw new Error("Unhandled 'error' event");
    }
    for (const handler of [...list]) handler(...args);
  }
}

test("isWhisperHallucination filters silence artifacts and keeps short speech", () => {
  for (const artifact of [
    "",
    "   ",
    "thank you",
    "Thank you.",
    "Thanks for watching!",
    "you",
    "Bye",
    "[blank_audio]",
    "[silence]",
    "Subtitle by someone",
    "...",
  ]) {
    assert.equal(isWhisperHallucination(artifact), true, artifact);
  }

  for (const speech of ["wait what", "Capri hi", "no, stop", "hello there"]) {
    assert.equal(isWhisperHallucination(speech), false, speech);
  }
});

test("encodeWav16kMono writes a 44-byte RIFF header and rejects bad PCM", () => {
  const pcm = Buffer.alloc(320, 1);
  const wav = encodeWav16kMono(pcm);

  assert.equal(wav.length, 44 + pcm.length);
  assert.equal(wav.subarray(0, 4).toString("ascii"), "RIFF");
  assert.equal(wav.readUInt32LE(4), 36 + pcm.length);
  assert.equal(wav.subarray(8, 12).toString("ascii"), "WAVE");
  assert.equal(wav.subarray(12, 16).toString("ascii"), "fmt ");
  assert.equal(wav.readUInt32LE(16), 16);
  assert.equal(wav.readUInt16LE(20), 1);
  assert.equal(wav.readUInt16LE(22), 1);
  assert.equal(wav.readUInt32LE(24), 16000);
  assert.equal(wav.readUInt32LE(28), 32000);
  assert.equal(wav.readUInt16LE(32), 2);
  assert.equal(wav.readUInt16LE(34), 16);
  assert.equal(wav.subarray(36, 40).toString("ascii"), "data");
  assert.equal(wav.readUInt32LE(40), pcm.length);
  assert.ok(wav.subarray(44).equals(pcm));

  assert.throws(() => encodeWav16kMono(Buffer.alloc(3)), RangeError);
  assert.throws(() => encodeWav16kMono("nope"), TypeError);
  assert.throws(() => encodeWav16kMono(Buffer.alloc(DEFAULT_MAX_PCM_BYTES * 2 + 2)), RangeError);
});

test("createFasterWhisperAdapter posts raw PCM, falls back to multipart, and bounds responses", async () => {
  const calls = [];
  const adapter = createFasterWhisperAdapter({
    baseUrl: "http://127.0.0.1:4373",
    fetchImpl: async (url, options) => {
      calls.push({ url: String(url), options });
      return jsonResponse({ text: "hello there" });
    },
  });

  assert.equal(adapter.name(), "faster-whisper");
  assert.doesNotThrow(() => assertAdapterContract(adapter, CONTRACT, "stt adapter"));
  assert.equal(await adapter.transcribe(PCM), "hello there");

  assert.ok(calls[0].url.endsWith("/v1/audio/pcm"));
  assert.equal(calls[0].options.method, "POST");
  assert.equal(calls[0].options.redirect, "error");
  assert.equal(calls[0].options.headers["content-type"], "application/octet-stream");
  assert.equal(calls[0].options.headers["x-sample-rate"], "16000");
  assert.equal(calls[0].options.headers["x-language"], "en");
  assert.ok(Buffer.from(calls[0].options.body).equals(PCM));

  const fallbackCalls = [];
  const fallback = createFasterWhisperAdapter({
    baseUrl: "http://127.0.0.1:4373",
    fetchImpl: async (url, options) => {
      fallbackCalls.push({ url: String(url), options });
      if (fallbackCalls.length === 1) return new Response("missing", { status: 404 });
      return jsonResponse({ text: "fallback text" });
    },
  });

  assert.equal(await fallback.transcribe(PCM), "fallback text");
  assert.equal(fallbackCalls.length, 2);
  assert.ok(fallbackCalls[1].url.endsWith("/v1/audio/transcriptions"));
  const form = fallbackCalls[1].options.body;
  assert.ok(form instanceof FormData);
  const file = form.get("file");
  assert.ok(file instanceof Blob);
  const wav = Buffer.from(await file.arrayBuffer());
  assert.equal(wav.subarray(0, 4).toString("ascii"), "RIFF");
  assert.equal(wav.length, 44 + PCM.length);

  const oversized = createFasterWhisperAdapter({
    baseUrl: "http://127.0.0.1:4373",
    fetchImpl: async () => new Response(Buffer.alloc(DEFAULT_MAX_RESPONSE_BYTES + 4)),
  });
  await assert.rejects(() => oversized.transcribe(PCM), /exceeded/i);

  const leaky = createFasterWhisperAdapter({
    baseUrl: "http://127.0.0.1:4373",
    fetchImpl: async () => new Response("upstream secret detail", { status: 500 }),
  });
  await assert.rejects(
    () => leaky.transcribe(PCM),
    (error) => {
      assert.match(error.message, /500/);
      assert.equal(error.message.includes("secret"), false);
      return true;
    },
  );
});

test("createRestWhisperAdapter posts multipart to Groq and OpenAI with bounded sanitized reads", async () => {
  const calls = [];
  const groq = createRestWhisperAdapter({
    provider: "groq",
    apiKey: "gsk_test_key",
    fetchImpl: async (url, options) => {
      calls.push({ url: String(url), options });
      return jsonResponse({ text: "from groq" });
    },
  });

  assert.equal(groq.name(), "groq");
  assert.doesNotThrow(() => assertAdapterContract(groq, CONTRACT, "stt adapter"));
  assert.equal(await groq.transcribe(PCM), "from groq");
  assert.equal(calls[0].url, "https://api.groq.com/openai/v1/audio/transcriptions");
  assert.equal(calls[0].options.redirect, "error");
  assert.equal(calls[0].options.headers.Authorization, "Bearer gsk_test_key");
  assert.ok(calls[0].options.body instanceof FormData);
  assert.equal(calls[0].options.body.get("model"), "whisper-large-v3-turbo");

  const openaiCalls = [];
  const openai = createRestWhisperAdapter({
    provider: "openai",
    apiKey: "sk_test_key",
    fetchImpl: async (url, options) => {
      openaiCalls.push({ url: String(url), options });
      return jsonResponse({ text: "thank you" });
    },
  });

  assert.equal(openai.name(), "openai");
  assert.equal(await openai.transcribe(PCM), "");
  assert.equal(openaiCalls[0].url, "https://api.openai.com/v1/audio/transcriptions");
  assert.equal(openaiCalls[0].options.body.get("model"), "whisper-1");

  let zeroCalls = 0;
  const zero = createRestWhisperAdapter({
    provider: "openai",
    apiKey: "sk_test_key",
    fetchImpl: async () => {
      zeroCalls += 1;
      return jsonResponse({ text: "" });
    },
  });
  assert.equal(await zero.transcribe(Buffer.alloc(0)), "");
  assert.equal(zeroCalls, 0);

  const controller = new AbortController();
  let observedSignal;
  const aborting = createRestWhisperAdapter({
    provider: "openai",
    apiKey: "sk_test_key",
    fetchImpl: (url, options) =>
      new Promise((resolve, reject) => {
        observedSignal = options.signal;
        options.signal.addEventListener("abort", () => {
          const error = new Error("aborted by client");
          error.name = "AbortError";
          reject(error);
        });
      }),
  });

  const pending = aborting.transcribe(PCM, { signal: controller.signal });
  await tick();
  controller.abort();
  await assert.rejects(pending, /aborted/i);
  assert.equal(observedSignal.aborted, true);

  const leaky = createRestWhisperAdapter({
    provider: "groq",
    apiKey: "gsk_test_key",
    fetchImpl: async () => new Response("super secret upstream text", { status: 401 }),
  });
  await assert.rejects(
    () => leaky.transcribe(PCM),
    (error) => {
      assert.match(error.message, /401/);
      assert.equal(error.message.includes("secret"), false);
      return true;
    },
  );

  assert.throws(
    () => createRestWhisperAdapter({ provider: "whisper", apiKey: "k" }),
    RangeError,
  );
  assert.throws(
    () => createRestWhisperAdapter({ provider: "openai", apiKey: "" }),
    RangeError,
  );
});

test("createDeepgramSttAdapter streams PCM then Finalize and reads the final transcript", async () => {
  MockWebSocket.reset();
  const adapter = createDeepgramSttAdapter({
    apiKey: "dg_test_key",
    WebSocketImpl: MockWebSocket,
  });

  assert.equal(adapter.name(), "deepgram");
  assert.doesNotThrow(() => assertAdapterContract(adapter, CONTRACT, "stt adapter"));

  const promise = adapter.transcribe(PCM);
  await tick();
  const socket = MockWebSocket.last();

  assert.ok(socket.url.startsWith("wss://api.deepgram.com/v1/listen"));
  const parsed = new URL(socket.url);
  assert.equal(parsed.searchParams.get("model"), "nova-3");
  assert.equal(parsed.searchParams.get("language"), "en");
  assert.equal(parsed.searchParams.get("encoding"), "linear16");
  assert.equal(parsed.searchParams.get("sample_rate"), "16000");
  assert.equal(parsed.searchParams.get("channels"), "1");
  assert.equal(parsed.searchParams.get("interim_results"), "false");
  assert.equal(socket.options.headers.Authorization, "Token dg_test_key");
  assert.equal(socket.options.maxPayload, DEFAULT_MAX_RESPONSE_BYTES);
  assert.ok(Buffer.isBuffer(socket.sent[0]));
  assert.ok(socket.sent[0].equals(PCM));
  assert.deepEqual(JSON.parse(socket.sent[1]), { type: "Finalize" });

  socket.emit(
    "message",
    JSON.stringify({
      is_final: true,
      channel: { alternatives: [{ transcript: " Hello there. " }] },
    }),
  );

  assert.equal(await promise, "Hello there.");
  assert.equal(socket.closed, true);
  assert.doesNotThrow(() => socket.emit("error", new Error("late error after close")));
});

test("createScribeSttAdapter sends base64 audio chunks and awaits the committed transcript", async () => {
  MockWebSocket.reset();
  const adapter = createScribeSttAdapter({
    apiKey: "el_test_key",
    WebSocketImpl: MockWebSocket,
  });

  assert.equal(adapter.name(), "elevenlabs");
  const promise = adapter.transcribe(PCM);
  await tick();
  const socket = MockWebSocket.last();

  assert.ok(socket.url.startsWith("wss://api.elevenlabs.io/v1/speech-to-text/realtime"));
  const parsed = new URL(socket.url);
  assert.equal(parsed.searchParams.get("model_id"), "scribe_v2_realtime");
  assert.equal(parsed.searchParams.get("audio_format"), "pcm_16000");
  assert.equal(parsed.searchParams.get("language_code"), "en");
  assert.equal(socket.options.headers["xi-api-key"], "el_test_key");

  const payload = JSON.parse(socket.sent[0]);
  assert.equal(payload.message_type, "input_audio_chunk");
  assert.equal(payload.audio_base_64, PCM.toString("base64"));
  assert.equal(payload.commit, true);
  assert.equal(payload.sample_rate, 16000);

  socket.emit(
    "message",
    JSON.stringify({ message_type: "committed_transcript", text: "scribe says hi" }),
  );
  assert.equal(await promise, "scribe says hi");
});

test("WebSocket adapters reject insecure URLs, cap frames, and sanitize failures", async () => {
  assert.throws(
    () => createDeepgramSttAdapter({ apiKey: "k", url: "ws://example.com/v1/listen" }),
    RangeError,
  );
  assert.doesNotThrow(() =>
    createDeepgramSttAdapter({ apiKey: "k", url: "ws://127.0.0.1:9999/v1/listen" }),
  );
  assert.throws(() => createDeepgramSttAdapter({ apiKey: "" }), RangeError);
  assert.throws(() => createScribeSttAdapter({ apiKey: "k", url: "ws://scribe.example.com" }), RangeError);

  MockWebSocket.reset();
  const premature = createDeepgramSttAdapter({ apiKey: "k", WebSocketImpl: MockWebSocket });
  const prematurePromise = premature.transcribe(PCM);
  await tick();
  MockWebSocket.last().emit("close", 1000, "done");
  await assert.rejects(prematurePromise, /closed/i);

  MockWebSocket.reset();
  const oversized = createDeepgramSttAdapter({ apiKey: "k", WebSocketImpl: MockWebSocket });
  const oversizedPromise = oversized.transcribe(PCM);
  await tick();
  MockWebSocket.last().emit("message", Buffer.alloc(DEFAULT_MAX_RESPONSE_BYTES + 1));
  await assert.rejects(oversizedPromise, /exceeded/i);

  MockWebSocket.reset();
  const leaky = createDeepgramSttAdapter({ apiKey: "k", WebSocketImpl: MockWebSocket });
  const leakyPromise = leaky.transcribe(PCM);
  await tick();
  MockWebSocket.last().emit("message", JSON.stringify({ error: "top secret upstream" }));
  await assert.rejects(leakyPromise, (error) => {
    assert.equal(error.message.includes("secret"), false);
    return true;
  });

  MockWebSocket.reset();
  MockWebSocket.autoOpen = false;
  const connecting = createDeepgramSttAdapter({ apiKey: "k", WebSocketImpl: MockWebSocket });
  const connectingPromise = connecting.transcribe(PCM);
  await tick();
  const socket = MockWebSocket.last();
  await connecting.close();
  await assert.rejects(connectingPromise, /closed|failed/i);
  assert.doesNotThrow(() => socket.emit("error", new Error("post close error")));
});

test("createSttAdapter maps config.stt keys and rejects unknown providers", () => {
  const stt = {
    deepgramApiKey: "dg_key",
    elevenlabsApiKey: "el_key",
    openaiApiKey: "openai_key",
    groqApiKey: "groq_key",
    localUrl: "http://127.0.0.1:4373",
    language: "en",
  };

  const expectations = [
    ["deepgram", "deepgram"],
    ["elevenlabs", "elevenlabs"],
    ["faster-whisper", "faster-whisper"],
    ["openai", "openai"],
    ["groq", "groq"],
  ];

  for (const [provider, name] of expectations) {
    const adapter = createSttAdapter(
      { stt: { ...stt, provider } },
      {
        fetchImpl: async () => jsonResponse({ text: "ok" }),
        WebSocketImpl: MockWebSocket,
      },
    );
    assert.equal(adapter.name(), name);
    assert.doesNotThrow(() => assertAdapterContract(adapter, CONTRACT, "stt adapter"));
  }

  assert.throws(
    () => createSttAdapter({ stt: { ...stt, provider: "unsupported" } }),
    RangeError,
  );
  assert.throws(
    () => createSttAdapter({ stt: { ...stt, provider: "deepgram", deepgramApiKey: "" } }),
    RangeError,
  );
});
