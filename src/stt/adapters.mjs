// Pluggable STT adapters: Deepgram Nova-3 and ElevenLabs Scribe v2 over
// WebSockets, OpenAI/Groq over REST with an in-memory WAV, and a local
// faster-whisper PCM fast path. No temp files, no key in any URL or message.

import { validateEndpointUrl } from "../config/config.mjs";
import { assertAdapterContract } from "../types.mjs";

export const DEFAULT_MAX_PCM_BYTES = 2 * 1024 * 1024;
export const MAX_PCM_BYTES_LIMIT = 4 * 1024 * 1024;
export const DEFAULT_MAX_RESPONSE_BYTES = 256 * 1024;
export const DEFAULT_STT_TIMEOUT_MS = 15000;

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "::1", "[::1]"]);
const CONTROL_CHAR_PATTERN = /[\u0000-\u001f\u007f]/;
const HALLUCINATIONS = new Set([
  "thank you",
  "thanks for watching",
  "you",
  "bye",
  "[blank_audio]",
  "[silence]",
  "...",
]);

class SttRequestError extends Error {}

function assertPositiveLimit(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive integer`);
  }
}

function assertMaxPcmBytes(value) {
  if (!Number.isSafeInteger(value) || value <= 0 || value > MAX_PCM_BYTES_LIMIT) {
    throw new RangeError(`maxPcmBytes must be a positive integer <= ${MAX_PCM_BYTES_LIMIT}`);
  }
}

function toBuffer(input) {
  if (Buffer.isBuffer(input)) return input;
  return Buffer.from(input.buffer, input.byteOffset, input.byteLength);
}

function assertValidPcm16kMono(pcm16kMono, maxPcmBytes = DEFAULT_MAX_PCM_BYTES) {
  assertMaxPcmBytes(maxPcmBytes);
  if (!Buffer.isBuffer(pcm16kMono) && !(pcm16kMono instanceof Uint8Array)) {
    throw new TypeError("pcm16kMono must be a Buffer or Uint8Array");
  }
  if (pcm16kMono.byteLength % 2 !== 0) {
    throw new RangeError("pcm16kMono must contain whole 16-bit samples");
  }
  if (pcm16kMono.byteLength > maxPcmBytes) {
    throw new RangeError(`pcm16kMono exceeds the ${maxPcmBytes} byte limit`);
  }
  return pcm16kMono.byteLength;
}

function assertWebSocketUrl(rawUrl, label = "url") {
  const text = String(rawUrl ?? "");
  if (!text || CONTROL_CHAR_PATTERN.test(text)) {
    throw new RangeError(`${label} contains invalid characters`);
  }
  let parsed;
  try {
    parsed = new URL(text);
  } catch {
    throw new RangeError(`${label} must be a valid URL`);
  }
  if (parsed.username || parsed.password) {
    throw new RangeError(`${label} must not contain credentials`);
  }
  if (parsed.hash) {
    throw new RangeError(`${label} must not contain a fragment`);
  }
  if (parsed.protocol === "wss:") return parsed;
  if (parsed.protocol === "ws:" && LOOPBACK_HOSTS.has(parsed.hostname.toLowerCase())) {
    return parsed;
  }
  throw new RangeError(`${label} must use wss: (ws: is allowed only on loopback hosts)`);
}

/** True for empty text and the usual Whisper silence artifacts. */
export function isWhisperHallucination(text) {
  if (typeof text !== "string") return true;
  const normalized = text
    .trim()
    .toLowerCase()
    .replace(/^[.,!?;:\s]+/, "")
    .replace(/[.,!?;:\s]+$/, "");
  if (!normalized) return true;
  if (HALLUCINATIONS.has(normalized)) return true;
  return normalized.startsWith("subtitle by");
}

/** Build a 16kHz mono 16-bit WAV entirely in memory, with a 44-byte header. */
export function encodeWav16kMono(pcm16kMono) {
  if (!Buffer.isBuffer(pcm16kMono) && !(pcm16kMono instanceof Uint8Array)) {
    throw new TypeError("pcm16kMono must be a Buffer or Uint8Array");
  }
  if (pcm16kMono.byteLength % 2 !== 0) {
    throw new RangeError("pcm16kMono must contain whole 16-bit samples");
  }
  if (pcm16kMono.byteLength > MAX_PCM_BYTES_LIMIT) {
    throw new RangeError(`pcm16kMono exceeds the ${MAX_PCM_BYTES_LIMIT} byte limit`);
  }

  const pcm = toBuffer(pcm16kMono);
  const header = Buffer.alloc(44);
  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVE", 8, "ascii");
  header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(16000, 24);
  header.writeUInt32LE(32000, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36, "ascii");
  header.writeUInt32LE(pcm.length, 40);

  return Buffer.concat([header, pcm]);
}

async function cancelBody(res) {
  try {
    await res?.body?.cancel?.();
  } catch {
    // Cancelling is best effort; the response is discarded either way.
  }
}

async function readBoundedBody(res, maxResponseBytes) {
  if (res?.body && typeof res.body.getReader === "function") {
    const reader = res.body.getReader();
    const chunks = [];
    let total = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = Buffer.isBuffer(value) ? value : toBuffer(value);
        total += chunk.byteLength;
        if (total > maxResponseBytes) {
          await reader.cancel?.().catch?.(() => {});
          throw new RangeError("STT response exceeded the maximum allowed size");
        }
        chunks.push(chunk);
      }
    } finally {
      reader.releaseLock?.();
    }
    return Buffer.concat(chunks).toString("utf8");
  }

  if (typeof res?.arrayBuffer === "function") {
    const buffer = Buffer.from(await res.arrayBuffer());
    if (buffer.byteLength > maxResponseBytes) {
      throw new RangeError("STT response exceeded the maximum allowed size");
    }
    return buffer.toString("utf8");
  }

  const text = typeof res?.text === "function" ? await res.text() : "";
  if (Buffer.byteLength(text, "utf8") > maxResponseBytes) {
    throw new RangeError("STT response exceeded the maximum allowed size");
  }
  return text;
}

async function runHttpRequest({
  fetchImpl,
  url,
  options,
  label,
  signal,
  timeoutMs,
  handleResponse,
}) {
  if (signal?.aborted) throw new SttRequestError(`${label} aborted`);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const onAbort = () => controller.abort();
  signal?.addEventListener("abort", onAbort, { once: true });

  try {
    const res = await fetchImpl(url, {
      ...options,
      signal: controller.signal,
      redirect: "error",
    });
    return await handleResponse(res);
  } catch (error) {
    if (error instanceof SttRequestError || error instanceof RangeError || error instanceof TypeError) {
      throw error;
    }
    if (controller.signal.aborted) {
      throw new SttRequestError(signal?.aborted ? `${label} aborted` : `${label} timed out`);
    }
    throw new SttRequestError(`${label} request failed`);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

function parseTranscript(text, label) {
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    throw new SttRequestError(`${label} returned invalid JSON`);
  }
  const transcript = typeof payload?.text === "string" ? payload.text : "";
  return isWhisperHallucination(transcript) ? "" : transcript.trim();
}

function onSocket(socket, event, handler) {
  if (typeof socket.on === "function") {
    socket.on(event, handler);
    return () => socket.off?.(event, handler);
  }
  if (typeof socket.addEventListener === "function") {
    socket.addEventListener(event, handler);
    return () => socket.removeEventListener?.(event, handler);
  }
  return () => {};
}

function attachPermanentErrorHandler(socket) {
  const noop = () => {};
  if (typeof socket.on === "function") socket.on("error", noop);
  else if (typeof socket.addEventListener === "function") socket.addEventListener("error", noop);
}

function socketMessageLength(data) {
  if (typeof data === "string") return Buffer.byteLength(data, "utf8");
  if (data instanceof ArrayBuffer) return data.byteLength;
  if (ArrayBuffer.isView(data)) return data.byteLength;
  return 0;
}

function socketMessageText(data) {
  if (typeof data === "string") return data;
  if (data instanceof ArrayBuffer) return Buffer.from(data).toString("utf8");
  if (ArrayBuffer.isView(data)) {
    return Buffer.from(data.buffer, data.byteOffset, data.byteLength).toString("utf8");
  }
  return "";
}

async function resolveWebSocketImpl(impl) {
  if (impl) return impl;
  const module = await import("ws");
  return module.WebSocket ?? module.default;
}

async function transcribeOverSocket({
  label,
  socketUrl,
  headers,
  WebSocketImpl,
  pcm16kMono,
  signal,
  timeoutMs,
  maxResponseBytes,
  buildMessages,
  readFinal,
  registerSocket,
}) {
  if (signal?.aborted) throw new SttRequestError(`${label} aborted`);

  const Impl = await resolveWebSocketImpl(WebSocketImpl);
  const socket = new Impl(socketUrl, { headers, maxPayload: maxResponseBytes });
  attachPermanentErrorHandler(socket);
  registerSocket?.(socket);

  return await new Promise((resolve, reject) => {
    let settled = false;
    let timer;
    const detach = [];

    const finish = (error, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      for (const off of detach) {
        try {
          off();
        } catch {
          // Listener removal is best effort.
        }
      }
      registerSocket?.(null);
      try {
        socket.close?.();
      } catch {
        // Closing a socket that already failed is expected.
      }
      if (error) reject(error);
      else resolve(value);
    };

    const onAbort = () => finish(new SttRequestError(`${label} aborted`));
    timer = setTimeout(() => finish(new SttRequestError(`${label} timed out`)), timeoutMs);
    signal?.addEventListener("abort", onAbort, { once: true });

    detach.push(
      onSocket(socket, "open", () => {
        try {
          for (const message of buildMessages(pcm16kMono)) socket.send(message);
        } catch {
          finish(new SttRequestError(`${label} failed to send audio`));
        }
      }),
    );

    detach.push(
      onSocket(socket, "message", (data) => {
        if (socketMessageLength(data) > maxResponseBytes) {
          finish(new SttRequestError(`${label} response exceeded the size limit`));
          return;
        }
        let parsed;
        try {
          parsed = JSON.parse(socketMessageText(data));
        } catch {
          return;
        }
        if (parsed && typeof parsed === "object" && parsed.error) {
          finish(new SttRequestError(`${label} returned an error`));
          return;
        }
        const transcript = readFinal(parsed);
        if (transcript === undefined) return;
        finish(null, isWhisperHallucination(transcript) ? "" : String(transcript).trim());
      }),
    );

    detach.push(onSocket(socket, "error", () => finish(new SttRequestError(`${label} failed`))));
    detach.push(
      onSocket(socket, "close", () =>
        finish(new SttRequestError(`${label} closed before a final transcript`)),
      ),
    );
  });
}

export function createDeepgramSttAdapter({
  apiKey,
  model = "nova-3",
  language = "en",
  url,
  maxPcmBytes = DEFAULT_MAX_PCM_BYTES,
  maxResponseBytes = DEFAULT_MAX_RESPONSE_BYTES,
  timeoutMs = DEFAULT_STT_TIMEOUT_MS,
  WebSocketImpl,
} = {}) {
  assertMaxPcmBytes(maxPcmBytes);
  assertPositiveLimit(maxResponseBytes, "maxResponseBytes");
  assertPositiveLimit(timeoutMs, "timeoutMs");
  if (typeof apiKey !== "string" || apiKey.trim().length === 0) {
    throw new RangeError("Deepgram STT adapter requires an API key");
  }

  const target = assertWebSocketUrl(url ?? "wss://api.deepgram.com/v1/listen", "stt.deepgramUrl");
  target.searchParams.set("model", model);
  target.searchParams.set("language", language);
  target.searchParams.set("encoding", "linear16");
  target.searchParams.set("sample_rate", "16000");
  target.searchParams.set("channels", "1");
  target.searchParams.set("interim_results", "false");
  const socketUrl = target.toString();

  let activeSocket = null;

  return {
    name() {
      return "deepgram";
    },
    async transcribe(pcm16kMono, { signal } = {}) {
      const byteLength = assertValidPcm16kMono(pcm16kMono, maxPcmBytes);
      if (byteLength === 0) return "";
      return await transcribeOverSocket({
        label: "Deepgram STT",
        socketUrl,
        headers: { Authorization: `Token ${apiKey}` },
        WebSocketImpl,
        pcm16kMono,
        signal,
        timeoutMs,
        maxResponseBytes,
        buildMessages: (pcm) => [pcm, JSON.stringify({ type: "Finalize" })],
        readFinal: (parsed) => {
          if (!parsed || (parsed.is_final !== true && parsed.speech_final !== true)) {
            return undefined;
          }
          const transcript = parsed.channel?.alternatives?.[0]?.transcript;
          return typeof transcript === "string" ? transcript : "";
        },
        registerSocket: (socket) => {
          activeSocket = socket;
        },
      });
    },
    close() {
      const socket = activeSocket;
      activeSocket = null;
      try {
        socket?.close?.();
      } catch {
        // Idempotent shutdown.
      }
    },
  };
}

export function createScribeSttAdapter({
  apiKey,
  modelId = "scribe_v2_realtime",
  languageCode = "en",
  url,
  maxPcmBytes = DEFAULT_MAX_PCM_BYTES,
  maxResponseBytes = DEFAULT_MAX_RESPONSE_BYTES,
  timeoutMs = DEFAULT_STT_TIMEOUT_MS,
  WebSocketImpl,
} = {}) {
  assertMaxPcmBytes(maxPcmBytes);
  assertPositiveLimit(maxResponseBytes, "maxResponseBytes");
  assertPositiveLimit(timeoutMs, "timeoutMs");
  if (typeof apiKey !== "string" || apiKey.trim().length === 0) {
    throw new RangeError("ElevenLabs STT adapter requires an API key");
  }

  const target = assertWebSocketUrl(
    url ?? "wss://api.elevenlabs.io/v1/speech-to-text/realtime",
    "stt.elevenlabsUrl",
  );
  target.searchParams.set("model_id", modelId);
  target.searchParams.set("audio_format", "pcm_16000");
  target.searchParams.set("language_code", languageCode);
  const socketUrl = target.toString();

  let activeSocket = null;

  return {
    name() {
      return "elevenlabs";
    },
    async transcribe(pcm16kMono, { signal } = {}) {
      const byteLength = assertValidPcm16kMono(pcm16kMono, maxPcmBytes);
      if (byteLength === 0) return "";
      return await transcribeOverSocket({
        label: "ElevenLabs STT",
        socketUrl,
        headers: { "xi-api-key": apiKey },
        WebSocketImpl,
        pcm16kMono,
        signal,
        timeoutMs,
        maxResponseBytes,
        buildMessages: (pcm) => [
          JSON.stringify({
            message_type: "input_audio_chunk",
            audio_base_64: toBuffer(pcm).toString("base64"),
            commit: true,
            sample_rate: 16000,
          }),
        ],
        readFinal: (parsed) => {
          if (!parsed) return undefined;
          if (
            parsed.message_type !== "committed_transcript" &&
            parsed.message_type !== "final_transcript"
          ) {
            return undefined;
          }
          return typeof parsed.text === "string" ? parsed.text : "";
        },
        registerSocket: (socket) => {
          activeSocket = socket;
        },
      });
    },
    close() {
      const socket = activeSocket;
      activeSocket = null;
      try {
        socket?.close?.();
      } catch {
        // Idempotent shutdown.
      }
    },
  };
}

export function createFasterWhisperAdapter({
  baseUrl = "http://127.0.0.1:4373",
  language = "en",
  prompt = "",
  maxPcmBytes = DEFAULT_MAX_PCM_BYTES,
  maxResponseBytes = DEFAULT_MAX_RESPONSE_BYTES,
  timeoutMs = DEFAULT_STT_TIMEOUT_MS,
  fetchImpl = fetch,
} = {}) {
  assertMaxPcmBytes(maxPcmBytes);
  assertPositiveLimit(maxResponseBytes, "maxResponseBytes");
  assertPositiveLimit(timeoutMs, "timeoutMs");

  const base = validateEndpointUrl(baseUrl, "stt.localUrl");
  const primaryUrl = `${base}/v1/audio/pcm`;
  const fallbackUrl = `${base}/v1/audio/transcriptions`;

  async function transcribe(pcm16kMono, { signal } = {}) {
    const byteLength = assertValidPcm16kMono(pcm16kMono, maxPcmBytes);
    if (byteLength === 0) return "";

    const headers = {
      "content-type": "application/octet-stream",
      "x-sample-rate": "16000",
      "x-language": language,
    };
    if (prompt) headers["x-prompt"] = prompt;

    const primary = await runHttpRequest({
      fetchImpl,
      url: primaryUrl,
      options: { method: "POST", headers, body: pcm16kMono },
      label: "faster-whisper STT",
      signal,
      timeoutMs,
      handleResponse: async (res) => {
        if (res.status === 404 || res.status === 405) {
          await cancelBody(res);
          return { fallback: true };
        }
        if (!res.ok) {
          await cancelBody(res);
          throw new SttRequestError(`faster-whisper STT request failed with status ${res.status}`);
        }
        return { fallback: false, text: await readBoundedBody(res, maxResponseBytes) };
      },
    });

    if (!primary.fallback) return parseTranscript(primary.text, "faster-whisper STT");

    const form = new FormData();
    form.append("file", new Blob([encodeWav16kMono(pcm16kMono)], { type: "audio/wav" }), "audio.wav");
    form.append("language", language);
    if (prompt) form.append("prompt", prompt);

    const text = await runHttpRequest({
      fetchImpl,
      url: fallbackUrl,
      options: { method: "POST", body: form },
      label: "faster-whisper STT",
      signal,
      timeoutMs,
      handleResponse: async (res) => {
        if (!res.ok) {
          await cancelBody(res);
          throw new SttRequestError(`faster-whisper STT request failed with status ${res.status}`);
        }
        return await readBoundedBody(res, maxResponseBytes);
      },
    });

    return parseTranscript(text, "faster-whisper STT");
  }

  return {
    name() {
      return "faster-whisper";
    },
    transcribe,
    close() {
      // No persistent connection to release.
    },
  };
}

const REST_PRESETS = Object.freeze({
  openai: { baseUrl: "https://api.openai.com/v1", model: "whisper-1" },
  groq: { baseUrl: "https://api.groq.com/openai/v1", model: "whisper-large-v3-turbo" },
});

export function createRestWhisperAdapter({
  provider = "openai",
  apiKey,
  baseUrl,
  model,
  language = "en",
  maxPcmBytes = DEFAULT_MAX_PCM_BYTES,
  maxResponseBytes = DEFAULT_MAX_RESPONSE_BYTES,
  timeoutMs = DEFAULT_STT_TIMEOUT_MS,
  fetchImpl = fetch,
} = {}) {
  assertMaxPcmBytes(maxPcmBytes);
  assertPositiveLimit(maxResponseBytes, "maxResponseBytes");
  assertPositiveLimit(timeoutMs, "timeoutMs");

  const preset = REST_PRESETS[provider];
  if (!preset) {
    throw new RangeError(`unsupported REST STT provider: ${String(provider)}`);
  }
  if (typeof apiKey !== "string" || apiKey.trim().length === 0) {
    throw new RangeError(`${provider} STT adapter requires an API key`);
  }

  const base = validateEndpointUrl(baseUrl ?? preset.baseUrl, "stt.baseUrl");
  const resolvedModel = model ?? preset.model;
  const endpoint = `${base}/audio/transcriptions`;

  async function transcribe(pcm16kMono, { signal } = {}) {
    const byteLength = assertValidPcm16kMono(pcm16kMono, maxPcmBytes);
    if (byteLength === 0) return "";

    const form = new FormData();
    form.append("file", new Blob([encodeWav16kMono(pcm16kMono)], { type: "audio/wav" }), "audio.wav");
    form.append("model", resolvedModel);
    form.append("language", language);

    const text = await runHttpRequest({
      fetchImpl,
      url: endpoint,
      options: {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}` },
        body: form,
      },
      label: `${provider} STT`,
      signal,
      timeoutMs,
      handleResponse: async (res) => {
        if (!res.ok) {
          await cancelBody(res);
          throw new SttRequestError(`${provider} STT request failed with status ${res.status}`);
        }
        return await readBoundedBody(res, maxResponseBytes);
      },
    });

    return parseTranscript(text, `${provider} STT`);
  }

  return {
    name() {
      return provider;
    },
    transcribe,
    close() {
      // No persistent connection to release.
    },
  };
}

function compact(entries) {
  return Object.fromEntries(
    Object.entries(entries).filter(([, value]) => value !== undefined && value !== ""),
  );
}

export function createSttAdapter(config = {}, deps = {}) {
  const stt = config?.stt ?? {};
  let adapter;

  switch (stt.provider) {
    case "deepgram":
      adapter = createDeepgramSttAdapter({
        ...deps,
        ...compact({
          apiKey: stt.deepgramApiKey,
          model: stt.model,
          language: stt.language,
        }),
      });
      break;
    case "elevenlabs":
      adapter = createScribeSttAdapter({
        ...deps,
        ...compact({
          apiKey: stt.elevenlabsApiKey,
          modelId: stt.model,
          languageCode: stt.language,
        }),
      });
      break;
    case "faster-whisper":
      adapter = createFasterWhisperAdapter({
        ...deps,
        ...compact({
          baseUrl: stt.localUrl,
          language: stt.language,
          prompt: stt.prompt,
        }),
      });
      break;
    case "openai":
      adapter = createRestWhisperAdapter({
        ...deps,
        ...compact({
          provider: "openai",
          apiKey: stt.openaiApiKey,
          model: stt.model,
          language: stt.language,
        }),
      });
      break;
    case "groq":
      adapter = createRestWhisperAdapter({
        ...deps,
        ...compact({
          provider: "groq",
          apiKey: stt.groqApiKey,
          model: stt.model,
          language: stt.language,
        }),
      });
      break;
    default:
      throw new RangeError(`unsupported STT provider: ${String(stt.provider)}`);
  }

  assertAdapterContract(adapter, ["name", "transcribe", "close"], "stt adapter");
  return adapter;
}
