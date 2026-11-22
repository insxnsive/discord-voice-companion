import { validateEndpointUrl } from "../config/config.mjs";
import { assertAdapterContract } from "../types.mjs";

const DEFAULT_MAX_AUDIO_BYTES = 4 * 1024 * 1024;
const DEFAULT_TTS_TIMEOUT_MS = 15000;
const SAFE_IDENTIFIER_PATTERN = /^[a-zA-Z0-9._-]{1,128}$/;
const MAX_WAV_SCAN_BYTES = 4096;

function assertSafeIdentifier(value, label) {
  if (typeof value !== "string" || !SAFE_IDENTIFIER_PATTERN.test(value)) {
    throw new RangeError(`${label} must be a non-empty alphanumeric/._- identifier`);
  }
  return value;
}

function createAbortError(signal) {
  if (signal?.reason instanceof Error) return signal.reason;
  const err = new Error("Aborted");
  err.name = "AbortError";
  return err;
}

function combineSignals(primary, secondary) {
  if (!primary) return secondary;
  if (!secondary) return primary;
  return AbortSignal.any([primary, secondary]);
}

function createTimeoutController(timeoutMs, label) {
  const controller = new AbortController();
  const timer = setTimeout(() => {
    const err = new Error(`${label} timed out after ${timeoutMs}ms`);
    err.name = "TimeoutError";
    controller.abort(err);
  }, timeoutMs);
  return {
    signal: controller.signal,
    clear: () => clearTimeout(timer),
  };
}

async function resolveWebSocketClass(WebSocketImpl) {
  if (WebSocketImpl) return WebSocketImpl;
  try {
    const mod = await import("ws");
    return mod.default ?? mod.WebSocket ?? globalThis.WebSocket;
  } catch {
    if (typeof globalThis.WebSocket === "function") return globalThis.WebSocket;
    throw new Error("No WebSocket implementation available (install 'ws' or pass WebSocketImpl)");
  }
}

async function throwHttpError(response) {
  await response.body?.cancel().catch(() => {});
  throw new Error(`TTS request failed (HTTP ${response.status})`);
}

async function readBoundedAudioStream(response, { maxAudioBytes, signal, now, onChunk }) {
  let byteLength = 0;
  let firstByteAt = null;

  if (!response.body) {
    const buf = Buffer.from(await response.arrayBuffer());
    if (buf.length > maxAudioBytes) {
      throw new Error("TTS audio exceeded size limit");
    }
    if (buf.length > 0) {
      firstByteAt = now();
      onChunk(buf);
      byteLength = buf.length;
    }
    return { firstByteAt, byteLength };
  }

  const reader = response.body.getReader();
  try {
    while (true) {
      if (signal?.aborted) {
        throw createAbortError(signal);
      }
      const { value, done } = await reader.read();
      if (done) break;
      const buf = Buffer.isBuffer(value) ? value : Buffer.from(value);
      if (buf.length === 0) continue;

      byteLength += buf.length;
      if (byteLength > maxAudioBytes) {
        await reader.cancel().catch(() => {});
        throw new Error("TTS audio exceeded size limit");
      }

      if (firstByteAt === null) firstByteAt = now();
      onChunk(buf);
    }
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
  }

  return { firstByteAt, byteLength };
}

/**
 * Strip a RIFF/WAVE header (44-byte or extended) and return the raw PCM slice.
 */
export function stripWavHeaderIfPresent(buf) {
  const buffer = Buffer.isBuffer(buf) ? buf : Buffer.from(buf);
  if (
    buffer.length < 12 ||
    buffer.toString("ascii", 0, 4) !== "RIFF" ||
    buffer.toString("ascii", 8, 12) !== "WAVE"
  ) {
    return buffer;
  }

  const scanLimit = Math.min(buffer.length, MAX_WAV_SCAN_BYTES);
  let offset = 12;
  while (offset + 8 <= scanLimit) {
    const chunkId = buffer.toString("ascii", offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    if (chunkId === "data") {
      return buffer.subarray(offset + 8);
    }
    const step = 8 + chunkSize + (chunkSize % 2);
    if (step < 8) break;
    offset += step;
  }
  return buffer;
}

function attachSocketListeners(ws, handlers) {
  const cleanups = [];
  if (typeof ws.on === "function") {
    for (const [event, fn] of Object.entries(handlers)) {
      ws.on(event, fn);
      cleanups.push(() => {
        if (typeof ws.removeListener === "function") ws.removeListener(event, fn);
        else if (typeof ws.off === "function") ws.off(event, fn);
      });
    }
  } else if (typeof ws.addEventListener === "function") {
    for (const [event, fn] of Object.entries(handlers)) {
      const wrapped = (evt) => fn(evt?.data ?? evt);
      ws.addEventListener(event, wrapped);
      cleanups.push(() => ws.removeEventListener(event, wrapped));
    }
  }
  return () => {
    for (const cleanup of cleanups) cleanup();
  };
}

function closeSocket(ws) {
  if (!ws) return;
  try {
    if (typeof ws.close === "function") ws.close(1000);
    else if (typeof ws.terminate === "function") ws.terminate();
  } catch {
    // Ignore close errors during teardown.
  }
}

/**
 * Create an ElevenLabs WebSocket stream-input adapter (default lowest-latency path).
 */
export function createElevenLabsWsAdapter({
  apiKey = "",
  voiceId,
  modelId = "eleven_flash_v2_5",
  maxAudioBytes = DEFAULT_MAX_AUDIO_BYTES,
  timeoutMs = DEFAULT_TTS_TIMEOUT_MS,
  WebSocketImpl,
  now = () => performance.now(),
} = {}) {
  const safeVoiceId = assertSafeIdentifier(voiceId, "tts.voiceId");
  const safeModelId = assertSafeIdentifier(modelId, "tts.model");
  const wsUrl =
    `wss://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(safeVoiceId)}` +
    `/stream-input?model_id=${encodeURIComponent(safeModelId)}&output_format=pcm_24000`;

  let idleSocket = null;
  let activeSocket = null;

  async function openSocket(signal) {
    if (signal?.aborted) throw createAbortError(signal);
    const WsClass = await resolveWebSocketClass(WebSocketImpl);
    if (signal?.aborted) throw createAbortError(signal);

    const ws = new WsClass(wsUrl, {
      headers: { "xi-api-key": apiKey },
    });

    if (ws.readyState === 1) return ws;

    await new Promise((resolve, reject) => {
      let detach = () => {};
      const onAbort = () => {
        detach();
        closeSocket(ws);
        reject(createAbortError(signal));
      };
      detach = attachSocketListeners(ws, {
        open() {
          signal?.removeEventListener("abort", onAbort);
          detach();
          resolve();
        },
        error(err) {
          signal?.removeEventListener("abort", onAbort);
          detach();
          closeSocket(ws);
          reject(err instanceof Error ? err : new Error("TTS WebSocket connection failed"));
        },
        close() {
          signal?.removeEventListener("abort", onAbort);
          detach();
          reject(new Error("TTS WebSocket closed before opening"));
        },
      });
      signal?.addEventListener("abort", onAbort, { once: true });
      if (signal?.aborted) onAbort();
    });

    return ws;
  }

  async function warmup() {
    if (idleSocket && idleSocket.readyState === 1) return;
    const ws = await openSocket();
    idleSocket = ws;
    const clearIdle = attachSocketListeners(ws, {
      close() {
        if (idleSocket === ws) idleSocket = null;
        clearIdle();
      },
      error() {
        if (idleSocket === ws) idleSocket = null;
        clearIdle();
      },
    });
  }

  async function synthesizeClause({ text, signal, onPcmChunk } = {}) {
    if (signal?.aborted) throw createAbortError(signal);

    const deadline = createTimeoutController(timeoutMs, "TTS WebSocket");
    const combinedSignal = combineSignals(signal, deadline.signal);
    let ws = null;

    try {
      if (idleSocket && idleSocket.readyState === 1) {
        ws = idleSocket;
        idleSocket = null;
      } else {
        if (idleSocket) {
          closeSocket(idleSocket);
          idleSocket = null;
        }
        ws = await openSocket(combinedSignal);
      }

      activeSocket = ws;

      return await new Promise((resolve, reject) => {
        let settled = false;
        let firstByteAt = null;
        let byteLength = 0;
        let detach = () => {};

        const finish = (err, result) => {
          if (settled) return;
          settled = true;
          combinedSignal?.removeEventListener("abort", onAbort);
          detach();
          if (activeSocket === ws) activeSocket = null;
          closeSocket(ws);
          if (err) reject(err);
          else resolve(result);
        };

        const onAbort = () => {
          finish(createAbortError(combinedSignal));
        };

        detach = attachSocketListeners(ws, {
          message(raw) {
            let textPayload;
            if (typeof raw === "string") {
              textPayload = raw;
            } else if (Buffer.isBuffer(raw)) {
              textPayload = raw.toString("utf8");
            } else if (raw instanceof ArrayBuffer || ArrayBuffer.isView(raw)) {
              textPayload = Buffer.from(raw.buffer ?? raw).toString("utf8");
            } else {
              return;
            }

            let msg;
            try {
              msg = JSON.parse(textPayload);
            } catch {
              return;
            }

            if (msg?.error) {
              finish(new Error(`ElevenLabs WS error: ${String(msg.error).slice(0, 120)}`));
              return;
            }

            if (typeof msg?.audio === "string" && msg.audio.length > 0) {
              const pcm = Buffer.from(msg.audio, "base64");
              if (pcm.length > 0) {
                byteLength += pcm.length;
                if (byteLength > maxAudioBytes) {
                  finish(new Error("TTS audio exceeded size limit"));
                  return;
                }
                if (firstByteAt === null) firstByteAt = now();
                if (typeof onPcmChunk === "function") onPcmChunk(pcm);
              }
            }

            if (msg?.isFinal === true) {
              finish(null, { firstByteAt, byteLength });
            }
          },
          error(err) {
            finish(err instanceof Error ? err : new Error("TTS WebSocket stream error"));
          },
          close(code) {
            if (settled) return;
            if (code === 1000 && byteLength > 0) {
              finish(null, { firstByteAt, byteLength });
            } else {
              finish(new Error("TTS WebSocket closed before completion"));
            }
          },
        });

        combinedSignal?.addEventListener("abort", onAbort, { once: true });
        if (combinedSignal?.aborted) {
          onAbort();
          return;
        }

        ws.send(
          JSON.stringify({
            text: " ",
            voice_settings: { stability: 0.45, similarity_boost: 0.85 },
          }),
        );
        ws.send(
          JSON.stringify({
            text: `${text} `,
            try_trigger_generation: true,
          }),
        );
        ws.send(JSON.stringify({ text: "" }));
      });
    } finally {
      deadline.clear();
      if (activeSocket === ws) activeSocket = null;
    }
  }

  function close() {
    closeSocket(idleSocket);
    idleSocket = null;
    closeSocket(activeSocket);
    activeSocket = null;
  }

  const adapter = {
    name: () => "elevenlabs-ws",
    synthesizeClause,
    warmup,
    close,
  };
  assertAdapterContract(adapter, ["name", "synthesizeClause", "warmup", "close"], "tts adapter");
  return adapter;
}

/**
 * Create an ElevenLabs chunked HTTP streaming adapter (for eleven_v4 / eleven_v4_turbo).
 */
export function createElevenLabsHttpStreamAdapter({
  apiKey = "",
  voiceId,
  modelId = "eleven_v4_turbo",
  maxAudioBytes = DEFAULT_MAX_AUDIO_BYTES,
  timeoutMs = DEFAULT_TTS_TIMEOUT_MS,
  fetchImpl = fetch,
  now = () => performance.now(),
} = {}) {
  const safeVoiceId = assertSafeIdentifier(voiceId, "tts.voiceId");
  const safeModelId = assertSafeIdentifier(modelId, "tts.model");
  const url =
    `https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(safeVoiceId)}` +
    `/stream?output_format=pcm_24000`;
  const omitStyle = safeModelId.startsWith("eleven_v4") || safeModelId.startsWith("eleven_v3");

  let activeController = null;

  async function warmup() {
    return undefined;
  }

  async function synthesizeClause({ text, signal, onPcmChunk } = {}) {
    if (signal?.aborted) throw createAbortError(signal);
    const deadline = createTimeoutController(timeoutMs, "TTS HTTP stream");
    const requestController = new AbortController();
    activeController = requestController;
    const combinedSignal = combineSignals(combineSignals(signal, deadline.signal), requestController.signal);

    try {
      const response = await fetchImpl(url, {
        method: "POST",
        headers: {
          "xi-api-key": apiKey,
          "Content-Type": "application/json",
          Accept: "audio/pcm",
        },
        body: JSON.stringify({
          text,
          model_id: safeModelId,
          voice_settings: {
            stability: 0.45,
            similarity_boost: 0.85,
            ...(omitStyle ? {} : { style: 0.3 }),
            use_speaker_boost: true,
          },
        }),
        redirect: "error",
        signal: combinedSignal,
      });

      if (!response.ok) {
        await throwHttpError(response);
      }

      return await readBoundedAudioStream(response, {
        maxAudioBytes,
        signal: combinedSignal,
        now,
        onChunk: (buf) => {
          if (typeof onPcmChunk === "function") onPcmChunk(buf);
        },
      });
    } finally {
      deadline.clear();
      if (activeController === requestController) activeController = null;
    }
  }

  function close() {
    activeController?.abort();
    activeController = null;
  }

  const adapter = {
    name: () => "elevenlabs-http-stream",
    synthesizeClause,
    warmup,
    close,
  };
  assertAdapterContract(adapter, ["name", "synthesizeClause", "warmup", "close"], "tts adapter");
  return adapter;
}

/**
 * Create an OpenAI / OpenAI-compatible HTTP TTS streaming adapter.
 */
export function createOpenAiTtsAdapter({
  apiKey = "",
  baseUrl = "https://api.openai.com/v1",
  model = "tts-1",
  voice = "nova",
  maxAudioBytes = DEFAULT_MAX_AUDIO_BYTES,
  timeoutMs = DEFAULT_TTS_TIMEOUT_MS,
  fetchImpl = fetch,
  now = () => performance.now(),
} = {}) {
  const validatedBaseUrl = validateEndpointUrl(baseUrl, "tts.baseUrl");
  if (!validatedBaseUrl) {
    throw new RangeError("tts.baseUrl is required");
  }
  const url = `${validatedBaseUrl}/audio/speech`;

  let activeController = null;

  async function warmup() {
    return undefined;
  }

  async function synthesizeClause({ text, signal, onPcmChunk } = {}) {
    if (signal?.aborted) throw createAbortError(signal);
    const deadline = createTimeoutController(timeoutMs, "OpenAI TTS");
    const requestController = new AbortController();
    activeController = requestController;
    const combinedSignal = combineSignals(combineSignals(signal, deadline.signal), requestController.signal);

    try {
      const headers = {
        "Content-Type": "application/json",
      };
      if (apiKey) {
        headers.Authorization = `Bearer ${apiKey}`;
      }

      const response = await fetchImpl(url, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model,
          voice,
          input: text,
          response_format: "pcm",
        }),
        redirect: "error",
        signal: combinedSignal,
      });

      if (!response.ok) {
        await throwHttpError(response);
      }

      let isFirstChunk = true;
      let emittedBytes = 0;
      let firstByteAt = null;

      await readBoundedAudioStream(response, {
        maxAudioBytes,
        signal: combinedSignal,
        now,
        onChunk: (buf) => {
          const payload = isFirstChunk ? stripWavHeaderIfPresent(buf) : buf;
          isFirstChunk = false;
          if (payload.length === 0) return;
          if (firstByteAt === null) firstByteAt = now();
          emittedBytes += payload.length;
          if (typeof onPcmChunk === "function") onPcmChunk(payload);
        },
      });

      return { firstByteAt, byteLength: emittedBytes };
    } finally {
      deadline.clear();
      if (activeController === requestController) activeController = null;
    }
  }

  function close() {
    activeController?.abort();
    activeController = null;
  }

  const adapter = {
    name: () => "openai-tts",
    synthesizeClause,
    warmup,
    close,
  };
  assertAdapterContract(adapter, ["name", "synthesizeClause", "warmup", "close"], "tts adapter");
  return adapter;
}

/**
 * Instantiate the configured TTS adapter and verify its runtime contract.
 */
export function createTtsAdapter(config, deps = {}) {
  const tts = config?.tts ?? {};
  if (tts.baseUrl) {
    validateEndpointUrl(tts.baseUrl, "tts.baseUrl");
  }

  let adapter;
  if (tts.provider === "elevenlabs") {
    const modelId = String(tts.model || "");
    if (tts.expressiveTags === true || modelId.startsWith("eleven_v4")) {
      adapter = createElevenLabsHttpStreamAdapter({
        apiKey: tts.apiKey,
        voiceId: tts.voiceId,
        modelId: modelId || "eleven_v4_turbo",
        ...deps,
      });
    } else {
      adapter = createElevenLabsWsAdapter({
        apiKey: tts.apiKey,
        voiceId: tts.voiceId,
        modelId: modelId || "eleven_flash_v2_5",
        ...deps,
      });
    }
  } else if (tts.provider === "openai" || tts.provider === "openai-compatible") {
    adapter = createOpenAiTtsAdapter({
      apiKey: tts.apiKey,
      baseUrl: tts.baseUrl || "https://api.openai.com/v1",
      model: tts.model || "tts-1",
      voice: tts.voiceId || "nova",
      ...deps,
    });
  } else {
    throw new RangeError(`Unsupported tts.provider: ${tts.provider}`);
  }

  assertAdapterContract(adapter, ["name", "synthesizeClause", "warmup", "close"], "tts adapter");
  return adapter;
}
