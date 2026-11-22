import { validateEndpointUrl } from "../config/config.mjs";
import { assertAdapterContract } from "../types.mjs";

const DEFAULT_MAX_AUDIO_BYTES = 4 * 1024 * 1024;
const MAX_AUDIO_BYTES_LIMIT = 16 * 1024 * 1024;
const MAX_WS_PAYLOAD_BYTES = 4 * 1024 * 1024;
const DEFAULT_TTS_TIMEOUT_MS = 15000;
const SAFE_IDENTIFIER_PATTERN = /^[a-zA-Z0-9._-]{1,128}$/;
const MAX_WAV_SCAN_BYTES = 512;

function assertValidMaxAudioBytes(maxAudioBytes) {
  if (
    !Number.isSafeInteger(maxAudioBytes) ||
    maxAudioBytes <= 0 ||
    maxAudioBytes > MAX_AUDIO_BYTES_LIMIT
  ) {
    throw new RangeError(`maxAudioBytes must be a positive integer <= ${MAX_AUDIO_BYTES_LIMIT}`);
  }
  return maxAudioBytes;
}

function getRawByteLength(raw) {
  if (typeof raw === "string") return Buffer.byteLength(raw, "utf8");
  if (Buffer.isBuffer(raw)) return raw.length;
  if (raw instanceof ArrayBuffer || ArrayBuffer.isView(raw)) return raw.byteLength;
  return -1;
}

function createPcmEvenEmitter(onPcmChunk) {
  let carry = null;
  let emittedBytes = 0;

  return {
    push(raw) {
      const buf = Buffer.isBuffer(raw) ? raw : Buffer.from(raw);
      if (buf.length === 0) return 0;

      let data = buf;
      if (carry !== null) {
        data = Buffer.allocUnsafe(1 + buf.length);
        data[0] = carry;
        buf.copy(data, 1);
        carry = null;
      }

      const evenLen = data.length - (data.length % 2);
      if (data.length % 2 === 1) {
        carry = data[data.length - 1];
      }
      if (evenLen >= 2) {
        const slice = evenLen === data.length ? data : data.subarray(0, evenLen);
        emittedBytes += slice.length;
        if (typeof onPcmChunk === "function") onPcmChunk(slice);
        return slice.length;
      }
      return 0;
    },
    flush() {
      if (carry !== null) {
        throw new Error("TTS PCM stream ended with incomplete sample frame");
      }
      return emittedBytes;
    },
    get emittedBytes() {
      return emittedBytes;
    },
  };
}

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

      if (byteLength + buf.length > maxAudioBytes) {
        await reader.cancel().catch(() => {});
        throw new Error("TTS audio exceeded size limit");
      }
      byteLength += buf.length;

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
 * Validate and strip a RIFF/WAVE header (44-byte or extended) and return the raw PCM slice.
 */
export function stripWavHeaderIfPresent(buf) {
  const buffer = Buffer.isBuffer(buf) ? buf : Buffer.from(buf);
  if (buffer.length < 4 || buffer.toString("ascii", 0, 4) !== "RIFF") {
    return buffer;
  }
  if (buffer.length < 12 || buffer.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error("Invalid WAV header");
  }

  const scanLimit = Math.min(buffer.length, MAX_WAV_SCAN_BYTES);
  let offset = 12;
  let sawFmt = false;
  while (offset + 8 <= scanLimit) {
    const chunkId = buffer.toString("ascii", offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    if (chunkId === "data") {
      if (!sawFmt) {
        throw new Error("WAV header missing fmt chunk before data");
      }
      const dataStart = offset + 8;
      if (chunkSize !== 0xffffffff) {
        if (dataStart + chunkSize > buffer.length) {
          throw new Error("Truncated WAV data chunk");
        }
        return buffer.subarray(dataStart, dataStart + chunkSize);
      }
      return buffer.subarray(dataStart);
    }
    const step = 8 + chunkSize + (chunkSize % 2);
    if (chunkSize > MAX_WAV_SCAN_BYTES || step < 8) {
      throw new Error("Invalid WAV chunk size");
    }
    if (chunkId === "fmt ") {
      if (chunkSize < 16 || offset + 8 + chunkSize > scanLimit) {
        throw new Error("Invalid or truncated WAV fmt chunk");
      }
      const audioFormat = buffer.readUInt16LE(offset + 8);
      const bitsPerSample = buffer.readUInt16LE(offset + 22);
      if (audioFormat !== 1 || bitsPerSample !== 16) {
        throw new Error("Unsupported non-PCM WAV format");
      }
      sawFmt = true;
    }
    offset += step;
  }
  throw new Error("Truncated or unsupported WAV header");
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
  const safeMaxAudioBytes = assertValidMaxAudioBytes(maxAudioBytes);
  const maxPayload = Math.min(safeMaxAudioBytes, MAX_WS_PAYLOAD_BYTES);
  const maxWsMessageBytes = Math.min(
    MAX_WS_PAYLOAD_BYTES,
    Math.ceil((safeMaxAudioBytes * 4) / 3) + 65536,
  );
  const safeVoiceId = assertSafeIdentifier(voiceId, "tts.voiceId");
  const safeModelId = assertSafeIdentifier(modelId, "tts.model");
  const wsUrl =
    `wss://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(safeVoiceId)}` +
    `/stream-input?model_id=${encodeURIComponent(safeModelId)}&output_format=pcm_24000`;

  let idleSocket = null;
  let activeSocket = null;
  let activeAbortController = null;

  function attachErrorTrap(ws) {
    const onErrorTrap = () => {};
    let cleanupClose = () => {};
    const onCloseTrap = () => {
      cleanupClose();
      if (typeof ws.removeListener === "function") ws.removeListener("error", onErrorTrap);
      else if (typeof ws.off === "function") ws.off("error", onErrorTrap);
    };
    if (typeof ws.on === "function") {
      ws.on("error", onErrorTrap);
      ws.on("close", onCloseTrap);
      cleanupClose = () => {
        if (typeof ws.removeListener === "function") ws.removeListener("close", onCloseTrap);
        else if (typeof ws.off === "function") ws.off("close", onCloseTrap);
      };
    } else if (typeof ws.addEventListener === "function") {
      ws.addEventListener("error", onErrorTrap);
      ws.addEventListener("close", onCloseTrap);
      cleanupClose = () => {
        ws.removeEventListener("close", onCloseTrap);
        ws.removeEventListener("error", onErrorTrap);
      };
    }
  }

  async function openSocket(signal) {
    if (signal?.aborted) throw createAbortError(signal);
    const WsClass = await resolveWebSocketClass(WebSocketImpl);
    if (signal?.aborted) throw createAbortError(signal);

    const ws = new WsClass(wsUrl, {
      headers: { "xi-api-key": apiKey },
      maxPayload,
    });
    attachErrorTrap(ws);

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
        error() {
          signal?.removeEventListener("abort", onAbort);
          detach();
          closeSocket(ws);
          reject(new Error("TTS WebSocket connection failed"));
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
    const deadline = createTimeoutController(timeoutMs, "TTS WebSocket warmup");
    try {
      const ws = await openSocket(deadline.signal);
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
    } finally {
      deadline.clear();
    }
  }

  async function synthesizeClause({ text, signal, onPcmChunk } = {}) {
    if (signal?.aborted) throw createAbortError(signal);

    const deadline = createTimeoutController(timeoutMs, "TTS WebSocket");
    const closeController = new AbortController();
    activeAbortController = closeController;
    const combinedSignal = combineSignals(combineSignals(signal, deadline.signal), closeController.signal);
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
      const emitter = createPcmEvenEmitter(onPcmChunk);

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
            const rawBytes = getRawByteLength(raw);
            if (rawBytes < 0) return;
            if (rawBytes > maxWsMessageBytes) {
              finish(new Error("TTS WebSocket payload exceeded size limit"));
              return;
            }

            let textPayload;
            if (typeof raw === "string") {
              textPayload = raw;
            } else if (Buffer.isBuffer(raw)) {
              textPayload = raw.toString("utf8");
            } else {
              textPayload = Buffer.from(raw.buffer ?? raw, raw.byteOffset ?? 0, raw.byteLength).toString("utf8");
            }

            let msg;
            try {
              msg = JSON.parse(textPayload);
            } catch {
              return;
            }

            if (msg?.error) {
              finish(new Error("TTS WebSocket upstream error"));
              return;
            }

            if (typeof msg?.audio === "string" && msg.audio.length > 0) {
              const remainingBytes = safeMaxAudioBytes - byteLength;
              const maxBase64Len = Math.ceil((Math.max(0, remainingBytes) * 4) / 3) + 4;
              if (msg.audio.length > maxBase64Len) {
                finish(new Error("TTS audio exceeded size limit"));
                return;
              }
              const pcm = Buffer.from(msg.audio, "base64");
              if (pcm.length > 0) {
                if (byteLength + pcm.length > safeMaxAudioBytes) {
                  finish(new Error("TTS audio exceeded size limit"));
                  return;
                }
                byteLength += pcm.length;
                if (firstByteAt === null) firstByteAt = now();
                try {
                  emitter.push(pcm);
                } catch (cbErr) {
                  finish(cbErr instanceof Error ? cbErr : new Error("TTS WebSocket chunk callback failed"));
                  return;
                }
              }
            }

            if (msg?.isFinal === true) {
              try {
                emitter.flush();
              } catch (frameErr) {
                finish(frameErr);
                return;
              }
              finish(null, { firstByteAt, byteLength: emitter.emittedBytes });
            }
          },
          error() {
            finish(new Error("TTS WebSocket stream error"));
          },
          close() {
            finish(new Error("TTS WebSocket closed before completion"));
          },
        });

        combinedSignal?.addEventListener("abort", onAbort, { once: true });
        if (combinedSignal?.aborted) {
          onAbort();
          return;
        }

        try {
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
        } catch (sendErr) {
          finish(sendErr instanceof Error ? sendErr : new Error("TTS WebSocket send failed"));
        }
      });
    } finally {
      deadline.clear();
      if (activeAbortController === closeController) activeAbortController = null;
      if (activeSocket === ws) activeSocket = null;
    }
  }

  function close() {
    if (activeAbortController) {
      activeAbortController.abort(new Error("TTS WebSocket closed"));
      activeAbortController = null;
    }
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
  const safeMaxAudioBytes = assertValidMaxAudioBytes(maxAudioBytes);
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

      const emitter = createPcmEvenEmitter(onPcmChunk);
      const result = await readBoundedAudioStream(response, {
        maxAudioBytes: safeMaxAudioBytes,
        signal: combinedSignal,
        now,
        onChunk: (buf) => {
          emitter.push(buf);
        },
      });
      emitter.flush();
      return result;
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
  const safeMaxAudioBytes = assertValidMaxAudioBytes(maxAudioBytes);
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

      let headerResolved = false;
      let headerBuf = Buffer.alloc(0);
      let remainingDataBytes = Infinity;
      let firstByteAt = null;
      const emitter = createPcmEvenEmitter((chunk) => {
        if (firstByteAt === null) firstByteAt = now();
        if (typeof onPcmChunk === "function") onPcmChunk(chunk);
      });

      const pushBoundedPcm = (buf) => {
        if (buf.length === 0 || remainingDataBytes === 0) return;
        const slice = Number.isFinite(remainingDataBytes)
          ? buf.subarray(0, Math.min(buf.length, remainingDataBytes))
          : buf;
        if (Number.isFinite(remainingDataBytes)) {
          remainingDataBytes -= slice.length;
        }
        if (slice.length > 0) emitter.push(slice);
      };

      const tryResolveHeader = () => {
        const prefixLen = Math.min(4, headerBuf.length);
        const expectedPrefix = "RIFF".slice(0, prefixLen);
        if (headerBuf.toString("ascii", 0, prefixLen) !== expectedPrefix) {
          headerResolved = true;
          const raw = headerBuf;
          headerBuf = null;
          emitter.push(raw);
          return;
        }
        if (headerBuf.length < 12) return;
        if (headerBuf.toString("ascii", 8, 12) !== "WAVE") {
          throw new Error("Invalid WAV header");
        }

        let offset = 12;
        let sawFmt = false;
        while (offset + 8 <= headerBuf.length) {
          const chunkId = headerBuf.toString("ascii", offset, offset + 4);
          const chunkSize = headerBuf.readUInt32LE(offset + 4);
          if (chunkId === "data") {
            if (!sawFmt) {
              throw new Error("WAV header missing fmt chunk before data");
            }
            remainingDataBytes = chunkSize !== 0xffffffff ? chunkSize : Infinity;
            headerResolved = true;
            const pcmStart = headerBuf.subarray(offset + 8);
            headerBuf = null;
            pushBoundedPcm(pcmStart);
            return;
          }
          const step = 8 + chunkSize + (chunkSize % 2);
          if (chunkSize > MAX_WAV_SCAN_BYTES || step < 8) {
            throw new Error("Invalid WAV chunk size");
          }
          if (chunkId === "fmt ") {
            if (chunkSize < 16) {
              throw new Error("Invalid WAV fmt chunk");
            }
            if (offset + step > headerBuf.length) {
              if (headerBuf.length > MAX_WAV_SCAN_BYTES) {
                throw new Error("Truncated WAV fmt chunk");
              }
              return;
            }
            const audioFormat = headerBuf.readUInt16LE(offset + 8);
            const bitsPerSample = headerBuf.readUInt16LE(offset + 22);
            if (audioFormat !== 1 || bitsPerSample !== 16) {
              throw new Error("Unsupported non-PCM WAV format");
            }
            sawFmt = true;
          } else if (offset + step > headerBuf.length) {
            if (headerBuf.length > MAX_WAV_SCAN_BYTES) {
              throw new Error("WAV header exceeded maximum scan size");
            }
            return;
          }
          offset += step;
        }

        if (headerBuf.length > MAX_WAV_SCAN_BYTES) {
          throw new Error("WAV header exceeded maximum scan size");
        }
      };

      await readBoundedAudioStream(response, {
        maxAudioBytes: safeMaxAudioBytes,
        signal: combinedSignal,
        now,
        onChunk: (buf) => {
          if (headerResolved) {
            pushBoundedPcm(buf);
            return;
          }
          headerBuf = headerBuf.length === 0 ? buf : Buffer.concat([headerBuf, buf]);
          tryResolveHeader();
        },
      });

      if (!headerResolved && headerBuf !== null && headerBuf.length > 0) {
        const prefixLen = Math.min(4, headerBuf.length);
        if (headerBuf.toString("ascii", 0, prefixLen) === "RIFF".slice(0, prefixLen)) {
          throw new Error("Truncated WAV header");
        }
        emitter.push(headerBuf);
        headerBuf = null;
      }
      if (headerResolved && Number.isFinite(remainingDataBytes) && remainingDataBytes > 0) {
        throw new Error("Truncated WAV data chunk");
      }
      emitter.flush();

      return { firstByteAt, byteLength: emitter.emittedBytes };
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
