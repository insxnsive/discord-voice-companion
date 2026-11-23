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
    // A body that is not a stream cannot be bounded: arrayBuffer() allocates the whole payload
    // before any cap can run, and a declared content-length is caller-controlled metadata
    // that does not limit that allocation. Only a readable stream is accepted.
    throw new Error("TTS response body is not a stream, so its size cannot be bounded");
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

function assertValidPcmWavFmt(buffer, offset, expectedSampleRate = 24000) {
  const audioFormat = buffer.readUInt16LE(offset + 8);
  const numChannels = buffer.readUInt16LE(offset + 10);
  const wavSampleRate = buffer.readUInt32LE(offset + 12);
  const blockAlign = buffer.readUInt16LE(offset + 20);
  const bitsPerSample = buffer.readUInt16LE(offset + 22);
  if (
    audioFormat !== 1 ||
    numChannels !== 1 ||
    wavSampleRate !== expectedSampleRate ||
    blockAlign !== 2 ||
    bitsPerSample !== 16
  ) {
    throw new Error(`Unsupported non-PCM or non-${expectedSampleRate}Hz-mono WAV format`);
  }
}

/**
 * Validate and strip a RIFF/WAVE header (44-byte or extended) and return the raw PCM slice.
 */
export function stripWavHeaderIfPresent(buf, { sampleRate = 24000 } = {}) {
  const buffer = Buffer.isBuffer(buf) ? buf : Buffer.from(buf);
  if (buffer.length < 4 || buffer.toString("ascii", 0, 4) !== "RIFF") {
    return buffer;
  }
  if (buffer.length < 12 || buffer.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error("Invalid WAV header");
  }

  // The whole buffer is known here, so the declared RIFF container size must match it exactly
  // once the data chunk is located (truncation is reported first). 0xffffffff is the
  // streaming sentinel and skips the comparison.
  const riffSize = buffer.readUInt32LE(4);
  const assertRiffMatchesBuffer = () => {
    if (riffSize !== 0xffffffff && riffSize + 8 !== buffer.length) {
      throw new Error("WAV RIFF length does not match buffer size");
    }
  };

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
        assertRiffMatchesBuffer();
        return buffer.subarray(dataStart, dataStart + chunkSize);
      }
      assertRiffMatchesBuffer();
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
      assertValidPcmWavFmt(buffer, offset, sampleRate);
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
  let currentVoiceId = assertSafeIdentifier(voiceId, "tts.voiceId");
  const safeModelId = assertSafeIdentifier(modelId, "tts.model");
  const buildWsUrl = (vid) =>
    `wss://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(vid)}` +
    `/stream-input?model_id=${encodeURIComponent(safeModelId)}&output_format=pcm_24000`;

  let idleSocket = null;
  let idleVoiceId = null;
  let activeSocket = null;
  let activeAbortController = null;
  let warmupControllers = new Set();
  let warmupPromise = null;
  let closed = false;
  let turnToken = null;
  let turnVoiceId = null;
  let pendingVoiceId = null;

  function effectiveVoiceId() {
    return turnToken ? turnVoiceId : currentVoiceId;
  }

  function closeIdleSocket() {
    if (idleSocket) {
      try {
        closeSocket(idleSocket);
      } catch {
        // The idle socket is discarded either way.
      }
      idleSocket = null;
    }
    idleVoiceId = null;
  }

  function applyVoiceId(nextVoiceId) {
    currentVoiceId = nextVoiceId;
    closeIdleSocket();
  }

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

  async function openSocket(signal, voiceId = effectiveVoiceId()) {
    if (signal?.aborted) throw createAbortError(signal);
    const WsClass = await resolveWebSocketClass(WebSocketImpl);
    if (signal?.aborted) throw createAbortError(signal);

    const ws = new WsClass(buildWsUrl(voiceId), {
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

  function warmup() {
    if (closed) {
      return Promise.reject(new Error("TTS WebSocket adapter is closed"));
    }
    if (idleSocket && idleSocket.readyState === 1 && idleVoiceId === currentVoiceId) {
      return Promise.resolve();
    }
    if (warmupPromise) return warmupPromise;
    warmupPromise = runWarmup().finally(() => {
      warmupPromise = null;
    });
    return warmupPromise;
  }

  async function runWarmup() {
    const deadline = createTimeoutController(timeoutMs, "TTS WebSocket warmup");
    const warmupCloseController = new AbortController();
    warmupControllers.add(warmupCloseController);
    const warmupSignal = combineSignals(deadline.signal, warmupCloseController.signal);
    try {
      for (;;) {
        const voiceForWarmup = currentVoiceId;
        const ws = await openSocket(warmupSignal, voiceForWarmup);
        if (closed) {
          closeSocket(ws);
          throw new Error("TTS WebSocket adapter is closed");
        }
        if (currentVoiceId !== voiceForWarmup) {
          closeSocket(ws);
          continue;
        }
        if (idleSocket && idleSocket !== ws) {
          if (idleSocket.readyState === 1) {
            closeSocket(ws);
            return;
          }
          closeSocket(idleSocket);
          idleSocket = null;
          idleVoiceId = null;
        }
        idleSocket = ws;
        idleVoiceId = voiceForWarmup;
        const clearIdle = attachSocketListeners(ws, {
          close() {
            if (idleSocket === ws) {
              idleSocket = null;
              idleVoiceId = null;
            }
            clearIdle();
          },
          error() {
            if (idleSocket === ws) {
              idleSocket = null;
              idleVoiceId = null;
            }
            clearIdle();
          },
        });
        return;
      }
    } finally {
      warmupControllers.delete(warmupCloseController);
      deadline.clear();
    }
  }

  async function synthesizeClause({ text, signal, onPcmChunk } = {}) {
    if (closed) {
      throw new Error("TTS WebSocket adapter is closed");
    }
    if (signal?.aborted) throw createAbortError(signal);

    const deadline = createTimeoutController(timeoutMs, "TTS WebSocket");
    const closeController = new AbortController();
    activeAbortController = closeController;
    const combinedSignal = combineSignals(combineSignals(signal, deadline.signal), closeController.signal);
    let ws = null;

    const voiceForClause = effectiveVoiceId();
    try {
      if (idleSocket && idleSocket.readyState === 1 && idleVoiceId === voiceForClause) {
        ws = idleSocket;
        idleSocket = null;
        idleVoiceId = null;
      } else {
        if (idleSocket) {
          closeSocket(idleSocket);
          idleSocket = null;
          idleVoiceId = null;
        }
        ws = await openSocket(combinedSignal, voiceForClause);
        if (closed) {
          closeSocket(ws);
          throw new Error("TTS WebSocket adapter is closed");
        }
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
    closed = true;
    for (const controller of warmupControllers) {
      controller.abort(new Error("TTS WebSocket closed"));
    }
    warmupControllers.clear();
    if (activeAbortController) {
      activeAbortController.abort(new Error("TTS WebSocket closed"));
      activeAbortController = null;
    }
    closeSocket(idleSocket);
    idleSocket = null;
    idleVoiceId = null;
    closeSocket(activeSocket);
    activeSocket = null;
    turnToken = null;
    turnVoiceId = null;
    pendingVoiceId = null;
  }

  const adapter = {
    name: () => "elevenlabs-ws",
    synthesizeClause,
    warmup,
    close,
    beginTurn() {
      // A selection made during a turn is queued, not applied. The next
      // beginTurn adopts it as that turn's pin, and a stale endTurn token can
      // neither clear the current turn nor consume its queued selection.
      const token = {};
      if (turnToken && pendingVoiceId && pendingVoiceId !== currentVoiceId) {
        applyVoiceId(pendingVoiceId);
      }
      pendingVoiceId = null;
      turnToken = token;
      turnVoiceId = currentVoiceId;
      return token;
    },
    endTurn(token) {
      if (!token || token !== turnToken) return false;
      turnToken = null;
      turnVoiceId = null;
      const pending = pendingVoiceId;
      pendingVoiceId = null;
      if (pending && pending !== currentVoiceId) {
        applyVoiceId(pending);
        return true;
      }
      return false;
    },
    setVoiceId(nextVoiceId) {
      const safe = assertSafeIdentifier(nextVoiceId, "tts.voiceId");
      if (!turnToken) {
        if (safe === currentVoiceId) return false;
        applyVoiceId(safe);
        return true;
      }
      if (safe === turnVoiceId) {
        pendingVoiceId = null;
        return false;
      }
      pendingVoiceId = safe;
      return true;
    },
    getVoiceId() {
      return turnToken ? turnVoiceId : currentVoiceId;
    },
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
      // Outer RIFF length is only known for streamed input once the header arrives, so it is
      // enforced against the running total and again at end of stream. A 0xffffffff declared
      // size is the streaming sentinel and keeps only the bounded header scan.
      let declaredRiffEnd = null;
      let receivedBytes = 0;
      const assertWithinDeclaredRiff = () => {
        if (declaredRiffEnd !== null && receivedBytes > declaredRiffEnd) {
          throw new Error("WAV RIFF length exceeded by streamed data");
        }
      };
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
        if (declaredRiffEnd === null) {
          const riffSize = headerBuf.readUInt32LE(4);
          if (riffSize !== 0xffffffff) declaredRiffEnd = riffSize + 8;
        }
        // The overrun may already be known from the first chunk. Reject it here, before the
        // header scan hands any PCM from that chunk to the emitter.
        assertWithinDeclaredRiff();

        const scanLimit = Math.min(headerBuf.length, MAX_WAV_SCAN_BYTES);
        let offset = 12;
        let sawFmt = false;
        while (offset + 8 <= scanLimit) {
          const chunkId = headerBuf.toString("ascii", offset, offset + 4);
          const chunkSize = headerBuf.readUInt32LE(offset + 4);
          if (chunkId === "data") {
            if (!sawFmt) {
              throw new Error("WAV header missing fmt chunk before data");
            }
            if (
              declaredRiffEnd !== null &&
              chunkSize !== 0xffffffff &&
              offset + 8 + chunkSize > declaredRiffEnd
            ) {
              throw new Error("WAV data chunk exceeds RIFF length");
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
            if (offset + step > scanLimit) {
              if (headerBuf.length > MAX_WAV_SCAN_BYTES) {
                throw new Error("Truncated WAV fmt chunk");
              }
              return;
            }
            assertValidPcmWavFmt(headerBuf, offset, 24000);
            sawFmt = true;
          } else if (offset + step > scanLimit) {
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

      const streamResult = await readBoundedAudioStream(response, {
        maxAudioBytes: safeMaxAudioBytes,
        signal: combinedSignal,
        now,
        onChunk: (buf) => {
          receivedBytes += buf.length;
          if (headerResolved) {
            assertWithinDeclaredRiff();
            pushBoundedPcm(buf);
            return;
          }
          headerBuf = headerBuf.length === 0 ? buf : Buffer.concat([headerBuf, buf]);
          tryResolveHeader();
          assertWithinDeclaredRiff();
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
      if (declaredRiffEnd !== null && streamResult.byteLength !== declaredRiffEnd) {
        throw new Error("WAV RIFF length does not match streamed size");
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

class VoicesRequestError extends Error {}

// Intrinsic typed-array accessors. A chunk can shadow byteLength, length,
// buffer and byteOffset, so the real extent is read from %TypedArray%.prototype.
const TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(Uint8Array.prototype);
const INTRINSIC_BYTE_LENGTH = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTOTYPE,
  "byteLength",
).get;
const INTRINSIC_BYTE_OFFSET = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTOTYPE,
  "byteOffset",
).get;
const INTRINSIC_BUFFER = Object.getOwnPropertyDescriptor(TYPED_ARRAY_PROTOTYPE, "buffer").get;

function measureStreamChunk(value) {
  if (!ArrayBuffer.isView(value)) return null;
  try {
    return {
      buffer: Reflect.apply(INTRINSIC_BUFFER, value, []),
      offset: Reflect.apply(INTRINSIC_BYTE_OFFSET, value, []),
      extent: Reflect.apply(INTRINSIC_BYTE_LENGTH, value, []),
    };
  } catch {
    return null;
  }
}

async function readBoundedText(res, maxBytes) {
  const body = res?.body;
  if (!body || typeof body.getReader !== "function") {
    throw new VoicesRequestError("ElevenLabs voices request failed");
  }
  const reader = body.getReader();
  const chunks = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      const measured = measureStreamChunk(value);
      if (!measured || !Number.isInteger(measured.extent) || measured.extent < 0) {
        throw new VoicesRequestError("ElevenLabs voices request failed");
      }
      if (total + measured.extent > maxBytes) {
        try {
          await reader.cancel();
        } catch {
          // Cancellation is best effort; the owned limit error still stands.
        }
        throw new VoicesRequestError("ElevenLabs voices response exceeded the size limit");
      }
      if (measured.extent === 0) continue;

      // Copy exactly the intrinsic extent, never the caller-visible metadata.
      const source = Buffer.from(measured.buffer, measured.offset, measured.extent);
      const chunk = Buffer.allocUnsafe(measured.extent);
      const copied = source.copy(chunk, 0, 0, measured.extent);
      if (copied !== measured.extent) {
        throw new VoicesRequestError("ElevenLabs voices request failed");
      }
      total += copied;
      if (total > maxBytes) {
        throw new VoicesRequestError("ElevenLabs voices response exceeded the size limit");
      }
      chunks.push(chunk);
    }
  } finally {
    try {
      reader.releaseLock?.();
    } catch {
      // Releasing a cancelled reader is best effort.
    }
  }
  return Buffer.concat(chunks, total).toString("utf8");
}

/** List the account's ElevenLabs voices. The key stays in the header only. */
export async function fetchElevenLabsVoices({
  apiKey = "",
  fetchImpl = fetch,
  timeoutMs = 8000,
  maxResponseBytes = 256 * 1024,
} = {}) {
  if (typeof apiKey !== "string" || apiKey.trim().length === 0) {
    throw new VoicesRequestError("ElevenLabs API key is required to list voices");
  }
  if (/[\r\n\0]/.test(apiKey)) {
    throw new VoicesRequestError("ElevenLabs API key contains invalid characters");
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetchImpl("https://api.elevenlabs.io/v1/voices", {
      method: "GET",
      headers: { "xi-api-key": apiKey, accept: "application/json" },
      signal: controller.signal,
      redirect: "error",
    });
    if (!res.ok) {
      try {
        await res.body?.cancel?.();
      } catch {
        // Best effort.
      }
      throw new VoicesRequestError(`ElevenLabs voices request failed with status ${res.status}`);
    }
    const payload = JSON.parse(await readBoundedText(res, maxResponseBytes));
    const voices = Array.isArray(payload?.voices) ? payload.voices : [];
    return voices
      .filter((voice) => voice && typeof voice.voice_id === "string" && voice.voice_id.length > 0)
      .map((voice) => ({
        voiceId: voice.voice_id,
        name:
          typeof voice.name === "string" && voice.name.trim().length > 0
            ? voice.name.trim()
            : voice.voice_id,
      }));
  } catch (error) {
    if (error instanceof VoicesRequestError) throw error;
    if (error?.name === "AbortError") {
      throw new VoicesRequestError("ElevenLabs voices request timed out");
    }
    throw new VoicesRequestError("ElevenLabs voices request failed");
  } finally {
    clearTimeout(timer);
  }
}
