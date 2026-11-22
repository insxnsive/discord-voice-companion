import { validateEndpointUrl } from "../config/config.mjs";

const AUDIO_TAG_BEFORE_PUNCT = /\s*\[[a-z][a-z0-9 _-]{1,32}\](?=\s*[.,!?;:])/gi;
const AUDIO_TAG_INLINE = /\s*\[[a-z][a-z0-9 _-]{1,32}\]\s*/gi;

const DEFAULT_REPLY_TIMEOUT_MS = 15000;
const DEFAULT_MAX_STREAM_BYTES = 262144;
const DEFAULT_MAX_LINE_CHARS = 16384;
const DEFAULT_MAX_REPLY_CHARS = 8192;
const MAX_GATE_BYTES = 16384;

/**
 * Remove bracketed performance tags like [laughs] or [whispers] while keeping
 * surrounding spacing and punctuation intact.
 */
export function stripAudioTags(text) {
  if (typeof text !== "string") return "";
  return text
    .replace(AUDIO_TAG_BEFORE_PUNCT, "")
    .replace(AUDIO_TAG_INLINE, " ")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

function countWords(text) {
  const speakable = stripAudioTags(text);
  return speakable ? speakable.split(/\s+/).length : 0;
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function findPunctuationSplit(text) {
  let depth = 0;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === "[") {
      depth += 1;
      continue;
    }
    if (ch === "]") {
      if (depth > 0) depth -= 1;
      continue;
    }
    if (depth !== 0) continue;

    if (ch === "." || ch === "!" || ch === "?" || ch === "\n") {
      let end = i + 1;
      while (end < text.length && (text[end] === "." || text[end] === "!" || text[end] === "?" || text[end] === "\n")) {
        end += 1;
      }
      return end;
    }

    if (ch === "," || ch === ";") {
      let end = i + 1;
      while (end < text.length && (text[end] === "," || text[end] === ";")) {
        end += 1;
      }
      if (countWords(text.slice(0, end)) >= 3) {
        return end;
      }
    }
  }
  return -1;
}

function analyzeBracketDepth(text) {
  let depth = 0;
  let lastDepth0Ws = -1;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === "[") {
      depth += 1;
      continue;
    }
    if (ch === "]") {
      if (depth > 0) depth -= 1;
      continue;
    }
    if (depth === 0 && /\s/.test(ch)) {
      lastDepth0Ws = i;
    }
  }
  return { depth, lastDepth0Ws };
}

/**
 * Buffer streamed LLM text deltas into speakable clauses without splitting
 * inside pending audio tags like [soft laugh].
 */
export function createClauseSplitter({
  minWords = 5,
  maxHoldMs = 120,
  expressiveTags = false,
  now = () => performance.now(),
  onClause,
} = {}) {
  let buffer = "";
  let clauseStartAt = null;

  function emitClause(rawClause, emitted) {
    const prepared = expressiveTags ? rawClause : stripAudioTags(rawClause);
    const cleaned = prepared.replace(/[ \t]{2,}/g, " ").trim();
    if (!cleaned) return;
    emitted.push(cleaned);
    if (typeof onClause === "function") onClause(cleaned);
  }

  function extractReady(emitted) {
    while (buffer.length > 0) {
      const punctEnd = findPunctuationSplit(buffer);
      if (punctEnd > 0) {
        const head = buffer.slice(0, punctEnd);
        buffer = buffer.slice(punctEnd).replace(/^[ \t]+/, "");
        emitClause(head, emitted);
        clauseStartAt = buffer.length > 0 ? now() : null;
        continue;
      }

      const elapsed = clauseStartAt === null ? 0 : now() - clauseStartAt;
      if (elapsed >= maxHoldMs) {
        const { depth, lastDepth0Ws } = analyzeBracketDepth(buffer);
        if (depth === 0) {
          if ((/\s$/.test(buffer) || /\]$/.test(buffer)) && countWords(buffer) >= minWords) {
            const head = buffer;
            buffer = "";
            emitClause(head, emitted);
            clauseStartAt = null;
            continue;
          }

          if (lastDepth0Ws > 0) {
            const head = buffer.slice(0, lastDepth0Ws);
            if (countWords(head) >= minWords) {
              buffer = buffer.slice(lastDepth0Ws + 1).replace(/^[ \t]+/, "");
              emitClause(head, emitted);
              clauseStartAt = buffer.length > 0 ? now() : null;
              continue;
            }
          }
        }
      }

      break;
    }
  }

  return {
    push(delta) {
      if (typeof delta !== "string" || delta.length === 0) return [];
      if (buffer.length === 0) {
        const trimmedStart = delta.replace(/^[ \t]+/, "");
        if (!trimmedStart) return [];
        buffer = trimmedStart;
        clauseStartAt = now();
      } else {
        buffer += delta;
        if (clauseStartAt === null) clauseStartAt = now();
      }

      const emitted = [];
      extractReady(emitted);
      return emitted;
    },

    flush() {
      if (!buffer) return [];
      const tail = buffer;
      buffer = "";
      clauseStartAt = null;
      const emitted = [];
      emitClause(tail, emitted);
      return emitted;
    },
  };
}

function combineSignals(primary, secondary) {
  if (!primary) return secondary;
  if (!secondary) return primary;
  return AbortSignal.any([primary, secondary]);
}

function isAliasAddressed(transcript, botAliases) {
  if (!Array.isArray(botAliases) || botAliases.length === 0) return false;
  return botAliases.some((alias) => {
    const clean = typeof alias === "string" ? alias.trim() : "";
    if (!clean) return false;
    return new RegExp(`\\b${escapeRegExp(clean)}\\b`, "i").test(transcript);
  });
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

async function readBoundedJson(response, maxBytes = MAX_GATE_BYTES) {
  if (!response.body) {
    const buf = await response.arrayBuffer();
    if (buf.byteLength > maxBytes) {
      throw new Error("Gate response exceeded size limit");
    }
    return JSON.parse(new TextDecoder("utf-8").decode(buf));
  }

  const reader = response.body.getReader();
  const chunks = [];
  let totalBytes = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      totalBytes += value.byteLength;
      if (totalBytes > maxBytes) {
        await reader.cancel().catch(() => {});
        throw new Error("Gate response exceeded size limit");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const combined = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return JSON.parse(new TextDecoder("utf-8").decode(combined));
}

/**
 * Create an OpenAI-compatible streaming reply and 1-token turn-gate client.
 */
export function createLlmClient({
  config,
  fetchImpl = fetch,
  now = () => performance.now(),
  maxStreamBytes = DEFAULT_MAX_STREAM_BYTES,
  maxLineChars = DEFAULT_MAX_LINE_CHARS,
  maxReplyChars = DEFAULT_MAX_REPLY_CHARS,
} = {}) {
  const llm = config?.llm;
  if (!llm?.baseUrl) {
    throw new Error("LLM client requires config.llm.baseUrl");
  }
  if (!llm?.replyModel) {
    throw new Error("LLM client requires config.llm.replyModel");
  }

  const validatedBaseUrl = validateEndpointUrl(llm.baseUrl, "llm.baseUrl");
  const endpoint = `${validatedBaseUrl}/chat/completions`;
  const expressiveTags = Boolean(config?.tts?.expressiveTags);
  const defaultChime = config?.chime ?? {};
  const replyTimeoutMs = llm.replyTimeoutMs ?? DEFAULT_REPLY_TIMEOUT_MS;

  function buildRequestInit({ model, messages, stream, maxTokens, temperature, signal }) {
    const headers = {
      "Content-Type": "application/json",
      ...(llm.headers ?? {}),
    };
    if (llm.apiKey) {
      headers.Authorization = `Bearer ${llm.apiKey}`;
    }

    const payload = {
      ...(llm.extraBody ?? {}),
      model,
      messages,
      stream,
      max_tokens: maxTokens,
    };
    if (temperature !== undefined) {
      payload.temperature = temperature;
    }

    return {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      redirect: "error",
      signal,
    };
  }

  async function streamReply({ messages, signal, onClause, onToken } = {}) {
    const deadline = createTimeoutController(replyTimeoutMs, "LLM stream");
    const streamSignal = combineSignals(signal, deadline.signal);

    let response;
    try {
      response = await fetchImpl(
        endpoint,
        buildRequestInit({
          model: llm.replyModel,
          messages,
          stream: true,
          maxTokens: llm.maxTokens ?? 140,
          signal: streamSignal,
        }),
      );
    } catch (error) {
      deadline.clear();
      throw error;
    }

    if (!response.ok) {
      deadline.clear();
      await response.body?.cancel().catch(() => {});
      throw new Error(`LLM stream failed (HTTP ${response.status})`);
    }
    if (!response.body) {
      deadline.clear();
      throw new Error("LLM stream response missing body");
    }

    const clauses = [];
    let firstClauseAt = null;
    const splitter = createClauseSplitter({
      expressiveTags,
      now,
      onClause(clause) {
        if (firstClauseAt === null) firstClauseAt = now();
        clauses.push(clause);
        if (typeof onClause === "function") onClause(clause);
      },
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let carry = "";
    let fullText = "";
    let totalBytes = 0;
    let done = false;

    const appendDelta = (delta) => {
      if (typeof delta !== "string" || delta.length === 0) return;
      if (fullText.length + delta.length > maxReplyChars) {
        throw new Error("LLM stream exceeded size limit");
      }
      fullText += delta;
      if (typeof onToken === "function") onToken(delta);
      splitter.push(delta);
    };

    try {
      while (!done) {
        if (streamSignal?.aborted) {
          throw streamSignal.reason ?? Object.assign(new Error("Aborted"), { name: "AbortError" });
        }

        const { value, done: streamDone } = await reader.read();
        if (streamDone) break;

        totalBytes += value.byteLength;
        if (totalBytes > maxStreamBytes) {
          throw new Error("LLM stream exceeded size limit");
        }

        carry += decoder.decode(value, { stream: true });
        const lines = carry.split(/\r?\n/);
        carry = lines.pop() ?? "";
        if (carry.length > maxLineChars) {
          throw new Error("LLM stream exceeded size limit");
        }

        for (const rawLine of lines) {
          if (rawLine.length > maxLineChars) {
            throw new Error("LLM stream exceeded size limit");
          }
          const line = rawLine.trim();
          if (!line || line.startsWith(":")) continue;
          if (!line.startsWith("data:")) continue;

          const data = line.slice(5).trim();
          if (data === "[DONE]") {
            done = true;
            await reader.cancel().catch(() => {});
            break;
          }

          let chunk;
          try {
            chunk = JSON.parse(data);
          } catch {
            continue;
          }

          appendDelta(chunk?.choices?.[0]?.delta?.content);
        }
      }

      if (!done && carry.trim().startsWith("data:")) {
        if (carry.length > maxLineChars) {
          throw new Error("LLM stream exceeded size limit");
        }
        const tailData = carry.trim().slice(5).trim();
        if (tailData && tailData !== "[DONE]") {
          let chunk = null;
          try {
            chunk = JSON.parse(tailData);
          } catch {
            // Ignore incomplete trailing JSON fragment at stream EOF.
          }
          if (chunk) {
            appendDelta(chunk?.choices?.[0]?.delta?.content);
          }
        }
      }
    } catch (error) {
      await reader.cancel().catch(() => {});
      throw error;
    } finally {
      deadline.clear();
      reader.releaseLock();
    }

    splitter.flush();

    return { fullText, clauses, firstClauseAt };
  }

  async function shouldChimeIn({
    transcript,
    speakerName = "Speaker",
    botAliases = defaultChime.botAliases ?? [],
    lastBotReplyAt = null,
    followUpWindowMs = defaultChime.followUpWindowMs ?? 12000,
    chimeMode = defaultChime.mode ?? "smart",
    gateTimeoutMs = defaultChime.gateTimeoutMs ?? 350,
    signal,
    now: clock = () => Date.now(),
  } = {}) {
    const cleanTranscript = typeof transcript === "string" ? transcript.trim() : "";
    if (!cleanTranscript) {
      return { shouldReply: false, reason: "empty", usedGateModel: false };
    }
    if (chimeMode === "always") {
      return { shouldReply: true, reason: "always", usedGateModel: false };
    }
    if (isAliasAddressed(cleanTranscript, botAliases)) {
      return { shouldReply: true, reason: "addressed", usedGateModel: false };
    }
    if (chimeMode === "mention") {
      return { shouldReply: false, reason: "mention_only", usedGateModel: false };
    }
    if (
      lastBotReplyAt !== null &&
      lastBotReplyAt !== undefined &&
      clock() - lastBotReplyAt <= followUpWindowMs
    ) {
      return { shouldReply: true, reason: "followup_window", usedGateModel: false };
    }

    const timeout = createTimeoutController(gateTimeoutMs, "Gate request");
    const gateSignal = combineSignals(signal, timeout.signal);
    let response;
    try {
      response = await fetchImpl(
        endpoint,
        buildRequestInit({
          model: llm.gateModel || llm.replyModel,
          messages: [
            {
              role: "system",
              content:
                "You are a Discord voice-call turn gate. Answer ONLY YES or NO: should the AI companion chime in right now?",
            },
            {
              role: "user",
              content: `[${speakerName}]: ${cleanTranscript}`,
            },
          ],
          stream: false,
          maxTokens: 2,
          temperature: 0,
          signal: gateSignal,
        }),
      );

      if (!response.ok) {
        await response.body?.cancel().catch(() => {});
        return { shouldReply: false, reason: "gate_http_error", usedGateModel: true };
      }

      const payload = await readBoundedJson(response, MAX_GATE_BYTES);
      const answer = payload?.choices?.[0]?.message?.content?.trim() ?? "";
      if (/^yes$/i.test(answer)) {
        return { shouldReply: true, reason: "gate_yes", usedGateModel: true };
      }
      return { shouldReply: false, reason: "gate_no", usedGateModel: true };
    } catch {
      // Fail closed so network errors, parse issues, or timeouts never trigger unwanted chatter.
      await response?.body?.cancel().catch(() => {});
      return { shouldReply: false, reason: "gate_error", usedGateModel: true };
    } finally {
      timeout.clear();
    }
  }

  return {
    streamReply,
    shouldChimeIn,
  };
}
