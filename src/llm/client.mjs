const AUDIO_TAG_BEFORE_PUNCT = /\s*\[[a-z][a-z0-9 _-]{1,32}\](?=\s*[.,!?;:])/gi;
const AUDIO_TAG_INLINE = /\s*\[[a-z][a-z0-9 _-]{1,32}\]\s*/gi;

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

function hasUnclosedBracket(text) {
  const openIndex = text.lastIndexOf("[");
  const closeIndex = text.lastIndexOf("]");
  return openIndex !== -1 && openIndex > closeIndex;
}

function countWords(text) {
  const speakable = stripAudioTags(text);
  return speakable ? speakable.split(/\s+/).length : 0;
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Buffer streamed LLM text deltas into speakable clauses without splitting
 * inside pending audio tags like [laughs].
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
      if (hasUnclosedBracket(buffer)) break;

      const sentenceMatch = buffer.match(/[.!?\n]+/);
      if (sentenceMatch) {
        const end = sentenceMatch.index + sentenceMatch[0].length;
        const head = buffer.slice(0, end);
        buffer = buffer.slice(end).replace(/^[ \t]+/, "");
        emitClause(head, emitted);
        clauseStartAt = buffer.length > 0 ? now() : null;
        continue;
      }

      const pauseMatch = buffer.match(/[,;]+/);
      if (pauseMatch) {
        const end = pauseMatch.index + pauseMatch[0].length;
        const head = buffer.slice(0, end);
        if (countWords(head) >= 3) {
          buffer = buffer.slice(end).replace(/^[ \t]+/, "");
          emitClause(head, emitted);
          clauseStartAt = buffer.length > 0 ? now() : null;
          continue;
        }
      }

      const elapsed = clauseStartAt === null ? 0 : now() - clauseStartAt;
      if (elapsed >= maxHoldMs) {
        if (/\s$/.test(buffer) && countWords(buffer) >= minWords) {
          const head = buffer;
          buffer = "";
          emitClause(head, emitted);
          clauseStartAt = null;
          continue;
        }

        const wsMatch = buffer.match(/\s+(?=\S*$)/);
        if (wsMatch) {
          const head = buffer.slice(0, wsMatch.index);
          if (countWords(head) >= minWords) {
            buffer = buffer.slice(wsMatch.index + wsMatch[0].length);
            emitClause(head, emitted);
            clauseStartAt = buffer.length > 0 ? now() : null;
            continue;
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

/**
 * Create an OpenAI-compatible streaming reply and 1-token turn-gate client.
 */
export function createLlmClient({
  config,
  fetchImpl = fetch,
  now = () => performance.now(),
} = {}) {
  const llm = config?.llm;
  if (!llm?.baseUrl) {
    throw new Error("LLM client requires config.llm.baseUrl");
  }
  if (!llm?.replyModel) {
    throw new Error("LLM client requires config.llm.replyModel");
  }

  const endpoint = `${llm.baseUrl.replace(/\/+$/, "")}/chat/completions`;
  const expressiveTags = Boolean(config?.tts?.expressiveTags);
  const defaultChime = config?.chime ?? {};

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
      signal,
    };
  }

  async function streamReply({ messages, signal, onClause, onToken } = {}) {
    const response = await fetchImpl(
      endpoint,
      buildRequestInit({
        model: llm.replyModel,
        messages,
        stream: true,
        maxTokens: llm.maxTokens ?? 140,
        signal,
      }),
    );

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`LLM stream failed (${response.status}): ${detail.slice(0, 200)}`);
    }
    if (!response.body) {
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
    let done = false;

    try {
      while (!done) {
        if (signal?.aborted) {
          throw signal.reason ?? Object.assign(new Error("Aborted"), { name: "AbortError" });
        }

        const { value, done: streamDone } = await reader.read();
        if (streamDone) break;

        carry += decoder.decode(value, { stream: true });
        const lines = carry.split(/\r?\n/);
        carry = lines.pop() ?? "";

        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line || line.startsWith(":")) continue;
          if (!line.startsWith("data:")) continue;

          const data = line.slice(5).trim();
          if (data === "[DONE]") {
            done = true;
            break;
          }

          let chunk;
          try {
            chunk = JSON.parse(data);
          } catch {
            continue;
          }

          const delta = chunk?.choices?.[0]?.delta?.content;
          if (typeof delta === "string" && delta.length > 0) {
            fullText += delta;
            if (typeof onToken === "function") onToken(delta);
            splitter.push(delta);
          }
        }
      }

      if (!done && carry.trim().startsWith("data:")) {
        const tailData = carry.trim().slice(5).trim();
        if (tailData && tailData !== "[DONE]") {
          try {
            const chunk = JSON.parse(tailData);
            const delta = chunk?.choices?.[0]?.delta?.content;
            if (typeof delta === "string" && delta.length > 0) {
              fullText += delta;
              if (typeof onToken === "function") onToken(delta);
              splitter.push(delta);
            }
          } catch {
            // Ignore incomplete trailing JSON fragment at stream EOF.
          }
        }
      }
    } catch (error) {
      await reader.cancel().catch(() => {});
      throw error;
    } finally {
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

    const timeoutAc = new AbortController();
    const timer = setTimeout(() => {
      const err = new Error(`Gate request timed out after ${gateTimeoutMs}ms`);
      err.name = "TimeoutError";
      timeoutAc.abort(err);
    }, gateTimeoutMs);
    const gateSignal = combineSignals(signal, timeoutAc.signal);
    try {
      const response = await fetchImpl(
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
        return { shouldReply: false, reason: "gate_http_error", usedGateModel: true };
      }

      const payload = await response.json();
      const answer = payload?.choices?.[0]?.message?.content?.trim() ?? "";
      if (/^YES\b/i.test(answer)) {
        return { shouldReply: true, reason: "gate_yes", usedGateModel: true };
      }
      return { shouldReply: false, reason: "gate_no", usedGateModel: true };
    } catch {
      // Fail closed so network errors or timeouts never trigger unwanted chatter.
      return { shouldReply: false, reason: "gate_error", usedGateModel: true };
    } finally {
      clearTimeout(timer);
    }
  }

  return {
    streamReply,
    shouldChimeIn,
  };
}
