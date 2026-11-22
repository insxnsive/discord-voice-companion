import test from "node:test";
import assert from "node:assert/strict";

import { LLM_PRESETS, loadConfig, parseDotEnv, redactConfig } from "../src/config/config.mjs";
import { createClauseSplitter, createLlmClient, stripAudioTags } from "../src/llm/client.mjs";

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

function makeSseResponse(chunks) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

test("loadConfig selects elevenlabs default vs expressive model and enforces provider presets", () => {
  const fsMock = makeMemoryFs({
    ".env": [
      "# comment",
      "DISCORD_BOT_TOKEN='disc-secret'",
      'LLM_PROVIDER="opencode"',
      "LLM_API_KEY=oc-secret",
      "LLM_BASE_URL=",
      "ELEVENLABS_API_KEY=el-secret",
      "ELEVENLABS_VOICE_ID=voice-123",
      "ELEVENLABS_EXPRESSIVE_TAGS=false",
      "DEEPGRAM_API_KEY=dg-secret",
    ].join("\n"),
    "SOUL.md": "Speak short and punchy.",
  });

  const fast = loadConfig({ env: {}, fsImpl: fsMock });
  assert.equal(fast.tts.provider, "elevenlabs");
  assert.equal(fast.tts.expressiveTags, false);
  assert.equal(fast.tts.model, "eleven_flash_v2_5");
  assert.equal(fast.llm.provider, "opencode");
  assert.equal(fast.llm.baseUrl, LLM_PRESETS.opencode.baseUrl);
  assert.deepEqual(fast.llm.headers, { "x-opencode-client": "discord-voice-companion" });
  assert.equal(fast.personaPrompt, "Speak short and punchy.");
  assert.equal(fast.reactions.mode, "adaptive");
  assert.equal(fast.reactions.fillerDelayMs, 165);
  assert.equal(fast.reactions.cooldownTurns, 10);
  assert.equal(fast.chime.mode, "smart");
  assert.equal(fast.mcp.enabled, false);
  assert.ok(Object.isFrozen(LLM_PRESETS));
  assert.ok(Object.isFrozen(LLM_PRESETS.deepseek.extraBody.thinking));

  const expressive = loadConfig({
    env: {
      ELEVENLABS_EXPRESSIVE_TAGS: "true",
      LLM_PROVIDER: "deepseek",
    },
    fsImpl: fsMock,
  });
  assert.equal(expressive.tts.expressiveTags, true);
  assert.equal(expressive.tts.model, "eleven_v4_turbo");
  assert.equal(expressive.llm.provider, "deepseek");
  assert.deepEqual(expressive.llm.extraBody, { thinking: { type: "disabled" } });
  assert.deepEqual(parseDotEnv("A='1'\n# skip\nB=\"two\""), { A: "1", B: "two" });

  assert.throws(
    () => loadConfig({ env: {}, fsImpl: makeMemoryFs({ "config.json": "{bad json" }) }),
    SyntaxError,
  );
  assert.throws(
    () => loadConfig({ env: { LLM_BASE_URL: "ftp://example.com" }, fsImpl: makeMemoryFs() }),
    RangeError,
  );
});

test("redactConfig strips tokens, API keys, and Authorization headers while keeping settings intact", () => {
  const cfg = loadConfig({
    env: {
      DISCORD_BOT_TOKEN: "bot-token-secret",
      DISCORD_APP_ID: "app-999",
      LLM_PROVIDER: "opencode",
      LLM_API_KEY: "llm-secret-key",
      ELEVENLABS_API_KEY: "el-secret-key",
      ELEVENLABS_VOICE_ID: "voice-123",
      DEEPGRAM_API_KEY: "dg-secret-key",
    },
    configPath: "config.json",
    fsImpl: makeMemoryFs({
      "config.json": JSON.stringify({
        llm: {
          headers: {
            Authorization: "Bearer super-secret",
            "x-opencode-client": "discord-voice-companion",
          },
        },
      }),
    }),
  });

  const redacted = redactConfig(cfg);

  assert.equal(redacted.discord.token, "[REDACTED]");
  assert.equal(redacted.discord.appId, "app-999");
  assert.equal(redacted.llm.apiKey, "[REDACTED]");
  assert.equal(redacted.llm.headers.Authorization, "[REDACTED]");
  assert.equal(redacted.llm.headers["x-opencode-client"], "discord-voice-companion");
  assert.equal(redacted.tts.apiKey, "[REDACTED]");
  assert.equal(redacted.tts.voiceId, "voice-123");
  assert.equal(redacted.stt.deepgramApiKey, "[REDACTED]");
  assert.equal(cfg.discord.token, "bot-token-secret");
});

test("createClauseSplitter preserves or strips audio tags, never splits mid-bracket, and respects timing", () => {
  assert.equal(stripAudioTags("[laughs] Hey, [sighs] wait up!"), "Hey, wait up!");

  let t = 1000;
  const expressiveClauses = [];
  const expressiveSplitter = createClauseSplitter({
    minWords: 5,
    maxHoldMs: 120,
    expressiveTags: true,
    now: () => t,
    onClause: (c) => expressiveClauses.push(c),
  });

  assert.deepEqual(expressiveSplitter.push("Hold on [lau"), []);
  t = 1200;
  assert.deepEqual(expressiveSplitter.push("ghs]"), []);
  assert.deepEqual(expressiveSplitter.push(" what happened?"), ["Hold on [laughs] what happened?"]);
  assert.deepEqual(expressiveSplitter.push("Wait a minute, let me check"), ["Wait a minute,"]);
  assert.deepEqual(expressiveSplitter.flush(), ["let me check"]);
  assert.deepEqual(expressiveClauses, [
    "Hold on [laughs] what happened?",
    "Wait a minute,",
    "let me check",
  ]);

  t = 2000;
  const plainClauses = [];
  const plainSplitter = createClauseSplitter({
    minWords: 5,
    maxHoldMs: 120,
    expressiveTags: false,
    now: () => t,
    onClause: (c) => plainClauses.push(c),
  });

  assert.deepEqual(plainSplitter.push("[laughs] One two three four "), []);
  t = 2080;
  assert.deepEqual(plainSplitter.push("five "), []);
  t = 2130;
  assert.deepEqual(plainSplitter.push("six "), ["One two three four five six"]);
  assert.deepEqual(plainSplitter.flush(), []);
  assert.deepEqual(plainClauses, ["One two three four five six"]);
});

test("streamReply parses split SSE chunks, stops on [DONE], omits auth for local Ollama, and aborts cleanly", async () => {
  const cfg = loadConfig({
    env: {
      LLM_PROVIDER: "opencode",
      LLM_API_KEY: "oc-key",
    },
    fsImpl: makeMemoryFs(),
  });

  let seenRequest = null;
  const streamedTokens = [];
  const streamedClauses = [];
  const client = createLlmClient({
    config: cfg,
    fetchImpl: async (url, init) => {
      seenRequest = { url, init };
      return makeSseResponse([
        'data: {"choices":[{"delta":{"content":"Hey "}}]}\n',
        'data: {"choices":[{"delta":{"content":"there"}}',
        ']}\n\ndata: {"choices":[{"delta":{"content":"! How are you?"}}]}\n',
        "data: [DONE]\n",
      ]);
    },
  });

  const result = await client.streamReply({
    messages: [{ role: "user", content: "hi" }],
    onToken: (tok) => streamedTokens.push(tok),
    onClause: (clause) => streamedClauses.push(clause),
  });

  assert.equal(seenRequest.url, "https://opencode.ai/zen/v1/chat/completions");
  assert.equal(seenRequest.init.headers.Authorization, "Bearer oc-key");
  assert.equal(seenRequest.init.headers["x-opencode-client"], "discord-voice-companion");
  assert.equal(result.fullText, "Hey there! How are you?");
  assert.deepEqual(result.clauses, ["Hey there!", "How are you?"]);
  assert.deepEqual(streamedClauses, ["Hey there!", "How are you?"]);
  assert.equal(streamedTokens.join(""), "Hey there! How are you?");
  assert.equal(typeof result.firstClauseAt, "number");

  const ollamaCfg = loadConfig({ env: { LLM_PROVIDER: "ollama" }, fsImpl: makeMemoryFs() });
  let ollamaHeaders = null;
  const ollamaClient = createLlmClient({
    config: ollamaCfg,
    fetchImpl: async (_url, init) => {
      ollamaHeaders = init.headers;
      return makeSseResponse(['data: {"choices":[{"delta":{"content":"Yo!"}}]}\n', "data: [DONE]\n"]);
    },
  });
  await ollamaClient.streamReply({ messages: [{ role: "user", content: "yo" }] });
  assert.equal(ollamaHeaders.Authorization, undefined);

  const abortClient = createLlmClient({
    config: cfg,
    fetchImpl: (_url, init) =>
      new Promise((_resolve, reject) => {
        init.signal.addEventListener("abort", () => {
          const err = new Error("Aborted");
          err.name = "AbortError";
          reject(err);
        });
      }),
  });
  const ac = new AbortController();
  const pending = abortClient.streamReply({
    messages: [{ role: "user", content: "wait" }],
    signal: ac.signal,
  });
  ac.abort();
  await assert.rejects(pending, { name: "AbortError" });
});

test("shouldChimeIn fast-paths aliases and followup window, parses gate YES/NO, and fails closed", async () => {
  const cfg = loadConfig({
    env: {
      LLM_PROVIDER: "deepseek",
      LLM_API_KEY: "ds-key",
    },
    configPath: "config.json",
    fsImpl: makeMemoryFs({
      "config.json": JSON.stringify({
        chime: { botAliases: ["Capri"], followUpWindowMs: 12000, gateTimeoutMs: 50 },
      }),
    }),
  });

  let fetchCalls = 0;
  let nextGateReply = "YES";
  let shouldTimeout = false;

  const client = createLlmClient({
    config: cfg,
    fetchImpl: (_url, init) => {
      fetchCalls += 1;
      if (shouldTimeout) {
        return new Promise((_resolve, reject) => {
          init.signal.addEventListener("abort", () => {
            const err = new Error("Timed out");
            err.name = "TimeoutError";
            reject(err);
          });
        });
      }
      return Promise.resolve(
        new Response(
          JSON.stringify({ choices: [{ message: { content: nextGateReply } }] }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    },
  });

  const empty = await client.shouldChimeIn({ transcript: "   " });
  assert.deepEqual(empty, { shouldReply: false, reason: "empty", usedGateModel: false });

  const always = await client.shouldChimeIn({ transcript: "random", chimeMode: "always" });
  assert.deepEqual(always, { shouldReply: true, reason: "always", usedGateModel: false });

  const mentionSkip = await client.shouldChimeIn({ transcript: "random", chimeMode: "mention" });
  assert.deepEqual(mentionSkip, { shouldReply: false, reason: "mention_only", usedGateModel: false });

  const addressed = await client.shouldChimeIn({
    transcript: "Hey Capri what do you think?",
    speakerName: "Nate",
    now: () => 50000,
  });
  assert.deepEqual(addressed, { shouldReply: true, reason: "addressed", usedGateModel: false });
  assert.equal(fetchCalls, 0);

  const followUp = await client.shouldChimeIn({
    transcript: "Wait what about that?",
    speakerName: "Nate",
    lastBotReplyAt: 42000,
    now: () => 50000,
  });
  assert.deepEqual(followUp, { shouldReply: true, reason: "followup_window", usedGateModel: false });
  assert.equal(fetchCalls, 0);

  nextGateReply = "YES";
  const gateYes = await client.shouldChimeIn({
    transcript: "Anyone know a good movie?",
    speakerName: "Nate",
    lastBotReplyAt: 10000,
    now: () => 50000,
  });
  assert.equal(gateYes.shouldReply, true);
  assert.equal(gateYes.usedGateModel, true);
  assert.equal(fetchCalls, 1);

  nextGateReply = "NO";
  const gateNo = await client.shouldChimeIn({
    transcript: "Pass me the water bottle",
    speakerName: "Alex",
    lastBotReplyAt: 10000,
    now: () => 50000,
  });
  assert.equal(gateNo.shouldReply, false);
  assert.equal(gateNo.usedGateModel, true);

  shouldTimeout = true;
  const gateFailClosed = await client.shouldChimeIn({
    transcript: "Hello?",
    speakerName: "Alex",
    lastBotReplyAt: 10000,
    now: () => 50000,
  });
  assert.equal(gateFailClosed.shouldReply, false);
  assert.equal(gateFailClosed.usedGateModel, true);
});
