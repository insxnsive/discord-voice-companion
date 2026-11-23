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
  assert.deepEqual(fast.llm.headers, {
    "x-opencode-client": "discord-voice-companion",
    "x-opencode-session": "discord-voice-companion",
  });
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

  assert.equal(seenRequest.url, "https://opencode.ai/zen/go/v1/chat/completions");
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

test("createClauseSplitter never splits inside [soft laugh] on maxHoldMs and scans commas left-to-right", () => {
  let t = 0;
  const stripped = createClauseSplitter({
    minWords: 5,
    maxHoldMs: 120,
    expressiveTags: false,
    now: () => t,
  });
  assert.deepEqual(stripped.push("One two three four five [soft"), []);
  t = 121;
  assert.deepEqual(stripped.push(" laugh]"), ["One two three four five"]);
  assert.deepEqual(stripped.flush(), []);

  t = 0;
  const expressive = createClauseSplitter({
    minWords: 5,
    maxHoldMs: 120,
    expressiveTags: true,
    now: () => t,
  });
  assert.deepEqual(expressive.push("One two three four five [soft"), []);
  t = 121;
  assert.deepEqual(expressive.push(" laugh]"), ["One two three four five [soft laugh]"]);
  assert.deepEqual(expressive.flush(), []);

  const commaBeforePeriod = createClauseSplitter({ expressiveTags: true });
  assert.deepEqual(commaBeforePeriod.push("Wait a minute, I am here."), [
    "Wait a minute,",
    "I am here.",
  ]);

  const skipShortComma = createClauseSplitter({ expressiveTags: true });
  assert.deepEqual(skipShortComma.push("Well, I am here, waiting"), ["Well, I am here,"]);
  assert.deepEqual(skipShortComma.flush(), ["waiting"]);
});

test("loadConfig and redactConfig enforce HTTPS on non-loopback, allowlist headers, and block __proto__", () => {
  assert.throws(
    () => loadConfig({ env: { LLM_BASE_URL: "http://api.example.com/v1" }, fsImpl: makeMemoryFs() }),
    RangeError,
  );
  assert.throws(
    () => loadConfig({ env: { LLM_BASE_URL: "https://user:pass@api.example.com/v1" }, fsImpl: makeMemoryFs() }),
    RangeError,
  );
  assert.throws(
    () => loadConfig({ env: { LLM_BASE_URL: "https://api.example.com/v1?foo=bar" }, fsImpl: makeMemoryFs() }),
    RangeError,
  );
  assert.throws(
    () => loadConfig({ env: { LLM_BASE_URL: "https://api.example.com/v1#frag" }, fsImpl: makeMemoryFs() }),
    RangeError,
  );

  const localOk = loadConfig({
    env: { LLM_PROVIDER: "ollama", LLM_BASE_URL: "http://127.0.0.1:11434/v1" },
    fsImpl: makeMemoryFs(),
  });
  assert.equal(localOk.llm.baseUrl, "http://127.0.0.1:11434/v1");

  const maliciousJson = '{"llm":{"headers":{"x-opencode-client":"discord-voice-companion","x-custom-secret":"hide-me"}},"__proto__":{"polluted":true}}';
  const cfg = loadConfig({
    env: { LLM_PROVIDER: "opencode" },
    configPath: "config.json",
    fsImpl: makeMemoryFs({ "config.json": maliciousJson }),
  });
  const redacted = redactConfig(cfg);
  assert.equal(redacted.llm.headers["x-opencode-client"], "discord-voice-companion");
  assert.equal(redacted.llm.headers["x-custom-secret"], "[REDACTED]");
  assert.equal(({}).polluted, undefined);
});

test("createLlmClient sets redirect error, sanitizes HTTP errors, bounds SSE size, and requires exact YES", async () => {
  const cfg = loadConfig({
    env: { LLM_PROVIDER: "opencode", LLM_API_KEY: "oc-key" },
    fsImpl: makeMemoryFs(),
  });

  let seenRedirect = null;
  let cancelled500 = false;
  const errClient = createLlmClient({
    config: cfg,
    fetchImpl: async (_url, init) => {
      seenRedirect = init.redirect;
      const body = new ReadableStream({
        start(c) {
          c.enqueue(new TextEncoder().encode("secret=abc"));
        },
        cancel() {
          cancelled500 = true;
        },
      });
      return new Response(body, { status: 500 });
    },
  });

  await assert.rejects(
    () => errClient.streamReply({ messages: [{ role: "user", content: "hi" }] }),
    (error) => {
      assert.equal(error.message, "LLM stream failed (HTTP 500)");
      assert.doesNotMatch(error.message, /secret=abc/);
      return true;
    },
  );
  assert.equal(seenRedirect, "error");
  assert.equal(cancelled500, true);

  let cancelledOversize = false;
  const bigClient = createLlmClient({
    config: cfg,
    fetchImpl: async () => {
      const stream = new ReadableStream({
        pull(controller) {
          const huge = "x".repeat(150000);
          controller.enqueue(new TextEncoder().encode(`data: {"choices":[{"delta":{"content":"${huge}"}}]}\n`));
        },
        cancel() {
          cancelledOversize = true;
        },
      });
      return new Response(stream, { status: 200 });
    },
  });

  await assert.rejects(
    () => bigClient.streamReply({ messages: [{ role: "user", content: "big" }] }),
    /size limit/i,
  );
  assert.equal(cancelledOversize, true);

  const gateClient = createLlmClient({
    config: cfg,
    fetchImpl: async () =>
      new Response(
        JSON.stringify({ choices: [{ message: { content: "YES maybe later" } }] }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
  });
  const fuzzyGate = await gateClient.shouldChimeIn({
    transcript: "Anyone want pizza?",
    speakerName: "Nate",
  });
  assert.equal(fuzzyGate.shouldReply, false);
  assert.equal(fuzzyGate.usedGateModel, true);
});

test("B1-REWORK-2: trailing SSE catch, bounded gate response, extraBody omission, and empty ?/# URLs", async () => {
  // 1. loadConfig rejects trailing empty ? and # URL delimiters
  assert.throws(
    () => loadConfig({ env: { LLM_BASE_URL: "https://api.example.com/v1?" }, fsImpl: makeMemoryFs() }),
    RangeError,
  );
  assert.throws(
    () => loadConfig({ env: { LLM_BASE_URL: "https://api.example.com/v1#" }, fsImpl: makeMemoryFs() }),
    RangeError,
  );

  // 2. redactConfig omits llm.extraBody completely so arbitrary fields never leak
  const cfgWithExtra = loadConfig({
    env: { LLM_PROVIDER: "opencode", LLM_API_KEY: "oc-key" },
    configPath: "config.json",
    fsImpl: makeMemoryFs({
      "config.json": JSON.stringify({
        llm: {
          headers: { "x-opencode-client": "discord-voice-companion" },
          extraBody: { custom_auth: "leak" },
        },
      }),
    }),
  });
  const redacted = redactConfig(cfgWithExtra);
  assert.equal(redacted.llm.headers["x-opencode-client"], "discord-voice-companion");
  assert.equal(Object.hasOwn(redacted.llm, "extraBody"), false);

  // 3. Final SSE data: chunk without trailing \n must not swallow onToken or maxReplyChars errors
  const tokenErrClient = createLlmClient({
    config: cfgWithExtra,
    fetchImpl: async () => makeSseResponse(['data: {"choices":[{"delta":{"content":"tail"}}]}']),
  });
  await assert.rejects(
    () =>
      tokenErrClient.streamReply({
        messages: [{ role: "user", content: "hi" }],
        onToken: () => {
          throw new Error("token sink boom");
        },
      }),
    /token sink boom/,
  );

  const maxCharsClient = createLlmClient({
    config: cfgWithExtra,
    maxReplyChars: 3,
    fetchImpl: async () => makeSseResponse(['data: {"choices":[{"delta":{"content":"abcd"}}]}']),
  });
  await assert.rejects(
    () => maxCharsClient.streamReply({ messages: [{ role: "user", content: "hi" }] }),
    /size limit/i,
  );

  // 4. shouldChimeIn cancels and fails closed with gate_error on >16384-byte response body
  let cancelledBigGate = false;
  const bigGateClient = createLlmClient({
    config: cfgWithExtra,
    fetchImpl: async () => {
      const stream = new ReadableStream({
        pull(controller) {
          controller.enqueue(new TextEncoder().encode("x".repeat(20000)));
        },
        cancel() {
          cancelledBigGate = true;
        },
      });
      return new Response(stream, { status: 200 });
    },
  });
  const bigGate = await bigGateClient.shouldChimeIn({
    transcript: "Anyone home?",
    speakerName: "Nate",
  });
  assert.deepEqual(bigGate, { shouldReply: false, reason: "gate_error", usedGateModel: true });
  assert.equal(cancelledBigGate, true);
});

test("C5 diagnostics: redactConfig reports scheme, host and port only and never echoes any path content", () => {
  const redacted = redactConfig({
    llm: { baseUrl: "https://api.example.com/v1/tok_AbC123xyz9876" },
    stt: { localUrl: "http://127.0.0.1:8080/v1" },
    tts: { baseUrl: "https://tts.example.com:8443/Secret-Path/abc" },
    extra: { baseUrl: "https://host.example.com/a/b/c/d/e/f" },
    label: { baseUrl: "https://proxy.example.com/private-label/v1" },
    broken: { baseUrl: "not a url" },
  });
  assert.equal(redacted.llm.baseUrl, "https://api.example.com");
  assert.equal(redacted.stt.localUrl, "http://127.0.0.1:8080");
  assert.equal(redacted.tts.baseUrl, "https://tts.example.com:8443");
  assert.equal(redacted.label.baseUrl, "https://proxy.example.com");
  assert.equal(redacted.extra.baseUrl, "https://host.example.com");
  assert.equal(redacted.broken.baseUrl, "[REDACTED]");
  assert.equal(JSON.stringify(redacted).includes("tok_AbC123xyz9876"), false);
  assert.equal(JSON.stringify(redacted).includes("Secret-Path"), false);
  assert.equal(JSON.stringify(redacted).includes("private-label"), false);
});

test("C5 diagnostics: custom values of allowlisted header names are reported as set, not echoed", () => {
  const redacted = redactConfig({
    llm: {
      headers: {
        "x-opencode-client": "discord-voice-companion",
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        "user-agent": "sk-live-leaked-token",
        "x-opencode-client-extra": "custom-value",
        empty: "",
      },
    },
  });
  assert.equal(redacted.llm.headers["x-opencode-client"], "discord-voice-companion");
  assert.equal(redacted.llm.headers["Content-Type"], "application/json");
  assert.equal(redacted.llm.headers.Accept, "text/event-stream");
  assert.equal(redacted.llm.headers["user-agent"], "[REDACTED]");
  assert.equal(redacted.llm.headers["x-opencode-client-extra"], "[REDACTED]");
  assert.equal(redacted.llm.headers.empty, "");

  const viaLoad = loadConfig({
    env: { LLM_PROVIDER: "opencode" },
    configPath: "config.json",
    fsImpl: makeMemoryFs({
      "config.json": JSON.stringify({ llm: { headers: { "x-opencode-client": "custom-tenant-4411" } } }),
    }),
  });
  assert.equal(redactConfig(viaLoad).llm.headers["x-opencode-client"], "[REDACTED]");
});

test("the opencode preset targets the Go gateway with both required headers", async () => {
  const config = loadConfig({
    env: { LLM_PROVIDER: "opencode", LLM_API_KEY: "oc_sk_test_key" },
    fsImpl: makeMemoryFs(),
  });
  const calls = [];
  const client = createLlmClient({
    config,
    fetchImpl: async (url, init) => {
      calls.push({ url: String(url), headers: init.headers });
      return makeSseResponse([
        'data: {"choices":[{"delta":{"content":"Hi there."}}]}\n\n',
        "data: [DONE]\n\n",
      ]);
    },
  });

  const clauses = [];
  await client.streamReply({
    messages: [{ role: "user", content: "hello" }],
    onClause: (clause) => clauses.push(clause),
  });

  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "https://opencode.ai/zen/go/v1/chat/completions");
  assert.equal(calls[0].headers["x-opencode-client"], "discord-voice-companion");
  assert.equal(calls[0].headers["x-opencode-session"], "discord-voice-companion");
  assert.deepEqual(clauses, ["Hi there."]);
});

test("the opencode preset disables reasoning and the gate timeout default is 1500", async () => {
  const config = loadConfig({
    env: { LLM_PROVIDER: "opencode", LLM_API_KEY: "oc_sk_test_key" },
    fsImpl: makeMemoryFs(),
  });
  assert.equal(config.chime.gateTimeoutMs, 1500);
  assert.deepEqual(config.llm.extraBody, { thinking: { type: "disabled" } });

  const bodies = [];
  const client = createLlmClient({
    config,
    fetchImpl: async (url, init) => {
      bodies.push(JSON.parse(init.body));
      return new Response(JSON.stringify({ choices: [{ message: { content: "NO" } }] }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    },
  });

  await client.shouldChimeIn({ transcript: "hello there", chimeMode: "smart" });
  assert.ok(bodies.length >= 1);
  assert.equal(bodies[0].thinking.type, "disabled");

  const auto = loadConfig({ env: {}, fsImpl: makeMemoryFs() });
  assert.equal(auto.stt.language, "");
  const portuguese = loadConfig({
    env: { STT_LANGUAGE: "pt" },
    fsImpl: makeMemoryFs(),
  });
  assert.equal(portuguese.stt.language, "pt");
});

test("the Go SSE shape speaks only delta.content and ignores role and reasoning deltas", async () => {
  const config = loadConfig({
    env: { LLM_PROVIDER: "opencode", LLM_API_KEY: "oc_sk_test_key" },
    fsImpl: makeMemoryFs(),
  });
  const client = createLlmClient({
    config,
    fetchImpl: async () =>
      makeSseResponse([
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
        'data: {"choices":[{"delta":{"reasoning_content":"hidden reasoning"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"there."}}]}\n\n',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        "data: [DONE]\n\n",
      ]),
  });

  const clauses = [];
  const result = await client.streamReply({
    messages: [{ role: "user", content: "hello" }],
    onClause: (clause) => clauses.push(clause),
  });

  assert.deepEqual(clauses, ["Hello there."]);
  assert.equal(result.fullText, "Hello there.");
  assert.equal(result.fullText.includes("hidden reasoning"), false);
});
