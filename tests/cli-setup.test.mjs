import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildEnvFileContent,
  pregenerateReactionClips,
  runSetupWizard,
} from "../src/cli/setup.mjs";
import * as entryMod from "../src/index.mjs";
import { createCompanionApp } from "../src/index.mjs";
import { loadConfig } from "../src/config/config.mjs";
import { createLlmClient } from "../src/llm/client.mjs";
import { REACTION_CLIP_PROMPTS } from "../src/tts/reactions.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function createMockSetupFs() {
  const dirs = new Set([REPO_ROOT, path.posix.resolve("/trusted/root")]);
  const files = new Map();
  const symlinks = new Set();
  const modes = new Map();
  const chmodCalls = [];
  const unlinkCalls = [];

  return {
    dirs,
    files,
    symlinks,
    modes,
    chmodCalls,
    unlinkCalls,
    existsSync(target) {
      const key = path.resolve(target);
      return dirs.has(key) || files.has(key) || symlinks.has(key);
    },
    mkdirSync(target, opts = {}) {
      const key = path.resolve(target);
      dirs.add(key);
      if (opts.mode !== undefined) modes.set(key, opts.mode);
    },
    unlinkSync(target) {
      const key = path.resolve(target);
      unlinkCalls.push(key);
      files.delete(key);
      symlinks.delete(key);
    },
    renameSync(oldTarget, newTarget) {
      const oldKey = path.resolve(oldTarget);
      const newKey = path.resolve(newTarget);
      if (!files.has(oldKey)) {
        const err = new Error(`ENOENT: ${oldTarget}`);
        err.code = "ENOENT";
        throw err;
      }
      files.set(newKey, files.get(oldKey));
      files.delete(oldKey);
      if (modes.has(oldKey)) {
        modes.set(newKey, modes.get(oldKey));
        modes.delete(oldKey);
      }
    },
    writeFileSync(target, data, opts = {}) {
      const key = path.resolve(target);
      if (opts?.flag === "wx" && (files.has(key) || dirs.has(key) || symlinks.has(key))) {
        const err = new Error(`EEXIST: ${target}`);
        err.code = "EEXIST";
        throw err;
      }
      files.set(key, Buffer.isBuffer(data) ? Buffer.from(data) : String(data));
      if (typeof opts === "object" && opts?.mode !== undefined) {
        modes.set(key, opts.mode);
      }
    },
    readFileSync(target, encoding) {
      const key = path.resolve(target);
      if (!files.has(key)) {
        const err = new Error(`ENOENT: ${target}`);
        err.code = "ENOENT";
        throw err;
      }
      const val = files.get(key);
      if (encoding === "utf8") {
        return Buffer.isBuffer(val) ? val.toString("utf8") : val;
      }
      return Buffer.isBuffer(val) ? Buffer.from(val) : Buffer.from(val, "utf8");
    },
    chmodSync(target, mode) {
      const key = path.resolve(target);
      modes.set(key, mode);
      chmodCalls.push({ target: key, mode });
    },
    lstatSync(target) {
      const key = path.resolve(target);
      if (symlinks.has(key)) {
        return { isSymbolicLink: () => true, isDirectory: () => false, isFile: () => false, dev: 1, ino: 9 };
      }
      if (dirs.has(key)) {
        return { isSymbolicLink: () => false, isDirectory: () => true, isFile: () => false, dev: 1, ino: 1 };
      }
      if (files.has(key)) {
        const val = files.get(key);
        const size = Buffer.isBuffer(val) ? val.length : Buffer.byteLength(val, "utf8");
        return { isSymbolicLink: () => false, isDirectory: () => false, isFile: () => true, size, dev: 1, ino: 2 };
      }
      const err = new Error(`ENOENT: ${target}`);
      err.code = "ENOENT";
      throw err;
    },
    realpathSync(target) {
      return path.resolve(target);
    },
    readdirSync(target) {
      const prefix = path.resolve(target) + path.sep;
      const out = [];
      for (const fileKey of files.keys()) {
        if (fileKey.startsWith(prefix)) {
          out.push(path.basename(fileKey));
        }
      }
      return out;
    },
  };
}

test("buildEnvFileContent formats valid .env lines, blocks control chars cleanly, and validates endpoint URLs", () => {
  const envText = buildEnvFileContent({
    discordToken: "disc-token-123",
    discordAppId: "app-456",
    llmProvider: "deepseek",
    llmApiKey: "llm-secret-789",
    sttProvider: "deepgram",
    deepgramApiKey: "dg-secret-012",
    ttsProvider: "elevenlabs",
    elevenlabsApiKey: "el-secret-345",
    elevenlabsVoiceId: "voice_capri",
    expressiveTags: false,
    reactionMode: "adaptive",
    chimeMode: "smart",
    chimeBotAliases: "capri, blue",
    reactionCacheDir: ".cache/reactions",
  });

  assert.match(envText, /^DISCORD_BOT_TOKEN=disc-token-123$/m);
  assert.match(envText, /^LLM_PROVIDER=deepseek$/m);
  assert.match(envText, /^STT_PROVIDER=deepgram$/m);
  assert.match(envText, /^DEEPGRAM_API_KEY=dg-secret-012$/m);
  assert.match(envText, /^TTS_PROVIDER=elevenlabs$/m);
  assert.match(envText, /^ELEVENLABS_VOICE_ID=voice_capri$/m);
  assert.match(envText, /^CHIME_MODE=smart$/m);
  assert.match(envText, /^REACTION_CACHE_DIR=\.cache\/reactions$/m);

  assert.throws(
    () =>
      buildEnvFileContent({
        discordToken: "secret-leaked-part\nINJECTED_KEY=bad",
      }),
    (err) =>
      err instanceof RangeError &&
      err.message === "Setup field contains invalid control characters" &&
      !err.message.includes("secret-leaked-part"),
  );

  assert.throws(
    () =>
      buildEnvFileContent({
        llmProvider: "custom",
        llmBaseUrl: "http://remote.example.com/v1",
      }),
    RangeError,
  );
});

test("pregenerateReactionClips synthesizes 14 clips at 0o600 and rejects cacheDir outside trustedRootDir", async () => {
  const mockFs = createMockSetupFs();
  const root = path.resolve(REPO_ROOT, "tmp-setup-test");
  mockFs.dirs.add(root);

  const calls = [];
  const fakeTts = {
    async synthesizeClause({ text, onPcmChunk }) {
      calls.push(text);
      onPcmChunk(Buffer.from([1, 2, 3, 4]));
      return { firstByteAt: 10, byteLength: 4 };
    },
  };

  const result = await pregenerateReactionClips({
    ttsAdapter: fakeTts,
    cacheDir: ".cache/reactions",
    trustedRootDir: root,
    fsImpl: mockFs,
  });

  assert.equal(result.generatedCount, 14);
  assert.equal(calls.length, REACTION_CLIP_PROMPTS.length);
  assert.equal(result.files.length, 14);
  for (const filePath of result.files) {
    assert.equal(mockFs.modes.get(filePath), 0o600);
  }

  await assert.rejects(
    () =>
      pregenerateReactionClips({
        ttsAdapter: fakeTts,
        cacheDir: "../outside-reactions",
        trustedRootDir: root,
        fsImpl: mockFs,
      }),
    RangeError,
  );
});

test("runSetupWizard writes 0600 .env, pregenerates clips, closes TTS adapter, and returns redacted config", async () => {
  const mockFs = createMockSetupFs();
  const root = path.resolve(REPO_ROOT, "wizard-root");
  mockFs.dirs.add(root);

  let adapterClosed = false;
  const fakeTtsFactory = () => ({
    async synthesizeClause({ onPcmChunk }) {
      onPcmChunk(Buffer.from([10, 20, 30, 40]));
      return { firstByteAt: 5, byteLength: 4 };
    },
    close() {
      adapterClosed = true;
    },
  });

  const out = await runSetupWizard({
    answers: {
      discordToken: "super-secret-discord-token",
      discordAppId: "1234567890",
      llmProvider: "deepseek",
      llmApiKey: "super-secret-llm-key",
      sttProvider: "deepgram",
      deepgramApiKey: "super-secret-dg-key",
      ttsProvider: "elevenlabs",
      elevenlabsApiKey: "super-secret-el-key",
      elevenlabsVoiceId: "voice-capri",
    },
    ttsAdapterFactory: fakeTtsFactory,
    fsImpl: mockFs,
    envPath: ".env",
    cacheDir: ".cache/reactions",
    trustedRootDir: root,
  });

  assert.equal(mockFs.modes.get(out.envPath), 0o600);
  assert.equal(adapterClosed, true);
  assert.equal(out.generatedClips.generatedCount, 14);
  assert.equal(out.config.discord.token, "[REDACTED]");
  assert.equal(out.config.llm.apiKey, "[REDACTED]");
  assert.equal(out.config.tts.apiKey, "[REDACTED]");
  const serialized = JSON.stringify(out);
  assert.equal(serialized.includes("super-secret"), false);
});

test("createCompanionApp loads clips inside trustedRootDir, absorbs warmup failure, returns redacted config, and stops idempotently", async () => {
  const mockFs = createMockSetupFs();
  const root = path.resolve(REPO_ROOT, "app-root");
  const cacheDir = path.resolve(root, ".cache", "reactions");
  mockFs.dirs.add(root);
  mockFs.dirs.add(cacheDir);
  mockFs.files.set(path.join(cacheDir, "subtle_breath_01.pcm"), Buffer.from([1, 2, 3, 4]));
  mockFs.files.set(
    path.join(root, ".env"),
    [
      "DISCORD_BOT_TOKEN=secret-bot-token",
      "DISCORD_APP_ID=app-1",
      "LLM_PROVIDER=deepseek",
      "LLM_API_KEY=secret-llm",
      "STT_PROVIDER=deepgram",
      "DEEPGRAM_API_KEY=secret-dg",
      "TTS_PROVIDER=elevenlabs",
      "ELEVENLABS_API_KEY=secret-el",
      "ELEVENLABS_VOICE_ID=voice-1",
      "REACTION_CACHE_DIR=.cache/reactions",
    ].join("\n"),
  );

  let warmupCalls = 0;
  let orchestratorStarted = 0;
  let orchestratorClosed = 0;
  let ttsClosed = 0;
  let sttClosed = 0;

  const app = await createCompanionApp({
    env: {},
    envPath: path.join(root, ".env"),
    configPath: path.join(root, "config.json"),
    soulPath: path.join(root, "SOUL.md"),
    trustedRootDir: root,
    fsImpl: mockFs,
    createLlmClientImpl: () => ({ streamReply: async () => {}, shouldChimeIn: async () => false }),
    createTtsAdapterImpl: () => ({
      name: () => "elevenlabs-ws",
      async warmup() {
        warmupCalls += 1;
        throw new Error("Transient warmup error");
      },
      async synthesizeClause() {},
      close() {
        ttsClosed += 1;
      },
    }),
    createSttAdapterImpl: () => ({
      name: () => "deepgram",
      async transcribe() {},
      close() {
        sttClosed += 1;
      },
    }),
    createVoiceOrchestratorImpl: () => ({
      async start() {
        orchestratorStarted += 1;
      },
      close() {
        orchestratorClosed += 1;
      },
    }),
    createCommandHandlerImpl: () => ({
      commands: [],
      async handleInteraction() {},
    }),
  });

  assert.equal(app.reactionPool.size, 1);
  assert.equal(app.config.discord.token, "[REDACTED]");
  assert.equal(app.config.llm.apiKey, "[REDACTED]");

  const startStatus = await app.start();
  assert.equal(warmupCalls, 1);
  assert.equal(orchestratorStarted, 1);
  assert.equal(startStatus.config.discord.token, "[REDACTED]");

  await app.stop();
  await app.stop();
  assert.equal(orchestratorClosed, 1);
  assert.equal(ttsClosed, 1);
  assert.equal(sttClosed, 1);
});

test("SOUL.md and README.md pass the prose gate with required content and zero AI formatting tells", () => {
  const soulPath = path.join(REPO_ROOT, "SOUL.md");
  const readmePath = path.join(REPO_ROOT, "README.md");
  const soulText = fs.readFileSync(soulPath, "utf8");
  const readmeText = fs.readFileSync(readmePath, "utf8");

  for (const [name, content] of [
    ["SOUL.md", soulText],
    ["README.md", readmeText],
  ]) {
    assert.ok(content.trim().length > 40, `${name} must contain real prose`);
    assert.equal(content.includes("\u2014"), false, `${name} must not contain em dashes`);
    assert.equal(content.includes("\u2013"), false, `${name} must not contain en dashes`);
    assert.equal(/^- \*\*/m.test(content), false, `${name} must not contain bolded inline-header lists`);
  }

  assert.match(readmeText, /700\s*ms/);
  assert.match(readmeText, /22\.12\.0/);
  assert.match(readmeText, /0600/);
  assert.match(readmeText, /0700/);
  assert.match(readmeText, /WSS/i);
  assert.match(readmeText, /automatically registers/i);
  assert.match(readmeText, /\/join/);
  assert.match(readmeText, /\/mute-user/);
  assert.match(readmeText, /faster-whisper/);
  assert.match(readmeText, /eleven_flash_v2_5/);
});

test("interactive custom LLM setup prompts for reply/gate models, marks secrets, enforces 0600/0700 chmods, and blocks symlinks", async () => {
  const mockFs = createMockSetupFs();
  const root = path.resolve(REPO_ROOT, "wizard-custom-root");
  mockFs.dirs.add(root);

  const promptsSeen = [];
  const scriptedReplies = {
    "Discord bot token": "disc-secret",
    "Discord application id": "app-999",
    "LLM provider": "custom",
    "LLM API key": "custom-llm-secret",
    "Custom LLM base URL": "https://llm.example.com/v1",
    "Custom LLM reply model": "custom-fast-chat",
    "Custom LLM gate model": "custom-fast-gate",
    "STT provider": "deepgram",
    "deepgram STT API key": "dg-secret",
    "TTS provider": "elevenlabs",
    "elevenlabs TTS API key": "el-secret",
    "TTS voice id": "voice-capri",
  };

  const out = await runSetupWizard({
    promptImpl: async (question, fallback = "", opts = {}) => {
      promptsSeen.push({ question, fallback, opts });
      for (const [prefix, val] of Object.entries(scriptedReplies)) {
        if (question.startsWith(prefix)) return val;
      }
      return fallback;
    },
    skipPregenerate: false,
    ttsAdapterFactory: () => ({
      async synthesizeClause({ onPcmChunk }) {
        onPcmChunk(Buffer.from([1, 2, 3, 4]));
        return { firstByteAt: 1, byteLength: 4 };
      },
      close() {},
    }),
    fsImpl: mockFs,
    envPath: ".env",
    cacheDir: ".cache/reactions",
    trustedRootDir: root,
  });

  // Secret prompts must include { secret: true }
  const secretPrompts = promptsSeen.filter((p) => p.opts?.secret === true);
  assert.equal(secretPrompts.length, 4);
  assert.ok(promptsSeen.some((p) => p.question.includes("reply model")));
  assert.ok(promptsSeen.some((p) => p.question.includes("gate model")));

  // Resulting .env loads cleanly into createLlmClient without throwing
  const reloaded = loadConfig({ env: {}, envPath: out.envPath, fsImpl: mockFs });
  assert.equal(reloaded.llm.replyModel, "custom-fast-chat");
  assert.equal(reloaded.llm.gateModel, "custom-fast-gate");
  const llm = createLlmClient({ config: reloaded });
  assert.equal(typeof llm.streamReply, "function");

  // chmod 0600 on .env and 0700 on cacheDir
  assert.ok(mockFs.chmodCalls.some((c) => c.target === out.envPath && c.mode === 0o600));
  assert.ok(mockFs.chmodCalls.some((c) => c.target === out.cacheDir && c.mode === 0o700));

  // Symlinked .env and out-of-root envPath must both reject with RangeError
  mockFs.symlinks.add(path.resolve(root, "link.env"));
  await assert.rejects(
    () =>
      runSetupWizard({
        answers: {
          discordToken: "t",
          discordAppId: "a",
          llmProvider: "deepseek",
          sttProvider: "deepgram",
          ttsProvider: "elevenlabs",
        },
        skipPregenerate: true,
        fsImpl: mockFs,
        envPath: "link.env",
        trustedRootDir: root,
      }),
    RangeError,
  );
  await assert.rejects(
    () =>
      runSetupWizard({
        answers: {
          discordToken: "t",
          discordAppId: "a",
          llmProvider: "deepseek",
          sttProvider: "deepgram",
          ttsProvider: "elevenlabs",
        },
        skipPregenerate: true,
        fsImpl: mockFs,
        envPath: "../outside.env",
        trustedRootDir: root,
      }),
    RangeError,
  );
});

test("createCompanionApp constructs Discord client, binds interactionCreate, logs in on start(), and closes adapters even when orchestrator.close rejects", async () => {
  const { EventEmitter } = await import("node:events");
  const mockFs = createMockSetupFs();
  const root = path.resolve(REPO_ROOT, "app-client-root");
  mockFs.dirs.add(root);
  mockFs.files.set(
    path.join(root, ".env"),
    [
      "DISCORD_BOT_TOKEN=bot-login-token",
      "DISCORD_APP_ID=app-1",
      "LLM_PROVIDER=deepseek",
      "LLM_API_KEY=llm-k",
      "STT_PROVIDER=deepgram",
      "DEEPGRAM_API_KEY=dg-k",
      "TTS_PROVIDER=elevenlabs",
      "ELEVENLABS_API_KEY=el-k",
      "ELEVENLABS_VOICE_ID=v-1",
    ].join("\n"),
  );

  let fakeClient = null;
  const interactionsHandled = [];
  let loginTokenSeen = null;
  let clientDestroyed = false;
  let ttsClosed = false;
  let sttClosed = false;

  const app = await createCompanionApp({
    env: {},
    envPath: path.join(root, ".env"),
    configPath: path.join(root, "config.json"),
    soulPath: path.join(root, "SOUL.md"),
    trustedRootDir: root,
    fsImpl: mockFs,
    createDiscordClientImpl: () => {
      fakeClient = new EventEmitter();
      fakeClient.login = async (tok) => {
        loginTokenSeen = tok;
      };
      fakeClient.destroy = () => {
        clientDestroyed = true;
      };
      return fakeClient;
    },
    createLlmClientImpl: () => ({ streamReply: async () => {}, shouldChimeIn: async () => false }),
    createTtsAdapterImpl: () => ({
      name: () => "elevenlabs-ws",
      async warmup() {},
      async synthesizeClause() {},
      close() {
        ttsClosed = true;
      },
    }),
    createSttAdapterImpl: () => ({
      name: () => "deepgram",
      async transcribe() {},
      close() {
        sttClosed = true;
      },
    }),
    createVoiceOrchestratorImpl: () => ({
      async start() {},
      async close() {
        throw new Error("orchestrator teardown failed");
      },
    }),
    createCommandHandlerImpl: () => ({
      commands: [],
      async handleInteraction(interaction) {
        interactionsHandled.push(interaction.id);
      },
    }),
  });

  assert.equal(fakeClient.listenerCount("interactionCreate"), 1);
  fakeClient.emit("interactionCreate", { id: "ix-1" });
  assert.deepEqual(interactionsHandled, ["ix-1"]);

  await app.start();
  assert.equal(loginTokenSeen, "bot-login-token");

  await app.stop();
  assert.equal(clientDestroyed, true);
  assert.equal(ttsClosed, true);
  assert.equal(sttClosed, true);
  assert.equal(fakeClient.listenerCount("interactionCreate"), 0);
});

test("runSetupWizard preserves existing .env on ENOSPC via atomic rename and refuses symlink cache ancestor before mkdir", async () => {
  const mockFs = createMockSetupFs();
  const root = path.resolve(REPO_ROOT, "atomic-env-root");
  const envTarget = path.join(root, ".env");
  mockFs.dirs.add(root);
  mockFs.files.set(envTarget, "DISCORD_BOT_TOKEN=existing-token\n");

  const origWrite = mockFs.writeFileSync.bind(mockFs);
  mockFs.writeFileSync = (target, data, opts) => {
    if (String(target).includes(".tmp-")) {
      const err = new Error("ENOSPC: no space left on device");
      err.code = "ENOSPC";
      throw err;
    }
    return origWrite(target, data, opts);
  };

  await assert.rejects(
    () =>
      runSetupWizard({
        answers: {
          discordToken: "new-token",
          discordAppId: "app-2",
          llmProvider: "deepseek",
          sttProvider: "deepgram",
          ttsProvider: "elevenlabs",
        },
        skipPregenerate: true,
        fsImpl: mockFs,
        envPath: ".env",
        trustedRootDir: root,
      }),
    /ENOSPC/,
  );

  // Original .env must remain intact because staging temp write failed before renameSync
  assert.equal(mockFs.files.get(envTarget), "DISCORD_BOT_TOKEN=existing-token\n");
  assert.equal(mockFs.unlinkCalls.includes(envTarget), false);

  // Symlinked ancestor under trustedRootDir must fail before mkdirSync is called
  const linkDir = path.join(root, "symlink-parent");
  mockFs.symlinks.add(linkDir);
  let mkdirCalledForSymlink = false;
  mockFs.mkdirSync = (target) => {
    if (path.resolve(target).startsWith(linkDir)) {
      mkdirCalledForSymlink = true;
    }
  };
  await assert.rejects(
    () =>
      pregenerateReactionClips({
        ttsAdapter: { async synthesizeClause() {} },
        cacheDir: "symlink-parent/reactions",
        trustedRootDir: root,
        fsImpl: mockFs,
      }),
    RangeError,
  );
  assert.equal(mkdirCalledForSymlink, false);
});

test("createCompanionApp wires opusTranscoder, registers slashCommands on start(), and rejects start() if stop() runs mid-startup", async () => {
  const { EventEmitter } = await import("node:events");
  const mockFs = createMockSetupFs();
  const root = path.resolve(REPO_ROOT, "app-slash-root");
  mockFs.dirs.add(root);
  mockFs.files.set(
    path.join(root, ".env"),
    [
      "DISCORD_BOT_TOKEN=bot-token-slash",
      "DISCORD_APP_ID=app-1",
      "LLM_PROVIDER=deepseek",
      "LLM_API_KEY=llm-k",
      "STT_PROVIDER=deepgram",
      "DEEPGRAM_API_KEY=dg-k",
      "TTS_PROVIDER=elevenlabs",
      "ELEVENLABS_API_KEY=el-k",
      "ELEVENLABS_VOICE_ID=v-1",
    ].join("\n"),
  );

  const expectedCommands = [{ name: "join" }, { name: "leave" }, { name: "mode" }, { name: "status" }, { name: "mute-user" }];
  const fakeTranscoder = { backend: "mock-opus", encode: () => Buffer.alloc(4), decode: () => Buffer.alloc(4) };
  let seenTranscoder = null;
  let registeredCommands = null;

  const client1 = new EventEmitter();
  client1.login = async () => {};
  client1.destroy = () => {};
  client1.application = {
    commands: {
      async set(cmds) {
        registeredCommands = cmds;
      },
    },
  };

  const app1 = await createCompanionApp({
    env: {},
    envPath: path.join(root, ".env"),
    configPath: path.join(root, "config.json"),
    soulPath: path.join(root, "SOUL.md"),
    trustedRootDir: root,
    fsImpl: mockFs,
    client: client1,
    slashCommands: expectedCommands,
    createOpusTranscoderImpl: async () => fakeTranscoder,
    createLlmClientImpl: () => ({ streamReply: async () => {}, shouldChimeIn: async () => false }),
    createTtsAdapterImpl: () => ({ name: () => "el", async warmup() {}, async synthesizeClause() {}, close() {} }),
    createSttAdapterImpl: () => ({ name: () => "dg", async transcribe() {}, close() {} }),
    createVoiceOrchestratorImpl: (opts) => {
      seenTranscoder = opts.opusTranscoder;
      return { async start() {}, close() {} };
    },
    createCommandHandlerImpl: () => ({ commands: expectedCommands, async handleInteraction() {} }),
  });

  assert.equal(seenTranscoder, fakeTranscoder);
  await app1.start();
  assert.deepEqual(registeredCommands, expectedCommands);

  // If stop() runs while orchestrator.start() is awaiting, start() must reject cleanly
  let releaseOrchStart;
  const orchStartGate = new Promise((resolve) => {
    releaseOrchStart = resolve;
  });
  let app2Closed = false;
  const app2 = await createCompanionApp({
    env: {},
    envPath: path.join(root, ".env"),
    configPath: path.join(root, "config.json"),
    soulPath: path.join(root, "SOUL.md"),
    trustedRootDir: root,
    fsImpl: mockFs,
    createDiscordClientImpl: () => ({ login: async () => {}, destroy() {}, on() {}, off() {} }),
    createOpusTranscoderImpl: async () => fakeTranscoder,
    createLlmClientImpl: () => ({ streamReply: async () => {}, shouldChimeIn: async () => false }),
    createTtsAdapterImpl: () => ({ name: () => "el", async warmup() {}, async synthesizeClause() {}, close() { app2Closed = true; } }),
    createSttAdapterImpl: () => ({ name: () => "dg", async transcribe() {}, close() {} }),
    createVoiceOrchestratorImpl: () => ({
      async start() {
        await orchStartGate;
      },
      close() {},
    }),
    createCommandHandlerImpl: () => ({ commands: [], async handleInteraction() {} }),
  });

  const startPromise = app2.start();
  await new Promise((r) => setTimeout(r, 10));
  const stopPromise = app2.stop();
  releaseOrchStart();
  await assert.rejects(startPromise, /stopped during startup/i);
  await stopPromise;
  assert.equal(app2Closed, true);
});

test("normalizeSlashCommands converts string option types and bare string choices without mutating inputs", () => {
  assert.equal(typeof entryMod.normalizeSlashCommands, "function");

  const a3Commands = [
    {
      name: "mode",
      description: "Switch mode",
      options: [
        {
          name: "chime",
          description: "Chime mode",
          type: "STRING",
          required: false,
          choices: ["smart", "always", { name: "Mention", value: "mention" }],
        },
        {
          name: "group",
          type: "SUB_COMMAND_GROUP",
          options: [
            {
              name: "sub",
              type: "SUB_COMMAND",
              options: [
                { name: "count", type: "INTEGER" },
                { name: "flag", type: "BOOLEAN" },
                { name: "who", type: "USER" },
                { name: "chan", type: "CHANNEL" },
                { name: "role", type: "ROLE" },
                { name: "any", type: "MENTIONABLE" },
                { name: "ratio", type: "NUMBER" },
                { name: "clip", type: "ATTACHMENT" },
              ],
            },
          ],
        },
      ],
    },
    {
      name: "join",
      description: "Join channel",
      options: [
        {
          name: "channel",
          type: 7,
          choices: [{ name: "Lobby", value: "lobby" }],
        },
      ],
    },
  ];
  const snapshotBefore = JSON.stringify(a3Commands);

  const normalized = entryMod.normalizeSlashCommands(a3Commands);
  assert.equal(JSON.stringify(a3Commands), snapshotBefore);

  assert.equal(normalized[0].options[0].type, 3);
  assert.deepEqual(normalized[0].options[0].choices, [
    { name: "smart", value: "smart" },
    { name: "always", value: "always" },
    { name: "Mention", value: "mention" },
  ]);
  assert.equal(normalized[0].options[1].type, 2);
  assert.equal(normalized[0].options[1].options[0].type, 1);
  assert.deepEqual(
    normalized[0].options[1].options[0].options.map((o) => o.type),
    [4, 5, 6, 7, 8, 9, 10, 11],
  );
  assert.equal(normalized[1].options[0].type, 7);
  assert.deepEqual(normalized[1].options[0].choices, [{ name: "Lobby", value: "lobby" }]);
});

test("stop() cleans up deferred ready/clientReady listeners and startup failure before ready detaches fallback listeners", async () => {
  const { EventEmitter } = await import("node:events");
  const mockFs = createMockSetupFs();
  const root = path.resolve(REPO_ROOT, "app-ready-cleanup-root");
  mockFs.dirs.add(root);
  mockFs.files.set(
    path.join(root, ".env"),
    [
      "DISCORD_BOT_TOKEN=bot-token-ready",
      "DISCORD_APP_ID=app-1",
      "LLM_PROVIDER=deepseek",
      "LLM_API_KEY=llm-k",
      "STT_PROVIDER=deepgram",
      "DEEPGRAM_API_KEY=dg-k",
      "TTS_PROVIDER=elevenlabs",
      "ELEVENLABS_API_KEY=el-k",
      "ELEVENLABS_VOICE_ID=v-1",
    ].join("\n"),
  );

  const client1 = new EventEmitter();
  client1.login = async () => {};
  client1.destroy = () => {};
  const fakeTranscoder = { backend: "mock-opus", encode: () => Buffer.alloc(4), decode: () => Buffer.alloc(4) };

  const app1 = await createCompanionApp({
    env: {},
    envPath: path.join(root, ".env"),
    configPath: path.join(root, "config.json"),
    soulPath: path.join(root, "SOUL.md"),
    trustedRootDir: root,
    fsImpl: mockFs,
    client: client1,
    slashCommands: [{ name: "mode", options: [{ name: "kind", type: "STRING", choices: ["smart"] }] }],
    createOpusTranscoderImpl: async () => fakeTranscoder,
    createLlmClientImpl: () => ({ streamReply: async () => {}, shouldChimeIn: async () => false }),
    createTtsAdapterImpl: () => ({ name: () => "el", async warmup() {}, async synthesizeClause() {}, close() {} }),
    createSttAdapterImpl: () => ({ name: () => "dg", async transcribe() {}, close() {} }),
    createVoiceOrchestratorImpl: () => ({ async start() {}, close() {} }),
    createCommandHandlerImpl: () => ({ commands: [], async handleInteraction() {} }),
  });

  await app1.start();
  assert.ok(client1.listenerCount("ready") >= 1);
  await app1.stop();
  assert.equal(client1.listenerCount("ready"), 0);
  assert.equal(client1.listenerCount("clientReady"), 0);

  // If orchestrator.start() throws before ready, fallback listeners must still be cleaned up
  const client2 = new EventEmitter();
  client2.login = async () => {};
  client2.destroy = () => {};
  const app2 = await createCompanionApp({
    env: {},
    envPath: path.join(root, ".env"),
    configPath: path.join(root, "config.json"),
    soulPath: path.join(root, "SOUL.md"),
    trustedRootDir: root,
    fsImpl: mockFs,
    client: client2,
    slashCommands: [{ name: "join" }],
    createOpusTranscoderImpl: async () => fakeTranscoder,
    createLlmClientImpl: () => ({ streamReply: async () => {}, shouldChimeIn: async () => false }),
    createTtsAdapterImpl: () => ({ name: () => "el", async warmup() {}, async synthesizeClause() {}, close() {} }),
    createSttAdapterImpl: () => ({ name: () => "dg", async transcribe() {}, close() {} }),
    createVoiceOrchestratorImpl: () => ({
      async start() {
        throw new Error("boom during start");
      },
      close() {},
    }),
    createCommandHandlerImpl: () => ({ commands: [], async handleInteraction() {} }),
  });

  await assert.rejects(() => app2.start(), /boom during start/);
  assert.equal(client2.listenerCount("ready"), 0);
  assert.equal(client2.listenerCount("clientReady"), 0);
});

test("runSetupWizard does not unlink an unowned pre-existing .tmp- file on EEXIST", async () => {
  const mockFs = createMockSetupFs();
  const root = path.resolve(REPO_ROOT, "eexist-tmp-root");
  mockFs.dirs.add(root);
  const envTarget = path.join(root, ".env");
  mockFs.files.set(envTarget, "DISCORD_BOT_TOKEN=keep-original\n");

  const unlinkedTargets = [];
  mockFs.openSync = () => {
    const err = new Error("EEXIST: file already exists");
    err.code = "EEXIST";
    throw err;
  };
  mockFs.writeSync = () => 0;
  mockFs.closeSync = () => {};
  mockFs.unlinkSync = (target) => {
    unlinkedTargets.push(path.resolve(target));
  };

  await assert.rejects(
    () =>
      runSetupWizard({
        answers: {
          discordToken: "new-token",
          discordAppId: "app-2",
          llmProvider: "deepseek",
          sttProvider: "deepgram",
          ttsProvider: "elevenlabs",
        },
        skipPregenerate: true,
        fsImpl: mockFs,
        envPath: ".env",
        trustedRootDir: root,
      }),
    /EEXIST/,
  );

  assert.equal(unlinkedTargets.length, 0);
  assert.equal(mockFs.files.get(envTarget), "DISCORD_BOT_TOKEN=keep-original\n");
});

test("writePrivateFile rejects short 0-byte writes without calling renameSync, preserves existing .env, uses fchmodSync, and clears stageCreated after rename", async () => {
  const mockFs = createMockSetupFs();
  const root = path.resolve(REPO_ROOT, "short-write-root");
  mockFs.dirs.add(root);
  const envTarget = path.join(root, ".env");
  mockFs.files.set(envTarget, "DISCORD_BOT_TOKEN=original-preserved\n");

  let writeCalls = 0;
  let renameCalls = 0;
  const unlinkedTmpPaths = [];
  const origUnlink = mockFs.unlinkSync.bind(mockFs);
  const origRename = mockFs.renameSync.bind(mockFs);

  mockFs.openSync = (target) => {
    mockFs.files.set(path.resolve(target), Buffer.alloc(0));
    return 77;
  };
  mockFs.writeSync = () => {
    writeCalls += 1;
    return writeCalls === 1 ? 3 : 0;
  };
  mockFs.closeSync = () => {};
  mockFs.renameSync = (...args) => {
    renameCalls += 1;
    return origRename(...args);
  };
  mockFs.unlinkSync = (target) => {
    unlinkedTmpPaths.push(path.resolve(target));
    return origUnlink(target);
  };

  await assert.rejects(
    () =>
      runSetupWizard({
        answers: {
          discordToken: "new-token",
          discordAppId: "app-3",
          llmProvider: "deepseek",
          sttProvider: "deepgram",
          ttsProvider: "elevenlabs",
        },
        skipPregenerate: true,
        fsImpl: mockFs,
        envPath: ".env",
        trustedRootDir: root,
      }),
    /Short write|Incomplete write/i,
  );

  assert.equal(renameCalls, 0);
  assert.equal(unlinkedTmpPaths.length, 1);
  assert.ok(path.basename(unlinkedTmpPaths[0]).startsWith(".env.tmp-"));
  assert.equal(mockFs.files.get(envTarget), "DISCORD_BOT_TOKEN=original-preserved\n");

  // Full write calls fchmodSync before closeSync and clears stageCreated after renameSync
  const callOrder = [];
  const postRenameUnlinks = [];
  const fullFs = createMockSetupFs();
  fullFs.dirs.add(root);
  fullFs.files.set(envTarget, "DISCORD_BOT_TOKEN=original-preserved\n");
  fullFs.openSync = (target) => {
    fullFs.files.set(path.resolve(target), Buffer.alloc(0));
    callOrder.push("openSync");
    return 88;
  };
  fullFs.writeSync = (fd, buf, off, len) => {
    assert.equal(fd, 88);
    callOrder.push("writeSync");
    return len;
  };
  fullFs.fchmodSync = (fd, mode) => {
    assert.equal(fd, 88);
    assert.equal(mode, 0o600);
    callOrder.push("fchmodSync");
  };
  fullFs.closeSync = () => {
    callOrder.push("closeSync");
  };
  const baseRename = fullFs.renameSync.bind(fullFs);
  fullFs.renameSync = (from, to) => {
    callOrder.push("renameSync");
    return baseRename(from, to);
  };
  fullFs.chmodSync = (target) => {
    if (path.resolve(target) === envTarget) {
      throw new Error("post-rename chmod failure");
    }
  };
  fullFs.unlinkSync = (target) => {
    postRenameUnlinks.push(path.resolve(target));
  };

  await assert.rejects(
    () =>
      runSetupWizard({
        answers: {
          discordToken: "ok-token",
          discordAppId: "app-4",
          llmProvider: "deepseek",
          sttProvider: "deepgram",
          ttsProvider: "elevenlabs",
        },
        skipPregenerate: true,
        fsImpl: fullFs,
        envPath: ".env",
        trustedRootDir: root,
      }),
    /post-rename chmod failure/,
  );

  assert.equal(postRenameUnlinks.length, 0);
  assert.ok(callOrder.indexOf("fchmodSync") < callOrder.indexOf("closeSync"));
});

test("createCompanionApp.start registers A3-shaped slashCommands normalized with numeric option types and { name, value } choices", async () => {
  const { EventEmitter } = await import("node:events");
  const mockFs = createMockSetupFs();
  const root = path.resolve(REPO_ROOT, "app-a3-normalize-root");
  mockFs.dirs.add(root);
  mockFs.files.set(
    path.join(root, ".env"),
    [
      "DISCORD_BOT_TOKEN=bot-token-norm",
      "DISCORD_APP_ID=app-1",
      "LLM_PROVIDER=deepseek",
      "LLM_API_KEY=llm-k",
      "STT_PROVIDER=deepgram",
      "DEEPGRAM_API_KEY=dg-k",
      "TTS_PROVIDER=elevenlabs",
      "ELEVENLABS_API_KEY=el-k",
      "ELEVENLABS_VOICE_ID=v-1",
    ].join("\n"),
  );

  let setPayload = null;
  const fakeClient = new EventEmitter();
  fakeClient.login = async () => {};
  fakeClient.destroy = () => {};
  fakeClient.application = {
    commands: {
      async set(cmds) {
        setPayload = cmds;
      },
    },
  };

  const a3Commands = [
    {
      name: "mode",
      description: "Configure companion behavior",
      options: [
        {
          name: "chime",
          description: "Chime mode",
          type: "STRING",
          choices: ["smart", "always", "mention"],
        },
        {
          name: "expressive",
          description: "Expressive tags",
          type: "BOOLEAN",
        },
      ],
    },
  ];

  const app = await createCompanionApp({
    env: {},
    envPath: path.join(root, ".env"),
    configPath: path.join(root, "config.json"),
    soulPath: path.join(root, "SOUL.md"),
    trustedRootDir: root,
    fsImpl: mockFs,
    client: fakeClient,
    slashCommands: a3Commands,
    createOpusTranscoderImpl: async () => ({ backend: "mock-opus", encode: () => Buffer.alloc(4), decode: () => Buffer.alloc(4) }),
    createLlmClientImpl: () => ({ streamReply: async () => {}, shouldChimeIn: async () => false }),
    createTtsAdapterImpl: () => ({ name: () => "el", async warmup() {}, async synthesizeClause() {}, close() {} }),
    createSttAdapterImpl: () => ({ name: () => "dg", async transcribe() {}, close() {} }),
    createVoiceOrchestratorImpl: () => ({ async start() {}, close() {} }),
    createCommandHandlerImpl: () => ({ commands: a3Commands, async handleInteraction() {} }),
  });

  await app.start();
  assert.equal(setPayload.length, 1);
  assert.equal(setPayload[0].options[0].type, 3);
  assert.equal(setPayload[0].options[1].type, 5);
  assert.deepEqual(setPayload[0].options[0].choices, [
    { name: "smart", value: "smart" },
    { name: "always", value: "always" },
    { name: "mention", value: "mention" },
  ]);
  await app.stop();
});
