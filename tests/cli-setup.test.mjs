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
import { createCompanionApp } from "../src/index.mjs";
import { REACTION_CLIP_PROMPTS } from "../src/tts/reactions.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function createMockSetupFs() {
  const dirs = new Set([REPO_ROOT, path.posix.resolve("/trusted/root")]);
  const files = new Map();
  const modes = new Map();
  const chmodCalls = [];

  return {
    dirs,
    files,
    modes,
    chmodCalls,
    existsSync(target) {
      const key = path.resolve(target);
      return dirs.has(key) || files.has(key);
    },
    mkdirSync(target, opts = {}) {
      const key = path.resolve(target);
      dirs.add(key);
      if (opts.mode !== undefined) modes.set(key, opts.mode);
    },
    writeFileSync(target, data, opts = {}) {
      const key = path.resolve(target);
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
  assert.match(readmeText, /\/join/);
  assert.match(readmeText, /\/mute-user/);
  assert.match(readmeText, /faster-whisper/);
  assert.match(readmeText, /eleven_flash_v2_5/);
});
