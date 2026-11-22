import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { CHIME_MODES, FILLER_MODES, PROVIDERS } from "../types.mjs";
import { LLM_PRESETS, loadConfig, redactConfig, validateEndpointUrl } from "../config/config.mjs";
import { createTtsAdapter } from "../tts/adapters.mjs";
import { REACTION_CLIP_PROMPTS } from "../tts/reactions.mjs";

const CONTROL_CHAR_PATTERN = /[\u0000-\u001f\u007f]/;

function assertNoControlChars(value) {
  if (typeof value === "string" && CONTROL_CHAR_PATTERN.test(value)) {
    throw new RangeError("Setup field contains invalid control characters");
  }
  return typeof value === "string" ? value.trim() : "";
}

function assertEnum(value, allowed, fallback, label) {
  const clean = assertNoControlChars(value ?? "");
  const candidate = clean || fallback;
  if (!allowed.includes(candidate)) {
    throw new RangeError(`${label} must be one of [${allowed.join(", ")}]`);
  }
  return candidate;
}

function isWithinRoot(candidatePath, rootPath) {
  const normCandidate = path.normalize(candidatePath);
  const normRoot = path.normalize(rootPath);
  const prefix = normRoot.endsWith(path.sep) ? normRoot : `${normRoot}${path.sep}`;
  return normCandidate === normRoot || normCandidate.startsWith(prefix);
}

function resolveContainedPath(baseRoot, targetPath, label) {
  const cleanTarget = assertNoControlChars(String(targetPath ?? ""));
  if (!cleanTarget) {
    throw new RangeError(`${label} must be a non-empty path`);
  }
  const resolvedRoot = path.resolve(baseRoot);
  const resolvedTarget = path.resolve(resolvedRoot, cleanTarget);
  if (!isWithinRoot(resolvedTarget, resolvedRoot)) {
    throw new RangeError(`${label} is outside trusted root`);
  }
  return { resolvedRoot, resolvedTarget };
}

function assertNoSymlinksOnPath(rootPath, targetPath, fsImpl) {
  if (typeof fsImpl.lstatSync !== "function") return;
  const rel = path.relative(rootPath, targetPath);
  const segments = rel ? rel.split(path.sep).filter(Boolean) : [];
  let cursor = rootPath;
  const check = (candidate) => {
    let st;
    try {
      st = fsImpl.lstatSync(candidate);
    } catch (error) {
      if (error && error.code === "ENOENT") return;
      throw error;
    }
    if (st?.isSymbolicLink?.()) {
      throw new RangeError("Setup target path cannot contain symlinks");
    }
  };
  check(cursor);
  for (const seg of segments) {
    cursor = path.join(cursor, seg);
    check(cursor);
  }
}

let tempFileCounter = 0;

function assertTargetNotSymlink(targetPath, fsImpl) {
  if (typeof fsImpl.lstatSync !== "function") return null;
  let existing = null;
  try {
    existing = fsImpl.lstatSync(targetPath);
  } catch (error) {
    if (!error || error.code !== "ENOENT") throw error;
  }
  if (existing?.isSymbolicLink?.()) {
    throw new RangeError("Refusing to write to a symlink target");
  }
  if (existing && !existing.isFile?.()) {
    throw new RangeError("Target path is not a regular file");
  }
  return existing;
}

function writePrivateFile(targetPath, data, { encoding, fsImpl }) {
  assertTargetNotSymlink(targetPath, fsImpl);

  if (typeof fsImpl.renameSync === "function") {
    tempFileCounter += 1;
    const tmpPath = `${targetPath}.tmp-${process.pid}-${tempFileCounter}`;
    assertTargetNotSymlink(tmpPath, fsImpl);
    const writeOpts = { mode: 0o600, flag: "wx" };
    if (encoding) writeOpts.encoding = encoding;
    try {
      fsImpl.writeFileSync(tmpPath, data, writeOpts);
      fsImpl.chmodSync?.(tmpPath, 0o600);
      assertTargetNotSymlink(targetPath, fsImpl);
      fsImpl.renameSync(tmpPath, targetPath);
      fsImpl.chmodSync?.(targetPath, 0o600);
    } catch (error) {
      try {
        fsImpl.unlinkSync?.(tmpPath);
      } catch {
        // Ignore temp cleanup errors.
      }
      throw error;
    }
    return;
  }

  const fallbackOpts = { mode: 0o600, flag: "w" };
  if (encoding) fallbackOpts.encoding = encoding;
  fsImpl.writeFileSync(targetPath, data, fallbackOpts);
  fsImpl.chmodSync?.(targetPath, 0o600);
}

/**
 * Validate setup answers and serialize an LF-delimited .env file string.
 */
export function buildEnvFileContent(answers = {}) {
  if (answers === null || typeof answers !== "object") {
    throw new TypeError("Setup answers must be an object");
  }

  const llmProvider = assertEnum(answers.llmProvider, PROVIDERS.llm, "deepseek", "llmProvider");
  const sttProvider = assertEnum(answers.sttProvider, PROVIDERS.stt, "deepgram", "sttProvider");
  const ttsProvider = assertEnum(answers.ttsProvider, PROVIDERS.tts, "elevenlabs", "ttsProvider");
  const reactionMode = assertEnum(answers.reactionMode, FILLER_MODES, "adaptive", "reactionMode");
  const chimeMode = assertEnum(answers.chimeMode, CHIME_MODES, "smart", "chimeMode");

  const discordToken = assertNoControlChars(answers.discordToken ?? "");
  const discordAppId = assertNoControlChars(answers.discordAppId ?? "");
  const llmApiKey = assertNoControlChars(answers.llmApiKey ?? "");
  const llmReplyModel = assertNoControlChars(answers.llmReplyModel ?? "");
  const llmGateModel = assertNoControlChars(answers.llmGateModel ?? "");
  const sttApiKey = assertNoControlChars(answers.sttApiKey ?? "");
  const ttsApiKey = assertNoControlChars(answers.ttsApiKey ?? "");
  const elevenlabsVoiceId = assertNoControlChars(
    answers.elevenlabsVoiceId ?? answers.ttsVoiceId ?? "",
  );
  const expressiveTags = Boolean(answers.expressiveTags);
  const reactionCacheDir =
    assertNoControlChars(answers.reactionCacheDir ?? ".cache/reactions") || ".cache/reactions";

  const rawAliases = Array.isArray(answers.chimeBotAliases)
    ? answers.chimeBotAliases.map((item) => assertNoControlChars(String(item ?? ""))).filter(Boolean).join(", ")
    : assertNoControlChars(answers.chimeBotAliases ?? "");

  const llmBaseUrlRaw = assertNoControlChars(answers.llmBaseUrl ?? "");
  const localSttUrlRaw = assertNoControlChars(answers.localSttUrl ?? "");
  const ttsBaseUrlRaw = assertNoControlChars(answers.ttsBaseUrl ?? "");

  const llmBaseUrl = llmBaseUrlRaw ? validateEndpointUrl(llmBaseUrlRaw, "llm.baseUrl") : "";
  const localSttUrl = localSttUrlRaw ? validateEndpointUrl(localSttUrlRaw, "stt.localUrl") : "";
  const ttsBaseUrl = ttsBaseUrlRaw ? validateEndpointUrl(ttsBaseUrlRaw, "tts.baseUrl") : "";

  const deepgramApiKey =
    assertNoControlChars(answers.deepgramApiKey ?? "") ||
    (sttProvider === "deepgram" ? sttApiKey : "");
  const elevenlabsApiKey =
    assertNoControlChars(answers.elevenlabsApiKey ?? "") ||
    (sttProvider === "elevenlabs" ? sttApiKey : "") ||
    (ttsProvider === "elevenlabs" ? ttsApiKey : "");
  const openaiApiKey =
    assertNoControlChars(answers.openaiApiKey ?? "") ||
    (sttProvider === "openai" ? sttApiKey : "") ||
    (ttsProvider === "openai" ? ttsApiKey : "");
  const groqApiKey =
    assertNoControlChars(answers.groqApiKey ?? "") ||
    (sttProvider === "groq" ? sttApiKey : "");

  const resolvedTtsApiKey =
    ttsApiKey ||
    (ttsProvider === "elevenlabs"
      ? elevenlabsApiKey
      : ttsProvider === "openai"
        ? openaiApiKey
        : "");

  const pairs = [
    ["DISCORD_BOT_TOKEN", discordToken],
    ["DISCORD_APP_ID", discordAppId],
    ["LLM_PROVIDER", llmProvider],
    ["LLM_API_KEY", llmApiKey],
    ["LLM_BASE_URL", llmBaseUrl],
    ["LLM_REPLY_MODEL", llmReplyModel],
    ["LLM_GATE_MODEL", llmGateModel],
    ["STT_PROVIDER", sttProvider],
    ["DEEPGRAM_API_KEY", deepgramApiKey],
    ["ELEVENLABS_API_KEY", elevenlabsApiKey],
    ["OPENAI_API_KEY", openaiApiKey],
    ["GROQ_API_KEY", groqApiKey],
    ["LOCAL_STT_URL", localSttUrl],
    ["TTS_PROVIDER", ttsProvider],
    ["TTS_API_KEY", resolvedTtsApiKey],
    ["TTS_BASE_URL", ttsBaseUrl],
    ["ELEVENLABS_VOICE_ID", elevenlabsVoiceId],
    ["ELEVENLABS_EXPRESSIVE_TAGS", expressiveTags ? "true" : "false"],
    ["REACTION_FILLERS", reactionMode],
    ["CHIME_MODE", chimeMode],
    ["CHIME_BOT_ALIASES", rawAliases],
    ["REACTION_CACHE_DIR", reactionCacheDir],
  ];

  return `${pairs.map(([key, val]) => `${key}=${val}`).join("\n")}\n`;
}

/**
 * Sequentially pregenerate voice-matched reaction PCM clips and write them with 0600 permissions.
 */
export async function pregenerateReactionClips({
  ttsAdapter,
  cacheDir = ".cache/reactions",
  trustedRootDir,
  prompts = REACTION_CLIP_PROMPTS,
  fsImpl = fs,
} = {}) {
  if (!ttsAdapter || typeof ttsAdapter.synthesizeClause !== "function") {
    throw new TypeError("pregenerateReactionClips requires a valid ttsAdapter");
  }

  const cleanCacheDir = assertNoControlChars(String(cacheDir ?? ""));
  if (!cleanCacheDir) {
    throw new RangeError("cacheDir must be a non-empty path");
  }

  let resolvedCacheDir;
  let symlinkCheckRoot;
  if (trustedRootDir) {
    const contained = resolveContainedPath(trustedRootDir, cleanCacheDir, "Reaction cache directory");
    resolvedCacheDir = contained.resolvedTarget;
    symlinkCheckRoot = contained.resolvedRoot;
    if (typeof fsImpl.realpathSync === "function") {
      try {
        const canonicalRoot = fsImpl.realpathSync(contained.resolvedRoot);
        let ancestor = resolvedCacheDir;
        while (ancestor && ancestor !== path.dirname(ancestor)) {
          if (typeof fsImpl.existsSync !== "function" || fsImpl.existsSync(ancestor)) {
            const canonicalAncestor = fsImpl.realpathSync(ancestor);
            if (!isWithinRoot(canonicalAncestor, canonicalRoot)) {
              throw new RangeError("Reaction cache directory is outside trusted root");
            }
            break;
          }
          if (ancestor === contained.resolvedRoot) break;
          ancestor = path.dirname(ancestor);
        }
      } catch (error) {
        if (error instanceof RangeError) throw error;
      }
    }
  } else {
    resolvedCacheDir = path.resolve(cleanCacheDir);
    symlinkCheckRoot = path.dirname(resolvedCacheDir);
  }

  assertNoSymlinksOnPath(symlinkCheckRoot, resolvedCacheDir, fsImpl);

  fsImpl.mkdirSync(resolvedCacheDir, { recursive: true, mode: 0o700 });
  fsImpl.chmodSync?.(resolvedCacheDir, 0o700);
  if (trustedRootDir && typeof fsImpl.realpathSync === "function") {
    const canonicalRoot = fsImpl.realpathSync(path.resolve(trustedRootDir));
    const canonicalDir = fsImpl.realpathSync(resolvedCacheDir);
    if (!isWithinRoot(canonicalDir, canonicalRoot)) {
      throw new RangeError("Reaction cache directory is outside trusted root");
    }
    resolvedCacheDir = canonicalDir;
  }

  const files = [];
  for (const prompt of prompts) {
    const safeId = assertNoControlChars(String(prompt?.id ?? ""));
    if (!safeId || safeId.includes("/") || safeId.includes("\\") || safeId.includes("..")) {
      throw new RangeError("Reaction prompt id contains invalid path characters");
    }
    const chunks = [];
    await ttsAdapter.synthesizeClause({
      text: prompt.text,
      onPcmChunk: (chunk) => {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
      },
    });
    const audioBuffer = Buffer.concat(chunks);
    const filePath = path.join(resolvedCacheDir, `${safeId}.pcm`);
    writePrivateFile(filePath, audioBuffer, { fsImpl });
    files.push(filePath);
  }

  return {
    cacheDir: resolvedCacheDir,
    generatedCount: files.length,
    files,
  };
}

async function collectInteractiveAnswers(promptImpl, defaults = {}) {
  const ask = async (label, fallback = "", opts = {}) => {
    const answer = await promptImpl(label, fallback, opts);
    return typeof answer === "string" && answer.trim().length > 0 ? answer.trim() : fallback;
  };

  const discordToken = await ask("Discord bot token", "", { secret: true });
  const discordAppId = await ask("Discord application id");
  const llmProvider = await ask("LLM provider (opencode/deepseek/openai/openrouter/groq/ollama/custom)", defaults.llmProvider ?? "deepseek");
  const llmApiKey = llmProvider === "ollama" ? "" : await ask("LLM API key", "", { secret: true });
  const llmBaseUrl = llmProvider === "custom" ? await ask("Custom LLM base URL") : "";
  const preset = LLM_PRESETS[llmProvider];
  let llmReplyModel = "";
  let llmGateModel = "";
  if (llmProvider === "custom" || !preset?.replyModel) {
    llmReplyModel = assertNoControlChars(await ask("Custom LLM reply model"));
    if (!llmReplyModel) {
      throw new RangeError("Custom LLM provider requires a reply model");
    }
    llmGateModel = assertNoControlChars(await ask("Custom LLM gate model", llmReplyModel)) || llmReplyModel;
  }
  const sttProvider = await ask("STT provider (deepgram/elevenlabs/openai/groq/faster-whisper)", defaults.sttProvider ?? "deepgram");
  const sttApiKey = sttProvider === "faster-whisper" ? "" : await ask(`${sttProvider} STT API key`, "", { secret: true });
  const localSttUrl = sttProvider === "faster-whisper" ? await ask("Local faster-whisper URL", "http://127.0.0.1:4373") : "";
  const ttsProvider = await ask("TTS provider (elevenlabs/openai/openai-compatible)", defaults.ttsProvider ?? "elevenlabs");
  const ttsApiKey = await ask(`${ttsProvider} TTS API key`, "", { secret: true });
  const ttsBaseUrl = ttsProvider === "openai-compatible" ? await ask("TTS base URL") : "";
  const elevenlabsVoiceId = await ask("TTS voice id", ttsProvider === "openai" ? "nova" : "");
  const expressiveInput = ttsProvider === "elevenlabs" ? await ask("Enable ElevenLabs v4 expressive audio tags? (true/false)", "false") : "false";
  const reactionMode = await ask("Reaction filler mode (adaptive/subtle/off)", "adaptive");
  const chimeMode = await ask("Chime mode (smart/always/mention)", "smart");
  const chimeBotAliases = await ask("Wake aliases (comma-separated, optional)", "");

  return {
    discordToken,
    discordAppId,
    llmProvider,
    llmApiKey,
    llmBaseUrl,
    llmReplyModel,
    llmGateModel,
    sttProvider,
    sttApiKey,
    localSttUrl,
    ttsProvider,
    ttsApiKey,
    ttsBaseUrl,
    elevenlabsVoiceId,
    expressiveTags: expressiveInput.toLowerCase() === "true",
    reactionMode,
    chimeMode,
    chimeBotAliases,
  };
}

/**
 * Run interactive or non-interactive setup, write 0600 .env, pregenerate reaction clips, and return redacted config.
 */
export async function runSetupWizard({
  answers,
  promptImpl,
  ttsAdapterFactory = createTtsAdapter,
  fsImpl = fs,
  envPath = ".env",
  cacheDir = ".cache/reactions",
  trustedRootDir = process.cwd(),
  skipPregenerate = false,
} = {}) {
  const rawAnswers = answers ?? (typeof promptImpl === "function" ? await collectInteractiveAnswers(promptImpl) : null);
  if (!rawAnswers) {
    throw new TypeError("runSetupWizard requires answers or promptImpl");
  }

  const effectiveAnswers = {
    ...rawAnswers,
    reactionCacheDir: rawAnswers.reactionCacheDir ?? cacheDir,
  };

  const envContent = buildEnvFileContent(effectiveAnswers);
  const { resolvedRoot, resolvedTarget: resolvedEnvPath } = resolveContainedPath(
    trustedRootDir,
    envPath,
    "envPath",
  );
  if (typeof fsImpl.realpathSync === "function") {
    const canonicalRoot = fsImpl.realpathSync(resolvedRoot);
    const parentCandidate = path.dirname(resolvedEnvPath);
    if (typeof fsImpl.existsSync !== "function" || fsImpl.existsSync(parentCandidate)) {
      const canonicalParent = fsImpl.realpathSync(parentCandidate);
      if (!isWithinRoot(canonicalParent, canonicalRoot)) {
        throw new RangeError("envPath parent directory is outside trusted root");
      }
    }
  }
  assertNoSymlinksOnPath(resolvedRoot, resolvedEnvPath, fsImpl);

  const envDir = path.dirname(resolvedEnvPath);
  if (typeof fsImpl.mkdirSync === "function") {
    fsImpl.mkdirSync(envDir, { recursive: true, mode: 0o700 });
  }
  writePrivateFile(resolvedEnvPath, envContent, { encoding: "utf8", fsImpl });

  const loadedConfig = loadConfig({ env: {}, envPath: resolvedEnvPath, fsImpl });
  const targetCacheDir = loadedConfig.reactions.cacheDir || cacheDir;

  let generatedClips = {
    cacheDir: resolveContainedPath(resolvedRoot, targetCacheDir, "Reaction cache directory").resolvedTarget,
    generatedCount: 0,
    files: [],
  };

  const hasTtsCredential = Boolean(loadedConfig.tts.apiKey || loadedConfig.tts.provider === "openai-compatible");
  if (!skipPregenerate && hasTtsCredential && loadedConfig.reactions.mode !== "off") {
    const adapter = ttsAdapterFactory(loadedConfig);
    try {
      generatedClips = await pregenerateReactionClips({
        ttsAdapter: adapter,
        cacheDir: targetCacheDir,
        trustedRootDir: resolvedRoot,
        fsImpl,
      });
    } finally {
      await adapter?.close?.();
    }
  }

  return {
    envPath: resolvedEnvPath,
    cacheDir: generatedClips.cacheDir,
    generatedClips,
    config: redactConfig(loadedConfig),
  };
}

const isCliEntry =
  typeof process !== "undefined" &&
  Array.isArray(process.argv) &&
  Boolean(process.argv[1]) &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isCliEntry) {
  const readline = await import("node:readline/promises");
  const { Writable } = await import("node:stream");
  let muteOutput = false;
  const silentOrEchoOutput = new Writable({
    write(chunk, encoding, callback) {
      if (!muteOutput) {
        process.stdout.write(chunk, encoding);
      }
      callback();
    },
  });
  const rl = readline.createInterface({
    input: process.stdin,
    output: silentOrEchoOutput,
    terminal: Boolean(process.stdin.isTTY),
  });
  try {
    const result = await runSetupWizard({
      promptImpl: async (question, fallback = "", opts = {}) => {
        const suffix = fallback ? ` [${fallback}]` : "";
        if (opts.secret) {
          process.stdout.write(`${question}${suffix}: `);
          muteOutput = true;
          try {
            const line = await rl.question("");
            return line.trim() || fallback;
          } finally {
            muteOutput = false;
            process.stdout.write("\n");
          }
        }
        const line = await rl.question(`${question}${suffix}: `);
        return line.trim() || fallback;
      },
    });
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  } finally {
    rl.close();
  }
}
