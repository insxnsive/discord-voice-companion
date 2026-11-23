import fs from "node:fs";
import { CHIME_MODES, FILLER_MODES, PROVIDERS } from "../types.mjs";

const DEFAULT_PERSONA_PROMPT =
  "You are a concise, natural voice companion in a Discord call. Reply in 1 to 2 short spoken sentences with no markdown or lists.";

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "::1", "[::1]"]);
const FORBIDDEN_KEYS = new Set(["__proto__", "constructor", "prototype"]);
// Header names whose values may be echoed in diagnostics, and only when the value is one of
// the fixed literals below. Any other value for these names, and every value for any other
// header name, is reported as set ("[REDACTED]") without its content.
const SAFE_HEADER_VALUES = new Map([
  ["x-opencode-client", new Set(["discord-voice-companion"])],
  ["content-type", new Set(["application/json"])],
  ["accept", new Set(["application/json", "text/event-stream", "*/*"])],
  ["user-agent", new Set(["discord-voice-companion"])],
]);
const CONTROL_CHAR_PATTERN = /[\u0000-\u001f\u007f]/;

function deepFreeze(value) {
  if (value === null || typeof value !== "object") return value;
  for (const key of Object.keys(value)) {
    deepFreeze(value[key]);
  }
  return Object.freeze(value);
}

/**
 * Frozen provider presets for OpenAI-compatible chat and turn-gate models.
 */
export const LLM_PRESETS = deepFreeze({
  // The same OpenCode key authorizes two doors: Zen (pay-per-use, zen/v1) and
  // Go (subscription, zen/go/v1). This preset targets the Go subscription door;
  // Zen users can override llm.baseUrl through config.
  opencode: {
    baseUrl: "https://opencode.ai/zen/go/v1",
    replyModel: "deepseek-v4.1-flash",
    gateModel: "deepseek-v4.1-flash",
    headers: {
      "x-opencode-client": "discord-voice-companion",
      "x-opencode-session": "discord-voice-companion",
    },
    // Reasoning models spend the gate's 2-token budget on reasoning_content.
    // Disabled by default; re-enable through config if you need thinking.
    extraBody: { thinking: { type: "disabled" } },
  },
  deepseek: {
    baseUrl: "https://api.deepseek.com/v1",
    replyModel: "deepseek-chat",
    gateModel: "deepseek-chat",
    headers: {},
    extraBody: { thinking: { type: "disabled" } },
  },
  openai: {
    baseUrl: "https://api.openai.com/v1",
    replyModel: "gpt-4o-mini",
    gateModel: "gpt-4o-mini",
    headers: {},
    extraBody: {},
  },
  openrouter: {
    baseUrl: "https://openrouter.ai/api/v1",
    replyModel: "deepseek/deepseek-chat",
    gateModel: "deepseek/deepseek-chat",
    headers: {},
    extraBody: {},
  },
  groq: {
    baseUrl: "https://api.groq.com/openai/v1",
    replyModel: "llama-3.3-70b-versatile",
    gateModel: "llama-3.1-8b-instant",
    headers: {},
    extraBody: {},
  },
  ollama: {
    baseUrl: "http://127.0.0.1:11434/v1",
    replyModel: "llama3.2",
    gateModel: "llama3.2",
    headers: {},
    extraBody: {},
  },
  custom: {
    baseUrl: "",
    replyModel: "",
    gateModel: "",
    headers: {},
    extraBody: {},
  },
});

/**
 * Parse a .env text blob into a plain key/value map without mutating process.env.
 */
export function parseDotEnv(text) {
  if (typeof text !== "string" || text.length === 0) return {};
  const parsed = {};

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    const clean = line.startsWith("export ") ? line.slice(7).trimStart() : line;
    const eqIndex = clean.indexOf("=");
    if (eqIndex <= 0) continue;

    const key = clean.slice(0, eqIndex).trim();
    if (!key) continue;

    let value = clean.slice(eqIndex + 1).trim();
    const quoted =
      value.length >= 2 &&
      ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'")));

    if (quoted) {
      value = value.slice(1, -1);
    } else {
      value = value.replace(/\s+#.*$/, "").trim();
    }

    parsed[key] = value;
  }

  return parsed;
}

function readOptionalFile(fsImpl, filePath) {
  if (!filePath) return null;
  if (typeof fsImpl.existsSync === "function" && !fsImpl.existsSync(filePath)) {
    return null;
  }
  try {
    return fsImpl.readFileSync(filePath, "utf8");
  } catch (error) {
    if (error && error.code === "ENOENT") return null;
    throw error;
  }
}

function pickNonEmpty(...candidates) {
  for (const value of candidates) {
    if (typeof value === "string" && value.trim().length > 0) {
      return value.trim();
    }
  }
  return "";
}

/**
 * Validate an HTTP(S) endpoint URL, disallowing control chars, credentials,
 * query strings, fragments, and plaintext HTTP on non-loopback hosts.
 */
export function validateEndpointUrl(rawUrl, label = "url") {
  if (!rawUrl) return "";
  if (CONTROL_CHAR_PATTERN.test(rawUrl)) {
    throw new RangeError(`${label} contains invalid control characters`);
  }
  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new RangeError(`${label} must be a valid URL`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new RangeError(`${label} must use http: or https:`);
  }
  if (parsed.username || parsed.password) {
    throw new RangeError(`${label} must not contain credentials`);
  }
  if (rawUrl.includes("?") || rawUrl.includes("#") || parsed.search || parsed.hash) {
    throw new RangeError(`${label} must not contain query parameters or fragments`);
  }
  if (parsed.protocol === "http:" && !LOOPBACK_HOSTS.has(parsed.hostname.toLowerCase())) {
    throw new RangeError(`${label} must use https: for non-loopback hosts`);
  }
  return `${parsed.origin}${parsed.pathname}`.replace(/\/+$/, "");
}

function parseEnum(raw, allowed, fallback, label) {
  if (raw === undefined || raw === null || raw === "") return fallback;
  const value = String(raw).trim();
  if (!allowed.includes(value)) {
    throw new RangeError(`${label} must be one of [${allowed.join(", ")}] (got "${value}")`);
  }
  return value;
}

function parseBoolean(raw, fallback, label) {
  if (raw === undefined || raw === null || raw === "") return fallback;
  if (typeof raw === "boolean") return raw;
  const normalized = String(raw).trim().toLowerCase();
  if (normalized === "true" || normalized === "1") return true;
  if (normalized === "false" || normalized === "0") return false;
  throw new RangeError(`${label} must be a boolean (got "${raw}")`);
}

function parseNumberSetting(raw, fallback, { min, max, clamp = false, integer = false, label }) {
  if (raw === undefined || raw === null || raw === "") return fallback;
  const num = Number(raw);
  if (!Number.isFinite(num)) {
    throw new RangeError(`${label} must be a finite number (got "${raw}")`);
  }
  if (clamp) {
    const bounded = Math.min(max, Math.max(min, num));
    return integer ? Math.round(bounded) : bounded;
  }
  if (integer && !Number.isInteger(num)) {
    throw new RangeError(`${label} must be an integer (got "${raw}")`);
  }
  if (min !== undefined && num < min) {
    throw new RangeError(`${label} must be >= ${min} (got ${num})`);
  }
  if (max !== undefined && num > max) {
    throw new RangeError(`${label} must be <= ${max} (got ${num})`);
  }
  return num;
}

function parseAliases(envValue, jsonValue) {
  if (typeof envValue === "string" && envValue.trim().length > 0) {
    return envValue
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  if (Array.isArray(jsonValue)) {
    return jsonValue.map((item) => String(item).trim()).filter(Boolean);
  }
  return [];
}

function clonePlain(value) {
  if (Array.isArray(value)) return value.map(clonePlain);
  if (value !== null && typeof value === "object") {
    const out = {};
    for (const key of Object.keys(value)) {
      if (FORBIDDEN_KEYS.has(key)) continue;
      out[key] = clonePlain(value[key]);
    }
    return out;
  }
  return value;
}

/**
 * Load and normalize configuration from env, .env, config.json, and SOUL.md.
 * Precedence: explicit env > .env > config.json > provider/app defaults.
 */
export function loadConfig({
  env = process.env,
  configPath = "config.json",
  envPath = ".env",
  soulPath = "SOUL.md",
  fsImpl = fs,
} = {}) {
  const dotEnv = parseDotEnv(readOptionalFile(fsImpl, envPath) ?? "");
  const rawConfig = readOptionalFile(fsImpl, configPath);
  let configJson = {};
  if (rawConfig !== null && rawConfig.trim().length > 0) {
    try {
      const parsed = JSON.parse(rawConfig);
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("root value must be a JSON object");
      }
      configJson = clonePlain(parsed);
    } catch {
      throw new SyntaxError(`Invalid JSON in config file (${configPath})`);
    }
  }

  const rawSoul = readOptionalFile(fsImpl, soulPath);
  const personaPrompt = pickNonEmpty(rawSoul, configJson.personaPrompt, DEFAULT_PERSONA_PROMPT);

  const getEnv = (key) => {
    if (env && env[key] !== undefined && env[key] !== null) return String(env[key]);
    if (dotEnv[key] !== undefined) return dotEnv[key];
    return undefined;
  };

  const llmProvider = parseEnum(
    pickNonEmpty(getEnv("LLM_PROVIDER"), configJson.llm?.provider),
    PROVIDERS.llm,
    "deepseek",
    "llm.provider",
  );
  const preset = LLM_PRESETS[llmProvider];

  const sttProvider = parseEnum(
    pickNonEmpty(getEnv("STT_PROVIDER"), configJson.stt?.provider),
    PROVIDERS.stt,
    "deepgram",
    "stt.provider",
  );

  const ttsProvider = parseEnum(
    pickNonEmpty(getEnv("TTS_PROVIDER"), configJson.tts?.provider),
    PROVIDERS.tts,
    "elevenlabs",
    "tts.provider",
  );

  const expressiveTags = parseBoolean(
    getEnv("ELEVENLABS_EXPRESSIVE_TAGS") ??
      getEnv("TTS_EXPRESSIVE_TAGS") ??
      configJson.tts?.expressiveTags,
    false,
    "tts.expressiveTags",
  );

  const explicitTtsModel = pickNonEmpty(
    getEnv("ELEVENLABS_MODEL"),
    getEnv("TTS_MODEL"),
    configJson.tts?.model,
  );
  const defaultTtsModel =
    ttsProvider === "elevenlabs"
      ? expressiveTags
        ? "eleven_v4_turbo"
        : "eleven_flash_v2_5"
      : "";

  // One resolved ElevenLabs credential serves both listing and synthesis.
  const explicitElevenlabsKey = pickNonEmpty(
    getEnv("ELEVENLABS_API_KEY"),
    configJson.tts?.elevenlabsApiKey,
  );
  const elevenlabsApiKey =
    ttsProvider === "elevenlabs"
      ? pickNonEmpty(explicitElevenlabsKey, getEnv("TTS_API_KEY"), configJson.tts?.apiKey)
      : explicitElevenlabsKey;
  const ttsFallbackKey = ttsProvider === "openai" ? getEnv("OPENAI_API_KEY") : "";

  return {
    personaPrompt,
    discord: {
      token: pickNonEmpty(getEnv("DISCORD_BOT_TOKEN"), configJson.discord?.token),
      appId: pickNonEmpty(getEnv("DISCORD_APP_ID"), configJson.discord?.appId),
    },
    llm: {
      provider: llmProvider,
      baseUrl: validateEndpointUrl(
        pickNonEmpty(getEnv("LLM_BASE_URL"), configJson.llm?.baseUrl, preset.baseUrl),
        "llm.baseUrl",
      ),
      apiKey: pickNonEmpty(getEnv("LLM_API_KEY"), configJson.llm?.apiKey),
      replyModel: pickNonEmpty(
        getEnv("LLM_REPLY_MODEL"),
        configJson.llm?.replyModel,
        preset.replyModel,
      ),
      gateModel: pickNonEmpty(
        getEnv("LLM_GATE_MODEL"),
        configJson.llm?.gateModel,
        preset.gateModel,
      ),
      headers: {
        ...clonePlain(preset.headers),
        ...(configJson.llm?.headers && typeof configJson.llm.headers === "object"
          ? clonePlain(configJson.llm.headers)
          : {}),
      },
      extraBody: {
        ...clonePlain(preset.extraBody),
        ...(configJson.llm?.extraBody && typeof configJson.llm.extraBody === "object"
          ? clonePlain(configJson.llm.extraBody)
          : {}),
      },
      maxTokens: parseNumberSetting(
        getEnv("LLM_MAX_TOKENS") ?? configJson.llm?.maxTokens,
        140,
        { min: 1, integer: true, label: "llm.maxTokens" },
      ),
    },
    stt: {
      provider: sttProvider,
      deepgramApiKey: pickNonEmpty(getEnv("DEEPGRAM_API_KEY"), configJson.stt?.deepgramApiKey),
      elevenlabsApiKey: pickNonEmpty(
        getEnv("ELEVENLABS_API_KEY"),
        configJson.stt?.elevenlabsApiKey,
      ),
      openaiApiKey: pickNonEmpty(getEnv("OPENAI_API_KEY"), configJson.stt?.openaiApiKey),
      groqApiKey: pickNonEmpty(getEnv("GROQ_API_KEY"), configJson.stt?.groqApiKey),
      localUrl: validateEndpointUrl(
        pickNonEmpty(getEnv("LOCAL_STT_URL"), configJson.stt?.localUrl, "http://127.0.0.1:4373"),
        "stt.localUrl",
      ),
      language: pickNonEmpty(getEnv("STT_LANGUAGE"), configJson.stt?.language),
    },
    tts: {
      provider: ttsProvider,
      apiKey:
        ttsProvider === "elevenlabs"
          ? elevenlabsApiKey
          : pickNonEmpty(getEnv("TTS_API_KEY"), ttsFallbackKey, configJson.tts?.apiKey),
      elevenlabsApiKey,
      baseUrl: validateEndpointUrl(
        pickNonEmpty(getEnv("TTS_BASE_URL"), configJson.tts?.baseUrl),
        "tts.baseUrl",
      ),
      voiceId: pickNonEmpty(
        getEnv("ELEVENLABS_VOICE_ID"),
        getEnv("TTS_VOICE_ID"),
        configJson.tts?.voiceId,
      ),
      expressiveTags,
      model: explicitTtsModel || defaultTtsModel,
    },
    reactions: {
      mode: parseEnum(
        pickNonEmpty(getEnv("REACTION_FILLERS"), configJson.reactions?.mode),
        FILLER_MODES,
        "adaptive",
        "reactions.mode",
      ),
      fillerDelayMs: parseNumberSetting(
        getEnv("REACTION_FILLER_DELAY_MS") ?? configJson.reactions?.fillerDelayMs,
        165,
        { min: 150, max: 180, clamp: true, integer: true, label: "reactions.fillerDelayMs" },
      ),
      cooldownTurns: parseNumberSetting(
        getEnv("REACTION_COOLDOWN_TURNS") ?? configJson.reactions?.cooldownTurns,
        10,
        { min: 10, integer: true, label: "reactions.cooldownTurns" },
      ),
      cacheDir: pickNonEmpty(
        getEnv("REACTION_CACHE_DIR"),
        configJson.reactions?.cacheDir,
        ".cache/reactions",
      ),
    },
    chime: {
      mode: parseEnum(
        pickNonEmpty(getEnv("CHIME_MODE"), configJson.chime?.mode),
        CHIME_MODES,
        "smart",
        "chime.mode",
      ),
      followUpWindowMs: parseNumberSetting(
        getEnv("CHIME_FOLLOW_UP_WINDOW_MS") ?? configJson.chime?.followUpWindowMs,
        12000,
        { min: 0, integer: true, label: "chime.followUpWindowMs" },
      ),
      botAliases: parseAliases(getEnv("CHIME_BOT_ALIASES"), configJson.chime?.botAliases),
      gateTimeoutMs: parseNumberSetting(
        getEnv("CHIME_GATE_TIMEOUT_MS") ?? configJson.chime?.gateTimeoutMs,
        1500,
        { min: 1, integer: true, label: "chime.gateTimeoutMs" },
      ),
    },
    mcp: {
      enabled: parseBoolean(
        getEnv("MCP_ENABLED") ?? configJson.mcp?.enabled,
        false,
        "mcp.enabled",
      ),
      serverCommand: pickNonEmpty(getEnv("MCP_SERVER_COMMAND"), configJson.mcp?.serverCommand),
    },
  };
}

const SECRET_KEY_PATTERN = /^(token|apiKey|.*ApiKey|secret|password)$/i;
const URL_KEY_PATTERN = /^(baseUrl|localUrl)$/;

function isNonEmptySecret(value) {
  if (typeof value === "string") return value.length > 0;
  return value !== null && value !== undefined && value !== "";
}

function redactUrlCredentials(value) {
  if (typeof value !== "string" || value.length === 0) return value;
  try {
    const parsed = new URL(value);
    if (parsed.origin === "null") return "[REDACTED]";
    // Diagnostics report scheme, host and port only. Any path can carry a tenant label or a
    // token, so none is echoed.
    return parsed.origin;
  } catch {
    return "[REDACTED]";
  }
}

function redactNode(value, inHeaders = false) {
  if (Array.isArray(value)) {
    return value.map((item) => redactNode(item, false));
  }
  if (value === null || typeof value !== "object") {
    return value;
  }

  const copy = {};
  for (const key of Object.keys(value)) {
    if (FORBIDDEN_KEYS.has(key) || key === "extraBody") continue;
    const entry = value[key];
    if (inHeaders) {
      if (!isNonEmptySecret(entry)) {
        copy[key] = redactNode(entry, false);
      } else if (typeof entry === "string" && SAFE_HEADER_VALUES.get(key.toLowerCase())?.has(entry)) {
        copy[key] = entry;
      } else {
        copy[key] = "[REDACTED]";
      }
    } else if (SECRET_KEY_PATTERN.test(key) && isNonEmptySecret(entry)) {
      copy[key] = "[REDACTED]";
    } else if (URL_KEY_PATTERN.test(key) && typeof entry === "string" && entry.length > 0) {
      copy[key] = redactUrlCredentials(entry);
    } else {
      copy[key] = redactNode(entry, key === "headers");
    }
  }
  return copy;
}

/**
 * Return a deep copy of config with credentials and authorization headers masked.
 */
export function redactConfig(config) {
  return redactNode(config, false);
}
