import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { LatencyTracker } from "./types.mjs";
import { createOpusTranscoder } from "./audio/pcm.mjs";
import { loadConfig, redactConfig } from "./config/config.mjs";
import { createLlmClient } from "./llm/client.mjs";
import { createTtsAdapter } from "./tts/adapters.mjs";
import { ReactionPool } from "./tts/reactions.mjs";

const DISCORD_OPTION_TYPE_MAP = Object.freeze({
  SUB_COMMAND: 1,
  SUB_COMMAND_GROUP: 2,
  STRING: 3,
  INTEGER: 4,
  BOOLEAN: 5,
  USER: 6,
  CHANNEL: 7,
  ROLE: 8,
  MENTIONABLE: 9,
  NUMBER: 10,
  ATTACHMENT: 11,
});

function normalizeChoice(choice) {
  if (typeof choice === "string") {
    return { name: String(choice), value: choice };
  }
  if (choice && typeof choice === "object") {
    return { ...choice };
  }
  return choice;
}

function normalizeCommandOption(option) {
  if (!option || typeof option !== "object") return option;
  const normalized = { ...option };
  if (typeof normalized.type === "string") {
    const mapped = DISCORD_OPTION_TYPE_MAP[normalized.type.toUpperCase()];
    if (mapped !== undefined) {
      normalized.type = mapped;
    }
  }
  if (Array.isArray(option.choices)) {
    normalized.choices = option.choices.map(normalizeChoice);
  }
  if (Array.isArray(option.options)) {
    normalized.options = option.options.map(normalizeCommandOption);
  }
  return normalized;
}

/**
 * Normalize slash command option types and choice entries into cloned Discord API-compatible objects.
 */
export function normalizeSlashCommands(commands = []) {
  if (!Array.isArray(commands)) return [];
  return commands.map((command) => {
    if (!command || typeof command !== "object") return command;
    const cloned = { ...command };
    if (Array.isArray(command.options)) {
      cloned.options = command.options.map(normalizeCommandOption);
    }
    return cloned;
  });
}

async function loadOptionalModule(specifier) {
  try {
    return await import(specifier);
  } catch (error) {
    if (error && error.code === "ERR_MODULE_NOT_FOUND") {
      return null;
    }
    throw error;
  }
}

/**
 * Assemble all companion adapters, caches, and lifecycle hooks with optional test dependency injection.
 */
export async function createCompanionApp({
  env = process.env,
  configPath = "config.json",
  envPath = ".env",
  soulPath = "SOUL.md",
  trustedRootDir = process.cwd(),
  fsImpl = fs,
  createDiscordClientImpl,
  createOpusTranscoderImpl = createOpusTranscoder,
  createLlmClientImpl = createLlmClient,
  createTtsAdapterImpl = createTtsAdapter,
  createSttAdapterImpl,
  createVoiceOrchestratorImpl,
  createCommandHandlerImpl,
  opusTranscoder: providedOpusTranscoder,
  slashCommands: providedSlashCommands,
  client,
  voiceDeps,
} = {}) {
  const config = loadConfig({ env, configPath, envPath, soulPath, fsImpl });
  const redactedConfig = redactConfig(config);

  const resolvedTrustedRoot = path.resolve(trustedRootDir);
  const cacheDir = path.resolve(resolvedTrustedRoot, config.reactions.cacheDir);

  const reactionPool = new ReactionPool({
    cooldownTurns: config.reactions.cooldownTurns,
  });
  reactionPool.loadFromDirectory(cacheDir, {
    fsImpl,
    trustedRootDir: resolvedTrustedRoot,
    // Missing cache components are created at 0o700 through the verified walk, so the
    // private directory is the only writable thing below the trusted root.
    createDir: true,
  });

  const latencyTracker = new LatencyTracker();
  const llmClient = createLlmClientImpl({ config });
  const ttsAdapter = createTtsAdapterImpl(config);

  const missingRuntimeModules = [];

  let opusTranscoder = providedOpusTranscoder ?? null;
  if (!opusTranscoder && typeof createOpusTranscoderImpl === "function") {
    try {
      opusTranscoder = await createOpusTranscoderImpl();
    } catch (error) {
      if (createOpusTranscoderImpl !== createOpusTranscoder) {
        throw error;
      }
      missingRuntimeModules.push("@discordjs/opus|opusscript");
    }
  }

  let resolvedClient = client ?? null;
  if (!resolvedClient) {
    if (typeof createDiscordClientImpl === "function") {
      resolvedClient = createDiscordClientImpl({ config });
    } else if (config.discord.token) {
      const discordMod = await loadOptionalModule("discord.js");
      if (discordMod?.Client && discordMod?.GatewayIntentBits) {
        resolvedClient = new discordMod.Client({
          intents: [
            discordMod.GatewayIntentBits.Guilds,
            discordMod.GatewayIntentBits.GuildVoiceStates,
          ],
        });
      } else {
        missingRuntimeModules.push("discord.js");
      }
    }
  }

  let sttAdapter = null;
  if (typeof createSttAdapterImpl === "function") {
    sttAdapter = createSttAdapterImpl(config);
  } else {
    const sttMod = await loadOptionalModule("./stt/adapters.mjs");
    if (sttMod && typeof sttMod.createSttAdapter === "function") {
      sttAdapter = sttMod.createSttAdapter(config);
    } else {
      missingRuntimeModules.push("./stt/adapters.mjs");
    }
  }

  let orchestrator = null;
  if (typeof createVoiceOrchestratorImpl === "function") {
    orchestrator = createVoiceOrchestratorImpl({
      config,
      reactionPool,
      latencyTracker,
      llmClient,
      ttsAdapter,
      sttAdapter,
      opusTranscoder,
      client: resolvedClient,
      voiceDeps,
    });
  } else {
    const sessionMod = await loadOptionalModule("./discord/session.mjs");
    const sessionFactory =
      sessionMod?.createVoiceSessionOrchestrator ?? sessionMod?.createVoiceOrchestrator;
    if (typeof sessionFactory === "function") {
      orchestrator = sessionFactory({
        config,
        reactionPool,
        latencyTracker,
        llmClient,
        ttsAdapter,
        sttAdapter,
        opusTranscoder,
        client: resolvedClient,
        voiceDeps,
      });
    } else {
      missingRuntimeModules.push("./discord/session.mjs");
    }
  }

  let commandHandler = null;
  const createVoiceChooserForTts = async () => {
    if (config?.tts?.provider !== "elevenlabs") return null;
    const commandsMod = await loadOptionalModule("./discord/commands.mjs");
    if (typeof commandsMod?.createVoiceChooser !== "function") return null;
    return commandsMod.createVoiceChooser({
      adapter: ttsAdapter,
      apiKey: config?.tts?.elevenlabsApiKey,
      provider: config.tts.provider,
    });
  };
  let resolvedSlashCommands = Array.isArray(providedSlashCommands) ? providedSlashCommands : null;
  if (typeof createCommandHandlerImpl === "function") {
    commandHandler = createCommandHandlerImpl({
      config,
      reactionPool,
      latencyTracker,
      orchestrator,
      client: resolvedClient,
      voiceChooser: await createVoiceChooserForTts(),
    });
    if (!resolvedSlashCommands && Array.isArray(commandHandler?.commands)) {
      resolvedSlashCommands = commandHandler.commands;
    }
  } else {
    const commandMod = await loadOptionalModule("./discord/commands.mjs");
    if (commandMod && typeof commandMod.createCommandHandler === "function") {
      commandHandler = commandMod.createCommandHandler({
        config,
        reactionPool,
        latencyTracker,
        orchestrator,
        client: resolvedClient,
        voiceChooser:
          config?.tts?.provider === "elevenlabs" &&
          typeof commandMod.createVoiceChooser === "function"
            ? commandMod.createVoiceChooser({
                adapter: ttsAdapter,
                apiKey: config?.tts?.elevenlabsApiKey,
                provider: config.tts.provider,
              })
            : null,
      });
      if (!resolvedSlashCommands && Array.isArray(commandMod.SLASH_COMMANDS)) {
        resolvedSlashCommands = commandMod.SLASH_COMMANDS;
      } else if (!resolvedSlashCommands && Array.isArray(commandHandler?.commands)) {
        resolvedSlashCommands = commandHandler.commands;
      }
    } else {
      missingRuntimeModules.push("./discord/commands.mjs");
    }
  }

  let interactionListener = null;
  if (
    resolvedClient &&
    typeof resolvedClient.on === "function" &&
    commandHandler &&
    typeof commandHandler.handleInteraction === "function"
  ) {
    interactionListener = (interaction) => {
      if (typeof interaction?.isAutocomplete === "function" && interaction.isAutocomplete()) {
        if (typeof commandHandler.handleAutocomplete === "function") {
          return commandHandler.handleAutocomplete(interaction);
        }
        return undefined;
      }
      return commandHandler.handleInteraction(interaction);
    };
    resolvedClient.on("interactionCreate", interactionListener);
  }

  let readyListeners = [];

  function detachListener(event, fn) {
    if (!resolvedClient || !fn) return;
    if (typeof resolvedClient.off === "function") {
      resolvedClient.off(event, fn);
    } else if (typeof resolvedClient.removeListener === "function") {
      resolvedClient.removeListener(event, fn);
    }
  }

  function clearReadyListeners() {
    for (const { event, fn } of readyListeners) {
      detachListener(event, fn);
    }
    readyListeners = [];
  }

  let stopped = false;
  let stoppingPromise = null;
  let activeStart = null;

  async function registerSlashCommands() {
    if (!resolvedClient || !Array.isArray(resolvedSlashCommands) || resolvedSlashCommands.length === 0) {
      return;
    }
    const normalizedCommands = normalizeSlashCommands(resolvedSlashCommands);
    if (typeof resolvedClient.application?.commands?.set === "function") {
      await resolvedClient.application.commands.set(normalizedCommands);
      return;
    }
    if (typeof resolvedClient.once === "function") {
      clearReadyListeners();
      let registered = false;
      const onReady = () => {
        if (registered || stopped || stoppingPromise) return;
        registered = true;
        clearReadyListeners();
        Promise.resolve(resolvedClient.application?.commands?.set?.(normalizedCommands)).catch(
          () => {},
        );
      };
      for (const event of ["ready", "clientReady"]) {
        readyListeners.push({ event, fn: onReady });
        resolvedClient.once(event, onReady);
      }
    }
  }

  async function ensureNotStoppedDuringStartup() {
    if (!stopped && !stoppingPromise) return;
    if (stoppingPromise) await stoppingPromise;
    else await stop();
    throw new Error("Companion app stopped during startup");
  }

  async function start() {
    if (stopped || stoppingPromise) {
      throw new Error("Companion has already been stopped");
    }
    if (activeStart) return activeStart;

    activeStart = (async () => {
      try {
        if (!orchestrator && missingRuntimeModules.length > 0) {
          throw new Error(
            `Missing runtime module(s) required to start companion: ${missingRuntimeModules.join(", ")}`,
          );
        }
        if (!orchestrator && config.discord.token && !resolvedClient && missingRuntimeModules.includes("discord.js")) {
          throw new Error("discord.js package is required when DISCORD_BOT_TOKEN is configured");
        }
        if (typeof resolvedClient?.login === "function" && config.discord.token) {
          await resolvedClient.login(config.discord.token);
        }
        await ensureNotStoppedDuringStartup();
        await registerSlashCommands();
        await ensureNotStoppedDuringStartup();
        await ttsAdapter?.warmup?.().catch(() => {});
        await ensureNotStoppedDuringStartup();
        await orchestrator?.start?.({ client: resolvedClient });
        await ensureNotStoppedDuringStartup();
        return {
          started: true,
          config: redactedConfig,
          loadedReactionClips: reactionPool.size,
        };
      } catch (error) {
        clearReadyListeners();
        throw error;
      }
    })();

    try {
      return await activeStart;
    } finally {
      activeStart = null;
    }
  }

  async function stop() {
    if (stopped) return;
    if (stoppingPromise) return stoppingPromise;
    stoppingPromise = (async () => {
      if (resolvedClient && interactionListener) {
        detachListener("interactionCreate", interactionListener);
        interactionListener = null;
      }
      clearReadyListeners();
      await Promise.allSettled([
        Promise.resolve().then(() => resolvedClient?.destroy?.()),
        Promise.resolve().then(() => orchestrator?.close?.()),
        Promise.resolve().then(() => ttsAdapter?.close?.()),
        Promise.resolve().then(() => sttAdapter?.close?.()),
      ]);
      stopped = true;
    })();
    return stoppingPromise;
  }

  return {
    config: redactedConfig,
    client: resolvedClient,
    reactionPool,
    latencyTracker,
    llmClient,
    ttsAdapter,
    sttAdapter,
    opusTranscoder,
    orchestrator,
    commandHandler,
    start,
    stop,
  };
}

/**
 * Create and start the companion app, optionally hooking SIGINT/SIGTERM for clean shutdown.
 */
export async function startCompanion({ registerSignals = false, ...options } = {}) {
  const app = await createCompanionApp(options);
  await app.start();

  if (registerSignals) {
    const handleSignal = () => {
      app.stop().finally(() => {
        process.exit(0);
      });
    };
    process.once("SIGINT", handleSignal);
    process.once("SIGTERM", handleSignal);
  }

  return app;
}

const isCliEntry =
  typeof process !== "undefined" &&
  Array.isArray(process.argv) &&
  Boolean(process.argv[1]) &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isCliEntry) {
  const app = await startCompanion({ registerSignals: true });
  const diagnostic = {
    status: "running",
    llmProvider: app.config.llm.provider,
    sttProvider: app.config.stt.provider,
    ttsProvider: app.config.tts.provider,
    reactionClips: app.reactionPool.size,
  };
  process.stdout.write(`${JSON.stringify(diagnostic)}\n`);
}
