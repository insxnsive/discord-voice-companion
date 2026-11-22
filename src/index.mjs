import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { LatencyTracker } from "./types.mjs";
import { loadConfig, redactConfig } from "./config/config.mjs";
import { createLlmClient } from "./llm/client.mjs";
import { createTtsAdapter } from "./tts/adapters.mjs";
import { ReactionPool } from "./tts/reactions.mjs";

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
  createLlmClientImpl = createLlmClient,
  createTtsAdapterImpl = createTtsAdapter,
  createSttAdapterImpl,
  createVoiceOrchestratorImpl,
  createCommandHandlerImpl,
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
  });

  const latencyTracker = new LatencyTracker();
  const llmClient = createLlmClientImpl({ config });
  const ttsAdapter = createTtsAdapterImpl(config);

  const missingRuntimeModules = [];

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
        client: resolvedClient,
        voiceDeps,
      });
    } else {
      missingRuntimeModules.push("./discord/session.mjs");
    }
  }

  let commandHandler = null;
  if (typeof createCommandHandlerImpl === "function") {
    commandHandler = createCommandHandlerImpl({
      config,
      reactionPool,
      latencyTracker,
      orchestrator,
      client: resolvedClient,
    });
  } else {
    const commandMod = await loadOptionalModule("./discord/commands.mjs");
    if (commandMod && typeof commandMod.createCommandHandler === "function") {
      commandHandler = commandMod.createCommandHandler({
        config,
        reactionPool,
        latencyTracker,
        orchestrator,
        client: resolvedClient,
      });
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
    interactionListener = (interaction) => commandHandler.handleInteraction(interaction);
    resolvedClient.on("interactionCreate", interactionListener);
  }

  let stopped = false;
  let stoppingPromise = null;
  let activeStart = null;

  async function start() {
    if (stopped || stoppingPromise) {
      throw new Error("Companion has already been stopped");
    }
    if (activeStart) return activeStart;

    activeStart = (async () => {
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
      if (stopped || stoppingPromise) {
        throw new Error("Companion stopped during startup");
      }
      await ttsAdapter?.warmup?.().catch(() => {});
      if (stopped || stoppingPromise) {
        throw new Error("Companion stopped during startup");
      }
      await orchestrator?.start?.({ client: resolvedClient });
      return {
        started: true,
        config: redactedConfig,
        loadedReactionClips: reactionPool.size,
      };
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
        if (typeof resolvedClient.off === "function") {
          resolvedClient.off("interactionCreate", interactionListener);
        } else if (typeof resolvedClient.removeListener === "function") {
          resolvedClient.removeListener("interactionCreate", interactionListener);
        }
        interactionListener = null;
      }
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
