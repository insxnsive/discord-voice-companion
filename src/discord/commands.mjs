// Slash command definitions and the interaction handler. Every reply is
// ephemeral and every failure is a fixed generic message, so a token or stack
// trace can never reach Discord.

import { CHIME_MODES, FILLER_MODES } from "../types.mjs";
import { fetchElevenLabsVoices } from "../tts/adapters.mjs";

const MODERATOR_PERMISSIONS = Object.freeze(["ManageGuild", "ModerateMembers"]);

function freezeCommand(command) {
  const options = (command.options ?? []).map((option) =>
    Object.freeze({
      ...option,
      ...(option.choices ? { choices: Object.freeze([...option.choices]) } : {}),
    }),
  );
  return Object.freeze({ ...command, options: Object.freeze(options) });
}

export const SLASH_COMMANDS = Object.freeze([
  freezeCommand({
    name: "join",
    description: "Join your current voice channel.",
  }),
  freezeCommand({
    name: "leave",
    description: "Leave the voice channel.",
  }),
  freezeCommand({
    name: "mode",
    description: "Set the chime or reaction filler mode.",
    options: [
      {
        name: "chime",
        description: "When the bot may speak.",
        type: "STRING",
        required: false,
        choices: CHIME_MODES,
      },
      {
        name: "filler",
        description: "Reaction filler style.",
        type: "STRING",
        required: false,
        choices: FILLER_MODES,
      },
    ],
  }),
  freezeCommand({
    name: "status",
    description: "Show session, mode, DAVE, and latency status.",
  }),
  freezeCommand({
    name: "mute-user",
    description: "Ignore or un-ignore a user in voice.",
    options: [
      {
        name: "user",
        description: "Discord user ID.",
        type: "STRING",
        required: true,
      },
      {
        name: "muted",
        description: "Mute state, defaults to true.",
        type: "BOOLEAN",
        required: false,
      },
    ],
  }),
  freezeCommand({
    name: "choose-voice",
    description: "Pick the bot's voice from the voices on this account.",
    options: [
      {
        name: "voice",
        description: "Voice name, searched live from your ElevenLabs account.",
        type: "STRING",
        required: true,
        autocomplete: true,
      },
    ],
  }),
]);

const VOICES_CACHE_TTL_MS = 60_000;
const VOICES_CACHE_MAX_TTL_MS = 300_000;

// List voices on demand and apply a pick to the live adapter. Picks target the
// elevenlabs WS adapter, which can reopen its socket on the new voice; the
// HTTP-stream adapter has no setVoiceId and fails cleanly. The cache keeps
// autocomplete typing from firing one provider request per keystroke, and a
// single in-flight promise collapses concurrent refreshes.
export function createVoiceChooser({
  adapter,
  apiKey,
  provider = "elevenlabs",
  fetchImpl = fetch,
  ttlMs = VOICES_CACHE_TTL_MS,
  now = () => Date.now(),
} = {}) {
  const safeTtl =
    Number.isFinite(ttlMs) && ttlMs > 0
      ? Math.min(ttlMs, VOICES_CACHE_MAX_TTL_MS)
      : VOICES_CACHE_TTL_MS;
  let cache = null;
  let inFlight = null;

  function refresh() {
    if (provider !== "elevenlabs") {
      return Promise.reject(new Error("voice switching requires the ElevenLabs provider"));
    }
    if (inFlight) return inFlight;
    inFlight = (async () => {
      const voices = await fetchElevenLabsVoices({ apiKey, fetchImpl });
      cache = { voices, at: now() };
      return voices;
    })().finally(() => {
      inFlight = null;
    });
    return inFlight;
  }

  return {
    async list() {
      if (cache && now() - cache.at < safeTtl) return cache.voices;
      return await refresh();
    },
    select(voiceId) {
      if (typeof voiceId !== "string" || voiceId.length === 0) {
        throw new Error("voice is required");
      }
      if (!cache) throw new Error("voices have not been listed yet");
      if (!cache.voices.some((voice) => voice.voiceId === voiceId)) {
        throw new Error("unknown voice");
      }
      if (typeof adapter?.setVoiceId !== "function") {
        throw new Error("live voice switching is unavailable");
      }
      adapter.setVoiceId(voiceId);
      return true;
    },
  };
}

/** One compact ephemeral status block with no secrets. */
export function formatStatusReply(status = {}) {
  const channel = status.channelId ? ` in <#${status.channelId}>` : "";
  return [
    `Session: ${status.active ? "active" : "idle"}${channel}`,
    `Chime mode: ${status.chimeMode ?? "unknown"}`,
    `Filler mode: ${status.fillerMode ?? "unknown"}`,
    `DAVE E2EE: ${status.daveReady ? "ready" : "not ready"}`,
    `Latency: ${status.latencySummary ?? "No turns recorded yet."}`,
    ...(status.timingSummary ? [`Timing: ${status.timingSummary}`] : []),
  ].join("\n");
}

function hasModeratorPermission(interaction) {
  if (!interaction?.guildId) return false;
  for (const source of [interaction.memberPermissions, interaction.member?.permissions]) {
    if (!source) continue;
    if (typeof source.has === "function") {
      if (source.has("ManageGuild") || source.has("ModerateMembers")) return true;
      continue;
    }
    if (Array.isArray(source)) {
      if (source.includes("ManageGuild") || source.includes("ModerateMembers")) return true;
    }
  }
  return false;
}

function optionValue(interaction, name) {
  const options = interaction?.options;
  if (!options) return undefined;
  if (typeof options.get === "function") {
    const entry = options.get(name);
    if (entry !== undefined && entry !== null) return entry.value ?? entry;
  }
  for (const method of ["getString", "getBoolean", "getInteger"]) {
    if (typeof options[method] === "function") {
      const value = options[method](name);
      if (value !== undefined && value !== null) return value;
    }
  }
  return options[name];
}

export function createCommandHandler({ orchestrator, voiceChooser = null } = {}) {
  if (!orchestrator) {
    throw new TypeError("createCommandHandler requires an orchestrator");
  }

  async function reply(interaction, content) {
    const payload = { content, ephemeral: true };
    if (typeof interaction?.reply === "function") return await interaction.reply(payload);
    return payload;
  }

  async function handleInteraction(interaction = {}) {
    try {
      const guildId = interaction?.guildId;
      if (typeof guildId !== "string" || guildId.length === 0) {
        return await reply(interaction, "This command only works inside a server.");
      }
      const status = orchestrator.getStatus?.();
      if (status?.guildId && status.guildId !== guildId) {
        return await reply(interaction, "This voice session belongs to another server.");
      }

      const commandName = interaction.commandName;
      const command = SLASH_COMMANDS.find((entry) => entry.name === commandName);
      if (!command) return await reply(interaction, "Unknown command.");

      switch (commandName) {
        case "join": {
          const channelId = interaction?.member?.voice?.channelId;
          if (!channelId) return await reply(interaction, "Join a voice channel first.");
          if (
            status?.active &&
            status?.channelId &&
            status.channelId !== channelId &&
            !hasModeratorPermission(interaction)
          ) {
            return await reply(
              interaction,
              "Join the active voice channel or use a moderator role to move the bot.",
            );
          }
          const adapterCreator =
            interaction?.guild?.voiceAdapterCreator ??
            interaction?.member?.guild?.voiceAdapterCreator;
          await orchestrator.joinChannel({ guildId, channelId, adapterCreator });
          return await reply(interaction, "Joined your voice channel.");
        }
        case "leave": {
          const memberChannelId = interaction?.member?.voice?.channelId;
          if (
            status?.active &&
            status?.channelId &&
            memberChannelId !== status.channelId &&
            !hasModeratorPermission(interaction)
          ) {
            return await reply(
              interaction,
              "You must be in the active voice channel or have moderator permission to disconnect the bot.",
            );
          }
          await orchestrator.leaveChannel({ guildId });
          return await reply(interaction, "Left the voice channel.");
        }
        case "mode": {
          if (!hasModeratorPermission(interaction)) {
            return await reply(interaction, "You need permission to change modes.");
          }
          if (!status?.active || status?.guildId !== guildId) {
            return await reply(interaction, "No active voice session in this server.");
          }
          const chime = optionValue(interaction, "chime");
          const filler = optionValue(interaction, "filler");
          if (chime === undefined && filler === undefined) {
            return await reply(interaction, "Choose a chime or filler mode.");
          }
          if (chime !== undefined) {
            if (!CHIME_MODES.includes(chime)) {
              return await reply(interaction, "Invalid chime mode.");
            }
            orchestrator.setChimeMode(chime);
          }
          if (filler !== undefined) {
            if (!FILLER_MODES.includes(filler)) {
              return await reply(interaction, "Invalid filler mode.");
            }
            orchestrator.setFillerMode(filler);
          }
          return await reply(interaction, "Mode updated.");
        }
        case "status": {
          return await reply(interaction, formatStatusReply(orchestrator.getStatus()));
        }
        case "mute-user": {
          if (!hasModeratorPermission(interaction)) {
            return await reply(interaction, "You need permission to mute users.");
          }
          if (!status?.active || status?.guildId !== guildId) {
            return await reply(interaction, "No active voice session in this server.");
          }
          const userId = optionValue(interaction, "user");
          if (typeof userId !== "string" || userId.length === 0) {
            return await reply(interaction, "A user ID is required.");
          }
          const muted = optionValue(interaction, "muted");
          orchestrator.muteUser(userId, muted === undefined ? true : Boolean(muted));
          return await reply(interaction, "Mute state updated.");
        }
        case "choose-voice": {
          if (!status?.active || status?.guildId !== guildId) {
            return await reply(interaction, "No active voice session in this server.");
          }
          if (!voiceChooser || typeof voiceChooser.select !== "function") {
            return await reply(interaction, "Voice switching is not configured.");
          }
          const voiceId = optionValue(interaction, "voice");
          if (typeof voiceId !== "string" || voiceId.length === 0) {
            return await reply(interaction, "Choose a voice from the list.");
          }
          try {
            await voiceChooser.select(voiceId);
            return await reply(interaction, "Voice updated.");
          } catch {
            return await reply(interaction, "Could not change the voice right now.");
          }
        }
        default:
          return await reply(interaction, "Unknown command.");
      }
    } catch {
      return await reply(interaction, "That command could not be completed.");
    }
  }

  async function respondAutocomplete(interaction, results) {
    if (typeof interaction?.respond === "function") {
      try {
        await interaction.respond(results);
      } catch {
        // The interaction may already be closed; the returned list still stands.
      }
    }
    return results;
  }

  async function handleAutocomplete(interaction = {}) {
    try {
      if (interaction.commandName !== "choose-voice") {
        return await respondAutocomplete(interaction, []);
      }
      const guildId = interaction?.guildId;
      const status = orchestrator.getStatus?.();
      if (
        typeof guildId !== "string" ||
        guildId.length === 0 ||
        !status?.active ||
        status?.guildId !== guildId
      ) {
        return await respondAutocomplete(interaction, []);
      }
      if (!voiceChooser || typeof voiceChooser.list !== "function") {
        return await respondAutocomplete(interaction, []);
      }
      const focused = String(interaction.options?.getFocused?.() ?? "").toLowerCase();
      const voices = await voiceChooser.list();
      const after = orchestrator.getStatus?.();
      if (!after?.active || after.guildId !== guildId) {
        return await respondAutocomplete(interaction, []);
      }
      const results = voices
        .filter((voice) => voice.name.toLowerCase().includes(focused))
        .slice(0, 25)
        .map((voice) => ({ name: voice.name.slice(0, 100), value: voice.voiceId }));
      return await respondAutocomplete(interaction, results);
    } catch {
      return await respondAutocomplete(interaction, []);
    }
  }

  return { handleInteraction, handleAutocomplete };
}
