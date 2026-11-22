// Slash command definitions and the interaction handler. Every reply is
// ephemeral and every failure is a fixed generic message, so a token or stack
// trace can never reach Discord.

import { CHIME_MODES, FILLER_MODES } from "../types.mjs";

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
]);

/** One compact ephemeral status block with no secrets. */
export function formatStatusReply(status = {}) {
  const channel = status.channelId ? ` in <#${status.channelId}>` : "";
  return [
    `Session: ${status.active ? "active" : "idle"}${channel}`,
    `Chime mode: ${status.chimeMode ?? "unknown"}`,
    `Filler mode: ${status.fillerMode ?? "unknown"}`,
    `DAVE E2EE: ${status.daveReady ? "ready" : "not ready"}`,
    `Latency: ${status.latencySummary ?? "No turns recorded yet."}`,
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

export function createCommandHandler({ orchestrator } = {}) {
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
          const adapterCreator =
            interaction?.guild?.voiceAdapterCreator ??
            interaction?.member?.guild?.voiceAdapterCreator;
          await orchestrator.joinChannel({ guildId, channelId, adapterCreator });
          return await reply(interaction, "Joined your voice channel.");
        }
        case "leave": {
          await orchestrator.leaveChannel({ guildId });
          return await reply(interaction, "Left the voice channel.");
        }
        case "mode": {
          if (!hasModeratorPermission(interaction)) {
            return await reply(interaction, "You need permission to change modes.");
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
          const userId = optionValue(interaction, "user");
          if (typeof userId !== "string" || userId.length === 0) {
            return await reply(interaction, "A user ID is required.");
          }
          const muted = optionValue(interaction, "muted");
          orchestrator.muteUser(userId, muted === undefined ? true : Boolean(muted));
          return await reply(interaction, "Mute state updated.");
        }
        default:
          return await reply(interaction, "Unknown command.");
      }
    } catch {
      return await reply(interaction, "That command could not be completed.");
    }
  }

  return { handleInteraction };
}
