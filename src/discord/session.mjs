// Discord voice session pipeline: DAVE prerequisite verification, Opus ingress
// and egress, barge-in cancellation, and the STT -> filler -> gate -> LLM -> TTS
// turn. Discord gateway wiring is injected or lazily loaded.

import { PassThrough } from "node:stream";

import {
  CHIME_MODES,
  FILLER_MODES,
  LatencyTracker,
  TurnTimingTrace,
  assertAdapterContract,
} from "../types.mjs";
import {
  createFrameAccumulator,
  createOpusTranscoder,
  createUpsampler24kTo48kStereo,
  downsample48kStereoTo16kMono,
} from "../audio/pcm.mjs";
import { PerSpeakerVad, createRmsSpeechDetector } from "../audio/vad.mjs";
import { isWhisperHallucination } from "../stt/adapters.mjs";
import { createClauseSplitter, formatSpeakerLabel } from "../llm/client.mjs";
import { scheduleReactionFiller } from "../tts/reactions.mjs";
import { redactConfig } from "../config/config.mjs";

export const REQUIRED_GATEWAY_INTENTS = Object.freeze(["Guilds", "GuildVoiceStates"]);
export const REQUIRED_VOICE_PERMISSIONS = Object.freeze(["Connect", "Speak"]);

const MIN_NODE = "22.12.0";
const MAX_SPEAKER_CACHE = 256;
const MAX_SPEAKER_DECODERS = 32;
const MAX_SPEAKER_FETCHES = 64;
const SPEAKER_NEGATIVE_TTL_MS = 60000;
const DAVE_MODULES = Object.freeze(["@discordjs/voice", "@snazzah/davey"]);
const MAX_PCM_CHUNK_BYTES = 245760;
const MAX_CLAUSES_PER_TURN = 24;
const MAX_TURN_AUDIO_BYTES = 16 * 1024 * 1024;
const MAX_QUEUED_OPUS_FRAMES = 1500;
const MAX_PENDING_TURNS = 8;

function parseNodeVersion(version) {
  const [major = 0, minor = 0] = String(version)
    .replace(/^v/, "")
    .split(".")
    .map((part) => Number.parseInt(part, 10) || 0);
  return { major, minor };
}

export async function verifyDavePrerequisites({
  nodeVersion = process.versions.node,
  loadModule = async (id) => import(id),
} = {}) {
  const { major, minor } = parseNodeVersion(nodeVersion);
  if (major < 22 || (major === 22 && minor < 12)) {
    throw new Error(`Node.js ${nodeVersion} is too old: DAVE E2EE requires Node >= ${MIN_NODE}`);
  }

  for (const moduleId of DAVE_MODULES) {
    try {
      await loadModule(moduleId);
    } catch {
      throw new Error(`${moduleId} is not installed. Run npm install.`);
    }
  }

  let opusBackend = "opusscript";
  try {
    await loadModule("@discordjs/opus");
    opusBackend = "native";
  } catch {
    opusBackend = "opusscript";
  }

  return { nodeVersion, daveReady: true, opusBackend };
}

function boundedHistory(history, maxHistoryTurns) {
  const maxMessages = Math.max(2, Math.floor(maxHistoryTurns) * 2);
  while (history.length > maxMessages) history.shift();
}

export function createVoiceOrchestrator({
  config = {},
  botUserId = "",
  client = null,
  voiceDeps = null,
  loadVoiceDeps = async () => import("@discordjs/voice"),
  sttAdapter,
  llmClient,
  ttsAdapter,
  reactionPool,
  latencyTracker = new LatencyTracker(),
  opusTranscoder = null,
  createOpusTranscoderImpl = createOpusTranscoder,
  vadFactory,
  createClauseSplitterImpl = createClauseSplitter,
  scheduleReactionFillerImpl = scheduleReactionFiller,
  verifyDaveImpl = verifyDavePrerequisites,
  playerSink = null,
  now = () => performance.now(),
  maxHistoryTurns = 12,
} = {}) {
  assertAdapterContract(sttAdapter, ["name", "transcribe", "close"], "stt adapter");
  assertAdapterContract(ttsAdapter, ["name", "synthesizeClause", "close"], "tts adapter");
  assertAdapterContract(llmClient, ["shouldChimeIn", "streamReply"], "llm client");
  if (reactionPool && typeof reactionPool.pick !== "function") {
    throw new TypeError("reactionPool must expose pick()");
  }
  if (
    opusTranscoder &&
    (typeof opusTranscoder.encode !== "function" ||
      typeof opusTranscoder.decode !== "function")
  ) {
    throw new TypeError("opusTranscoder must expose encode() and decode()");
  }

  if (
    config?.turnDeadlineMs !== undefined &&
    (!Number.isFinite(config.turnDeadlineMs) || config.turnDeadlineMs <= 0)
  ) {
    throw new RangeError("turnDeadlineMs must be a positive finite number");
  }

  let resolvedBotUserId = botUserId || client?.user?.id || "";
  let transcoder = opusTranscoder;
  let transcoderPromise = null;
  let resolvedVoiceDeps = voiceDeps ?? null;
  let joinNonce = 0;
  let pendingGuildId = null;
  let activePlaybackUserId = null;
  let activePlaybackTurn = null;
  let activePlaybackResource = null;
  let activePlaybackStream = null;
  const pendingTurns = new Set();
  const voicePresence = new Set();
  const recentVoiceArrivals = [];
  const MAX_VOICE_ROSTER = 64;
  const MAX_RECENT_ARRIVALS = 8;

  const speakerBotCache = new Map();
  const speakerFetchInFlight = new Map();
  const speakerDecoders = new Map();
  const timingTrace = new TurnTimingTrace();

  const state = {
    active: false,
    guildId: null,
    channelId: null,
    daveReady: false,
    chimeMode: CHIME_MODES.includes(config?.chime?.mode) ? config.chime.mode : "smart",
    fillerMode: FILLER_MODES.includes(config?.reactions?.mode)
      ? config.reactions.mode
      : "adaptive",
    mutedUsers: new Set(),
    isBotSpeaking: false,
    lastBotReplyAt: null,
    lastBotReplyUserId: null,
    lastError: null,
    sessionEpoch: 0,
    sessionToken: null,
    closed: false,
    client,
  };

  let history = [];
  let currentTurn = null;
  let committedTurnOrder = null;
  let turnCounter = 0;
  let voiceConnection = null;
  let voicePlayer = null;
  let speakingListener = null;
  let voiceStateListener = null;
  let connectionStateListener = null;
  let playerStateListener = null;
  let connectionErrorListener = null;
  let playerErrorListener = null;
  const userStreams = new Map();

  async function ensureTranscoder() {
    if (transcoder) return transcoder;
    if (!transcoderPromise) transcoderPromise = createOpusTranscoderImpl();
    transcoder = await transcoderPromise;
    return transcoder;
  }

  function verifyLiveChannelBinding() {
    if (!voiceConnection) return true;
    const joinChannelId = voiceConnection.joinConfig?.channelId;
    if (joinChannelId !== undefined && joinChannelId !== null && joinChannelId !== state.channelId) {
      return false;
    }
    if (joinChannelId === null && state.channelId) return false;
    if (resolvedBotUserId && state.guildId) {
      const botMember = state.client?.guilds?.cache
        ?.get?.(state.guildId)
        ?.members?.cache?.get?.(resolvedBotUserId);
      if (
        botMember?.voice?.channelId !== undefined &&
        botMember.voice.channelId !== state.channelId
      ) {
        return false;
      }
    }
    return true;
  }

  function captureSessionSnapshot() {
    return {
      epoch: state.sessionEpoch,
      token: state.sessionToken,
      connection: voiceConnection,
      guildId: state.guildId,
      channelId: state.channelId,
    };
  }

  function isSnapshotCurrent(snap) {
    return Boolean(
      snap &&
        state.active &&
        !state.closed &&
        state.daveReady &&
        state.sessionEpoch === snap.epoch &&
        state.sessionToken === snap.token &&
        voiceConnection === snap.connection &&
        state.guildId === snap.guildId &&
        state.channelId === snap.channelId &&
        verifyLiveChannelBinding(),
    );
  }

  function isStaleTurnOrder(turn) {
    if (!committedTurnOrder) return false;
    if (turn.endedAt !== committedTurnOrder.endedAt) {
      return turn.endedAt < committedTurnOrder.endedAt;
    }
    return turn.seq < committedTurnOrder.seq;
  }

  function isTurnLive(turn) {
    return Boolean(
      turn &&
        !state.closed &&
        state.active &&
        verifyLiveChannelBinding() &&
        state.sessionEpoch === turn.sessionEpoch &&
        state.guildId === turn.guildId &&
        state.channelId === turn.channelId &&
        (turn.committed === false || currentTurn === turn) &&
        !turn.controller.signal.aborted &&
        !state.mutedUsers.has(turn.userId),
    );
  }

  function finishOutput(turn) {
    if (!turn?.outStream) return;
    try {
      turn.outStream.end?.();
    } catch {
      // The output stream may already be closed.
    }
    turn.outStream = null;
  }

  function stopPlayback(expectedTurn = null) {
    if (expectedTurn && !isPlaybackOwner(expectedTurn)) return false;
    finishOutput(expectedTurn ?? activePlaybackTurn ?? currentTurn);
    activePlaybackUserId = null;
    activePlaybackTurn = null;
    activePlaybackResource = null;
    activePlaybackStream = null;
    try {
      playerSink?.stop?.();
      playerSink?.clear?.();
    } catch {
      // Playback sinks are best effort during teardown.
    }
    try {
      voicePlayer?.stop?.(true);
    } catch {
      // The player may have already been destroyed.
    }
    return true;
  }

  function isPlaybackOwner(turn) {
    return Boolean(
      turn &&
      activePlaybackTurn === turn &&
      activePlaybackTurn.sessionEpoch === state.sessionEpoch,
    );
  }

  function refreshSinkPlaybackOwner() {
    if (typeof playerSink?.playOpusFrame !== "function" || typeof playerSink?.isPlaying !== "function") return false;
    let playing;
    try {
      playing = playerSink.isPlaying();
    } catch {
      return false;
    }
    if (playing !== false) return false;
    activePlaybackUserId = null;
    activePlaybackTurn = null;
    activePlaybackResource = null;
    activePlaybackStream = null;
    state.isBotSpeaking = false;
    return true;
  }

  function stopOwnedPlayback(turn) {
    if (!isPlaybackOwner(turn)) return false;
    state.isBotSpeaking = false;
    return stopPlayback(turn);
  }

  function sinkIsPlaying() {
    return Boolean(playerSink?.isPlaying?.() || voicePlayer?.state?.status === "playing");
  }

  const vadCallbacks = {
    onSpeechStart: () => {},
    onSpeechEnd: (event) => handleSpeechEnd(event),
    onBargeIn: (event) => handleBargeIn(event),
  };

  const vad = vadFactory
    ? vadFactory(vadCallbacks)
    : new PerSpeakerVad({
        botUserId: resolvedBotUserId,
        detectorFactory: () => createRmsSpeechDetector(),
        isBotSpeaking: () => Boolean(state.isBotSpeaking || sinkIsPlaying()),
        now,
        ...vadCallbacks,
      });

  function resolveDisplayName(userId, guildId) {
    const member = state.client?.guilds?.cache?.get?.(guildId)?.members?.cache?.get?.(userId);
    return (
      member?.displayName ??
      member?.user?.username ??
      state.client?.users?.cache?.get?.(userId)?.username ??
      "Unknown"
    );
  }

  function clearVoicePresence() {
    voicePresence.clear();
    recentVoiceArrivals.length = 0;
  }

  function setVoicePresence(userId, { arrival = false } = {}) {
    if (!userId || userId === resolvedBotUserId) return;
    if (voicePresence.has(userId) || voicePresence.size >= MAX_VOICE_ROSTER) return;
    voicePresence.add(userId);
    if (arrival) {
      recentVoiceArrivals.push(userId);
      while (recentVoiceArrivals.length > MAX_RECENT_ARRIVALS) recentVoiceArrivals.shift();
    }
  }

  function seedVoicePresence(guildId, channelId) {
    clearVoicePresence();
    const guild = state.client?.guilds?.cache?.get?.(guildId);
    const states = guild?.voiceStates?.cache;
    if (states && typeof states.values === "function") {
      for (const voiceState of states.values()) {
        const userId = voiceState?.id ?? voiceState?.member?.id;
        if (!userId || voiceState?.channelId !== channelId) continue;
        if (isVoiceStateBot(voiceState, userId, guildId)) continue;
        setVoicePresence(userId);
      }
    }
    const members = guild?.members?.cache;
    if (members && typeof members.values === "function") {
      for (const member of members.values()) {
        const userId = member?.id ?? member?.user?.id;
        if (!userId || member?.voice?.channelId !== channelId || isVoiceStateBot({ member }, userId, guildId)) {
          continue;
        }
        setVoicePresence(userId);
      }
    }
  }

  function voicePresenceMessage() {
    const participantCount = voicePresence.size;
    let recentArrivalCount = 0;
    for (const userId of recentVoiceArrivals) {
      if (voicePresence.has(userId)) recentArrivalCount += 1;
    }
    if (participantCount === 0 && recentArrivalCount === 0) return null;
    return {
      role: "user",
      content: `Voice room presence (ephemeral; counts only): ${JSON.stringify({
        participantCount,
        recentArrivalCount,
      })}`,
    };
  }

  function isVoiceStateBot(voiceState, userId, guildId) {
    const member =
      voiceState?.member ??
      state.client?.guilds?.cache?.get?.(guildId)?.members?.cache?.get?.(userId);
    return Boolean(member?.user?.bot ?? member?.bot ?? lookupBot(userId, guildId));
  }

  function handleHumanVoiceState(oldState, newState) {
    const userId = newState?.id ?? oldState?.id ?? newState?.member?.id ?? oldState?.member?.id;
    const guildId = newState?.guildId ?? newState?.guild?.id ?? oldState?.guildId ?? oldState?.guild?.id;
    if (!userId || userId === resolvedBotUserId || guildId !== state.guildId) return;
    if (isVoiceStateBot(newState, userId, guildId) || isVoiceStateBot(oldState, userId, guildId)) return;
    if (!Object.hasOwn(newState ?? {}, "channelId")) return;
    const beforeChannelId = Object.hasOwn(oldState ?? {}, "channelId") ? oldState.channelId : null;
    const afterChannelId = newState?.channelId;
    if (beforeChannelId === afterChannelId || (beforeChannelId == null && afterChannelId == null)) return;

    if (afterChannelId === state.channelId && beforeChannelId !== state.channelId) {
      setVoicePresence(userId, { arrival: true });
    } else if (beforeChannelId === state.channelId && afterChannelId !== state.channelId) {
      voicePresence.delete(userId);
      for (let index = recentVoiceArrivals.length - 1; index >= 0; index -= 1) {
        if (recentVoiceArrivals[index] === userId) recentVoiceArrivals.splice(index, 1);
      }
    }
  }

  function lookupBot(userId, guildId) {
    const usersCache = state.client?.users?.cache;
    if (usersCache && typeof usersCache.get === "function") {
      const user = usersCache.get(userId);
      if (user && typeof user.bot === "boolean") return user.bot;
    }
    const member = state.client?.guilds?.cache?.get?.(guildId)?.members?.cache?.get?.(userId);
    if (member?.user && typeof member.user.bot === "boolean") return member.user.bot;
    return undefined;
  }

  function rememberSpeaker(userId, isBot, expiresAt) {
    if (speakerBotCache.size >= MAX_SPEAKER_CACHE) {
      const oldest = speakerBotCache.keys().next().value;
      speakerBotCache.delete(oldest);
    }
    speakerBotCache.set(userId, { isBot, expiresAt });
  }

  function cachedSpeakerBot(userId) {
    const entry = speakerBotCache.get(userId);
    if (!entry) return undefined;
    if (entry.expiresAt > now()) return entry.isBot;
    speakerBotCache.delete(userId);
    return undefined;
  }

  async function ensureSpeakerKnown(userId, guildId) {
    const usersCache = state.client?.users?.cache;
    const guildsCache = state.client?.guilds?.cache;
    if (!usersCache && !guildsCache) return false;

    const cached = cachedSpeakerBot(userId);
    if (cached !== undefined) return cached;

    const known = lookupBot(userId, guildId);
    if (known !== undefined) {
      rememberSpeaker(userId, known, Infinity);
      return known;
    }

    const inFlight = speakerFetchInFlight.get(userId);
    if (inFlight) return await inFlight;
    if (speakerFetchInFlight.size >= MAX_SPEAKER_FETCHES) return true;

    const pending = (async () => {
      const lookupEpoch = state.sessionEpoch;
      try {
        const guild = guildsCache?.get?.(guildId);
        let fetched = null;
        if (typeof guild?.members?.fetch === "function") {
          fetched = await guild.members.fetch(userId);
        } else if (typeof state.client?.users?.fetch === "function") {
          fetched = await state.client.users.fetch(userId);
        }
        if (fetched) {
          const isBot = Boolean(fetched.user?.bot ?? fetched.bot);
          if (state.sessionEpoch === lookupEpoch && !state.closed) {
            rememberSpeaker(userId, isBot, Infinity);
          }
          return isBot;
        }
      } catch {
        // Fall through to the negative cache.
      }
      if (state.sessionEpoch === lookupEpoch && !state.closed) {
        rememberSpeaker(userId, true, now() + SPEAKER_NEGATIVE_TTL_MS);
      }
      return true;
    })();

    speakerFetchInFlight.set(userId, pending);
    try {
      return await pending;
    } finally {
      if (speakerFetchInFlight.get(userId) === pending) speakerFetchInFlight.delete(userId);
    }
  }

  function detachReceiver() {
    const speaking = voiceConnection?.receiver?.speaking;
    if (speaking && speakingListener) {
      try {
        speaking.off?.("start", speakingListener);
      } catch {
        // A throwing receiver removal must not stop teardown.
      }
      try {
        speaking.removeListener?.("start", speakingListener);
      } catch {
        // Best effort.
      }
    }
    speakingListener = null;

    for (const stream of userStreams.values()) {
      try {
        stream.destroy?.();
      } catch {
        // Stream teardown is best effort.
      }
    }
    userStreams.clear();
  }

  function bindReceiver(boundGuildId, boundChannelId, boundToken) {
    const speaking = voiceConnection?.receiver?.speaking;
    if (!speaking || typeof speaking.on !== "function") return;
    speakingListener = (userId) => {
      Promise.resolve(onSpeakingStart(userId, boundGuildId, boundChannelId, boundToken)).catch(
        () => {},
      );
    };
    speaking.on("start", speakingListener);
  }

  function detachConnectionListeners() {
    if (voiceConnection) {
      if (voiceStateListener && state.client?.off) {
        try {
          state.client.off("voiceStateUpdate", voiceStateListener);
        } catch {
          // A throwing removal hook must not stop teardown.
        }
        try {
          state.client.removeListener?.("voiceStateUpdate", voiceStateListener);
        } catch {
          // Best effort.
        }
      }
      if (connectionStateListener) {
        try {
          voiceConnection.off?.("stateChange", connectionStateListener);
        } catch {
          // Best effort.
        }
        try {
          voiceConnection.removeListener?.("stateChange", connectionStateListener);
        } catch {
          // Best effort.
        }
      }
      if (connectionErrorListener) {
        try {
          voiceConnection.off?.("error", connectionErrorListener);
        } catch {
          // Best effort.
        }
        try {
          voiceConnection.removeListener?.("error", connectionErrorListener);
        } catch {
          // Best effort.
        }
      }
    }
    if (voicePlayer) {
      if (playerStateListener) {
        try {
          voicePlayer.off?.("stateChange", playerStateListener);
        } catch {
          // Best effort.
        }
        try {
          voicePlayer.removeListener?.("stateChange", playerStateListener);
        } catch {
          // Best effort.
        }
      }
      if (playerErrorListener) {
        try {
          voicePlayer.off?.("error", playerErrorListener);
        } catch {
          // Best effort.
        }
        try {
          voicePlayer.removeListener?.("error", playerErrorListener);
        } catch {
          // Best effort.
        }
      }
    }
    voiceStateListener = null;
    connectionStateListener = null;
    playerStateListener = null;
    connectionErrorListener = null;
    playerErrorListener = null;
  }

  function destroyVoice() {
    detachReceiver();
    detachConnectionListeners();

    finishOutput(currentTurn);
    try {
      voicePlayer?.stop?.(true);
    } catch {
      // Player teardown is best effort.
    }
    try {
      voiceConnection?.destroy?.();
    } catch {
      // Connection teardown is best effort.
    }
    voicePlayer = null;
    voiceConnection = null;
    activePlaybackUserId = null;
    activePlaybackTurn = null;
    activePlaybackResource = null;
    activePlaybackStream = null;
  }

  function clearGuildScopedState() {
    state.mutedUsers.clear();
    vad.setIgnoredUsers?.([]);
    state.chimeMode = CHIME_MODES.includes(config?.chime?.mode) ? config.chime.mode : "smart";
    state.fillerMode = FILLER_MODES.includes(config?.reactions?.mode)
      ? config.reactions.mode
      : "adaptive";
  }

  function resetTurnAndSession() {
    clearVoicePresence();
    for (const pending of pendingTurns) {
      try {
        timingTrace.setStopReason(pending.seq, "session_reset");
      } catch {
        // The trace may already be trimmed.
      }
      pending.controller.abort();
      pending.filler?.cancel();
      clearTimeout(pending.deadline);
      finishOutput(pending);
      if (!pending.committed && pending.fillerEmitted && currentTurn === null && isPlaybackOwner(pending)) {
        state.isBotSpeaking = false;
        stopPlayback(pending);
      }
    }
    clearTimeout(currentTurn?.deadline);
    if (currentTurn) {
      try {
        timingTrace.setStopReason(currentTurn.seq, "session_reset");
      } catch {
        // The trace may already be trimmed.
      }
    }
    currentTurn?.controller.abort();
    currentTurn?.filler?.cancel();
    finishOutput(currentTurn);
    currentTurn = null;
    committedTurnOrder = null;
    state.isBotSpeaking = false;
    stopPlayback();
    vad.clear?.();
    history = [];
    state.lastBotReplyAt = null;
    state.lastBotReplyUserId = null;
    for (const decoder of speakerDecoders.values()) {
      try {
        decoder?.close?.();
      } catch {
        // Decoder teardown is best effort.
      }
    }
    speakerDecoders.clear();
    speakerBotCache.clear();
    timingTrace.reset();
    state.sessionEpoch += 1;
    state.sessionToken = null;
  }

  function handleExternalVoiceTransition(nextChannelId) {
    if (state.closed) return;
    const guildId = state.guildId;
    resetTurnAndSession();
    detachReceiver();

    if (!nextChannelId) {
      destroyVoice();
      state.guildId = null;
      state.channelId = null;
      // External disconnect is a guild boundary: drop moderation and modes so
      // a later join cannot inherit them.
      state.mutedUsers.clear();
      vad.setIgnoredUsers?.([]);
      state.chimeMode = CHIME_MODES.includes(config?.chime?.mode) ? config.chime.mode : "smart";
      state.fillerMode = FILLER_MODES.includes(config?.reactions?.mode)
        ? config.reactions.mode
        : "adaptive";
      return;
    }
    if (!guildId) return;
    state.guildId = guildId;
    state.channelId = nextChannelId;
    state.active = true;
    state.sessionToken = Symbol("voice-session");
    seedVoicePresence(guildId, nextChannelId);
    bindReceiver(guildId, nextChannelId, state.sessionToken);
  }

  async function onSpeakingStart(userId, boundGuildId, boundChannelId, boundToken) {
    try {
      if (!userId || userId === resolvedBotUserId) return;
      if (state.closed || state.sessionToken !== boundToken) return;
      if (state.guildId !== boundGuildId || state.channelId !== boundChannelId) return;
      if (!verifyLiveChannelBinding()) return;
      if (state.mutedUsers.has(userId)) return;

      const snap = captureSessionSnapshot();
      if (await ensureSpeakerKnown(userId, boundGuildId)) return;
      if (
        !isSnapshotCurrent(snap) ||
        snap.token !== boundToken ||
        snap.guildId !== boundGuildId ||
        snap.channelId !== boundChannelId ||
        state.mutedUsers.has(userId) ||
        userStreams.has(userId)
      ) {
        return;
      }

      const codec = await ensureTranscoder();
      if (!isSnapshotCurrent(snap)) return;
      let streamDecoder = speakerDecoders.get(userId);
      if (streamDecoder) {
        speakerDecoders.delete(userId);
        speakerDecoders.set(userId, streamDecoder);
      } else {
        if (speakerDecoders.size >= MAX_SPEAKER_DECODERS) {
          let oldestId = null;
          for (const id of speakerDecoders.keys()) {
            if (!userStreams.has(id)) {
              oldestId = id;
              break;
            }
          }
          if (!oldestId) {
            // Every cached decoder is still serving a live stream. Decline the
            // new subscription rather than closing a decoder in use.
            return;
          }
          const evicted = speakerDecoders.get(oldestId);
          speakerDecoders.delete(oldestId);
          try {
            evicted?.close?.();
          } catch {
            // Decoder teardown is best effort.
          }
        }
        streamDecoder = codec?.createDecoder ? codec.createDecoder() : codec;
        speakerDecoders.set(userId, streamDecoder);
      }

      const EndBehaviorType = resolvedVoiceDeps?.EndBehaviorType;
      const stream = voiceConnection?.receiver?.subscribe?.(userId, {
        end: { behavior: EndBehaviorType?.AfterSilence ?? 1, duration: 240 },
      });
      if (!stream) return;

      userStreams.set(userId, stream);
      const cleanup = () => {
        if (userStreams.get(userId) === stream) userStreams.delete(userId);
      };

      stream.on?.("data", (chunk) => {
        Promise.resolve(
          handleOpusPacket({
            sessionToken: boundToken,
            guildId: boundGuildId,
            channelId: boundChannelId,
            userId,
            displayName: resolveDisplayName(userId, boundGuildId),
            opusPacket: chunk,
            decoder: streamDecoder,
          }),
        ).catch(() => {});
      });
      stream.on?.("end", () => {
        cleanup();
        if (isSnapshotCurrent(snap) && !state.mutedUsers.has(userId)) {
          Promise.resolve(vad.flushSpeaker?.(userId, now())).catch(() => {});
        }
      });
      stream.on?.("close", cleanup);
      stream.on?.("error", cleanup);
    } catch {
      state.lastError = "voice receiver failed";
    }
  }

  function ensureTurnStream(turn) {
    if (playerSink?.playOpusFrame) return null;
    if (!voicePlayer) return null;
    if (
      turn.outStream &&
      activePlaybackTurn === turn &&
      activePlaybackStream === turn.outStream &&
      !turn.outStream.destroyed &&
      !turn.outStream.writableEnded &&
      !turn.outStream.writableFinished
    ) return turn.outStream;
    if (turn.outStream) finishOutput(turn);

    const stream = new PassThrough({ objectMode: true });
    const resource = resolvedVoiceDeps?.createAudioResource
      ? resolvedVoiceDeps.createAudioResource(stream, {
          inputType: resolvedVoiceDeps?.StreamType?.Opus ?? "opus",
        })
      : stream;
    turn.outStream = stream;
    turn.resourceOrdinal = timingTrace.assignResource(turn.seq);
    activePlaybackUserId = turn.userId;
    activePlaybackTurn = turn;
    activePlaybackResource = resource;
    activePlaybackStream = stream;
    voicePlayer.play?.(resource);
    return stream;
  }

  function playPacket(turn, packet, { isFiller = false } = {}) {
    if (isFiller && activePlaybackTurn && activePlaybackTurn !== turn) return;
    turn.queuedOpusFrames += 1;
    if (turn.queuedOpusFrames > MAX_QUEUED_OPUS_FRAMES) {
      state.lastError = "turn frame limit exceeded";
      timingTrace.setStopReason(turn.seq, "frame_cap");
      turn.controller.abort();
      stopPlayback();
      return;
    }
    if (activePlaybackTurn !== turn) {
      activePlaybackResource = null;
      activePlaybackStream = null;
    }
    activePlaybackUserId = turn.userId;
    activePlaybackTurn = turn;
    if (playerSink?.playOpusFrame) {
      playerSink.playOpusFrame(packet);
      if (!isFiller) {
        turn.replyFrameEmitted = true;
        if (!turn.firstAcceptedMarked) {
          timingTrace.mark(turn.seq, "firstAcceptedFrameMs", Math.max(0, now() - turn.startedAt));
          timingTrace.mark(turn.seq, "frameBackpressure", false);
          turn.firstAcceptedMarked = true;
        }
        turn.lastQueuedFrameAt = now();
      }
      return;
    }
    const stream = ensureTurnStream(turn);
    if (stream?.write && !stream.destroyed && !stream.writableEnded && !stream.writableFinished) {
      const accepted = stream.write(packet);
      if (!isFiller) {
        turn.replyFrameEmitted = true;
        if (!turn.firstAcceptedMarked) {
          timingTrace.mark(turn.seq, "firstAcceptedFrameMs", Math.max(0, now() - turn.startedAt));
          timingTrace.mark(turn.seq, "frameBackpressure", !accepted);
          turn.firstAcceptedMarked = true;
        }
        turn.lastQueuedFrameAt = now();
      }
      if (!accepted && (stream.readableLength ?? 0) > MAX_QUEUED_OPUS_FRAMES) {
        state.lastError = "turn frame limit exceeded";
        timingTrace.setStopReason(turn.seq, "frame_cap");
        turn.controller.abort();
        stopPlayback();
      }
    } else if (!stream) {
      activePlaybackUserId = null;
      activePlaybackTurn = null;
    }
  }

  function emitFrame(turn, packet, { isFiller = false } = {}) {
    if (turn.controller.signal.aborted) return;
    if (!isFiller && !turn.firstFrameMarked) {
      latencyTracker.mark(turn.id, "opusFirstFrameAt", now());
      turn.firstFrameMarked = true;
    }
    playPacket(turn, packet, { isFiller });
  }

  function emitPcm48(turn, pcm48, options) {
    const buffer = Buffer.isBuffer(pcm48) ? pcm48 : Buffer.from(pcm48);
    for (let offset = 0; offset < buffer.length; offset += MAX_PCM_CHUNK_BYTES) {
      const slice = buffer.subarray(offset, Math.min(offset + MAX_PCM_CHUNK_BYTES, buffer.length));
      for (const frame of turn.accumulator.push(slice)) {
        emitFrame(turn, transcoder.encode(frame), options);
      }
    }
  }

  function flushFrames(turn, options) {
    const tail = turn.accumulator.flushPadded();
    if (tail) emitFrame(turn, transcoder.encode(tail), options);
  }

  function emitClip(clip, turn) {
    if (!clip || turn.controller.signal.aborted) return;
    if (Array.isArray(clip.frames48kOpus) && clip.frames48kOpus.length > 0) {
      for (const packet of clip.frames48kOpus) emitFrame(turn, packet, { isFiller: true });
      return;
    }
    if (clip.pcm24k) {
      emitPcm48(turn, turn.upsampler.push(clip.pcm24k), { isFiller: true });
      flushFrames(turn, { isFiller: true });
    }
  }

  function abortTurn(turn = currentTurn) {
    if (!turn) return;
    const ownsPlayback = isPlaybackOwner(turn);
    if (ownsPlayback) state.isBotSpeaking = false;
    try {
      latencyTracker.mark(turn.id, "bargeInCancelled", true);
    } catch {
      // The turn may already be finished.
    }
    turn.controller.abort();
    turn.filler?.cancel();
    clearTimeout(turn.deadline);
    if (ownsPlayback) {
      stopPlayback(turn);
    } else if (!activePlaybackTurn) {
      state.isBotSpeaking = false;
      stopPlayback();
    }
  }

  function handleBargeIn(event) {
    refreshSinkPlaybackOwner();
    if (typeof event?.userId !== "string" || event.userId.length === 0) return;
    if (currentTurn?.userId === event.userId) {
      timingTrace.setStopReason(currentTurn.seq, "barge_in_same_speaker");
      abortTurn(currentTurn);
      return;
    }
    const playbackTurn = activePlaybackTurn;
    if (playbackTurn?.userId !== event.userId) return;
    timingTrace.setStopReason(playbackTurn.seq, "barge_in_same_speaker");
    if (playbackTurn.committed === false) abortTurn(playbackTurn);
    else stopOwnedPlayback(playbackTurn);
  }

  async function speakClause(clause, turn) {
    if (!clause || !isTurnLive(turn)) return;
    if (!turn.fillerCancelled) {
      turn.filler?.cancel();
      turn.fillerCancelled = true;
    }
    const clauseAt = now();
    if (!turn.llmMarked) {
      latencyTracker.mark(turn.id, "llmFirstClauseAt", clauseAt);
      turn.llmMarked = true;
    }
    if (!turn.firstClauseMarked) {
      timingTrace.mark(turn.seq, "firstClauseMs", Math.max(0, clauseAt - turn.startedAt));
      turn.firstClauseMarked = true;
    }

    await ttsAdapter.synthesizeClause({
      text: clause,
      signal: turn.controller.signal,
      onPcmChunk: (pcm24) => {
        if (!isTurnLive(turn)) return;
        turn.pcmBytes += Buffer.isBuffer(pcm24) ? pcm24.length : Buffer.byteLength(pcm24);
        if (turn.pcmBytes > MAX_TURN_AUDIO_BYTES) {
          state.lastError = "turn audio limit exceeded";
          timingTrace.setStopReason(turn.seq, "audio_cap");
          turn.controller.abort();
          stopPlayback();
          return;
        }
        if (!turn.ttsMarked) {
          const pcmAt = now();
          latencyTracker.mark(turn.id, "ttsFirstByteAt", pcmAt);
          timingTrace.mark(turn.seq, "firstPcmMs", Math.max(0, pcmAt - turn.startedAt));
          turn.ttsMarked = true;
        }
        state.isBotSpeaking = true;
        emitPcm48(turn, turn.upsampler.push(pcm24));
      },
    });

    if (isTurnLive(turn)) flushFrames(turn);
  }

  async function handleSpeechEnd({ userId, displayName, pcm16kMono, endedAt, lastVoicedAt } = {}) {
    if (state.closed) return;
    if (pendingTurns.size >= MAX_PENDING_TURNS) return;
    for (const pending of pendingTurns) {
      if (pending.committed === false && !pending.controller.signal.aborted && pending.userId === userId) {
        pending.controller.abort();
        pending.filler?.cancel();
        clearTimeout(pending.deadline);
        finishOutput(pending);
        if (!currentTurn && pending.fillerEmitted && isPlaybackOwner(pending)) {
          state.isBotSpeaking = false;
          stopPlayback(pending);
        }
      }
    }

    const turnSeq = (turnCounter += 1);
    const turnId = `turn-${turnSeq}`;
    const controller = new AbortController();

    const callbackAt = now();
    const startedAt = endedAt === undefined ? callbackAt : endedAt;
    const turn = {
      id: turnId,
      seq: turnSeq,
      startedAt,
      endedAt: startedAt,
      userId,
      controller,
      filler: null,
      fillerCancelled: false,
      upsampler: createUpsampler24kTo48kStereo(),
      accumulator: createFrameAccumulator(),
      firstFrameMarked: false,
      llmMarked: false,
      pcmBytes: 0,
      queuedOpusFrames: 0,
      ttsMarked: false,
      replyFrameEmitted: false,
      outStream: null,
      sessionEpoch: state.sessionEpoch,
      guildId: state.guildId,
      channelId: state.channelId,
      fillerEmitted: false,
    };
    pendingTurns.add(turn);
    turn.committed = false;
    turn.voiceToken = undefined;
    timingTrace.startTurn(turnSeq, {
      lastVoicedAt: Number.isFinite(lastVoicedAt) ? lastVoicedAt : null,
      endedAt: startedAt,
      callbackAt,
    });
    latencyTracker.startTurn(turnId, startedAt);

    const turnDeadlineMs = Number.isFinite(config?.turnDeadlineMs)
      ? config.turnDeadlineMs
      : 10000;
    turn.deadline = setTimeout(() => {
      if (turn.controller.signal.aborted) return;
      timingTrace.setStopReason(turn.seq, "deadline");
      turn.controller.abort();
      turn.filler?.cancel();
      finishOutput(turn);
      if (
        state.closed || !state.active || state.sessionEpoch !== turn.sessionEpoch ||
        state.guildId !== turn.guildId || state.channelId !== turn.channelId
      ) return;
      if (currentTurn === turn) {
        state.lastError = "turn deadline exceeded";
        if (isPlaybackOwner(turn)) {
          state.isBotSpeaking = false;
          stopPlayback(turn);
        }
      } else if (currentTurn === null) {
        if (!activePlaybackTurn || isPlaybackOwner(turn)) {
          state.lastError = "turn deadline exceeded";
        }
        if (turn.fillerEmitted && isPlaybackOwner(turn)) {
          state.isBotSpeaking = false;
          stopPlayback(turn);
        }
      }
    }, turnDeadlineMs);

    const fillerDelay =
      state.fillerMode === "off" ? 0 : (config?.reactions?.fillerDelayMs ?? 165);
    let ttsError = null;

    try {
      await ensureTranscoder();
      if (!isTurnLive(turn)) return;

      const transcript = await sttAdapter.transcribe(pcm16kMono, { signal: controller.signal });
      if (!isTurnLive(turn)) return;
      const sttAt = now();
      latencyTracker.mark(turnId, "sttDoneAt", sttAt);
      timingTrace.mark(turnSeq, "sttMs", Math.max(0, sttAt - startedAt));
      if (
        typeof transcript !== "string" ||
        transcript.trim().length === 0 ||
        isWhisperHallucination(transcript)
      ) {
        return;
      }

      turn.filler = scheduleReactionFillerImpl({
        delayMs: fillerDelay,
        onFire: () => {
          try {
            refreshSinkPlaybackOwner();
            if (
              !isTurnLive(turn) ||
              isStaleTurnOrder(turn) ||
              (currentTurn && currentTurn !== turn) ||
              (activePlaybackTurn && activePlaybackTurn !== turn) ||
              turn.gateDenied ||
              turn.ttsMarked
            ) return;
            const clip = reactionPool?.pick(transcript, { mode: state.fillerMode });
            if (!clip) return;
            turn.fillerEmitted = true;
            latencyTracker.mark(turnId, "fillerFiredAt", now());
            state.isBotSpeaking = true;
            emitClip(clip, turn);
            if (!isPlaybackOwner(turn) && !activePlaybackTurn) state.isBotSpeaking = false;
          } catch {
            state.lastError = "reaction filler failed";
          }
        },
      });

      const decision = await llmClient.shouldChimeIn({
        transcript,
        chimeMode: state.chimeMode,
        speakerName: displayName,
        userId,
        botAliases: config?.chime?.botAliases,
        lastBotReplyAt: state.lastBotReplyAt,
        lastBotReplyUserId: state.lastBotReplyUserId,
        followUpWindowMs: config?.chime?.followUpWindowMs,
        gateTimeoutMs: config?.chime?.gateTimeoutMs,
        signal: controller.signal,
        now,
      });
      if (!isTurnLive(turn)) return;
      const gateAt = now();
      latencyTracker.mark(turnId, "gateDoneAt", gateAt);
      timingTrace.mark(turnSeq, "gateMs", Math.max(0, gateAt - sttAt));
      const chimeIn = decision?.chimeIn ?? decision?.shouldReply;
      timingTrace.setGateReason(
        turnSeq,
        chimeIn !== true
          ? chimeIn === false
            ? "no_reply"
            : "unknown"
          : isStaleTurnOrder(turn)
            ? "no_reply"
            : "reply",
      );
      if (chimeIn !== true || isStaleTurnOrder(turn)) {
        turn.gateDenied = true;
        turn.filler?.cancel();
        if (!turn.committed && turn.fillerEmitted && currentTurn === null && isPlaybackOwner(turn)) {
          state.isBotSpeaking = false;
          stopPlayback(turn);
        }
        return;
      }

      if (currentTurn && currentTurn !== turn) {
        timingTrace.setStopReason(currentTurn.seq, "newer_turn");
        currentTurn.controller.abort();
        currentTurn.filler?.cancel();
        clearTimeout(currentTurn.deadline);
        finishOutput(currentTurn);
        stopPlayback();
      }
      committedTurnOrder = { endedAt: turn.endedAt, seq: turn.seq };
      currentTurn = turn;
      turn.committed = true;
      pendingTurns.delete(turn);
      try {
        turn.voiceToken = ttsAdapter.beginTurn?.();
      } catch {
        // A voice hook must never block the turn or its cleanup.
        turn.voiceToken = undefined;
      }

      const speakerLabel = formatSpeakerLabel(displayName);
      history.push({ role: "user", content: `[${speakerLabel}]: ${transcript}` });
      boundedHistory(history, maxHistoryTurns);

      let receivedOnClause = false;
      let enqueuedClauses = 0;
      let tokenBuffer = "";
      let chain = Promise.resolve();
      const enqueueClause = (clause) => {
        if (!clause || enqueuedClauses >= MAX_CLAUSES_PER_TURN) return;
        enqueuedClauses += 1;
        chain = chain
          .then(() => speakClause(clause, turn))
          .catch((error) => {
            ttsError ??= error;
          });
      };
      const splitter = createClauseSplitterImpl({
        expressiveTags: Boolean(config?.tts?.expressiveTags),
        now,
        onClause: enqueueClause,
      });

      const reply = await llmClient.streamReply({
        messages: (() => {
          const messages = history.slice();
          const presence = voicePresenceMessage();
          if (presence) messages.splice(Math.max(0, messages.length - 1), 0, presence);
          return messages;
        })(),
        signal: controller.signal,
        onClause: (clause) => {
          receivedOnClause = true;
          enqueueClause(clause);
        },
        onToken: (delta) => {
          if (!delta) return;
          if (!turn.firstTokenMarked) {
            timingTrace.mark(turnSeq, "firstContentTokenMs", Math.max(0, now() - startedAt));
            turn.firstTokenMarked = true;
          }
          if (!receivedOnClause) tokenBuffer += delta;
        },      });
      if (!isTurnLive(turn)) return;

      if (!receivedOnClause && tokenBuffer) {
        splitter?.push?.(tokenBuffer);
        const beforeFlush = enqueuedClauses;
        const tail = splitter?.flush?.() ?? [];
        if (enqueuedClauses === beforeFlush) {
          for (const clause of tail) enqueueClause(clause);
        }
      }
      await chain;

      if (!isTurnLive(turn)) return;
      if (ttsError && !controller.signal.aborted) {
        state.lastError = "voice turn failed";
        timingTrace.setStopReason(turnSeq, "provider_error");
      }

      if (typeof reply?.fullText === "string" && reply.fullText.trim().length > 0) {
        history.push({ role: "assistant", content: reply.fullText });
        boundedHistory(history, maxHistoryTurns);
      }
      if (!ttsError && turn.replyFrameEmitted) {
        state.lastBotReplyAt = now();
        state.lastBotReplyUserId = userId ?? null;
      }
    } catch (error) {
      if (!controller.signal.aborted && error?.name !== "AbortError") {
        state.lastError = "voice turn failed";
        timingTrace.setStopReason(turnSeq, "provider_error");
        if (!turn.committed) timingTrace.setGateReason(turnSeq, "error");
      }
    } finally {
      pendingTurns.delete(turn);
      if (!turn.committed && turn.fillerEmitted && currentTurn === null && isPlaybackOwner(turn)) {
        state.isBotSpeaking = false;
        stopPlayback(turn);
      }
      try {
        ttsAdapter.endTurn?.(turn.voiceToken);
      } catch {
        // A voice hook must never skip deadline clearing or latency finalization.
      }
      if (currentTurn === turn) {
        currentTurn = null;
        state.isBotSpeaking = false;
      }
      turn.filler?.cancel();
      clearTimeout(turn.deadline);
      finishOutput(turn);
      try {
        latencyTracker.finishTurn(turnId);
      } catch {
        // Already finished by a racing close.
      }
    }
  }

  async function handleOpusPacket({
    sessionToken,
    guildId,
    channelId,
    userId,
    displayName,
    isBot = false,
    opusPacket,
    decoder,
    timestamp,
  } = {}) {
    if (
      !state.active ||
      state.closed ||
      !state.daveReady ||
      !state.guildId ||
      !state.channelId ||
      !state.sessionToken ||
      !verifyLiveChannelBinding()
    ) {
      return { ignored: true, dropped: "inactive_session" };
    }

    const tokenMatches = sessionToken !== undefined && sessionToken === state.sessionToken;
    if (sessionToken !== undefined && !tokenMatches) {
      return { ignored: true, dropped: "wrong_guild" };
    }
    const idsMatch =
      typeof guildId === "string" &&
      guildId.length > 0 &&
      guildId === state.guildId &&
      typeof channelId === "string" &&
      channelId.length > 0 &&
      channelId === state.channelId;

    if (!tokenMatches && !idsMatch) {
      return { ignored: true, dropped: "wrong_guild" };
    }
    if (tokenMatches) {
      if (guildId !== undefined && guildId !== state.guildId) {
        return { ignored: true, dropped: "wrong_guild" };
      }
      if (channelId !== undefined && channelId !== state.channelId) {
        return { ignored: true, dropped: "wrong_guild" };
      }
    }

    if (!userId || userId === resolvedBotUserId || isBot || state.mutedUsers.has(userId)) {
      return { ignored: true };
    }

    const snap = captureSessionSnapshot();
    if (await ensureSpeakerKnown(userId, state.guildId)) {
      return { ignored: true };
    }
    if (!isSnapshotCurrent(snap) || state.mutedUsers.has(userId) || userId === resolvedBotUserId) {
      return { ignored: true, dropped: "inactive_session" };
    }

    if (!opusPacket || typeof opusPacket.byteLength !== "number") {
      return { ignored: true, dropped: "invalid" };
    }

    let pcm48;
    try {
      const codec = decoder ?? (await ensureTranscoder());
      pcm48 = await codec.decode(opusPacket);
    } catch {
      return { ignored: true, dropped: "decode" };
    }

    if (!isSnapshotCurrent(snap)) {
      return { ignored: true, dropped: "inactive_session" };
    }
    if (
      state.mutedUsers.has(userId) ||
      userId === resolvedBotUserId ||
      cachedSpeakerBot(userId) === true
    ) {
      return { ignored: true };
    }

    const frame = {
      userId,
      displayName,
      pcm16kMono: downsample48kStereoTo16kMono(pcm48),
    };
    if (timestamp !== undefined) frame.timestampMs = timestamp;
    return await vad.pushFrame(frame);
  }

  async function start({ client: startClient } = {}) {
    if (state.closed) throw new Error("Voice orchestrator is closed");
    if (startClient) {
      state.client = startClient;
      if (!resolvedBotUserId) resolvedBotUserId = startClient?.user?.id || "";
    }
    const verification = await verifyDaveImpl();
    state.daveReady = verification?.daveReady === true;
    state.active = true;
    return getStatus();
  }

  async function joinChannel({ guildId, channelId, adapterCreator } = {}) {
    if (state.closed) throw new Error("Voice orchestrator is closed");
    if (typeof guildId !== "string" || guildId.length === 0) {
      throw new Error("joinChannel requires a non-empty guildId");
    }
    if (typeof channelId !== "string" || channelId.length === 0) {
      throw new Error("joinChannel requires a non-empty channelId");
    }
    if (state.active && state.guildId && state.guildId !== guildId) {
      throw new Error("Voice session already active in another guild");
    }
    if (pendingGuildId && pendingGuildId !== guildId) {
      throw new Error("Voice join already in progress for another guild");
    }

    const myNonce = (joinNonce += 1);
    pendingGuildId = guildId;

    try {
      await ensureTranscoder();

      if (!resolvedVoiceDeps) {
        try {
          resolvedVoiceDeps = await loadVoiceDeps();
        } catch (error) {
          if (!playerSink) throw error;
          resolvedVoiceDeps = null;
        }
      }

      if (
        state.closed ||
        joinNonce !== myNonce ||
        (state.active && state.guildId && state.guildId !== guildId)
      ) {
        throw new Error("Voice join aborted");
      }

      resetTurnAndSession();
      destroyVoice();

      state.guildId = guildId;
      state.channelId = channelId;
      state.active = true;
      state.sessionToken = Symbol("voice-session");
      seedVoicePresence(guildId, channelId);

      if (resolvedVoiceDeps?.joinVoiceChannel) {
        try {
          voiceConnection = resolvedVoiceDeps.joinVoiceChannel({
            guildId,
            channelId,
            adapterCreator,
            selfDeaf: false,
            selfMute: false,
          });

          bindReceiver(guildId, channelId, state.sessionToken);

          connectionErrorListener = () => {
            state.lastError = "voice connection error";
          };
          voiceConnection?.on?.("error", connectionErrorListener);

          connectionStateListener = (_oldState, nextState) => {
            try {
              if (state.closed) return;
              if (nextState?.status === "disconnected" || nextState?.status === "destroyed") {
                if (voiceConnection?.joinConfig?.channelId !== state.channelId) {
                  handleExternalVoiceTransition(null);
                }
                return;
              }
              const bound = voiceConnection?.joinConfig?.channelId;
              if (bound && bound !== state.channelId) {
                handleExternalVoiceTransition(bound);
              }
            } catch {
              state.lastError = "voice connection error";
            }
          };
          voiceConnection?.on?.("stateChange", connectionStateListener);

          if (typeof state.client?.on === "function") {
            const listener = (_oldState, newState) => {
              try {
                if (voiceStateListener !== listener || state.closed || !resolvedBotUserId) return;
                const userId = newState?.id ?? _oldState?.id;
                const guildId =
                  newState?.guildId ??
                  newState?.guild?.id ??
                  _oldState?.guildId ??
                  _oldState?.guild?.id;
                if (userId === resolvedBotUserId) {
                  if (guildId && state.guildId && guildId !== state.guildId) return;
                  if (!Object.hasOwn(newState ?? {}, "channelId")) return;
                  const nextChannelId = newState.channelId;
                  if (nextChannelId !== state.channelId) handleExternalVoiceTransition(nextChannelId ?? null);
                  return;
                }
                handleHumanVoiceState(_oldState, newState);
              } catch {
                state.lastError = "voice state failed";
              }
            };
            voiceStateListener = listener;
            state.client.on("voiceStateUpdate", listener);
          }

          if (
            typeof resolvedVoiceDeps.createAudioPlayer === "function" &&
            typeof voiceConnection?.subscribe === "function"
          ) {
            voicePlayer = resolvedVoiceDeps.createAudioPlayer();
            voiceConnection.subscribe(voicePlayer);

            playerErrorListener = () => {
              state.lastError = "voice player error";
              if (activePlaybackTurn) {
                timingTrace.setStopReason(activePlaybackTurn.seq, "player_error");
              }
            };
            voicePlayer?.on?.("error", playerErrorListener);

            playerStateListener = (oldState, nextState) => {
              const endedResource = nextState?.resource ?? oldState?.resource;
              if (nextState?.status === "playing") {
                if (
                  endedResource === activePlaybackResource &&
                  activePlaybackTurn &&
                  activePlaybackTurn.playerState !== "playing"
                ) {
                  activePlaybackTurn.playerState = "playing";
                  timingTrace.notePlayer(activePlaybackTurn.seq, { playerState: "playing" });
                }
                return;
              }
              if (nextState?.status !== "idle" || !activePlaybackTurn || !activePlaybackResource) return;
              if (endedResource !== activePlaybackResource) return;
              const endedTurn = activePlaybackTurn;
              const queueDepth = activePlaybackStream?.readableLength ?? null;
              const writableEnded = activePlaybackStream
                ? Boolean(activePlaybackStream.writableEnded)
                : null;
              const outputOpen = Boolean(
                activePlaybackStream &&
                  !activePlaybackStream.destroyed &&
                  !activePlaybackStream.writableEnded,
              );
              const sinceLastQueuedFrameMs =
                endedTurn.lastQueuedFrameAt === undefined
                  ? null
                  : Math.max(0, now() - endedTurn.lastQueuedFrameAt);
              timingTrace.notePlayer(endedTurn.seq, {
                playerState: "idle",
                queueDepth,
                writableEnded,
                sinceLastQueuedFrameMs,
                idleObservation:
                  outputOpen && (queueDepth ?? 0) === 0
                    ? "output_open_empty_queue"
                    : "normal",
              });
              timingTrace.setStopReason(endedTurn.seq, "normal_drain", { onlyIfUnset: true });
              if (endedTurn?.outStream === activePlaybackStream) finishOutput(endedTurn);
              activePlaybackUserId = null;
              activePlaybackTurn = null;
              activePlaybackResource = null;
              activePlaybackStream = null;
              state.isBotSpeaking = false;
            };
            voicePlayer?.on?.("stateChange", playerStateListener);
          }
        } catch (error) {
          destroyVoice();
          clearVoicePresence();
          state.guildId = null;
          state.channelId = null;
          state.active = false;
          state.sessionToken = null;
          clearGuildScopedState();
          throw error;
        }
      }

      return getStatus();
    } finally {
      if (joinNonce === myNonce) pendingGuildId = null;
    }
  }

  async function leaveChannel({ guildId } = {}) {
    if (
      guildId &&
      ((state.guildId && guildId !== state.guildId) ||
        (!state.guildId && pendingGuildId && guildId !== pendingGuildId))
    ) {
      throw new Error("Voice session belongs to another guild");
    }
    joinNonce += 1;
    pendingGuildId = null;
    resetTurnAndSession();
    destroyVoice();
    state.guildId = null;
    state.channelId = null;
    // Guild boundary: mutes and per-guild modes revert to configured defaults.
    state.mutedUsers.clear();
    vad.setIgnoredUsers?.([]);
    state.chimeMode = CHIME_MODES.includes(config?.chime?.mode) ? config.chime.mode : "smart";
    state.fillerMode = FILLER_MODES.includes(config?.reactions?.mode)
      ? config.reactions.mode
      : "adaptive";
    return getStatus();
  }

  function setChimeMode(mode) {
    if (!CHIME_MODES.includes(mode)) {
      throw new RangeError(`chime mode must be one of [${CHIME_MODES.join(", ")}]`);
    }
    state.chimeMode = mode;
  }

  function setFillerMode(mode) {
    if (!FILLER_MODES.includes(mode)) {
      throw new RangeError(`filler mode must be one of [${FILLER_MODES.join(", ")}]`);
    }
    state.fillerMode = mode;
  }

  function muteUser(userId, muted = true) {
    if (typeof userId !== "string" || userId.length === 0) {
      throw new RangeError("userId must be a non-empty string");
    }
    if (muted) state.mutedUsers.add(userId);
    else state.mutedUsers.delete(userId);
    vad.setIgnoredUsers?.([...state.mutedUsers]);

    if (!muted) return;
    refreshSinkPlaybackOwner();
    if (currentTurn?.userId === userId) {
      abortTurn(currentTurn);
    } else if (activePlaybackUserId === userId) {
      if (activePlaybackTurn?.committed === false) abortTurn(activePlaybackTurn);
      else stopOwnedPlayback(activePlaybackTurn);
    }
  }

  function getStatus() {
    return {
      active: state.active,
      guildId: state.guildId,
      channelId: state.channelId,
      daveReady: state.daveReady,
      isBotSpeaking: state.isBotSpeaking,
      chimeMode: state.chimeMode,
      fillerMode: state.fillerMode,
      mutedUsers: [...state.mutedUsers],
      config: redactConfig(config ?? {}),
      latencySummary: latencyTracker.formatSummary(),
      latencyLast: latencyTracker.getLast(),
      timingSummary: timingTrace.formatSummary(),
      timingLast: timingTrace.getLatest(),
      lastError: state.lastError,
    };
  }

  async function close() {
    if (state.closed) return;
    state.closed = true;
    state.active = false;
    joinNonce += 1;
    pendingGuildId = null;
    resetTurnAndSession();
    destroyVoice();
    state.guildId = null;
    state.channelId = null;
    state.mutedUsers.clear();
    vad.setIgnoredUsers?.([]);
    state.chimeMode = CHIME_MODES.includes(config?.chime?.mode) ? config.chime.mode : "smart";
    state.fillerMode = FILLER_MODES.includes(config?.reactions?.mode)
      ? config.reactions.mode
      : "adaptive";
  }

  return {
    start,
    joinChannel,
    leaveChannel,
    handleOpusPacket,
    setChimeMode,
    setFillerMode,
    muteUser,
    getStatus,
    close,
  };
}
