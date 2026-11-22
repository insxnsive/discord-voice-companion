// Discord voice session pipeline: DAVE prerequisite verification, Opus ingress
// and egress, barge-in cancellation, and the STT -> filler -> gate -> LLM -> TTS
// turn. Discord gateway wiring is injected or lazily loaded.

import { PassThrough } from "node:stream";

import { CHIME_MODES, FILLER_MODES, LatencyTracker, assertAdapterContract } from "../types.mjs";
import {
  createFrameAccumulator,
  createOpusTranscoder,
  createUpsampler24kTo48kStereo,
  downsample48kStereoTo16kMono,
} from "../audio/pcm.mjs";
import { PerSpeakerVad, createRmsSpeechDetector } from "../audio/vad.mjs";
import { isWhisperHallucination } from "../stt/adapters.mjs";
import { createClauseSplitter } from "../llm/client.mjs";
import { scheduleReactionFiller } from "../tts/reactions.mjs";
import { redactConfig } from "../config/config.mjs";

export const REQUIRED_GATEWAY_INTENTS = Object.freeze(["Guilds", "GuildVoiceStates"]);
export const REQUIRED_VOICE_PERMISSIONS = Object.freeze(["Connect", "Speak"]);

const MIN_NODE = "22.12.0";
const MAX_SPEAKER_CACHE = 256;
const DAVE_MODULES = Object.freeze(["@discordjs/voice", "@snazzah/davey"]);
const MAX_PCM_CHUNK_BYTES = 245760;

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

  let resolvedBotUserId = botUserId || client?.user?.id || "";
  let transcoder = opusTranscoder;
  let transcoderPromise = null;
  let resolvedVoiceDeps = voiceDeps ?? null;
  let joinNonce = 0;
  let pendingGuildId = null;
  let activePlaybackUserId = null;

  const speakerBotCache = new Map();

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
    lastBotReplyAt: 0,
    lastError: null,
    sessionEpoch: 0,
    sessionToken: null,
    closed: false,
    client,
  };

  let history = [];
  let currentTurn = null;
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

  function isTurnLive(turn) {
    return Boolean(
      turn &&
        !state.closed &&
        state.active &&
        verifyLiveChannelBinding() &&
        state.sessionEpoch === turn.sessionEpoch &&
        state.guildId === turn.guildId &&
        state.channelId === turn.channelId &&
        currentTurn === turn &&
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

  function stopPlayback() {
    finishOutput(currentTurn);
    activePlaybackUserId = null;
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

  function rememberSpeaker(userId, isBot) {
    if (speakerBotCache.size >= MAX_SPEAKER_CACHE) {
      const oldest = speakerBotCache.keys().next().value;
      speakerBotCache.delete(oldest);
    }
    speakerBotCache.set(userId, isBot);
  }

  async function ensureSpeakerKnown(userId, guildId) {
    const usersCache = state.client?.users?.cache;
    const guildsCache = state.client?.guilds?.cache;
    if (!usersCache && !guildsCache) return false;
    if (speakerBotCache.has(userId)) return speakerBotCache.get(userId);

    const known = lookupBot(userId, guildId);
    if (known !== undefined) {
      rememberSpeaker(userId, known);
      return known;
    }

    const guild = guildsCache?.get?.(guildId);
    try {
      let fetched = null;
      if (typeof guild?.members?.fetch === "function") {
        fetched = await guild.members.fetch(userId);
      } else if (typeof state.client?.users?.fetch === "function") {
        fetched = await state.client.users.fetch(userId);
      }
      if (fetched) {
        const isBot = Boolean(fetched.user?.bot ?? fetched.bot);
        rememberSpeaker(userId, isBot);
        return isBot;
      }
    } catch {
      // Fall through to fail closed.
    }
    return true;
  }

  function detachReceiver() {
    const speaking = voiceConnection?.receiver?.speaking;
    if (speaking && speakingListener) {
      speaking.off?.("start", speakingListener);
      speaking.removeListener?.("start", speakingListener);
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
        state.client.off("voiceStateUpdate", voiceStateListener);
        state.client.removeListener?.("voiceStateUpdate", voiceStateListener);
      }
      if (connectionStateListener) {
        voiceConnection.off?.("stateChange", connectionStateListener);
        voiceConnection.removeListener?.("stateChange", connectionStateListener);
      }
      if (connectionErrorListener) {
        voiceConnection.off?.("error", connectionErrorListener);
        voiceConnection.removeListener?.("error", connectionErrorListener);
      }
    }
    if (voicePlayer) {
      if (playerStateListener) {
        voicePlayer.off?.("stateChange", playerStateListener);
        voicePlayer.removeListener?.("stateChange", playerStateListener);
      }
      if (playerErrorListener) {
        voicePlayer.off?.("error", playerErrorListener);
        voicePlayer.removeListener?.("error", playerErrorListener);
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
  }

  function resetTurnAndSession() {
    currentTurn?.controller.abort();
    currentTurn?.filler?.cancel();
    finishOutput(currentTurn);
    currentTurn = null;
    state.isBotSpeaking = false;
    stopPlayback();
    vad.clear?.();
    history = [];
    state.lastBotReplyAt = 0;
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
      return;
    }
    if (!guildId) return;
    state.guildId = guildId;
    state.channelId = nextChannelId;
    state.active = true;
    state.sessionToken = Symbol("voice-session");
    bindReceiver(guildId, nextChannelId, state.sessionToken);
  }

  async function onSpeakingStart(userId, boundGuildId, boundChannelId, boundToken) {
    try {
      if (!userId || userId === resolvedBotUserId) return;
      if (state.closed || state.sessionToken !== boundToken) return;
      if (state.guildId !== boundGuildId || state.channelId !== boundChannelId) return;
      if (!verifyLiveChannelBinding()) return;
      if (state.mutedUsers.has(userId)) return;
      if (await ensureSpeakerKnown(userId, boundGuildId)) return;
      if (userStreams.has(userId)) return;

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
          }),
        ).catch(() => {});
      });
      stream.on?.("end", cleanup);
      stream.on?.("close", cleanup);
      stream.on?.("error", cleanup);
    } catch {
      state.lastError = "voice receiver failed";
    }
  }

  function ensureTurnStream(turn) {
    if (playerSink?.playOpusFrame) return null;
    if (!voicePlayer) return null;
    if (turn.outStream) return turn.outStream;

    const stream = new PassThrough({ objectMode: true });
    const resource = resolvedVoiceDeps?.createAudioResource
      ? resolvedVoiceDeps.createAudioResource(stream, {
          inputType: resolvedVoiceDeps?.StreamType?.Opus ?? "opus",
        })
      : stream;
    turn.outStream = stream;
    activePlaybackUserId = turn.userId;
    voicePlayer.play?.(resource);
    return stream;
  }

  function playPacket(turn, packet) {
    activePlaybackUserId = turn.userId;
    if (playerSink?.playOpusFrame) {
      playerSink.playOpusFrame(packet);
      return;
    }
    const stream = ensureTurnStream(turn);
    stream?.write?.(packet);
  }

  function emitFrame(turn, packet) {
    if (turn.controller.signal.aborted) return;
    if (!turn.firstFrameMarked) {
      latencyTracker.mark(turn.id, "opusFirstFrameAt", now());
      turn.firstFrameMarked = true;
    }
    playPacket(turn, packet);
  }

  function emitPcm48(turn, pcm48) {
    const buffer = Buffer.isBuffer(pcm48) ? pcm48 : Buffer.from(pcm48);
    for (let offset = 0; offset < buffer.length; offset += MAX_PCM_CHUNK_BYTES) {
      const slice = buffer.subarray(offset, Math.min(offset + MAX_PCM_CHUNK_BYTES, buffer.length));
      for (const frame of turn.accumulator.push(slice)) {
        emitFrame(turn, transcoder.encode(frame));
      }
    }
  }

  function flushFrames(turn) {
    const tail = turn.accumulator.flushPadded();
    if (tail) emitFrame(turn, transcoder.encode(tail));
  }

  function emitClip(clip, turn) {
    if (!clip || turn.controller.signal.aborted) return;
    if (Array.isArray(clip.frames48kOpus) && clip.frames48kOpus.length > 0) {
      for (const packet of clip.frames48kOpus) emitFrame(turn, packet);
      return;
    }
    if (clip.pcm24k) {
      emitPcm48(turn, turn.upsampler.push(clip.pcm24k));
      flushFrames(turn);
    }
  }

  function abortTurn() {
    const turn = currentTurn;
    state.isBotSpeaking = false;
    if (!turn) return;
    try {
      latencyTracker.mark(turn.id, "bargeInCancelled", true);
    } catch {
      // The turn may already be finished.
    }
    turn.controller.abort();
    turn.filler?.cancel();
    stopPlayback();
  }

  function handleBargeIn() {
    abortTurn();
  }

  async function speakClause(clause, turn) {
    if (!clause || !isTurnLive(turn)) return;
    if (!turn.fillerCancelled) {
      turn.filler?.cancel();
      turn.fillerCancelled = true;
    }
    if (!turn.llmMarked) {
      latencyTracker.mark(turn.id, "llmFirstClauseAt", now());
      turn.llmMarked = true;
    }

    await ttsAdapter.synthesizeClause({
      text: clause,
      signal: turn.controller.signal,
      onPcmChunk: (pcm24) => {
        if (!isTurnLive(turn)) return;
        if (!turn.ttsMarked) {
          latencyTracker.mark(turn.id, "ttsFirstByteAt", now());
          turn.ttsMarked = true;
        }
        state.isBotSpeaking = true;
        emitPcm48(turn, turn.upsampler.push(pcm24));
      },
    });

    if (isTurnLive(turn)) flushFrames(turn);
  }

  async function handleSpeechEnd({ userId, displayName, pcm16kMono, endedAt } = {}) {
    if (state.closed) return;

    const turnId = `turn-${(turnCounter += 1)}`;
    const controller = new AbortController();
    if (currentTurn) {
      currentTurn.controller.abort();
      currentTurn.filler?.cancel();
      finishOutput(currentTurn);
      stopPlayback();
    }

    const startedAt = endedAt === undefined ? now() : endedAt;
    const turn = {
      id: turnId,
      userId,
      controller,
      filler: null,
      fillerCancelled: false,
      upsampler: createUpsampler24kTo48kStereo(),
      accumulator: createFrameAccumulator(),
      firstFrameMarked: false,
      llmMarked: false,
      ttsMarked: false,
      outStream: null,
      sessionEpoch: state.sessionEpoch,
      guildId: state.guildId,
      channelId: state.channelId,
    };
    currentTurn = turn;
    latencyTracker.startTurn(turnId, startedAt);

    const fillerDelay =
      state.fillerMode === "off" ? 0 : (config?.reactions?.fillerDelayMs ?? 165);
    let ttsError = null;

    try {
      await ensureTranscoder();
      if (!isTurnLive(turn)) return;

      const transcript = await sttAdapter.transcribe(pcm16kMono, { signal: controller.signal });
      if (!isTurnLive(turn)) return;
      latencyTracker.mark(turnId, "sttDoneAt", now());
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
            if (!isTurnLive(turn)) return;
            const clip = reactionPool?.pick(transcript, { mode: state.fillerMode });
            if (!clip) return;
            latencyTracker.mark(turnId, "fillerFiredAt", now());
            state.isBotSpeaking = true;
            emitClip(clip, turn);
          } catch {
            state.lastError = "reaction filler failed";
          }
        },
      });

      const decision = await llmClient.shouldChimeIn({
        transcript,
        chimeMode: state.chimeMode,
        speakerName: displayName,
        botAliases: config?.chime?.botAliases,
        lastBotReplyAt: state.lastBotReplyAt,
        followUpWindowMs: config?.chime?.followUpWindowMs,
        gateTimeoutMs: config?.chime?.gateTimeoutMs,
        signal: controller.signal,
        now,
      });
      if (!isTurnLive(turn)) return;
      latencyTracker.mark(turnId, "gateDoneAt", now());
      const chimeIn = decision?.chimeIn ?? decision?.shouldReply;
      if (chimeIn !== true) {
        turn.filler?.cancel();
        return;
      }

      history.push({ role: "user", content: transcript });
      boundedHistory(history, maxHistoryTurns);

      let chain = Promise.resolve();
      const enqueueClause = (clause) => {
        chain = chain
          .then(() => speakClause(clause, turn))
          .catch((error) => {
            ttsError ??= error;
          });
      };
      const splitter = createClauseSplitterImpl({ onClause: enqueueClause });

      const reply = await llmClient.streamReply({
        messages: history.slice(),
        signal: controller.signal,
        onToken: (delta) => splitter?.push?.(delta),
      });
      if (!isTurnLive(turn)) return;

      for (const clause of splitter?.flush?.() ?? []) enqueueClause(clause);
      await chain;

      if (!isTurnLive(turn)) return;
      if (ttsError && !controller.signal.aborted) {
        state.lastError = "voice turn failed";
      }

      if (typeof reply?.fullText === "string" && reply.fullText.trim().length > 0) {
        history.push({ role: "assistant", content: reply.fullText });
        boundedHistory(history, maxHistoryTurns);
      }
      state.lastBotReplyAt = now();
    } catch (error) {
      if (!controller.signal.aborted && error?.name !== "AbortError") {
        state.lastError = "voice turn failed";
      }
    } finally {
      if (currentTurn === turn) {
        currentTurn = null;
        state.isBotSpeaking = false;
      }
      turn.filler?.cancel();
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
    if (await ensureSpeakerKnown(userId, state.guildId)) {
      return { ignored: true };
    }
    if (!opusPacket || typeof opusPacket.byteLength !== "number") {
      return { ignored: true, dropped: "invalid" };
    }

    const packetEpoch = state.sessionEpoch;
    const packetToken = state.sessionToken;
    let pcm48;
    try {
      const codec = await ensureTranscoder();
      pcm48 = await codec.decode(opusPacket);
    } catch {
      return { ignored: true, dropped: "decode" };
    }

    if (
      !state.active ||
      state.closed ||
      !state.daveReady ||
      !verifyLiveChannelBinding() ||
      state.sessionEpoch !== packetEpoch ||
      state.sessionToken !== packetToken
    ) {
      return { ignored: true, dropped: "inactive_session" };
    }
    if (
      state.mutedUsers.has(userId) ||
      userId === resolvedBotUserId ||
      (await ensureSpeakerKnown(userId, state.guildId))
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

      if (resolvedVoiceDeps?.joinVoiceChannel) {
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
          voiceStateListener = (_oldState, newState) => {
            try {
              if (state.closed || !resolvedBotUserId) return;
              if ((newState?.id ?? _oldState?.id) !== resolvedBotUserId) return;
              if (newState?.guild?.id && state.guildId && newState.guild.id !== state.guildId) {
                return;
              }
              const nextChannelId = newState?.channelId ?? null;
              if (nextChannelId !== state.channelId) {
                handleExternalVoiceTransition(nextChannelId);
              }
            } catch {
              state.lastError = "voice state failed";
            }
          };
          state.client.on("voiceStateUpdate", voiceStateListener);
        }

        if (
          typeof resolvedVoiceDeps.createAudioPlayer === "function" &&
          typeof voiceConnection?.subscribe === "function"
        ) {
          voicePlayer = resolvedVoiceDeps.createAudioPlayer();
          voiceConnection.subscribe(voicePlayer);

          playerErrorListener = () => {
            state.lastError = "voice player error";
          };
          voicePlayer?.on?.("error", playerErrorListener);

          playerStateListener = (_oldState, nextState) => {
            if (nextState?.status === "idle") activePlaybackUserId = null;
          };
          voicePlayer?.on?.("stateChange", playerStateListener);
        }
      }

      return getStatus();
    } finally {
      if (joinNonce === myNonce) pendingGuildId = null;
    }
  }

  async function leaveChannel({ guildId } = {}) {
    joinNonce += 1;
    pendingGuildId = null;
    if (guildId && state.guildId && guildId !== state.guildId) {
      throw new Error("Voice session belongs to another guild");
    }
    resetTurnAndSession();
    destroyVoice();
    state.guildId = null;
    state.channelId = null;
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
    if (currentTurn?.userId === userId) {
      abortTurn();
    } else if (activePlaybackUserId === userId) {
      state.isBotSpeaking = false;
      stopPlayback();
    }
  }

  function getStatus() {
    return {
      active: state.active,
      guildId: state.guildId,
      channelId: state.channelId,
      daveReady: state.daveReady,
      chimeMode: state.chimeMode,
      fillerMode: state.fillerMode,
      mutedUsers: [...state.mutedUsers],
      config: redactConfig(config ?? {}),
      latencySummary: latencyTracker.formatSummary(),
      latencyLast: latencyTracker.getLast(),
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
