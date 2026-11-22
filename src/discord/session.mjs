// Discord voice session pipeline: DAVE prerequisite verification, Opus ingress
// and egress, barge-in cancellation, and the STT -> filler -> gate -> LLM -> TTS
// turn. Discord gateway wiring is injected so the unit suite needs no network.

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
const DAVE_MODULES = Object.freeze(["@discordjs/voice", "@snazzah/davey"]);

function parseNodeVersion(version) {
  const [major = 0, minor = 0] = String(version)
    .replace(/^v/, "")
    .split(".")
    .map((part) => Number.parseInt(part, 10) || 0);
  return { major, minor };
}

/**
 * Fail fast when the runtime cannot run DAVE E2EE. The voice and davey
 * packages are probed through an injected loader so tests stay offline.
 */
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

/**
 * Build the voice orchestrator. All I/O (STT, LLM, TTS, Opus, player, VAD) is
 * injected, so the pipeline can be driven frame by frame in tests.
 */
export function createVoiceOrchestrator({
  config = {},
  botUserId = "",
  client = null,
  voiceDeps = null,
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
    closed: false,
    client,
  };

  let history = [];
  let currentTurn = null;
  let turnCounter = 0;
  let voiceConnection = null;
  let voicePlayer = null;
  let voiceReceiver = null;

  function playPacket(packet) {
    if (playerSink?.playOpusFrame) {
      playerSink.playOpusFrame(packet);
      return;
    }
    voicePlayer?.play?.(packet);
  }

  function stopPlayback() {
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

  async function ensureTranscoder() {
    if (transcoder) return transcoder;
    if (!transcoderPromise) transcoderPromise = createOpusTranscoderImpl();
    transcoder = await transcoderPromise;
    return transcoder;
  }

  function destroyVoice() {
    try {
      voiceReceiver?.destroy?.();
    } catch {
      // Receiver teardown is best effort.
    }
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
    voiceReceiver = null;
    voicePlayer = null;
    voiceConnection = null;
  }

  function resetTurnAndSession() {
    currentTurn?.controller.abort();
    currentTurn?.filler?.cancel();
    currentTurn = null;
    state.isBotSpeaking = false;
    stopPlayback();
    vad.clear?.();
    history = [];
    state.lastBotReplyAt = 0;
  }

  function emitFrame(turn, packet) {
    if (turn.controller.signal.aborted) return;
    if (!turn.firstFrameMarked) {
      latencyTracker.mark(turn.id, "opusFirstFrameAt", now());
      turn.firstFrameMarked = true;
    }
    playPacket(packet);
  }

  function emitPcm48(turn, pcm48) {
    for (const frame of turn.accumulator.push(pcm48)) {
      emitFrame(turn, transcoder.encode(frame));
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
    if (!clause || turn.controller.signal.aborted) return;
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
        if (turn.controller.signal.aborted) return;
        if (!turn.ttsMarked) {
          latencyTracker.mark(turn.id, "ttsFirstByteAt", now());
          turn.ttsMarked = true;
        }
        state.isBotSpeaking = true;
        emitPcm48(turn, turn.upsampler.push(pcm24));
      },
    });

    if (!turn.controller.signal.aborted) flushFrames(turn);
  }

  function resolveIsBot(userId, guildId) {
    const direct = state.client?.users?.cache?.get?.(userId)?.bot;
    if (typeof direct === "boolean") return direct;
    const member = state.client?.guilds?.cache?.get?.(guildId)?.members?.cache?.get?.(userId);
    return member?.user?.bot === true;
  }

  async function handleSpeechEnd({ userId, displayName, pcm16kMono, endedAt } = {}) {
    if (state.closed) return;

    const turnId = `turn-${(turnCounter += 1)}`;
    const controller = new AbortController();
    if (currentTurn) {
      currentTurn.controller.abort();
      currentTurn.filler?.cancel();
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
    };
    currentTurn = turn;
    latencyTracker.startTurn(turnId, startedAt);

    const fillerDelay =
      state.fillerMode === "off" ? 0 : (config?.reactions?.fillerDelayMs ?? 165);
    let ttsError = null;

    try {
      await ensureTranscoder();
      const transcript = await sttAdapter.transcribe(pcm16kMono, { signal: controller.signal });
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
          if (controller.signal.aborted) return;
          const clip = reactionPool?.pick(transcript, { mode: state.fillerMode });
          if (!clip) return;
          latencyTracker.mark(turnId, "fillerFiredAt", now());
          state.isBotSpeaking = true;
          emitClip(clip, turn);
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

      for (const clause of splitter?.flush?.() ?? []) enqueueClause(clause);
      await chain;

      if (typeof reply?.fullText === "string" && reply.fullText.trim().length > 0) {
        history.push({ role: "assistant", content: reply.fullText });
        boundedHistory(history, maxHistoryTurns);
      }
      state.lastBotReplyAt = now();
      void ttsError;
    } catch (error) {
      // A barge-in or a failed provider ends this turn only.
      if (!controller.signal.aborted && error?.name !== "AbortError") {
        state.lastError = "voice turn failed";
      }
    } finally {
      if (currentTurn === turn) currentTurn = null;
      state.isBotSpeaking = false;
      turn.filler?.cancel();
      try {
        latencyTracker.finishTurn(turnId);
      } catch {
        // Already finished by a racing close.
      }
    }
  }

  async function handleOpusPacket({
    guildId,
    channelId,
    userId,
    displayName,
    isBot = false,
    opusPacket,
    timestamp,
  } = {}) {
    if (!state.active || !state.daveReady || !state.guildId || !state.channelId) {
      return { ignored: true, dropped: "inactive_session" };
    }
    if (guildId !== undefined && guildId !== state.guildId) {
      return { ignored: true, dropped: "wrong_guild" };
    }
    if (channelId !== undefined && channelId !== state.channelId) {
      return { ignored: true, dropped: "wrong_guild" };
    }
    if (
      !userId ||
      userId === resolvedBotUserId ||
      isBot ||
      state.mutedUsers.has(userId) ||
      resolveIsBot(userId, state.guildId)
    ) {
      return { ignored: true };
    }
    if (!opusPacket || typeof opusPacket.byteLength !== "number") {
      return { ignored: true, dropped: "invalid" };
    }

    let pcm48;
    try {
      const codec = await ensureTranscoder();
      pcm48 = await codec.decode(opusPacket);
    } catch {
      return { ignored: true, dropped: "decode" };
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
      if (!resolvedBotUserId) {
        resolvedBotUserId = startClient?.user?.id || "";
      }
    }
    const verification = await verifyDaveImpl();
    state.daveReady = verification?.daveReady === true;
    state.active = true;
    return getStatus();
  }

  async function joinChannel({ guildId, channelId, adapterCreator } = {}) {
    if (typeof guildId !== "string" || guildId.length === 0) {
      throw new Error("joinChannel requires a non-empty guildId");
    }
    if (typeof channelId !== "string" || channelId.length === 0) {
      throw new Error("joinChannel requires a non-empty channelId");
    }
    if (state.active && state.guildId && state.guildId !== guildId) {
      throw new Error("Voice session already active in another guild");
    }

    await ensureTranscoder();
    resetTurnAndSession();
    destroyVoice();

    state.guildId = guildId;
    state.channelId = channelId;
    state.active = true;

    if (voiceDeps?.joinVoiceChannel) {
      voiceConnection = voiceDeps.joinVoiceChannel({
        guildId,
        channelId,
        adapterCreator,
        selfDeaf: false,
        selfMute: false,
      });
      if (typeof voiceConnection?.receiver?.subscribe === "function") {
        voiceReceiver = voiceConnection.receiver.subscribe(resolvedBotUserId || "bot");
      }
      if (
        typeof voiceDeps.createAudioPlayer === "function" &&
        typeof voiceConnection?.subscribe === "function"
      ) {
        voicePlayer = voiceDeps.createAudioPlayer();
        voiceConnection.subscribe(voicePlayer);
      }
    }

    return getStatus();
  }

  async function leaveChannel({ guildId } = {}) {
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

    if (muted && currentTurn?.userId === userId) {
      abortTurn();
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
    };
  }

  async function close() {
    if (state.closed) return;
    state.closed = true;
    state.active = false;
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
