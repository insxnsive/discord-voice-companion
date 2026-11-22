import fs from "node:fs";
import path from "node:path";

import { REACTION_MOODS } from "../types.mjs";
import { stripWavHeaderIfPresent } from "./adapters.mjs";

const CLIP_FILENAME_PATTERN =
  /^(subtle_breath|curious_hmm|light_chuckle|playful_scoff)_([a-z0-9_-]{1,32})\.(pcm|wav)$/i;

/**
 * 14 short reaction prompts balanced across the 4 reaction moods for one-time
 * voice-matched reaction generation on first setup.
 */
export const REACTION_CLIP_PROMPTS = Object.freeze([
  Object.freeze({ id: "subtle_breath_01", mood: "subtle_breath", text: "[sighs]" }),
  Object.freeze({ id: "subtle_breath_02", mood: "subtle_breath", text: "Mm." }),
  Object.freeze({ id: "subtle_breath_03", mood: "subtle_breath", text: "Ah." }),
  Object.freeze({ id: "curious_hmm_01", mood: "curious_hmm", text: "Hmm?" }),
  Object.freeze({ id: "curious_hmm_02", mood: "curious_hmm", text: "Ooh?" }),
  Object.freeze({ id: "curious_hmm_03", mood: "curious_hmm", text: "Wait, what?" }),
  Object.freeze({ id: "curious_hmm_04", mood: "curious_hmm", text: "Huh." }),
  Object.freeze({ id: "light_chuckle_01", mood: "light_chuckle", text: "Hehe!" }),
  Object.freeze({ id: "light_chuckle_02", mood: "light_chuckle", text: "Pfft, okay." }),
  Object.freeze({ id: "light_chuckle_03", mood: "light_chuckle", text: "Hah!" }),
  Object.freeze({ id: "light_chuckle_04", mood: "light_chuckle", text: "Ooh, nice." }),
  Object.freeze({ id: "playful_scoff_01", mood: "playful_scoff", text: "Tch, seriously?" }),
  Object.freeze({ id: "playful_scoff_02", mood: "playful_scoff", text: "Oh, please." }),
  Object.freeze({ id: "playful_scoff_03", mood: "playful_scoff", text: "As if." }),
]);

/**
 * Map an utterance transcript into one of the four reaction moods using
 * zero-latency local pattern heuristics.
 */
export function classifyTranscriptMood(transcript) {
  const text = typeof transcript === "string" ? transcript.trim() : "";
  if (!text) return "subtle_breath";
  if (/\b(seriously|bro|bruh|nah|really|skill issue)\b/i.test(text)) {
    return "playful_scoff";
  }
  if (text.includes("?") || /\b(why|what|how|who|when|wait)\b/i.test(text)) {
    return "curious_hmm";
  }
  if (text.includes("!") || /\b(haha|lol|lmao|funny|nice|crazy|wall|crash)\b/i.test(text)) {
    return "light_chuckle";
  }
  return "subtle_breath";
}

const DEFAULT_MAX_CLIP_BYTES = 512 * 1024;
const DEFAULT_MAX_POOL_BYTES = 8 * 1024 * 1024;

function computeClipBytes({ frames48kOpus = [], pcm24k = null }) {
  let total = pcm24k ? (pcm24k.byteLength ?? pcm24k.length ?? 0) : 0;
  if (Array.isArray(frames48kOpus)) {
    for (const frame of frames48kOpus) {
      if (frame) total += frame.byteLength ?? frame.length ?? 0;
    }
  }
  return total;
}

/**
 * Mood-bucketed reaction audio pool with an LRU cooldown window so short
 * reaction cues do not repeat back-to-back across long voice sessions.
 */
export class ReactionPool {
  constructor({
    cooldownTurns = 10,
    maxClips = 64,
    maxFileBytes = DEFAULT_MAX_CLIP_BYTES,
    maxTotalBytes = DEFAULT_MAX_POOL_BYTES,
  } = {}) {
    this.cooldownTurns =
      Number.isInteger(cooldownTurns) && cooldownTurns > 0 ? cooldownTurns : 10;
    this.maxClips = Number.isInteger(maxClips) && maxClips > 0 ? maxClips : 64;
    this.maxFileBytes =
      Number.isInteger(maxFileBytes) && maxFileBytes > 0 ? maxFileBytes : DEFAULT_MAX_CLIP_BYTES;
    this.maxTotalBytes =
      Number.isInteger(maxTotalBytes) && maxTotalBytes > 0 ? maxTotalBytes : DEFAULT_MAX_POOL_BYTES;
    this.clips = new Map();
    this.recentIds = [];
    this.totalBytes = 0;
  }

  get size() {
    return this.clips.size;
  }

  registerClip({ id, mood, frames48kOpus = [], pcm24k = null } = {}) {
    if (typeof id !== "string" || id.trim().length === 0) {
      throw new RangeError("Reaction clip id must be a non-empty string");
    }
    if (!REACTION_MOODS.includes(mood)) {
      throw new RangeError(`Invalid reaction mood: ${mood}`);
    }
    const clipBytes = computeClipBytes({ frames48kOpus, pcm24k });
    if (clipBytes > this.maxFileBytes) {
      throw new RangeError("Reaction clip exceeds maximum file byte limit");
    }
    if (this.clips.size >= this.maxClips && !this.clips.has(id)) {
      return false;
    }
    const prevBytes = this.clips.has(id) ? computeClipBytes(this.clips.get(id)) : 0;
    const nextTotalBytes = this.totalBytes - prevBytes + clipBytes;
    if (nextTotalBytes > this.maxTotalBytes) {
      throw new RangeError("ReactionPool exceeded 8 MiB aggregate byte limit");
    }
    this.clips.set(id, { id, mood, frames48kOpus, pcm24k });
    this.totalBytes = nextTotalBytes;
    return true;
  }

  moodCounts() {
    const counts = {};
    for (const mood of REACTION_MOODS) counts[mood] = 0;
    for (const clip of this.clips.values()) {
      counts[clip.mood] += 1;
    }
    return counts;
  }

  pick(transcript, { mode = "adaptive", rng = Math.random } = {}) {
    if (mode === "off" || this.clips.size === 0) return null;

    const targetMood = mode === "subtle" ? "subtle_breath" : classifyTranscriptMood(transcript);
    const allClips = Array.from(this.clips.values());
    const recentSet = new Set(this.recentIds);
    const bucket = allClips.filter((clip) => clip.mood === targetMood);

    let candidates = bucket.filter((clip) => !recentSet.has(clip.id));
    if (candidates.length === 0 && (mode !== "subtle" || bucket.length === 0)) {
      candidates = allClips.filter((clip) => !recentSet.has(clip.id));
    }
    if (candidates.length === 0) {
      const globalLru = [...allClips].sort(
        (a, b) => this.recentIds.indexOf(a.id) - this.recentIds.indexOf(b.id),
      );
      candidates = [globalLru[0]];
    }

    const rawRoll = Number(rng());
    const boundedRoll = Number.isFinite(rawRoll) ? Math.min(0.999999, Math.max(0, rawRoll)) : 0;
    const chosen = candidates[Math.floor(boundedRoll * candidates.length)];

    const existingIdx = this.recentIds.indexOf(chosen.id);
    if (existingIdx !== -1) this.recentIds.splice(existingIdx, 1);
    this.recentIds.push(chosen.id);
    if (this.recentIds.length > this.cooldownTurns) {
      this.recentIds.splice(0, this.recentIds.length - this.cooldownTurns);
    }

    return chosen;
  }

  loadFromDirectory(dirPath, { fsImpl = fs, maxFileBytes = this.maxFileBytes } = {}) {
    if (!dirPath) return 0;
    if (typeof fsImpl.existsSync === "function" && !fsImpl.existsSync(dirPath)) {
      return 0;
    }

    if (typeof fsImpl.lstatSync === "function") {
      let rootStat;
      try {
        rootStat = fsImpl.lstatSync(dirPath);
      } catch (error) {
        if (error && error.code === "ENOENT") return 0;
        throw error;
      }
      if (rootStat?.isSymbolicLink?.()) {
        throw new RangeError("Reaction cache directory cannot be a symlink");
      }
      if (!rootStat?.isDirectory?.()) {
        return 0;
      }
    }

    const effectiveMaxFileBytes =
      Number.isInteger(maxFileBytes) && maxFileBytes > 0
        ? Math.min(this.maxFileBytes, maxFileBytes)
        : this.maxFileBytes;

    let entries;
    try {
      entries = fsImpl.readdirSync(dirPath);
    } catch (error) {
      if (error && error.code === "ENOENT") return 0;
      throw error;
    }

    const hasDescriptorFs =
      typeof fsImpl.openSync === "function" &&
      typeof fsImpl.fstatSync === "function" &&
      typeof fsImpl.readSync === "function" &&
      typeof fsImpl.closeSync === "function";

    let loaded = 0;
    for (const entry of [...entries].sort()) {
      if (this.size >= this.maxClips) break;
      const match = CLIP_FILENAME_PATTERN.exec(entry);
      if (!match) continue;

      const fullPath = path.join(dirPath, entry);
      let buffer = null;

      if (hasDescriptorFs) {
        const flags =
          (fsImpl.constants?.O_RDONLY ?? fs.constants.O_RDONLY) |
          (fsImpl.constants?.O_NOFOLLOW ?? fs.constants.O_NOFOLLOW ?? 0);
        let fd = null;
        try {
          fd = fsImpl.openSync(fullPath, flags);
          const stat = fsImpl.fstatSync(fd);
          if (
            stat?.isSymbolicLink?.() ||
            !stat?.isFile?.() ||
            stat.size <= 0 ||
            stat.size > effectiveMaxFileBytes
          ) {
            continue;
          }
          const target = Buffer.allocUnsafe(stat.size);
          let offset = 0;
          while (offset < stat.size) {
            const bytesRead = fsImpl.readSync(fd, target, offset, stat.size - offset, offset);
            if (bytesRead <= 0) break;
            offset += bytesRead;
          }
          if (offset !== stat.size) continue;
          buffer = target;
        } catch {
          continue;
        } finally {
          if (fd !== null) {
            try {
              fsImpl.closeSync(fd);
            } catch {
              // Ignore close errors.
            }
          }
        }
      } else {
        let stat;
        try {
          stat = fsImpl.lstatSync(fullPath);
        } catch {
          continue;
        }
        if (
          stat?.isSymbolicLink?.() ||
          !stat?.isFile?.() ||
          stat.size <= 0 ||
          stat.size > effectiveMaxFileBytes
        ) {
          continue;
        }
        const raw = fsImpl.readFileSync(fullPath);
        buffer = Buffer.isBuffer(raw) ? raw : Buffer.from(raw);
      }

      const mood = match[1].toLowerCase();
      const slug = match[2];
      const ext = match[3].toLowerCase();
      let pcm24k;
      try {
        pcm24k = ext === "wav" ? stripWavHeaderIfPresent(buffer) : buffer;
      } catch {
        continue;
      }
      if (pcm24k.length < 2 || pcm24k.length % 2 !== 0) {
        continue;
      }
      if (this.totalBytes + pcm24k.length > this.maxTotalBytes) {
        break;
      }

      if (this.registerClip({ id: `${mood}_${slug}`, mood, pcm24k })) {
        loaded += 1;
      }
    }

    return loaded;
  }
}

/**
 * Schedule a one-shot reaction cue after 150-180ms so fast live responses
 * cancel it before playback, while slower turns mask dead air naturally.
 */
export function scheduleReactionFiller({
  delayMs = 165,
  onFire,
  setTimer = setTimeout,
  clearTimer = clearTimeout,
} = {}) {
  if (!Number.isFinite(delayMs) || delayMs <= 0) {
    return {
      cancel() {},
      get fired() {
        return false;
      },
    };
  }

  const clampedDelay = Math.min(180, Math.max(150, Math.round(delayMs)));
  let fired = false;
  let cancelled = false;
  let timerId = setTimer(() => {
    if (cancelled || fired) return;
    fired = true;
    timerId = null;
    if (typeof onFire === "function") onFire();
  }, clampedDelay);

  return {
    cancel() {
      if (cancelled || fired) return;
      cancelled = true;
      if (timerId !== null) {
        clearTimer(timerId);
        timerId = null;
      }
    },
    get fired() {
      return fired;
    },
  };
}
