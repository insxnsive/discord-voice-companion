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

function assertAudioBytes(value, label) {
  if (!Buffer.isBuffer(value) && !(value instanceof Uint8Array)) {
    throw new TypeError(`${label} must be a Buffer or Uint8Array`);
  }
  return value;
}

function computeClipBytes({ frames48kOpus = [], pcm24k = null }) {
  let total = 0;
  if (pcm24k !== null && pcm24k !== undefined) {
    total += assertAudioBytes(pcm24k, "pcm24k").byteLength;
  }
  if (!Array.isArray(frames48kOpus)) {
    throw new TypeError("frames48kOpus must be an array of Buffer or Uint8Array values");
  }
  for (const frame of frames48kOpus) {
    total += assertAudioBytes(frame, "frames48kOpus entry").byteLength;
  }
  return total;
}

// Intrinsic typed-array accessors. A subclass or exotic value can override byteLength,
// length, buffer and byteOffset, so the real extent is read from %TypedArray%.prototype.
const TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(Uint8Array.prototype);
const INTRINSIC_BYTE_LENGTH = Object.getOwnPropertyDescriptor(TYPED_ARRAY_PROTOTYPE, "byteLength").get;
const INTRINSIC_BYTE_OFFSET = Object.getOwnPropertyDescriptor(TYPED_ARRAY_PROTOTYPE, "byteOffset").get;
const INTRINSIC_BUFFER = Object.getOwnPropertyDescriptor(TYPED_ARRAY_PROTOTYPE, "buffer").get;

function intrinsicByteLength(value) {
  return Reflect.apply(INTRINSIC_BYTE_LENGTH, value, []);
}

/**
 * Take one fixed snapshot of everything a clip will copy: the pcm24k container and each entry
 * of the frames list, in order. The frames list length and each element are read once. For
 * every container the backing buffer, offset and extent are measured once through the
 * intrinsic accessors and saved as plain numbers, and only then is the caller's declared
 * byteLength read, once. A fixed-length view over a resizable buffer goes out of bounds when
 * the store shrinks and regains its length when it regrows, so no view is kept: every check
 * works on the saved numbers, and the copy rebuilds an exact-extent view that throws if the
 * store no longer covers it. No caller accessor runs after this function returns.
 */
function snapshotClipSources({ frames48kOpus = [], pcm24k = null }) {
  const sources = [];
  const add = (value, label) => {
    assertAudioBytes(value, label);
    const buffer = Reflect.apply(INTRINSIC_BUFFER, value, []);
    const offset = Reflect.apply(INTRINSIC_BYTE_OFFSET, value, []);
    const extent = intrinsicByteLength(value);
    const declared = value.byteLength;
    sources.push({ label, buffer, offset, extent, declared });
  };
  if (pcm24k !== null && pcm24k !== undefined) add(pcm24k, "pcm24k");
  if (!Array.isArray(frames48kOpus)) {
    throw new TypeError("frames48kOpus must be an array of Buffer or Uint8Array values");
  }
  const count = frames48kOpus.length;
  const frames = [];
  for (let index = 0; index < count; index += 1) frames.push(frames48kOpus[index]);
  for (const frame of frames) add(frame, "frames48kOpus entry");
  return sources;
}

/**
 * Copy exactly the saved extent. The source view is built with an explicit length, so a
 * backing store that shrank below the saved extent throws a RangeError instead of yielding a
 * short or regrown copy, and the allocation is exactly the extent that the caps already bounded.
 */
function copySavedExtent({ buffer, offset, extent }) {
  const source = new Uint8Array(buffer, offset, extent);
  const clone = Buffer.alloc(extent);
  clone.set(source);
  return clone;
}

const FILE_ATTRIBUTE_REPARSE_POINT = 0x400;

/**
 * True for anything that is not a plain directory entry: symlinks, and on Windows junctions
 * and other reparse points (Node reports junctions as symlinks; fs implementations that
 * expose a reparse flag or raw file attributes are honoured too).
 */
function isLinkLike(stats) {
  if (!stats) return true;
  if (stats.isSymbolicLink?.() === true) return true;
  if (stats.isReparsePoint?.() === true || stats.isJunction?.() === true) return true;
  if (
    Number.isInteger(stats.fileAttributes) &&
    (stats.fileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) !== 0
  ) {
    return true;
  }
  return false;
}

function sameIdentity(recorded, stats) {
  if (
    Number.isFinite(recorded.dev) &&
    Number.isFinite(stats?.dev) &&
    typeof recorded.ino === "number" &&
    recorded.ino > 0 &&
    typeof stats?.ino === "number" &&
    stats.ino > 0
  ) {
    return recorded.dev === stats.dev && recorded.ino === stats.ino;
  }
  return true;
}

function lstatOrNull(fsImpl, target) {
  try {
    return fsImpl.lstatSync(target);
  } catch (error) {
    if (error && error.code === "ENOENT") return null;
    throw error;
  }
}

/**
 * Walk every component from the trusted root down to the cache directory with lstat and
 * return the verified chain, or null when the directory is absent or not a directory. Any
 * symlink or reparse point below the root throws. With createDir, missing components are
 * created one at a time (mode 0o700) only after every existing ancestor above them passed.
 */
function walkTrustedChain(dirPath, trustedRootDir, fsImpl, createDir) {
  const absRoot = path.resolve(trustedRootDir);
  const absDir = path.resolve(dirPath);
  const rel = path.relative(absRoot, absDir);
  if (rel === ".." || rel.startsWith(`..${path.sep}`) || path.isAbsolute(rel)) {
    throw new RangeError("Reaction cache directory is outside trusted root");
  }

  let current = absRoot;
  if (typeof fsImpl.realpathSync === "function") {
    try {
      current = path.normalize(fsImpl.realpathSync(absRoot));
    } catch (error) {
      if (error && error.code === "ENOENT") return null;
      throw error;
    }
  }

  const segments = rel === "" ? [] : rel.split(path.sep);
  const chain = [];
  for (let index = 0; index < segments.length; index += 1) {
    const isLeaf = index === segments.length - 1;
    current = path.join(current, segments[index]);
    let stats = lstatOrNull(fsImpl, current);
    if (!stats) {
      // A missing component ends the load. Skipping a missing intermediate would leave it
      // unchecked by every later step if it appears or is replaced concurrently.
      if (!createDir || typeof fsImpl.mkdirSync !== "function") return null;
      fsImpl.mkdirSync(current, { mode: 0o700 });
      stats = fsImpl.lstatSync(current);
    }
    if (isLinkLike(stats)) {
      throw new RangeError(
        isLeaf
          ? "Reaction cache directory cannot be a symlink"
          : "Reaction cache path component cannot be a symlink or reparse point",
      );
    }
    if (!stats.isDirectory?.()) return null;
    if (process.platform !== "win32" && Number.isInteger(stats.mode) && (stats.mode & 0o002) !== 0) {
      throw new RangeError("Reaction cache path component cannot be world-writable");
    }
    chain.push({ path: current, dev: stats.dev, ino: stats.ino });
  }
  return chain;
}

function verifyChainUnchanged(chain, fsImpl) {
  for (const entry of chain) {
    let stats = null;
    try {
      stats = fsImpl.lstatSync(entry.path);
    } catch {
      stats = null;
    }
    if (isLinkLike(stats) || !stats.isDirectory?.() || !sameIdentity(entry, stats)) {
      throw new RangeError("Reaction cache path changed during load");
    }
  }
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
    const sources = snapshotClipSources({ frames48kOpus, pcm24k });
    let clipBytes = 0;
    for (const source of sources) clipBytes += source.declared;
    if (clipBytes > this.maxFileBytes) {
      throw new RangeError("Reaction clip exceeds maximum file byte limit");
    }
    // The declared byteLength is caller-controlled. Bound the copy by the saved extents, so
    // peak allocation never exceeds one clip cap and the copy cannot see a larger extent.
    let pinnedBytes = 0;
    for (const source of sources) pinnedBytes += source.extent;
    if (pinnedBytes > this.maxFileBytes) {
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
    for (const source of sources) {
      if (source.declared !== source.extent) {
        throw new RangeError(`${source.label} declared byte length does not match its extent`);
      }
    }
    const clones = sources.map((source) => ({ label: source.label, clone: copySavedExtent(source) }));
    const clonedPcm = clones.find((entry) => entry.label === "pcm24k")?.clone ?? null;
    const clonedFrameList = clones
      .filter((entry) => entry.label !== "pcm24k")
      .map((entry) => entry.clone);
    const clonedBytes = clones.reduce((sum, entry) => sum + entry.clone.length, 0);
    if (clonedBytes > this.maxFileBytes) {
      throw new RangeError("Reaction clip exceeds maximum file byte limit");
    }
    const clonedTotalBytes = this.totalBytes - prevBytes + clonedBytes;
    if (clonedTotalBytes > this.maxTotalBytes) {
      throw new RangeError("ReactionPool exceeded 8 MiB aggregate byte limit");
    }
    const clonedFrames = Object.freeze(clonedFrameList);
    this.clips.set(
      id,
      Object.freeze({ id, mood, frames48kOpus: clonedFrames, pcm24k: clonedPcm }),
    );
    this.totalBytes = clonedTotalBytes;
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

  loadFromDirectory(
    dirPath,
    { fsImpl = fs, maxFileBytes = this.maxFileBytes, trustedRootDir, createDir = false } = {},
  ) {
    if (!dirPath) return 0;

    // With a trusted root, every component below it is verified with lstat before anything
    // is read, and re-verified around each read. Residual race: Node has no openat, so a
    // component swapped and restored between a verification and the path-based open can
    // still go unseen; the descriptor identity checks below and the 0o700 private directory
    // (POSIX only, Windows relies on profile ACLs) narrow that window but do not close it.
    let chain = [];
    const anchored =
      typeof trustedRootDir === "string" &&
      trustedRootDir.trim().length > 0 &&
      typeof fsImpl.lstatSync === "function";
    if (anchored) {
      chain = walkTrustedChain(dirPath, trustedRootDir, fsImpl, createDir === true);
      if (chain === null) return 0;
    } else if (typeof fsImpl.existsSync === "function" && !fsImpl.existsSync(dirPath)) {
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

    let resolvedDir = dirPath;
    if (typeof fsImpl.realpathSync === "function") {
      try {
        resolvedDir = path.normalize(fsImpl.realpathSync(dirPath));
      } catch (error) {
        if (error && error.code === "ENOENT") return 0;
        throw error;
      }
      if (typeof trustedRootDir === "string" && trustedRootDir.trim().length > 0) {
        const resolvedRoot = path.normalize(fsImpl.realpathSync(trustedRootDir));
        const rootPrefix = resolvedRoot.endsWith(path.sep)
          ? resolvedRoot
          : `${resolvedRoot}${path.sep}`;
        if (resolvedDir !== resolvedRoot && !resolvedDir.startsWith(rootPrefix)) {
          throw new RangeError("Reaction cache directory is outside trusted root");
        }
      }
    }

    const dirPrefix = resolvedDir.endsWith(path.sep) ? resolvedDir : `${resolvedDir}${path.sep}`;

    const effectiveMaxFileBytes =
      Number.isInteger(maxFileBytes) && maxFileBytes > 0
        ? Math.min(this.maxFileBytes, maxFileBytes)
        : this.maxFileBytes;

    // Verify the ancestors immediately before the listing and again right after it, so the
    // listing is bracketed by two checks and a swap during it cannot go unseen.
    verifyChainUnchanged(chain, fsImpl);
    let entries;
    try {
      entries = fsImpl.readdirSync(resolvedDir);
    } catch (error) {
      if (error && error.code === "ENOENT") return 0;
      throw error;
    }
    verifyChainUnchanged(chain, fsImpl);

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
      verifyChainUnchanged(chain, fsImpl);

      const fullPath = path.join(resolvedDir, entry);
      if (typeof fsImpl.realpathSync === "function") {
        let resolvedEntry;
        try {
          resolvedEntry = path.normalize(fsImpl.realpathSync(fullPath));
        } catch {
          continue;
        }
        if (!resolvedEntry.startsWith(dirPrefix)) {
          continue;
        }
      }

      let entryLstat = null;
      if (typeof fsImpl.lstatSync === "function") {
        try {
          entryLstat = fsImpl.lstatSync(fullPath);
        } catch {
          continue;
        }
        if (
          entryLstat?.isSymbolicLink?.() ||
          !entryLstat?.isFile?.() ||
          entryLstat.size <= 0 ||
          entryLstat.size > effectiveMaxFileBytes
        ) {
          continue;
        }
      }

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
          if (
            entryLstat &&
            Number.isFinite(entryLstat.dev) &&
            Number.isFinite(stat?.dev) &&
            typeof entryLstat.ino === "number" &&
            entryLstat.ino > 0 &&
            typeof stat?.ino === "number" &&
            stat.ino > 0 &&
            (entryLstat.ino !== stat.ino || entryLstat.dev !== stat.dev)
          ) {
            continue;
          }
          if (
            typeof entryLstat?.ino === "number" &&
            entryLstat.ino > 0 &&
            typeof stat?.ino === "number" &&
            stat.ino > 0 &&
            entryLstat.ino !== stat.ino
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
        if (!entryLstat) continue;
        const raw = fsImpl.readFileSync(fullPath);
        buffer = Buffer.isBuffer(raw) ? raw : Buffer.from(raw);
      }

      const mood = match[1].toLowerCase();
      verifyChainUnchanged(chain, fsImpl);
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
