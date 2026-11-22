// Zero-ffmpeg PCM helpers for Discord voice. Discord wants 48kHz stereo signed
// 16-bit LE in 20ms frames; STT wants 16kHz mono and TTS returns 24kHz mono.

export const DISCORD_SAMPLE_RATE = 48000;
export const DISCORD_CHANNELS = 2;
export const DISCORD_FRAME_SAMPLES = 960;
export const DISCORD_FRAME_BYTES = 3840;
export const VAD_SAMPLE_RATE = 16000;
export const TTS_SAMPLE_RATE = 24000;

// Hard caps so a malformed or hostile producer cannot force a huge allocation.
export const MAX_AUDIO_CHUNK_BYTES = 1048576;
export const MAX_OPUS_PACKET_BYTES = 4000;

function toBuffer(input, what = "PCM16LE audio") {
  if (Buffer.isBuffer(input)) return input;
  if (input instanceof Uint8Array) {
    return Buffer.from(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new TypeError(`expected a Buffer or Uint8Array of ${what}`);
}

function assertAudioChunk(input, maxBytes = MAX_AUDIO_CHUNK_BYTES) {
  if (input.byteLength > maxBytes) {
    throw new RangeError(
      `audio chunk is ${input.byteLength} bytes, above the ${maxBytes} byte limit`,
    );
  }
}

function assertPositiveLimit(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0 || value > MAX_AUDIO_CHUNK_BYTES) {
    throw new RangeError(
      `${label} must be a positive integer <= ${MAX_AUDIO_CHUNK_BYTES}, got ${value}`,
    );
  }
}

/**
 * 48kHz stereo to 16kHz mono by averaging left, right, and the three source
 * frames that collapse into one output sample (12 input bytes -> 2 output
 * bytes). A trailing partial group is ignored rather than thrown away loudly.
 */
export function downsample48kStereoTo16kMono(pcm48Stereo) {
  const input = toBuffer(pcm48Stereo);
  assertAudioChunk(input);
  const groupBytes = 12;
  const groups = Math.floor(input.length / groupBytes);
  const output = Buffer.allocUnsafe(groups * 2);

  for (let group = 0; group < groups; group += 1) {
    const base = group * groupBytes;
    let sum = 0;
    for (let sample = 0; sample < 6; sample += 1) {
      sum += input.readInt16LE(base + sample * 2);
    }
    let value = Math.round(sum / 6);
    if (value > 32767) value = 32767;
    if (value < -32768) value = -32768;
    output.writeInt16LE(value, group * 2);
  }

  return output;
}

/** Signed 16-bit LE mono PCM to float32 samples in [-1, 1] for Silero VAD. */
export function pcm16MonoToFloat32(pcm16Mono) {
  const input = toBuffer(pcm16Mono);
  assertAudioChunk(input);
  const count = Math.floor(input.length / 2);
  const samples = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    samples[i] = input.readInt16LE(i * 2) / 32768;
  }
  return samples;
}

/**
 * Stateful 24kHz mono to 48kHz stereo upsampler. Each source sample emits the
 * midpoint between the previous sample and itself, then the sample, both
 * duplicated to left and right. Carrying prevSample across pushes keeps
 * streaming TTS chunks click-free; reset() drops that carry.
 */
export function createUpsampler24kTo48kStereo() {
  let prevSample = 0;

  return {
    push(pcm24Mono) {
      const input = toBuffer(pcm24Mono);
      assertAudioChunk(input);
      const samples = Math.floor(input.length / 2);
      const output = Buffer.allocUnsafe(samples * 8);
      let offset = 0;

      for (let i = 0; i < samples; i += 1) {
        const sample = input.readInt16LE(i * 2);
        const midpoint = Math.round((prevSample + sample) / 2);

        output.writeInt16LE(midpoint, offset);
        offset += 2;
        output.writeInt16LE(midpoint, offset);
        offset += 2;
        output.writeInt16LE(sample, offset);
        offset += 2;
        output.writeInt16LE(sample, offset);
        offset += 2;

        prevSample = sample;
      }

      return output;
    },
    reset() {
      prevSample = 0;
    },
  };
}

/** One-shot wrapper for callers that do not need cross-chunk continuity. */
export function upsample24kMonoTo48kStereo(pcm24Mono) {
  return createUpsampler24kTo48kStereo().push(pcm24Mono);
}

/**
 * Buffers arbitrary chunk sizes and hands back exact Opus frame buffers. The
 * short tail is retained until flushPadded() turns it into a final frame.
 */
export function createFrameAccumulator(
  frameBytes = DISCORD_FRAME_BYTES,
  { maxChunkBytes = MAX_AUDIO_CHUNK_BYTES } = {},
) {
  assertPositiveLimit(frameBytes, "frameBytes");
  assertPositiveLimit(maxChunkBytes, "maxChunkBytes");
  if (frameBytes > maxChunkBytes) {
    throw new RangeError(
      `frameBytes (${frameBytes}) must not exceed maxChunkBytes (${maxChunkBytes})`,
    );
  }
  let pending = Buffer.alloc(0);

  return {
    push(chunk) {
      const input = toBuffer(chunk);
      assertAudioChunk(input, maxChunkBytes);
      if (input.length === 0) return [];
      pending = pending.length === 0 ? Buffer.from(input) : Buffer.concat([pending, input]);

      const frames = [];
      let offset = 0;
      while (pending.length - offset >= frameBytes) {
        frames.push(Buffer.from(pending.subarray(offset, offset + frameBytes)));
        offset += frameBytes;
      }
      // Copy the short remainder instead of keeping a view that pins the
      // concatenated parent buffer.
      if (offset > 0) pending = Buffer.from(pending.subarray(offset));
      return frames;
    },
    flushPadded() {
      if (pending.length === 0) return null;
      const frame = Buffer.alloc(frameBytes);
      pending.copy(frame, 0, 0, Math.min(pending.length, frameBytes));
      pending = Buffer.alloc(0);
      return frame;
    },
    reset() {
      pending = Buffer.alloc(0);
    },
  };
}

function describeError(error) {
  return error instanceof Error ? error.message : String(error);
}

function assertPcmFrame(frame) {
  if (!Buffer.isBuffer(frame) && !(frame instanceof Uint8Array)) {
    throw new TypeError("pcm48StereoFrame must be a Buffer or Uint8Array");
  }
  if (frame.byteLength !== DISCORD_FRAME_BYTES) {
    throw new RangeError(
      `pcm48StereoFrame must be exactly ${DISCORD_FRAME_BYTES} bytes, got ${frame.byteLength}`,
    );
  }
}

function assertOpusPacket(packet) {
  if (!Buffer.isBuffer(packet) && !(packet instanceof Uint8Array)) {
    throw new TypeError("opusPacket must be a Buffer or Uint8Array");
  }
  if (packet.byteLength < 1 || packet.byteLength > MAX_OPUS_PACKET_BYTES) {
    throw new RangeError(
      `opusPacket must be 1..${MAX_OPUS_PACKET_BYTES} bytes, got ${packet.byteLength}`,
    );
  }
}

function wrapEncoder(backend, encoder, decoderFactory) {
  return {
    backend,
    encode(pcm48StereoFrame) {
      assertPcmFrame(pcm48StereoFrame);
      if (backend === "opusscript") {
        return encoder.encode(pcm48StereoFrame, DISCORD_FRAME_SAMPLES);
      }
      return encoder.encode(pcm48StereoFrame);
    },
    decode(opusPacket) {
      assertOpusPacket(opusPacket);
      return encoder.decode(opusPacket);
    },
    createDecoder() {
      if (typeof decoderFactory === "function") {
        const decoder = decoderFactory();
        if (decoder && typeof decoder.decode === "function") {
          let closed = false;
          return {
            decode(opusPacket) {
              assertOpusPacket(opusPacket);
              return decoder.decode(opusPacket);
            },
            close() {
              if (closed) return;
              closed = true;
              try {
                decoder.delete?.();
              } catch {
                // Teardown is best effort.
              }
              try {
                decoder.close?.();
              } catch {
                // Teardown is best effort.
              }
            },
          };
        }
      }
      return {
        decode(opusPacket) {
          assertOpusPacket(opusPacket);
          return encoder.decode(opusPacket);
        },
        close() {
          // The shared encoder must never be deleted.
        },
      };
    },
  };
}

/**
 * Build an Opus encoder for 48kHz stereo frames, preferring the native
 * @discordjs/opus binding and falling back to the pure-JS opusscript wasm
 * encoder. Injection hooks exist so tests never need either package.
 */
export async function createOpusTranscoder({ loadNative, loadJs } = {}) {
  const nativeLoader = loadNative ?? (() => import("@discordjs/opus"));
  const jsLoader = loadJs ?? (() => import("opusscript"));

  let nativeFailure;
  try {
    const module = await nativeLoader();
    const OpusEncoder = module?.OpusEncoder ?? module?.default?.OpusEncoder ?? module?.default;
    if (typeof OpusEncoder !== "function") {
      throw new TypeError("@discordjs/opus did not expose an OpusEncoder constructor");
    }
    const encoder = new OpusEncoder(DISCORD_SAMPLE_RATE, DISCORD_CHANNELS);
    return wrapEncoder("native", encoder, () => new OpusEncoder(DISCORD_SAMPLE_RATE, DISCORD_CHANNELS));
  } catch (error) {
    nativeFailure = error;
  }

  let jsFailure;
  try {
    const module = await jsLoader();
    const OpusScript = module?.default ?? module;
    if (typeof OpusScript !== "function") {
      throw new TypeError("opusscript did not expose a constructor");
    }
    const application = (OpusScript.Application ?? module?.Application ?? { AUDIO: 2049 }).AUDIO ?? 2049;
    const encoder = new OpusScript(DISCORD_SAMPLE_RATE, DISCORD_CHANNELS, application);
    return wrapEncoder(
      "opusscript",
      encoder,
      () => new OpusScript(DISCORD_SAMPLE_RATE, DISCORD_CHANNELS, application),
    );
  } catch (error) {
    jsFailure = error;
  }

  throw new Error(
    `No Opus backend available: install @discordjs/opus or opusscript ` +
      `(native: ${describeError(nativeFailure)}; opusscript: ${describeError(jsFailure)})`,
  );
}
