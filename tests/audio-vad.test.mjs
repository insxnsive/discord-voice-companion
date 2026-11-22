import test from "node:test";
import assert from "node:assert/strict";

import {
  DISCORD_FRAME_BYTES,
  DISCORD_FRAME_SAMPLES,
  DISCORD_SAMPLE_RATE,
  MAX_AUDIO_CHUNK_BYTES,
  MAX_OPUS_PACKET_BYTES,
  TTS_SAMPLE_RATE,
  VAD_SAMPLE_RATE,
  createFrameAccumulator,
  createOpusTranscoder,
  createUpsampler24kTo48kStereo,
  downsample48kStereoTo16kMono,
  pcm16MonoToFloat32,
  upsample24kMonoTo48kStereo,
} from "../src/audio/pcm.mjs";

import {
  PerSpeakerVad,
  computeRms,
  createRmsSpeechDetector,
  createSileroDetector,
} from "../src/audio/vad.mjs";

const VAD_FRAME_SAMPLES = (VAD_SAMPLE_RATE * 20) / 1000;
const VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * 2;

function monoFrame(amplitude = 0, samples = VAD_FRAME_SAMPLES) {
  const buffer = Buffer.alloc(samples * 2);
  for (let i = 0; i < samples; i += 1) buffer.writeInt16LE(amplitude, i * 2);
  return buffer;
}

function stereoFrames(pairs) {
  const buffer = Buffer.alloc(pairs.length * 4);
  pairs.forEach(([left, right], index) => {
    buffer.writeInt16LE(left, index * 4);
    buffer.writeInt16LE(right, index * 4 + 2);
  });
  return buffer;
}

function rampPcm24k(samples) {
  const buffer = Buffer.alloc(samples * 2);
  for (let i = 0; i < samples; i += 1) buffer.writeInt16LE(-1000 + i * 8, i * 2);
  return buffer;
}

function makeVad(overrides = {}) {
  const events = { starts: [], ends: [], barges: [], drops: [] };
  const vad = new PerSpeakerVad({
    onSpeechStart: (event) => events.starts.push(event),
    onSpeechEnd: (event) => events.ends.push(event),
    onBargeIn: (event) => events.barges.push(event),
    onNoiseDrop: (event) => events.drops.push(event),
    ...overrides,
  });
  return { vad, events };
}

test("pcm constants describe a Discord 20ms stereo frame and the resampling rates", () => {
  assert.equal(DISCORD_SAMPLE_RATE, 48000);
  assert.equal(DISCORD_FRAME_SAMPLES, 960);
  assert.equal(DISCORD_FRAME_BYTES, 3840);
  assert.equal(VAD_SAMPLE_RATE, 16000);
  assert.equal(TTS_SAMPLE_RATE, 24000);
});

test("downsample48kStereoTo16kMono averages channels and yields one sixth of the bytes", () => {
  const pairs = [];
  for (let group = 0; group < 320; group += 1) {
    pairs.push([0, 0], [3000, 3000], [0, 0]);
  }
  const input = stereoFrames(pairs);
  assert.equal(input.length, 3840);

  const output = downsample48kStereoTo16kMono(input);
  assert.equal(output.length, 640);
  assert.equal(output.length, input.length / 6);
  for (let i = 0; i < output.length; i += 2) {
    assert.equal(output.readInt16LE(i), 1000);
  }

  const skewed = downsample48kStereoTo16kMono(
    stereoFrames([
      [1000, 3000],
      [1000, 3000],
      [1000, 3000],
    ]),
  );
  assert.equal(skewed.length, 2);
  assert.equal(skewed.readInt16LE(0), 2000);

  const trailing = Buffer.concat([input, Buffer.alloc(11)]);
  assert.equal(downsample48kStereoTo16kMono(trailing).length, 640);
  const complete = Buffer.concat([input, Buffer.alloc(12)]);
  assert.equal(downsample48kStereoTo16kMono(complete).length, 642);
});

test("pcm16MonoToFloat32 normalizes signed samples into the -1..1 range", () => {
  const input = Buffer.alloc(6);
  input.writeInt16LE(0, 0);
  input.writeInt16LE(16384, 2);
  input.writeInt16LE(-32768, 4);

  const samples = pcm16MonoToFloat32(input);
  assert.equal(samples.length, 3);
  assert.equal(samples[0], 0);
  assert.equal(samples[1], 0.5);
  assert.equal(samples[2], -1);
});

test("createUpsampler24kTo48kStereo interpolates, quadruples bytes, and survives chunking", () => {
  const samples = 100;
  const full = rampPcm24k(samples);
  const stateful = createUpsampler24kTo48kStereo();

  const streamed = Buffer.concat([stateful.push(full.subarray(0, 40)), stateful.push(full.subarray(40))]);
  assert.equal(streamed.length, 4 * full.length);

  const stateless = upsample24kMonoTo48kStereo(full);
  assert.ok(streamed.equals(stateless), "chunked output must match one-shot output");

  assert.equal(stateless.readInt16LE(0), -500);
  assert.equal(stateless.readInt16LE(2), -500);
  assert.equal(stateless.readInt16LE(4), -1000);
  assert.equal(stateless.readInt16LE(6), -1000);
  assert.equal(stateless.readInt16LE(8), -996);
  assert.equal(stateless.readInt16LE(10), -996);

  stateful.reset();
  const restarted = stateful.push(rampPcm24k(1));
  assert.equal(restarted.readInt16LE(0), -500);
  assert.equal(restarted.readInt16LE(4), -1000);
});

test("createFrameAccumulator slices exact frames and zero-pads the tail", () => {
  const accumulator = createFrameAccumulator();
  const chunk = Buffer.alloc(3840 + 1500);
  for (let i = 0; i < chunk.length; i += 1) chunk[i] = i % 251;

  const frames = accumulator.push(chunk);
  assert.equal(frames.length, 1);
  assert.equal(frames[0].length, 3840);
  assert.ok(frames[0].equals(chunk.subarray(0, 3840)));

  const padded = accumulator.flushPadded();
  assert.equal(padded.length, 3840);
  assert.ok(padded.subarray(0, 1500).equals(chunk.subarray(3840)));
  assert.ok(padded.subarray(1500).every((byte) => byte === 0));

  assert.equal(accumulator.flushPadded(), null);
  assert.deepEqual(accumulator.push(Buffer.alloc(0)), []);

  const exact = createFrameAccumulator().push(Buffer.alloc(3840 * 2));
  assert.equal(exact.length, 2);
});

test("createOpusTranscoder prefers native, falls back to opusscript, and forwards packets", async () => {
  class MockNativeEncoder {
    constructor(rate, channels) {
      this.args = [rate, channels];
    }

    encode(frame) {
      return Buffer.concat([Buffer.from([0x0a]), frame.subarray(0, 2)]);
    }

    decode(packet) {
      return Buffer.concat([Buffer.from([0x0b]), packet]);
    }
  }

  const nativeOnly = await createOpusTranscoder({
    loadNative: async () => ({ OpusEncoder: MockNativeEncoder }),
  });
  assert.equal(nativeOnly.backend, "native");
  assert.deepEqual(nativeOnly.encode(Buffer.alloc(DISCORD_FRAME_BYTES, 3))[0], 0x0a);

  const calls = [];
  class MockOpusScript {
    static Application = { AUDIO: 2049 };

    constructor(rate, channels, application) {
      calls.push([rate, channels, application]);
    }

    encode(frame) {
      return Buffer.concat([Buffer.from([0x01]), frame.subarray(0, 4)]);
    }

    decode(packet) {
      return Buffer.concat([Buffer.from([0x02]), packet]);
    }
  }

  const transcoder = await createOpusTranscoder({
    loadNative: async () => {
      throw new Error("native module not installed");
    },
    loadJs: async () => ({ default: MockOpusScript }),
  });

  assert.equal(transcoder.backend, "opusscript");
  assert.deepEqual(calls, [[DISCORD_SAMPLE_RATE, 2, 2049]]);

  const frame = Buffer.alloc(DISCORD_FRAME_BYTES, 7);
  frame.writeUInt8(9, 0);
  const encoded = transcoder.encode(frame);
  assert.equal(encoded[0], 0x01);
  assert.ok(encoded.subarray(1).equals(frame.subarray(0, 4)));

  const decoded = transcoder.decode(Buffer.from([4, 3]));
  assert.equal(decoded[0], 0x02);
  assert.ok(decoded.subarray(1).equals(Buffer.from([4, 3])));

  await assert.rejects(
    () =>
      createOpusTranscoder({
        loadNative: async () => {
          throw new Error("no native");
        },
        loadJs: async () => {
          throw new Error("no js");
        },
      }),
    /Opus backend/,
  );
});

test("PerSpeakerVad ignores the bot, configured users, and frames without a user id", async () => {
  const { vad, events } = makeVad({ botUserId: "bot-1", ignoredUserIds: ["muted-1"] });

  for (let i = 0; i < 5; i += 1) {
    const timestampMs = i * 20;
    assert.deepEqual(
      await vad.pushFrame({ userId: "bot-1", pcm16kMono: monoFrame(9000), timestampMs }),
      { ignored: true },
    );
    assert.deepEqual(
      await vad.pushFrame({ userId: "muted-1", pcm16kMono: monoFrame(9000), timestampMs }),
      { ignored: true },
    );
    assert.deepEqual(
      await vad.pushFrame({ userId: null, pcm16kMono: monoFrame(9000), timestampMs }),
      { ignored: true },
    );
  }

  assert.equal(events.starts.length, 0);
  assert.equal(events.ends.length, 0);

  vad.setIgnoredUsers([]);
  const result = await vad.pushFrame({ userId: "muted-1", pcm16kMono: monoFrame(9000), timestampMs: 200 });
  assert.equal(result.ignored, false);
  assert.equal(events.starts.length, 1);
});

test("PerSpeakerVad drops sub-250ms blips and emits merged PCM for real utterances", async () => {
  const { vad, events } = makeVad();

  for (let i = 0; i < 5; i += 1) {
    await vad.pushFrame({ userId: "u1", displayName: "Ana", pcm16kMono: monoFrame(9000), timestampMs: i * 20 });
  }
  for (let i = 0; i < 12; i += 1) {
    await vad.pushFrame({ userId: "u1", displayName: "Ana", pcm16kMono: monoFrame(0), timestampMs: 100 + i * 20 });
  }

  assert.equal(events.drops.length, 1);
  assert.equal(events.drops[0].speechMs, 100);
  assert.equal(events.ends.length, 0);

  let timestampMs = 1000;
  for (let i = 0; i < 15; i += 1) {
    timestampMs += 20;
    await vad.pushFrame({ userId: "u1", displayName: "Ana", pcm16kMono: monoFrame(9000), timestampMs });
  }
  for (let i = 0; i < 12; i += 1) {
    timestampMs += 20;
    await vad.pushFrame({ userId: "u1", displayName: "Ana", pcm16kMono: monoFrame(0), timestampMs });
  }

  assert.equal(events.ends.length, 1);
  const utterance = events.ends[0];
  assert.equal(utterance.userId, "u1");
  assert.equal(utterance.displayName, "Ana");
  assert.equal(utterance.durationMs, 300);
  assert.equal(utterance.endedAt, timestampMs);
  assert.equal(utterance.pcm16kMono.length, (15 + 12) * VAD_FRAME_BYTES);
});

test("PerSpeakerVad fires barge-in once per utterance and caps long utterances", async () => {
  const { vad, events } = makeVad({
    botUserId: "bot-1",
    isBotSpeaking: () => true,
  });

  let timestampMs = 0;
  for (let i = 0; i < 8; i += 1) {
    timestampMs += 20;
    await vad.pushFrame({ userId: "u2", displayName: "Ben", pcm16kMono: monoFrame(9000), timestampMs });
  }
  assert.equal(events.barges.length, 0);

  for (let i = 0; i < 6; i += 1) {
    timestampMs += 20;
    await vad.pushFrame({ userId: "u2", displayName: "Ben", pcm16kMono: monoFrame(9000), timestampMs });
  }

  assert.equal(events.barges.length, 1);
  assert.equal(events.barges[0].userId, "u2");
  assert.equal(events.barges[0].displayName, "Ben");
  assert.equal(events.barges[0].timestampMs, 180);
  assert.equal(events.ends.length, 0);

  const capped = makeVad({ maxUtteranceMs: 60, minSpeechMs: 40, hangoverMs: 500 });
  for (let i = 0; i < 3; i += 1) {
    await capped.vad.pushFrame({ userId: "u3", pcm16kMono: monoFrame(9000), timestampMs: i * 20 });
  }
  assert.equal(capped.events.ends.length, 1);
  assert.equal(capped.events.ends[0].durationMs, 60);

  await capped.vad.pushFrame({ userId: "u3", pcm16kMono: monoFrame(9000), timestampMs: 80 });
  assert.equal(capped.events.starts.length, 2);
});

test("computeRms tracks amplitude and the RMS detector maps it to confidence", () => {
  assert.equal(computeRms(new Float32Array(0)), 0);
  assert.equal(computeRms(new Float32Array([0, 0, 0])), 0);

  const halfScale = new Float32Array(160);
  halfScale.fill(0.5);
  assert.equal(computeRms(halfScale), 0.5);

  const detect = createRmsSpeechDetector({ threshold: 0.018 });
  assert.equal(detect(new Float32Array(160)), 0);
  assert.equal(detect(halfScale), 0.9);

  const quiet = new Float32Array(160);
  quiet.fill(0.009);
  const quietProbability = detect(quiet);
  assert.ok(quietProbability > 0);
  assert.ok(quietProbability < 0.5);
});

test("PerSpeakerVad rejects malformed frames before creating speaker state", async () => {
  const { vad } = makeVad();

  await assert.rejects(
    () => vad.pushFrame({ userId: "u1", pcm16kMono: Buffer.alloc(641), timestampMs: 0 }),
    RangeError,
  );
  await assert.rejects(
    () => vad.pushFrame({ userId: "u1", pcm16kMono: Buffer.alloc(6401), timestampMs: 0 }),
    RangeError,
  );
  await assert.rejects(
    () => vad.pushFrame({ userId: "u1", pcm16kMono: new Float32Array(320), timestampMs: 0 }),
    TypeError,
  );

  assert.equal(vad.speakers.size, 0);
});

test("PerSpeakerVad stops an utterance at the byte budget during short pauses", async () => {
  const { vad, events } = makeVad({ minSpeechMs: 40, maxUtteranceBytes: 1920, hangoverMs: 240 });
  let timestampMs = 0;

  await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000), timestampMs: (timestampMs += 20) });
  await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000), timestampMs: (timestampMs += 20) });
  await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(0), timestampMs: (timestampMs += 20) });

  assert.equal(events.drops.length, 0);
  assert.equal(events.ends.length, 1);
  assert.equal(events.ends[0].durationMs, 40);
  assert.equal(events.ends[0].pcm16kMono.length, 3 * VAD_FRAME_BYTES);
});

test("PerSpeakerVad bounds the speaker map and evicts idle speakers", async () => {
  const bounded = makeVad({ maxSpeakers: 3 }).vad;
  for (let i = 0; i < 5; i += 1) {
    await bounded.pushFrame({ userId: `user-${i}`, pcm16kMono: monoFrame(0), timestampMs: 1000 });
  }
  assert.equal(bounded.speakers.size, 3);

  const transient = makeVad({ maxSpeakers: 4, idleSpeakerTtlMs: 100 }).vad;
  for (let i = 0; i < 100; i += 1) {
    await transient.pushFrame({ userId: `transient-${i}`, pcm16kMono: monoFrame(0), timestampMs: i * 1000 });
  }
  assert.ok(transient.speakers.size <= 4);
});

test("PerSpeakerVad drops buffered state when muted or reset mid-detection", async () => {
  let releaseMute;
  const muteGate = new Promise((resolve) => {
    releaseMute = resolve;
  });
  const muted = { starts: [], ends: [] };
  const vadMuted = new PerSpeakerVad({
    detectorFactory: () => async () => {
      await muteGate;
      return 0.9;
    },
    onSpeechStart: (event) => muted.starts.push(event),
    onSpeechEnd: (event) => muted.ends.push(event),
  });

  const mutedFrame = vadMuted.pushFrame({
    userId: "u1",
    displayName: "Ana",
    pcm16kMono: monoFrame(9000),
    timestampMs: 20,
  });
  await new Promise((resolve) => setImmediate(resolve));
  vadMuted.setIgnoredUsers(["u1"]);
  releaseMute();
  await mutedFrame;

  assert.equal(vadMuted.speakers.has("u1"), false);
  assert.equal(muted.starts.length, 0);
  assert.equal(muted.ends.length, 0);

  let releaseReset;
  const resetGate = new Promise((resolve) => {
    releaseReset = resolve;
  });
  const reset = { ends: [] };
  const vadReset = new PerSpeakerVad({
    detectorFactory: () => async () => {
      await resetGate;
      return 0.9;
    },
    onSpeechEnd: (event) => reset.ends.push(event),
  });

  const resetFrame = vadReset.pushFrame({ userId: "u2", pcm16kMono: monoFrame(9000), timestampMs: 20 });
  await new Promise((resolve) => setImmediate(resolve));
  vadReset.resetSpeaker("u2");
  releaseReset();
  await resetFrame;

  assert.equal(vadReset.speakers.has("u2"), false);
  assert.equal(reset.ends.length, 0);
});

test("PerSpeakerVad serializes concurrent frames and applies backpressure", async () => {
  let factoryCalls = 0;
  const vad = new PerSpeakerVad({
    maxPendingFrames: 4,
    detectorFactory: async () => {
      factoryCalls += 1;
      await new Promise((resolve) => setTimeout(resolve, 5));
      return () => 0.9;
    },
  });

  const results = await Promise.all(
    Array.from({ length: 10 }, (_, index) =>
      vad.pushFrame({ userId: "new-user", pcm16kMono: monoFrame(9000), timestampMs: index * 20 }),
    ),
  );

  assert.equal(factoryCalls, 1);
  assert.equal(vad.speakers.size, 1);
  assert.ok(results.filter((result) => result?.dropped === "backpressure").length > 0);
});

test("createFrameAccumulator copies frames and rejects oversized chunks", () => {
  assert.throws(() => createFrameAccumulator().push(Buffer.alloc(MAX_AUDIO_CHUNK_BYTES + 2)), RangeError);

  const accumulator = createFrameAccumulator(DISCORD_FRAME_BYTES, { maxChunkBytes: 8192 });
  assert.throws(() => accumulator.push(Buffer.alloc(8193)), RangeError);

  const chunk = Buffer.alloc(8192, 7);
  const frames = accumulator.push(chunk);
  assert.equal(frames.length, 2);
  assert.notEqual(frames[0].buffer, chunk.buffer);
  chunk.fill(0);
  assert.equal(frames[0][0], 7);
  assert.equal(frames[1][0], 7);
});

test("createOpusTranscoder validates frame size and packet bounds", async () => {
  class MockEncoder {
    encode(frame) {
      return Buffer.from([frame.length % 256]);
    }

    decode(packet) {
      return Buffer.alloc(packet.length);
    }
  }

  const transcoder = await createOpusTranscoder({
    loadNative: async () => ({ OpusEncoder: MockEncoder }),
  });

  assert.throws(() => transcoder.encode(Buffer.alloc(DISCORD_FRAME_BYTES - 2)), RangeError);
  assert.throws(() => transcoder.encode(Buffer.alloc(DISCORD_FRAME_BYTES + 2)), RangeError);
  assert.throws(() => transcoder.encode("not pcm"), TypeError);
  assert.equal(transcoder.encode(Buffer.alloc(DISCORD_FRAME_BYTES))[0], DISCORD_FRAME_BYTES % 256);

  assert.throws(() => transcoder.decode(Buffer.alloc(0)), RangeError);
  assert.throws(() => transcoder.decode(Buffer.alloc(MAX_OPUS_PACKET_BYTES + 1)), RangeError);
  assert.throws(() => transcoder.decode(new Float32Array(2)), TypeError);
  assert.equal(transcoder.decode(Buffer.alloc(MAX_OPUS_PACKET_BYTES)).length, MAX_OPUS_PACKET_BYTES);
  assert.equal(transcoder.decode(Buffer.alloc(1)).length, 1);
});

function createOrtMock(probabilities) {
  const runs = [];
  const ort = {
    Tensor: class {
      constructor(type, data, dims) {
        this.type = type;
        this.data = data;
        this.dims = dims;
      }
    },
    InferenceSession: {
      create: async () => ({
        run: async (feeds) => {
          runs.push({ dims: feeds.input.dims, samples: feeds.input.data.length });
          const probability = probabilities[Math.min(runs.length - 1, probabilities.length - 1)];
          return {
            output: { data: Float32Array.from([probability]) },
            stateN: { data: new Float32Array(256), dims: [2, 1, 128] },
          };
        },
      }),
    },
  };
  return { ort, runs };
}

test("createSileroDetector rebuffers into 512-sample windows and validates its input", async () => {
  const silent = createOrtMock([0.5]);

  await assert.rejects(
    () => createSileroDetector({ modelPath: "silero.onnx", sampleRate: 8000, ort: silent.ort }),
    RangeError,
  );
  await assert.rejects(
    () => createSileroDetector({ modelPath: "silero\u0000.onnx", ort: silent.ort }),
    (error) => error instanceof TypeError || error instanceof RangeError,
  );

  const { ort, runs } = createOrtMock([0.2, 0.9]);
  const detect = await createSileroDetector({ modelPath: "silero.onnx", ort });

  assert.equal(await detect(new Float32Array(320)), 0);
  assert.equal(runs.length, 0);

  assert.equal(await detect(new Float32Array(192)), Math.fround(0.2));
  assert.equal(runs.length, 1);
  assert.deepEqual(runs[0].dims, [1, 512]);

  assert.equal(await detect(new Float32Array(1024)), Math.fround(0.9));
  assert.equal(runs.length, 3);
  assert.deepEqual(runs[2].dims, [1, 512]);

  detect.reset();
  assert.equal(await detect(new Float32Array(320)), 0);

  await assert.rejects(() => detect(new Float32Array([Number.NaN])), TypeError);
  await assert.rejects(() => detect(new Float32Array([Number.POSITIVE_INFINITY])), TypeError);
});
