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

test("PerSpeakerVad evicts the oldest idle speaker and admits the newcomer", async () => {
  const bounded = makeVad({ maxSpeakers: 3 }).vad;
  for (let i = 0; i < 5; i += 1) {
    await bounded.pushFrame({ userId: `user-${i}`, pcm16kMono: monoFrame(0), timestampMs: 1000 });
  }

  assert.equal(bounded.speakers.size, 3);
  assert.equal(bounded.speakers.has("user-4"), true);
  assert.equal(bounded.speakers.has("user-0"), false);
  assert.equal(bounded.speakers.has("user-1"), false);
  assert.equal(bounded.speakers.has("user-2"), true);
  assert.equal(bounded.speakers.has("user-3"), true);

  const transient = makeVad({ maxSpeakers: 4, idleSpeakerTtlMs: 100 }).vad;
  for (let i = 0; i < 100; i += 1) {
    await transient.pushFrame({ userId: `transient-${i}`, pcm16kMono: monoFrame(0), timestampMs: i * 1000 });
  }
  assert.ok(transient.speakers.size <= 4);
});

test("PerSpeakerVad expires an idle speaker at the TTL before evicting", async () => {
  const vad = makeVad({ maxSpeakers: 2, idleSpeakerTtlMs: 100 }).vad;

  await vad.pushFrame({ userId: "a", pcm16kMono: monoFrame(0), timestampMs: 0 });
  await vad.pushFrame({ userId: "b", pcm16kMono: monoFrame(0), timestampMs: 60 });
  assert.equal(vad.speakers.has("a"), true);

  await vad.pushFrame({ userId: "c", pcm16kMono: monoFrame(0), timestampMs: 150 });
  assert.equal(vad.speakers.has("a"), false);
  assert.equal(vad.speakers.has("b"), true);
  assert.equal(vad.speakers.has("c"), true);
  assert.equal(vad.speakers.size, 2);
});

test("PerSpeakerVad refuses newcomers when every tracked speaker is active", async () => {
  const vad = new PerSpeakerVad({
    maxSpeakers: 2,
    detectorFactory: () => () => 0.9,
  });

  await vad.pushFrame({ userId: "active-1", pcm16kMono: monoFrame(9000), timestampMs: 0 });
  await vad.pushFrame({ userId: "active-2", pcm16kMono: monoFrame(9000), timestampMs: 0 });
  assert.equal(vad.speakers.get("active-1").active, true);
  assert.equal(vad.speakers.get("active-2").active, true);

  const result = await vad.pushFrame({ userId: "newcomer", pcm16kMono: monoFrame(0), timestampMs: 10 });
  assert.deepEqual(result, { ignored: true, dropped: "max_speakers" });
  assert.equal(vad.speakers.has("newcomer"), false);
  assert.equal(vad.speakers.has("active-1"), true);
  assert.equal(vad.speakers.has("active-2"), true);
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

test("PerSpeakerVad expires a stalled active speaker past the TTL", async () => {
  const vad = new PerSpeakerVad({
    maxSpeakers: 2,
    idleSpeakerTtlMs: 100,
    detectorFactory: () => () => 0.9,
  });

  await vad.pushFrame({ userId: "stalled", pcm16kMono: monoFrame(9000), timestampMs: 0 });
  assert.equal(vad.speakers.get("stalled").active, true);
  assert.equal(vad.speakers.get("stalled").pendingCount, 0);

  await vad.pushFrame({ userId: "fresh", pcm16kMono: monoFrame(0), timestampMs: 150 });
  assert.equal(vad.speakers.has("stalled"), false);
  assert.equal(vad.speakers.has("fresh"), true);
});

test("PerSpeakerVad cancels queued work when the speaker is dropped first", async () => {
  let factoryCalls = 0;
  let detectorCalls = 0;
  const vad = new PerSpeakerVad({
    detectorFactory: () => {
      factoryCalls += 1;
      return () => {
        detectorCalls += 1;
        return 0.9;
      };
    },
  });

  const first = vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000), timestampMs: 0 });
  const second = vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000), timestampMs: 20 });
  vad.resetSpeaker("u1");
  await Promise.all([first, second]);

  assert.equal(factoryCalls, 0);
  assert.equal(detectorCalls, 0);
  assert.equal(vad.speakers.has("u1"), false);

  const clearedFactoryCalls = { count: 0 };
  const cleared = new PerSpeakerVad({
    detectorFactory: () => {
      clearedFactoryCalls.count += 1;
      return () => 0.9;
    },
  });
  const pending = cleared.pushFrame({ userId: "u2", pcm16kMono: monoFrame(9000), timestampMs: 0 });
  cleared.clear();
  await pending;

  assert.equal(clearedFactoryCalls.count, 0);
  assert.equal(cleared.speakers.size, 0);
});

test("PerSpeakerVad skips busy speakers when evicting", async () => {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const vad = new PerSpeakerVad({
    maxSpeakers: 2,
    detectorFactory: (userId) => async () => {
      if (userId === "newcomer") return 0.9;
      await gate;
      return 0.9;
    },
  });

  const busyA = vad.pushFrame({ userId: "busy-a", pcm16kMono: monoFrame(9000), timestampMs: 0 });
  const busyB = vad.pushFrame({ userId: "busy-b", pcm16kMono: monoFrame(9000), timestampMs: 0 });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(vad.speakers.get("busy-a").pendingCount, 1);
  assert.equal(vad.speakers.get("busy-b").pendingCount, 1);

  const newcomer = await vad.pushFrame({ userId: "newcomer", pcm16kMono: monoFrame(0), timestampMs: 10 });

  release();
  await Promise.allSettled([busyA, busyB]);

  assert.deepEqual(newcomer, { ignored: true, dropped: "max_speakers" });
  assert.equal(vad.speakers.has("newcomer"), false);
  assert.equal(vad.speakers.has("busy-a"), true);
  assert.equal(vad.speakers.has("busy-b"), true);
});

test("PerSpeakerVad caps in-flight detections across all speakers", async () => {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const vad = new PerSpeakerVad({
    maxSpeakers: 100,
    maxPendingFrames: 8,
    detectorFactory: () => async () => {
      await gate;
      return 0.9;
    },
  });

  const pushes = [];
  for (let i = 0; i < 70; i += 1) {
    pushes.push(vad.pushFrame({ userId: `bulk-${i}`, pcm16kMono: monoFrame(9000), timestampMs: 0 }));
  }

  await new Promise((resolve) => setImmediate(resolve));
  const inFlightDuringBurst = vad.inFlightCount;

  release();
  const results = await Promise.all(pushes);

  assert.ok(inFlightDuringBurst <= 64);
  const accepted = results.filter((result) => result?.ignored === false);
  const dropped = results.filter((result) => result?.dropped === "global_backpressure");
  assert.equal(accepted.length, 64);
  assert.equal(dropped.length, 6);
  assert.equal(vad.inFlightCount, 0);
});

test("createFrameAccumulator rejects invalid frame and chunk limits", () => {
  for (const invalidFrame of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 2, MAX_AUDIO_CHUNK_BYTES + 2]) {
    assert.throws(() => createFrameAccumulator(invalidFrame), RangeError);
  }

  for (const invalidChunk of [0, -1, 1.5, MAX_AUDIO_CHUNK_BYTES + 2]) {
    assert.throws(
      () => createFrameAccumulator(DISCORD_FRAME_BYTES, { maxChunkBytes: invalidChunk }),
      RangeError,
    );
  }

  assert.throws(
    () => createFrameAccumulator(4096, { maxChunkBytes: 2048 }),
    RangeError,
  );
  assert.doesNotThrow(() => createFrameAccumulator(2048, { maxChunkBytes: 2048 }));
});

test("resetSpeaker inside onSpeechStart suppresses the utterance", async () => {
  const ends = [];
  let vad;
  vad = new PerSpeakerVad({
    detectorFactory: () => () => 0.9,
    minSpeechMs: 10,
    maxUtteranceBytes: 640,
    onSpeechStart: () => vad.resetSpeaker("u1"),
    onSpeechEnd: (event) => ends.push(event),
  });

  await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000), timestampMs: 20 });

  assert.equal(ends.length, 0);
  assert.equal(vad.speakers.has("u1"), false);
});

test("resetSpeaker inside onBargeIn suppresses the stale speech end", async () => {
  const ends = [];
  const barges = [];
  let vad;
  vad = new PerSpeakerVad({
    detectorFactory: () => () => 0.9,
    minSpeechMs: 40,
    maxUtteranceBytes: 9 * VAD_FRAME_BYTES,
    isBotSpeaking: () => true,
    onBargeIn: (event) => {
      barges.push(event);
      vad.resetSpeaker("u2");
    },
    onSpeechEnd: (event) => ends.push(event),
  });

  for (let i = 0; i < 9; i += 1) {
    await vad.pushFrame({
      userId: "u2",
      displayName: "Ben",
      pcm16kMono: monoFrame(9000),
      timestampMs: i * 20,
    });
  }

  assert.equal(barges.length, 1);
  assert.equal(ends.length, 0);
});

test("a slow onSpeechEnd does not hold VAD in-flight slots", async () => {
  const vad = new PerSpeakerVad({
    detectorFactory: () => () => 0.9,
    minSpeechMs: 40,
    maxUtteranceBytes: 3 * VAD_FRAME_BYTES,
    hangoverMs: 240,
    maxInFlightFrames: 1,
    onSpeechEnd: () => new Promise(() => {}),
  });

  await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000), timestampMs: 20 });
  await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000), timestampMs: 40 });

  const hung = vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(0), timestampMs: 60 });
  await new Promise((resolve) => setImmediate(resolve));

  const next = await vad.pushFrame({
    userId: "u1",
    pcm16kMono: monoFrame(9000),
    timestampMs: 80,
  });

  assert.notEqual(next.dropped, "global_backpressure");
  assert.equal(vad.inFlightCount, 0);
  void hung;
});

test("maxInFlightFrames must be an integer between 1 and 256", () => {
  for (const invalid of [0, -1, 1.5, 257]) {
    assert.throws(() => new PerSpeakerVad({ maxInFlightFrames: invalid }), RangeError);
  }
  assert.doesNotThrow(() => new PerSpeakerVad({ maxInFlightFrames: 1 }));
  assert.doesNotThrow(() => new PerSpeakerVad({ maxInFlightFrames: 256 }));

  for (const invalid of [0, -1, 1.5, 30001]) {
    assert.throws(() => new PerSpeakerVad({ detectorTimeoutMs: invalid }), RangeError);
  }
  assert.doesNotThrow(() => new PerSpeakerVad({ detectorTimeoutMs: 5000 }));
});

test("resetSpeaker inside isBotSpeaking suppresses barge-in dispatch", async () => {
  const barges = [];
  const ends = [];
  let vad;
  vad = new PerSpeakerVad({
    detectorFactory: () => () => 0.9,
    minSpeechMs: 40,
    maxUtteranceBytes: 9 * VAD_FRAME_BYTES,
    isBotSpeaking: () => {
      vad.resetSpeaker("u3");
      return true;
    },
    onBargeIn: (event) => barges.push(event),
    onSpeechEnd: (event) => ends.push(event),
  });

  for (let i = 0; i < 9; i += 1) {
    await vad.pushFrame({ userId: "u3", pcm16kMono: monoFrame(9000), timestampMs: i * 20 });
  }

  assert.equal(barges.length, 0);
  assert.equal(ends.length, 0);
});

test("a stalled detector times out and frees the in-flight slot", { timeout: 3000 }, async () => {
  const vad = new PerSpeakerVad({
    detectorFactory: () => () => new Promise(() => {}),
    detectorTimeoutMs: 25,
    maxInFlightFrames: 1,
  });

  await assert.rejects(
    () => vad.pushFrame({ userId: "u4", pcm16kMono: monoFrame(9000), timestampMs: 0 }),
    /timed out/i,
  );

  assert.equal(vad.inFlightCount, 0);
  assert.equal(vad.speakers.get("u4").pendingCount, 0);
});

test("a reset queued on the final frame suppresses onSpeechEnd", async () => {
  const ends = [];
  let vad;
  let resetQueued = false;
  vad = new PerSpeakerVad({
    detectorFactory: () => () => {
      if (!resetQueued) {
        resetQueued = true;
        queueMicrotask(() => vad.resetSpeaker("u6"));
      }
      return 0.9;
    },
    minSpeechMs: 10,
    maxUtteranceBytes: VAD_FRAME_BYTES,
    onSpeechEnd: (event) => ends.push(event),
  });

  await vad.pushFrame({ userId: "u6", pcm16kMono: monoFrame(9000), timestampMs: 20 });
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(ends.length, 0);
});

test("a timed-out detector is quarantined until it settles", async () => {
  let calls = 0;
  let rejectHung;
  const hung = new Promise((resolve, reject) => {
    rejectHung = reject;
  });

  const vad = new PerSpeakerVad({
    detectorFactory: () => {
      calls += 1;
      if (calls === 1) return () => hung;
      return () => 0.9;
    },
    detectorTimeoutMs: 20,
  });

  await assert.rejects(
    () => vad.pushFrame({ userId: "u7", pcm16kMono: monoFrame(9000), timestampMs: 0 }),
    /timed out/i,
  );
  assert.equal(calls, 1);

  const quarantined = await vad.pushFrame({
    userId: "u7",
    pcm16kMono: monoFrame(9000),
    timestampMs: 20,
  });
  assert.equal(quarantined.dropped, "detector_quarantined");
  assert.equal(calls, 1);

  vad.resetSpeaker("u7");
  const afterReset = await vad.pushFrame({
    userId: "u7",
    pcm16kMono: monoFrame(9000),
    timestampMs: 30,
  });
  assert.equal(afterReset.dropped, "detector_quarantined");
  assert.equal(calls, 1);

  rejectHung(new Error("late detector rejection"));
  await new Promise((resolve) => setImmediate(resolve));

  const recovered = await vad.pushFrame({
    userId: "u7",
    pcm16kMono: monoFrame(9000),
    timestampMs: 40,
  });
  assert.notEqual(recovered.dropped, "detector_quarantined");
  assert.equal(calls, 2);
});

test("a detector reset that drops the speaker suppresses onSpeechEnd", async () => {
  const ends = [];
  let vad;
  vad = new PerSpeakerVad({
    detectorFactory: () => {
      const detect = () => 0.9;
      detect.reset = () => vad.resetSpeaker("u8");
      return detect;
    },
    minSpeechMs: 10,
    maxUtteranceBytes: VAD_FRAME_BYTES,
    onSpeechEnd: (event) => ends.push(event),
  });

  await vad.pushFrame({ userId: "u8", pcm16kMono: monoFrame(9000), timestampMs: 20 });
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(ends.length, 0);
});

test("resetSpeaker while a detector is unsettled quarantines the speaker", async () => {
  let calls = 0;
  let resolveDetector;
  const pendingDetector = new Promise((resolve) => {
    resolveDetector = resolve;
  });

  const vad = new PerSpeakerVad({
    detectorFactory: () => {
      calls += 1;
      return () => pendingDetector;
    },
    detectorTimeoutMs: 5000,
    maxInFlightFrames: 1,
  });

  const pendingFrame = vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000), timestampMs: 0 });
  await new Promise((resolve) => setImmediate(resolve));
  vad.resetSpeaker("u1");

  const blocked = await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000), timestampMs: 20 });
  assert.deepEqual(blocked, { ignored: true, dropped: "detector_quarantined" });
  assert.equal(calls, 1);

  const other = await vad.pushFrame({ userId: "u2", pcm16kMono: monoFrame(9000), timestampMs: 20 });
  assert.equal(other.dropped, "global_backpressure");

  resolveDetector(0.9);
  await pendingFrame.catch(() => {});
  await new Promise((resolve) => setImmediate(resolve));

  const recovered = await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000), timestampMs: 40 });
  assert.notEqual(recovered.dropped, "detector_quarantined");
  assert.equal(calls, 2);
});

test("opusscript encode receives the 960 sample count and createDecoder is independent", async () => {
  const calls = [];
  class MockOpusScript {
    static Application = { AUDIO: 2049 };

    constructor(rate, channels, application) {
      this.args = [rate, channels, application];
    }

    encode(frame, samples) {
      calls.push(samples);
      return Buffer.from([1]);
    }

    decode() {
      return Buffer.alloc(2);
    }
  }

  const transcoder = await createOpusTranscoder({
    loadNative: async () => {
      throw new Error("no native");
    },
    loadJs: async () => ({ default: MockOpusScript }),
  });

  assert.equal(transcoder.backend, "opusscript");
  transcoder.encode(Buffer.alloc(DISCORD_FRAME_BYTES));
  assert.deepEqual(calls, [960]);

  const decoder = transcoder.createDecoder();
  assert.equal(typeof decoder.decode, "function");
  assert.equal(decoder.decode(Buffer.from([1])).length, 2);
});

test("flushSpeaker finalizes the active utterance", async () => {
  const ends = [];
  const vad = new PerSpeakerVad({
    detectorFactory: () => () => 0.9,
    minSpeechMs: 20,
    onSpeechEnd: (event) => ends.push(event),
  });

  await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000), timestampMs: 20 });
  const result = await vad.flushSpeaker("u1", 100);

  assert.equal(result.flushed, true);
  assert.equal(ends.length, 1);
  assert.equal(ends[0].durationMs, 20);
  assert.equal(ends[0].endedAt, 100);
});

test("VAD frame duration follows the actual sample count", async () => {
  const ends = [];
  const vad = new PerSpeakerVad({
    detectorFactory: () => (samples) => (computeRms(samples) > 0 ? 0.9 : 0),
    minSpeechMs: 10,
    hangoverMs: 40,
    frameMs: 20,
    onSpeechEnd: (event) => ends.push(event),
  });

  await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000, 160), timestampMs: 10 });
  await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000, 160), timestampMs: 20 });
  for (let i = 0; i < 4; i += 1) {
    await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(0, 160), timestampMs: 30 + i * 10 });
  }

  assert.equal(ends.length, 1);
  assert.equal(ends[0].durationMs, 20);
});

test("maxSpeakers counts quarantined identities and clear preserves them", async () => {
  let calls = 0;
  let rejectHung;
  const hung = new Promise((resolve, reject) => {
    rejectHung = reject;
  });

  const vad = new PerSpeakerVad({
    detectorFactory: () => {
      calls += 1;
      return calls === 1 ? () => hung : () => 0.9;
    },
    detectorTimeoutMs: 20,
    maxSpeakers: 1,
    maxInFlightFrames: 4,
  });

  await assert.rejects(
    () => vad.pushFrame({ userId: "a", pcm16kMono: monoFrame(9000), timestampMs: 0 }),
    /timed out/i,
  );
  assert.equal(vad.quarantinedUsers.has("a"), true);

  const blocked = await vad.pushFrame({ userId: "b", pcm16kMono: monoFrame(9000), timestampMs: 10 });
  assert.equal(blocked.dropped, "max_speakers");

  vad.clear();
  assert.equal(vad.quarantinedUsers.has("a"), true);

  rejectHung(new Error("late"));
  await new Promise((resolve) => setImmediate(resolve));
});

test("createDecoder close deletes the real backend decoder exactly once", async () => {
  let deletes = 0;

  class MockOpusScript {
    static Application = { AUDIO: 2049 };

    encode() {
      return Buffer.from([1]);
    }

    decode() {
      return Buffer.alloc(2);
    }

    delete() {
      deletes += 1;
    }
  }

  const transcoder = await createOpusTranscoder({
    loadNative: async () => {
      throw new Error("no native");
    },
    loadJs: async () => ({ default: MockOpusScript }),
  });

  const first = transcoder.createDecoder();
  const second = transcoder.createDecoder();
  assert.notEqual(first, second);
  assert.equal(first.decode(Buffer.from([1])).length, 2);
  assert.equal(second.decode(Buffer.from([1])).length, 2);

  first.close();
  first.close();
  assert.equal(deletes, 1);
  second.close();
  assert.equal(deletes, 2);
});

test("stateful decoders through the real transcoder do not share state", async () => {
  class StatefulBackend {
    static Application = { AUDIO: 2049 };

    constructor() {
      this.count = 0;
    }

    encode() {
      return Buffer.from([1]);
    }

    decode() {
      this.count += 1;
      return Buffer.alloc(2, this.count);
    }

    delete() {}
  }

  const transcoder = await createOpusTranscoder({
    loadNative: async () => {
      throw new Error("no native");
    },
    loadJs: async () => ({ default: StatefulBackend }),
  });

  const first = transcoder.createDecoder();
  const second = transcoder.createDecoder();

  assert.equal(first.decode(Buffer.from([1]))[0], 1);
  assert.equal(first.decode(Buffer.from([1]))[0], 2);
  assert.equal(second.decode(Buffer.from([1]))[0], 1);
  assert.equal(first.decode(Buffer.from([1]))[0], 3);
});

test("60 ms silence frames count as 60 ms toward the hangover", async () => {
  const ends = [];
  const vad = new PerSpeakerVad({
    detectorFactory: () => (samples) => (computeRms(samples) > 0 ? 0.9 : 0),
    minSpeechMs: 60,
    hangoverMs: 120,
    frameMs: 20,
    onSpeechEnd: (event) => ends.push(event),
  });

  await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(9000, 960), timestampMs: 60 });
  await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(0, 960), timestampMs: 120 });
  await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(0, 960), timestampMs: 180 });

  assert.equal(ends.length, 1);
  assert.equal(ends[0].durationMs, 60);
  assert.equal(ends[0].endedAt, 180);
});

test("union cap counts quarantined users and never evicts them", async () => {
  let calls = 0;
  let rejectHung;
  const hung = new Promise((resolve, reject) => {
    rejectHung = reject;
  });
  const vad = new PerSpeakerVad({
    detectorFactory: () => {
      calls += 1;
      return calls === 1 ? () => hung : () => 0.9;
    },
    detectorTimeoutMs: 20,
    maxSpeakers: 2,
    maxInFlightFrames: 8,
  });

  await assert.rejects(
    () => vad.pushFrame({ userId: "q", pcm16kMono: monoFrame(9000), timestampMs: 0 }),
    /timed out/i,
  );
  assert.equal(vad.quarantinedUsers.has("q"), true);

  await vad.pushFrame({ userId: "a", pcm16kMono: monoFrame(9000), timestampMs: 10 });
  assert.equal(vad.speakers.get("a").active, true);

  const blocked = await vad.pushFrame({
    userId: "b",
    pcm16kMono: monoFrame(9000),
    timestampMs: 20,
  });
  assert.equal(blocked.dropped, "max_speakers");
  assert.equal(vad.quarantinedUsers.has("q"), true);
  assert.equal(vad.speakers.has("a"), true);
  assert.equal(vad.speakers.has("b"), false);

  rejectHung(new Error("late"));
});

test("union size counts quarantined users that left the speaker map", async () => {
  let calls = 0;
  let rejectHung;
  const hung = new Promise((resolve, reject) => {
    rejectHung = reject;
  });
  const vad = new PerSpeakerVad({
    detectorFactory: () => {
      calls += 1;
      return calls === 1 ? () => hung : () => 0.9;
    },
    detectorTimeoutMs: 20,
    maxSpeakers: 2,
    maxInFlightFrames: 8,
  });

  await assert.rejects(
    () => vad.pushFrame({ userId: "q", pcm16kMono: monoFrame(9000), timestampMs: 0 }),
    /timed out/i,
  );
  vad.resetSpeaker("q");
  assert.equal(vad.speakers.has("q"), false);
  assert.equal(vad.quarantinedUsers.has("q"), true);

  await vad.pushFrame({ userId: "a", pcm16kMono: monoFrame(9000), timestampMs: 10 });
  const blocked = await vad.pushFrame({
    userId: "b",
    pcm16kMono: monoFrame(9000),
    timestampMs: 20,
  });
  assert.equal(blocked.dropped, "max_speakers");

  rejectHung(new Error("late"));
});

test("eviction picks the LRU live speaker and never the quarantined one", async () => {
  let factoryCalls = 0;
  let rejectHung;
  const hung = new Promise((resolve, reject) => {
    rejectHung = reject;
  });
  const vad = new PerSpeakerVad({
    detectorFactory: () => {
      factoryCalls += 1;
      if (factoryCalls === 1) return () => hung;
      return (samples) => (computeRms(samples) > 0 ? 0.9 : 0);
    },
    detectorTimeoutMs: 20,
    maxSpeakers: 3,
    maxInFlightFrames: 8,
  });

  await assert.rejects(
    () => vad.pushFrame({ userId: "q", pcm16kMono: monoFrame(9000), timestampMs: 0 }),
    /timed out/i,
  );
  assert.equal(vad.quarantinedUsers.has("q"), true);

  await vad.pushFrame({ userId: "a", pcm16kMono: monoFrame(0), timestampMs: 10 });
  await vad.pushFrame({ userId: "b", pcm16kMono: monoFrame(0), timestampMs: 20 });
  assert.equal(vad.speakers.size, 3);

  await vad.pushFrame({ userId: "c", pcm16kMono: monoFrame(0), timestampMs: 30 });

  assert.equal(vad.speakers.has("a"), false);
  assert.equal(vad.speakers.has("b"), true);
  assert.equal(vad.speakers.has("c"), true);
  assert.equal(vad.quarantinedUsers.has("q"), true);

  const qFrame = await vad.pushFrame({
    userId: "q",
    pcm16kMono: monoFrame(9000),
    timestampMs: 40,
  });
  assert.equal(qFrame.dropped, "detector_quarantined");

  rejectHung(new Error("late"));
});

test("decode after close throws instead of reusing a deleted decoder", async () => {
  class DeleteCountingBackend {
    static Application = { AUDIO: 2049 };

    constructor() {
      this.deleted = false;
    }

    encode() {
      return Buffer.from([1]);
    }

    decode() {
      return Buffer.alloc(2);
    }

    delete() {
      this.deleted = true;
    }
  }

  const transcoder = await createOpusTranscoder({
    loadNative: async () => {
      throw new Error("no native");
    },
    loadJs: async () => ({ default: DeleteCountingBackend }),
  });

  const decoder = transcoder.createDecoder();
  assert.equal(decoder.decode(Buffer.from([1])).length, 2);
  decoder.close();
  assert.throws(() => decoder.decode(Buffer.from([1])), /closed/i);
});

test("the post-eviction union recheck rejects a re-entrant admission", async () => {
  let vad;
  const reentry = { fired: false };
  vad = new PerSpeakerVad({
    detectorFactory: () => {
      const detect = () => 0.9;
      detect.reset = () => {
        if (!reentry.fired) {
          reentry.fired = true;
          vad.pushFrame({ userId: "v", pcm16kMono: monoFrame(0), timestampMs: 5 });
        }
      };
      return detect;
    },
    minSpeechMs: 40,
    maxSpeakers: 2,
    maxInFlightFrames: 8,
  });

  await vad.pushFrame({ userId: "a", pcm16kMono: monoFrame(0), timestampMs: 10 });
  await vad.pushFrame({ userId: "b", pcm16kMono: monoFrame(0), timestampMs: 20 });

  const c = await vad.pushFrame({ userId: "c", pcm16kMono: monoFrame(0), timestampMs: 30 });

  assert.equal(c.dropped, "max_speakers");
  assert.ok(vad.speakers.size <= 2);
  assert.equal(vad.speakers.has("c"), false);
});

test("eviction reset that re-enters admission is caught by the recheck", async () => {
  let vad;
  let resetCalls = 0;
  let reentrantPush = null;
  const frame = Buffer.alloc(640, 1);

  vad = new PerSpeakerVad({
    detectorFactory: () => {
      const detect = async () => 0.1;
      detect.reset = () => {
        resetCalls += 1;
        reentrantPush = vad?.pushFrame({
          userId: "v",
          displayName: "V",
          pcm16kMono: frame,
          timestampMs: 300,
        });
      };
      return detect;
    },
    maxSpeakers: 2,
    idleSpeakerTtlMs: 60000,
    maxInFlightFrames: 8,
  });

  await vad.pushFrame({ userId: "a", pcm16kMono: frame, timestampMs: 100 });
  await vad.pushFrame({ userId: "b", pcm16kMono: frame, timestampMs: 200 });
  const c = await vad.pushFrame({ userId: "c", pcm16kMono: frame, timestampMs: 300 });
  if (reentrantPush) await reentrantPush;

  assert.equal(resetCalls, 1);
  assert.deepEqual(c, { ignored: true, dropped: "max_speakers" });
  assert.deepEqual([...vad.speakers.keys()].sort(), ["b", "v"]);
  assert.equal(vad.speakers.size, 2);
});

test("the VAD default timestamp uses the monotonic clock, not Date.now", async () => {
  const realDateNow = Date.now;
  Date.now = () => 1_000_000_000_000;
  try {
    const vad = new PerSpeakerVad({
      detectorFactory: () => () => 0.1,
      maxSpeakers: 2,
    });
    await vad.pushFrame({ userId: "u1", pcm16kMono: monoFrame(0) });
    const state = vad.speakers.get("u1");
    assert.ok(state.lastSeenAt < 1_000_000_000_000);
  } finally {
    Date.now = realDateNow;
  }
});

test("flushSpeaker finalizes after frames queued behind the speaker chain", async () => {
  const ends = [];
  let releaseGate;
  const gate = new Promise((resolve) => {
    releaseGate = resolve;
  });
  let detections = 0;

  const vad = new PerSpeakerVad({
    detectorFactory: () => async () => {
      detections += 1;
      if (detections === 1) await gate;
      return 0.9;
    },
    minSpeechMs: 20,
    onSpeechEnd: (event) => ends.push(event),
  });

  const frame = monoFrame(9000);
  const first = vad.pushFrame({ userId: "u1", pcm16kMono: frame, timestampMs: 20 });
  await new Promise((resolve) => setImmediate(resolve));
  const flush = vad.flushSpeaker("u1", 100);
  releaseGate();
  await first;
  const result = await flush;

  assert.equal(result.flushed, true);
  assert.equal(ends.length, 1);
  assert.equal(ends[0].pcm16kMono.length, VAD_FRAME_BYTES);
  assert.equal(ends[0].durationMs, 20);
});
