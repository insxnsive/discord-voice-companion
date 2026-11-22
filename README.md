# Discord Voice Companion

Real-time Discord voice companion bot built on Node 22 with mandatory Discord DAVE end-to-end voice encryption. It aims for a 700 ms end-to-end turn budget from end of speech to first spoken audio byte and fills slower model turns after 165 ms with a short, barge-in-cancellable reaction clip. Speech synthesis defaults to ElevenLabs Flash v2.5 (`eleven_flash_v2_5`) over WebSocket and switches to HTTP streaming (`eleven_v4_turbo`) when `ELEVENLABS_EXPRESSIVE_TAGS=true` is enabled.

## Requirements

- Node.js `>= 22.12.0` for `@discordjs/voice@0.19.2` and `@snazzah/davey@0.2.0`
- Optional native `@discordjs/opus` build, with automatic fallback to `opusscript`
- Discord Gateway Intents: `Guilds` and `GuildVoiceStates`
- Discord Bot Permissions: `Connect` and `Speak`

## Quickstart

```bash
npm install
npm run setup
npm start
npm test
```

## Configuration and providers

Configuration loads with explicit environment variables overriding `.env`, followed by `config.json`, and then provider defaults. Persona instructions come from `SOUL.md`.

| Area | Options | Notes |
|---|---|---|
| LLM presets (`LLM_PROVIDER`) | `deepseek`, `opencode`, `openai`, `openrouter`, `groq`, `ollama`, `custom` | Powers both the streaming reply clause splitter and the 1-token turn gate |
| STT adapters (`STT_PROVIDER`) | `deepgram`, `elevenlabs`, `faster-whisper`, `openai`, `groq` | `faster-whisper` sends raw in-RAM PCM to `/v1/audio/pcm` with no disk writes |
| TTS adapters (`TTS_PROVIDER`) | `elevenlabs`, `openai`, `openai-compatible` | `elevenlabs` defaults to `eleven_flash_v2_5` WS or uses `eleven_v4_turbo` HTTP when expressive tags are on |
| Turn gate (`CHIME_MODE`) | `smart`, `always`, `mention` | Supports wakeword aliases via `CHIME_BOT_ALIASES` and a `12000` ms `followUpWindowMs` conversation window |
| Reaction fillers (`REACTION_FILLERS`) | `adaptive`, `subtle`, `off` | Fires after `165` ms (clamped to `150` to `180` ms) with a 10-turn no-repeat LRU window |

## Discord slash commands

- `/join` connects the companion to your active voice channel
- `/leave` disconnects from voice playback and tears down listeners
- `/mode` updates the live turn-taking chime or reaction mode
- `/status` shows active providers, loaded reaction clip count, and recent turn latency averages
- `/mute-user` stops transcribing and replying to a selected user in the channel

## Audio security and privacy

`npm run setup` writes `.env` at the project root with `0600` POSIX permissions and writes pregenerated reaction `.pcm` clips under `.cache/reactions` (`0700` directory and `0600` files on POSIX, or user profile ACLs on Windows). Remote HTTP endpoints require `HTTPS` and reject redirects, while remote WebSocket endpoints require `WSS` (plaintext `HTTP` and `WS` are allowed only on loopback hosts like `127.0.0.1` for local Ollama or `faster-whisper`). `ReactionPool.loadFromDirectory` resolves `trustedRootDir` with `realpathSync`, skips symlinks before opening file descriptors, matches device and inode numbers against `fstatSync`, and bounds loaded reaction memory to 8 MiB. Diagnostic output always passes through `redactConfig` so tokens and API keys stay masked.
