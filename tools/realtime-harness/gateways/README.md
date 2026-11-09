# Gateways in front of llama-server

llama.cpp does not recommend exposing `llama-server` to untrusted networks (see `SECURITY.md`). These are tested ways to keep it on localhost and put a component built for the internet in front of its `/v1/realtime` WebSocket.

| Gateway | License | What it adds | Result |
| --- | --- | --- | --- |
| [Pipecat](https://github.com/pipecat-ai/pipecat) SmallWebRTC + `OpenAIRealtimeLLMService` | BSD-2 | WebRTC for browsers (aiortc), Opus | Browser session works: greeting, two spoken questions, barge-in with truncate |
| [LiteLLM](https://github.com/BerriAI/litellm) proxy | MIT (core) | API keys, budgets, rate limits for the WebSocket | All harness scenarios pass, no measurable latency; its WebRTC routes only forward to an upstream that speaks WebRTC itself |

## Pipecat: WebRTC for browsers

```bash
pip install "pipecat-ai[webrtc,openai,runner]"
llama-server -m gemma-4-E2B-it-Q4_0.gguf --mmproj mmproj-gemma-4-E2B-it-Q8_0.gguf --host 127.0.0.1 --port 8080 \
    --vad-model silero-vad.gguf --voice-model "model=pocket-en.gguf,mmproj=mmproj-pocket-en.gguf"
LLAMA_REALTIME_URL=ws://127.0.0.1:8080/v1/realtime python pipecat_bot.py -t webrtc --host 127.0.0.1 --port 7860
```

The browser connects to the bot at `http://127.0.0.1:7860/client/` (Pipecat's prebuilt client) or with its own `RTCPeerConnection`, posting the offer to `/api/offer`. `webrtc_browser.js` does the latter in headless Chromium with a WAV file as the microphone and records what the bot says:

```bash
node webrtc_browser.js mic.wav reply.webm
```

`TURN=server` selects `server_vad` instead of `semantic_vad`, `TRANSCRIBE=0` turns off the input transcription, `GREET=0` the greeting.

With Gemma 4 E2B and Pocket TTS on 4 CPU threads, the connection is up in 0.4-0.6 s and Pipecat measures 1.1-1.6 s to the first audio of an answer. With input transcription on, it runs on a second slot of the same model at the same time and the answer takes about twice as long on CPU.

## LiteLLM: keys for the WebSocket

```bash
pip install "litellm[proxy]"
litellm --config litellm.yaml --host 127.0.0.1 --port 4000
LOCAL_API_KEY=sk-spike-master python ../harness.py record --target local \
    --url "ws://127.0.0.1:4000/v1/realtime?model=local-realtime" --out out/litellm
```

Connections without a valid key get HTTP 403. Use virtual keys (`/key/generate`) rather than the master key for clients.
