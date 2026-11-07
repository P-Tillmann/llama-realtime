# Models for realtime turn detection, speech recognition and voices

`llama-server` can detect turns in `/v1/realtime` sessions (`turn_detection` of the OpenAI realtime API) with small helper models. Without them a level-based VAD is used, which handles steady noise but not clicks or background voices.

| Model | Flag | Used for | Convert |
| --- | --- | --- | --- |
| [Silero VAD](https://github.com/snakers4/silero-vad) (MIT) | `--vad-model` | speech start and end (`server_vad`, `semantic_vad`) | `convert_silero_vad.py` |
| [Smart Turn v3](https://github.com/pipecat-ai/smart-turn) (BSD-2) | `--turn-model` | end of turn by content and prosody (`semantic_vad`) | `convert_smart_turn.py` |

## Silero VAD

```bash
pip install silero-vad torch gguf
python convert_silero_vad.py silero-vad.gguf
llama-server -hf LiquidAI/LFM3-Audio-2B-GGUF:Q4_K_M --vad-model silero-vad.gguf
```

The converter takes the weights of the package's default model (`load_silero_vad()`). The `silero_vad_16k.safetensors` file of the package holds different weights. The GGUF is about 1.2 MB, a 32 ms frame takes about 0.07 ms on one CPU thread, and the probabilities match the PyTorch model to within 2e-6.

Silero runs on 16 kHz audio. For models with another input rate the level-based VAD is used.

## Smart Turn v3

```bash
pip install onnx gguf numpy
curl -LO https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-gpu.onnx
python convert_smart_turn.py smart-turn-v3.2-gpu.onnx smart-turn-v3.2.gguf
llama-server ... --vad-model silero-vad.gguf --turn-model smart-turn-v3.2.gguf
```

The `-gpu` file is fp32, the `-cpu` file int8; both convert (the int8 weights are dequantized), `--type f16` or `--type q8_0` makes the GGUF smaller. With `--type f32` the probabilities match ONNX Runtime to four decimals; one check takes about 60 ms on 4 CPU threads.

With a turn model, `semantic_vad` checks the turn after 200 ms of silence and then every 200 ms. The turn ends when the probability reaches the threshold of the eagerness (low 0.7, medium/auto 0.5, high 0.3), or at the longest silence (low 8 s, medium/auto 4 s, high 2 s). Without a turn model it ends after a fixed silence (low 1.5 s, medium/auto 0.8 s, high 0.5 s).

Then enable turn detection in the session:

```json
{"type": "session.update", "session": {"audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 500}}}}}
{"type": "session.update", "session": {"audio": {"input": {"turn_detection": {"type": "semantic_vad", "eagerness": "auto"}}}}}
```

## Voices

A model that writes text but no audio (e.g. Gemma 4 E2B/E4B, which hears audio) can answer `/v1/realtime` sessions with speech: `--voice-model` loads text-to-speech models into the server process, and each response is spoken sentence by sentence while the language model still writes. Any audio generation model of libmtmd works as a voice; each one gets its own context and worker thread.

```bash
llama-server -hf ggml-org/gemma-4-E2B-it-GGUF \
    --vad-model silero-vad.gguf \
    --voice-model "model=pocket-de.gguf,mmproj=mmproj-pocket-de.gguf,name=de,lang=de;model=pocket-en.gguf,mmproj=mmproj-pocket-en.gguf,name=en"
```

`SPEC` is `model=FILE,mmproj=FILE[,name=NAME][,lang=LANG][,voice=VOICE][,speaker=FILE]`. Separate several with `;` or repeat the option.

| Key | Meaning |
| --- | --- |
| `name` | what sessions select the model by (default: the file name) |
| `lang` | language passed to models that take one (Qwen3-TTS, LFM3-Audio, Chatterbox Multilingual) |
| `voice` | default voice: a predefined voice of the model, or a voice of its tts template |
| `speaker` | a WAV to clone the voice from |

A session selects a voice with `session.audio.output.voice` set to `NAME`, `VOICE` or `NAME/VOICE`. Unknown names, e.g. OpenAI's `marin`, fall back to the first model and its default voice. With voices loaded, `output_modalities: ["audio"]` returns `response.output_audio.delta`, resampled from the voice to the session rate, together with `response.output_audio_transcript.delta` for each spoken piece as its audio starts. `response.cancel` and barge-in by VAD stop the voice; `conversation.item.truncate` cuts the transcript in the conversation where the audio was cut.

Tested with Gemma 4 E2B (Q4_0) on 4 CPU threads, measured from `speech_stopped` to the first audio:

| Voice | Languages | First audio | Note |
| --- | --- | --- | --- |
| Pocket TTS | de, en (one model per language) | 1.0–1.3 s (en), 1.6–1.8 s (de) | about 6× faster than real time |
| Soprano 80M | en | about 1.4 s | |
| LFM3-Audio 2B | 19 languages, `lang=` | 1.3–1.7 s | about real time |
| Chatterbox Turbo | en | about 9 s | the vocoder runs once per sentence, no streaming |

Qwen3-TTS and Chatterbox Multilingual load the same way but are untested here. Voices are loaded from local files; there is no `-hf` download for them yet.

## Speech recognition (cascade)

With `--asr-model`, a model without audio input, any text model, holds `/v1/realtime` voice sessions: a speech-to-text model in the server process transcribes each user audio item, and its text goes into the conversation. Together with `--voice-model` this is a full cascade: speech-to-text, the language model, text-to-speech, all in one process.

```bash
llama-server -m gemma-4-E2B-it-Q4_0.gguf --no-mmproj \
    --asr-model "model=LFM3-Audio-2B-Q4_K_M.gguf,mmproj=mmproj-LFM3-Audio-2B-F16.gguf,lang=de" \
    --vad-model silero-vad.gguf \
    --voice-model "model=pocket-de.gguf,mmproj=mmproj-pocket-de.gguf,name=de,lang=de"
```

`SPEC` is `model=FILE,mmproj=FILE[,lang=LANG]`; `lang` is the default language, `audio.input.transcription.language` of a session overrides it. Any audio model of libmtmd works:

- A model with an audio stream and an `asr` chat template (LFM3-Audio, Voxtral Realtime) transcribes while the user speaks. When the turn ends, only the last fraction of a second is left to transcribe.
- Other audio models (Qwen3-ASR, Voxtral, Gemma 4, Ultravox, ...) transcribe the audio of a turn when it ends, with the transcription prompt of their chat template.

The user audio item keeps its audio for `conversation.item.retrieve`, its `transcript` holds the recognized text. With `audio.input.transcription` set, the `conversation.item.input_audio_transcription.*` events carry the same text, and a `transcription` session is served by the ASR model. If the main model hears audio itself, the ASR model only serves these transcription events.

LFM3-Audio without a language transcribes into English; give it `lang=` for other languages.

Tested on 4 CPU threads, with server VAD, from `speech_stopped` to the first token of the response:

| Main model | ASR | Transcription done | First token | First audio (Pocket TTS) |
| --- | --- | --- | --- | --- |
| Gemma 4 E2B, text only | LFM3-Audio 2B (streaming) | 0.3 s | 0.6 s | 1.5–1.8 s (de) |
| Gemma 4 E2B, text only | Qwen3-ASR 0.6B (at the end) | 0.8–1.2 s | 1.1–1.3 s | 2.3–2.4 s (de) |
| Llama 3.2 1B | LFM3-Audio 2B (streaming) | 0.3 s | 0.6 s | |

A text model that reasons before it answers adds that time to every turn; the LFM3-Audio backbone does when it runs as a text model.
