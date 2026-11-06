# Models for realtime turn detection

`llama-server` can detect turns in `/v1/realtime` sessions (`turn_detection` of the OpenAI realtime API) with small helper models. Without them a level-based VAD is used, which handles steady noise but not clicks or background voices.

| Model | Flag | Used for | Convert |
| --- | --- | --- | --- |
| [Silero VAD](https://github.com/snakers4/silero-vad) (MIT) | `--vad-model` | speech start and end (`server_vad`, `semantic_vad`) | `convert_silero_vad.py` |

## Silero VAD

```bash
pip install silero-vad torch gguf
python convert_silero_vad.py silero-vad.gguf
llama-server -hf LiquidAI/LFM3-Audio-2B-GGUF:Q4_K_M --vad-model silero-vad.gguf
```

The converter takes the weights of the package's default model (`load_silero_vad()`). The `silero_vad_16k.safetensors` file of the package holds different weights. The GGUF is about 1.2 MB, a 32 ms frame takes about 0.07 ms on one CPU thread, and the probabilities match the PyTorch model to within 2e-6.

Silero runs on 16 kHz audio. For models with another input rate the level-based VAD is used.

Then enable turn detection in the session:

```json
{"type": "session.update", "session": {"audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 500}}}}}
```
