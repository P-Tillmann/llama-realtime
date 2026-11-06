# Models for realtime turn detection

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
