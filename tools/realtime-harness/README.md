# realtime-harness

Fork-only conformance harness for `/v1/realtime`. It plays scripted sessions against the OpenAI Realtime API and against `llama-server`, then compares the two event streams by structure: event order, field paths and latency. Text, audio and ids are expected to differ and are ignored.

| File | Purpose |
| --- | --- |
| `scenarios.py` | Scripted sessions, each tagged with the roadmap phase after which llama-server should match |
| `harness.py` | `record` runs scenarios against `openai` or `local`; `compare` writes a Markdown report |
| `compare.py` | Structural comparison |
| `gen_audio.py` | Regenerates `audio/*.wav` with Mistral TTS (`MISTRAL_API_KEY`) |
| `fixtures/openai/` | Recorded OpenAI sessions, written by the `Realtime - record OpenAI fixtures` workflow |

## Run locally

```bash
pip install -r requirements.txt
llama-server -hf LiquidAI/LFM3-Audio-2B-GGUF:Q4_K_M --temp 0 --port 8080
python harness.py record --target local --out out/local
python harness.py compare fixtures/openai out/local --report out/report.md
```

Recording against OpenAI needs `OPENAI_API_KEY` and a direct WebSocket connection:

```bash
python harness.py record --target openai --model gpt-realtime --out fixtures/openai
```

## CI

- `Realtime - record OpenAI fixtures` (manual): records the reference sessions and commits them. Needs the repository secret `OPEN_AI_KEY`.
- `Realtime - conformance` (on push to `realtime`, `rt/**`, `up/**`): builds `llama-server`, runs all scenarios with LFM3-Audio-2B and puts the comparison in the job summary.

Recorded logs keep every server event, except that base64 audio is replaced by its byte count.
