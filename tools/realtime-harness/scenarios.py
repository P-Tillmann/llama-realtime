"""Scripted realtime sessions. Each scenario is a list of steps:

  {"do": "send", "event": {...}}                       send a client event
  {"do": "audio", "file": "x.wav", ...}                 append a WAV (resampled to the session rate), 100 ms chunks
  {"do": "wait", "for": "response.done"}                wait for a server event type (or a list of types)
  {"do": "tool_output", "output": "..."}                answer the last function call

"phase" names the roadmap phase after which llama-server is expected to match OpenAI.
"""


def session(**kw):
    s = {"type": "realtime", "instructions": "You are a helpful assistant. Answer in one short sentence."}
    s.update(kw)
    return {"do": "send", "event": {"type": "session.update", "session": s}}


def no_vad(**kw):
    # explicit turn handling: OpenAI defaults to server_vad, which would answer on its own
    # no format here: the audio follows the rate the server announces (24 kHz is tested in session_echo_24k)
    return session(audio={"input": {"turn_detection": None}}, **kw)


WAIT_UPDATED = {"do": "wait", "for": "session.updated"}
WAIT_DONE = {"do": "wait", "for": "response.done"}
CREATE = {"do": "send", "event": {"type": "response.create"}}
COMMIT = {"do": "send", "event": {"type": "input_audio_buffer.commit"}}

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def user_text(text):
    return {"do": "send", "event": {"type": "conversation.item.create", "item": {
        "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}}}


SCENARIOS = [
    {
        "name": "text_in_text_out",
        "phase": "P0",
        "steps": [no_vad(output_modalities=["text"]), WAIT_UPDATED,
                  user_text("What is the capital of France?"), CREATE, WAIT_DONE],
    },
    {
        "name": "audio_in_text_out",
        "phase": "P0",
        "steps": [no_vad(output_modalities=["text"]), WAIT_UPDATED,
                  {"do": "audio", "file": "q_capital.wav"}, COMMIT, CREATE, WAIT_DONE],
    },
    {
        "name": "audio_in_audio_out",
        "phase": "P1",  # needs response.output_audio_transcript.*
        "steps": [no_vad(output_modalities=["audio"]), WAIT_UPDATED,
                  {"do": "audio", "file": "q_capital.wav"}, COMMIT, CREATE, WAIT_DONE],
    },
    {
        "name": "two_turns",
        "phase": "P0",
        "steps": [no_vad(output_modalities=["text"]), WAIT_UPDATED,
                  {"do": "audio", "file": "q_capital.wav"}, COMMIT, CREATE, WAIT_DONE,
                  {"do": "audio", "file": "q_population.wav"}, COMMIT, CREATE, WAIT_DONE],
    },
    {
        "name": "tool_call",
        "phase": "P0",
        "steps": [no_vad(output_modalities=["text"], tools=[WEATHER_TOOL], tool_choice="auto"), WAIT_UPDATED,
                  user_text("What's the weather in Berlin right now?"), CREATE, WAIT_DONE,
                  {"do": "tool_output", "output": "{\"temperature_c\": 12, \"condition\": \"light rain\"}"},
                  CREATE, WAIT_DONE],
    },
    {
        "name": "server_vad_turn",
        "phase": "P2",
        "steps": [session(output_modalities=["text"], audio={"input": {
                      "format": {"type": "audio/pcm", "rate": 24000},
                      "turn_detection": {"type": "server_vad", "silence_duration_ms": 500, "create_response": True}}}),
                  WAIT_UPDATED,
                  {"do": "audio", "file": "q_capital.wav", "silence_before_ms": 500, "silence_after_ms": 1500, "realtime": True},
                  {"do": "wait", "for": "response.done", "timeout": 30}],
    },
    {
        "name": "semantic_vad_pause",
        "phase": "P2",
        "steps": [session(output_modalities=["text"], audio={"input": {
                      "format": {"type": "audio/pcm", "rate": 24000},
                      "turn_detection": {"type": "semantic_vad", "eagerness": "medium", "create_response": True}}}),
                  WAIT_UPDATED,
                  {"do": "audio", "file": "um_label.wav", "silence_before_ms": 500, "silence_after_ms": 3000, "realtime": True},
                  {"do": "wait", "for": "response.done", "timeout": 30}],
    },
    {
        "name": "session_echo_24k",
        "phase": "P1",  # 24 kHz input must be accepted, session fields echoed
        "steps": [session(output_modalities=["text"], audio={"input": {
                      "format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": None,
                      "transcription": {"model": "gpt-4o-mini-transcribe", "language": "en"}},
                      "output": {"voice": "marin"}}, max_output_tokens=200),
                  WAIT_UPDATED],
    },
]
