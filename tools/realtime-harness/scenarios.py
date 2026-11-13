"""Scripted realtime sessions. Each scenario is a list of steps:

  {"do": "send", "event": {...}}                       send a client event ("$item[i]": id of the i-th conversation item)
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
    {
        "name": "item_delete",
        "phase": "P3",
        "steps": [no_vad(output_modalities=["text"]), WAIT_UPDATED,
                  user_text("My name is Peter."), CREATE, WAIT_DONE,
                  user_text("Actually, my name is Anna."), CREATE, WAIT_DONE,
                  {"do": "send", "event": {"type": "conversation.item.delete", "item_id": "$item[2]"}},
                  {"do": "wait", "for": "conversation.item.deleted"},
                  {"do": "send", "event": {"type": "conversation.item.delete", "item_id": "$item[3]"}},
                  {"do": "wait", "for": "conversation.item.deleted"},
                  user_text("What is my name?"), CREATE, WAIT_DONE],
    },
    {
        "name": "item_insert_retrieve",
        "phase": "P3",
        "steps": [no_vad(output_modalities=["text"]), WAIT_UPDATED,
                  user_text("My name is Peter."), CREATE, WAIT_DONE,
                  {"do": "send", "event": {"type": "conversation.item.create", "previous_item_id": "$item[0]", "item": {
                      "type": "message", "role": "user", "content": [{"type": "input_text", "text": "I am 42 years old."}]}}},
                  {"do": "wait", "for": "conversation.item.done"},
                  {"do": "send", "event": {"type": "conversation.item.retrieve", "item_id": "$item[-1]"}},
                  {"do": "wait", "for": "conversation.item.retrieved"},
                  user_text("How old am I?"), CREATE, WAIT_DONE],
    },
    {
        "name": "audio_clear",
        "phase": "P3",
        "steps": [no_vad(output_modalities=["text"]), WAIT_UPDATED,
                  {"do": "audio", "file": "q_capital.wav"},
                  {"do": "send", "event": {"type": "input_audio_buffer.clear"}},
                  {"do": "wait", "for": "input_audio_buffer.cleared"},
                  user_text("What is two plus two?"), CREATE, WAIT_DONE],
    },
    {
        "name": "audio_truncate",
        "phase": "P3",
        "steps": [no_vad(output_modalities=["audio"]), WAIT_UPDATED,
                  user_text("Tell me about Paris in two sentences."), CREATE, WAIT_DONE,
                  {"do": "send", "event": {"type": "conversation.item.truncate", "item_id": "$item[1]", "content_index": 0,
                                           "audio_end_ms": 1000}},
                  {"do": "wait", "for": "conversation.item.truncated"},
                  {"do": "send", "event": {"type": "conversation.item.retrieve", "item_id": "$item[1]"}},
                  {"do": "wait", "for": "conversation.item.retrieved"}],
    },
    {
        # what a client does on barge-in (e.g. Pipecat): the user speaks, the heard part of the answer is truncated
        "name": "truncate_while_speaking",
        "phase": "P3",
        "steps": [no_vad(output_modalities=["audio"]), WAIT_UPDATED,
                  user_text("Tell me about Paris in two sentences."), CREATE, WAIT_DONE,
                  {"do": "audio", "file": "q_population.wav"},
                  {"do": "send", "event": {"type": "conversation.item.truncate", "item_id": "$item[1]", "content_index": 0,
                                           "audio_end_ms": 800}},
                  {"do": "wait", "for": "conversation.item.truncated"},
                  COMMIT, CREATE, WAIT_DONE],
    },
    {
        "name": "instructions_change",
        "phase": "P3",
        "steps": [no_vad(output_modalities=["text"]), WAIT_UPDATED,
                  user_text("My name is Peter."), CREATE, WAIT_DONE,
                  {"do": "send", "event": {"type": "session.update", "session": {"type": "realtime",
                      "instructions": "Answer in German, in one short sentence."}}},
                  WAIT_UPDATED,
                  user_text("What is my name?"), CREATE, WAIT_DONE],
    },
    {
        "name": "oob_response",
        "phase": "P4",
        "steps": [no_vad(output_modalities=["text"]), WAIT_UPDATED,
                  user_text("My name is Peter and I love sailing."), CREATE, WAIT_DONE,
                  {"do": "send", "event": {"type": "response.create", "response": {"conversation": "none",
                      "metadata": {"topic": "classification"}, "output_modalities": ["text"]}}},
                  WAIT_DONE,
                  user_text("What is my name?"), CREATE, WAIT_DONE],
    },
    {
        "name": "oob_input",
        "phase": "P4",
        "steps": [no_vad(output_modalities=["text"]), WAIT_UPDATED,
                  user_text("My name is Peter and I love sailing."), CREATE, WAIT_DONE,
                  {"do": "send", "event": {"type": "response.create", "response": {"conversation": "none", "input": [
                      {"type": "item_reference", "id": "$item[0]"},
                      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What hobby did I mention? One word."}]}]}}},
                  WAIT_DONE],
    },
    {
        "name": "response_instructions",
        "phase": "P4",
        "steps": [no_vad(output_modalities=["text"]), WAIT_UPDATED,
                  user_text("Say hello to me."),
                  {"do": "send", "event": {"type": "response.create", "response": {"instructions": "Answer only in capital letters."}}},
                  WAIT_DONE,
                  user_text("Say goodbye to me."), CREATE, WAIT_DONE],
    },
    {
        # tools only in response.create, none in the session: the tool output still goes into the conversation
        "name": "response_tools",
        "phase": "P4",
        "steps": [no_vad(output_modalities=["text"], tools=[]), WAIT_UPDATED,
                  user_text("What's the weather in Berlin right now?"),
                  {"do": "send", "event": {"type": "response.create", "response": {
                      "output_modalities": ["text"], "tools": [WEATHER_TOOL], "tool_choice": "required",
                      "instructions": "Call the matching tool.", "metadata": {"turn": "1"}}}},
                  WAIT_DONE,
                  {"do": "tool_output", "output": "{\"temperature_c\": 12, \"condition\": \"light rain\"}"},
                  {"do": "wait", "for": "conversation.item.done"},
                  {"do": "send", "event": {"type": "response.create", "response": {
                      "output_modalities": ["text"], "tools": [],
                      "instructions": "Tell the user the weather in one short sentence."}}},
                  WAIT_DONE],
    },
    {
        # server VAD interrupts a spoken response that has its own instructions; the client truncates what was heard
        "name": "barge_in_response_instructions",
        "phase": "P4",
        "steps": [session(output_modalities=["audio"], audio={"input": {
                      "format": {"type": "audio/pcm", "rate": 24000},
                      "turn_detection": {"type": "server_vad", "silence_duration_ms": 500,
                                         "create_response": False, "interrupt_response": True}}}),
                  WAIT_UPDATED,
                  user_text("Tell me about Paris."),
                  {"do": "send", "event": {"type": "response.create", "response": {
                      "output_modalities": ["audio"], "tools": [],
                      "instructions": "Answer in at least eight long sentences."}}},
                  {"do": "wait", "for": "response.output_audio.delta"},
                  {"do": "audio", "file": "q_population.wav", "silence_before_ms": 300, "silence_after_ms": 1200, "realtime": True},
                  {"do": "wait", "for": "input_audio_buffer.speech_started"},
                  {"do": "wait", "for": "response.done"},
                  {"do": "send", "event": {"type": "conversation.item.truncate", "item_id": "$item[1]", "content_index": 0,
                                           "audio_end_ms": 500}},
                  {"do": "wait", "for": "conversation.item.truncated"}],
    },
    {
        "name": "input_transcription",
        "phase": "P4",
        "steps": [session(output_modalities=["text"], audio={"input": {"turn_detection": None,
                      "transcription": {"model": "gpt-4o-mini-transcribe", "language": "en"}}}), WAIT_UPDATED,
                  {"do": "audio", "file": "q_capital.wav"}, COMMIT,
                  {"do": "wait", "for": "conversation.item.input_audio_transcription.completed"},
                  CREATE, WAIT_DONE],
    },
]
