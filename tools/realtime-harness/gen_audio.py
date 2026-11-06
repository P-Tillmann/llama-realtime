#!/usr/bin/env python3
"""Generate the scenario audio clips with Mistral TTS (needs MISTRAL_API_KEY).

The clips are committed, so this only runs when clips are added or changed.
"""

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "audio"

# file -> (voice slug, text); voices: GET https://api.mistral.ai/v1/audio/voices
CLIPS = {
    "q_capital.wav":    ("voice_3150", "What is the capital of France?"),
    "q_population.wav": ("voice_478",  "And how many people live there?"),
    "um_label.wav":     ("voice_3150", "I can't seem to, um... find the return label."),
}


def tts(voice: str, text: str) -> bytes:
    body = json.dumps({"model": "voxtral-mini-tts-3", "input": text, "voice": voice, "response_format": "wav"}).encode()
    req = urllib.request.Request("https://api.mistral.ai/v1/audio/speech", data=body, method="POST", headers={
        "Authorization": f"Bearer {os.environ.get('MISTRAL_API_KEY', '')}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return base64.b64decode(json.load(r)["audio_data"])


def main(names):
    OUT.mkdir(exist_ok=True)
    for name, (voice, text) in CLIPS.items():
        if names and name not in names:
            continue
        (OUT / name).write_bytes(tts(voice, text))
        print(f"{name}: {text!r} ({voice})")


if __name__ == "__main__":
    main(sys.argv[1:])
