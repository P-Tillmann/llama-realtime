#!/usr/bin/env python3
"""Run realtime API scenarios against OpenAI or llama-server and record the server events.

  python harness.py record --target openai --out fixtures/openai            # needs OPENAI_API_KEY
  python harness.py record --target local  --out out/local --url ws://127.0.0.1:8080/v1/realtime
  python harness.py compare fixtures/openai out/local --report out/report.md

Recorded logs keep every event in full, except that base64 audio is replaced by its byte count.
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import websockets  # type: ignore  # not in requirements-all.txt, see requirements.txt here

from scenarios import SCENARIOS

HERE = Path(__file__).resolve().parent
AUDIO_DIR = HERE / "audio"


def load_pcm16(name: str, rate_out: int) -> bytes:
    with wave.open(str(AUDIO_DIR / name)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, f"{name}: need mono PCM16"
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    if rate != rate_out:
        # linear resampling is good enough for test input
        n = int(len(x) * rate_out / rate)
        x = np.interp(np.arange(n) * rate / rate_out, np.arange(len(x)), x)
    return np.clip(x, -32768, 32767).astype(np.int16).tobytes()


def strip_audio(ev: dict) -> dict:
    """Replace base64 audio payloads by their size, so logs stay small and diffable."""
    ev = json.loads(json.dumps(ev))
    if ev.get("type", "").endswith("audio.delta") and isinstance(ev.get("delta"), str):
        ev["delta"] = {"audio_bytes": len(base64.b64decode(ev["delta"]))}
    return ev


class Session:
    def __init__(self, ws, t0):
        self.ws = ws
        self.t0 = t0
        self.events = []  # (t, event)
        self.queue: asyncio.Queue = asyncio.Queue()

    async def reader(self):
        try:
            async for msg in self.ws:
                ev = json.loads(msg)
                self.events.append({"t": round(time.monotonic() - self.t0, 3), "event": strip_audio(ev)})
                await self.queue.put(ev)
        except websockets.ConnectionClosed:
            pass

    async def send(self, ev: dict):
        self.events.append({"t": round(time.monotonic() - self.t0, 3), "sent": strip_audio(ev) if ev["type"] != "input_audio_buffer.append" else {"type": ev["type"], "audio": {"audio_bytes": len(base64.b64decode(ev["audio"]))}}})
        await self.ws.send(json.dumps(ev))

    async def wait(self, types, timeout):
        """Wait for one of the event types; an error event ends the wait (and the scenario)."""
        types = {types} if isinstance(types, str) else set(types)
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no {sorted(types)} within {timeout}s")
            try:
                ev = await asyncio.wait_for(self.queue.get(), left)
            except asyncio.TimeoutError:
                raise TimeoutError(f"no {sorted(types)} within {timeout}s") from None
            if ev["type"] in types:
                return ev
            if ev["type"] == "error":
                err = ev.get("error", {})
                raise RuntimeError(f"error event while waiting for {sorted(types)}: {err.get('code') or err.get('type')}: {err.get('message')}")


async def run_scenario(sc: dict, target: str, url: str, model: str, timeout: float) -> dict:
    headers = {}
    if target == "openai":
        url = f"wss://api.openai.com/v1/realtime?model={model}"
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("OPENAI_API_KEY is empty")
        headers["Authorization"] = f"Bearer {key}"
    elif os.environ.get("LOCAL_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['LOCAL_API_KEY']}"  # e.g. a gateway in front of llama-server
    t0 = time.monotonic()
    proxy = None if target == "local" else True  # local server: never through an HTTPS proxy
    async with websockets.connect(url, additional_headers=headers, max_size=None, proxy=proxy) as ws:
        s = Session(ws, t0)
        reader = asyncio.create_task(s.reader())
        status, error = "ok", None
        state = {"rate": 24000, "call_id": None}
        try:
            created = await s.wait({"session.created", "transcription_session.created"}, timeout)
            fmt = created["session"].get("audio", {}).get("input", {}).get("format", {})
            state["rate"] = fmt.get("rate", 24000)
            for step in sc["steps"]:
                await run_step(s, step, state, timeout)
        except Exception as e:  # recorded, the comparison shows where the run stopped
            status, error = "failed", f"{type(e).__name__}: {e}"
        await asyncio.sleep(0.5)
        reader.cancel()
    n_err = sum(1 for e in s.events if e.get("event", {}).get("type") == "error")
    if status == "ok" and n_err:
        status, error = "errors", f"{n_err} error event(s)"
    return {"scenario": sc["name"], "target": target, "model": model if target == "openai" else url,
            "status": status, "error": error, "log": s.events}


def item_ids(s: "Session") -> list:
    """Ids of the conversation items in the order the server added them."""
    ids = []
    for e in s.events:
        ev = e.get("event", {})
        if ev.get("type") == "conversation.item.added" and ev["item"]["id"] not in ids:
            ids.append(ev["item"]["id"])
    return ids


def resolve(obj, s: "Session"):
    """Replaces "$item[i]" strings by the id of the i-th conversation item (negative i counts from the end)."""
    if isinstance(obj, dict):
        return {k: resolve(v, s) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve(v, s) for v in obj]
    if isinstance(obj, str) and obj.startswith("$item[") and obj.endswith("]"):
        return item_ids(s)[int(obj[6:-1])]
    return obj


async def run_step(s: Session, step: dict, state: dict, timeout: float):
    kind = step["do"]
    if kind == "send":
        await s.send(resolve(step["event"], s))
    elif kind == "audio":
        pcm = load_pcm16(step["file"], state["rate"])
        silence = b"\0\0" * int(state["rate"] * step.get("silence_after_ms", 0) / 1000)
        pcm = b"\0\0" * int(state["rate"] * step.get("silence_before_ms", 0) / 1000) + pcm + silence
        chunk = state["rate"] // 10 * 2  # 100 ms
        for i in range(0, len(pcm), chunk):
            await s.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i:i + chunk]).decode()})
            if step.get("realtime"):
                await asyncio.sleep(0.1)
    elif kind == "wait":
        ev = await s.wait(step["for"], step.get("timeout", timeout))
        if ev["type"] == "response.done":
            for item in ev["response"].get("output", []):
                if item.get("type") == "function_call":
                    state["call_id"] = item.get("call_id")
    elif kind == "tool_output":
        await s.send({"type": "conversation.item.create", "item": {
            "type": "function_call_output", "call_id": state["call_id"], "output": step["output"]}})
    else:
        raise ValueError(f"unknown step {kind}")


async def record(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.scenario or [sc["name"] for sc in SCENARIOS]
    failed = 0
    for sc in SCENARIOS:
        if sc["name"] not in names:
            continue
        res = await run_scenario(sc, args.target, args.url, args.model, args.timeout)
        (out / f"{sc['name']}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
        n = sum(1 for e in res["log"] if "event" in e)
        print(f"{sc['name']:24s} {res['status']:6s} {n:4d} events  {res['error'] or ''}")
        failed += res["status"] != "ok"
    return failed


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--target", choices=["openai", "local"], required=True)
    r.add_argument("--url", default="ws://127.0.0.1:8080/v1/realtime")
    r.add_argument("--model", default="gpt-realtime")
    r.add_argument("--out", required=True)
    r.add_argument("--scenario", action="append", help="run only these (repeatable)")
    r.add_argument("--timeout", type=float, default=120)
    r.add_argument("--strict", action="store_true", help="exit 1 if any scenario did not end ok")
    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.add_argument("--report")
    args = ap.parse_args()
    if args.cmd == "record":
        failed = asyncio.run(record(args))
        if args.strict and failed:
            return 1
    else:
        from compare import compare_dirs
        report = compare_dirs(Path(args.reference), Path(args.candidate))
        if args.report:
            Path(args.report).write_text(report)
        print(report)


if __name__ == "__main__":
    sys.exit(main())
