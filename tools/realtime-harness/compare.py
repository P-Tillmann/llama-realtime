"""Structural comparison of two recorded runs (reference = OpenAI, candidate = llama-server).

Content (text, audio, ids) is expected to differ and is ignored. Compared are:
  - the order of server event types (runs of the same type collapsed),
  - the field paths of each event type,
  - latency from each response.create to the first output delta and to response.done.
"""

import difflib
import json
from pathlib import Path

from scenarios import SCENARIOS

PHASE = {sc["name"]: sc["phase"] for sc in SCENARIOS}


def received(run):
    return [(e["t"], e["event"]) for e in run["log"] if "event" in e]


def collapsed_types(run):
    out = []
    for _, ev in received(run):
        t = ev.get("type", "?")
        if out and out[-1][0] == t:
            out[-1][1] += 1
        else:
            out.append([t, 1])
    return out


def paths(obj, prefix=""):
    """Field paths of a JSON value; list items are merged under []."""
    res = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            res.add(p)
            res |= paths(v, p)
    elif isinstance(obj, list):
        for v in obj:
            res |= paths(v, prefix + "[]")
    return res


def schema(run):
    by_type = {}
    for _, ev in received(run):
        by_type.setdefault(ev.get("type", "?"), set()).update(paths(ev))
    return by_type


def latencies(run):
    """(first output delta, response.done) in seconds after each response.create, or after a commit with VAD."""
    out, start, first = [], None, None
    for e in run["log"]:
        if "sent" in e and e["sent"]["type"] == "response.create":
            start, first = e["t"], None
        elif "event" in e:
            t = e["event"].get("type", "")
            if start is None and t == "input_audio_buffer.speech_stopped":
                start, first = e["t"], None
            if start is not None and first is None and t.startswith("response.output_") and t.endswith(".delta"):
                first = e["t"] - start
            if start is not None and t == "response.done":
                out.append((first, e["t"] - start))
                start = None
    return out


def fmt_lat(lats):
    if not lats:
        return "-"
    return ", ".join(f"{(f or 0):.2f}/{d:.2f}s" if f is not None else f"-/{d:.2f}s" for f, d in lats)


def compare_run(name, ref, cand):
    lines = [f"## {name} (phase {PHASE.get(name, '?')})", ""]
    lines.append(f"- reference: {ref['status']} {ref.get('error') or ''}".rstrip())
    lines.append(f"- candidate: {cand['status']} {cand.get('error') or ''}".rstrip())
    lines.append(f"- latency first-delta/done: reference {fmt_lat(latencies(ref))}, candidate {fmt_lat(latencies(cand))}")
    lines.append("")

    rt, ct = collapsed_types(ref), collapsed_types(cand)
    fmt = lambda x: f"{x[0]} x{x[1]}" if x[1] > 1 else x[0]
    diff = list(difflib.unified_diff([x[0] for x in rt], [x[0] for x in ct], "reference", "candidate", lineterm="", n=100))
    if diff:
        lines += ["Event order (`-` only in reference, `+` only in candidate):", "", "```diff"] + diff[2:] + ["```", ""]
    else:
        lines += ["Event order: identical (" + ", ".join(fmt(x) for x in rt) + ")", ""]

    rs, cs = schema(ref), schema(cand)
    missing_types = sorted(set(rs) - set(cs))
    field_rows = []
    for t in sorted(set(rs) & set(cs)):
        miss, extra = sorted(rs[t] - cs[t]), sorted(cs[t] - rs[t])
        if miss or extra:
            field_rows.append((t, miss, extra))
    if missing_types:
        lines += ["Event types missing in candidate: " + ", ".join(f"`{t}`" for t in missing_types), ""]
    if field_rows:
        lines += ["| Event | Fields missing in candidate | Extra in candidate |", "| --- | --- | --- |"]
        for t, miss, extra in field_rows:
            lines.append(f"| `{t}` | {', '.join(miss) or '-'} | {', '.join(extra) or '-'} |")
        lines.append("")
    n_missing = sum(len(m) for _, m, _ in field_rows)
    summary = (name, PHASE.get(name, "?"), ref["status"], cand["status"], len(missing_types), n_missing,
               "yes" if not diff else "no")
    return summary, lines


def compare_dirs(ref_dir: Path, cand_dir: Path) -> str:
    summaries, body = [], []
    for ref_path in sorted(ref_dir.glob("*.json")):
        cand_path = cand_dir / ref_path.name
        ref = json.loads(ref_path.read_text())
        if not cand_path.exists():
            summaries.append((ref_path.stem, PHASE.get(ref_path.stem, "?"), ref["status"], "not run", "-", "-", "-"))
            continue
        s, lines = compare_run(ref_path.stem, ref, json.loads(cand_path.read_text()))
        summaries.append(s)
        body += lines
    head = ["# Realtime conformance report", "",
            f"Reference: `{ref_dir}`, candidate: `{cand_dir}`", "",
            "| Scenario | Phase | Reference | Candidate | Missing event types | Missing fields | Same event order |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    head += [f"| {' | '.join(str(x) for x in s)} |" for s in summaries]
    return "\n".join(head + [""] + body) + "\n"
