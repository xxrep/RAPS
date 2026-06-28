#!/usr/bin/env python3
"""
Single-agent mini-swe-agent baseline WITH per-task metrics.

Records per instance: rounds (api calls), prompt/completion/total tokens,
USD (tokens x gpt-4o-mini price), wall-time, exit status, patch-nonempty.
Reuses mini-swe-agent's process_instance (backticks + litellm_textbased mode);
adds wall-timing + exact token capture via a litellm success callback.

Usage: run_baseline_metrics.py <ids_file> <out_dir> <workers>
"""
import json, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
import tiktoken
from datasets import load_dataset
from minisweagent.config import get_config_from_spec
from minisweagent.utils.serialize import recursive_merge
from minisweagent.run.benchmarks.swebench import process_instance

ROOT = Path("/home/lirui/raps_swe/RAPS-main")
IDS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/eval20.txt"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs/baseline_v2"
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

# gpt-4o price (USD per token): $2.50/1M in, $10.00/1M out
PRICE_IN, PRICE_OUT = 2.50 / 1e6, 10.00 / 1e6
_lock = threading.Lock()


class StubPM:
    """No-op progress manager (skip the rich Live UI)."""
    def on_instance_start(self, *a, **k): pass
    def update_instance_status(self, *a, **k): pass
    def on_instance_end(self, *a, **k): pass


# ---------- config: backticks prompts + textbased model + gpt-4o ----------
specs = ["swebench_backticks.yaml", "environment.pull_timeout=1800"]
configs = [get_config_from_spec(s) for s in specs]
configs.append({
    "environment": {"environment_class": "docker"},
    "model": {"model_name": "gpt-4o", "model_class": "litellm_textbased"},
})
config = recursive_merge(*configs)

# ---------- instances ----------
want = [x.strip() for x in IDS_FILE.read_text().split() if x.strip()]
ds = list(load_dataset("princeton-nlp/SWE-bench_Lite", split="test"))
instances = [x for x in ds if x["instance_id"] in set(want)]
OUT.mkdir(parents=True, exist_ok=True)
print(f"running {len(instances)} instances, workers={WORKERS}, out={OUT}")

_timing: dict = {}


def run_one(instance):
    iid = instance["instance_id"]
    t0 = time.time()
    try:
        process_instance(instance, OUT, config, StubPM())
    finally:
        with _lock:
            _timing[iid] = time.time() - t0
    print(f"  done {iid} in {_timing[iid]:.0f}s")


t_start = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(run_one, i) for i in instances]
    for f in as_completed(futs):
        try:
            f.result()
        except Exception as e:
            print("  ERROR:", e)
total_wall = time.time() - t_start

# ---------- tiktoken cross-check (billed-style: each turn re-sends history) ----------
enc = tiktoken.get_encoding("o200k_base")


def tiktoken_billed(messages):
    p = c = 0
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            ctx = messages[:i]
            p += sum(len(enc.encode(str(x.get("content", "")))) for x in ctx) + 4 * len(ctx)
            c += len(enc.encode(str(m.get("content", ""))))
    return p, c


# ---------- merge + report ----------
rows = []
for iid in want:
    tf = OUT / iid / f"{iid}.traj.json"
    if not tf.exists():
        rows.append({"instance_id": iid, "exit_status": "MISSING"}); continue
    d = json.load(open(tf)); info = d["info"]; ms = info.get("model_stats", {})
    tk_p, tk_c = tiktoken_billed(d.get("messages", []))  # billed-style: history re-sent each turn
    rows.append({
        "instance_id": iid,
        "exit_status": info.get("exit_status"),
        "rounds": ms.get("api_calls"),
        "prompt_tokens": tk_p, "completion_tokens": tk_c, "total_tokens": tk_p + tk_c,
        "usd_tokens": round(tk_p * PRICE_IN + tk_c * PRICE_OUT, 6),
        "usd_litellm": round(ms.get("instance_cost") or 0.0, 6),
        "wall_time_s": round(_timing.get(iid, 0), 1),
        "patch_nonempty": bool((info.get("submission") or "").strip()),
    })

summary = {
    "n": len(rows),
    "total_wall_time_s": round(total_wall, 1),
    "sum_total_tokens": sum(r.get("total_tokens", 0) for r in rows),
    "sum_usd_tokens": round(sum(r.get("usd_tokens", 0) for r in rows), 4),
    "sum_usd_litellm": round(sum(r.get("usd_litellm", 0) for r in rows), 4),
    "avg_rounds": round(sum((r.get("rounds") or 0) for r in rows) / max(1, len(rows)), 1),
    "patches_nonempty": sum(1 for r in rows if r.get("patch_nonempty")),
}
json.dump({"summary": summary, "rows": rows}, open(OUT / "metrics.json", "w"), indent=2)

print("\n=== per-task metrics ===")
hdr = f"{'instance':32s} {'exit':14s} {'rnds':>4s} {'tok':>7s} {'$tok':>8s} {'$ll':>8s} {'wall_s':>7s} {'patch':>5s}"
print(hdr)
for r in rows:
    print(f"{r['instance_id']:32s} {str(r.get('exit_status'))[:14]:14s} "
          f"{str(r.get('rounds','')):>4s} {r.get('total_tokens',0):>7d} "
          f"{r.get('usd_tokens',0):>8.4f} {r.get('usd_litellm',0):>8.4f} "
          f"{r.get('wall_time_s',0):>7.0f} {str(r.get('patch_nonempty','')):>5s}")
print("\n=== summary ===")
print(json.dumps(summary, indent=2))
