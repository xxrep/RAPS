#!/usr/bin/env python3
"""RAPS-on-SWE-bench batch runner (per-task RAPS dynamic ad-hoc network).

Usage: run_swebench.py <ids_file> <out_dir> <workers> <step_limit>
Produces swebench-compatible preds.json + per-task metrics.json
(rounds / llm_calls / tokens / USD / wall-time). Resolved% via swebench harness afterwards.
"""
import copy
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = "/home/lirui/raps_swe/RAPS-main"
sys.path.insert(0, ROOT)

from datasets import load_dataset
from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_sb_environment, update_preds_file
from minisweagent.utils.serialize import recursive_merge

from RAPS.swe.agent import SWERAPSAgent
from RAPS.swe.team import DEFAULT_TEAM

IDS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(ROOT) / "runs/dev6.txt"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(ROOT) / "runs/raps_v1"
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
STEP_LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 40
PRICE_IN, PRICE_OUT = 2.50 / 1e6, 10.00 / 1e6

BASE = recursive_merge(
    get_config_from_spec("swebench_backticks.yaml"),
    {
        "environment": {"environment_class": "docker", "pull_timeout": 1800},
        "model": {"model_name": "gpt-4o", "model_class": "litellm_textbased"},
        "agent": {"step_limit": STEP_LIMIT, "cost_limit": 2.0},
    },
)

want = [x.strip() for x in IDS_FILE.read_text().split() if x.strip()]
ds = list(load_dataset("princeton-nlp/SWE-bench_Lite", split="test"))
instances = [x for x in ds if x["instance_id"] in set(want)]
OUT.mkdir(parents=True, exist_ok=True)
print(f"RAPS on {len(instances)} instances | workers={WORKERS} | step_limit={STEP_LIMIT}")

_timing: dict = {}
_lock = threading.Lock()


def run_one(instance):
    iid = instance["instance_id"]
    cfg = copy.deepcopy(BASE)  # per-instance copy (get_sb_environment mutates it)
    (OUT / iid).mkdir(parents=True, exist_ok=True)
    traj = OUT / iid / f"{iid}.traj.json"
    t0 = time.time()
    exit_status, submission = None, ""
    try:
        env = get_sb_environment(cfg, instance)
        agent = SWERAPSAgent(get_model(config=cfg["model"]), env, team=DEFAULT_TEAM,
                             **{**cfg["agent"], "output_path": str(traj)})
        task = instance["problem_statement"]
        if os.getenv("ORACLE_LOC") == "1":  # ablation: seed gold-patch files (NOT a fair number)
            gold_files = sorted(set(re.findall(r"^\+\+\+ b/(.+)$", instance.get("patch", "") or "", re.M)))
            if gold_files:
                task += ("\n\n## Localization hint — the fix is expected to modify ONLY these "
                         "file(s):\n" + "\n".join(f"- {f}" for f in gold_files))
        info = agent.run(task)
        exit_status, submission = info.get("exit_status"), info.get("submission") or ""
    except Exception as e:
        exit_status = type(e).__name__
        print(f"  ERROR {iid}: {e}")
    finally:
        with _lock:
            _timing[iid] = time.time() - t0
        update_preds_file(OUT / "preds.json", iid, BASE["model"]["model_name"], submission)
    print(f"  done {iid}: {exit_status} in {_timing[iid]:.0f}s")


t_start = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(run_one, i) for i in instances]
    for f in as_completed(futs):
        try:
            f.result()
        except Exception as e:
            print("  future error:", e)
total_wall = time.time() - t_start

rows = []
for iid in want:
    tf = OUT / iid / f"{iid}.traj.json"
    if not tf.exists():
        rows.append({"instance_id": iid, "exit": "MISSING"}); continue
    d = json.load(open(tf))
    info = d.get("info", {})
    rs = info.get("raps_stats", {})
    ms = info.get("model_stats", {})
    sub = info.get("submission") or ""
    rows.append({
        "instance_id": iid,
        "exit": info.get("exit_status"),
        "rounds": rs.get("rounds") or ms.get("api_calls"),
        "llm_calls": rs.get("llm_calls"),
        "tokens": rs.get("prompt_tokens", 0) + rs.get("completion_tokens", 0),
        "usd": rs.get("usd") if rs else round(ms.get("instance_cost", 0), 6),
        "wall_s": round(_timing.get(iid, 0), 1),
        "patch_nonempty": bool(str(sub).strip()),
    })
summary = {
    "n": len(rows),
    "total_wall_time_s": round(total_wall, 1),
    "patches_nonempty": sum(1 for r in rows if r.get("patch_nonempty")),
    "sum_usd": round(sum((r.get("usd") or 0) for r in rows), 4),
    "sum_tokens": sum(r.get("tokens", 0) for r in rows),
    "avg_rounds": round(sum((r.get("rounds") or 0) for r in rows) / max(1, len(rows)), 1),
    "avg_llm_calls": round(sum((r.get("llm_calls") or 0) for r in rows) / max(1, len(rows)), 1),
}
json.dump({"summary": summary, "rows": rows}, open(OUT / "metrics.json", "w"), indent=2)
print("\n=== RAPS metrics ===")
for r in rows:
    print(r)
print("\n=== summary ===")
print(json.dumps(summary, indent=2))
