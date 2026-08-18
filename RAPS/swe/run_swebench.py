#!/usr/bin/env python3
"""RAPS-on-SWE-bench batch runner (per-step RAPS dynamic ad-hoc network).

Aligned with the revision's Long-Horizon Scenario (§4.5): both SWE-bench
releases taken in full, a 75-step budget, five single-responsibility
personas (RAPS/swe/team.py), one backbone per run via --model (gpt-4o / gpt-5.2 /
gpt-4o-mini as the floor), four independent repeats per instance (--repeats), and
the subscription-freshness ablation behind --refine_once (§2.6).

Usage:
    python RAPS/swe/run_swebench.py --dataset lite --out swe_runs/raps_4o_lite \
        --model gpt-4o --workers 4 --step_limit 75 --repeats 4
    python RAPS/swe/run_swebench.py --dataset verified \
        --out swe_runs/raps_gpt52_verified --model gpt-5.2 --workers 8

Each repeat is a self-contained run directory, <out>/repeat-<n>/, holding its own
swebench-compatible preds.json, its per-instance trajectories and a metrics.json
(rounds / llm_calls / tokens / USD / wall-time). <out>/metrics.json carries the
per-repeat summaries and their mean. The swebench harness scores the patches of
each repeat separately, run against that repeat's preds.json.
"""
import argparse
import copy
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # repo root, portable across machines
sys.path.insert(0, str(ROOT))

from datasets import load_dataset
from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_sb_environment, update_preds_file
from minisweagent.utils.serialize import recursive_merge

from RAPS.swe.agent import SWERAPSAgent
from RAPS.swe.team import DEFAULT_TEAM

DATASETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
}


def parse_args():
    p = argparse.ArgumentParser(description="RAPS on SWE-bench (Lite / Verified)")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="lite")
    p.add_argument("--out", default=str(ROOT / "swe_runs/raps_swe"),
                   help="directory the trajectories, predictions and metrics go to")
    p.add_argument("--model", default="gpt-4o",
                   help="backbone served by the litellm route; the reported runs use "
                        "gpt-4o and gpt-5.2, with gpt-4o-mini as a floor reference")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1,
                   help="independent repeats per instance; above one, each repeat is written "
                        "to its own repeat-<n>/ directory with its own preds.json. The "
                        "long-horizon results of §4.5 are the mean over four")
    p.add_argument("--step_limit", type=int, default=75, help="per-episode action budget (§4.5)")
    p.add_argument("--cost_limit", type=float, default=2.0)
    p.add_argument("--refine_once", action="store_true",
                   help="ablation: compute each persona's subscription once per episode (§2.6)")
    p.add_argument("--watchdog", action="store_true",
                   help="run the local watchdog and the reputation gate over the hosts, "
                        "one extra backbone call per round (off by default)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    cfg_base = recursive_merge(
        get_config_from_spec("swebench_backticks.yaml"),
        {
            "environment": {"environment_class": "docker", "pull_timeout": 1800},
            "model": {"model_name": args.model, "model_class": "litellm_textbased"},
            "agent": {"step_limit": args.step_limit, "cost_limit": args.cost_limit,
                      "refine_once": args.refine_once, "watchdog": args.watchdog},
        },
    )

    instances = list(load_dataset(DATASETS[args.dataset], split="test"))
    out.mkdir(parents=True, exist_ok=True)
    repeats = range(1, max(1, args.repeats) + 1)

    def run_dir(repeat):
        """A single run writes straight into --out; repeats each get their own directory."""
        return out if len(repeats) == 1 else out / f"repeat-{repeat}"

    print(f"RAPS on {len(instances)} {args.dataset} instances x {len(repeats)} repeat(s) "
          f"| model={args.model} | workers={args.workers} | step_limit={args.step_limit} "
          f"| refine_once={args.refine_once} | watchdog={args.watchdog}")

    _timing: dict = {}
    _lock = threading.Lock()

    def run_one(instance, repeat):
        iid = instance["instance_id"]
        d = run_dir(repeat)
        cfg = copy.deepcopy(cfg_base)   # per-episode copy (get_sb_environment mutates it)
        (d / iid).mkdir(parents=True, exist_ok=True)
        traj = d / iid / f"{iid}.traj.json"
        t0 = time.time()
        exit_status, submission = None, ""
        try:
            env = get_sb_environment(cfg, instance)
            agent = SWERAPSAgent(get_model(config=cfg["model"]), env, team=DEFAULT_TEAM,
                                 **{**cfg["agent"], "output_path": str(traj)})
            info = agent.run(instance["problem_statement"])
            exit_status, submission = info.get("exit_status"), info.get("submission") or ""
        except Exception as e:
            exit_status = type(e).__name__
            print(f"  ERROR r{repeat} {iid}: {e}")
        finally:
            elapsed = time.time() - t0
            with _lock:
                _timing[(repeat, iid)] = elapsed
                update_preds_file(d / "preds.json", iid,
                                  cfg_base["model"]["model_name"], submission)
        print(f"  done r{repeat} {iid}: {exit_status} in {elapsed:.0f}s")

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, i, r) for r in repeats for i in instances]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print("  future error:", e)
    total_wall = time.time() - t_start

    def collect(repeat):
        """Per-instance metrics of one repeat, read back from its trajectories."""
        rows = []
        for instance in instances:
            iid = instance["instance_id"]
            tf = run_dir(repeat) / iid / f"{iid}.traj.json"
            if not tf.exists():
                rows.append({"instance_id": iid, "exit": "MISSING"})
                continue
            info = json.load(open(tf)).get("info", {})
            rs, ms = info.get("raps_stats", {}), info.get("model_stats", {})
            sub = info.get("submission") or ""
            rows.append({
                "instance_id": iid,
                "exit": info.get("exit_status"),
                "rounds": rs.get("rounds") or ms.get("api_calls"),
                "llm_calls": rs.get("llm_calls"),
                "tokens": rs.get("prompt_tokens", 0) + rs.get("completion_tokens", 0),
                "usd": rs.get("usd") if rs else round(ms.get("instance_cost", 0), 6),
                "wall_s": round(_timing.get((repeat, iid), 0), 1),
                "patch_nonempty": bool(str(sub).strip()),
            })
        return rows

    def summarize(rows, repeat):
        n = max(1, len(rows))
        summary = {
            "n": len(rows),
            "model": args.model,
            "dataset": args.dataset,
            "total_wall_time_s": round(total_wall, 1),
            "patches_nonempty": sum(1 for r in rows if r.get("patch_nonempty")),
            "sum_usd": round(sum((r.get("usd") or 0) for r in rows), 4),
            "sum_tokens": sum(r.get("tokens", 0) for r in rows),
            "avg_rounds": round(sum((r.get("rounds") or 0) for r in rows) / n, 1),
            "avg_llm_calls": round(sum((r.get("llm_calls") or 0) for r in rows) / n, 1),
        }
        return summary if len(repeats) == 1 else {"repeat": repeat, **summary}

    per_repeat = []
    for repeat in repeats:
        rows = collect(repeat)
        summary = summarize(rows, repeat)
        json.dump({"summary": summary, "rows": rows},
                  open(run_dir(repeat) / "metrics.json", "w"), indent=2)
        per_repeat.append(summary)
        if len(repeats) == 1:
            print("\n=== RAPS metrics ===")
            for r in rows:
                print(r)
        else:
            print(f"\n=== repeat {repeat} ===\n{json.dumps(summary, indent=2)}")

    if len(repeats) == 1:
        print(f"\n=== summary ===\n{json.dumps(per_repeat[0], indent=2)}")
        preds = f"{out}/preds.json"
    else:
        keys = ["patches_nonempty", "sum_usd", "sum_tokens", "avg_rounds", "avg_llm_calls"]
        across = {
            "model": args.model, "dataset": args.dataset, "repeats": len(per_repeat),
            "total_wall_time_s": round(total_wall, 1),
            "mean": {k: round(sum(s[k] for s in per_repeat) / len(per_repeat), 4) for k in keys},
            "per_repeat": per_repeat,
        }
        json.dump(across, open(out / "metrics.json", "w"), indent=2)
        print(f"\n=== mean over {len(per_repeat)} repeats ===\n{json.dumps(across['mean'], indent=2)}")
        preds = f"{out}/repeat-<n>/preds.json"
    print(f"\nresolved% comes from the swebench harness, run on {preds}")


if __name__ == "__main__":
    main()
