"""Cost of coordination, decomposed by mechanism (Supp. D, Table S.11).

Buckets every LLM call recorded in the task logs by its trace type and
reports per-task token estimates per mechanism, so the terms that grow with
the population (generation / routing prompts / subscription embeddings /
watchdog / reputation state) can be compared at N=20 vs N=80.

Usage:
    python analysis/cost_decomposition.py "logs/log_mmlu_*.json"
"""
import sys
from collections import defaultdict
from typing import Any, Dict, List

from log_io import load_task_logs, count_tokens, iter_llm_traces

# Trace type -> mechanism bucket of Table S.11.
BUCKETS = {
    "publish": "agent execution",
    "refine_system_prompt": "subscription refinement",
    "broker_route": "routing decisions",
    "subscription_embedding": "subscription embedding",
    "watchdog_evaluate": "watchdog verifications",
    "tool_compute": "tool execution",
    "code_repair": "tool execution",
    "edge_cases": "tool execution",
    "retrieve": "retrieval",
}


def decompose(task_log: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Per-mechanism {calls, prompt_tokens, completion_tokens} for one episode."""
    per = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    for tr in iter_llm_traces(task_log):
        bucket = BUCKETS.get(tr["type"], tr["type"])
        per[bucket]["calls"] += 1
        per[bucket]["prompt_tokens"] += sum(count_tokens(m.get("content", "")) for m in tr["messages"])
        per[bucket]["completion_tokens"] += count_tokens(tr["output"])
    # reputation state: one Beta record (two floats) per observed peer
    snap = {}
    for step in reversed(task_log.get("steps", [])):
        if step.get("reputation"):
            snap = step["reputation"]
            break
    per["reputation state (records)"] = {
        "calls": sum(len(peers) for peers in snap.values()), "prompt_tokens": 0, "completion_tokens": 0}
    return dict(per)


def summarize(task_logs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Mean per-task decomposition over a run, plus engaged-agents count."""
    totals = defaultdict(lambda: {"calls": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0})
    engaged = []
    for log in task_logs:
        for mech, vals in decompose(log).items():
            for k in vals:
                totals[mech][k] += vals[k]
        team = log.get("team", {})
        if team.get("active_per_step"):
            engaged.append(sum(team["active_per_step"]) / len(team["active_per_step"]))
    n = max(1, len(task_logs))
    out = {m: {k: round(v / n, 1) for k, v in vals.items()} for m, vals in sorted(totals.items())}
    if engaged:
        out["engaged agents (mean/step)"] = {"calls": round(sum(engaged) / len(engaged), 2),
                                             "prompt_tokens": 0, "completion_tokens": 0}
    return out


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else "logs/log_*.json"
    logs = load_task_logs(pattern)
    if not logs:
        sys.exit(f"no task logs matched {pattern!r}")
    summary = summarize(logs)
    print(f"{len(logs)} task logs | mean per-task cost by mechanism")
    print(f"{'mechanism':<32}{'calls':>8}{'prompt_tok':>12}{'completion_tok':>16}")
    for mech, vals in summary.items():
        print(f"{mech:<32}{vals['calls']:>8}{vals['prompt_tokens']:>12}{vals['completion_tokens']:>16}")
