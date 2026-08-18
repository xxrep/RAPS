"""Role evolution within SWE-bench episodes (§2.6), from trajectory raps_logs.

Reads <out>/<iid>/<iid>.traj.json files and measures, over all episodes:
  * modal acting role per episode stage (early / mid / late third of rounds);
  * host re-selection rate (fraction of rounds whose acting persona differs
    from the previous round's);
  * subscription specificity — fraction of refined roles naming a concrete
    file, symbol or test (rather than generic guidance);
  * subscription-action agreement — fraction of rounds whose command matches
    the acting persona's action vocabulary;
  * for episodes run with the watchdog, the share of publications judged faulty
    and how often the gate withheld a host.

Usage:
    python analysis/swe_role_evolution.py "swe_runs/raps_4o_lite/*/*.traj.json"
"""
import glob
import json
import re
import sys
from collections import Counter
from typing import Any, Dict, List

# Action vocabulary per persona: a round "agrees" when its command uses the
# verbs of the role that published it.
ACTION_VOCAB = {
    "Issue Analyst": (r"cat\b", r"sed -n", r"head", r"grep", r"less"),
    "Code Locator": (r"grep", r"\brg\b", r"find\b", r"sed -n", r"ctags"),
    "Patch Author": (r"apply_edit", r"git --no-pager diff", r"patch"),
    "Test Runner": (r"raps_repro", r"pytest", r"python\s+\S*test", r"python /tmp"),
    "Reviewer": (r"git --no-pager diff", r"git status", r"git log"),
}

_SPECIFIC = re.compile(
    r"/testbed/\S+|[^\s`]+\.py\b|`[^`]+`|test_\w+|def \w+|class \w+", re.IGNORECASE)


def episode_rows(raps_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-episode role-evolution quantities from one trajectory's raps_log."""
    n = len(raps_log)
    if not n:
        return {}
    # modal role per third of the episode
    thirds = [raps_log[: n // 3 or 1], raps_log[n // 3 or 1: 2 * (n // 3 or 1)],
              raps_log[2 * (n // 3 or 1):]]
    modal = []
    for part in thirds:
        roles = Counter(r.get("routed_to") for r in part)
        modal.append(roles.most_common(1)[0][0] if roles else None)
    reselected = sum(1 for r in raps_log if r.get("reselected"))
    refined = [r for r in raps_log if r.get("refined_role")]
    specific = sum(1 for r in refined if _SPECIFIC.search(r["refined_role"]))
    agreed = 0
    comparable = 0
    for r in raps_log:
        vocab = ACTION_VOCAB.get(r.get("routed_to"))
        if not vocab:
            continue
        comparable += 1
        if any(re.search(p, r.get("command", "")) for p in vocab):
            agreed += 1
    judged = [r for r in raps_log if r.get("verdict")]
    return {"rounds": n, "modal_by_stage": modal,
            "reselection_rate": reselected / n,
            "specificity": (specific / len(refined)) if refined else None,
            "agreement": (agreed / comparable) if comparable else None,
            # present only for episodes run with the watchdog
            "faulty_rate": (sum(1 for r in judged if r["verdict"] == "faulty") / len(judged))
                           if judged else None,
            "withholdings": sum(len(r.get("withheld") or []) for r in raps_log)}


def summarize(pattern: str) -> Dict[str, Any]:
    episodes = []
    for path in sorted(glob.glob(pattern)):
        try:
            data = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        raps_log = data.get("raps_log") or []
        row = episode_rows(raps_log)
        if row:
            row["instance"] = path.split("/")[-1].replace(".traj.json", "")
            episodes.append(row)
    if not episodes:
        return {"episodes": 0}

    def mean(key):
        vals = [e[key] for e in episodes if e.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    stage_modal = [Counter(e["modal_by_stage"][i] for e in episodes
                           if len(e.get("modal_by_stage", [])) > i).most_common(3)
                   for i in range(3)]
    return {
        "episodes": len(episodes),
        "modal_roles_by_stage": stage_modal,
        "mean_reselection_rate": mean("reselection_rate"),
        "mean_subscription_specificity": mean("specificity"),
        "mean_action_agreement": mean("agreement"),
        "mean_faulty_rate": mean("faulty_rate"),
        "mean_withholdings": mean("withholdings"),
        "per_episode": episodes,
    }


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else "swe_runs/*/*/*.traj.json"
    out = summarize(pattern)
    if not out["episodes"]:
        sys.exit(f"no trajectories matched {pattern!r}")
    print(f"{out['episodes']} episodes")
    for i, stage in enumerate(("early", "mid", "late")):
        print(f"  modal roles ({stage}): {out['modal_roles_by_stage'][i]}")
    print(f"  reselection rate:            {out['mean_reselection_rate']}")
    print(f"  subscription specificity:    {out['mean_subscription_specificity']}")
    print(f"  subscription-action agreement: {out['mean_action_agreement']}")
    if out["mean_faulty_rate"] is not None:
        print(f"  publications judged faulty:  {out['mean_faulty_rate']}")
        print(f"  hosts withheld per episode:  {out['mean_withholdings']}")
