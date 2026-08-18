"""Watchdog quality and reputation calibration from task logs (Supp. F.3, Tables S.18/S.19).

Scores the local watchdog as an evaluator against an objective outcome label:
a publication is labelled SOUND when its extracted answer equals the gold
answer stored in the task log, FAULTY otherwise. The watchdog's verdict is
read from the log's broker decisions (blocked = judged faulty).

  * Table S.18: agreement / precision / recall / false-rejection rate
    (positive class = faulty publication).
  * Table S.19: calibration — publications bucketed by the mean posterior
    the team held about their sender at that round, against realized
    reliability (fraction actually sound).

Usage:
    python analysis/watchdog_eval.py "logs/log_gsm8k_*.json" --domain gsm8k
"""
import argparse
from collections import defaultdict
from typing import Any, Callable, Dict, List

from log_io import load_task_logs
from raps_data.gsm8k_dataset import gsm_get_predict
from raps_data.aqua_dataset import aqua_get_predict

EXTRACTORS: Dict[str, Callable[[Any], str]] = {
    "gsm8k": gsm_get_predict,
    "svamp": gsm_get_predict,
    "aqua": aqua_get_predict,
    "mmlu": aqua_get_predict,   # same letter-after-'answer is' extraction, A-D subset
}


def _norm(pred: str, domain: str) -> str:
    return str(pred).strip().upper() if domain in ("aqua", "mmlu") else str(pred).strip()


def collect(task_logs: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
    """One record per publication: {sound, blocked, posterior}."""
    extract = EXTRACTORS[domain]
    records = []
    for log in task_logs:
        answer = log.get("answer")
        if answer is None:
            continue
        for step in log.get("steps", []):
            if not isinstance(step.get("step"), int):
                continue
            blocked_senders = {d.get("sender") for d in step.get("broker_decisions", [])
                               if d.get("status") == "blocked"}
            snap = step.get("reputation", {})
            for pub in step.get("publications", []):
                sender = pub.get("sender_id")
                sound = _norm(extract(pub.get("content", "")), domain) == _norm(answer, domain)
                posteriors = [peers[sender] for peers in snap.values() if sender in peers]
                records.append({
                    "sound": sound,
                    "blocked": sender in blocked_senders,
                    "posterior": sum(posteriors) / len(posteriors) if posteriors else None,
                })
    return records


def quality(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Table S.18 metrics (positive class = faulty; predicted faulty = blocked)."""
    tp = sum(1 for r in records if r["blocked"] and not r["sound"])
    fp = sum(1 for r in records if r["blocked"] and r["sound"])
    fn = sum(1 for r in records if not r["blocked"] and not r["sound"])
    tn = sum(1 for r in records if not r["blocked"] and r["sound"])
    n = max(1, len(records))

    def safe(num, den):
        return round(num / den, 4) if den else None

    return {"n": len(records), "agreement": round((tp + tn) / n, 4),
            "precision": safe(tp, tp + fp), "recall": safe(tp, tp + fn),
            "false_rejection": safe(fp, fp + tn)}


BUCKETS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]


def calibration(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Table S.19: realized reliability per posterior-expectation bucket."""
    rows = []
    for lo, hi in BUCKETS:
        group = [r for r in records if r["posterior"] is not None and lo <= r["posterior"] < hi]
        if not group:
            continue
        rows.append({
            "bucket": f"[{lo},{min(hi, 1.0)})",
            "publications": len(group),
            "mean_score": round(sum(r["posterior"] for r in group) / len(group), 3),
            "realized_reliability": round(sum(1 for r in group if r["sound"]) / len(group), 3),
        })
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--domain", choices=sorted(EXTRACTORS), default="gsm8k")
    args = ap.parse_args()
    logs = load_task_logs(args.pattern)
    if not logs:
        raise SystemExit(f"no task logs matched {args.pattern!r}")
    recs = collect(logs, args.domain)
    q = quality(recs)
    print(f"{len(logs)} task logs, {q['n']} publications | agreement={q['agreement']} "
          f"precision={q['precision']} recall={q['recall']} false_rejection={q['false_rejection']}")
    print(f"{'bucket':<12}{'publications':>14}{'mean_score':>12}{'realized_reliability':>22}")
    for row in calibration(recs):
        print(f"{row['bucket']:<12}{row['publications']:>14}{row['mean_score']:>12}"
              f"{row['realized_reliability']:>22}")
