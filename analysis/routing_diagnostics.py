"""Routing and subscription diagnostics from task logs (Supp. E.1/E.2, Fig. S.1).

Per communication round, over a run of episodes:
  * routing structure — edge density, in-degree concentration (Herfindahl),
    edge persistence between consecutive rounds, distinct receiving roles;
  * reactive subscription — update rate (fraction of agent-rounds whose
    refined subscription differs from the previous one);
  * selectivity — the fraction of a publisher's peers that receives from it,
    against 1.0 under broadcast (Table S.14);
  * admitted share — publications a receiver admits rather than the local check
    blocking, and how often the reputation gate withheld a candidate.

Usage:
    python analysis/routing_diagnostics.py "logs/log_mmlu_*.json"
"""
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from log_io import load_task_logs


def round_edges(step: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """(sender, receiver) pairs the broker produced in one round."""
    edges = set()
    for dec in step.get("broker_decisions", []):
        sender = dec.get("sender")
        for r in dec.get("receivers", []):
            edges.add((sender, r))
    return edges


def structure_by_round(task_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Structural quantities of Fig. S.1d for each round of one episode."""
    n_agents = len({a for s in task_log.get("steps", [])
                    if isinstance(s.get("step"), int) for a in s.get("active_agents", [])})
    rows = []
    prev = None
    for step in task_log.get("steps", []):
        if not isinstance(step.get("step"), int):
            continue
        edges = round_edges(step)
        possible = max(1, n_agents * (n_agents - 1))
        density = len(edges) / possible
        indeg = Counter(r for _, r in edges)
        total = max(1, sum(indeg.values()))
        concentration = sum((c / total) ** 2 for c in indeg.values())
        persistence = (len(edges & prev) / len(edges | prev)) if prev is not None and (edges | prev) else None
        rows.append({
            "round": step["step"], "edges": len(edges), "density": round(density, 4),
            "in_degree_concentration": round(concentration, 4),
            "edge_persistence": None if persistence is None else round(persistence, 4),
            "receiving_roles": len(indeg),
        })
        prev = edges
    return rows


def pool_size(task_log: Dict[str, Any]) -> int:
    """The hosts routing could engage, from the team roster when present, else the agents
    the episode engaged. Selectivity is a fraction of a publisher's peers, so the roster
    sets the denominator rather than the agents that happened to act."""
    roster = task_log.get("team", {}).get("roster")
    if roster:
        return len(roster)
    return len({a for s in task_log.get("steps", []) if isinstance(s.get("step"), int)
                for a in s.get("active_agents", [])})


def selectivity(task_log: Dict[str, Any]) -> List[float]:
    """Table S.14, first property: the fraction of a publisher's peers that receives from
    it, one value per publication that was routed."""
    peers = max(1, pool_size(task_log) - 1)
    out = []
    for step in task_log.get("steps", []):
        for dec in step.get("broker_decisions", []):
            if "receivers" in dec:
                out.append(len(dec["receivers"]) / peers)
    return out


def admitted_share(task_log: Dict[str, Any]) -> Dict[str, int]:
    """Table S.12, last columns: publications a receiver admits against those the local
    check blocked, and how often the reputation gate withheld a candidate."""
    published = blocked = withheld = 0
    for step in task_log.get("steps", []):
        if not isinstance(step.get("step"), int):
            continue
        published += len(step.get("publications", []))
        for dec in step.get("broker_decisions", []):
            blocked += dec.get("status") == "blocked"
            withheld += len(dec.get("withheld") or [])
    return {"published": published, "blocked": blocked, "withheld": withheld}


def subscription_updates(task_log: Dict[str, Any]) -> Dict[str, int]:
    """Update rate of reactive subscription (Table S.12, first column)."""
    changed = total = 0
    last: Dict[str, str] = {}
    for step in task_log.get("steps", []):
        for agent_id, ref in step.get("refinements", {}).items():
            post = ref.get("post_refinement", "")
            if agent_id in last:
                total += 1
                changed += post != last[agent_id]
            last[agent_id] = post
    return {"changed": changed, "total": total}


def summarize(task_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_round = defaultdict(lambda: {"density": [], "in_degree_concentration": [],
                                    "edge_persistence": [], "receiving_roles": []})
    changed = total = 0
    selective, admitted = [], {"published": 0, "blocked": 0, "withheld": 0}
    for log in task_logs:
        selective.extend(selectivity(log))
        for key, value in admitted_share(log).items():
            admitted[key] += value
        for row in structure_by_round(log):
            by_round[row["round"]]["density"].append(row["density"])
            by_round[row["round"]]["in_degree_concentration"].append(row["in_degree_concentration"])
            if row["edge_persistence"] is not None:
                by_round[row["round"]]["edge_persistence"].append(row["edge_persistence"])
            by_round[row["round"]]["receiving_roles"].append(row["receiving_roles"])
        u = subscription_updates(log)
        changed += u["changed"]
        total += u["total"]

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    published = max(1, admitted["published"])
    return {
        "rounds": {r: {k: mean(v) for k, v in sorted(vals.items())}
                   for r, vals in sorted(by_round.items())},
        "subscription_update_rate": round(changed / total, 4) if total else None,
        "selectivity": mean(selective),
        "admitted_share": round(1 - admitted["blocked"] / published, 4),
        "gate_withholdings": admitted["withheld"],
    }


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else "logs/log_*.json"
    logs = load_task_logs(pattern)
    if not logs:
        sys.exit(f"no task logs matched {pattern!r}")
    out = summarize(logs)
    print(f"{len(logs)} task logs")
    print(f"{'round':<7}{'density':>9}{'indeg_conc':>12}{'persistence':>13}{'recv_roles':>12}")
    for r, vals in out["rounds"].items():
        cells = [str(vals[k]) for k in ("density", "in_degree_concentration",
                                        "edge_persistence", "receiving_roles")]
        print(f"{r:<7}{cells[0]:>9}{cells[1]:>12}{cells[2]:>13}{cells[3]:>12}")
    print(f"subscription update rate: {out['subscription_update_rate']}")
    print(f"selectivity:              {out['selectivity']}")
    print(f"admitted share:           {out['admitted_share']}")
    print(f"gate withholdings:        {out['gate_withholdings']}")
