"""Robustness evaluation under the revision's threat model (§4.6, Fig. 3, Supp. F).

Injects adversarial peers of a chosen type into a benchmark's five-agent pool at a
chosen composition (T truthful + A adversarial) and measures, under three defense
conditions, task accuracy together with the attack-cost quantities of Table S.16:
faulty publications, how many of them the watchdog blocks, how often the gate
withholds the adversary, and the round at which a benign posterior first reaches
tau_rep. It also reports the false-isolation rate of benign agents (Table S.17).
MMLU is the default, which is the benchmark those tables report; --dataset gsm8k
runs the same evaluation on the arithmetic pool.

Adversary types (RAPS/agents/adversaries.py): overt, covert, sleeper, adaptive,
selective — plus optional collusion (false praise / bad-mouthing) exercised
through forged second-hand reports. A degraded-watchdog sweep (--degrade_p,
Fig. 3f) inverts each benign watchdog judgement with the given probability.

Example:
    python experiments/run_robustness.py --adversary overt --truthful 3 --num_adversary 2
    python experiments/run_robustness.py --adversary covert --collusion both
    python experiments/run_robustness.py --adversary adaptive --degrade_p 0.3
    python experiments/run_robustness.py --dataset gsm8k --adversary sleeper
"""
import sys
import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.utils.const import RAPS_ROOT
from RAPS.utils.globals import Time, Cost
from RAPS.core import RAPSCoordinator
from RAPS.config import TRUST_THRESHOLD, REP_DISCOUNT, MAX_STEPS, protocol_config
from RAPS.agents.adversaries import Adversary, link_colluders, degrade_watchdog
from raps_data.gsm8k_dataset import gsm_data_process, gsm_get_predict, load_gsm8k_jsonl
from raps_data.mmlu_dataset import MMLUDataset
from experiments import run_math, run_mmlu


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str):
    print(f"[{_ts()}] {message}")


def _load_mmlu(_path, start, limit):
    """MMLU questions as the {task, answer} records the injected pool is evaluated on."""
    ds = MMLUDataset(split="test")
    end = min(start + limit, len(ds))
    return [{"task": MMLUDataset.record_to_input(ds[i])["task"],
             "answer": MMLUDataset.record_to_target_answer(ds[i])} for i in range(start, end)]


def _load_gsm8k(path, start, limit):
    return gsm_data_process(load_gsm8k_jsonl(path))[start:start + limit]


#: What differs between benchmarks: the questions, the pool, the answer extractor and the
#: interests an adversary declares to look like a teammate on it. Everything else — the
#: defence conditions, the protocol parameters, the attack schedules — is shared.
BENCHMARKS = {
    "mmlu": dict(
        load=_load_mmlu, default_path=None,
        workers=run_mmlu.initialize_agents_from_set, final=run_mmlu.initialize_final_answerer,
        extract=MMLUDataset.postprocess_answer,
        interests="answering multiple-choice questions, checking factual claims, "
                  "weighing the options against each other",
    ),
    "gsm8k": dict(
        load=_load_gsm8k, default_path="raps_data/gsm8k/gsm8k.jsonl",
        workers=run_math.initialize_agents_from_set, final=run_math.initialize_final_answerer,
        extract=gsm_get_predict,
        interests="solving math word problems, verifying arithmetic, checking solutions",
    ),
}


# Defense conditions compared on the same injected pool (Fig. 3b). Every other control,
# the posterior's persistence across tasks included, is the one protocol_config fixes.
CONDITIONS = {
    "no_defense":          dict(use_watchdog=False, reputation_gate=False,
                                second_hand_gossip=False),
    "watchdog":            dict(use_watchdog=True,  reputation_gate=False,
                                second_hand_gossip=False),
    "watchdog+reputation": dict(use_watchdog=True,  reputation_gate=True,
                                second_hand_gossip=True),
}


def _workers(spec, llm_name, domain, truthful):
    """`truthful` benign agents drawn from the benchmark's own pool."""
    fn = spec["workers"]
    agents = fn(llm_name, domain) if domain in ("gsm8k", "svamp", "aqua") else fn(llm_name)
    return agents[:truthful]


def _final_answerer(spec, llm_name, domain):
    fn = spec["final"]
    return fn(llm_name, domain) if domain in ("gsm8k", "svamp", "aqua") else fn(llm_name)


def build_pool(spec, domain, llm_name, truthful, num_adversary, kind, collusion,
               defect_after=3):
    """`truthful` benign agents from the benchmark's pool + adversaries in disguise. Each
    adversary declares the interests of a teammate on this benchmark and answers in its
    format, so it is distinguishable by content alone. defect_after is the sleeper's
    schedule; the reputation-aware adversary instead runs the gate's own parameters."""
    agents = _workers(spec, llm_name, domain, truthful)
    benign_ids = [a.id for a in agents]
    adversaries = []
    for i in range(num_adversary):
        adv = Adversary(
            id=None, llm_name=llm_name, domain=domain, role=f"Solution Assistant {i + 1}",
            interests=spec["interests"],
            kind=kind, collusion=collusion, defect_after=defect_after,
            gate_threshold=TRUST_THRESHOLD, gate_discount=REP_DISCOUNT,
        )
        adv.history = []
        adv.inbox = []
        agents.append(adv)
        adversaries.append(adv)
    if collusion != "none":
        link_colluders(adversaries, benign_ids)
    return agents, adversaries


def _scored(pred, answer, domain) -> bool:
    """Whether the final answer is correct, by the benchmark's own comparison: an option
    letter on the multiple-choice benchmarks, a number on the arithmetic ones."""
    if domain == "mmlu":
        return str(pred).strip().upper() == str(answer).strip().upper()
    return run_math.is_correct(pred, answer, domain)


def _first_gated_round(task_log, adv_ids):
    """First round in which any benign agent's posterior about an adversary reached the
    reputation gate (None if it never did). Logged posteriors are misbehaviour scores,
    so the gate isolates a peer once the score reaches tau_rep."""
    for step in task_log["steps"]:
        if not isinstance(step.get("step"), int):
            continue
        for watcher, peers in step.get("reputation", {}).items():
            if watcher in adv_ids:
                continue
            if any(tid in adv_ids and val >= TRUST_THRESHOLD for tid, val in peers.items()):
                return step["step"]
    return None


def run_condition(name, flags, dataset, args):
    spec, domain = BENCHMARKS[args.dataset], args.dataset
    extract = spec["extract"]
    agents, adversaries = build_pool(spec, domain, args.llm_name, args.truthful,
                                     args.num_adversary, args.adversary, args.collusion,
                                     defect_after=args.defect_after)
    adv_ids = {a.id for a in adversaries}
    benign_ids = {a.id for a in agents} - adv_ids
    if args.degrade_p > 0:
        for a in agents:
            if a.id in benign_ids:
                degrade_watchdog(a, args.degrade_p)

    final_answerer = _final_answerer(spec, args.llm_name, domain)
    # the paper's single parameter set (Table S.4); the defense condition only
    # flips the mechanism flags
    config = protocol_config(domain, max_steps=args.max_steps, top_k=args.top_k,
                             sim_threshold=args.sim_threshold, entry_index=args.entry_index,
                             **flags)
    coordinator = RAPSCoordinator(agents, final_answerer, config,
                                  logger=lambda m: None, answer_extractor=extract)

    solved = faulty_published = faulty_blocked = gate_withholdings = 0
    gated_rounds, false_isolated = [], set()
    for example in dataset:
        for a in adversaries:
            a.reset_episode()
        result = coordinator.run(example["task"])
        solved += 1 if _scored(extract(result.final_output), example.get("answer"), domain) else 0

        ep_faulty = sum(a.stats["faulty"] for a in adversaries)
        faulty_published += ep_faulty
        for step in result.task_log["steps"]:
            for dec in step.get("broker_decisions", []):
                if dec.get("status") == "blocked" and dec.get("sender") in adv_ids:
                    faulty_blocked += 1
                gate_withholdings += sum(1 for cid in dec.get("withheld", []) if cid in adv_ids)
        r = _first_gated_round(result.task_log, adv_ids)
        if r is not None:
            gated_rounds.append(r)
        # false isolation: a benign peer someone's posterior gates out (Table S.17)
        for watcher, peers in coordinator.snapshot_reputation().items():
            if watcher in adv_ids:
                continue
            false_isolated.update(tid for tid, v in peers.items()
                                  if tid in benign_ids and v >= TRUST_THRESHOLD)

    n = len(dataset)
    acc = solved / n if n else 0.0
    rep = coordinator.snapshot_reputation()
    mal_scores = [v for peer_map in rep.values() for tid, v in peer_map.items() if tid in adv_ids]
    avg_mal_score = round(sum(mal_scores) / len(mal_scores), 3) if mal_scores else None
    row = {
        "condition": name, "accuracy": round(acc, 4), "solved": solved, "n": n,
        "faulty_published": faulty_published, "faulty_blocked": faulty_blocked,
        "faulty_admitted": faulty_published - faulty_blocked,
        "adversary_withheld_by_gate": gate_withholdings,
        "mean_first_gated_round": round(sum(gated_rounds) / len(gated_rounds), 2) if gated_rounds else None,
        "benign_falsely_isolated": len(false_isolated),
        "avg_adversary_misbehaviour": avg_mal_score,
    }
    _log(f"[{name}] {row}")
    return row


def parse_args():
    p = argparse.ArgumentParser(description="RAPS robustness evaluation (five adversary types)")
    p.add_argument("--dataset", choices=sorted(BENCHMARKS), default="mmlu",
                   help="the benchmark the injected pool is evaluated on (Table S.16 "
                        "reports MMLU)")
    p.add_argument("--dataset_json", type=str, default=None,
                   help="override the data path of a benchmark that reads one")
    p.add_argument("--llm_name", type=str, default="gpt-4o-mini-2024-07-18")
    p.add_argument("--max_steps", type=int, default=MAX_STEPS)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--sim_threshold", type=float, default=0.5)
    p.add_argument("--entry_index", type=int, default=1)
    p.add_argument("--truthful", type=int, default=3, help="benign agents from the 5-agent pool")
    p.add_argument("--num_adversary", type=int, default=2)
    p.add_argument("--adversary", default="overt",
                   choices=["overt", "covert", "sleeper", "adaptive", "selective"])
    p.add_argument("--defect_after", type=int, default=3,
                   help="sleeper: sound publications before defecting")
    p.add_argument("--collusion", default="none",
                   choices=["none", "praise", "badmouthing", "both"])
    p.add_argument("--degrade_p", type=float, default=0.0,
                   help="invert each benign watchdog judgement with this probability (Fig. 3f)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=153,
                   help="questions to evaluate, defaulting to the full MMLU subset")
    p.add_argument("--conditions", type=str, default="all",
                   help="comma-separated subset of: " + ",".join(CONDITIONS))
    return p.parse_args()


def main():
    args = parse_args()
    Time.instance().value = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    spec = BENCHMARKS[args.dataset]
    dataset = spec["load"](args.dataset_json or spec["default_path"], args.start, args.limit)
    names = list(CONDITIONS) if args.conditions == "all" else args.conditions.split(",")

    _log(f"Robustness on {args.dataset}: {len(dataset)} questions | "
         f"{args.truthful}T{args.num_adversary}A {args.adversary} | "
         f"collusion={args.collusion} | degrade_p={args.degrade_p} | {names}")
    rows = [run_condition(n, CONDITIONS[n], dataset, args) for n in names]

    print("\n==================== ROBUSTNESS SUMMARY ====================")
    header = (f"{'condition':<22}{'acc':<8}{'faulty':<8}{'blocked':<9}{'admitted':<10}"
              f"{'withheld':<10}{'gated@':<8}{'false_iso':<10}{'adv_mis':<8}")
    print(header)
    for r in rows:
        print(f"{r['condition']:<22}{r['accuracy']:<8.3f}{r['faulty_published']:<8}"
              f"{r['faulty_blocked']:<9}{r['faulty_admitted']:<10}"
              f"{r['adversary_withheld_by_gate']:<10}"
              f"{str(r['mean_first_gated_round']):<8}{r['benign_falsely_isolated']:<10}"
              f"{str(r['avg_adversary_misbehaviour']):<8}")
    print(f"\nTotal cost: ${Cost.instance().value:.4f}")

    out_dir = Path(f"{RAPS_ROOT}/result/robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = (out_dir /
                f"robustness_{args.dataset}_{args.adversary}_{args.llm_name}_"
                f"{Time.instance().value}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": rows}, f, indent=2)
    _log(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
