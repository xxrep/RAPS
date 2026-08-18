"""Open-membership evaluation (Fig. 4a-c): membership changes mid-episode.

Three modes, all driven through RAPSConfig.membership_hook — the coordination
loop is untouched; the hook only calls coordinator.depart/arrive:

  churn     each worker departs after each round with probability --leave_prob
  targeted  at the midpoint round, remove the agent that has received the most
            publications (approximated by its ingested-history length)
  newcomer  at round --arrive_at, add a fresh specialist discovered purely by
            its declared subscription; its subsequent engagement is measured

Accuracy under churn is reported next to a static control run of the same
episodes, giving the retention ratio of Fig. 4a-c.

Example:
    python experiments/run_membership.py --mode churn --leave_prob 0.25 --limit 15
    python experiments/run_membership.py --mode targeted --limit 15
    python experiments/run_membership.py --mode newcomer --arrive_at 2 --limit 15
"""
import sys
import os
import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.utils.const import RAPS_ROOT
from RAPS.utils.globals import Time, Cost
from RAPS.agents.agent_registry import AgentRegistry
from RAPS.core import RAPSCoordinator
from RAPS.config import BENCHMARKS, protocol_config
from RAPS.agents.math_solver import MathAgent
from raps_data.gsm8k_dataset import gsm_data_process, gsm_get_predict, load_gsm8k_jsonl
from experiments.run_math import initialize_agents_from_set, initialize_final_answerer, is_correct


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str):
    print(f"[{_ts()}] {message}")


NEWCOMER = dict(
    role="Arithmetic Specialist",
    capabilities="You are an arithmetic specialist. Recompute numeric results carefully, "
                 "step by step, and flag calculation errors. End with 'The answer is X'.",
    interests="recomputing arithmetic, verifying intermediate numeric results, totals and rates",
)


def make_hook(args, rng):
    if args.mode == "churn":
        def hook(step, coord):
            for a in list(coord.agents):
                if a is not coord.final_answerer and rng.random() < args.leave_prob:
                    coord.depart(a.id)
        return hook

    if args.mode == "targeted":
        mid = max(1, args.max_steps // 2)

        def hook(step, coord):
            if step != mid:
                return
            workers = [a for a in coord.agents if a is not coord.final_answerer]
            if workers:
                busiest = max(workers, key=lambda a: len(a.history))
                coord.depart(busiest.id)
        return hook

    # newcomer
    def hook(step, coord):
        if step != args.arrive_at:
            return
        spec = BENCHMARKS["gsm8k"]
        agent = AgentRegistry.get(
            spec.agent_class, id=None, llm_name=args.llm_name, domain="gsm8k",
            role=NEWCOMER["role"], capabilities=NEWCOMER["capabilities"],
            interests=NEWCOMER["interests"], additional_instructions="", few_shot="")
        coord.arrive(agent)
    return hook


def run_episodes(dataset, args, hook=None):
    agents = initialize_agents_from_set(args.llm_name, "gsm8k")
    final_answerer = initialize_final_answerer(args.llm_name, "gsm8k")
    config = protocol_config("gsm8k", max_steps=args.max_steps, entry_index=args.entry_index,
                             membership_hook=hook)
    coordinator = RAPSCoordinator(agents, final_answerer, config,
                                  logger=lambda m: None, answer_extractor=gsm_get_predict)
    solved = 0
    newcomer_received = 0
    for example in dataset:
        result = coordinator.run(example["task"])
        solved += 1 if is_correct(gsm_get_predict(result.final_output), example.get("answer"), "gsm8k") else 0
        if args.mode == "newcomer":
            for step in result.task_log["steps"]:
                for dec in step.get("broker_decisions", []):
                    newcomer_received += sum(1 for r in dec.get("receivers", [])
                                             if NEWCOMER["role"] in r)
    n = len(dataset)
    return {"accuracy": solved / n if n else 0.0, "solved": solved, "n": n,
            "newcomer_received": newcomer_received}


def parse_args():
    p = argparse.ArgumentParser(description="RAPS open-membership evaluation")
    p.add_argument("--mode", choices=["churn", "targeted", "newcomer"], required=True)
    p.add_argument("--dataset_json", type=str, default="raps_data/gsm8k/gsm8k.jsonl")
    p.add_argument("--llm_name", type=str, default="gpt-4o-mini-2024-07-18")
    p.add_argument("--max_steps", type=int, default=5)
    p.add_argument("--entry_index", type=int, default=1)
    p.add_argument("--leave_prob", type=float, default=0.25)
    p.add_argument("--arrive_at", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=10**9,
                   help="questions to evaluate, defaulting to the whole set")
    return p.parse_args()


def main():
    args = parse_args()
    Time.instance().value = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    dataset = gsm_data_process(load_gsm8k_jsonl(args.dataset_json))[args.start:args.start + args.limit]

    rng = random.Random(args.seed)
    hook = make_hook(args, rng)
    _log(f"Membership mode={args.mode} on {len(dataset)} questions")
    test = run_episodes(dataset, args, hook=hook)

    summary = {"mode": args.mode, "test": test}
    if args.mode in ("churn", "targeted"):
        control = run_episodes(dataset, args, hook=None)
        retention = test["accuracy"] / control["accuracy"] if control["accuracy"] else 0.0
        summary["control"] = control
        summary["retention"] = round(retention, 4)
        _log(f"test acc={test['accuracy']:.3f} | control acc={control['accuracy']:.3f} "
             f"| retention={retention:.3f}")
    else:
        _log(f"test acc={test['accuracy']:.3f} | newcomer engagements={test['newcomer_received']}")

    out_dir = Path(f"{RAPS_ROOT}/result/membership")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"membership_{args.mode}_{Time.instance().value}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=2)
    _log(f"Saved -> {out_file} | Total cost: ${Cost.instance().value:.4f}")


if __name__ == "__main__":
    main()
