"""Entry point of the compared methods (Tables 1-2); mirrors the flags of the RAPS
runners (experiments/) so a baseline row differs from a RAPS row only in the
coordination protocol.

    python baselines/run.py --method single --dataset gsm8k --limit 100
    python baselines/run.py --method gptswarm --dataset mmlu --search_budget 200
    python baselines/run.py --method dylan --dataset mmlu --adversary covert --num_adversary 2
"""
import argparse
import importlib
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from baselines import common

METHODS = ("single", "debate", "dylan", "gptswarm", "aflow", "puppeteer")
PEERLESS = ("single", "aflow")   # no peer participants for an adversary to join
POOL_METHODS = ("debate", "dylan", "gptswarm")   # methods coordinating the shared pool
ADVERSARIES = ("overt", "covert", "sleeper", "adaptive", "selective")


def parse_args():
    p = argparse.ArgumentParser(description="Compared methods on the RAPS benchmarks")
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--dataset", required=True, choices=sorted(common.BENCHMARKS))
    p.add_argument("--dataset_json", default=None,
                   help="override the benchmark's default data path")
    p.add_argument("--llm_name", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=10**9)
    p.add_argument("--adversary", default="none", choices=("none",) + ADVERSARIES)
    p.add_argument("--num_adversary", type=int, default=1)
    p.add_argument("--defect_after", type=int, default=3,
                   help="sleeper: sound publications before defecting")
    p.add_argument("--search_budget", type=int, default=None,
                   help="optimization iterations on the held-out slice; defaults to "
                        "each method's declared value (aflow: 20, gptswarm: 0)")
    p.add_argument("--search_pool", type=int, default=100,
                   help="held-out records the search draws from")
    p.add_argument("--policy", default="llm", choices=("llm", "random"),
                   help="puppeteer orchestrator variant")
    p.add_argument("--rounds", type=int, default=None,
                   help="communication rounds (debate/dylan)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.adversary != "none" and args.method in PEERLESS:
        raise SystemExit(f"--adversary does not apply to {args.method}: "
                         "no peer participants")
    Time.instance().value = (Time.instance().value
                             or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    bench = common.BENCHMARKS[args.dataset]
    records = bench.load(args.dataset_json, args.start, args.limit)

    dev_records = None
    dev_split = "none"
    if args.search_budget or (args.method == "aflow" and args.search_budget is None):
        if bench.load_dev:
            dev_records = bench.load_dev(args.dataset_json, args.search_pool)
            dev_split = "val"
        else:
            dev_records = bench.load(args.dataset_json,
                                     args.start + len(records), args.search_pool)
            dev_split = "post-slice"
        if not dev_records:
            print("[warn] no held-out records available; the search budget is spent "
                  "on the evaluation slice")
            dev_records = records
            dev_split = "eval-fallback"

    adversary = None
    if args.adversary != "none":
        adversary = {"kind": args.adversary, "count": args.num_adversary,
                     "defect_after": args.defect_after}
    pool = (common.build_pool(args.dataset, args.llm_name, adversary)
            if args.method in POOL_METHODS else None)

    module = importlib.import_module(f"baselines.{args.method}")
    solver = module.build(bench=bench, llm_name=args.llm_name, pool=pool,
                          adversary=adversary, search_budget=args.search_budget,
                          dev_records=dev_records, policy=args.policy,
                          rounds=args.rounds, seed=args.seed)

    print(f"[run] {args.method} on {args.dataset}: {len(records)} tasks")
    summary = common.evaluate(solver, bench, records)
    summary["config"] = {**vars(args), "dev_split": dev_split}
    path = common.save_result(args.method, args.dataset, args.llm_name, summary)

    print(f"Accuracy: {summary['accuracy']:.4f} ({summary['solved']}/{summary['n']})")
    if "attack" in summary:
        print(f"Attack: {summary['attack']}")
    print(f"Total cost: ${Cost.instance().value:.4f} | "
          f"PromptTokens: {PromptTokens.instance().value} | "
          f"CompletionTokens: {CompletionTokens.instance().value}")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
