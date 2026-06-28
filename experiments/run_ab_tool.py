"""
A/B: Program-Aided computation (tool_verify) vs baseline on GSM8K, deterministic (temp=0).

baseline    : reasoning ladder only (current default).
tool_verify : same ladder, but the final numeric answer is computed by EXECUTING Python
              (the Programming Expert's real job), then handed to the Final Answerer.
"""
import sys
import os
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.utils.globals import Time, Cost
from RAPS.core import RAPSCoordinator, RAPSConfig
from raps_data.gsm8k_dataset import gsm_data_process, gsm_get_predict
from experiments.run_gsm8k import (initialize_agents_from_set, initialize_final_answerer,
                                   is_correct, load_jsonl)


def run_condition(tool, dataset, llm_name, max_steps, top_k):
    agents = initialize_agents_from_set(llm_name)
    fa = initialize_final_answerer(llm_name)
    cfg = RAPSConfig(domain="gsm8k", max_steps=max_steps, top_k=top_k, entry_index=1,
                     tool_verify=tool)
    coord = RAPSCoordinator(agents, fa, cfg, logger=lambda m: None, answer_extractor=gsm_get_predict)
    c0 = Cost.instance().value
    solved = 0
    wrong = []
    for i, ex in enumerate(dataset):
        res = coord.run(ex["task"])
        ok = is_correct(gsm_get_predict(res.final_output), ex.get("answer"))
        solved += 1 if ok else 0
        if not ok:
            wrong.append(i)
    return {"acc": solved / len(dataset), "solved": solved, "n": len(dataset),
            "cost": Cost.instance().value - c0, "wrong": wrong}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm_name", type=str, default="gpt-4o-mini-2024-07-18")
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=3)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--dataset_json", type=str, default="raps_data/gsm8k/gsm8k.jsonl")
    args = p.parse_args()
    Time.instance().value = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    data = gsm_data_process(load_jsonl(args.dataset_json))[args.start:args.start + args.limit]
    print(f"A/B tool_verify on GSM8K | {len(data)} questions | top_k={args.top_k} max_steps={args.max_steps}")

    base = run_condition(False, data, args.llm_name, args.max_steps, args.top_k)
    print(f"[baseline]    acc={base['acc']:.3f} ({base['solved']}/{base['n']}) cost=${base['cost']:.4f}")
    tool = run_condition(True, data, args.llm_name, args.max_steps, args.top_k)
    print(f"[tool_verify] acc={tool['acc']:.3f} ({tool['solved']}/{tool['n']}) cost=${tool['cost']:.4f}")

    fixed = sorted(set(base["wrong"]) - set(tool["wrong"]))
    broke = sorted(set(tool["wrong"]) - set(base["wrong"]))
    print("\n==================== A/B (tool_verify) ====================")
    print(f"{'metric':<16}{'baseline':<14}{'tool_verify':<14}")
    print(f"{'accuracy':<16}{base['acc']:<14.3f}{tool['acc']:<14.3f}")
    print(f"{'cost ($)':<16}{base['cost']:<14.4f}{tool['cost']:<14.4f}")
    print(f"\ntool_verify FIXED {len(fixed)} baseline-failures {fixed}; newly BROKE {len(broke)} {broke}")


if __name__ == "__main__":
    main()
