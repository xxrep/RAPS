"""
A/B validation of the prompt improvements on GSM8K.

baseline : original prompts (refine can bake in a solution path; broker defaults to a verifier)
improved : persona-only refinement + gap-driven broker + an independent re-deriving Inspector

Reports, on the SAME questions:
  - accuracy
  - solution-leak rate in refined prompts (should drop to ~0 under 'improved')
  - Programming Expert engagement (exact computation; should rise under 'improved')
  - cost
"""
import sys
import os
import re
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.utils.globals import Time, Cost
from RAPS.core import RAPSCoordinator, RAPSConfig
from raps_data.gsm8k_dataset import gsm_data_process, gsm_get_predict
from experiments.run_gsm8k import (initialize_agents_from_set, initialize_final_answerer,
                                   is_correct, load_jsonl)

INSPECTOR_IMPROVED = (
    "You are an independent Inspector. Do NOT restate or paraphrase the prior solution. "
    "Re-derive the answer YOURSELF from the original problem using a DIFFERENT method than the "
    "previous agent (recompute each arithmetic step independently, or reason via explicit equations). "
    "If your result differs from the prior agent's, explicitly flag the discrepancy and state which "
    "is correct and why. The last line of your output contains only the final result without any "
    "units, for example: The answer is 140"
)

_LEAK = re.compile(r"=\s*\$?\d|calculate the|\$\d|\d+\s*[×x*]\s*\d")


def refined_prompts(task_log):
    out = []
    for st in task_log.get("steps", []):
        for _, r in st.get("refinements", {}).items():
            if isinstance(r, dict):
                out.append(str(r.get("post_refinement", "")))
    return out


def run_condition(improved, dataset, llm_name, max_steps, top_k):
    agents = initialize_agents_from_set(llm_name)
    if improved:
        for a in agents:
            if a.role == "Inspector":
                a.system_prompt = INSPECTOR_IMPROVED
                a.refined_prompt = INSPECTOR_IMPROVED
    final_answerer = initialize_final_answerer(llm_name)
    cfg = RAPSConfig(domain="gsm8k", max_steps=max_steps, top_k=top_k, entry_index=1,
                     improved_prompts=improved)
    coord = RAPSCoordinator(agents, final_answerer, cfg, logger=lambda m: None,
                            answer_extractor=gsm_get_predict)

    c0 = Cost.instance().value
    solved = leak = leak_tot = prog = 0
    wrong_idx = []
    for i, ex in enumerate(dataset):
        res = coord.run(ex["task"])
        ok = is_correct(gsm_get_predict(res.final_output), ex.get("answer"))
        solved += 1 if ok else 0
        if not ok:
            wrong_idx.append(i)
        for t in refined_prompts(res.task_log):
            leak_tot += 1
            leak += 1 if _LEAK.search(t.lower()) else 0
        for p in res.publications:
            if p["role"] == "Programming Expert":
                prog += 1
    cost = Cost.instance().value - c0
    return {"solved": solved, "n": len(dataset), "acc": solved / len(dataset),
            "leak": leak, "leak_tot": leak_tot, "leak_rate": leak / max(leak_tot, 1),
            "prog_pubs": prog, "cost": cost, "wrong_idx": wrong_idx}


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
    print(f"A/B prompts on GSM8K | {len(data)} questions | top_k={args.top_k} max_steps={args.max_steps}")

    base = run_condition(False, data, args.llm_name, args.max_steps, args.top_k)
    print(f"[baseline] acc={base['acc']:.3f} ({base['solved']}/{base['n']}) "
          f"leak={base['leak_rate']:.2f} prog_pubs={base['prog_pubs']} cost=${base['cost']:.4f}")
    imp = run_condition(True, data, args.llm_name, args.max_steps, args.top_k)
    print(f"[improved] acc={imp['acc']:.3f} ({imp['solved']}/{imp['n']}) "
          f"leak={imp['leak_rate']:.2f} prog_pubs={imp['prog_pubs']} cost=${imp['cost']:.4f}")

    fixed = sorted(set(base["wrong_idx"]) - set(imp["wrong_idx"]))
    broke = sorted(set(imp["wrong_idx"]) - set(base["wrong_idx"]))
    print("\n==================== A/B SUMMARY ====================")
    print(f"{'metric':<26}{'baseline':<14}{'improved':<14}")
    print(f"{'accuracy':<26}{base['acc']:<14.3f}{imp['acc']:<14.3f}")
    print(f"{'refine solution-leak':<26}{base['leak_rate']:<14.2f}{imp['leak_rate']:<14.2f}")
    print(f"{'Programming Expert pubs':<26}{base['prog_pubs']:<14}{imp['prog_pubs']:<14}")
    print(f"{'cost ($)':<26}{base['cost']:<14.4f}{imp['cost']:<14.4f}")
    print(f"\nimproved fixed {len(fixed)} baseline-failures {fixed}; newly broke {len(broke)} {broke}")


if __name__ == "__main__":
    main()
