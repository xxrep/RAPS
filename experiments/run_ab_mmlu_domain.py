"""
MMLU multi-agent optimization via PROMPTS + COMMUNICATION only (no voting/sampling).
Deterministic (temp=0), top_k=3, sim_threshold=0.40.

old : current expert prompts + generic Final Answerer.
new : each expert declares its DOMAIN-FIT and commits an answer; the Final Answerer
      defers to the most domain-relevant expert (fixes the 'aggregation drift' where a
      correct in-domain expert is overruled by off-topic experts, e.g. Q12).

MMLU-only: touches just the AnalyzeAgent prompts + Final Answerer; GSM8K / HumanEval
use their own prompt sets and are unaffected.
"""
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.agents.agent_registry import AgentRegistry
from RAPS.agents.analyze_agent import AnalyzeAgent
from RAPS.core import RAPSCoordinator, RAPSConfig
from RAPS.prompt.mmlu_prompt_set import MMLUPromptSet
from raps_data.mmlu_dataset import MMLUDataset

LLM = "gpt-4o-mini-2024-07-18"
ROLES = ["Knowledge Expert", "Critic", "Mathematician", "Psychologist",
         "Historian", "Doctor", "Lawyer", "Economist", "Programmer"]

NEW_EXPERT = (
    "\n\nThis is a 4-option multiple-choice question (A/B/C/D). FIRST judge honestly whether it "
    "falls within YOUR specific field: write a line 'Domain-fit: high' (squarely your field), "
    "'medium' (partially), or 'low' (outside your field — do NOT overclaim). Then give a brief "
    "(<60 words) analysis from your expertise and end with a final line 'Answer: X'."
)
NEW_FINAL = (
    "You are the chief decision-maker for a panel of domain experts. Each expert reported a "
    "'Domain-fit' (high/medium/low) and an 'Answer'. Decide as follows: (1) identify the academic "
    "subject of the question; (2) find the expert(s) whose field genuinely matches that subject "
    "(prioritize those who declared high domain-fit AND are actually relevant); (3) adopt the "
    "in-domain expert's Answer unless its reasoning is clearly wrong — do NOT let off-topic or "
    "low-domain-fit experts override the qualified expert. Output ONLY one letter: A, B, C, or D."
)


def expert(role, new):
    desc = MMLUPromptSet().get_description(role)
    if new:
        cap = desc.strip() + NEW_EXPERT
    else:
        cap = (f"{desc}\n" if role == "Knowledge Expert"
               else f"{desc}\nYour reply must be less than 100 words and include a brief step by "
                    f"step analysis of the question.\n")
    a = AgentRegistry.get("AnalyzeAgent", id=None, llm_name=LLM, domain="mmlu",
                          role=role, capabilities=cap, interests="", additional_instructions="")
    a.history = []
    a.inbox = []
    return a


def build(new):
    agents = [expert(r, new) for r in ROLES]
    fa = AgentRegistry.get("AnalyzeAgent", id=None, llm_name=LLM, domain="mmlu",
                           role="Final Answerer", capabilities=MMLUPromptSet.get_decision_role(),
                           interests="",
                           additional_instructions=NEW_FINAL if new else MMLUPromptSet.get_decision_constraint())
    return agents, fa


def run(new, data, ds):
    agents, fa = build(new)
    cfg = RAPSConfig(domain="mmlu", max_steps=2, top_k=3, sim_threshold=0.40, entry_index=0)
    coord = RAPSCoordinator(agents, fa, cfg, logger=lambda m: None, answer_extractor=ds.postprocess_answer)
    res = {}
    for i, rec in enumerate(data):
        q = MMLUDataset.record_to_input(rec)["task"]
        gold = MMLUDataset.record_to_target_answer(rec)
        pred = ds.postprocess_answer(coord.run(q).final_output)
        res[i] = (pred == gold)
        print(f"PROG:: {'new' if new else 'old'} {i+1}/{len(data)} solved={sum(res.values())}", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=25)
    args = p.parse_args()
    ds = MMLUDataset(split="test")
    data = [ds[i] for i in range(args.n)]
    old = run(False, data, ds)
    new = run(True, data, ds)
    o, n = sum(old.values()), sum(new.values())
    print(f"\nRESULT:: OLD={o}/{args.n}={o/args.n:.3f}  NEW(domain-aware)={n}/{args.n}={n/args.n:.3f}  delta {(n-o)/args.n:+.3f}", flush=True)
    fixed = [i for i in old if not old[i] and new[i]]
    broke = [i for i in old if old[i] and not new[i]]
    print(f"RESULT:: fixed {len(fixed)} {fixed}; broke {len(broke)} {broke}", flush=True)


if __name__ == "__main__":
    main()
