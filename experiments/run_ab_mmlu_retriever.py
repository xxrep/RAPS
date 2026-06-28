"""
A/B: does a dedicated RetrieverAgent (Wiki grounding, as entry host) help the MMLU pipeline?
Deterministic (temp=0), top_k=3, sim_threshold=0.40.

no_ret  : current pipeline (9 experts + Final Answerer), entry = Knowledge Expert.
with_ret: RetrieverAgent grounds first (entry), then routes to the experts.
"""
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.agents.agent_registry import AgentRegistry
from RAPS.agents.analyze_agent import AnalyzeAgent       # ensure registration
from RAPS.agents.retriever_agent import RetrieverAgent   # ensure registration
from RAPS.core import RAPSCoordinator, RAPSConfig
from RAPS.prompt.mmlu_prompt_set import MMLUPromptSet
from raps_data.mmlu_dataset import MMLUDataset

LLM = "gpt-4o-mini-2024-07-18"
ROLES = ["Knowledge Expert", "Critic", "Mathematician", "Psychologist",
         "Historian", "Doctor", "Lawyer", "Economist", "Programmer"]

RETRIEVER_CAP = ("You retrieve factual reference material from Wikipedia to ground the team's "
                 "reasoning. Use this host when a question needs factual grounding or domain facts.")


def expert(role):
    desc = MMLUPromptSet().get_description(role)
    cap = (f"{desc}\n" if role == "Knowledge Expert"
           else f"{desc}\nYour reply must be less than 100 words and include a brief step by step "
                f"analysis of the question. Use any reference material provided to you.\n")
    a = AgentRegistry.get("AnalyzeAgent", id=None, llm_name=LLM, domain="mmlu",
                          role=role, capabilities=cap, interests="", additional_instructions="")
    a.history = []
    a.inbox = []
    return a


def build(with_ret):
    agents = []
    if with_ret:
        r = AgentRegistry.get("RetrieverAgent", id=None, llm_name=LLM, domain="mmlu",
                              role="Retriever", capabilities=RETRIEVER_CAP, interests="")
        r.history = []
        r.inbox = []
        agents.append(r)
    agents += [expert(role) for role in ROLES]
    fa = AgentRegistry.get("AnalyzeAgent", id=None, llm_name=LLM, domain="mmlu",
                           role="Final Answerer", capabilities=MMLUPromptSet.get_decision_role(),
                           interests="", additional_instructions=MMLUPromptSet.get_decision_constraint())
    return agents, fa


def run(with_ret, dataset, ds, top_k, thr, max_steps):
    agents, fa = build(with_ret)
    cfg = RAPSConfig(domain="mmlu", max_steps=max_steps, top_k=top_k, sim_threshold=thr, entry_index=0)
    coord = RAPSCoordinator(agents, fa, cfg, logger=lambda m: None, answer_extractor=ds.postprocess_answer)
    solved = 0
    for i, rec in enumerate(dataset):
        task = MMLUDataset.record_to_input(rec)["task"]
        gold = MMLUDataset.record_to_target_answer(rec)
        pred = ds.postprocess_answer(coord.run(task).final_output)
        solved += 1 if pred == gold else 0
        print(f"PROG:: {'with_ret' if with_ret else 'no_ret'} {i+1}/{len(dataset)} solved={solved}", flush=True)
    return solved / len(dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--sim_threshold", type=float, default=0.40)
    p.add_argument("--max_steps", type=int, default=3)
    args = p.parse_args()
    ds = MMLUDataset(split="test")
    dataset = [ds[i] for i in range(args.start, min(args.start + args.limit, len(ds)))]
    print(f"MMLU RetrieverAgent A/B | {len(dataset)} q | top_k={args.top_k} thr={args.sim_threshold}", flush=True)

    a = run(False, dataset, ds, args.top_k, args.sim_threshold, args.max_steps)
    print(f"RESULT:: no_ret    acc={a:.3f}", flush=True)
    b = run(True, dataset, ds, args.top_k, args.sim_threshold, args.max_steps)
    print(f"RESULT:: with_ret  acc={b:.3f}", flush=True)
    print(f"RESULT:: delta {b-a:+.3f}", flush=True)


if __name__ == "__main__":
    main()
