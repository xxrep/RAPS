"""
A/B on MMLU prompt optimization, deterministic (temp=0), top_k=3, sim_threshold=0.40.

old : current role prompts (vague "give a brief analysis"; experts never commit a letter)
new : each expert stays in its domain lane and COMMITS a decisive answer + confidence,
      and the Final Answerer weighs the committed answers by domain relevance/confidence.
"""
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.agents.agent_registry import AgentRegistry
from RAPS.agents.analyze_agent import AnalyzeAgent  # ensure registration
from RAPS.core import RAPSCoordinator, RAPSConfig
from RAPS.prompt.mmlu_prompt_set import MMLUPromptSet
from raps_data.mmlu_dataset import MMLUDataset

LLM = "gpt-4o-mini-2024-07-18"
ROLES = ["Knowledge Expert", "Critic", "Mathematician", "Psychologist",
         "Historian", "Doctor", "Lawyer", "Economist", "Programmer"]

# Optimized expert format: decisive, in-lane, committed answer + confidence.
NEW_ANSWER_FMT = (
    "\n\nThis is a 4-option multiple-choice question (A/B/C/D); exactly one option is correct. "
    "Answer strictly from YOUR domain of expertise. If the question is clearly outside your "
    "domain, say so in one phrase and give your best guess with LOW confidence; if it is in your "
    "domain, be decisive. Give a brief analysis (<70 words), then end with a final line EXACTLY:\n"
    "Answer: X | Confidence: high/medium/low   (X is one of A, B, C, D)"
)
NEW_DECISION = (
    "Each expert ended its reply with a line 'Answer: X | Confidence: ...'. Identify the question's "
    "subject, then weigh the experts' committed answers by (a) how relevant their domain is to this "
    "subject and (b) their confidence. Prefer the option supported by the most relevant, most "
    "confident, mutually-agreeing experts. Output ONLY one letter (A, B, C, or D) with no other text."
)


def make_prompt(role, desc, new):
    if new:
        return f"{desc.strip()}{NEW_ANSWER_FMT}"
    if role == "Knowledge Expert":
        return f"{desc}\n"
    return (f"{desc}\nYour reply must be less than 100 words and "
            f"include a brief step by step analysis of the question.\n")


def build(new):
    ps = MMLUPromptSet()
    agents = []
    for role in ROLES:
        agent = AgentRegistry.get("AnalyzeAgent", id=None, llm_name=LLM, domain="mmlu",
                                  role=role, capabilities=make_prompt(role, ps.get_description(role), new),
                                  interests="", additional_instructions="")
        agent.history = []
        agent.inbox = []
        agents.append(agent)
    fa = AgentRegistry.get("AnalyzeAgent", id=None, llm_name=LLM, domain="mmlu",
                           role="Final Answerer", capabilities=MMLUPromptSet.get_decision_role(),
                           interests="",
                           additional_instructions=NEW_DECISION if new else MMLUPromptSet.get_decision_constraint())
    return agents, fa


def run_condition(new, dataset, ds, top_k, thr, max_steps):
    agents, fa = build(new)
    cfg = RAPSConfig(domain="mmlu", max_steps=max_steps, top_k=top_k, sim_threshold=thr, entry_index=0)
    coord = RAPSCoordinator(agents, fa, cfg, logger=lambda m: None, answer_extractor=ds.postprocess_answer)
    solved = 0
    for i, rec in enumerate(dataset):
        task = MMLUDataset.record_to_input(rec)["task"]
        gold = MMLUDataset.record_to_target_answer(rec)
        pred = ds.postprocess_answer(coord.run(task).final_output)
        solved += 1 if pred == gold else 0
        print(f"PROG:: {'new' if new else 'old'} {i+1}/{len(dataset)} solved={solved}", flush=True)
    return solved / len(dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--sim_threshold", type=float, default=0.40)
    p.add_argument("--max_steps", type=int, default=5)
    args = p.parse_args()

    ds = MMLUDataset(split="test")
    dataset = [ds[i] for i in range(args.start, min(args.start + args.limit, len(ds)))]
    print(f"MMLU prompt A/B | {len(dataset)} q | top_k={args.top_k} thr={args.sim_threshold} temp=0", flush=True)

    old = run_condition(False, dataset, ds, args.top_k, args.sim_threshold, args.max_steps)
    print(f"RESULT:: OLD prompts  acc={old:.3f}", flush=True)
    new = run_condition(True, dataset, ds, args.top_k, args.sim_threshold, args.max_steps)
    print(f"RESULT:: NEW prompts  acc={new:.3f}", flush=True)
    print(f"RESULT:: delta {new-old:+.3f}  (baseline ref: original run ~0.837)", flush=True)


if __name__ == "__main__":
    main()
