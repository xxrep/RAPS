"""
Redesigned MMLU coordination (MMLU-only): break the serial chain.

  Dispatcher  -> classify the question's subject, pick the most relevant experts
  Panel       -> those experts answer INDEPENDENTLY (each sees only the question,
                 no shared history -> no anchoring / error-propagation)
  Critic      -> examines the independent answers + reasoning, finds the disagreement
                 and the likely error, gives a corrected analysis (reasoning, not voting)
  Synthesizer -> weighs the in-domain expert's answer + the critique -> final letter
                 (a reasoned decision, NOT majority voting)

No backbone change, no multi-round sampling/voting. GSM8K/HumanEval untouched
(this is a standalone MMLU flow using the AnalyzeAgent role prompts).
"""
import sys
import os
import re
import argparse
import glob
import json
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.llm.llm_registry import LLMRegistry
from RAPS.prompt.mmlu_prompt_set import ROLE_DESCRIPTION
from raps_data.mmlu_dataset import MMLUDataset

llm = LLMRegistry.get("gpt-4o-mini-2024-07-18")
EXPERTS = ["Knowledge Expert", "Critic", "Mathematician", "Psychologist", "Historian",
           "Doctor", "Lawyer", "Economist", "Programmer"]
# specialists the dispatcher can route to (Critic is reserved for the correction stage)
SPECIALISTS = [r for r in EXPERTS if r != "Critic"]


def letter(t):
    m = re.findall(r"[Aa]nswer\s*[:：]?\s*\(?([ABCD])\b", str(t))
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([ABCD])\b", str(t))
    return m[-1].upper() if m else "?"


def dispatch(question, k=3):
    opts = "\n".join(f"- {r}: {ROLE_DESCRIPTION[r].strip().splitlines()[0]}" for r in SPECIALISTS)
    r = llm.gen([{"role": "system", "content": "You route a question to the most relevant specialists."},
                 {"role": "user", "content": f"Question:\n{question}\n\nSpecialists:\n{opts}\n\n"
                  f"List the {k} MOST relevant specialist names for this question, comma-separated, no extra text."}])
    picked = [r2 for r2 in SPECIALISTS if r2.lower() in str(r).lower()][:k]
    return picked or ["Knowledge Expert", "Mathematician", "Historian"][:k]


def expert_answer(role, question):
    sys = (ROLE_DESCRIPTION[role].strip() +
           "\n\nAnswer this 4-option multiple-choice question from your expertise. Reason briefly "
           "(<80 words), then end with a line 'Answer: X' (X = A/B/C/D).")
    out = llm.gen([{"role": "system", "content": sys}, {"role": "user", "content": question}])
    return {"role": role, "letter": letter(out), "text": str(out)}


def panel(question, roles):
    with ThreadPoolExecutor(max_workers=len(roles)) as ex:
        return list(ex.map(lambda r: expert_answer(r, question), roles))


def critic(question, answers):
    block = "\n\n".join(f"[{a['role']} -> {a['letter']}] {a['text']}" for a in answers)
    # NEUTRAL analytical critic: examine the disagreement and reason toward the correct
    # option on the merits (NOT adversarial, NOT vote-counting).
    out = llm.gen([{"role": "system", "content":
                    "You are a careful examiner. The specialists answered independently and may "
                    "disagree. Identify exactly where and why they diverge, check each option against "
                    "the precise governing fact/definition/rule, and flag any reasoning that relies on "
                    "a misconception or a trap distractor. Conclude with the option best supported on "
                    "the merits and why. Reason rigorously; do not count votes."},
                   {"role": "user", "content": f"Question:\n{question}\n\nIndependent answers:\n{block}"}])
    return str(out)


def synthesize(question, answers, critique):
    block = "\n".join(f"[{a['role']}] Answer: {a['letter']}" for a in answers)
    out = llm.gen([{"role": "system", "content":
                    "You are the chief decision-maker. Using the independent specialists' answers and "
                    "the critic's analysis, decide the correct option. Weigh the specialist whose field "
                    "best matches the question's subject, and the critic's reasoning — do NOT simply take "
                    "the majority. Output ONLY one letter: A, B, C, or D."},
                   {"role": "user", "content": f"Question:\n{question}\n\nSpecialist answers:\n{block}\n\n"
                    f"Critic analysis:\n{critique}"}])
    return letter(out)


def solve(question):
    roles = dispatch(question)
    answers = panel(question, roles)
    crit = critic(question, answers)
    return synthesize(question, answers, crit), roles


def chain_baseline_from_logs():
    """Map task text -> chain pipeline's predicted letter, from the original max_steps=5 run."""
    ds = MMLUDataset(split="test")
    m = {}
    for fp in glob.glob("logs/log_mmlu_2026-06-03-18-59-41_*"):
        try:
            L = json.load(open(fp))
        except Exception:
            continue
        fo = ""
        for st in reversed(L.get("steps", [])):
            if st.get("step") == "Final" and st.get("publications"):
                fo = st["publications"][0]["content"]
                break
        m[L["task"]] = ds.postprocess_answer(fo)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--start", type=int, default=0)
    args = p.parse_args()
    ds = MMLUDataset(split="test")
    chain = chain_baseline_from_logs()
    new_ok = chain_ok = matched = 0
    fixed, broke = [], []
    for i in range(args.start, args.start + args.n):
        rec = ds[i]
        q = MMLUDataset.record_to_input(rec)["task"]
        gold = MMLUDataset.record_to_target_answer(rec)
        npred, roles = solve(q)
        new_ok += npred == gold
        cpred = chain.get(q)
        if cpred is not None:
            matched += 1
            chain_ok += cpred == gold
            if npred == gold and cpred != gold:
                fixed.append(i)
            if cpred == gold and npred != gold:
                broke.append(i)
        print(f"PROG:: {i-args.start+1}/{args.n} new={new_ok} | matched_chain={chain_ok}/{matched}", flush=True)
    print(f"\nRESULT:: NEW design = {new_ok}/{args.n} = {new_ok/args.n:.3f}", flush=True)
    if matched:
        print(f"RESULT:: chain baseline (same {matched} matched Qs) = {chain_ok}/{matched} = {chain_ok/matched:.3f}", flush=True)
        print(f"RESULT:: vs chain -> fixed {len(fixed)} {fixed}; broke {len(broke)} {broke}", flush=True)


if __name__ == "__main__":
    main()
