"""
Fair-comparison arbitration router for MMLU (design-only; no backbone change, no voting).

Diagnosis from the breakage dump: the arbiter was structurally biased toward the challenger,
because the cached chain output was a BARE LETTER while few-shot gave a full eloquent paragraph.
The arbiter got seduced by confident-but-wrong few-shot prose.

Fix: give the arbiter the chain's REAL reasoning (the last reasoning step from the log, not the
bare final letter), so both sides present comparable arguments. Symmetric arbiter (merits only).
"""
import sys, os, re, glob, json, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from RAPS.llm.llm_registry import LLMRegistry
from raps_data.mmlu_dataset import MMLUDataset

llm = LLMRegistry.get("gpt-4o-mini-2024-07-18")
ds = MMLUDataset(split="test")
COLS = ["question", "A", "B", "C", "D", "answer"]


def letter(t):
    m = re.findall(r"[Aa]nswer\s*[:：]?\s*\(?([ABCD])\b", str(t)) or re.findall(r"\b([ABCD])\b", str(t))
    return m[-1].upper() if m else "?"


def subject_map():
    m = {}
    for f in glob.glob("raps_data/MMLU/data/test/*_test.csv"):
        subj = os.path.basename(f)[:-len("_test.csv")]
        for q in pd.read_csv(f, header=None, names=COLS)["question"]:
            m[str(q)] = subj
    return m


SMAP = subject_map()


def chain_from_logs():
    """task -> (pred_letter, REAL_reasoning_text, gold). Reasoning = last non-Final step."""
    m = {}
    for fp in glob.glob("logs/log_mmlu_2026-06-03-18-59-41_*"):
        try:
            L = json.load(open(fp))
        except Exception:
            continue
        final_letter, reasoning = "", ""
        for st in reversed(L.get("steps", [])):
            pubs = st.get("publications")
            if not pubs:
                continue
            c = pubs[0]["content"]
            if st.get("step") == "Final":
                final_letter = c
            elif not reasoning:           # last reasoning step before Final
                reasoning = c
        g = L.get("answer")
        if g in "ABCD":
            m[L["task"]] = (ds.postprocess_answer(final_letter), str(reasoning), g)
    return m


def fewshot(task):
    stem = task.split("\nOption A:")[0].strip()
    subj = SMAP.get(stem)
    if not subj:
        out = llm.gen([{"role": "system", "content": "Answer the MCQ. Reason briefly, end with 'Answer: X'."},
                       {"role": "user", "content": task}])
        return letter(out), str(out)
    df = pd.read_csv(f"raps_data/MMLU/data/dev/{subj}_dev.csv", header=None, names=COLS).head(5)
    ex = "\n\n".join(f"{r['question']}\nOption A: {r['A']}\nOption B: {r['B']}\nOption C: {r['C']}\n"
                     f"Option D: {r['D']}\nAnswer: {r['answer']}" for _, r in df.iterrows())
    out = llm.gen([{"role": "user", "content":
            f"The following are multiple choice questions (with answers) about {subj.replace('_',' ')}. "
            f"For the LAST one, reason briefly then end with 'Answer: X'.\n\n{ex}\n\n{task}"}])
    return letter(out), str(out)


def arbiter(task, a_letter, a_text, b_letter, b_text):
    # SYMMETRIC, but now both sides carry real reasoning. Judge on the merits; reward the
    # argument grounded in a precise verifiable fact/rule, distrust confident-but-vague prose.
    out = llm.gen([{"role": "system", "content":
        "Two independent solutions to this multiple-choice question disagree. Each presents its "
        "reasoning below. Judge ONLY on correctness: check each chain of reasoning against the "
        "precise governing fact, definition, or rule. Confident or fluent prose is NOT evidence — "
        "an argument is only as good as the specific fact it rests on, and a plausible-sounding "
        "claim that is factually wrong must be rejected. Identify the decisive fact, say which "
        "solution got it right, and end with 'Answer: X'. Do not count votes."},
        {"role": "user", "content":
         f"{task}\n\n--- Solution 1 (answer {a_letter}) ---\n{a_text}\n\n"
         f"--- Solution 2 (answer {b_letter}) ---\n{b_text}\n\nWhich solution is correct?"}])
    return letter(out)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--limit", type=int, default=999); args = p.parse_args()
    chain = chain_from_logs()
    tasks = list(chain.items())[:args.limit]
    print(f"FAIR:: {len(tasks)} questions", flush=True)
    r_ok = c_ok = agree = arb = 0
    fixed, broke = [], []
    for i, (task, (cp, ctext, gold)) in enumerate(tasks):
        fp, ftext = fewshot(task)
        if fp == cp:
            final = cp; agree += 1
        else:
            arb += 1
            final = arbiter(task, cp, ctext, fp, ftext)
            if final not in "ABCD":
                final = cp
        r_ok += final == gold
        c_ok += cp == gold
        subj = SMAP.get(task.split("\nOption A:")[0].strip(), "?")
        if final == gold and cp != gold: fixed.append((i, subj))
        if cp == gold and final != gold: broke.append((i, subj))
        print(f"PROG:: {i+1}/{len(tasks)} router={r_ok} chain={c_ok} agree={agree} arb={arb} "
              f"fixed={len(fixed)} broke={len(broke)}", flush=True)
    n = len(tasks)
    print(f"\nFAIR:: chain  = {c_ok}/{n} = {c_ok/n:.3f}", flush=True)
    print(f"FAIR:: router = {r_ok}/{n} = {r_ok/n:.3f}  (net {r_ok-c_ok:+d})", flush=True)
    print(f"FAIR:: agreed {agree}, arbitrated {arb}; fixed {len(fixed)} {fixed}; broke {len(broke)} {broke}", flush=True)


if __name__ == "__main__":
    main()
