"""
Confidence-gated arbitration router for MMLU (design-only, no backbone change, no voting).

Per question:
  chain_pred  : cached from the original max_steps=5 run (the 0.837 baseline)
  fewshot_pred: 5-shot single solver (the broad recoverer: 7/9 of recoverable failures)
  - if they AGREE  -> keep it (zero risk)
  - if they DISAGREE -> an ARBITER reads BOTH chains of reasoning and decides which is
    correct on the merits (reasoning-based arbitration, NOT vote counting)

Measured NET on the full deduped 172-question set (breakage included): fixed vs broke vs chain.
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
    """task -> (pred_letter, final_text, gold)."""
    m = {}
    for fp in glob.glob("logs/log_mmlu_2026-06-03-18-59-41_*"):
        try:
            L = json.load(open(fp))
        except Exception:
            continue
        fo = ""
        for st in reversed(L.get("steps", [])):
            if st.get("step") == "Final" and st.get("publications"):
                fo = st["publications"][0]["content"]; break
        g = L.get("answer")
        if g in "ABCD":
            m[L["task"]] = (ds.postprocess_answer(fo), str(fo), g)
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
    # INCUMBENT-BIASED arbitration: the primary solution (Attempt 1) is the default; the
    # challenger only wins if it cites a specific, verifiable fact/rule that proves the
    # primary made a concrete error. Asymmetric burden of proof, not vote counting.
    out = llm.gen([{"role": "system", "content":
        "A PRIMARY solution and a CHALLENGER reached different answers on this multiple-choice "
        "question. The primary is the default and should stand UNLESS the challenger demonstrates, "
        "with a specific verifiable fact / definition / rule, that the primary made a concrete "
        "factual or logical error. A merely plausible alternative is NOT enough to overturn the "
        "primary — only a demonstrable error in the primary is. If the evidence is ambiguous or "
        "both are defensible, keep the PRIMARY. Reason step by step, then end with 'Answer: X'."},
        {"role": "user", "content":
         f"{task}\n\n--- PRIMARY (answer {a_letter}) ---\n{a_text}\n\n"
         f"--- CHALLENGER (answer {b_letter}) ---\n{b_text}\n\n"
         f"Does the challenger prove a concrete error in the primary? Decide the final answer."}])
    return letter(out)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--limit", type=int, default=999); args = p.parse_args()
    chain = chain_from_logs()
    tasks = list(chain.items())[:args.limit]
    print(f"ROUTER:: {len(tasks)} questions", flush=True)
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
        if final == gold and cp != gold: fixed.append((i, SMAP.get(task.split("\nOption A:")[0].strip(), "?")))
        if cp == gold and final != gold: broke.append((i, SMAP.get(task.split("\nOption A:")[0].strip(), "?")))
        print(f"PROG:: {i+1}/{len(tasks)} router={r_ok} chain={c_ok} agree={agree} arb={arb} "
              f"fixed={len(fixed)} broke={len(broke)}", flush=True)
    n = len(tasks)
    print(f"\nROUTER:: chain  = {c_ok}/{n} = {c_ok/n:.3f}", flush=True)
    print(f"ROUTER:: router = {r_ok}/{n} = {r_ok/n:.3f}  (net {r_ok-c_ok:+d})", flush=True)
    print(f"ROUTER:: agreed {agree}, arbitrated {arb}; fixed {len(fixed)} {fixed}; broke {len(broke)} {broke}", flush=True)


if __name__ == "__main__":
    main()
