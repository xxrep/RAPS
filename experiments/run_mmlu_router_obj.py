"""
Objective-evidence arbitration router for MMLU (design-only; no backbone change, no voting).

Same chain(primary) vs few-shot(challenger) gate as run_mmlu_router.py, but on DISAGREEMENT
the arbiter is grounded in evidence OUTSIDE the backbone:
  - computable subjects -> run Python (code execution) for an independent objective answer
  - factual subjects    -> local Wikipedia BM25 passages as evidence
The arbiter stays incumbent-biased (overturn the primary only on demonstrable error), but now
weighs an objective signal instead of just the model's own re-reasoning.
"""
import sys, os, re, glob, json, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from RAPS.llm.llm_registry import LLMRegistry
from raps_data.mmlu_dataset import MMLUDataset
from RAPS.tools.coding.python_executor import execute_code_get_return
from RAPS.tools.search.bm25_retriever import bm25_retrieve

llm = LLMRegistry.get("gpt-4o-mini-2024-07-18")
ds = MMLUDataset(split="test")
COLS = ["question", "A", "B", "C", "D", "answer"]
COMPUTABLE = {"formal_logic", "abstract_algebra", "high_school_mathematics", "college_mathematics",
              "elementary_mathematics", "high_school_statistics", "college_physics",
              "high_school_physics", "conceptual_physics"}


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


def subj_of(task):
    return SMAP.get(task.split("\nOption A:")[0].strip(), "?")


def chain_from_logs():
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
    subj = subj_of(task)
    if subj == "?":
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


def code_signal(task):
    raw = llm.gen([{"role": "system", "content": "If this MCQ can be settled by exact Python "
                    "(logic/math/counting/probability/algebra), write a program ending with "
                    "answer='<letter>'. Otherwise output NONE."},
                   {"role": "user", "content": task}])
    if "```" not in str(raw):
        return None
    try:
        code = raw.split("```python")[1].split("```")[0] if "```python" in raw else raw.split("```")[1].split("```")[0]
        v = str(execute_code_get_return(code)).strip().strip("'\"").upper()
        return v if v in "ABCD" else None
    except Exception:
        return None


def retrieve_signal(task):
    q = llm.gen([{"role": "system", "content": "Write ONE focused Wikipedia search query (<=12 words) "
                  "that would surface the fact needed to answer this question. Output only the query."},
                 {"role": "user", "content": task.split("\nOption A:")[0]}])
    try:
        return bm25_retrieve([str(q).strip().strip('"')], k=3, chars=500)
    except Exception:
        return ""


def arbiter(task, a_letter, a_text, b_letter, b_text, evidence_label, evidence):
    out = llm.gen([{"role": "system", "content":
        "A PRIMARY solution and a CHALLENGER disagree on this multiple-choice question. The primary "
        "is the default and stands UNLESS the OBJECTIVE EVIDENCE below demonstrates a concrete error "
        "in it. Trust the objective evidence over either attempt's assertions. If the evidence is "
        "irrelevant or inconclusive, keep the PRIMARY. Do not count votes. Reason step by step, then "
        "end with 'Answer: X'."},
        {"role": "user", "content":
         f"{task}\n\n--- PRIMARY (answer {a_letter}) ---\n{a_text}\n\n"
         f"--- CHALLENGER (answer {b_letter}) ---\n{b_text}\n\n"
         f"--- OBJECTIVE EVIDENCE ({evidence_label}) ---\n{evidence}\n\nDecide the final answer."}])
    return letter(out)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--limit", type=int, default=999); args = p.parse_args()
    chain = chain_from_logs()
    tasks = list(chain.items())[:args.limit]
    print(f"ROUTEROBJ:: {len(tasks)} questions", flush=True)
    r_ok = c_ok = agree = arb = 0
    fixed, broke = [], []
    for i, (task, (cp, ctext, gold)) in enumerate(tasks):
        fp, ftext = fewshot(task)
        if fp == cp:
            final = cp; agree += 1
        else:
            arb += 1
            subj = subj_of(task)
            if subj in COMPUTABLE:
                cs = code_signal(task)
                if cs:
                    final = arbiter(task, cp, ctext, fp, ftext, "Python execution result",
                                    f"An independent program computed the answer to be: {cs}")
                else:
                    final = cp  # no objective signal -> keep primary
            else:
                ev = retrieve_signal(task)
                final = arbiter(task, cp, ctext, fp, ftext, "Wikipedia passages",
                                ev) if ev else cp
            if final not in "ABCD":
                final = cp
        r_ok += final == gold
        c_ok += cp == gold
        if final == gold and cp != gold: fixed.append((i, subj_of(task)))
        if cp == gold and final != gold: broke.append((i, subj_of(task)))
        print(f"PROG:: {i+1}/{len(tasks)} router={r_ok} chain={c_ok} agree={agree} arb={arb} "
              f"fixed={len(fixed)} broke={len(broke)}", flush=True)
    n = len(tasks)
    print(f"\nROUTEROBJ:: chain  = {c_ok}/{n} = {c_ok/n:.3f}", flush=True)
    print(f"ROUTEROBJ:: router = {r_ok}/{n} = {r_ok/n:.3f}  (net {r_ok-c_ok:+d})", flush=True)
    print(f"ROUTEROBJ:: agreed {agree}, arbitrated {arb}; fixed {len(fixed)} {fixed}; broke {len(broke)} {broke}", flush=True)


if __name__ == "__main__":
    main()
