"""Evaluate candidate designs ON the baseline chain's FAILED MMLU questions (recovery rate)."""
import sys, os, re, glob, json, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from RAPS.llm.llm_registry import LLMRegistry
from RAPS.prompt.mmlu_prompt_set import ROLE_DESCRIPTION
from raps_data.mmlu_dataset import MMLUDataset
from experiments.mmlu_design import solve as panel_solve   # independent panel
from RAPS.tools.coding.python_executor import execute_code_get_return

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


def baseline_failures():
    """Chain's wrong questions from the original max_steps=5 logs."""
    out = []
    for fp in glob.glob("logs/log_mmlu_2026-06-03-18-59-41_*"):
        try:
            L = json.load(open(fp))
        except Exception:
            continue
        fo = ""
        for st in reversed(L.get("steps", [])):
            if st.get("step") == "Final" and st.get("publications"):
                fo = st["publications"][0]["content"]; break
        gold = L.get("answer")
        if gold in "ABCD" and ds.postprocess_answer(fo) != gold:
            out.append((L["task"], gold))
    return out


def plain(task):
    return letter(llm.gen([{"role": "system", "content": "Answer the MCQ. Reason briefly, end with 'Answer: X'."},
                           {"role": "user", "content": task}]))


def fewshot(task):
    stem = task.split("\nOption A:")[0].strip()
    subj = SMAP.get(stem)
    if not subj:
        return plain(task)
    df = pd.read_csv(f"raps_data/MMLU/data/dev/{subj}_dev.csv", header=None, names=COLS).head(5)
    ex = "\n\n".join(f"{r['question']}\nOption A: {r['A']}\nOption B: {r['B']}\nOption C: {r['C']}\n"
                     f"Option D: {r['D']}\nAnswer: {r['answer']}" for _, r in df.iterrows())
    return letter(llm.gen([{"role": "user", "content":
            f"The following are multiple choice questions (with answers) about {subj.replace('_',' ')}.\n\n{ex}\n\n{task}\nAnswer:"}]))


def code_exec(task):
    raw = llm.gen([{"role": "system", "content": "If this MCQ is solvable by exact Python (logic/math/counting/probability), "
                    "write a program ending with answer='<letter>'. Otherwise output NONE."},
                   {"role": "user", "content": task}])
    if "```" not in str(raw):
        return None
    try:
        code = raw.split("```python")[1].split("```")[0] if "```python" in raw else raw.split("```")[1].split("```")[0]
    except Exception:
        return None
    v = str(execute_code_get_return(code)).strip().strip("'\"").upper()
    return v if v in "ABCD" else None


def main():
    p = argparse.ArgumentParser(); p.add_argument("--limit", type=int, default=999); args = p.parse_args()
    fails = baseline_failures()[:args.limit]
    print(f"RECOVER:: baseline chain failures = {len(fails)}", flush=True)
    rows = []  # per-question {panel,fewshot,code} booleans + subject
    for i, (task, gold) in enumerate(fails):
        stem = task.split("\nOption A:")[0].strip()
        r = {"subj": SMAP.get(stem, "?"),
             "panel": int(panel_solve(task)[0] == gold),
             "fewshot": int(fewshot(task) == gold)}
        ce = code_exec(task); r["code"] = int(ce == gold) if ce else 0
        r["any"] = int(r["panel"] or r["fewshot"] or r["code"])
        rows.append(r)
        print(f"PROG:: {i+1}/{len(fails)} panel={sum(x['panel'] for x in rows)} "
              f"fewshot={sum(x['fewshot'] for x in rows)} code={sum(x['code'] for x in rows)} "
              f"union={sum(x['any'] for x in rows)}", flush=True)
    n = len(fails)
    print(f"\nRECOVER:: on {n} baseline failures (chain=0 by definition):", flush=True)
    for k in ["panel", "fewshot", "code", "any"]:
        s = sum(x[k] for x in rows)
        print(f"RECOVER:: {k:8s} {s}/{n} = {s/n:.3f}", flush=True)
    # which subjects are recoverable at all vs hopeless
    from collections import Counter
    hopeless = Counter(x["subj"] for x in rows if not x["any"])
    recov = Counter(x["subj"] for x in rows if x["any"])
    print(f"RECOVER:: recoverable-by-subject = {dict(recov)}", flush=True)
    print(f"RECOVER:: HOPELESS-by-subject     = {dict(hopeless)}", flush=True)
    json.dump(rows, open("result/mmlu_recover_rows.json", "w"), indent=2)


if __name__ == "__main__":
    main()
