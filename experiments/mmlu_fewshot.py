"""Test the standard MMLU lever: 0-shot vs 5-shot (per-subject dev exemplars). MMLU-only."""
import sys, os, re, glob, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from RAPS.llm.llm_registry import LLMRegistry
from raps_data.mmlu_dataset import MMLUDataset
llm = LLMRegistry.get("gpt-4o-mini-2024-07-18")
COLS = ["question", "A", "B", "C", "D", "answer"]


def subject_map():
    m = {}
    for f in glob.glob("raps_data/MMLU/data/test/*_test.csv"):
        subj = os.path.basename(f)[:-len("_test.csv")]
        df = pd.read_csv(f, header=None, names=COLS)
        for q in df["question"]:
            m[str(q)] = subj
    return m


def fmt(row, with_ans):
    s = f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\nAnswer:"
    return s + f" {row['answer']}\n" if with_ans else s


def shots(subject, k=5):
    df = pd.read_csv(f"raps_data/MMLU/data/dev/{subject}_dev.csv", header=None, names=COLS).head(k)
    return "\n".join(fmt(r, True) for _, r in df.iterrows())


def letter(t):
    m = re.findall(r"\b([ABCD])\b", str(t))
    return m[0].upper() if m else "?"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=60); args = p.parse_args()
    ds = MMLUDataset(split="test"); smap = subject_map()
    z = f = matched = 0
    for i in range(args.n):
        rec = ds[i]
        subj = smap.get(str(rec["question"]))
        gold = rec["correct_answer"]
        row = {"question": rec["question"], "A": rec["A"], "B": rec["B"], "C": rec["C"], "D": rec["D"]}
        qblock = fmt(row, False)
        # 0-shot
        z0 = letter(llm.gen([{"role": "user", "content":
              f"The following is a multiple choice question (with answer).\n\n{qblock}"}]))
        z += z0 == gold
        # 5-shot (same subject)
        if subj:
            matched += 1
            ctx = shots(subj)
            f5 = letter(llm.gen([{"role": "user", "content":
                  f"The following are multiple choice questions (with answers) about {subj.replace('_',' ')}.\n\n{ctx}\n{qblock}"}]))
            f += f5 == gold
        print(f"PROG:: {i+1}/{args.n} 0shot={z} 5shot={f} (matched {matched})", flush=True)
    print(f"\nRESULT:: single-solver  0-shot={z}/{args.n}={z/args.n:.3f}  5-shot={f}/{matched}={f/max(matched,1):.3f}", flush=True)


if __name__ == "__main__":
    main()
