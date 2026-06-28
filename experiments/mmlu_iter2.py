"""
Iteration 2: self-consistency + disagreement-triggered retrieval on the 25 MMLU questions.

- plain        : 1 sample (temp 0).
- self_consist : K samples (temp 0.7) -> majority vote.
- sc_plus_rag  : self-consistency; if the vote is SPLIT (real uncertainty), an agent
                 proposes a targeted query, retrieve, and re-vote with the facts.
"""
import sys
import os
import re
import argparse
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.llm.llm_registry import LLMRegistry
from RAPS.tools.search.bm25_retriever import bm25_retrieve
from raps_data.mmlu_dataset import MMLUDataset

llm = LLMRegistry.get("gpt-4o-mini-2024-07-18")
SYS = "Answer the 4-option multiple-choice question. Reason briefly, then end with 'Answer: X'."


def choice(t):
    m = re.findall(r"[Aa]nswer\s*[:：]?\s*\(?([ABCD])\b", str(t))
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([ABCD])\b", str(t))
    return m[-1].upper() if m else "?"


def sample(q, temp):
    return choice(llm.gen([{"role": "system", "content": SYS}, {"role": "user", "content": q}], temperature=temp))


def plain(q):
    return sample(q, 0.0)


def self_consist(q, k=5):
    votes = [sample(q, 0.7) for _ in range(k)]
    return Counter(votes).most_common(1)[0], votes


def query_for(q):
    r = llm.gen([{"role": "system", "content": "Write ONE concise Wikipedia search query for the "
                  "specific fact needed to answer this question. Output only the query."},
                 {"role": "user", "content": q}])
    return str(r).strip().splitlines()[0][:200]


def sc_plus_rag(q, k=5, agree=4):
    (top, n), votes = self_consist(q, k)
    if n >= agree:
        return top, False
    ctx = bm25_retrieve([query_for(q), q], k=2, chars=600)
    revote = [choice(llm.gen([{"role": "system", "content": "Use the reference if relevant. " + SYS},
                              {"role": "user", "content": f"Reference:\n{ctx}\n\nQuestion:\n{q}"}],
                             temperature=0.4)) for _ in range(3)]
    allv = votes + revote
    return Counter(allv).most_common(1)[0][0], True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=25)
    args = p.parse_args()
    ds = MMLUDataset(split="test")
    data = [ds[i] for i in range(args.n)]
    b = sc = scr = used = 0
    for i, rec in enumerate(data):
        q = MMLUDataset.record_to_input(rec)["task"]
        gold = MMLUDataset.record_to_target_answer(rec)
        b += plain(q) == gold
        (top, _), _ = self_consist(q)
        sc += top == gold
        a, u = sc_plus_rag(q)
        scr += a == gold
        used += u
        print(f"PROG:: {i+1}/{args.n} plain={b} sc={sc} sc_rag={scr} retrieved={used}", flush=True)
    n = args.n
    print(f"\nRESULT:: plain={b}/{n}={b/n:.3f}  self_consistency={sc}/{n}={sc/n:.3f}  "
          f"sc+rag={scr}/{n}={scr/n:.3f}  (retrieved on {used}/{n})", flush=True)


if __name__ == "__main__":
    main()
