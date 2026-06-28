"""
Iterating to a real gain on the 25 MMLU questions.

Design (agent-driven, selective retrieval):
  1. A reasoning agent attempts the question and, IF it needs an external fact, itself
     proposes a targeted 'SEARCH: <query>' (the upstream agent formulates the query).
  2. Only then does the retriever run that query (selective -> no distraction on
     questions the model already knows).
  3. The agent re-answers with the retrieved facts.
Compared against a plain no-retrieval solver on the same 25 questions.
"""
import sys
import os
import re
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.llm.llm_registry import LLMRegistry
from RAPS.tools.search.bm25_retriever import bm25_retrieve
from raps_data.mmlu_dataset import MMLUDataset

llm = LLMRegistry.get("gpt-4o-mini-2024-07-18")


def choice(t):
    m = re.findall(r"[Aa]nswer\s*[:：]?\s*\(?([ABCD])\b", str(t))
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([ABCD])\b", str(t))
    return m[-1].upper() if m else "?"


def plain(q):
    return choice(llm.gen([{"role": "system", "content": "Answer the multiple-choice question. "
                            "Reason briefly, then end with 'Answer: X'."},
                           {"role": "user", "content": q}]))


def agent_rag(q, log=None):
    # Step 1: reason; the agent decides whether/what to retrieve.
    r1 = llm.gen([{"role": "system", "content":
                   "You answer 4-option multiple-choice questions. First reason briefly. "
                   "If you are confident, end with 'Answer: X'. If a SPECIFIC external fact "
                   "would resolve your uncertainty, instead end with a line 'SEARCH: <concise "
                   "Wikipedia query for that exact fact>' (do NOT answer yet)."},
                  {"role": "user", "content": q}])
    m = re.search(r"SEARCH:\s*(.+)", r1)
    if not m:
        return choice(r1), False  # confident -> no retrieval
    query = m.group(1).strip()
    ctx = bm25_retrieve([query, q], k=2, chars=600)
    if log is not None:
        log["query"] = query
        log["ctx"] = ctx[:160]
    r2 = llm.gen([{"role": "system", "content": "Answer the multiple-choice question using the "
                   "reference if relevant (ignore irrelevant parts). End with 'Answer: X'."},
                  {"role": "user", "content": f"Reference:\n{ctx}\n\nQuestion:\n{q}"}])
    return choice(r2), True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=25)
    args = p.parse_args()
    ds = MMLUDataset(split="test")
    data = [ds[i] for i in range(args.n)]
    base = rag = retrieved = 0
    flips = []
    for i, rec in enumerate(data):
        q = MMLUDataset.record_to_input(rec)["task"]
        gold = MMLUDataset.record_to_target_answer(rec)
        b = plain(q)
        lg = {}
        a, used = agent_rag(q, lg)
        base += b == gold
        rag += a == gold
        retrieved += used
        tag = ""
        if b == gold and a != gold:
            tag = "FLIP_DOWN"
        elif b != gold and a == gold:
            tag = "FLIP_UP"
        if tag:
            flips.append((i, gold, b, a, used, tag, lg.get("query", "")))
        print(f"PROG:: {i+1}/{args.n} base={base} rag={rag} retrieved={retrieved}", flush=True)
    print(f"\nRESULT:: plain={base}/{args.n}={base/args.n:.3f}  agent_selective_rag={rag}/{args.n}={rag/args.n:.3f}  "
          f"(retrieved on {retrieved}/{args.n})", flush=True)
    for i, gold, b, a, used, tag, query in flips:
        print(f"  {tag} Q{i} gold={gold} plain={b} rag={a} used_retrieval={used} query={query!r}", flush=True)


if __name__ == "__main__":
    main()
