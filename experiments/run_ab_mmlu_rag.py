"""
MMLU RAG A/B (deterministic, temp=0) on a single solver.

A (no_rag) : a single strong solver answers the MCQ directly.
B (rag)    : the solver first generates Wikipedia search queries, retrieves real
             Wikipedia intros, and answers with that reference material in context.

Isolates the retrieval lever (the 9-expert pipeline did not help MMLU).
"""
import sys
import os
import re
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.llm.llm_registry import LLMRegistry
from RAPS.tools.search.bm25_retriever import bm25_retrieve as retrieve
from raps_data.mmlu_dataset import MMLUDataset

LLM = "gpt-4o-mini-2024-07-18"
llm = LLMRegistry.get(LLM)

SOLVE_SYS = ("You are an expert answering a 4-option multiple-choice question (A/B/C/D); "
             "exactly one option is correct. Reason briefly, then end with a final line "
             "'Answer: X' where X is one of A, B, C, D.")


def extract_choice(text: str) -> str:
    m = re.findall(r"[Aa]nswer\s*[:：]\s*\(?([ABCD])", str(text))
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([ABCD])\b", str(text))
    return m[-1].upper() if m else ""


def gen_queries(question: str):
    prompt = ("Give 1-2 concise Wikipedia search queries (key topics/entities) that would help "
              "answer the following question. Output ONLY the queries separated by ';'.\n\n" + question)
    raw = llm.gen([{"role": "system", "content": "You write search queries."},
                   {"role": "user", "content": prompt}])
    qs = [q.strip() for q in str(raw).replace("\n", ";").split(";") if q.strip()]
    return qs[:2]


def solve(question: str, rag: bool) -> str:
    if rag:
        try:
            # Use the FULL question (rich context) as the primary query, plus focused
            # sub-queries; text-only BM25 then returns the on-topic article.
            context = retrieve([question] + gen_queries(question), k=2, chars=600)
        except Exception:
            context = ""
        user = (f"Reference material from Wikipedia (use it only if relevant):\n{context}\n\n"
                f"{question}") if context else question
    else:
        user = question
    return llm.gen([{"role": "system", "content": SOLVE_SYS}, {"role": "user", "content": user}])


def run(rag: bool, dataset, ds):
    solved = 0
    for i, rec in enumerate(dataset):
        q = MMLUDataset.record_to_input(rec)["task"]
        gold = MMLUDataset.record_to_target_answer(rec)
        pred = extract_choice(solve(q, rag))
        solved += 1 if pred == gold else 0
        print(f"PROG:: {'rag' if rag else 'norag'} {i+1}/{len(dataset)} solved={solved}", flush=True)
    return solved / len(dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--start", type=int, default=0)
    args = p.parse_args()
    ds = MMLUDataset(split="test")
    dataset = [ds[i] for i in range(args.start, min(args.start + args.limit, len(ds)))]
    print(f"MMLU RAG A/B | single solver | {len(dataset)} questions | temp=0", flush=True)

    a = run(False, dataset, ds)
    print(f"RESULT:: no_rag   acc={a:.3f}", flush=True)
    b = run(True, dataset, ds)
    print(f"RESULT:: rag(wiki) acc={b:.3f}", flush=True)
    print(f"RESULT:: delta {b-a:+.3f}  (multi-agent pipeline ref ~0.837)", flush=True)


if __name__ == "__main__":
    main()
