"""Dump the FULL internals of the disagreement cases the router fixed (and broke)."""
import sys, os, re, glob, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from RAPS.llm.llm_registry import LLMRegistry
from raps_data.mmlu_dataset import MMLUDataset
from experiments.run_mmlu_router import chain_from_logs, fewshot, arbiter, subject_map, letter

llm = LLMRegistry.get("gpt-4o-mini-2024-07-18")
ds = MMLUDataset(split="test")
SMAP = subject_map()


def main():
    chain = chain_from_logs()
    tasks = list(chain.items())
    for i, (task, (cp, ctext, gold)) in enumerate(tasks):
        fp, ftext = fewshot(task)
        if fp == cp:
            continue
        final = arbiter(task, cp, ctext, fp, ftext)
        tag = None
        if final == gold and cp != gold:
            tag = "FIXED"
        elif cp == gold and final != gold:
            tag = "BROKE"
        if not tag:
            continue
        subj = SMAP.get(task.split("\nOption A:")[0].strip(), "?")
        print("=" * 90)
        print(f"[{tag}] idx={i} subj={subj}  chain={cp}  fewshot={fp}  arbiter={final}  GOLD={gold}")
        print("-" * 90)
        print("QUESTION:\n" + task.strip()[:900])
        print("-" * 90)
        print(f"CHAIN reasoning (answered {cp}, WRONG):\n" + ctext.strip()[:700])
        print("-" * 90)
        print(f"FEWSHOT reasoning (answered {fp}):\n" + ftext.strip()[:700])
        print("=" * 90 + "\n", flush=True)


if __name__ == "__main__":
    main()
