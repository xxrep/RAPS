"""
Measure whether the fixed ReAct retrieval (kstem index + discriminative keyword queries)
RECOVERS knowledge-intensive MMLU baseline failures, run through the full top_k=3 pipeline.

Recovery = baseline chain (top_k=1) got it wrong; the new ReAct pipeline gets it right.
We restrict to knowledge subjects where external facts plausibly help.
"""
import sys, os, re, glob, json, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from RAPS.core import RAPSCoordinator, RAPSConfig
from raps_data.mmlu_dataset import MMLUDataset
from experiments.run_mmlu import initialize_agents_from_set, initialize_final_answerer

ds = MMLUDataset(split="test")
COLS = ["question", "A", "B", "C", "D", "answer"]

KNOWLEDGE_SUBJECTS = {
    "professional_law", "international_law", "jurisprudence", "global_facts",
    "high_school_world_history", "high_school_european_history", "prehistory",
    "professional_medicine", "college_medicine", "clinical_knowledge", "anatomy",
    "human_aging", "human_sexuality", "sociology", "nutrition", "virology",
    "world_religions", "high_school_geography", "miscellaneous",
}


def subject_map():
    m = {}
    for f in glob.glob("raps_data/MMLU/data/test/*_test.csv"):
        subj = os.path.basename(f)[:-len("_test.csv")]
        for q in pd.read_csv(f, header=None, names=COLS)["question"]:
            m[str(q)] = subj
    return m


SMAP = subject_map()


def knowledge_failures():
    """baseline top_k=1 chain failures restricted to knowledge subjects."""
    out = []
    for fp in glob.glob("logs/log_mmlu_2026-06-03-18-59-41_*"):
        try:
            L = json.load(open(fp))
        except Exception:
            continue
        stem = L["task"].split("\nOption A:")[0].strip()
        subj = SMAP.get(stem, "?")
        if subj not in KNOWLEDGE_SUBJECTS:
            continue
        fo = ""
        for st in reversed(L.get("steps", [])):
            if st.get("step") == "Final" and st.get("publications"):
                fo = st["publications"][0]["content"]; break
        gold = L.get("answer")
        if gold in "ABCD" and ds.postprocess_answer(fo) != gold:
            out.append((L["task"], gold, subj))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--sim_threshold", type=float, default=0.30)
    args = p.parse_args()

    fails = knowledge_failures()[:args.limit]
    print(f"REACT-REC:: {len(fails)} knowledge-subject baseline failures", flush=True)

    agents = initialize_agents_from_set("gpt-4o-mini-2024-07-18")
    fa = initialize_final_answerer("gpt-4o-mini-2024-07-18")
    cfg = RAPSConfig(domain="mmlu", max_steps=5, top_k=args.top_k,
                     sim_threshold=args.sim_threshold, entry_index=0)

    recovered = 0
    for i, (task, gold, subj) in enumerate(fails):
        # fresh agents each question to avoid history bleed
        agents = initialize_agents_from_set("gpt-4o-mini-2024-07-18")
        fa = initialize_final_answerer("gpt-4o-mini-2024-07-18")
        coord = RAPSCoordinator(agents, fa, cfg, logger=lambda m: None,
                                answer_extractor=ds.postprocess_answer)
        pred = ds.postprocess_answer(coord.run(task).final_output)
        ok = pred == gold
        recovered += ok
        print(f"PROG:: {i+1}/{len(fails)} [{subj}] pred={pred} gold={gold} "
              f"{'RECOVERED' if ok else 'still-wrong'} | total_recovered={recovered}", flush=True)

    n = len(fails)
    print(f"\nREACT-REC:: recovered {recovered}/{n} = {recovered/n:.3f} of knowledge-subject "
          f"baseline failures (chain=0 by definition)", flush=True)


if __name__ == "__main__":
    main()
