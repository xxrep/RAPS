"""
RAPS MMLU pipeline WITH 5-shot exemplars (standard MMLU protocol). MMLU-only.

Per-subject 5-shot exemplars (from the official dev split) are prepended to the task,
so every expert and the Final Answerer reason with worked examples. Everything else
(the multi-agent chain) is unchanged. Compared to the 0-shot chain (from the original
max_steps=5 logs) on the same questions.
"""
import sys, os, re, glob, json, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from RAPS.core import RAPSCoordinator, RAPSConfig
from RAPS.prompt.mmlu_prompt_set import MMLUPromptSet
from raps_data.mmlu_dataset import MMLUDataset
from experiments.run_mmlu import initialize_agents_from_set, initialize_final_answerer

COLS = ["question", "A", "B", "C", "D", "answer"]


def subject_map():
    m = {}
    for f in glob.glob("raps_data/MMLU/data/test/*_test.csv"):
        subj = os.path.basename(f)[:-len("_test.csv")]
        for q in pd.read_csv(f, header=None, names=COLS)["question"]:
            m[str(q)] = subj
    return m


def shots(subject, k=5):
    df = pd.read_csv(f"raps_data/MMLU/data/dev/{subject}_dev.csv", header=None, names=COLS).head(k)
    blocks = [f"{r['question']}\nOption A: {r['A']}\nOption B: {r['B']}\nOption C: {r['C']}\n"
              f"Option D: {r['D']}\nAnswer: {r['answer']}" for _, r in df.iterrows()]
    return "\n\n".join(blocks)


def chain_baseline_from_logs():
    ds = MMLUDataset(split="test")
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
        m[L["task"]] = ds.postprocess_answer(fo)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--max_steps", type=int, default=5)
    args = p.parse_args()
    ds = MMLUDataset(split="test")
    smap = subject_map()
    chain = chain_baseline_from_logs()

    agents = initialize_agents_from_set("gpt-4o-mini-2024-07-18")
    fa = initialize_final_answerer("gpt-4o-mini-2024-07-18")
    cfg = RAPSConfig(domain="mmlu", max_steps=args.max_steps, top_k=1, sim_threshold=0.3, entry_index=0)
    coord = RAPSCoordinator(agents, fa, cfg, logger=lambda m: None, answer_extractor=ds.postprocess_answer)

    new_ok = chain_ok = matched = 0
    fixed, broke = [], []
    for i in range(args.n):
        rec = ds[i]
        q = MMLUDataset.record_to_input(rec)["task"]
        gold = MMLUDataset.record_to_target_answer(rec)
        subj = smap.get(str(rec["question"]))
        if subj:
            task = (f"The following are example multiple-choice questions (with answers) about "
                    f"{subj.replace('_', ' ')}:\n\n{shots(subj)}\n\n"
                    f"Now answer this question:\n{q}")
        else:
            task = q
        npred = ds.postprocess_answer(coord.run(task).final_output)
        new_ok += npred == gold
        cpred = chain.get(q)
        if cpred is not None:
            matched += 1
            chain_ok += cpred == gold
            if npred == gold and cpred != gold: fixed.append(i)
            if cpred == gold and npred != gold: broke.append(i)
        print(f"PROG:: {i+1}/{args.n} 5shot_chain={new_ok} | 0shot_chain={chain_ok}/{matched}", flush=True)
    print(f"\nRESULT:: 5-shot pipeline = {new_ok}/{args.n} = {new_ok/args.n:.3f}", flush=True)
    if matched:
        print(f"RESULT:: 0-shot chain (same {matched}) = {chain_ok}/{matched} = {chain_ok/matched:.3f}", flush=True)
        print(f"RESULT:: fixed {len(fixed)} {fixed}; broke {len(broke)} {broke}", flush=True)


if __name__ == "__main__":
    main()
