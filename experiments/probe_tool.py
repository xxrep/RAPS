"""Quick visible head-to-head: baseline vs tool_verify (PAL) on a few GSM8K questions."""
import sys, os, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RAPS.core import RAPSCoordinator, RAPSConfig
from raps_data.gsm8k_dataset import gsm_data_process, gsm_get_predict
from experiments.run_gsm8k import (initialize_agents_from_set, initialize_final_answerer,
                                   is_correct, load_jsonl)

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=20)
ap.add_argument("--n", type=int, default=8)
args = ap.parse_args()

data = gsm_data_process(load_jsonl("raps_data/gsm8k/gsm8k.jsonl"))[args.start:args.start + args.n]

def mk(tool):
    ag = initialize_agents_from_set("gpt-4o-mini-2024-07-18")
    fa = initialize_final_answerer("gpt-4o-mini-2024-07-18")
    return RAPSCoordinator(ag, fa, RAPSConfig(domain="gsm8k", max_steps=2, top_k=1,
                           entry_index=1, tool_verify=tool),
                           logger=lambda m: None, answer_extractor=gsm_get_predict)

cb, ct = mk(False), mk(True)
print(f"{'idx':<6}{'gold':<10}{'baseline':<14}{'tool_verify':<14}{'change'}", flush=True)
fixes = breaks = bcorr = tcorr = 0
for i, ex in enumerate(data):
    pb = gsm_get_predict(cb.run(ex["task"]).final_output); ob = is_correct(pb, ex["answer"])
    pt = gsm_get_predict(ct.run(ex["task"]).final_output); ot = is_correct(pt, ex["answer"])
    bcorr += ob; tcorr += ot
    tag = "  <== FIX" if (ot and not ob) else ("  <== BREAK" if (ob and not ot) else "")
    fixes += (ot and not ob); breaks += (ob and not ot)
    print(f"{args.start+i:<6}{str(ex['answer']):<10}{pb+(' OK' if ob else ' X'):<14}{pt+(' OK' if ot else ' X'):<14}{tag}", flush=True)
print(f"\nTOTAL baseline {bcorr}/{len(data)}  tool_verify {tcorr}/{len(data)}  | fixes={fixes} breaks={breaks}", flush=True)
