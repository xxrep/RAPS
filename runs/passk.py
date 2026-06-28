#!/usr/bin/env python3
"""DIAGNOSTIC pass@k (NOT a fair number): for each dev6 instance, give gpt-4o-mini the gold
file(s) + issue (oracle localization), sample K candidate fixes, and judge EACH against the REAL
FAIL_TO_PASS test via the swebench harness. Answers definitively: can gpt-4o-mini produce a
correct fix for these tasks at all (removing localization + self-repro confounds)?
"""
import difflib, json, os, re, subprocess, sys
sys.path.insert(0, "/home/lirui/raps_swe/RAPS-main")
import litellm
from datasets import load_dataset

K = 5
PY = "/home/lirui/anaconda3/envs/raps_swe/bin/python"
ROOT = "/home/lirui/raps_swe/RAPS-main"
ds = {x["instance_id"]: x for x in load_dataset("princeton-nlp/SWE-bench_Lite", split="test")}
ids = [l.strip() for l in open(f"{ROOT}/runs/dev6.txt") if l.strip()]
os.makedirs(f"{ROOT}/runs/passk", exist_ok=True)


def gold_files(patch):
    return sorted(set(re.findall(r"^\+\+\+ b/(.+)$", patch, re.M)))


def image(iid):
    return f"docker.io/swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest".lower()


def cat_file(iid, path):
    r = subprocess.run(["docker", "run", "--rm", image(iid), "cat", f"/testbed/{path}"],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


SYS = ("You are an expert bug-fixer. Output exactly ONE SEARCH/REPLACE block that fixes the issue:\n"
       "<<<<<<< SEARCH\n<exact existing lines copied verbatim from the file>\n=======\n"
       "<the replacement lines>\n>>>>>>> REPLACE\nCopy the SEARCH text EXACTLY (incl. indentation). "
       "Output ONLY the block, no prose.")

preds = {k: {} for k in range(K)}
for iid in ids:
    inst = ds[iid]
    f = gold_files(inst["patch"])[0]            # dev6 golds are single-file
    content = cat_file(iid, f)
    if content is None:
        print("cat FAILED", iid); continue
    user = f"# Issue\n{inst['problem_statement'][:4500]}\n\n# File: {f}\n```python\n{content[:13000]}\n```"
    for k in range(K):
        patch = ""
        try:
            r = litellm.completion(model="gpt-4o-mini",
                                   messages=[{"role": "system", "content": SYS},
                                             {"role": "user", "content": user}],
                                   temperature=0.0 if k == 0 else 0.8, max_tokens=1600, drop_params=True)
            out = r.choices[0].message.content or ""
            m = re.search(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)(?:\n>>>>>>> REPLACE|\Z)", out, re.DOTALL)
            if m and m.group(1) in content:
                new = content.replace(m.group(1), m.group(2), 1)
                diff = "".join(difflib.unified_diff(content.splitlines(True), new.splitlines(True),
                                                    fromfile=f"a/{f}", tofile=f"b/{f}"))
                if diff:
                    patch = f"diff --git a/{f} b/{f}\n" + diff
        except Exception as e:
            print("gen err", iid, k, str(e)[:60])
        preds[k][iid] = {"model_name_or_path": "passk", "instance_id": iid, "model_patch": patch}
    print("generated", iid, "| nonempty:", sum(1 for k in range(K) if preds[k][iid]["model_patch"]))

resolved_any = {iid: False for iid in ids}
for k in range(K):
    pf = f"{ROOT}/runs/passk/preds_{k}.json"
    json.dump(preds[k], open(pf, "w"), indent=2)
    subprocess.run([PY, "-m", "swebench.harness.run_evaluation", "-d", "princeton-nlp/SWE-bench_Lite",
                    "-s", "test", "-p", pf, "-i", *ids, "-id", f"passk_{k}",
                    "--max_workers", "4", "--cache_level", "env", "--clean", "True", "-n", "swebench"],
                   cwd=ROOT, env={**os.environ, "HF_ENDPOINT": "https://hf-mirror.com"},
                   capture_output=True, text=True)
    rep = f"{ROOT}/passk.passk_{k}.json"
    if os.path.exists(rep):
        rids = json.load(open(rep)).get("resolved_ids", [])
        for iid in rids:
            resolved_any[iid] = True
        print(f"[k={k}] resolved: {rids}")

print("\n=== pass@%d (oracle file, judged by REAL tests) ===" % K)
for iid in ids:
    print(f"  {iid:32s} {'RESOLVED by >=1 candidate' if resolved_any[iid] else 'no candidate resolved'}")
print(f"pass@{K}: {sum(resolved_any.values())}/{len(ids)}")
