#!/bin/bash
# Full-20 RAPS run + resolved eval. The headline RAPS-vs-single-agent number (baseline 2/20).
cd /home/lirui/raps_swe/RAPS-main
export HF_ENDPOINT=https://hf-mirror.com LITELLM_LOG=ERROR
PY=/home/lirui/anaconda3/envs/raps_swe/bin/python

echo "### RAPS full-20 run ###"
$PY RAPS/swe/run_swebench.py runs/eval20.txt runs/raps_full20 4 30

echo "### resolved eval ###"
IDS=$(paste -sd' ' runs/eval20.txt)
$PY -m swebench.harness.run_evaluation -d princeton-nlp/SWE-bench_Lite -s test \
    -p runs/raps_full20/preds.json -i $IDS -id raps_full20 \
    --max_workers 4 --cache_level env --clean True -n swebench > /tmp/eval_full20.log 2>&1

echo "### FULL-20 RESULT ###"
grep -iE 'Instances (resolved|completed|unresolved|with empty|with errors)' /tmp/eval_full20.log | tail -6
ls -t gpt-4o-mini.raps_full20.json passk.raps_full20.json raps_full20*.json 2>/dev/null | head -1 | \
  xargs -I{} $PY -c "import json;d=json.load(open('{}'));print('RESOLVED ids:',d.get('resolved_ids'));print('RAPS full-20 resolved:',d.get('resolved_instances'),'/ 20  (single-agent baseline: 2/20)')"
