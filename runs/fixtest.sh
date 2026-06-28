cd /home/lirui/raps_swe/RAPS-main
export HF_ENDPOINT=https://hf-mirror.com LITELLM_LOG=ERROR
PY=/home/lirui/anaconda3/envs/raps_swe/bin/python
$PY RAPS/swe/run_swebench.py runs/fix6.txt runs/raps_fix6 4 30
IDS=$(paste -sd' ' runs/fix6.txt)
$PY -m swebench.harness.run_evaluation -d princeton-nlp/SWE-bench_Lite -s test -p runs/raps_fix6/preds.json -i $IDS -id raps_fix6 --max_workers 3 --cache_level env --clean True -n swebench > /tmp/eval_fix6.log 2>&1
echo "### FIX6 RESULT ###"
$PY -c "import json;m=json.load(open('runs/raps_fix6/metrics.json'));print('patches_nonempty:',m['summary']['patches_nonempty'],'/6')"
grep -iE 'Instances (resolved|unresolved|empty|errors)' /tmp/eval_fix6.log | tail -4
$PY -c "import json,glob;f=sorted(glob.glob('*raps_fix6*.json'))[-1];d=json.load(open(f));print('RESOLVED:',d.get('resolved_ids'))"
