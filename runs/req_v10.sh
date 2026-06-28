cd /home/lirui/raps_swe/RAPS-main
export HF_ENDPOINT=https://hf-mirror.com LITELLM_LOG=ERROR
PY=/home/lirui/anaconda3/envs/raps_swe/bin/python
printf 'psf__requests-1963\n' > runs/dev1_req.txt
$PY RAPS/swe/run_swebench.py runs/dev1_req.txt runs/raps_req_v10 1 30
$PY -m swebench.harness.run_evaluation -d princeton-nlp/SWE-bench_Lite -s test -p runs/raps_req_v10/preds.json -i psf__requests-1963 -id raps_req_v10 --max_workers 1 --cache_level env --clean True -n swebench > /tmp/eval_req_v10.log 2>&1
echo "=== requests-1963 v10 ==="; grep -iE 'Instances (resolved|unresolved|empty)' /tmp/eval_req_v10.log | tail -3
$PY -c "import json,glob; f=sorted(glob.glob('*raps_req_v10*.json'))[-1]; d=json.load(open(f)); print('RESOLVED:', d.get('resolved_ids'))"
echo "mc fires=$(grep -roh 'multi-candidate' runs/raps_req_v10 2>/dev/null|wc -l) repro_rewrites? check traj"
