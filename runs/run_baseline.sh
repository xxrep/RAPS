#!/bin/bash
# Single-agent baseline: mini-swe-agent + gpt-4o-mini on the fixed 20-task eval set.
set -euo pipefail
cd /home/lirui/raps_swe/RAPS-main

FILTER="^($(paste -sd'|' runs/eval20.txt))$"
export HF_ENDPOINT=https://hf-mirror.com
export LITELLM_LOG=ERROR

echo "filter: $FILTER"
exec /home/lirui/anaconda3/envs/raps_swe/bin/python -m minisweagent.run.benchmarks.swebench \
  --subset lite --split test --filter "$FILTER" \
  -m gpt-4o-mini -w 4 -o runs/baseline_mini_4omini \
  -c swebench.yaml -c environment.pull_timeout=1800
