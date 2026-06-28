#!/usr/bin/env bash
# Launch the Qwen3-Embedding-4B service on GPU 0.
set -euo pipefail

export PYTHONPATH=""
export CUDA_VISIBLE_DEVICES=0          # GPU 0 only
export EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-/opt/tiger/RAPS/models/Qwen3-Embedding-4B}"
export EMBED_PORT="${EMBED_PORT:-8200}"
export TOKENIZERS_PARALLELISM=false

PY="$HOME/miniconda3/envs/embed/bin/python"
exec "$PY" /opt/tiger/RAPS/serving/embedding_server.py
