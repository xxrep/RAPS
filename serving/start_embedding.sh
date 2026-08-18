#!/usr/bin/env bash
# Launch the Qwen3-Embedding-4B service on GPU 0.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0          # GPU 0 only
export EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-Qwen/Qwen3-Embedding-4B}"
export EMBED_PORT="${EMBED_PORT:-8200}"
export EMBED_HOST="${EMBED_HOST:-127.0.0.1}"
export TOKENIZERS_PARALLELISM=false

# The interpreter that has torch and sentence-transformers; set EMBED_PYTHON to
# point at another environment.
PY="${EMBED_PYTHON:-$(command -v python3 || command -v python)}"
exec "$PY" "$(dirname "$0")/embedding_server.py"
