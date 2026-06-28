# RAPS Embedding Service (Qwen3-Embedding-4B, GPU 0)

OpenAI-compatible embedding server used by RAPS broker routing, so it replaces the
old `text-embedding-3-small` API calls (which the internal modelhub gateway doesn't expose).

## What runs where
- **conda env `embed`** (Python 3.12, torch 2.4.1+**cu121** — required because the box driver is CUDA 12.2; the
  default cu130 torch is incompatible). Holds the model + serving stack.
- **conda env `raps`** — the RAPS app itself (talks to this service over HTTP).
- Model weights: `/opt/tiger/RAPS/models/Qwen3-Embedding-4B` (bf16, ~8 GB on GPU 0, dim **2560**).

## Start / stop
```bash
# start (GPU 0 only)
nohup bash serving/start_embedding.sh > serving/embed_server.log 2>&1 &

# health
curl -s http://127.0.0.1:8200/health
# -> {"status":"ok","model":"qwen3-embedding-4b","dim":2560,"device":"cuda","dtype":"bfloat16"}

# stop
pkill -f serving/embedding_server.py
```

## Use from the app (OpenAI client)
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8200/v1", api_key="local")
r = client.embeddings.create(model="qwen3-embedding-4b", input=["text a", "text b"])
vec = r.data[0].embedding          # 2560-d, already L2-normalized -> cosine == dot
```

## Config (env vars)
`EMBED_PORT` (8200), `EMBED_MODEL_PATH`, `EMBED_DTYPE` (bfloat16), `EMBED_MAX_BATCH` (64),
`CUDA_VISIBLE_DEVICES` (pinned to 0 in `start_embedding.sh`).

## Notes
- Embeddings are L2-normalized server-side; the broker can use plain dot product.
- GPU calls are serialized by a lock (one model copy, GPU 0). Realistic broker batch (~6 texts) ≈ 80 ms/call.
- For higher throughput later: switch `encoding_format` to base64 / float16 over the wire, or move to vLLM/TEI.
