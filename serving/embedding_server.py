"""
OpenAI-compatible embedding server for Qwen3-Embedding-4B (GPU 0).

Exposes:
  GET  /health
  POST /v1/embeddings        (OpenAI Embeddings API schema)

So the RAPS app can talk to it via the standard OpenAI client:
    client = OpenAI(base_url="http://127.0.0.1:8200/v1", api_key="x")
    client.embeddings.create(model="qwen3-embedding-4b", input=[...])

Config via env vars (all optional):
  EMBED_MODEL_PATH   local path or hub id of the embedding weights
                     (default Qwen/Qwen3-Embedding-4B)
  EMBED_MODEL_NAME   default qwen3-embedding-4b   (the id echoed back)
  EMBED_DEVICE       default cuda
  EMBED_DTYPE        default bfloat16
  EMBED_MAX_BATCH    default 64
  EMBED_PORT         default 8200
  EMBED_HOST         default 127.0.0.1 (set 0.0.0.0 to serve other machines)
"""
import os
import asyncio
import threading
from typing import List, Union, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", "Qwen/Qwen3-Embedding-4B")
MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "qwen3-embedding-4b")
DEVICE = os.environ.get("EMBED_DEVICE", "cuda")
DTYPE = os.environ.get("EMBED_DTYPE", "bfloat16")
MAX_BATCH = int(os.environ.get("EMBED_MAX_BATCH", "64"))
PORT = int(os.environ.get("EMBED_PORT", "8200"))
HOST = os.environ.get("EMBED_HOST", "127.0.0.1")

print(f"[embed] loading {MODEL_PATH} on {DEVICE} ({DTYPE}) ...", flush=True)
_dtype = getattr(torch, DTYPE)
model = SentenceTransformer(MODEL_PATH, model_kwargs={"dtype": _dtype}, device=DEVICE)
model.eval()
EMBED_DIM = model.get_sentence_embedding_dimension()
# Serialize GPU access so concurrent HTTP requests don't contend on CUDA.
_gpu_lock = threading.Lock()
print(f"[embed] ready: dim={EMBED_DIM}", flush=True)

app = FastAPI(title="RAPS Embedding Server")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None
    # accepted for OpenAI compatibility, ignored:
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None


def _encode(texts: List[str]):
    with torch.inference_mode():
        return model.encode(
            texts,
            normalize_embeddings=True,   # cosine == dot product
            batch_size=MAX_BATCH,
            convert_to_numpy=True,
        )


def _count_tokens(texts: List[str]) -> int:
    """Tokens the model itself reads, which is what the caller's cost ledger charges the
    call by. Counted with the model's own tokenizer rather than estimated from words."""
    encoded = model.tokenizer(texts)["input_ids"]
    return sum(len(ids) for ids in encoded)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "dim": EMBED_DIM,
            "device": DEVICE, "dtype": DTYPE}


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    inputs = [req.input] if isinstance(req.input, str) else list(req.input)
    loop = asyncio.get_running_loop()

    def _run():
        with _gpu_lock:
            return _encode(inputs)

    emb = await loop.run_in_executor(None, _run)
    data = [{"object": "embedding", "index": i, "embedding": emb[i].tolist()}
            for i in range(len(inputs))]
    n_tok = _count_tokens(inputs)
    return {
        "object": "list",
        "data": data,
        "model": req.model or MODEL_NAME,
        "usage": {"prompt_tokens": n_tok, "total_tokens": n_tok},
    }


if __name__ == "__main__":
    # single worker: one model copy on GPU 0, GPU calls serialized by _gpu_lock
    uvicorn.run(app, host=HOST, port=PORT, workers=1, log_level="info")
