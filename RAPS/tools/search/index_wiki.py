"""
Index the local Wikipedia parquet dump into Elasticsearch (BM25) for offline retrieval.

Chunks each article into ~128-word passages (title prepended) and bulk-indexes them.
Run once; takes a while (millions of passages). No network/proxy: talks to local ES.
"""
import os
import glob
import sys

import pyarrow.parquet as pq
from elasticsearch import Elasticsearch, helpers

os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

ES = Elasticsearch("http://127.0.0.1:9200", request_timeout=120)
INDEX = "wiki"
WIKI_DIR = "/opt/tiger/RAPS/wiki/20231101.en"
PASSAGE_WORDS = 128


def create_index():
    if ES.indices.exists(index=INDEX):
        print(f"index '{INDEX}' already exists; delete it first to re-index.")
        return
    ES.indices.create(index=INDEX, body={
        "settings": {
            "number_of_shards": 4,
            "number_of_replicas": 0,
            "refresh_interval": "-1",
            "analysis": {"analyzer": {"default": {"type": "english"}}},
            "similarity": {"default": {"type": "BM25"}},
        },
        "mappings": {"properties": {
            "title": {"type": "text"},
            "text": {"type": "text"},
        }},
    })
    print(f"created index '{INDEX}'")


def passages(title, text):
    words = (text or "").split()
    for i in range(0, len(words), PASSAGE_WORDS):
        chunk = " ".join(words[i:i + PASSAGE_WORDS])
        if len(chunk) > 50:
            yield {"_index": INDEX, "title": title, "text": chunk}


def actions():
    files = sorted(glob.glob(f"{WIKI_DIR}/*.parquet"))
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=2000, columns=["title", "text"]):
            d = batch.to_pydict()
            for title, text in zip(d["title"], d["text"]):
                yield from passages(title, text)
        print(f"[file {fi+1}/{len(files)}] {os.path.basename(f)} done", flush=True)


def main():
    create_index()
    n = 0
    for ok, _ in helpers.parallel_bulk(ES, actions(), thread_count=8, chunk_size=2000,
                                       raise_on_error=False, queue_size=8):
        n += 1
        if n % 200000 == 0:
            print(f"  indexed {n} passages", flush=True)
    ES.indices.put_settings(index=INDEX, body={"index": {"refresh_interval": "1s"}})
    ES.indices.refresh(index=INDEX)
    print(f"DONE: indexed {n} passages. count={ES.count(index=INDEX)['count']}", flush=True)


if __name__ == "__main__":
    main()
