"""
Index the local Wikipedia parquet dump into Elasticsearch (BM25) for offline retrieval.

Chunks each article into ~128-word passages (title prepended), bulk-indexes them, and
re-indexes the result under the analyser the retriever queries. Run once; it takes a
while (millions of passages) and talks only to the Elasticsearch host at ES_URL.
"""
import os
import glob
import sys

import pyarrow.parquet as pq
from elasticsearch import Elasticsearch, helpers

from RAPS.tools.search.bm25_retriever import _bypass_proxy_for

ES_URL = os.environ.get("ES_URL", "http://127.0.0.1:9200")
_bypass_proxy_for(ES_URL)
ES = Elasticsearch(ES_URL, request_timeout=120)
INDEX = "wiki"                                        # the passages as ingested
KSTEM_INDEX = os.environ.get("WIKI_INDEX", "wiki_kstem")   # what the retriever queries
WIKI_DIR = os.environ.get("WIKI_DIR", "wiki/20231101.en")
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


def create_kstem_index():
    """The index the retriever reads, analysed with the light kstem stemmer. The default
    English analyser over-stems the discriminative terms a query relies on — it collapses
    international, internal and internment onto "intern" — so the passages are re-analysed
    rather than queried through it."""
    if ES.indices.exists(index=KSTEM_INDEX):
        print(f"index '{KSTEM_INDEX}' already exists; delete it first to rebuild.")
        return False
    ES.indices.create(index=KSTEM_INDEX, body={
        "settings": {
            "index": {"number_of_shards": 4, "number_of_replicas": 0, "refresh_interval": "-1"},
            "analysis": {"analyzer": {"default": {
                "type": "custom", "tokenizer": "standard",
                "filter": ["lowercase", "kstem"]}}},
        },
        "mappings": {"properties": {"title": {"type": "text"}, "text": {"type": "text"}}},
    })
    return True


def reindex_kstem():
    """Copy the passages of the base index into the one the retriever queries."""
    if not create_kstem_index():
        return
    ES.reindex(body={"source": {"index": INDEX, "size": 2000},
                     "dest": {"index": KSTEM_INDEX}},
               slices="auto", wait_for_completion=True, request_timeout=3600)
    ES.indices.put_settings(index=KSTEM_INDEX, body={"index": {"refresh_interval": "1s"}})
    ES.indices.refresh(index=KSTEM_INDEX)
    print(f"DONE: {KSTEM_INDEX} count={ES.count(index=KSTEM_INDEX)['count']}", flush=True)


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
    reindex_kstem()


if __name__ == "__main__":
    main()
