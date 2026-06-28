"""
Local Wikipedia BM25 retriever backed by Elasticsearch (offline, no proxy, no rate limit).
Index built by index_wiki.py (~27M ~128-word passages from the 2023-11 English dump).
"""
import os
from elasticsearch import Elasticsearch

os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")

_ES = Elasticsearch("http://127.0.0.1:9200", request_timeout=30)
# wiki_kstem uses a light (kstem) stemmer that avoids the `english` analyzer's
# over-stemming (international/internal/internment -> "intern"). Override via WIKI_INDEX.
INDEX = os.environ.get("WIKI_INDEX", "wiki_kstem")


def bm25_retrieve(queries, k: int = 3, chars: int = 600) -> str:
    """Search the local Wikipedia BM25 index; return concatenated top passages (deduped)."""
    if isinstance(queries, str):
        queries = [queries]
    seen, blocks = set(), []
    for q in queries:
        if not q:
            continue
        try:
            # text-only match (NO title boost: a title^2 boost made short queries like
            # "Injured State" match the baseball "Injured list" article).
            res = _ES.search(index=INDEX, size=k, query={"match": {"text": q}})
        except Exception:
            continue
        for hit in res["hits"]["hits"]:
            src = hit["_source"]
            key = (src["title"], src["text"][:40])
            if key in seen:
                continue
            seen.add(key)
            blocks.append(f"[{src['title']}] {src['text'][:chars]}")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    print(bm25_retrieve(["Factory Acts work hours child labour regulation 19th century Britain"], k=2))
