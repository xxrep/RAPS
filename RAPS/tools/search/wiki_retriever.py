"""
Real Wikipedia retriever for RAG.

Uses the Wikipedia REST/Action API directly with a proper User-Agent (the legacy
`wikipedia` package fails with HTTP 403 because it sends no User-Agent). Works
through the environment HTTP(S) proxy.
"""
import requests

_API = "https://en.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "RAPS-research/0.1 (academic MMLU benchmark; contact: research@example.com)"}
_session = requests.Session()
_session.headers.update(_HEADERS)


def _search_titles(query: str, limit: int = 2, timeout: int = 10):
    try:
        r = _session.get(_API, params={"action": "query", "list": "search",
                                        "srsearch": query[:300], "srlimit": limit,
                                        "format": "json"}, timeout=timeout)
        return [h["title"] for h in r.json().get("query", {}).get("search", [])]
    except Exception:
        return []


def _intro_extract(title: str, chars: int = 600, timeout: int = 10) -> str:
    try:
        r = _session.get(_API, params={"action": "query", "prop": "extracts",
                                        "exintro": 1, "explaintext": 1, "titles": title,
                                        "format": "json"}, timeout=timeout)
        pages = r.json().get("query", {}).get("pages", {})
        for p in pages.values():
            ext = p.get("extract", "")
            if ext:
                return ext[:chars]
    except Exception:
        pass
    return ""


def retrieve(queries, k_per_query: int = 1, chars: int = 600) -> str:
    """Search Wikipedia for each query, return concatenated intro extracts (deduped)."""
    if isinstance(queries, str):
        queries = [queries]
    seen, blocks = set(), []
    for q in queries:
        for title in _search_titles(q, limit=k_per_query):
            if title in seen:
                continue
            seen.add(title)
            ext = _intro_extract(title, chars=chars)
            if ext:
                blocks.append(f"[Wikipedia: {title}] {ext}")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    print(retrieve(["Calvin cycle photosynthesis", "Krebs cycle"]))
