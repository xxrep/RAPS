"""
Azure-compatible chat backend for RAPS.

Routes chat completions through the Azure-compatible deployment named in
AZURE_OPENAI_ENDPOINT (default model `gpt-4o-mini-2024-07-18`).

The broker's embedding model is chosen by name: `text-embedding-*` is served by
the OpenAI API, and any other name by the OpenAI-compatible service at
EMBED_BASE_URL, which is where the local Qwen3-Embedding service runs (see
serving/embedding_server.py). Both paths are charged to the token ledger at the
rate of the backbone, as the cost accounting requires.

All endpoints/keys are configurable via environment variables so secrets
never need to live in the repo:
  AZURE_OPENAI_API_KEY   (or OPENAI_API_KEY, or RAPS/llm/azure_key.txt)
  AZURE_OPENAI_ENDPOINT  the Azure-compatible endpoint to call (required)
  AZURE_OPENAI_API_VERSION  default 2024-02-01
  RAPS_TRACE_HEADER      optional "Name: value" request header for a gateway that
                         requires one; a random per-run id is substituted for "<run>"
  EMBED_MODEL            default qwen3-embedding-4b
  EMBED_BASE_URL         default http://127.0.0.1:8200/v1   (locally served models)
  EMBED_API_KEY          (or OPENAI_API_KEY) for an OpenAI-hosted embedding model
  EMBED_OPENAI_BASE_URL  default https://api.openai.com/v1
  RAPS_PRICE_IN/OUT      optional per-1M-token rates overriding the price table
"""
import os
from functools import lru_cache
from typing import List, Optional, Dict, Union

import shortuuid
from openai import AzureOpenAI, AsyncAzureOpenAI, OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from RAPS.llm.llm import LLM
from RAPS.llm.llm_registry import LLMRegistry
from RAPS.utils.const import RAPS_ROOT
from RAPS.utils.globals import Cost, PromptTokens, CompletionTokens

DEFAULT_API_VERSION = "2024-02-01"
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_EMBED_BASE_URL = "http://127.0.0.1:8200/v1"
DEFAULT_EMBED_MODEL = "qwen3-embedding-4b"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
# Embedding model names served by the OpenAI API rather than by a local service.
HOSTED_EMBED_PREFIXES = ("text-embedding-",)

# USD per 1K tokens; substring match against the model name, with a 0.0 fallback.
# These are the rates of the cost ledger reported in Table S.8, namely the OpenAI list
# prices of 1 July 2026: gpt-4o $2.50/$10.00, gpt-4o-mini $0.15/$0.60, gpt-5.2
# $1.75/$14.00 per 1M tokens. "gpt-4o-mini" must precede "gpt-4o", which
# substring-matches it.
_PRICE_PER_1K = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-5.2": {"input": 0.00175, "output": 0.014},
}


def price_override():
    """Per-1K rates taken from RAPS_PRICE_IN / RAPS_PRICE_OUT, which are given in USD per
    1M tokens, or None when they are unset. Setting them prices a run at another date's
    list rates without editing the tables that reproduce Table S.8."""
    if not os.getenv("RAPS_PRICE_IN"):
        return None
    return (float(os.environ["RAPS_PRICE_IN"]) / 1000.0,
            float(os.getenv("RAPS_PRICE_OUT", 0.0)) / 1000.0)


def _resolve_endpoint() -> str:
    """The Azure-compatible endpoint to call, which has no default because it is specific
    to the deployment."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT is not set. Point it at your Azure-compatible "
            "deployment, or set RAPS_LLM_BACKEND=GPTChat to call an OpenAI-compatible "
            "endpoint instead.")
    return endpoint


def _trace_header() -> Dict[str, str]:
    """The request header a gateway may require for tracing, given as "Name: value" in
    RAPS_TRACE_HEADER, where "<run>" stands for a fresh per-run identifier."""
    spec = os.getenv("RAPS_TRACE_HEADER", "")
    name, sep, value = spec.partition(":")
    if not sep or not name.strip():
        return {}
    run_id = f"raps-{shortuuid.ShortUUID().random(length=8)}"
    return {name.strip(): value.strip().replace("<run>", run_id)}


def _resolve_key() -> str:
    key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        key_path = RAPS_ROOT / "RAPS/llm/azure_key.txt"
        if key_path.exists():
            key = key_path.read_text().strip()
    if not key:
        raise ValueError(
            "Azure API key not found. Set AZURE_OPENAI_API_KEY (or OPENAI_API_KEY), "
            "or create RAPS/llm/azure_key.txt."
        )
    return key


def _price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    override = price_override()
    if override:
        return prompt_tokens * override[0] / 1000 + completion_tokens * override[1] / 1000
    for name, rate in _PRICE_PER_1K.items():
        if name in model:
            return prompt_tokens * rate["input"] / 1000 + completion_tokens * rate["output"] / 1000
    return 0.0


@lru_cache(maxsize=None)
def _openai_client(base_url: str, api_key: str) -> OpenAI:
    """One client per (endpoint, key), shared across the agents that route to it."""
    return OpenAI(base_url=base_url, api_key=api_key)


def embedding_client(model: str) -> OpenAI:
    """The client that serves `model`: an OpenAI-hosted embedding model is reached through
    the OpenAI API, and any other name through the OpenAI-compatible service that hosts it
    locally. Selecting the broker's embedding model is therefore a matter of naming it in
    EMBED_MODEL, with EMBED_BASE_URL pointing at the local service."""
    if model.startswith(HOSTED_EMBED_PREFIXES):
        key = os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                f"Embedding model {model!r} is served by the OpenAI API, which needs "
                "EMBED_API_KEY (or OPENAI_API_KEY). Set EMBED_MODEL to a locally served "
                f"model such as {DEFAULT_EMBED_MODEL!r} to use the local service instead.")
        return _openai_client(os.getenv("EMBED_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL), key)
    return _openai_client(os.getenv("EMBED_BASE_URL", DEFAULT_EMBED_BASE_URL), "local")


@LLMRegistry.register('AzureGPTChat')
class AzureGPTChat(LLM):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name or DEFAULT_MODEL
        self.endpoint = _resolve_endpoint()
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)
        self.headers = _trace_header()
        self.embed_model = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

        # Allow a deterministic override (e.g. RAPS_TEMPERATURE=0 for reproducible A/Bs).
        _temp = os.getenv("RAPS_TEMPERATURE")
        if _temp is not None:
            self.DEFAULT_TEMPERATURE = float(_temp)

        key = _resolve_key()
        self._client = AzureOpenAI(
            api_key=key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            default_headers=self.headers,
        )
        self._aclient: Optional[AsyncAzureOpenAI] = None

    # ---------------- chat ----------------

    @retry(wait=wait_random_exponential(max=50), stop=stop_after_attempt(3))
    def gen(
        self,
        messages: Union[str, List[Dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.DEFAULT_TEMPERATURE if temperature is None else temperature,
            max_tokens=self.DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
        )
        self._account(resp)
        return resp.choices[0].message.content

    @retry(wait=wait_random_exponential(max=50), stop=stop_after_attempt(3))
    async def agen(
        self,
        messages: Union[str, List[Dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> str:
        if self._aclient is None:
            self._aclient = AsyncAzureOpenAI(
                api_key=_resolve_key(),
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                default_headers=self.headers,
            )
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        resp = await self._aclient.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.DEFAULT_TEMPERATURE if temperature is None else temperature,
            max_tokens=self.DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
        )
        self._account(resp)
        return resp.choices[0].message.content

    def _account(self, resp) -> None:
        """Update global cost/token counters from the API usage field."""
        try:
            usage = resp.usage
            prompt_tokens = int(usage.prompt_tokens)
            completion_tokens = int(usage.completion_tokens)
        except Exception:
            return
        Cost.instance().value += _price(self.model_name, prompt_tokens, completion_tokens)
        PromptTokens.instance().value += prompt_tokens
        CompletionTokens.instance().value += completion_tokens

    def _account_embedding(self, resp) -> None:
        """Charge the broker's embedding call to the same ledger as a chat call, at the
        rate of the backbone rather than the lower rate of the embedding model, so the
        per-instance budget covers every call the coordination issues."""
        try:
            prompt_tokens = int(resp.usage.prompt_tokens)
        except Exception:
            return
        Cost.instance().value += _price(self.model_name, prompt_tokens, 0)
        PromptTokens.instance().value += prompt_tokens

    # ---------------- embeddings (local Qwen3 service) ----------------

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        return self.get_embeddings([text], model=model)[0]

    @retry(wait=wait_random_exponential(max=20), stop=stop_after_attempt(3))
    def get_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        cleaned = [(t if isinstance(t, str) else str(t)).replace("\n", " ") for t in texts]
        embed_model = model or self.embed_model
        resp = embedding_client(embed_model).embeddings.create(
            model=embed_model, input=cleaned
        )
        self._account_embedding(resp)
        # preserve request order
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in ordered]
