"""
Azure / modelhub gateway chat backend for RAPS.

Routes chat completions through the internal Azure-compatible gateway
(default model `gpt-4o-mini-2024-07-18`), and routes embeddings to the
local Qwen3-Embedding service (see serving/embedding_server.py), which
replaces the OpenAI `text-embedding-3-small` calls that the gateway does
not expose.

All endpoints/keys are configurable via environment variables so secrets
never need to live in the repo:
  AZURE_OPENAI_API_KEY   (or OPENAI_API_KEY, or RAPS/llm/azure_key.txt)
  AZURE_OPENAI_ENDPOINT  default https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl
  AZURE_OPENAI_API_VERSION  default 2024-02-01
  X_TT_LOGID             default raps-<random>
  EMBED_BASE_URL         default http://127.0.0.1:8200/v1
  EMBED_MODEL            default qwen3-embedding-4b
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

DEFAULT_AZURE_ENDPOINT = "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"
DEFAULT_API_VERSION = "2024-02-01"
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_EMBED_BASE_URL = "http://127.0.0.1:8200/v1"
DEFAULT_EMBED_MODEL = "qwen3-embedding-4b"

# USD per 1K tokens; substring match against the model name, with a 0.0 fallback.
_PRICE_PER_1K = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
}


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
    for name, rate in _PRICE_PER_1K.items():
        if name in model:
            return prompt_tokens * rate["input"] / 1000 + completion_tokens * rate["output"] / 1000
    return 0.0


@lru_cache(maxsize=None)
def _embedding_client() -> OpenAI:
    """Shared client for the local Qwen3 embedding service (OpenAI-compatible)."""
    base_url = os.getenv("EMBED_BASE_URL", DEFAULT_EMBED_BASE_URL)
    return OpenAI(base_url=base_url, api_key="local")


@LLMRegistry.register('AzureGPTChat')
class AzureGPTChat(LLM):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name or DEFAULT_MODEL
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT)
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)
        self.logid = os.getenv("X_TT_LOGID", f"raps-{shortuuid.ShortUUID().random(length=8)}")
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
            default_headers={"X-TT-LOGID": self.logid},
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
                default_headers={"X-TT-LOGID": self.logid},
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

    # ---------------- embeddings (local Qwen3 service) ----------------

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        return self.get_embeddings([text], model=model)[0]

    @retry(wait=wait_random_exponential(max=20), stop=stop_after_attempt(3))
    def get_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        cleaned = [(t if isinstance(t, str) else str(t)).replace("\n", " ") for t in texts]
        resp = _embedding_client().embeddings.create(
            model=model or self.embed_model, input=cleaned
        )
        # preserve request order
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in ordered]
