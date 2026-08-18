"""
OpenAI-compatible chat backend for RAPS.

Works with the public OpenAI API or any OpenAI-compatible gateway, selected via
OPENAI_BASE_URL. A hosted embedding model named in
EMBED_MODEL is served by that same endpoint, which already holds this backend's
key; any other name is served by the OpenAI-compatible service at EMBED_BASE_URL,
so the broker's embedding model is chosen the same way on every backbone.

Config (all via env so secrets stay out of the repo):
  OPENAI_API_KEY      or a gitignored file under RAPS/llm/ (see _resolve_key)
  OPENAI_BASE_URL     default https://api.openai.com/v1
  EMBED_MODEL         default text-embedding-3-small
  EMBED_BASE_URL      endpoint for a locally served embedding model
  RAPS_TEMPERATURE    optional deterministic override (e.g. 0)
  RAPS_PRICE_IN/OUT   optional per-1M-token rates overriding the price table

Activated by RAPS_LLM_BACKEND=GPTChat (the registry default is AzureGPTChat).
"""
import os
from typing import List, Union, Optional, Dict

import httpx
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from RAPS.llm.llm import LLM
from RAPS.llm.llm_registry import LLMRegistry
from RAPS.llm.azure_chat import embedding_client, price_override, HOSTED_EMBED_PREFIXES
from RAPS.utils.const import RAPS_ROOT
from RAPS.utils.globals import Cost, PromptTokens, CompletionTokens

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# USD per 1K tokens; substring match against the model name, 0.0 fallback. These are the
# rates of the cost ledger reported in Table S.8, namely the OpenAI list prices of
# 1 July 2026. "gpt-4o-mini" must precede "gpt-4o" and "gpt-4.1-mini" precede "gpt-4.1",
# each substring-matching the next.
_PRICE_PER_1K = {
    "gpt-4o-mini":  {"input": 0.00015, "output": 0.0006},
    "gpt-4o":       {"input": 0.0025,  "output": 0.010},
    "gpt-5.2":      {"input": 0.00175, "output": 0.014},
    "gpt-4.1-mini": {"input": 0.0004,  "output": 0.0016},
    "gpt-4.1":      {"input": 0.002,   "output": 0.008},
}


def _resolve_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        for fname in ("openai_key.txt", "gateway_key.txt"):
            p = RAPS_ROOT / "RAPS/llm" / fname
            if p.exists():
                key = p.read_text().strip()
                if key:
                    break
    if not key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY or create "
            "RAPS/llm/openai_key.txt."
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


@LLMRegistry.register('GPTChat')
class GPTChat(LLM):
    def __init__(self, model_name: str = 'gpt-4o-mini'):
        self.model_name = model_name or 'gpt-4o-mini'
        self.base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        self.embed_model = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

        _temp = os.getenv("RAPS_TEMPERATURE")
        if _temp is not None:
            self.DEFAULT_TEMPERATURE = float(_temp)

        # trust_env=False -> ignore an ambient HTTP proxy and reach the endpoint directly.
        self.client = OpenAI(
            api_key=_resolve_key(),
            base_url=self.base_url,
            http_client=httpx.Client(trust_env=False, timeout=httpx.Timeout(120.0, connect=15.0)),
        )

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
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.DEFAULT_TEMPERATURE if temperature is None else temperature,
            max_tokens=self.DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
        )
        self._account(resp)
        return resp.choices[0].message.content

    def _account(self, resp) -> None:
        try:
            usage = resp.usage
            pt, ct = int(usage.prompt_tokens), int(usage.completion_tokens)
        except Exception:
            return
        Cost.instance().value += _price(self.model_name, pt, ct)
        PromptTokens.instance().value += pt
        CompletionTokens.instance().value += ct

    def _account_embedding(self, resp) -> None:
        """Charge the broker's embedding call at the rate of the backbone, matching the
        chat ledger so the per-instance budget covers every coordination call."""
        try:
            pt = int(resp.usage.prompt_tokens)
        except Exception:
            return
        Cost.instance().value += _price(self.model_name, pt, 0)
        PromptTokens.instance().value += pt

    # ---------------- embeddings ----------------

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        return self.get_embeddings([text], model=model)[0]

    @retry(wait=wait_random_exponential(max=20), stop=stop_after_attempt(3))
    def get_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        cleaned = [(t if isinstance(t, str) else str(t)).replace("\n", " ") for t in texts]
        embed_model = model or self.embed_model
        # A hosted embedding model comes from this backend's own endpoint, which already
        # holds its key; any other name is served by the endpoint that hosts it.
        client = (self.client if embed_model.startswith(HOSTED_EMBED_PREFIXES)
                  else embedding_client(embed_model))
        resp = client.embeddings.create(model=embed_model, input=cleaned)
        self._account_embedding(resp)
        ordered = sorted(resp.data, key=lambda d: d.index)  # preserve request order
        return [d.embedding for d in ordered]
