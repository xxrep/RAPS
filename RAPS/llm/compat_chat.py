"""OpenAI-compatible chat backends for the cross-backbone comparison (Table S.21).

One client class covers any OpenAI-compatible endpoint; two registrations wire
the revision's two additional backbones:

  * ClaudeChat — Claude Sonnet 5 via Anthropic's OpenAI-compatible endpoint
    (ANTHROPIC_API_KEY; ANTHROPIC_BASE_URL to override).
  * QwenChat   — Qwen3-32B served locally with vLLM's OpenAI server
    (QWEN_BASE_URL, default http://127.0.0.1:8000/v1; no key needed).

Embeddings are not part of either endpoint, so both resolve the broker's embedding
model through the same selector the Azure backend uses — the routing layer is
therefore identical on every backbone, as the paper requires.
"""
import os
from typing import Dict, List, Optional, Union

from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from RAPS.llm.llm import LLM
from RAPS.llm.llm_registry import LLMRegistry
from RAPS.llm.azure_chat import embedding_client, price_override, DEFAULT_EMBED_MODEL
from RAPS.utils.globals import Cost, PromptTokens, CompletionTokens

# USD per 1K tokens; substring match against the lowered model name, 0.0 fallback.
# These are the rates of the cost ledger reported in Table S.8, namely the Anthropic list
# prices of 1 July 2026: Sonnet $3/$15, Haiku $1/$5, Opus $5/$25 per 1M tokens. Qwen3-32B
# is open weight and served locally, so it carries no bill. Set RAPS_PRICE_IN and
# RAPS_PRICE_OUT (USD per 1M tokens) to price a run at another date's rates.
_PRICE_PER_1K = {
    "sonnet": {"input": 0.003, "output": 0.015},
    "haiku": {"input": 0.001, "output": 0.005},
    "opus": {"input": 0.005, "output": 0.025},
    "qwen": {"input": 0.0, "output": 0.0},
}


def _price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    override = price_override()
    if override:
        return prompt_tokens * override[0] / 1000 + completion_tokens * override[1] / 1000
    for name, rate in _PRICE_PER_1K.items():
        if name in model.lower():
            return prompt_tokens * rate["input"] / 1000 + completion_tokens * rate["output"] / 1000
    return 0.0


class OpenAICompatChat(LLM):
    """Chat backend for any OpenAI-compatible endpoint."""

    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.embed_model = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
        _temp = os.getenv("RAPS_TEMPERATURE")   # deterministic override, as in AzureGPTChat
        if _temp is not None:
            self.DEFAULT_TEMPERATURE = float(_temp)
        self._client = OpenAI(base_url=base_url, api_key=api_key)

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

    async def agen(self, messages, max_tokens=None, temperature=None, num_comps=None):
        # The comparison runs are synchronous; keep one code path.
        return self.gen(messages, max_tokens=max_tokens, temperature=temperature, num_comps=num_comps)

    def _account(self, resp) -> None:
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
        """Charge the broker's embedding call at the rate of the backbone, matching the
        chat ledger so the per-instance budget covers every coordination call."""
        try:
            prompt_tokens = int(resp.usage.prompt_tokens)
        except Exception:
            return
        Cost.instance().value += _price(self.model_name, prompt_tokens, 0)
        PromptTokens.instance().value += prompt_tokens

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        return self.get_embeddings([text], model=model)[0]

    @retry(wait=wait_random_exponential(max=20), stop=stop_after_attempt(3))
    def get_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        cleaned = [(t if isinstance(t, str) else str(t)).replace("\n", " ") for t in texts]
        embed_model = model or self.embed_model
        resp = embedding_client(embed_model).embeddings.create(model=embed_model, input=cleaned)
        self._account_embedding(resp)
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in ordered]


@LLMRegistry.register("ClaudeChat")
class ClaudeChat(OpenAICompatChat):
    def __init__(self, model_name: str = "claude-sonnet-5"):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set; it is required to run the "
                             "comparison on the Claude backbone.")
        super().__init__(
            model_name=model_name,
            base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/"),
            api_key=key,
        )


@LLMRegistry.register("QwenChat")
class QwenChat(OpenAICompatChat):
    def __init__(self, model_name: str = "Qwen/Qwen3-32B"):
        super().__init__(
            model_name=model_name,
            base_url=os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key="local",
        )
