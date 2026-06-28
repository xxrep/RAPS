import os
from typing import Optional
from class_registry import ClassRegistry

from RAPS.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name == "":
            model_name = "gpt-4o-mini-2024-07-18"

        if model_name == 'mock':
            return cls.registry.get(model_name)

        # Default chat backend is the Azure/modelhub gateway; override with
        # RAPS_LLM_BACKEND=GPTChat to use the public OpenAI endpoint instead.
        backend = os.getenv("RAPS_LLM_BACKEND", "AzureGPTChat")
        return cls.registry.get(backend, model_name)
