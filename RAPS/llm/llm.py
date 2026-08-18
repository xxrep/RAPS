from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict


class LLM(ABC):
    # The shared decoding controls of Table S.1, applied to every agent call of every
    # backend unless a call passes an explicit override. RAPS/config.py re-exports these
    # as the canonical values so the controlled factor is declared in one place.
    DEFAULT_MAX_TOKENS = 2000
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_NUM_COMPLETIONS = 1

    # @abstractmethod
    # async def agen(
    #     self,
    #     messages: List[Dict],
    #     max_tokens: Optional[int] = None,
    #     temperature: Optional[float] = None,
    #     num_comps: Optional[int] = None,
    #     ) -> Union[List[str], str]:

    #     pass

    @abstractmethod
    def gen(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        pass