from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict

# from RAPS.llm.format import Message


class LLM(ABC):
    DEFAULT_MAX_TOKENS = 2000
    DEFAULT_TEMPERATURE = 0.5
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