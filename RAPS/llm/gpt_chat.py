import os
from typing import List, Union, Optional, Dict
from openai import OpenAI
# from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

# from RAPS.llm.format import Message
from RAPS.llm.price import cost_count
from RAPS.llm.llm import LLM
from RAPS.llm.llm_registry import LLMRegistry
from RAPS.utils.const import RAPS_ROOT


@LLMRegistry.register('GPTChat')
class GPTChat(LLM):
    def __init__(self, model_name: str = 'gpt-4o-mini'):
        self.model_name = model_name
        
        # Try to load key from environment variable first
        self.key = os.getenv("OPENAI_API_KEY")
        
        # If not in env, try loading from file
        if not self.key:
            key_path = RAPS_ROOT / "RAPS/llm/openai_key.txt"
            if key_path.exists():
                with open(key_path, "r") as f:
                    self.key = f.read().strip()
        
        if not self.key:
            raise ValueError("OpenAI API Key not found. Please set OPENAI_API_KEY env variable or add it to RAPS/llm/openai_key.txt")
            
        self.client = OpenAI(api_key=self.key)

    @retry(
        wait=wait_random_exponential(max=50),
        stop=stop_after_attempt(3)
    )
    def gen(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[str, List[str]]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFAULT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # OpenAI API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        output = response.choices[0].message.content
        prompt = "".join([m['content'] for m in messages])
        cost_count(prompt, output, self.model_name)

        return output

    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        text = text.replace("\n", " ")
        # Warning: Embedding APIs might also incur costs
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding
