from typing import List, Dict


class SeedAgentPool:
    """
    For each dataset/task name, define a list of seed agent configurations:
    Each config is a dict: {
        "name": agent_class_name_in_registry,
        "id": optional_agent_id,
        "llm_name": LLM name,
        ...other init kwargs...
    }
    """

    _pool: Dict[str, List[Dict]] = {
        "GSM8K": [
            {"name": "GenerateCoT",
             "id": "cot1", 
             "llm_name": "gpt-4o",
             "role": "Chain-of-Thought Reasoner",
             "capabilities": "step-by-step reasoning",
             "interests": "math word problems, equations, multi-step reasoning",
             "additional_instructions": "Explain reasoning before final answer."},

            {"name": "VerifierAgent", 
             "id": "ver1", 
             "llm_name": 
             "gpt-3.5",
             "role": "Solution Verifier",
             "capabilities": "logical verification, consistency checking",
             "interests": "arithmetic verification, proof validation",
             "additional_instructions": "Always cross-check with the original question."},
        ],

        "HotpotQA": [
            {"name": "GenerateCoT",
             "id": "cot1", 
             "llm_name": "gpt-4o",
             "role": "Chain-of-Thought Reasoner",
             "capabilities": "step-by-step reasoning",
             "interests": "math word problems, equations, multi-step reasoning",
             "additional_instructions": "Explain reasoning before final answer."},

            {"name": "VerifierAgent", 
             "id": "ver1", 
             "llm_name": 
             "gpt-3.5",
             "role": "Solution Verifier",
             "capabilities": "logical verification, consistency checking",
             "interests": "arithmetic verification, proof validation",
             "additional_instructions": "Always cross-check with the original question."},
        ],

        "MATH": [
            {"name": "GenerateCoT",
             "id": "cot1", 
             "llm_name": "gpt-4o",
             "role": "Chain-of-Thought Reasoner",
             "capabilities": "step-by-step reasoning",
             "interests": "math word problems, equations, multi-step reasoning",
             "additional_instructions": "Explain reasoning before final answer."},

            {"name": "VerifierAgent", 
             "id": "ver1", 
             "llm_name": 
             "gpt-3.5",
             "role": "Solution Verifier",
             "capabilities": "logical verification, consistency checking",
             "interests": "arithmetic verification, proof validation",
             "additional_instructions": "Always cross-check with the original question."},
        ]
    }

    @classmethod
    def get_seed_configs(cls, dataset: str) -> List[Dict]:
        return cls._pool.get(dataset, [])

    @classmethod
    def add_seed_config(cls, dataset: str, agent_config: Dict):
        if dataset not in cls._pool:
            cls._pool[dataset] = []
        cls._pool[dataset].append(agent_config)