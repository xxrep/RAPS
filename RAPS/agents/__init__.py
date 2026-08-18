"""The agent classes a benchmark pool draws from, plus the adversarial hosts of the
threat model. Each registers itself in AgentRegistry under the name a runner asks for.
"""

__all__ = ["AnalyzeAgent",      # MMLU, with the ReAct retrieval loop
           "MathAgent",         # GSM8K / SVAMP / AQuA
           "CodeWriting",       # HumanEval
           "Adversary",         # the five adversary types of the threat model
           "SeedAgentPool",     # the specialists recruitment can add
           "AgentRegistry",
           ]
