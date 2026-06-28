from RAPS.graph.node import Node
from RAPS.agents.agent_registry import AgentRegistry


@AgentRegistry.register("GenerateCoT")
class GenerateCoTAgent(Node):
    """
    A chain-of-thought agent that produces step-by-step reasoning
    followed by the final solution.
    """

    def __init__(self, 
                 id=None, 
                 llm_name="", 
                 domain="",
                 role="",
                 capabilities="",
                 interests:str = "", 
                 additional_instructions:str = "",
                 **kwargs):
        super().__init__(id, "GenerateCoT", domain, llm_name, role, capabilities, interests, additional_instructions)
        self.system_prompt = self.subscription_prompt.to_prompt()

    def generate_cot(self, problem: str, entry_point: str, instruction: str):

        system_prompt = (
            self.system_prompt
            + "\nYou must show your reasoning process (chain of thought) before giving the final answer."
        )

        user_prompt = (
            f"The task is:\n{problem}\n"
            "Please think step by step, explain your reasoning clearly, "
            "and then give the final answer."
        )

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.llm.gen(msgs)
        return {"response": response}

    def _process_inputs(self, task_context, **kwargs):
        problem = task_context.get("problem", "")
        entry_point = task_context.get("entry_point", "")
        instruction = task_context.get("instruction", "")

        return problem, entry_point, instruction

    def _execute(self, task_context, **kwargs):
        problem, entry_point, instruction = self._process_inputs(task_context)
        result = self.generate_cot(problem, entry_point, instruction)
        return result["response"]