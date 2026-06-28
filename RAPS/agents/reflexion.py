from RAPS.graph import Node
from RAPS.agents.agent_registry import AgentRegistry


@AgentRegistry.register("ReflexionAgent")
class ReflexionAgent(Node):
    """
    An agent that reflects on previous failed attempts and improves its solution.
    Implements the self-reflection paradigm from Reflexion.
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
        super().__init__(id, "ReflexionAgent", domain, llm_name, role, capabilities, interests, additional_instructions)
        self.system_prompt = self.subscription_prompt.to_prompt()
        self.previous_attempt: str = ""
        self.feedback: str = ""

    def generate_reflection(self, problem: str, instruction: str):
        """
        Uses the LLM to reflect on previous failure and generate improved solution.
        """

        # Main instruction
        user_prompt = (
            f"The task is:\n{problem}\n\n"
            f"Instruction: {instruction}\n\n"
        )

        # Append self-reflection context if available
        if self.previous_attempt or self.feedback:
            user_prompt += (
                f"Your previous attempt was:\n{self.previous_attempt}\n\n"
                f"The feedback on your previous attempt is:\n{self.feedback}\n\n"
                "Reflect on the failure and improve your approach this time.\n"
            )
        else:
            user_prompt += "Think carefully and solve the problem thoroughly.\n"

        # Final messages
        message = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm.gen(message)

        # Save for future reflection
        self.previous_attempt = response
        self.feedback = self.mock_feedback(response)

        return {"response": response}

    def mock_feedback(self, response: str) -> str:
        """
        Placeholder feedback generator.
        In real use, this could come from an evaluator/judge agent or test case checker.
        """
        if "error" in response.lower() or "fail" in response.lower():
            return "Your previous solution contained logical errors or was incomplete."
        elif len(response) < 30:
            return "Your response was too short and lacked sufficient reasoning."
        else:
            return "Although your answer is on the right track, further justification is needed."

    def _process_inputs(self, task_context, **kwargs):
        problem = task_context.get("problem", "")
        instruction = task_context.get("instruction", "")
        return problem, instruction

    def _execute(self, task_context, **kwargs):
        problem, instruction = self._process_inputs(task_context)
        result = self.generate_reflection(problem, instruction)
        return result["response"]
