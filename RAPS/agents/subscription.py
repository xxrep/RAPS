from typing import List, Optional
import time

class SubscriptionTemplate:
    def __init__(
        self,
        role: str,
        capabilities: str,
        interests: str,
        additional_instructions: Optional[str] = "",
        few_shot = "",
    ):
        self.role = role
        self.capabilities = capabilities
        self.interests = interests
        self.additional_instructions = additional_instructions
        self.prompt = self.to_prompt()
        self.few_shot = few_shot

    def to_prompt(self) -> str:
        """Generate a basic system instruction string from current fields."""
        instr = f"Your Role: {self.role.strip()}.\n"
        if self.capabilities:
            instr += "Your Capabilities: " + self.capabilities.strip() + ".\n"
        if self.interests:
            instr += "You Interests: " + self.interests.strip() + ".\n"
        if self.additional_instructions:
            instr += f"{self.additional_instructions.strip()}\n"
        return instr.strip()

    def refine(self, context: str, llm) -> str:
        """
        Use LLM to refine the instruction based on current fields + task context.
        Updates internal instruction and timestamp.
        """
        prompt = (
            "You are tasked with improving the subscription prompt for an LLM agent.\n"
            f"Current agent profile:\n{self.prompt}\n\n"
            f"Given the task context:\n{context}\n\n"
            "Please rewrite the agent's system instruction to be more specific, "
            "actionable, and task-aware. Return only the final instruction."
        )
        msg = [{"role": "system", "content": "You are a subscription prompt refiner."},
               {"role": "user", "content": prompt}]
        refined = llm.gen(msg)
        self.prompt = str(refined).strip()
        return self.prompt