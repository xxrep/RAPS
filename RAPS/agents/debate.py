from RAPS.graph import Node
from RAPS.agents.agent_registry import AgentRegistry


@AgentRegistry.register("DebateAgent")
class DebateAgent(Node):
    """
    An agent that simulates multiple LLMs debating different perspectives,
    and synthesizes their arguments into a final judgment.
    """

    def __init__(self,
                 id=None,
                 llm_name="",
                 domain="",
                 role="",
                 capabilities="",
                 interests:str = "", 
                 additional_instructions:str = "",
                 num_opponents: int = 3,
                 **kwargs):
        super().__init__(id, "DebateAgent", domain, llm_name, role, capabilities, interests, additional_instructions)

        self.system_prompt = self.subscription_prompt.to_prompt()
        self.num_opponents = num_opponents

        # Simulate different perspectives using same LLM
        self.debaters = [
            self.llm for _ in range(num_opponents)
        ]

    def generate_debate(self, problem: str, instruction: str):
        """
        Simulates multiple debaters (LLMs), each providing a different perspective.
        The final output summarizes the debate.
        """
        system_prompt = (
            self.system_prompt
            + "\nYou will simulate a multi-perspective debate before forming a final judgment."
        )

        debate_intro = (
            f"The task is:\n{problem}\n\n"
            f"Instruction: {instruction}\n"
            "Simulate a debate between multiple experts with different viewpoints. "
            "Each expert gives their opinion, followed by a summary decision."
        )

        # Create multiple debater turns
        debate_turns = []
        for i, llm in enumerate(self.debaters):
            persona = f"Expert {i+1}"
            role_prompt = (
                f"You are {persona}. Provide your independent analysis of the problem "
                "based on your own assumptions or viewpoint.\n"
            )
            message = [
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": debate_intro}
            ]
            response = llm.gen(message)
            debate_turns.append((persona, response))

        # Aggregate all arguments into a summary
        summary_input = "\n\n".join([f"{p}: {r}" for p, r in debate_turns])
        summary_prompt = (
            f"{system_prompt}\n"
            f"The following debate occurred:\n\n{summary_input}\n\n"
            "Please summarize the key points of each perspective and provide your final conclusion."
        )

        summary_message = [{"role": "system", "content": summary_prompt}]
        final_response = self.llm.gen(summary_message)

        return {
            "debate": debate_turns,
            "response": final_response
        }

    def _process_inputs(self, task_context, **kwargs):
        problem = task_context.get("problem", "")
        instruction = task_context.get("instruction", "")
        return problem, instruction

    def _execute(self, task_context, **kwargs):
        problem, instruction = self._process_inputs(task_context)
        result = self.generate_debate(problem, instruction)
        return result["response"]