import shortuuid
from typing import List, Any, Optional,Dict
from abc import ABC, abstractmethod
import asyncio
import numpy as np

from RAPS.llm.llm_registry import LLMRegistry
from RAPS.graph.reputation import ReputationManager


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


class Node(ABC):

    def __init__(self, 
                 id: Optional[str],
                 agent_name:str="",
                 domain:str="", 
                 llm_name:str = "",
                 role:str = "", 
                 capabilities:str = "", 
                 interests:str = "", 
                 additional_instructions:str = "",
                 few_shot: str = "",
                 ):
        """
        Initializes a new Node instance.
        """
        self.id:str = id if id is not None else role + '_' + shortuuid.ShortUUID().random(length=4)
        self.agent_name:str = agent_name
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.role = role
        self.capabilities = capabilities
        self.interests = interests
        self.additional_instructions = additional_instructions
        self.few_shot = few_shot
        self.last_memory: Dict[str,List[Any]] = {'inputs':[],'outputs':[],'raw_inputs':[]}
        self.llm = LLMRegistry.get(llm_name)
        # The standing subscription an agent declares, which the broker matches against and
        # which reactive subscription rewrites into `refined_prompt`.
        self.system_prompt = capabilities
        self.refined_prompt = self.system_prompt
        self.llm_trace = []
        # prompt strategy: "solve" (original) refines persona AND can bake in a
        # solution path; "persona" refines only the agent's expertise/lens.
        self.refine_mode = "solve"
        # broker strategy: "default" predicts a next role; "gap" asks what
        # capability is still MISSING (engages under-used specialists/tools).
        self.broker_mode = "default"

        self.rep_manager = ReputationManager(
            discount_alpha=0.999,
            discount_beta=0.999,
            merge_weight=0.05,
            deviation_threshold=0.2,
            trust_threshold=0.5
        )

    @property
    def node_name(self):
        return self.__class__.__name__
    
    def add_predecessor(self, operation: 'Node', st='spatial'):
        if st == 'spatial' and operation not in self.spatial_predecessors:
            self.spatial_predecessors.append(operation)
            operation.spatial_successors.append(self)
        elif st == 'temporal' and operation not in self.temporal_predecessors:
            self.temporal_predecessors.append(operation)
            operation.temporal_successors.append(self)

    def add_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation not in self.spatial_successors:
            self.spatial_successors.append(operation)
            operation.spatial_predecessors.append(self)
        elif st == 'temporal' and operation not in self.temporal_successors:
            self.temporal_successors.append(operation)
            operation.temporal_predecessors.append(self)

    def remove_predecessor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_predecessors:
            self.spatial_predecessors.remove(operation)
            operation.spatial_successors.remove(self)
        elif st =='temporal' and operation in self.temporal_predecessors:
            self.temporal_predecessors.remove(operation)
            operation.temporal_successors.remove(self)

    def remove_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_successors:
            self.spatial_successors.remove(operation)
            operation.spatial_predecessors.remove(self)
        elif st =='temporal' and operation in self.temporal_successors:
            self.temporal_successors.remove(operation)
            operation.temporal_predecessors.remove(self)

    def clear_connections(self):
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []        
    
    def update_memory(self):
        self.last_memory['inputs'] = self.inputs
        self.last_memory['outputs'] = self.outputs
        self.last_memory['raw_inputs'] = self.raw_inputs

    def refine_system_prompt(self, context: str, question: str) -> str:
        if getattr(self, "refine_mode", "solve") == "persona":
            # Persona-only refinement: shape the agent's expertise/lens, never the solution.
            prompt = (
                "### GOAL\n"
                "Specialize this agent's EXPERTISE and LENS for the current problem — NOT its solution.\n"
                "Describe what perspective, domain knowledge, and verification checks the agent should bring.\n\n"

                "### HARD CONSTRAINTS\n"
                "- Do NOT solve the problem, do NOT perform any calculation, do NOT state or imply a numeric/final answer.\n"
                "- Do NOT restate the problem's specific quantities or a step-by-step solution path.\n"
                "- Keep the agent's original output-format requirement intact.\n"
                "- Output 2-3 sentences describing the agent's specialized role for THIS problem.\n\n"

                "### INPUT DATA\n"
                f"1. **Base Profile:**\n{self.system_prompt}\n\n"
                f"2. **Current Question:**\n{question}\n\n"
                f"3. **Interaction Context:**\n{context}\n\n"

                "### OUTPUT FORMAT\n"
                "Return ONLY the refined system instruction string. No explanations, no solution."
            )
        else:
            prompt = (
                "### GOAL\n"
                "Your task is to dynamically specialize an LLM agent's persona to perfectly align with the specific needs of the current problem state.\n"
                "Do NOT simply summarize the old profile. Instead, evolve the agent's intent to be highly specific to the immediate context.\n\n"

                "### INPUT DATA\n"
                f"1. **Base Profile (Starting Point):**\n{self.system_prompt}\n\n"
                f"2. **Current Task/Question:**\n{question}\n\n"
                f"3. **Interaction Context (Message Flow):**\n{context}\n\n"

                "### INSTRUCTIONS\n"
                "- Analyze the `Current Task` and `Interaction Context` to identify what specific expertise, constraints, or output format is missing or needed right now.\n"
                "- Ignore generic traits in the `Base Profile` if they are not relevant to the current step.\n"
                "- Rewrite the System Instruction to explicitly guide the agent on *how* to process the specific input in the context.\n"
                "- The new instruction should act like a focused 'Mission Briefing' for this specific step of the chain.\n\n"

                "### OUTPUT FORMAT\n"
                "Return ONLY the refined system instruction string. No explanations."
            )
        msg = [{"role": "system", "content": "You are an expert prompt refiner who tailors agent instructions to the current question and context based on the evolving workflow."},
               {"role": "user", "content": prompt}]
        self.llm_trace.append({
            "type": "refine_system_prompt",
            "messages": msg
        })
        refined = self.llm.gen(msg)

        self.refined_prompt = str(refined).strip()

        # The deciding node keeps its output-format constraint through the
        # rewrite: runners mark it via role "Final Answerer" or the
        # is_final_answerer flag (the revision names it after a pool role,
        # e.g. "Summarizer" — Table S.2).
        if self.role == "Final Answerer" or getattr(self, "is_final_answerer", False):
            self.refined_prompt += self.additional_instructions

        self.llm_trace[-1]["output"] = self.refined_prompt
        return self.refined_prompt


    #: What the watchdog is asked to judge a publication against, per benchmark domain.
    _WATCHDOG_SUBJECT = {
        "mmlu": ("answering the multiple-choice question", "question"),
        "aqua": ("solving the multiple-choice math problem", "problem"),
        "gsm8k": ("solving the math problem", "problem"),
        "svamp": ("solving the math problem", "problem"),
        "math": ("solving the math problem", "problem"),
        "humaneval": ("implementing a correct program for the specification", "specification"),
    }

    def watchdog_evaluate(self, message_content: Any, task_domain: str = "gsm8k", question: str = "", existing_info: str = "") -> bool:
        if not message_content or len(str(message_content).strip()) == 0:
            return False
        domain = str(task_domain or "").lower()
        question_text = str(question).strip() if question is not None else ""
        existing_text = str(existing_info).strip() if existing_info is not None else "None"
        goal, subject = self._WATCHDOG_SUBJECT.get(domain, ("solving the task", "task"))
        prompt = (
            "You are a watchdog for an intelligent agent system.\n"
            "Please evaluate the following new message from another agent.\n"
            f"Decide whether it is logically relevant and valuable for {goal}:\n"
            f"{question_text}\n\n"
            "Your known information:\n"
            f"{existing_text}\n\n"
            "The new message from another agent:\n"
            f"{message_content}\n\n"
            f"A message is valuable if it is coherent, grounded in the {subject}, adds supporting evidence, provides a correction, or clarifies reasoning.\n"
            "Disagreement with others is acceptable when it is well-justified.\n"
            "Respond with 'YES' if it is good, 'NO' if it is bad or a hallucination.\n"
            "Do not provide explanations, just YES or NO."
        )
        try:
            msg = [{"role": "system", "content": "You are a watchdog evaluator."},
                   {"role": "user", "content": prompt}]
            self.llm_trace.append({
                "type": "watchdog_evaluate",
                "messages": msg
            })
            response = self.llm.gen(msg)
            self.llm_trace[-1]["output"] = response
            if isinstance(response, list):
                response = response[0]
            return "YES" in str(response).upper()
        except Exception:
            return False

    def check_trust(self, sender_id: str, threshold: float = 0.7,
                    min_observations: float = 1.0) -> bool:
        """Whether this agent's reliability-aware gate admits `sender_id` as a routing
        candidate. min_observations=0 removes the exemption for unobserved peers, which
        places the decision on the prior expectation (kept as an ablation)."""
        return self.rep_manager.admits(sender_id, threshold, min_observations)
    
    def export_report(self) -> Dict[str, tuple]:
        """The second-hand testimony this agent broadcasts during gossip.

        Defaults to its honest first-hand table; adversarial agents override it
        to forge reports (false praise / bad-mouthing, Supp. F). The coordinator
        always calls this method rather than the reputation manager directly.
        """
        return self.rep_manager.export_first_hand()

    def publish(self, task_context: Any, **kwargs):
        output = self._execute(task_context, **kwargs)
        self.outputs = output
        return output
    
    def broker_route(self, 
                    publications: List[str], 
                    all_subscriptions: Dict[str, str], 
                    method: str = "embedding", 
                    top_k: int = 2,
                    sim_threshold: float = 0.4) -> List[str]:

        # Step 1: summarize or predict next step
        pub_context = "\n\n".join(publications)
        candidate_subs = {aid: sub for aid, sub in all_subscriptions.items() if aid != self.id}
        if not candidate_subs:
            return []
        options = "\n".join([f"{i+1}. [{aid}] {sub}" for i, (aid, sub) in enumerate(candidate_subs.items())])
        if getattr(self, "broker_mode", "default") == "gap":
            # Gap-driven routing: identify the MISSING capability rather than defaulting
            # to a generic verifier, so under-used specialists/tools get engaged.
            instruction = (
                "You are coordinating a multi-agent system.\n"
                "Identify the single most important capability that is STILL MISSING to finish the task,\n"
                "then describe the downstream role best suited to provide it.\n"
                "If the work so far is only natural-language reasoning with no independent/exact "
                "verification (e.g. no code execution), prefer a role that provides that verification.\n"
                f"Previous agents' outputs:\n{pub_context}\n\n"
                f"Available downstream roles (current agent excluded):\n{options}\n\n"
                "Write a concise description of the needed role, grounded in the options above. "
                "Do not write a task plan. Keep it under 100 words."
            )
        else:
            instruction = (
                "You are coordinating a multi-agent system.\n"
                "Choose the most suitable downstream role based on the available role options.\n"
                f"Previous agents' outputs:\n{pub_context}\n\n"
                f"Available downstream roles (current agent excluded):\n{options}\n\n"
                "Write a concise role description grounded in the options above. Do not write a task plan. Keep it under 100 words."
            )

        msg = [
            {"role": "system", "content": "You are a semantic router. You bridge the gap between current progress and required expertise."},
            {"role": "user", "content": instruction}
        ]
        self.llm_trace.append({
            "type": "broker_route",
            "messages": msg
        })
        predicted_task = self.llm.gen(msg).strip()
        self.llm_trace[-1]["output"] = predicted_task
        print(f"[BROKER] Predicted next task: {predicted_task}")

        # Step 2: match with subscriptions
        # if method == "embedding":
        #     return self._match_by_embedding(predicted_task, all_subscriptions, top_k=top_k, threshold=sim_threshold)
        # elif method == "llm":
        #     return self._match_by_llm(predicted_task, all_subscriptions, top_k=top_k)
        # else:
        #     raise ValueError("Unknown broker method")
        return self._match_by_embedding(predicted_task, candidate_subs, top_k=top_k, threshold=sim_threshold)

    def _match_by_embedding(self, task: str, subs: Dict[str, str], top_k=2, threshold=0.4) -> List[str]:
        """The forwarding query is matched against the candidate subscriptions by cosine
        similarity, and the `top_k` highest-ranked candidates that reach `threshold` are
        retained. A round in which no candidate reaches it forwards to nobody, which ends
        the episode at that point."""
        if not subs:
            return []
        agent_ids = list(subs.keys())
        # One batched call: [predicted_task, sub_1, ..., sub_n] -> embeddings. The payload
        # is traced so the cost decomposition can charge the subscription-embedding term.
        payload = [task] + [subs[aid] for aid in agent_ids]
        self.llm_trace.append({
            "type": "subscription_embedding",
            "messages": [{"role": "input", "content": text} for text in payload],
            "output": "",
        })
        vectors = self.llm.get_embeddings(payload)
        task_vec = np.asarray(vectors[0])
        scored = [(aid, cosine_similarity(task_vec, np.asarray(vec)))
                  for aid, vec in zip(agent_ids, vectors[1:])]
        scored.sort(key=lambda x: x[1], reverse=True)

        top = [aid for aid, score in scored[:top_k] if score >= threshold]
        print(f"[BROKER-EMBEDDING] Top-{top_k} (Threshold {threshold}): {top or 'none reached'} "
              f"| scores={[round(s, 3) for _, s in scored[:top_k]]}")
        return top

    def _match_by_llm(self, task: str, subs: Dict[str, str], top_k=1) -> List[str]:
        """
        Use LLM to select the most suitable agent(s) from all subscriptions
        for the predicted task. Returns a list of agent_ids.
        """
        # Format options as choices
        options = "\n".join([f"{i+1}. [{aid}] {sub}" for i, (aid, sub) in enumerate(subs.items())])
        id_list = list(subs.keys())

        prompt = (
            f"The current task is:\n\"{task}\"\n\n"
            f"Below are system instructions from different agents:\n\n{options}\n\n"
            f"Select the top {top_k} agents that are best suited to handle the task. "
            f"Return ONLY their corresponding numbers, separated by commas (e.g., 2,4,5)."
        )

        msgs = [
            {"role": "system", "content": "You are a planner selecting the best agents for a task."},
            {"role": "user", "content": prompt}
        ]

        raw = self.llm.gen(msgs).strip()

        # Parse numbers from response
        import re
        choices = re.findall(r"\d+", raw)
        indices = [int(c)-1 for c in choices if c.isdigit() and 0 < int(c) <= len(id_list)]
        selected = [id_list[i] for i in indices][:top_k]

        print(f"[BROKER-LLM] Selected agents: {selected}")
        return selected

    async def async_execute(self, task_context: Any, **kwargs):
        self.outputs = []
        tasks = [asyncio.create_task(self._async_execute(task_context, **kwargs))]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs
               
    def _execute(self, task_context: Any, **kwargs):
        """ To be overriden by the descendant class """
        return ""

    @abstractmethod
    async def _async_execute(self, task_context: Any, **kwargs):
        return ""

    @abstractmethod
    def _process_inputs(self, task_context: Any, **kwargs)->List[Any]:
        return []
