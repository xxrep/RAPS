import shortuuid
from typing import List, Any, Optional,Dict
from abc import ABC, abstractmethod
import asyncio

from RAPS.llm.llm_registry import LLMRegistry
from RAPS.llm.profile_embedding import cosine_similarity
from RAPS.graph.reputation import ReputationManager

class SimpleProfile:
    def __init__(self, role: str, capabilities: str, interests: str, additional_instructions: str, few_shot: str):
        self.role = role
        self.capabilities = capabilities
        self.interests = interests
        self.additional_instructions = additional_instructions
        self.few_shot = few_shot
        self.prompt = self.to_prompt()

    def to_prompt(self) -> str:
        prompt = f"Your Role: {self.role.strip()}."
        if self.capabilities:
            prompt += f"\nYour Capabilities: {self.capabilities.strip()}."
        if self.interests:
            prompt += f"\nYour Interests: {self.interests.strip()}."
        if self.additional_instructions:
            prompt += f"\n{self.additional_instructions.strip()}"
        return prompt.strip()

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
        # self.subscription_prompt = SimpleProfile(
        #     role=role,
        #     capabilities=capabilities,
        #     interests=interests,
        #     additional_instructions=additional_instructions,
        #     few_shot=few_shot
        # )
        self.system_prompt = capabilities
        self.refined_prompt = self.system_prompt
        self.llm_trace = []

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

        if self.role == "Final Answerer":
            # self.refined_prompt += "You only need to output the final numerical answer. Do not provide any textual descriptions."
            self.refined_prompt += self.additional_instructions

        self.llm_trace[-1]["output"] = self.refined_prompt
        return self.refined_prompt


    def watchdog_evaluate(self, message_content: Any, task_domain: str = "gsm8k", question: str = "", existing_info: str = "") -> bool:
        if not message_content or len(str(message_content).strip()) == 0:
            return False
        domain = str(task_domain or "").lower()
        question_text = str(question).strip() if question is not None else ""
        existing_text = str(existing_info).strip() if existing_info is not None else "None"
        if domain == "mmlu":
            prompt = (
                "You are a watchdog for an intelligent agent system.\n"
                "Please evaluate the following new message from another agent.\n"
                "Decide whether it is logically relevant and valuable for answering the multiple-choice MMLU question:\n"
                f"{question_text}\n\n"
                "Your known information:\n"
                f"{existing_text}\n\n"
                "The new message from another agent:\n"
                f"{message_content}\n\n"
                "A message is valuable if it is coherent, grounded in the question, adds supporting evidence, provides a correction, or clarifies reasoning.\n"
                "Disagreement with others is acceptable when it is well-justified.\n"
                "Respond with 'YES' if it is good, 'NO' if it is bad or a hallucination.\n"
                "Do not provide explanations, just YES or NO."
            )
        else:
            prompt = (
                "You are a watchdog for an intelligent agent system.\n"
                "Please evaluate the following new message from another agent.\n"
                "Decide whether it is logically relevant and valuable for solving the math problem:\n"
                f"{question_text}\n\n"
                "Your known information:\n"
                f"{existing_text}\n\n"
                "The new message from another agent:\n"
                f"{message_content}\n\n"
                "A message is valuable if it is coherent, grounded in the problem, adds supporting evidence, provides a correction, or clarifies reasoning.\n"
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

    def check_trust(self, sender_id: str, threshold: float = 0.3) -> bool:
        rep = self.rep_manager.REP.get(sender_id, {"a": 1.0, "b": 1.0})
        trust_score = rep["a"] / (rep["a"] + rep["b"])
        return trust_score >= threshold
    
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
        # task_vec = get_sentence_embedding(task)
        task_vec = self.llm.get_embedding(task)
        results = []
        for agent_id, sub in subs.items():
            # sub_vec = get_sentence_embedding(sub)
            sub_vec = self.llm.get_embedding(sub)
            score = cosine_similarity(task_vec, sub_vec)
            if score >= threshold:
                results.append((agent_id, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        top = [agent_id for agent_id, _ in results[:top_k]]
        print(f"[BROKER-EMBEDDING] Top-{top_k} (Threshold {threshold}): {top}")
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
