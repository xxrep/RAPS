"""
RetrieverAgent — a dedicated agentic host that grounds the team in real facts.

In RAPS's content-centric pub/sub paradigm this is just another host whose declared
intent is "I provide factual grounding via Wikipedia retrieval". When it publishes
(typically as the entry host), it turns the question into search queries, retrieves
real Wikipedia content, and publishes that as reference material for the downstream
experts to reason over.
"""
from typing import List, Any, Dict

from RAPS.graph.node import Node
from RAPS.agents.agent_registry import AgentRegistry
from RAPS.tools.search.wiki_retriever import retrieve


@AgentRegistry.register('RetrieverAgent')
class RetrieverAgent(Node):
    def __init__(self, id=None, llm_name="", domain="", role="Retriever",
                 capabilities="", interests="", additional_instructions="", few_shot="", **kwargs):
        super().__init__(id, "RetrieverAgent", domain, llm_name, role,
                         capabilities, interests, additional_instructions, few_shot)
        self.refined_prompt = self.system_prompt

    def _process_inputs(self, task_context: Dict[str, str], **kwargs) -> str:
        return task_context.get("task", "")

    def _gen_queries(self, task: str) -> List[str]:
        prompt = ("Give 1-3 concise Wikipedia search queries (key topics/entities) whose articles "
                  "would contain facts relevant to answering the question below. "
                  "Output ONLY the queries separated by ';'.\n\n" + task)
        raw = self.llm.gen([{"role": "system", "content": "You write precise search queries."},
                            {"role": "user", "content": prompt}])
        return [q.strip() for q in str(raw).replace("\n", ";").split(";") if q.strip()][:3]

    def _retrieve(self, task: str) -> str:
        self.llm_trace.append({"type": "retrieve", "messages": [{"role": "user", "content": task[:200]}]})
        try:
            queries = self._gen_queries(task)
            context = retrieve(queries, k_per_query=2, chars=500)
        except Exception as e:
            queries, context = [], f""
        self.llm_trace[-1]["output"] = f"queries={queries}; chars={len(context)}"
        if not context:
            return "No relevant Wikipedia reference was found for this question."
        return ("Reference material retrieved from Wikipedia (use these facts to ground your "
                f"reasoning; ignore any that are irrelevant):\n{context}")

    def _execute(self, task_context: Dict[str, str], **kwargs):
        return self._retrieve(task_context.get("task", ""))

    async def _async_execute(self, task_context: Dict[str, str], **kwargs):
        return self._retrieve(task_context.get("task", ""))
