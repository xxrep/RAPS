import re
from typing import List, Any, Dict

from RAPS.graph.node import Node
from RAPS.agents.agent_registry import AgentRegistry

# Local Wikipedia BM25 tool (offline, no proxy). If Elasticsearch is down the ReAct
# loop degrades gracefully to pure reasoning.
try:
    from RAPS.tools.search.bm25_retriever import bm25_retrieve
except Exception:  # pragma: no cover
    bm25_retrieve = None

_SEARCH_RE = re.compile(r"SEARCH:\s*(.+)", re.IGNORECASE)


@AgentRegistry.register('AnalyzeAgent')
class AnalyzeAgent(Node):
    def __init__(self,
                 id=None,
                 llm_name="",
                 domain="",
                 role="",
                 capabilities="",
                 interests:str = "",
                 additional_instructions:str = "",
                 few_shot:str = "",
                 **kwargs):
        super().__init__(id, "AnalyzeAgent", domain, llm_name, role, capabilities, interests, additional_instructions, few_shot)
        self.refined_prompt = self.system_prompt
        # ReAct-style autonomous retrieval. Off by default so non-knowledge roles
        # (and the gsm8k/humaneval domains) are unaffected; enabled per-agent for
        # the knowledge experts in run_mmlu.initialize_agents_from_set.
        self.react_retrieve = False
        self.react_max_search = 2

    def _process_inputs(self, task_context: Dict[str, str], **kwargs) -> List[Any]:
        task = task_context.get('task', '')
        history = task_context.get('history', '')
        user_prompt = f"{self.few_shot}\n\nThe task is: {task}\n"
        if history:
            user_prompt += f"\nPrevious Discussions and Plans:\n{history}\n"
        return user_prompt

    def _run_search(self, query: str) -> str:
        """Execute one local BM25 retrieval; return an Observation string."""
        if bm25_retrieve is None:
            return "Retrieval tool unavailable."
        try:
            # each hit is a small ~128-word passage, so top-5 gives broader coverage
            obs = bm25_retrieve([query], k=5, chars=500)
        except Exception:
            obs = ""
        return obs or "No relevant reference was found for this query."

    def _react_gen(self, messages: List[Dict[str, str]]) -> str:
        """ReAct loop: the agent itself decides when and what to retrieve.

        The agent may emit a line 'SEARCH: <query>' to fetch real facts; the retrieved
        text is fed back as an 'Observation' and the agent continues, up to
        react_max_search retrievals, then must answer.
        """
        messages = list(messages)
        response = ""
        for it in range(self.react_max_search + 1):
            self.llm_trace.append({"type": "publish", "messages": list(messages)})
            response = str(self.llm.gen(messages))
            self.llm_trace[-1]["output"] = response
            m = _SEARCH_RE.search(response)
            if not m or it == self.react_max_search:
                break
            query = m.group(1).strip().strip('"').strip().splitlines()[0][:200]
            obs = self._run_search(query)
            self.llm_trace.append({"type": "retrieve",
                                   "messages": [{"role": "user", "content": query}],
                                   "output": obs[:400]})
            messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user", "content":
                    (f"Observation — facts retrieved from Wikipedia for your query '{query}':\n{obs}\n\n"
                     "Ground your reasoning in these facts where relevant (ignore any that are off-topic). "
                     "Continue your analysis and give your final answer. Issue another 'SEARCH: <query>' "
                     "ONLY if a specific decisive fact is still missing.")},
            ]
        return response

    def _execute(self, task_context: Dict[str, str], **kwargs):
        user_prompt = self._process_inputs(task_context)
        message = [{'role':'system','content':self.refined_prompt},{'role':'user','content':user_prompt}]
        if getattr(self, "react_retrieve", False):
            return self._react_gen(message)
        self.llm_trace.append({
            "type": "publish",
            "messages": message
        })
        response = self.llm.gen(message)
        self.llm_trace[-1]["output"] = response
        return response

    async def _async_execute(self, task_context: Dict[str, str], **kwargs):
        user_prompt = self._process_inputs(task_context)
        message = [{'role':'system','content':self.refined_prompt},{'role':'user','content':user_prompt}]
        if getattr(self, "react_retrieve", False):
            return self._react_gen(message)
        self.llm_trace.append({
            "type": "publish",
            "messages": message
        })
        response = await self.llm.agen(message)
        self.llm_trace[-1]["output"] = response
        return response
