from typing import List,Any,Dict
import re

from RAPS.graph.node import Node
from RAPS.agents.agent_registry import AgentRegistry
from RAPS.llm.llm_registry import LLMRegistry
from RAPS.prompt.prompt_set_registry import PromptSetRegistry
from RAPS.tools.search.wiki import search_wiki_main


@AgentRegistry.register('AdverarialAgent')
class AdverarialAgent(Node):
    def __init__(self, 
                 id=None, 
                 llm_name="", 
                 domain="",
                 role="",
                 capabilities="",
                 interests:str = "", 
                 additional_instructions:str = "",
                 **kwargs):
        super().__init__(id, "AdverarialAgent" , domain, llm_name, role, capabilities, interests, additional_instructions)
        self.system_prompt = self.subscription_prompt.to_prompt()
        
    def _process_inputs(self, task_context: Dict[str, Any], **kwargs)->List[Any]:
        system_prompt = self.system_prompt
        raw_inputs = task_context
        spatial_info = task_context.get("spatial_info", {})
        temporal_info = task_context.get("temporal_info", {})
        user_prompt = f"The task is: {raw_inputs['task']}\n"
        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            spatial_str += f"Agent {id}, output is:\n\n {info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id}, output is:\n\n {info['output']}\n\n"
        user_prompt += f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
        user_prompt += f"In the last round of dialogue, the outputs of other agents were: \n\n{temporal_str}" if len(temporal_str) else ""
        return system_prompt, user_prompt
                
    def _execute(self, task_context: Dict[str, Any], **kwargs):
        system_prompt, user_prompt = self._process_inputs(task_context)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, task_context: Dict[str, Any], **kwargs):
        system_prompt, user_prompt = self._process_inputs(task_context)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        return response
