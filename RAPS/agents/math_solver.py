from typing import List,Any,Dict,Union

from RAPS.graph.node import Node
from RAPS.agents.agent_registry import AgentRegistry
from RAPS.llm.llm_registry import LLMRegistry
from RAPS.prompt.prompt_set_registry import PromptSetRegistry
from RAPS.tools.coding.python_executor import execute_code_get_return
from raps_data.gsm8k_dataset import gsm_get_predict

@AgentRegistry.register('MathAgent')
class MathAgent(Node):
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
        super().__init__(id, "MathAgent" , domain, llm_name, role, capabilities, interests, additional_instructions, few_shot)
        
        self.refined_prompt = self.system_prompt
        
    def _process_inputs(self, task_context: Dict[str, str], **kwargs) -> List[Any]:
        """ Process the task_context (Dict) """
        
        task = task_context.get('task', '')
        history = task_context.get('history', '')
        
        user_prompt = f"{self.few_shot}\n\nThe question is: {task}\n"
        
        if history:
            user_prompt += f"\nPrevious Discussions and Plans:\n{history}\n"

        return user_prompt
    
    def _execute(self, task_context: Dict[str, str], **kwargs):
        """ Use the processed input to get the result """
        user_prompt = self._process_inputs(task_context)
        message = [{'role':'system','content':self.refined_prompt},{'role':'user','content':user_prompt}]
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
        self.llm_trace.append({
            "type": "publish",
            "messages": message
        })
        response = await self.llm.agen(message)
        self.llm_trace[-1]["output"] = response
        return response
