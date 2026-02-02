from typing import List,Any,Dict

from RAPS.graph.node import Node
from RAPS.agents.agent_registry import AgentRegistry
from RAPS.llm.llm_registry import LLMRegistry
from RAPS.prompt.prompt_set_registry import PromptSetRegistry
from RAPS.tools.coding.python_executor import PyExecutor

@AgentRegistry.register('FinalWriteCode')
class FinalWriteCode(Node):
    def __init__(self, 
                 id=None, 
                 llm_name="", 
                 domain="",
                 role="",
                 capabilities="",
                 interests:str = "", 
                 additional_instructions:str = "",
                 **kwargs):
        super().__init__(id, "FinalWriteCode" , domain, llm_name, role, capabilities, interests, additional_instructions)
        self.system_prompt = self.subscription_prompt.to_prompt()

    def extract_example(self, prompt: str) -> list:
        prompt = prompt['task']
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results
    
    def _process_inputs(self, task_context: Dict[str, Any], **kwargs)->List[Any]:
        system_prompt = self.system_prompt
        raw_inputs = task_context
        spatial_info = task_context.get("spatial_info", {})
        spatial_str = ""
        for id, info in spatial_info.items():
            if info['output'].startswith("```python") and info['output'].endswith("```"):  # is python code
                self.internal_tests = self.extract_example(raw_inputs)
                output = info['output'].lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                spatial_str += f"Agent {id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{info['output']}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                spatial_str += f"Agent {id} as a {info['role']} provides the following info: {info['output']}\n\n"
        user_prompt = f"The task is:\n\n{raw_inputs['task']}.\n At the same time, the outputs and feedbacks of other agents are as follows:\n\n{spatial_str}\n\n"
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


@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    def __init__(self, 
                 id=None, 
                 llm_name="", 
                 domain="",
                 role="",
                 capabilities="",
                 interests:str = "", 
                 additional_instructions:str = "",
                 decision_few_shot: str = "",
                 **kwargs):
        super().__init__(id, "FinalRefer" , domain, llm_name, role, capabilities, interests, additional_instructions, decision_few_shot)
        self.system_prompt = self.subscription_prompt.to_prompt()
        self.decision_few_shot = decision_few_shot

    def _process_inputs(self, task_context: Dict[str, Any], **kwargs)->List[Any]:
        system_prompt = self.system_prompt
        raw_inputs = task_context
        spatial_info = task_context.get("spatial_info", {})
        spatial_str = ""
        for id, info in spatial_info.items():
            spatial_str += id + ": " + info['output'] + "\n\n"
        # decision_few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = f"{self.subscription_prompt.few_shot}\n\nThe task is: {raw_inputs['task']}.\n At the same time, the output of other agents is as follows:\n\n{spatial_str}"
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
        print(f"################system prompt:{system_prompt}")
        print(f"################user prompt:{user_prompt}")
        print(f"################response:{response}")
        return response

@AgentRegistry.register('FinalDirect')
class FinalDirect(Node):
    def __init__(self, 
                 id=None, 
                 llm_name="", 
                 domain="",
                 role="",
                 capabilities="",
                 interests:str = "", 
                 additional_instructions:str = "",
                 **kwargs):
        """ Used for Directed IO """
        super().__init__(id, "FinalDirect", domain, llm_name, role, capabilities, interests, additional_instructions)
        self.system_prompt = self.subscription_prompt.to_prompt()
        
    def _process_inputs(self, task_context: Dict[str, Any], **kwargs)->List[Any]:
        return None
                
    def _execute(self, task_context: Dict[str, Any], **kwargs):
        output = ""
        info_list = []
        spatial_info = task_context.get("spatial_info", {})
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        return output
    
    async def _async_execute(self, task_context: Dict[str, Any], **kwargs):
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        return output


@AgentRegistry.register('FinalMajorVote')
class FinalMajorVote(Node):
    def __init__(self, 
                 id=None, 
                 llm_name="", 
                 domain="",
                 role="",
                 capabilities="",
                 interests:str = "", 
                 additional_instructions:str = "",
                 **kwargs):
        """ Used for Directed IO """
        super().__init__(id, "FinalMajorVote", domain, llm_name, role, capabilities, interests, additional_instructions)
        self.system_prompt = self.subscription_prompt.to_prompt()
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None
    
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = info['output']
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = info['output']
            print(processed_output)
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output
