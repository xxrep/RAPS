import sys
import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.utils.const import RAPS_ROOT
# from RAPS.graph.graph import Graph
# from RAPS.tools.reader.readers import JSONLReader
from RAPS.utils.globals import Time
from RAPS.agents.agent_registry import AgentRegistry
from RAPS.utils.globals import Cost, PromptTokens, CompletionTokens
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict
from RAPS.prompt.gsm8k_prompt_set import GSM8KPromptSet
# Explicitly import agent to ensure registration
from RAPS.agents.math_solver import MathAgent


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str):
    print(f"[{_ts()}] {message}")


def parse_args():
    parser = argparse.ArgumentParser(description="RAPS Experiments on gsm8k")
    parser.add_argument("--dataset_json", type=str, default="datasets/gsm8k/gsm8k.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument('--domain', type=str, default="gsm8k", help="Domain (the same as dataset name), default 'gsm8k'")
    parser.add_argument("--max_steps", type=int, default=3, help="Max communication step")
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the GDesigner')
    
    args = parser.parse_args()
    result_path = RAPS_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)

    return args

def initialize_agents_from_set(llm_name: str):
    """
    Initialize agents using GSM8KPromptSet.
    """
    agents = []
    
    # Iterate over roles defined in GSM8KPromptSet
    # ROLE_DESCRIPTION is a dict {role: description}
    # We will create one agent for each role.
    
    # Note: GSM8KPromptSet.ROLE_DESCRIPTION is a dict.
    # We can access it via GSM8KPromptSet.get_description(role) but we need the keys.
    # The file gsm8k_prompt_set.py defines ROLE_DESCRIPTION globally, but we should use the class if possible.
    # However, the class methods assume we know the role.
    # We can import ROLE_DESCRIPTION from the module if we want, or just hardcode the keys since they are standard.
    # Or better, let's inspect the module imports. 
    # Actually, GSM8KPromptSet does not expose a list of all roles directly except via 'roles' iterator which is infinite.
    # But we can see the keys in ROLE_DESCRIPTION in the file content.
    
    roles = ["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"]
    
    for i, role in enumerate(roles):
        agent_name = "MathAgent" # All are MathAgent class agents
        AgentClass = AgentRegistry.get_class(agent_name)
        
        description = GSM8KPromptSet().get_description(role) 
        
        from RAPS.prompt.gsm8k_prompt_set import FEW_SHOT_DATA
        few_shot = FEW_SHOT_DATA.get(role, "")
        
        agent = AgentClass(
            id=None,
            llm_name=llm_name,
            role=role,
            capabilities=description, 
            interests=f"",
            additional_instructions=None,
            few_shot=few_shot
        )
        
        # Initialize custom attributes for the workflow
        agent.history = []  # Short-term memory
        agent.inbox = []    # Current step messages
        agents.append(agent)
        
    return agents

def initialize_final_answerer(llm_name: str):
    
    AgentClass = AgentRegistry.get_class("MathAgent")
    
    role = "Final Answerer"
    
    agent = AgentClass(
        id=None,
        llm_name=llm_name,
        role=role,
        capabilities="You are the top decision-maker. You are Good at analyzing and summarizing mathematical problems, judging and summarizing other people's solutions, and giving final answers to math problems.",
        interests="",
        additional_instructions=GSM8KPromptSet.get_decision_constraint(),
        few_shot=GSM8KPromptSet.get_decision_few_shot()
    )
    return agent

def run_gsm8k_task(agents, final_answerer, task_example, args):
    """
    Main Execution Loop (Per Task)
    """
    task = task_example['task']
    answer = task_example.get('answer', None)

    _log(f"Task: {task}")
    _log(f"Answer: {answer}")
    
    # === Phase A: Setup ===
    
    # Create a log entry for this task
    task_log = {
        "task": task,
        "answer": answer,
        "steps": []
    }

    # 1. State Reset
    for agent in agents:
        agent.history = []
        agent.inbox = []
        # agent.current_subscription is managed by SubscriptionTemplate
        agent.refined_prompt = agent.system_prompt
    
    active_agents = []
    
    # 2. Bootstrapping
    # Identify Entry Agent (First defined agent, e.g. Math Agent)
    entry_agent = agents[1]
    active_agents = [entry_agent]
    
    # Inject user query into Entry Agent's inbox
    entry_agent.inbox.append({
        "sender_id": "User",
        "role": "User",
        "content": task
    })
    
    round_publications = [] # Global log
    refinement_record = {} 
    
    # === Phase B: Interaction Loop ===
    
    for step in range(1, args.max_steps + 1):
        # print(f"--- Step {step} ---")
        
        step_log = {
            "step": step,
            "active_agents": [a.id for a in active_agents],
            "publications": [],
            "refinements": {},
            "broker_decisions": [],
            "llm_calls": {}
        }

        _log(f"--- Step {step} ---")
        _log(f"Active Agents: {[a.id for a in active_agents]}")
        _log(f"Entry Agent: {entry_agent.id}")

        for a in agents:
            if hasattr(a, "llm_trace"):
                a.llm_trace = []

        current_step_pubs = []
        upstream_map = {a.id: set() for a in active_agents}
        upstream_info_map = {a.id: [] for a in active_agents}
        
        # 1. Pre-Computation (Inbox Processing)
        for agent in active_agents:
            for msg in agent.inbox:
                sender_id = msg.get("sender_id")
                
                if sender_id != "User":
                    upstream_map[agent.id].add(sender_id)
                    upstream_info_map[agent.id].append(f"{msg['role']} ({sender_id}): {msg['content']}")
                
                    # Memory Update (store all incoming messages)
                    agent.history.append(f"{msg['role']}: {msg['content']}")
            
            agent.inbox = []

        # 2. Adaptivity (Reactive Subscription)
        for agent in active_agents:
            # Capture pre-refinement prompt (base)

            # Context includes intermediate results and current plan (from history)
            if step == 1:
                context_str = "History:\nNone"
            else:
                context_str = "History:\n" + "\n".join(agent.history)
            refined_sub = agent.refine_system_prompt(context=context_str, question=task)
            # refinement_record[agent.id] = refined_sub
            
            # Log detail: Pre and Post refinement
            step_log["refinements"][agent.id] = {
                "role": agent.role,
                "pre_refinement": agent.system_prompt,
                "post_refinement": refined_sub
            }
            _log(f"Step {step} Subscribe Input | {agent.id} ({agent.role}) | question: {task}")
            _log(f"Step {step} Subscribe Context | {agent.id} ({agent.role}) | {context_str}")
            _log(f"Step {step} Subscribe Output | {agent.id} ({agent.role}) | {refined_sub}")
            # agent.subscription_prompt.set_current_prompt(refined_sub) # Already updated by refine_subscription
        
        # 3. Action (Publication)
        temp_pubs = []
        for agent in active_agents:
            context_str = "\n".join(agent.history)
            
            # Pass unified context
            publication_context = {
                "task": task,
                "history": context_str
            }
            output = agent.publish(publication_context)
            
            # Parse output as JSON if possible
            pub_obj = {
                "sender_id": agent.id,
                "role": agent.role,
                "content": output
            }
            
            temp_pubs.append(pub_obj)
            round_publications.append(pub_obj)
            current_step_pubs.append(pub_obj)
            step_log["publications"].append(pub_obj)
            _log(f"Step {step} Publish Input | {agent.id} ({agent.role}) | {publication_context}")
            _log(f"Step {step} Publish Output | {agent.id} ({agent.role}) | {output}")

        # 4. Upstream Watchdog Verification (Blocking Logic)
        if not current_step_pubs:
            break
            
        agent_map = {a.id: a for a in agents}
        valid_pubs = []
        passed_senders = set()
        for pub in current_step_pubs:
            sender_id = pub["sender_id"]
            content = pub["content"]
            other_info = "\n".join(upstream_info_map.get(sender_id, []))
            
            # Get upstream agents for this sender (from current step's inbox)
            upstream_ids = upstream_map.get(sender_id, set())
            
            is_blocked = False
            blocking_agents = []
            
            for u_id in upstream_ids:
                if u_id not in agent_map:
                    continue
                upstream_agent = agent_map[u_id]
                
                # Upstream agent verifies the output derived from their input
                is_valid = upstream_agent.watchdog_evaluate(content, task_domain="gsm8k", question=task, existing_info=other_info)
                upstream_agent.rep_manager.update_first_hand(sender_id, is_valid)
                if not is_valid:
                    is_blocked = True
                    blocking_agents.append(u_id)
                    break 
            
            if is_blocked:
                _log(f"[WATCHDOG-BLOCK] Agent {sender_id}'s output blocked by upstream: {blocking_agents}")
                step_log["broker_decisions"].append({
                    "sender": sender_id,
                    "blocked_by": blocking_agents,
                    "status": "blocked"
                })
            else:
                valid_pubs.append(pub)
                passed_senders.add(sender_id)
        
        current_step_pubs = valid_pubs

        if len(passed_senders) == 0:
             _log("[WATCHDOG-BLOCK] All publications blocked or empty.")
             break
            
        # 5. Routing (Embedding Brokerage)
        next_active_agents = set()
        
        all_subscriptions = {}
        
        for a in agents:
             curr_sub = a.refined_prompt
             # all_subscriptions[f"{a.id}"] = {"agent_id": a.id, "prompt": curr_sub}
             all_subscriptions[a.id] = curr_sub

        # Prepare context for broker from all publications in this step
        step_pubs_str = [f"Agent {p['sender_id']} ({p['role']}): {p['content']}" for p in current_step_pubs]
        
        # Check if all agents are ready to give final answer
        all_ready = True
        for pub in current_step_pubs:
            if "final answer: yes" not in pub["content"].lower():
                all_ready = False
                break
        
        if all_ready or step == args.max_steps:
             # Trigger Final Answerer
             break # Exit loop to Phase C

        # Broker Routing (Per-agent broker)
        for pub in current_step_pubs:
            sender_id = pub["sender_id"]
            if sender_id not in agent_map:
                continue

            matcher_agent = agent_map[sender_id]
            try:
                _log(f"Step {step} Broker Input | {sender_id} ({matcher_agent.role}) | publications: {step_pubs_str} | subscriptions: {all_subscriptions}")
                matched_keys = matcher_agent.broker_route(step_pubs_str, all_subscriptions, top_k=1, sim_threshold=0.3)
            except Exception as e:
                _log(f"Broker routing error: {e}")
                matched_keys = []

            selected_receivers = []

            for receiver_id in matched_keys:
                if receiver_id not in agent_map:
                    continue

                receiver = agent_map[receiver_id]
                if receiver_id == sender_id:
                    continue

                receiver.inbox.append(pub)
                next_active_agents.add(receiver)
                selected_receivers.append(receiver_id)
            _log(f"Step {step} Broker Output | {sender_id} ({matcher_agent.role}) | matched: {matched_keys} | receivers: {selected_receivers}")

            step_log["broker_decisions"].append({
                "sender": sender_id,
                "matched_keys": matched_keys,
                "receivers": selected_receivers
            })
        
        step_log["llm_calls"] = {
            a.id: a.llm_trace for a in agents if hasattr(a, "llm_trace") and len(a.llm_trace)
        }
        task_log["steps"].append(step_log)

        if not next_active_agents:
            break
        else:
            active_agents = list(next_active_agents)

    # === Phase C: Decision & Learning ===
    
    # Final Decision using Final Answerer
    combined_output = "\n".join([f"{p['role']} ({p['sender_id']}): {p['content']}" for p in round_publications])
    
    final_answerer.llm_trace = []

    # Final Answerer Refinement
    context_str = f"Original Task: {task}\n\nProcess History:\n{combined_output}"
    final_answerer.refine_system_prompt(context=context_str, question=task)
    
    # Final Answerer Publication
    decision_input = {
        "task": task,
        "history": combined_output
    }
    final_output = final_answerer.publish(decision_input)
    _log(f"Final Answerer Input | question: {task} | context: {context_str}")
    _log(f"Final Answerer Publish Input | {decision_input}")
    _log(f"Final Answerer Output | {final_output}")
    
    # Log Final Answerer Activity
    final_log = {
        "step": "Final",
        "active_agents": ["Final_Answerer"],
        "publications": [{
            "sender_id": "Final_Answerer",
            "role": "Final Answerer",
            "content": final_output,
            "embedding": None
        }],
        "refinements": {
            "Final_Answerer": final_answerer.refined_prompt
        },
        "llm_calls": {
            "Final_Answerer": final_answerer.llm_trace
        },
        "broker_decisions": []
    }
    task_log["steps"].append(final_log)
    
    pred = gsm_get_predict(final_output)
    
    # Evaluation
    is_solved = (float(pred) == float(answer)) if answer is not None else False
    
    # Save Log
    log_dir = RAPS_ROOT / "logs"
    os.makedirs(log_dir, exist_ok=True)
    current_time = Time.instance().value
    log_file = log_dir / f"log_{current_time}_{task[:20].replace(' ', '_')}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(task_log, f, indent=2)

    return {
        "publications": round_publications,
        "refinement_record": refinement_record,
        "final_output": final_output,
        "prediction": pred,
        "is_solved": is_solved
    }

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def main():
    args = parse_args()
    dataset = load_jsonl(args.dataset_json)
    dataset = gsm_data_process(dataset)
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{RAPS_ROOT}/result/gsm8k")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"

    # Initialize Agents from GSM8KPromptSet
    agents = initialize_agents_from_set(args.llm_name)
    final_answerer = initialize_final_answerer(args.llm_name)

    all_results = []
    total_solved, total_executed = (0, 0)

    # Test on first 50 examples
    test_dataset = dataset[54:100]

    for i, example in enumerate(test_dataset):
        _log(f"================ Question {i} ================")
        
        result_data = run_gsm8k_task(agents, final_answerer, example, args)
        
        task = example['task']
        answer = example.get('answer', None)
        publications = result_data["publications"]
        final_output = result_data["final_output"]
        pred = result_data["prediction"]
        is_solved = result_data["is_solved"]
        
        publications_str = ""
        for pubs in publications:
            publications_str += f"{pubs['role']}: {pubs['content']}\n"

        total_solved = total_solved + (1 if is_solved else 0)
        total_executed = total_executed + 1
        accuracy = total_solved / total_executed

        result = {
            "task": task,
            "answer": answer,
            "publications": publications_str,
            "response": final_output,
            "prediction": pred,
            "correct": is_solved,
            "total solved": total_solved,
            "total executed": total_executed,
            "accuracy": accuracy
        }
        
        all_results.append(result)
        
        _log(f"Predicted: {pred} | Answer: {answer} | Correct: {is_solved}")
        _log(f"Current Accuracy: {accuracy:.4f} ({total_solved}/{total_executed})")
    
    with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
    
    _log("====== Final Stats ======")
    _log(f"Accuracy: {accuracy:.4f}")
    _log(f"Total Solved: {total_solved}")
    _log(f"Total Executed: {total_executed}")
    _log(f"Total Cost: {Cost.instance().value}")
    _log(f"PromptTokens: {PromptTokens.instance().value}")
    _log(f"CompletionTokens: {CompletionTokens.instance().value}")


if __name__ == "__main__":
    main()
