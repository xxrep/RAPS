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
from RAPS.utils.globals import Time
from RAPS.agents.agent_registry import AgentRegistry
from RAPS.agents.code_writing import CodeWriting
from RAPS.prompt.humaneval_prompt_set import HumanEvalPromptSet
from RAPS.tools.coding.python_executor import PyExecutor


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str):
    print(f"[{_ts()}] {message}")


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_code(output):
    if not isinstance(output, str):
        return ""
    if "```python" in output:
        parts = output.split("```python", 1)[-1]
        if "```" in parts:
            return parts.split("```", 1)[0].strip()
        return parts.strip()
    if "```" in output:
        parts = output.split("```", 1)[-1]
        if "```" in parts:
            return parts.split("```", 1)[0].strip()
    return output.strip()


def initialize_agents_from_set(llm_name: str):
    agents = []
    roles = ["Project Manager", "Algorithm Designer", "Programming Expert", "Test Analyst", "Bug Fixer"]
    for role in roles:
        AgentClass = AgentRegistry.get_class("CodeWriting")
        description = HumanEvalPromptSet().get_description(role)
        agent = AgentClass(
            id=None,
            llm_name=llm_name,
            role=role,
            capabilities=description,
            interests="",
            additional_instructions="",
            few_shot=""
        )
        agent.history = []
        agent.inbox = []
        agents.append(agent)
    return agents


def initialize_final_answerer(llm_name: str):
    AgentClass = AgentRegistry.get_class("CodeWriting")
    role = "Final Answerer"
    description = HumanEvalPromptSet().get_decision_role()
    agent = AgentClass(
        id=None,
        llm_name=llm_name,
        role=role,
        capabilities=description,
        interests="",
        additional_instructions=HumanEvalPromptSet().get_decision_constraint(),
        few_shot=""
    )
    return agent


def run_humaneval_task(agents, final_answerer, record, args):
    task = record["prompt"]
    entry_point = record["entry_point"]
    test = record["test"]

    for agent in agents:
        agent.history = []
        agent.inbox = []
        agent.refined_prompt = agent.system_prompt

    entry_agent = agents[0]
    active_agents = [entry_agent]

    entry_agent.inbox.append({
        "sender_id": "User",
        "role": "User",
        "content": task
    })

    round_publications = []
    refinement_record = {}

    for step in range(1, args.max_steps + 1):
        _log(f"--- Step {step} ---")
        _log(f"Active Agents: {[a.id for a in active_agents]}")
        current_step_pubs = []
        upstream_map = {a.id: set() for a in active_agents}
        upstream_info_map = {a.id: [] for a in active_agents}

        for agent in active_agents:
            for msg in agent.inbox:
                sender_id = msg.get("sender_id")
                if sender_id != "User":
                    upstream_map[agent.id].add(sender_id)
                    upstream_info_map[agent.id].append(f"{msg['role']} ({sender_id}): {msg['content']}")
                    agent.history.append(f"{msg['role']}: {msg['content']}")
            agent.inbox = []

        for agent in active_agents:
            if step == 1:
                context_str = "History:\nNone"
            else:
                context_str = "History:\n" + "\n".join(agent.history)
            refined_sub = agent.refine_system_prompt(context=context_str, question=task)
            refinement_record[agent.id] = refined_sub
            _log(f"Step {step} Subscribe Input | {agent.id} ({agent.role}) | question: {task}")
            _log(f"Step {step} Subscribe Context | {agent.id} ({agent.role}) | {context_str}")
            _log(f"Step {step} Subscribe Output | {agent.id} ({agent.role}) | {refined_sub}")

        for agent in active_agents:
            publication_context = {
                "task": task,
                "history": "\n".join(agent.history)
            }
            output = agent.publish(publication_context)
            pub_obj = {
                "sender_id": agent.id,
                "role": agent.role,
                "content": output,
                "embedding": None
            }
            round_publications.append(pub_obj)
            current_step_pubs.append(pub_obj)
            _log(f"Step {step} Publish Input | {agent.id} ({agent.role}) | {publication_context}")
            _log(f"Step {step} Publish Output | {agent.id} ({agent.role}) | {output}")

        if not current_step_pubs:
            break

        agent_map = {a.id: a for a in agents}
        valid_pubs = []
        passed_senders = set()
        for pub in current_step_pubs:
            sender_id = pub["sender_id"]
            content = pub["content"]
            other_info = "\n".join(upstream_info_map.get(sender_id, []))
            upstream_ids = upstream_map.get(sender_id, set())
            is_blocked = False
            for u_id in upstream_ids:
                if u_id not in agent_map:
                    continue
                upstream_agent = agent_map[u_id]
                is_valid = upstream_agent.watchdog_evaluate(content, task_domain="humaneval", question=task, existing_info=other_info)
                upstream_agent.rep_manager.update_first_hand(sender_id, is_valid)
                if not is_valid:
                    is_blocked = True
                    break
            if not is_blocked:
                valid_pubs.append(pub)
                passed_senders.add(sender_id)

        current_step_pubs = valid_pubs
        if len(passed_senders) == 0:
            break

        if step == args.max_steps:
            break

        next_active_agents = set()
        all_subscriptions = {}
        for a in agents:
            all_subscriptions[a.id] = a.refined_prompt

        step_pubs_str = [f"Agent {p['sender_id']} ({p['role']}): {p['content']}" for p in current_step_pubs]

        for pub in current_step_pubs:
            sender_id = pub["sender_id"]
            if sender_id not in agent_map:
                continue
            matcher_agent = agent_map[sender_id]
            try:
                _log(f"Step {step} Broker Input | {sender_id} ({matcher_agent.role}) | publications: {step_pubs_str} | subscriptions: {all_subscriptions}")
                matched_keys = matcher_agent.broker_route(step_pubs_str, all_subscriptions, top_k=1, sim_threshold=0.3)
            except Exception:
                matched_keys = []
            selected_receivers = []
            for receiver_id in matched_keys:
                if receiver_id not in agent_map:
                    continue
                if receiver_id == sender_id:
                    continue
                receiver = agent_map[receiver_id]
                receiver.inbox.append(pub)
                next_active_agents.add(receiver)
                selected_receivers.append(receiver_id)
            _log(f"Step {step} Broker Output | {sender_id} ({matcher_agent.role}) | matched: {matched_keys} | receivers: {selected_receivers}")

        if not next_active_agents:
            break
        active_agents = list(next_active_agents)

    combined_output = "\n".join([f"{p['role']} ({p['sender_id']}): {p['content']}" for p in round_publications])
    final_answerer.refined_prompt = final_answerer.system_prompt
    context_str = f"Original Task: {task}\n\nProcess History:\n{combined_output}"
    final_answerer.refine_system_prompt(context=context_str, question=task)
    decision_input = {
        "task": task,
        "history": combined_output
    }
    final_output = final_answerer.publish(decision_input)
    _log(f"Final Answerer Input | question: {task} | context: {context_str}")
    _log(f"Final Answerer Publish Input | {decision_input}")
    _log(f"Final Answerer Output | {final_output}")
    code = extract_code(final_output)
    is_solved = PyExecutor().evaluate(entry_point, code, test)
    return {
        "final_output": final_output,
        "code": code,
        "is_solved": is_solved
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_steps", type=int, default=5)
    args = parser.parse_args()

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    data_path = Path(f"{RAPS_ROOT}/datasets/humaneval/humaneval-py.jsonl")
    dataset = load_jsonl(data_path)[:20]

    agents = initialize_agents_from_set(args.llm_name)
    final_answerer = initialize_final_answerer(args.llm_name)

    all_results = []
    total_solved, total_executed = (0, 0)

    for i, record in enumerate(dataset):
        _log(f"================ Question {i} ================")
        result_data = run_humaneval_task(agents, final_answerer, record, args)
        is_solved = result_data["is_solved"]
        total_solved = total_solved + (1 if is_solved else 0)
        total_executed = total_executed + 1
        pass_at_1 = total_solved / total_executed
        all_results.append({
            "name": record["name"],
            "entry_point": record["entry_point"],
            "pass@1": is_solved,
            "code": result_data["code"]
        })
        _log(f"Pass@1: {is_solved}")
        _log(f"Current Pass@1: {pass_at_1:.4f} ({total_solved}/{total_executed})")

    result_dir = Path(f"{RAPS_ROOT}/result/humaneval")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"humaneval_{args.llm_name}_{current_time}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    _log("====== Final Stats ======")
    _log(f"Pass@1: {total_solved / total_executed if total_executed else 0}")
    _log(f"Total Solved: {total_solved}")
    _log(f"Total Executed: {total_executed}")


if __name__ == "__main__":
    main()
