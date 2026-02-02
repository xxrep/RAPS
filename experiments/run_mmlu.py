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
from RAPS.agents.analyze_agent import AnalyzeAgent
from RAPS.prompt.mmlu_prompt_set import MMLUPromptSet
from datasets.mmlu_dataset import MMLUDataset


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str):
    print(f"[{_ts()}] {message}")


def initialize_agents_from_set(llm_name: str):
    agents = []
    roles = ["Knowledge Expert", "Critic", "Mathematician", "Psychologist", "Historian", "Doctor", "Lawyer", "Economist", "Programmer"]
    for role in roles:
        AgentClass = AgentRegistry.get_class("AnalyzeAgent")
        description = MMLUPromptSet().get_description(role)
        # constraint = MMLUPromptSet.get_constraint()
        if role == "Knowledge Expert":
            combined_prompt = f"{description}\n"
        else:
            combined_prompt = f"{description}\nYour reply must be less than 100 words and include a brief step by step analysis of the question.\n"
        agent = AgentClass(
            id=None,
            llm_name=llm_name,
            role=role,
            capabilities=combined_prompt,
            interests="",
            additional_instructions=""
        )
        agent.history = []
        agent.inbox = []
        agents.append(agent)
    return agents


def initialize_final_answerer(llm_name: str):
    AgentClass = AgentRegistry.get_class("AnalyzeAgent")
    role = "Final Answerer"
    # decision_prompt = f"{MMLUPromptSet.get_decision_role()}\n{MMLUPromptSet.get_decision_constraint()}"
    agent = AgentClass(
        id=None,
        llm_name=llm_name,
        role=role,
        capabilities=MMLUPromptSet.get_decision_role(),
        interests="",
        additional_instructions=MMLUPromptSet.get_decision_constraint()
    )
    return agent


def run_mmlu_task(agents, final_answerer, record, dataset, args):
    task_dict = MMLUDataset.record_to_input(record)
    task = task_dict["task"]
    answer = MMLUDataset.record_to_target_answer(record)

    _log(f"Task: {task}")
    _log(f"Answer: {answer}")

    task_log = {
        "task": task,
        "answer": answer,
        "steps": []
    }

    for agent in agents:
        agent.history = []
        agent.inbox = []
        agent.refined_prompt = agent.system_prompt

    active_agents = []

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

        for agent in active_agents:
            for msg in agent.inbox:
                sender_id = msg.get("sender_id")
                if sender_id != "User":
                    upstream_map[agent.id].add(sender_id)
                    upstream_info_map[agent.id].append(f"{msg['role']} ({sender_id}): {msg['content']}")
                    agent.history.append(f"{msg['role']}: {msg['content']}")
            agent.inbox = []

        for agent in active_agents:
            pre_refinement_prompt = agent.system_prompt
            if step == 1:
                context_str = "History:\nNone"
            else:
                context_str = "History:\n" + "\n".join(agent.history)
            refined_sub = agent.refine_system_prompt(context=context_str, question=task)
            refinement_record[agent.id] = refined_sub
            step_log["refinements"][agent.id] = {
                "role": agent.role,
                "pre_refinement": pre_refinement_prompt,
                "post_refinement": refined_sub
            }
            _log(f"Step {step} Subscribe Input | {agent.id} ({agent.role}) | question: {task}")
            _log(f"Step {step} Subscribe Context | {agent.id} ({agent.role}) | {context_str}")
            _log(f"Step {step} Subscribe Output | {agent.id} ({agent.role}) | {refined_sub}")

        temp_pubs = []
        for agent in active_agents:

            publication_context = {
                "task": task,
                "history": "\n".join(agent.history)
            }
            output = agent.publish(publication_context)
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

        if not current_step_pubs:
            step_log["llm_calls"] = {
                a.id: a.llm_trace for a in agents if hasattr(a, "llm_trace") and len(a.llm_trace)
            }
            task_log["steps"].append(step_log)
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
            blocking_agents = []
            for u_id in upstream_ids:
                if u_id not in agent_map:
                    continue
                upstream_agent = agent_map[u_id]
                is_valid = upstream_agent.watchdog_evaluate(content, task_domain="mmlu", question=task, existing_info=other_info)
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
            step_log["llm_calls"] = {
                a.id: a.llm_trace for a in agents if hasattr(a, "llm_trace") and len(a.llm_trace)
            }
            task_log["steps"].append(step_log)
            break

        next_active_agents = set()
        all_subscriptions = {}
        for a in agents:
            all_subscriptions[a.id] = a.refined_prompt

        step_pubs_str = [f"Agent {p['sender_id']} ({p['role']}): {p['content']}" for p in current_step_pubs]

        all_ready = True
        for pub in current_step_pubs:
            if "final answer: yes" not in str(pub["content"]).lower():
                all_ready = False
                break
        if all_ready or step == args.max_steps:
            step_log["llm_calls"] = {
                a.id: a.llm_trace for a in agents if hasattr(a, "llm_trace") and len(a.llm_trace)
            }
            task_log["steps"].append(step_log)
            _log(f"Max steps reached: {args.max_steps}")
            break

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
        active_agents = list(next_active_agents)

    combined_output = "\n".join([f"{p['role']} ({p['sender_id']}): {p['content']}" for p in round_publications])

    final_answerer.llm_trace = []
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

    pred = dataset.postprocess_answer(final_output)
    is_solved = (pred == answer)

    log_dir = RAPS_ROOT / "logs"
    os.makedirs(log_dir, exist_ok=True)
    current_time = Time.instance().value
    log_file = log_dir / f"log_mmlu_{current_time}_{task[:10]}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(task_log, f, indent=2)

    return {
        "publications": round_publications,
        "refinement_record": refinement_record,
        "final_output": final_output,
        "prediction": pred,
        "is_solved": is_solved
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_steps", type=int, default=5)
    args = parser.parse_args()

    dataset = MMLUDataset(split="test")
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    agents = initialize_agents_from_set(args.llm_name)
    final_answerer = initialize_final_answerer(args.llm_name)

    all_results = []
    total_solved, total_executed = (0, 0)

    # test_indices = list(range(min(159, len(dataset))))
    test_indices = range(159)
    for i in test_indices:
        record = dataset[i]
        _log(f"================ Question {i} ================")
        result_data = run_mmlu_task(agents, final_answerer, record, dataset, args)
        answer = MMLUDataset.record_to_target_answer(record)
        pred = result_data["prediction"]
        is_solved = result_data["is_solved"]
        total_solved = total_solved + (1 if is_solved else 0)
        total_executed = total_executed + 1
        accuracy = total_solved / total_executed
        result = {
            "answer": answer,
            "prediction": pred,
            "correct": is_solved,
            "total solved": total_solved,
            "total executed": total_executed,
            "accuracy": accuracy
        }
        all_results.append(result)
        _log(f"Predicted: {pred} | Answer: {answer} | Correct: {is_solved}")
        _log(f"Current Accuracy: {accuracy:.4f} ({total_solved}/{total_executed})")

    result_dir = Path(f"{RAPS_ROOT}/result/mmlu")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"mmlu_{args.llm_name}_{current_time}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    _log("====== Final Stats ======")
    _log(f"Accuracy: {total_solved / total_executed if total_executed else 0}")
    _log(f"Total Solved: {total_solved}")
    _log(f"Total Executed: {total_executed}")


if __name__ == "__main__":
    main()
