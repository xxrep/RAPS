import sys
import os
import re
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.utils.const import RAPS_ROOT
from RAPS.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from RAPS.agents.agent_registry import AgentRegistry
from RAPS.core import RAPSCoordinator, RAPSConfig
from raps_data.gsm8k_dataset import gsm_data_process, gsm_get_predict
from RAPS.prompt.gsm8k_prompt_set import GSM8KPromptSet, FEW_SHOT_DATA
# Explicitly import agent to ensure registration
from RAPS.agents.math_solver import MathAgent


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str):
    print(f"[{_ts()}] {message}")


ROLES = ["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"]


def initialize_agents_from_set(llm_name: str):
    """One MathAgent per GSM8K role."""
    agents = []
    AgentClass = AgentRegistry.get_class("MathAgent")
    prompt_set = GSM8KPromptSet()
    for role in ROLES:
        agent = AgentClass(
            id=None,
            llm_name=llm_name,
            role=role,
            capabilities=prompt_set.get_description(role),
            interests="",
            additional_instructions=None,
            few_shot=FEW_SHOT_DATA.get(role, ""),
        )
        agent.history = []
        agent.inbox = []
        agents.append(agent)
    return agents


def initialize_final_answerer(llm_name: str):
    AgentClass = AgentRegistry.get_class("MathAgent")
    return AgentClass(
        id=None,
        llm_name=llm_name,
        role="Final Answerer",
        capabilities="You are the top decision-maker. You are Good at analyzing and summarizing "
                     "mathematical problems, judging and summarizing other people's solutions, "
                     "and giving final answers to math problems.",
        interests="",
        additional_instructions=GSM8KPromptSet.get_decision_constraint(),
        few_shot=GSM8KPromptSet.get_decision_few_shot(),
    )


def is_correct(pred, answer) -> bool:
    """Robust numeric comparison with a string fallback (no crash on non-numeric pred)."""
    if answer is None:
        return False
    try:
        return abs(float(pred) - float(answer)) < 1e-6
    except (ValueError, TypeError):
        return str(pred).strip() == str(answer).strip()


def _safe_slug(text, n=20):
    """Filesystem-safe slug (avoids '/' etc. in task text breaking the path)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)[:n]).strip("_") or "task"


def write_task_log(task_log, answer):
    task_log["answer"] = answer
    log_dir = RAPS_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    task = task_log.get("task", "task")
    log_file = log_dir / f"log_gsm8k_{Time.instance().value}_{_safe_slug(task)}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(task_log, f, indent=2)


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_args():
    parser = argparse.ArgumentParser(description="RAPS Experiments on GSM8K")
    parser.add_argument("--dataset_json", type=str, default="raps_data/gsm8k/gsm8k.jsonl")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--domain", type=str, default="gsm8k")
    parser.add_argument("--max_steps", type=int, default=3, help="Max communication steps")
    parser.add_argument("--top_k", type=int, default=3, help="Broker fan-out cap per publisher")
    parser.add_argument("--sim_threshold", type=float, default=0.70,
                        help="Embedding gate; >=1 best match always engaged (fallback)")
    parser.add_argument("--entry_index", type=int, default=1, help="Entry agent index (default Mathematical Analyst)")
    parser.add_argument("--start", type=int, default=0, help="First example index")
    parser.add_argument("--limit", type=int, default=50, help="Number of examples to run")
    parser.add_argument("--reputation_gate", action="store_true", help="Isolate distrusted peers in routing")
    parser.add_argument("--second_hand_gossip", action="store_true", help="Exchange reputation reports each step")
    parser.add_argument("--dynamic_recruit", action="store_true", help="Recruit specialists from the seed pool on demand")
    parser.add_argument("--adaptive_capacity", action="store_true", help="Scale broker fan-out to task difficulty")
    parser.add_argument("--max_team_size", type=int, default=8)
    parser.add_argument("--max_top_k", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = gsm_data_process(load_jsonl(args.dataset_json))

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{RAPS_ROOT}/result/gsm8k")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"

    agents = initialize_agents_from_set(args.llm_name)
    final_answerer = initialize_final_answerer(args.llm_name)
    config = RAPSConfig(
        domain=args.domain, max_steps=args.max_steps, top_k=args.top_k,
        sim_threshold=args.sim_threshold, entry_index=args.entry_index,
        reputation_gate=args.reputation_gate, second_hand_gossip=args.second_hand_gossip,
        dynamic_recruit=args.dynamic_recruit, adaptive_capacity=args.adaptive_capacity,
        max_team_size=args.max_team_size, max_top_k=args.max_top_k,
    )
    coordinator = RAPSCoordinator(agents, final_answerer, config,
                                  logger=_log, answer_extractor=gsm_get_predict)

    test_dataset = dataset[args.start:args.start + args.limit]
    all_results = []
    total_solved = total_executed = 0

    for i, example in enumerate(test_dataset):
        _log(f"================ Question {args.start + i} ================")
        task, answer = example["task"], example.get("answer")
        _log(f"Task: {task}")

        result = coordinator.run(task)
        write_task_log(result.task_log, answer)

        pred = gsm_get_predict(result.final_output)
        solved = is_correct(pred, answer)
        total_solved += 1 if solved else 0
        total_executed += 1
        accuracy = total_solved / total_executed

        publications_str = "".join(f"{p['role']}: {p['content']}\n" for p in result.publications)
        all_results.append({
            "task": task, "answer": answer, "publications": publications_str,
            "response": result.final_output, "prediction": pred, "correct": solved,
            "total solved": total_solved, "total executed": total_executed, "accuracy": accuracy,
        })
        _log(f"Predicted: {pred} | Answer: {answer} | Correct: {solved}")
        _log(f"Current Accuracy: {accuracy:.4f} ({total_solved}/{total_executed})")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    _log("====== Final Stats ======")
    _log(f"Accuracy: {total_solved / total_executed if total_executed else 0:.4f}")
    _log(f"Total Solved: {total_solved} / {total_executed}")
    _log(f"Total Cost: ${Cost.instance().value:.4f} | PromptTokens: {PromptTokens.instance().value} "
         f"| CompletionTokens: {CompletionTokens.instance().value}")


if __name__ == "__main__":
    main()
