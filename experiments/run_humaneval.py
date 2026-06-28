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
from RAPS.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from RAPS.agents.agent_registry import AgentRegistry
from RAPS.core import RAPSCoordinator, RAPSConfig
from RAPS.agents.code_writing import CodeWriting
from RAPS.prompt.humaneval_prompt_set import HumanEvalPromptSet
from RAPS.tools.coding.python_executor import PyExecutor


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str):
    print(f"[{_ts()}] {message}")


ROLES = ["Project Manager", "Algorithm Designer", "Programming Expert", "Test Analyst", "Bug Fixer"]


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
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
    AgentClass = AgentRegistry.get_class("CodeWriting")
    prompt_set = HumanEvalPromptSet()
    for role in ROLES:
        agent = AgentClass(
            id=None, llm_name=llm_name, role=role,
            capabilities=prompt_set.get_description(role),
            interests="", additional_instructions="", few_shot="",
        )
        agent.history = []
        agent.inbox = []
        agents.append(agent)
    return agents


def initialize_final_answerer(llm_name: str):
    AgentClass = AgentRegistry.get_class("CodeWriting")
    return AgentClass(
        id=None, llm_name=llm_name, role="Final Answerer",
        capabilities=HumanEvalPromptSet().get_decision_role(),
        interests="", additional_instructions=HumanEvalPromptSet().get_decision_constraint(),
        few_shot="",
    )


def write_task_log(task_log, record):
    task_log["entry_point"] = record["entry_point"]
    log_dir = RAPS_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"log_humaneval_{Time.instance().value}_{record['entry_point'][:20]}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(task_log, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="RAPS Experiments on HumanEval")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--domain", type=str, default="humaneval")
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--sim_threshold", type=float, default=0.60)
    parser.add_argument("--entry_index", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--reputation_gate", action="store_true")
    parser.add_argument("--second_hand_gossip", action="store_true")
    parser.add_argument("--dynamic_recruit", action="store_true")
    parser.add_argument("--adaptive_capacity", action="store_true")
    parser.add_argument("--max_team_size", type=int, default=10)
    parser.add_argument("--max_top_k", type=int, default=2)
    parser.add_argument("--code_verify", action="store_true",
                        help="Execute-feedback-repair on public doctests before finalizing")
    parser.add_argument("--code_verify_max_iters", type=int, default=3)
    parser.add_argument("--edge_cases", action="store_true",
                        help="Edge-Case Writer agent generates extra tests for the Code Verifier")
    return parser.parse_args()


def main():
    args = parse_args()
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    data_path = Path(f"{RAPS_ROOT}/raps_data/humaneval/humaneval-py.jsonl")
    dataset = load_jsonl(data_path)[args.start:args.start + args.limit]

    agents = initialize_agents_from_set(args.llm_name)
    final_answerer = initialize_final_answerer(args.llm_name)
    config = RAPSConfig(
        domain=args.domain, max_steps=args.max_steps, top_k=args.top_k,
        sim_threshold=args.sim_threshold, entry_index=args.entry_index,
        reputation_gate=args.reputation_gate, second_hand_gossip=args.second_hand_gossip,
        dynamic_recruit=args.dynamic_recruit, adaptive_capacity=args.adaptive_capacity,
        max_team_size=args.max_team_size, max_top_k=args.max_top_k,
        code_verify=args.code_verify, code_verify_max_iters=args.code_verify_max_iters,
        edge_cases=args.edge_cases,
    )
    coordinator = RAPSCoordinator(agents, final_answerer, config, logger=_log)

    all_results = []
    total_solved = total_executed = 0

    for i, record in enumerate(dataset):
        _log(f"================ Question {args.start + i} ================")
        # Per-problem resilience: a transient gateway hiccup shouldn't kill the whole run.
        result = None
        for attempt in range(4):
            try:
                result = coordinator.run(record["prompt"])
                break
            except Exception as e:
                _log(f"[RETRY] problem {args.start + i} attempt {attempt + 1} failed: {e}")
                time.sleep(10 * (attempt + 1))
        if result is None:
            _log(f"[SKIP] problem {args.start + i} failed after retries; counting as fail")
            all_results.append({"name": record["name"], "entry_point": record["entry_point"],
                                "pass@1": False, "code": "", "errored": True})
            total_executed += 1
            continue
        write_task_log(result.task_log, record)

        code = extract_code(result.final_output)
        solved = PyExecutor().evaluate(record["entry_point"], code, record["test"])
        total_solved += 1 if solved else 0
        total_executed += 1
        pass_at_1 = total_solved / total_executed

        all_results.append({
            "name": record["name"], "entry_point": record["entry_point"],
            "pass@1": solved, "code": code,
        })
        _log(f"Pass@1: {solved} | Current Pass@1: {pass_at_1:.4f} ({total_solved}/{total_executed})")

    result_dir = Path(f"{RAPS_ROOT}/result/humaneval")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"humaneval_{args.llm_name}_{current_time}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    _log("====== Final Stats ======")
    _log(f"Pass@1: {total_solved / total_executed if total_executed else 0:.4f}")
    _log(f"Total Solved: {total_solved} / {total_executed}")
    _log(f"Total Cost: ${Cost.instance().value:.4f} | PromptTokens: {PromptTokens.instance().value} "
         f"| CompletionTokens: {CompletionTokens.instance().value}")


if __name__ == "__main__":
    main()
