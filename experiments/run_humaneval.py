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
from RAPS.core import RAPSCoordinator
from RAPS.config import (BENCHMARKS, NAIVE_POOL, TOP_K, add_mechanism_flags,
                         add_pool_flag, mechanism_overrides, pool_spec,
                         protocol_config)
from RAPS.agents.code_writing import CodeWriting
from RAPS.prompt.humaneval_prompt_set import HumanEvalPromptSet
from RAPS.tools.coding.python_executor import PyExecutor


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str):
    print(f"[{_ts()}] {message}")


#: The crafted pool of Table S.2, taken from the one place it is declared.
ROLES = BENCHMARKS["humaneval"].roles


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


def _subscription(role: str, naive_pool: bool) -> str:
    """The profile a host declares: the naive pool's generic instruction, or the role's
    domain description from the benchmark's prompt set."""
    return NAIVE_POOL[role] if naive_pool else HumanEvalPromptSet().get_description(role)


def _build_agent(role: str, llm_name: str, additional_instructions: str = "",
                 naive_pool: bool = False):
    agent = AgentRegistry.get(
        BENCHMARKS["humaneval"].agent_class, id=None, llm_name=llm_name, domain="humaneval",
        role=role, capabilities=_subscription(role, naive_pool), interests="",
        additional_instructions=additional_instructions, few_shot="",
    )
    agent.history = []
    agent.inbox = []
    return agent


def initialize_agents_from_set(llm_name: str, naive_pool: bool = False):
    """One CodeWriting agent per worker role of the HumanEval pool (Table S.2): the final
    answerer is one of the five, so it is excluded here and built separately."""
    spec = pool_spec("humaneval", naive_pool)
    return [_build_agent(role, llm_name, naive_pool=naive_pool)
            for role in spec.roles if role != spec.final_answerer]


def initialize_final_answerer(llm_name: str, naive_pool: bool = False):
    """The pool role that composes the final answer (Table S.2), carrying the decision
    constraint on top of its own subscription rather than replacing it."""
    agent = _build_agent(pool_spec("humaneval", naive_pool).final_answerer, llm_name,
                         additional_instructions=HumanEvalPromptSet().get_decision_constraint(),
                         naive_pool=naive_pool)
    agent.is_final_answerer = True   # keep the format constraint through refinement
    return agent


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
    parser.add_argument("--max_steps", type=int, default=None, help="default: paper value")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--sim_threshold", type=float, default=None)
    parser.add_argument("--entry_index", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=164)
    add_mechanism_flags(parser)
    add_pool_flag(parser)
    parser.add_argument("--dynamic_recruit", action="store_true")
    parser.add_argument("--adaptive_capacity", action="store_true")
    parser.add_argument("--max_team_size", type=int, default=10)
    parser.add_argument("--max_top_k", type=int, default=TOP_K,
                        help="fan-out ceiling under adaptive capacity, at the protocol cap")
    parser.add_argument("--budget_tokens", type=int, default=None,
                        help="per-task token cap on the coordination loop (default: uncapped)")
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

    agents = initialize_agents_from_set(args.llm_name, args.naive_pool)
    final_answerer = initialize_final_answerer(args.llm_name, args.naive_pool)
    overrides = dict(
        entry_index=args.entry_index,
        dynamic_recruit=args.dynamic_recruit, adaptive_capacity=args.adaptive_capacity,
        max_team_size=args.max_team_size, max_top_k=args.max_top_k,
        **mechanism_overrides(args),
        code_verify=args.code_verify, code_verify_max_iters=args.code_verify_max_iters,
        edge_cases=args.edge_cases,
    )
    for flag in ("max_steps", "top_k", "sim_threshold", "budget_tokens"):
        if getattr(args, flag) is not None:
            overrides[flag] = getattr(args, flag)
    config = protocol_config(args.domain, **overrides)
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
