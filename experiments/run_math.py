"""Unified RAPS runner for the three mathematical benchmarks (Table S.2/S.3).

One entry point for GSM8K, SVAMP and AQuA: they share the five-agent math pool
(Math Solver / Mathematical Analyst / Programming Expert / Inspector /
Summarizer), the Summarizer as final answerer, and the single protocol
parameter set of RAPS/config.py (Table S.4). Only the dataset loader and the
answer extractor differ — numeric for GSM8K/SVAMP, option letter for AQuA.

Backbones are assigned per agent: --llm_name sets the pool default and
--role_model 'Role=model' overrides individual roles, so one run can mix
backbones across the pool (Fig. 4d-e). --budget_tokens caps the tokens spent
inside the coordination loop of each task (the final decision always runs).

Example:
    python experiments/run_math.py --dataset svamp --limit 100
    python experiments/run_math.py --dataset aqua --no_reputation_gate   (ablation override)
    python experiments/run_math.py --dataset gsm8k --llm_name gpt-4o-mini-2024-07-18 \
        --role_model "Math Solver=claude-sonnet-5" --role_model "Inspector=Qwen/Qwen3-32B"
    python experiments/run_math.py --dataset gsm8k --budget_tokens 20000
"""
import sys
import os
import re
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from RAPS.utils.const import RAPS_ROOT
from RAPS.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from RAPS.agents.agent_registry import AgentRegistry
from RAPS.core import RAPSCoordinator
from RAPS.config import (BENCHMARKS, NAIVE_POOL, TOP_K, add_mechanism_flags,
                         add_pool_flag, mechanism_overrides, pool_spec,
                         protocol_config)
from RAPS.agents.math_solver import MathAgent
from RAPS.prompt.gsm8k_prompt_set import GSM8KPromptSet, FEW_SHOT_DATA
from raps_data.gsm8k_dataset import gsm_data_process, gsm_get_predict
from raps_data.svamp_dataset import load_svamp
from raps_data.aqua_dataset import load_aqua, aqua_get_predict


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str):
    print(f"[{_ts()}] {message}")


# AQuA answers are option letters; the final answerer is instructed accordingly.
_AQUA_DECISION_CONSTRAINT = (
    "You will be given a multiple-choice math problem and the analyses of other agents. "
    "Please find the most reliable option based on their analyses. "
    "Give reasons for making decisions. "
    "The last line of your output must be exactly: The answer is X, where X is one of A, B, C, D, E."
)

LOADERS = {
    "gsm8k": lambda path: gsm_data_process(_load_jsonl(path)),
    "svamp": lambda path: load_svamp(path) if path else load_svamp(),
    "aqua": lambda path: load_aqua(path) if path else load_aqua(),
}

EXTRACTORS = {"gsm8k": gsm_get_predict, "svamp": gsm_get_predict, "aqua": aqua_get_predict}

_DEFAULT_JSON = {"gsm8k": "raps_data/gsm8k/gsm8k.jsonl"}


def _load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_correct(pred, answer, dataset) -> bool:
    if dataset == "aqua":
        return str(pred).strip().upper() == str(answer).strip().upper()
    try:
        return abs(float(pred) - float(answer)) < 1e-6
    except (ValueError, TypeError):
        return str(pred).strip() == str(answer).strip()


def _subscription(role: str, naive_pool: bool) -> str:
    """The profile a host declares: the naive pool's generic instruction, or the role's
    domain description from the benchmark's prompt set."""
    return NAIVE_POOL[role] if naive_pool else GSM8KPromptSet().get_description(role)


def initialize_agents_from_set(llm_name: str, domain: str,
                               role_models: Optional[Dict[str, str]] = None,
                               naive_pool: bool = False):
    """One MathAgent per worker role of the benchmark's five-agent pool
    (Table S.2): the final answerer is one of the five, not an additional
    aggregator, so it is excluded here and built separately. `role_models`
    overrides the backbone of individual roles for a heterogeneous pool
    (Fig. 4d-e); every agent without an entry uses `llm_name`."""
    spec = pool_spec(domain, naive_pool)
    role_models = role_models or {}
    agents = []
    for role in spec.roles:
        if role == spec.final_answerer:
            continue
        agent = AgentRegistry.get(
            spec.agent_class, id=None, llm_name=role_models.get(role, llm_name), domain=domain, role=role,
            capabilities=_subscription(role, naive_pool), interests="",
            additional_instructions="", few_shot=FEW_SHOT_DATA.get(role, ""),
        )
        agent.history = []
        agent.inbox = []
        agents.append(agent)
    return agents


def initialize_final_answerer(llm_name: str, domain: str,
                              role_models: Optional[Dict[str, str]] = None,
                              naive_pool: bool = False):
    """The pool role that composes the final answer, carrying the answer-format constraint
    the harness applies to every method whichever pool is in use."""
    spec = pool_spec(domain, naive_pool)
    role_models = role_models or {}
    constraint = _AQUA_DECISION_CONSTRAINT if domain == "aqua" else GSM8KPromptSet.get_decision_constraint()
    capabilities = (NAIVE_POOL[spec.final_answerer] if naive_pool
                    else GSM8KPromptSet.get_decision_role())
    agent = AgentRegistry.get(
        spec.agent_class, id=None, llm_name=role_models.get(spec.final_answerer, llm_name), domain=domain,
        role=spec.final_answerer,
        capabilities=capabilities, interests="",
        additional_instructions=constraint,
        few_shot="" if domain == "aqua" else GSM8KPromptSet.get_decision_few_shot(),
    )
    agent.is_final_answerer = True   # keep the format constraint through refinement
    return agent


def _safe_slug(text, n=20):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)[:n]).strip("_") or "task"


def write_task_log(task_log, answer, domain):
    task_log["answer"] = answer
    log_dir = RAPS_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"log_{domain}_{Time.instance().value}_{_safe_slug(task_log.get('task', 'task'))}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(task_log, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="RAPS on the mathematical benchmarks")
    parser.add_argument("--dataset", choices=["gsm8k", "svamp", "aqua"], required=True)
    parser.add_argument("--dataset_json", type=str, default=None,
                        help="override the default data path of the dataset")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--max_steps", type=int, default=None, help="default: paper value (5)")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--sim_threshold", type=float, default=None)
    parser.add_argument("--entry_index", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10**9,
                        help="questions to evaluate, defaulting to the whole set")
    add_mechanism_flags(parser)
    add_pool_flag(parser)
    parser.add_argument("--dynamic_recruit", action="store_true")
    parser.add_argument("--adaptive_capacity", action="store_true")
    parser.add_argument("--max_team_size", type=int, default=8)
    parser.add_argument("--max_top_k", type=int, default=TOP_K)
    parser.add_argument("--role_model", action="append", default=None, metavar="'Role=model'",
                        help="heterogeneous pool (Fig. 4d-e): backbone of one role; repeatable. "
                             "Every role without an entry uses --llm_name")
    parser.add_argument("--budget_tokens", type=int, default=None,
                        help="per-task token cap on the coordination loop (default: uncapped)")
    parser.add_argument("--tool_verify", action="store_true",
                        help="program-aided arithmetic: compute the final answer by executing "
                             "Python rather than by generating it")
    return parser.parse_args()


def parse_role_models(pairs, domain: str) -> Dict[str, str]:
    """'Role=model' pairs -> {role: model}, each role checked against the benchmark pool
    so a misspelled role fails here instead of silently leaving the pool homogeneous."""
    roles = set(BENCHMARKS[domain].roles)
    out = {}
    for p in pairs or []:
        role, sep, model = p.partition("=")
        role, model = role.strip(), model.strip()
        if not sep or not role or not model:
            raise ValueError(f"--role_model expects 'Role=model', got {p!r}")
        if role not in roles:
            raise ValueError(f"--role_model role {role!r} is not in the {domain} pool "
                             f"{sorted(roles)}")
        out[role] = model
    return out


def main():
    args = parse_args()
    domain = args.dataset
    path = args.dataset_json or _DEFAULT_JSON.get(domain)
    dataset = LOADERS[domain](path)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{RAPS_ROOT}/result/{domain}")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"{domain}_{args.llm_name}_{current_time}.json"

    role_models = parse_role_models(args.role_model, domain)
    agents = initialize_agents_from_set(args.llm_name, domain, role_models, args.naive_pool)
    final_answerer = initialize_final_answerer(args.llm_name, domain, role_models,
                                               args.naive_pool)

    overrides = dict(
        dynamic_recruit=args.dynamic_recruit,
        adaptive_capacity=args.adaptive_capacity,
        max_team_size=args.max_team_size,
        max_top_k=args.max_top_k,
        entry_index=args.entry_index,
        tool_verify=args.tool_verify,
        **mechanism_overrides(args),
    )
    for flag in ("max_steps", "top_k", "sim_threshold", "budget_tokens"):
        if getattr(args, flag) is not None:
            overrides[flag] = getattr(args, flag)
    config = protocol_config(domain, **overrides)

    coordinator = RAPSCoordinator(agents, final_answerer, config,
                                  logger=_log, answer_extractor=EXTRACTORS[domain])

    test_dataset = dataset[args.start:args.start + args.limit]
    all_results = []
    total_solved = total_executed = 0

    for i, example in enumerate(test_dataset):
        _log(f"================ Question {args.start + i} ================")
        task, answer = example["task"], example.get("answer")
        _log(f"Task: {task}")

        result = coordinator.run(task)
        write_task_log(result.task_log, answer, domain)

        pred = EXTRACTORS[domain](result.final_output)
        solved = is_correct(pred, answer, domain)
        total_solved += 1 if solved else 0
        total_executed += 1
        accuracy = total_solved / total_executed

        all_results.append({
            "task": task, "answer": answer, "response": result.final_output,
            "prediction": pred, "correct": solved,
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
