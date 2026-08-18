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
from RAPS.core import RAPSCoordinator
from RAPS.config import (BENCHMARKS, NAIVE_POOL, NAIVE_TOOL_USER, TOP_K,
                         add_mechanism_flags, add_pool_flag, mechanism_overrides,
                         pool_spec, protocol_config)
from RAPS.agents.analyze_agent import AnalyzeAgent
from RAPS.prompt.mmlu_prompt_set import MMLUPromptSet
from raps_data.mmlu_dataset import MMLUDataset


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str):
    print(f"[{_ts()}] {message}")


#: The crafted pool of Table S.2, taken from the one place it is declared.
ROLES = BENCHMARKS["mmlu"].roles

# Knowledge-intensive roles whose answers hinge on external facts (statutes, dates,
# definitions, named results, empirical figures) get ReAct-style autonomous Wikipedia
# retrieval. Pure-reasoning roles (Mathematician, Programmer) are left unchanged. Names
# beyond the pool are listed so the additional profiles of the prompt set keep their
# retrieval behaviour when a larger population draws on them.
KNOWLEDGE_ROLES = {"Knowledge Expert", "Historian", "Doctor", "Lawyer", "Economist", "Psychologist"}

REACT_RETRIEVAL_INSTRUCTION = (
    "\n\nYou have access to a Wikipedia fact-retrieval tool. IMPORTANT: multiple-choice "
    "distractors are deliberately crafted to look correct, and confident recall is often "
    "wrong on specifics. Therefore, whenever the correct answer hinges on a precise external "
    "fact — a specific statute, legal rule or holding, a definition, a date, a named "
    "theorem/result, or an empirical figure — do NOT rely on memory alone: VERIFY the "
    "decisive fact first, even if you believe you know it. To retrieve, output a SINGLE line "
    "in EXACTLY this format and nothing else on that line:\n"
    "SEARCH: <discriminative keywords>\n"
    "Write the query as KEYWORDS for a keyword search engine, NOT a natural-language sentence: "
    "include only the specific, rare, identifying terms — proper names, technical/domain terms, "
    "statute or case names, the exact concept — and DROP generic filler like 'definition of', "
    "'what is', 'in international law', 'the following'. For example, instead of "
    "\"Definition of 'injured State' in international law\" write "
    "\"injured State responsibility breach obligation specially affected\".\n"
    "The retrieved facts are returned to you as an 'Observation'; then continue your analysis "
    "grounded in them. Answer directly only for questions of pure reasoning or ones you can "
    "derive from first principles. Use at most two searches, then give a brief step-by-step "
    "analysis and your answer.\n"
)


def _retrieves(role: str, naive_pool: bool) -> bool:
    """Whether the retrieval interface is granted to this profile: the crafted
    knowledge roles, or the one naive profile defined by acting on tools."""
    return role == NAIVE_TOOL_USER if naive_pool else role in KNOWLEDGE_ROLES


def _subscription(role: str, naive_pool: bool) -> str:
    """The profile a host declares. A naive profile is a complete instruction in itself, so
    it is taken verbatim; a crafted one carries the reply-length requirement of its role.
    Either way the profile granted retrieval also carries the instruction for using it."""
    if naive_pool:
        prompt = NAIVE_POOL[role]
    else:
        description = MMLUPromptSet().get_description(role)
        if role == "Knowledge Expert":
            prompt = f"{description}\n"
        elif role in KNOWLEDGE_ROLES:
            # knowledge roles need room to reason over retrieved facts -> relaxed length
            prompt = (f"{description}\nKeep your reply concise (under ~150 words) "
                      f"with a brief step by step analysis of the question.\n")
        else:
            prompt = (f"{description}\nYour reply must be less than 100 words and "
                      f"include a brief step by step analysis of the question.\n")
    return prompt + REACT_RETRIEVAL_INSTRUCTION if _retrieves(role, naive_pool) else prompt


def _build_agent(role: str, llm_name: str, additional_instructions: str = "",
                 naive_pool: bool = False):
    agent = AgentRegistry.get(
        BENCHMARKS["mmlu"].agent_class, id=None, llm_name=llm_name, domain="mmlu", role=role,
        capabilities=_subscription(role, naive_pool), interests="",
        additional_instructions=additional_instructions,
    )
    if _retrieves(role, naive_pool):
        agent.react_retrieve = True
    agent.history = []
    agent.inbox = []
    return agent


def initialize_agents_from_set(llm_name: str, naive_pool: bool = False):
    """One AnalyzeAgent per worker role of the MMLU pool (Table S.2): the final answerer
    is one of the five, so it is excluded here and built separately."""
    spec = pool_spec("mmlu", naive_pool)
    return [_build_agent(role, llm_name, naive_pool=naive_pool)
            for role in spec.roles if role != spec.final_answerer]


def initialize_final_answerer(llm_name: str, naive_pool: bool = False):
    """The pool role that composes the final answer (Table S.2), carrying the decision
    constraint on top of its own subscription rather than replacing it."""
    agent = _build_agent(pool_spec("mmlu", naive_pool).final_answerer, llm_name,
                         additional_instructions=MMLUPromptSet.get_decision_constraint(),
                         naive_pool=naive_pool)
    agent.is_final_answerer = True   # keep the format constraint through refinement
    return agent


def _safe_slug(text, n=10):
    """Filesystem-safe slug (avoids '/' etc. in task text breaking the path)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)[:n]).strip("_") or "task"


def write_task_log(task_log, answer, index=None):
    task_log["answer"] = answer
    if index is not None:
        task_log["index"] = index
    log_dir = RAPS_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    task = task_log.get("task", "task")
    # include the question index so questions sharing a 10-char prefix (very common in
    # MMLU: "Which of the...", "This question...") no longer overwrite each other.
    idx = "" if index is None else f"{index:05d}_"
    log_file = log_dir / f"log_mmlu_{Time.instance().value}_{idx}{_safe_slug(task)}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(task_log, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="RAPS Experiments on MMLU")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--domain", type=str, default="mmlu")
    parser.add_argument("--max_steps", type=int, default=None, help="default: paper value")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--sim_threshold", type=float, default=None)
    parser.add_argument("--entry_index", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=153)
    add_mechanism_flags(parser)
    add_pool_flag(parser)
    parser.add_argument("--dynamic_recruit", action="store_true")
    parser.add_argument("--adaptive_capacity", action="store_true")
    parser.add_argument("--max_team_size", type=int, default=12)
    parser.add_argument("--max_top_k", type=int, default=TOP_K,
                        help="fan-out ceiling under adaptive capacity, at the protocol cap")
    parser.add_argument("--budget_tokens", type=int, default=None,
                        help="per-task token cap on the coordination loop (default: uncapped)")
    parser.add_argument("--no_react", action="store_true",
                        help="disable ReAct retrieval for the knowledge roles (clean baseline)")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = MMLUDataset(split="test")
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    agents = initialize_agents_from_set(args.llm_name, args.naive_pool)
    if args.no_react:
        for a in agents:
            a.react_retrieve = False
    final_answerer = initialize_final_answerer(args.llm_name, args.naive_pool)
    overrides = dict(
        entry_index=args.entry_index,
        dynamic_recruit=args.dynamic_recruit, adaptive_capacity=args.adaptive_capacity,
        max_team_size=args.max_team_size, max_top_k=args.max_top_k,
        **mechanism_overrides(args),
    )
    for flag in ("max_steps", "top_k", "sim_threshold", "budget_tokens"):
        if getattr(args, flag) is not None:
            overrides[flag] = getattr(args, flag)
    config = protocol_config(args.domain, **overrides)
    coordinator = RAPSCoordinator(agents, final_answerer, config,
                                  logger=_log, answer_extractor=dataset.postprocess_answer)

    all_results = []
    total_solved = total_executed = 0

    end = min(args.start + args.limit, len(dataset))
    for i in range(args.start, end):
        record = dataset[i]
        _log(f"================ Question {i} ================")
        task = MMLUDataset.record_to_input(record)["task"]
        answer = MMLUDataset.record_to_target_answer(record)

        result = coordinator.run(task)
        write_task_log(result.task_log, answer, index=i)

        pred = dataset.postprocess_answer(result.final_output)
        solved = (pred == answer)
        total_solved += 1 if solved else 0
        total_executed += 1
        accuracy = total_solved / total_executed

        all_results.append({
            "answer": answer, "prediction": pred, "correct": solved,
            "total solved": total_solved, "total executed": total_executed, "accuracy": accuracy,
        })
        _log(f"Predicted: {pred} | Answer: {answer} | Correct: {solved}")
        _log(f"Current Accuracy: {accuracy:.4f} ({total_solved}/{total_executed})")

    result_dir = Path(f"{RAPS_ROOT}/result/mmlu")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"mmlu_{args.llm_name}_{current_time}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    _log("====== Final Stats ======")
    _log(f"Accuracy: {total_solved / total_executed if total_executed else 0:.4f}")
    _log(f"Total Solved: {total_solved} / {total_executed}")
    _log(f"Total Cost: ${Cost.instance().value:.4f} | PromptTokens: {PromptTokens.instance().value} "
         f"| CompletionTokens: {CompletionTokens.instance().value}")


if __name__ == "__main__":
    main()
