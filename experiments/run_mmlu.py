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
from RAPS.agents.analyze_agent import AnalyzeAgent
from RAPS.prompt.mmlu_prompt_set import MMLUPromptSet
from raps_data.mmlu_dataset import MMLUDataset


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str):
    print(f"[{_ts()}] {message}")


ROLES = ["Knowledge Expert", "Critic", "Mathematician", "Psychologist",
         "Historian", "Doctor", "Lawyer", "Economist", "Programmer"]


# Knowledge-intensive roles whose answers hinge on external facts (statutes, dates,
# definitions, named results, empirical figures) get ReAct-style autonomous Wikipedia
# retrieval. Pure-reasoning roles (Critic, Mathematician, Programmer) are left unchanged.
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


def initialize_agents_from_set(llm_name: str):
    agents = []
    AgentClass = AgentRegistry.get_class("AnalyzeAgent")
    prompt_set = MMLUPromptSet()
    for role in ROLES:
        description = prompt_set.get_description(role)
        is_knowledge = role in KNOWLEDGE_ROLES
        if role == "Knowledge Expert":
            combined_prompt = f"{description}\n"
        elif is_knowledge:
            # knowledge roles need room to reason over retrieved facts -> relaxed length
            combined_prompt = (f"{description}\nKeep your reply concise (under ~150 words) "
                               f"with a brief step by step analysis of the question.\n")
        else:
            combined_prompt = (f"{description}\nYour reply must be less than 100 words and "
                               f"include a brief step by step analysis of the question.\n")
        if is_knowledge:
            combined_prompt += REACT_RETRIEVAL_INSTRUCTION
        agent = AgentClass(
            id=None, llm_name=llm_name, role=role,
            capabilities=combined_prompt, interests="", additional_instructions="",
        )
        if is_knowledge:
            agent.react_retrieve = True
        agent.history = []
        agent.inbox = []
        agents.append(agent)
    return agents


def initialize_final_answerer(llm_name: str):
    AgentClass = AgentRegistry.get_class("AnalyzeAgent")
    return AgentClass(
        id=None, llm_name=llm_name, role="Final Answerer",
        capabilities=MMLUPromptSet.get_decision_role(),
        interests="", additional_instructions=MMLUPromptSet.get_decision_constraint(),
    )


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
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--sim_threshold", type=float, default=0.40)
    parser.add_argument("--entry_index", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=159)
    parser.add_argument("--reputation_gate", action="store_true")
    parser.add_argument("--second_hand_gossip", action="store_true")
    parser.add_argument("--dynamic_recruit", action="store_true")
    parser.add_argument("--adaptive_capacity", action="store_true")
    parser.add_argument("--max_team_size", type=int, default=12)
    parser.add_argument("--max_top_k", type=int, default=2)
    parser.add_argument("--no_react", action="store_true",
                        help="disable ReAct retrieval for the knowledge roles (clean baseline)")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = MMLUDataset(split="test")
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    agents = initialize_agents_from_set(args.llm_name)
    if args.no_react:
        for a in agents:
            a.react_retrieve = False
    final_answerer = initialize_final_answerer(args.llm_name)
    config = RAPSConfig(
        domain=args.domain, max_steps=args.max_steps, top_k=args.top_k,
        sim_threshold=args.sim_threshold, entry_index=args.entry_index,
        reputation_gate=args.reputation_gate, second_hand_gossip=args.second_hand_gossip,
        dynamic_recruit=args.dynamic_recruit, adaptive_capacity=args.adaptive_capacity,
        max_team_size=args.max_team_size, max_top_k=args.max_top_k,
    )
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
    _log(f"Total Cost: ${Cost.instance().value:.4f}")


if __name__ == "__main__":
    main()
