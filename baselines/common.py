"""Shared harness of the compared methods (Table 1) and their robustness rows (Table 2).

A compared method is built as `build(bench, llm_name, pool, **opts)` and returns a
solver with `solve(record) -> final output`. Everything a fair comparison must hold
constant lives here exactly once: benchmark loading and slicing, answer extraction
and scoring, the output-format instruction, the shared backends, and the five-role
pool (Table S.2) that every peer-based method coordinates — the same pieces the RAPS
runners use, so a baseline row and a RAPS row differ only in the coordination
protocol.
"""
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from RAPS.utils.const import RAPS_ROOT
from RAPS.utils.globals import Time
from RAPS.llm.llm_registry import LLMRegistry
from RAPS.config import BENCHMARKS as POOLS
from RAPS.prompt.gsm8k_prompt_set import GSM8KPromptSet
from RAPS.prompt.mmlu_prompt_set import MMLUPromptSet
from RAPS.prompt.humaneval_prompt_set import HumanEvalPromptSet
from RAPS.tools.coding.python_executor import PyExecutor

from raps_data.gsm8k_dataset import gsm_data_process, gsm_get_predict, load_gsm8k_jsonl
from raps_data.svamp_dataset import load_svamp
from raps_data.aqua_dataset import load_aqua, aqua_get_predict
from raps_data.mmlu_dataset import MMLUDataset


# ------------------------------------------------------------------ participants

_LLMS: Dict[str, object] = {}


def get_llm(llm_name: str):
    """One backend instance per backbone name, shared by every participant of a run."""
    if llm_name not in _LLMS:
        _LLMS[llm_name] = LLMRegistry.get(llm_name)
    return _LLMS[llm_name]


class Agent:
    """A participant of a compared method: the role profile it speaks from and the
    backbone it calls. Decoding follows the shared controls (Table S.1)."""

    def __init__(self, name: str, profile: str = "", llm_name: str = ""):
        self.name = name
        self.profile = profile
        self.llm = get_llm(llm_name)

    def respond(self, user: str, task: str = "") -> str:
        messages = ([{"role": "system", "content": self.profile}] if self.profile else [])
        messages.append({"role": "user", "content": user})
        return self.llm.gen(messages)


def inject_adversaries(agents: List[Agent], domain: str, llm_name: str,
                       adversary: Optional[dict] = None) -> List[Agent]:
    """Replace the last `count` members by adversarial participants wearing the same
    role profiles — the disguise of the shared threat model (Table 2)."""
    if not adversary:
        return agents
    from baselines.adversaries import AdversarialAgent   # late import: no module cycle
    count = min(adversary.get("count", 1), len(agents))
    for i in range(1, count + 1):
        a = agents[-i]
        agents[-i] = AdversarialAgent(a.name, a.profile, llm_name, domain,
                                      kind=adversary["kind"],
                                      defect_after=adversary.get("defect_after", 3))
    return agents


_PROMPT_SETS = {
    "gsm8k": GSM8KPromptSet, "svamp": GSM8KPromptSet, "aqua": GSM8KPromptSet,
    "mmlu": MMLUPromptSet, "humaneval": HumanEvalPromptSet,
}


def build_pool(domain: str, llm_name: str, adversary: Optional[dict] = None) -> List[Agent]:
    """The benchmark's five crafted roles (Table S.2) as participants, optionally with
    the trailing roles replaced by adversaries."""
    prompt_set = _PROMPT_SETS[domain]()
    agents = [Agent(role, prompt_set.get_description(role), llm_name)
              for role in POOLS[domain].roles]
    return inject_adversaries(agents, domain, llm_name, adversary)


# -------------------------------------------------------------------- benchmarks

#: The output-format instruction appended to every prompt a method issues, so the
#: shared extractors parse every method's output alike.
FORMAT_INSTRUCTIONS = {
    "number": ("The last line of your output contains only the final result without any "
               "units, for example: The answer is 140"),
    "letter4": ("The last line of your output must be exactly: The answer is X, "
                "where X is one of A, B, C, D."),
    "letter5": ("The last line of your output must be exactly: The answer is X, "
                "where X is one of A, B, C, D, E."),
    "code": ("Write your full implementation (restating the function signature) as a "
             "single ```python code block."),
}


def _number_equal(pred, answer) -> bool:
    try:
        return abs(float(pred) - float(answer)) < 1e-6
    except (TypeError, ValueError):
        return str(pred).strip() == str(answer).strip()


def _number_score(pred, rec) -> bool:
    return _number_equal(pred, rec["answer"])


def _letter_score(pred, rec) -> bool:
    return str(pred).strip().upper() == str(rec["answer"]).strip().upper()


def _code_score(pred, rec) -> bool:
    return PyExecutor().evaluate(rec["entry_point"], pred, rec["test"])


def extract_code(output) -> str:
    """The fenced python block of a final output, or the whole output when unfenced."""
    if not isinstance(output, str):
        return ""
    if "```python" in output:
        parts = output.split("```python", 1)[-1]
        return parts.split("```", 1)[0].strip() if "```" in parts else parts.strip()
    if "```" in output:
        parts = output.split("```", 1)[-1]
        return parts.split("```", 1)[0].strip() if "```" in parts else output.strip()
    return output.strip()


def _load_gsm8k(path, start, limit):
    path = path or RAPS_ROOT / "raps_data/gsm8k/gsm8k.jsonl"
    return gsm_data_process(load_gsm8k_jsonl(path))[start:start + limit]


def _load_svamp(path, start, limit):
    return load_svamp(path)[start:start + limit]


def _load_aqua(path, start, limit):
    return load_aqua(path)[start:start + limit]


def _mmlu_record(ds, i):
    return {"task": MMLUDataset.record_to_input(ds[i])["task"],
            "answer": MMLUDataset.record_to_target_answer(ds[i])}


def _load_mmlu(_path, start, limit):
    ds = MMLUDataset(split="test")
    return [_mmlu_record(ds, i) for i in range(start, min(start + limit, len(ds)))]


def _load_mmlu_dev(_path, n):
    """The MMLU validation split, the held-out slice a search budget is spent on."""
    ds = MMLUDataset(split="val")
    return [_mmlu_record(ds, i) for i in range(min(n, len(ds)))]


def _load_humaneval(path, start, limit):
    path = path or RAPS_ROOT / "raps_data/humaneval/humaneval-py.jsonl"
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [{"task": r["prompt"], "entry_point": r["entry_point"], "test": r["test"]}
            for r in records[start:start + limit]]


@dataclass(frozen=True)
class Benchmark:
    """A benchmark row: how records load, how a final output is extracted into a
    prediction, and how a prediction scores. `kind` picks the format instruction."""
    domain: str
    kind: str
    load: Callable[[Optional[str], int, int], List[dict]]
    extract: Callable[[str], str]
    score: Callable[[str, dict], bool]
    load_dev: Optional[Callable[[Optional[str], int], List[dict]]] = None

    @property
    def format(self) -> str:
        return FORMAT_INSTRUCTIONS[self.kind]


BENCHMARKS = {
    "gsm8k": Benchmark("gsm8k", "number", _load_gsm8k, gsm_get_predict, _number_score),
    "svamp": Benchmark("svamp", "number", _load_svamp, gsm_get_predict, _number_score),
    "aqua": Benchmark("aqua", "letter5", _load_aqua, aqua_get_predict, _letter_score),
    "mmlu": Benchmark("mmlu", "letter4", _load_mmlu, MMLUDataset.postprocess_answer,
                      _letter_score, load_dev=_load_mmlu_dev),
    "humaneval": Benchmark("humaneval", "code", _load_humaneval, extract_code, _code_score),
}


# -------------------------------------------------------------------- evaluation

def majority_vote(outputs: List[str], extract: Callable[[str], str]) -> str:
    """The output whose extracted prediction is the most frequent; ties keep the
    earliest such output, so the vote is deterministic."""
    if not outputs:
        return ""
    top = Counter(extract(o) for o in outputs).most_common(1)[0][0]
    return next(o for o in outputs if extract(o) == top)


def evaluate(solver, bench: Benchmark, records: List[dict], log=print) -> dict:
    """One pass over the records: solve, extract, score. Returns the per-task rows, the
    aggregate accuracy, and — when the pool holds adversarial participants — the
    attack volume the evaluation phase saw (Table 2)."""
    watched = [a for a in getattr(solver, "agents", []) if hasattr(a, "stats")]
    before = [dict(a.stats) for a in watched]
    rows, solved = [], 0
    for i, rec in enumerate(records):
        # per-task resilience: one failing task scores as wrong instead of killing the run
        try:
            output = solver.solve(rec)
        except Exception as e:
            output = ""
            log(f"[task {i}] solver error: {e}")
        pred = bench.extract(output)
        ok = bool(bench.score(pred, rec))
        solved += ok
        rows.append({"index": i, "prediction": pred, "correct": ok, "output": output})
        log(f"[task {i}] correct={ok} | running accuracy {solved / (i + 1):.4f}")
    summary = {"accuracy": solved / max(1, len(records)), "solved": solved,
               "n": len(records), "rows": rows}
    if watched:
        summary["attack"] = {
            "published": sum(a.stats["published"] - b["published"]
                             for a, b in zip(watched, before)),
            "faulty": sum(a.stats["faulty"] - b["faulty"]
                          for a, b in zip(watched, before)),
        }
    return summary


def save_result(method: str, dataset: str, llm_name: str, summary: dict) -> Path:
    """`result/<dataset>/<method>_<dataset>_<llm>_<time>.json`, next to the RAPS rows."""
    out_dir = Path(RAPS_ROOT) / "result" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{method}_{dataset}_{llm_name}_{Time.instance().value}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path
