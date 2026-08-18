"""AFlow: Monte Carlo search over code-represented workflows composed from the
operator set, run with a declared iteration budget on a held-out slice. A workflow is
a python snippet defining `run(task, ops, P)` — ops are the operators, P the custom
prompts of that workflow — and each iteration expands one parent workflow into a child
through a single optimizer call. The workflow scoring best on the held-out slice is
the one evaluated on the benchmark."""
import math
import random
import re
from types import SimpleNamespace

from baselines.common import get_llm
from baselines.aflow_operators import Operators

SEARCH_ROUNDS = 20   # optimizer iterations the declared budget buys
TOP_K = 4            # selection candidates; the initial workflow always stays first
LAMBDA = 0.3         # uniform mixture weight of the selection probability
ALPHA = 0.2          # softmax temperature of the selection probability (score x100)

INITIAL_GRAPH = 'def run(task, ops, P):\n    return ops.custom(task, "")\n'

OPERATOR_DESCRIPTION = """\
1. custom(input, instruction=""): one backbone call of instruction + input.
2. answer_generate(input): step-by-step thought plus a concise final answer field.
3. custom_code_generate(problem, instruction=""): a python implementation of the task's entry point (code tasks).
4. sc_ensemble(solutions, problem): one consistency vote over candidate solutions.
5. programmer(problem, analysis="None"): write-and-run a solve() function with error feedback (math tasks)."""

EXPAND_PROMPT = """\
You are building a workflow and corresponding prompts to jointly solve {kind} problems.
A workflow is a python snippet defining `run(task, ops, P)`, where `ops` provides the
operators below and `P` holds this workflow's custom prompts as attributes (P.XXX_PROMPT).

Referring to the given workflow and prompts, which form a basic example of a solution
approach, reconstruct and optimize them. You can add, modify, or delete operators,
parameters, or prompts. You can incorporate critical thinking methods like review,
revise, ensemble (generating multiple answers through different or similar prompts,
then voting or integrating the majority to obtain a final answer), selfAsk, etc.
Consider python's loops (for, while, list comprehensions) and conditional statements.
The workflow complexity should not exceed 10 operator calls.

<sample>
    <experience>{experience}</experience>
    <score>{score}</score>
    <graph>{graph}</graph>
    <prompt>{prompt}</prompt>
    <operator_description>{operator_description}</operator_description>
</sample>
Below are tasks the parent workflow answered incorrectly, as references for optimization:
{log}

Only one detail point can be modified at a time, and no more than 5 lines of code may
be changed per modification. Ensure that every prompt the workflow reads from P is
defined in the <prompt> field as `XXX_PROMPT = "..."` assignments, that generated
prompts contain no placeholders, and that `run` returns the final answer string.
Reply with exactly three XML fields:
<modification>the single difference from the parent</modification>
<graph>the full run(task, ops, P) function</graph>
<prompt>the prompt assignments, or empty</prompt>"""


def _xml_field(text: str, name: str) -> str:
    m = re.findall(rf"<{name}>(.*?)</{name}>", text or "", re.DOTALL)
    return m[-1].strip() if m else ""


class AFlow:
    def __init__(self, bench, llm_name, dev_records=None, search_budget=None, seed=0):
        self.bench = bench
        self.llm_name = llm_name
        self.optimizer = get_llm(llm_name)
        self.rng = random.Random(seed)
        budget = SEARCH_ROUNDS if search_budget is None else search_budget
        self.best = {"graph": INITIAL_GRAPH, "prompt": ""}
        if budget:
            if not dev_records:
                raise ValueError("aflow --search_budget needs held-out dev_records")
            self.best = self._optimize(dev_records, budget)
        self._call = self._compile(self.best["graph"], self.best["prompt"])

    # ------------------------------------------------------------- workflow exec

    def _compile(self, graph_code: str, prompt_code: str):
        """The workflow as a callable over records, or None when it does not load."""
        pns, gns = {}, {}
        try:
            exec(prompt_code or "", pns)
            exec(graph_code, gns)
            run = gns["run"]
        except Exception:
            return None
        prompts = SimpleNamespace(**{k: v for k, v in pns.items()
                                     if k.isupper() and isinstance(v, str)})

        def call(rec):
            ops = Operators(self.llm_name, rec.get("entry_point", ""))
            return str(run(rec["task"], ops, prompts))
        return call

    # ------------------------------------------------------------------ search

    def _validate(self, graph: str, prompt: str, records):
        """Mean accuracy over the held-out slice, plus a few failing tasks as the
        expansion log; None when the workflow does not load."""
        call = self._compile(graph, prompt)
        if call is None:
            return None
        solved, wrong = 0, []
        for rec in records:
            try:
                out = call(rec)
            except Exception:
                out = ""
            ok = bool(self.bench.score(self.bench.extract(out), rec))
            solved += ok
            if not ok:
                wrong.append(rec["task"][:200])
        return {"score": solved / max(1, len(records)), "wrong": wrong[:3]}

    def _select_parent(self, nodes):
        """One of the top-4 nodes — the initial workflow always stays a candidate —
        sampled from a uniform/softmax mixture over the validation scores."""
        ranked = sorted(nodes, key=lambda n: -n["score"])
        top = [nodes[0]] + [n for n in ranked if n is not nodes[0]][:TOP_K - 1]
        scores = [n["score"] * 100 for n in top]
        exps = [math.exp(ALPHA * (s - max(scores))) for s in scores]
        weights = [LAMBDA / len(top) + (1 - LAMBDA) * e / sum(exps) for e in exps]
        r = self.rng.random() * sum(weights)
        for node, w in zip(top, weights):
            r -= w
            if r <= 0:
                return node
        return top[-1]

    def _expand(self, parent, nodes):
        """One optimizer call producing a child workflow; its modification must differ
        from everything already tried from this parent."""
        experience = "\n".join(
            f"- parent score {n['parent_score']:.2f} -> {n['score']:.2f}: {n['modification']}"
            for n in nodes if n.get("parent") is parent) or "none"
        prompt = EXPAND_PROMPT.format(
            kind=self.bench.kind, experience=experience, score=f"{parent['score']:.2f}",
            graph=parent["graph"], prompt=parent["prompt"] or "(empty)",
            operator_description=OPERATOR_DESCRIPTION,
            log="\n".join(parent.get("wrong", [])) or "none")
        for _ in range(3):
            out = self.optimizer.gen(prompt)
            modification, graph = _xml_field(out, "modification"), _xml_field(out, "graph")
            if graph and "def run(" in graph and modification not in parent["tried"]:
                return {"graph": graph, "prompt": _xml_field(out, "prompt"),
                        "modification": modification, "parent": parent,
                        "parent_score": parent["score"], "tried": []}
        return None

    def _optimize(self, dev_records, budget: int) -> dict:
        initial = (self._validate(INITIAL_GRAPH, "", dev_records)
                   or {"score": 0.0, "wrong": []})
        nodes = [{"graph": INITIAL_GRAPH, "prompt": "", "modification": "initial",
                  "parent": None, "parent_score": 0.0, "tried": [], **initial}]
        best = nodes[0]
        for _ in range(budget):
            child = self._expand(self._select_parent(nodes), nodes)
            if child is None:
                continue
            child["parent"]["tried"].append(child["modification"])
            result = self._validate(child["graph"], child["prompt"], dev_records)
            if result is None:
                continue
            child.update(result)
            nodes.append(child)
            if child["score"] > best["score"]:
                best = child
        return best

    def solve(self, record: dict) -> str:
        return self._call(record)


def build(bench, llm_name, pool=None, search_budget=None, dev_records=None,
          seed=0, **opts) -> AFlow:
    return AFlow(bench, llm_name, dev_records, search_budget, seed)
