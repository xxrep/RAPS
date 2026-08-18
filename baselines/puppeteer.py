"""Puppeteer: a central orchestrator sequences functional agents over the evolving task
state. Each decision picks one to three next agents; picking more than one splits the
path. A path ends at the terminator or the step cap, the path's final answers vote
into one candidate per path, and the candidates vote into the answer. The default
orchestrator is an LLM policy (--policy llm); the untrained policy is uniform-random
orchestration (--policy random). The web and file tool roles are omitted because their
services are absent; the python role degrades to plain reasoning on the same terms."""
import random
import re

from baselines.common import Agent, get_llm, inject_adversaries, majority_vote

MAX_STEPS = 5      # global rounds; also the per-path length cap
MAX_PATHS = 4      # parallel-path ceiling
MAX_PICK = 3       # agents one decision may activate

ROSTER = [
    ("TerminatorAgent", "terminate",
     "You are an expert in terminating processes. Your task is to determine when the reasoning process should be terminated and provide the final answer."),
    ("PlannerAgent", "planning",
     "You are an expert in planning. Your task is to create detailed plans for achieving specific goals."),
    ("ReasoningAgent", "reasoning",
     "You are an expert in logical reasoning. Your task is to reason through complex problems and provide well-thought-out solutions."),
    ("CriticAgent", "critique",
     "You are an expert in critiquing. Your task is to critique the reasoning and solutions provided by others."),
    ("ReflectAgent", "reflect",
     "You are an expert in reflection. Your task is to reflect on the reasoning process and provide insights for improvement."),
    ("QuestionAgent", "question",
     "You are an expert in questioning. Your task is to propose relevant sub-questions that help in solving the main problem."),
    ("SummarizerAgent", "summarize",
     "You are an expert in summarizing. Your task is to summarize the information and provide concise conclusions."),
    ("ConcluderAgent", "conclude",
     "You are an expert in concluding. Your task is to provide final conclusions based on the reasoning process."),
    ("Modifier", "modify",
     "You are an expert in error correction and modification. Your task is to identify errors in previous reasoning, explain why they are incorrect, and provide accurate corrections."),
    ("PythonAgent", "run_python",
     "You are an expert in Python programming. Your task is to run Python code and provide the results."),
]

INSTRUCTIONS = {
    "planning": "Decompose the question and plan the next steps to address the question.",
    "reasoning": "Continue the reasoning to get closer to the correct answer.",
    "critique": "Critique the previous reasoning for plausibility and correctness.",
    "reflect": "Diagnose the potential cause of failure in the previous reasoning and "
               "outline a concise high-level plan to avoid it.",
    "question": "Propose the next sub-question that logically follows from the previous "
                "reasoning, and answer it.",
    "summarize": "Summarize the previous results and provide intermediate conclusions.",
    "conclude": "Conclude the task and provide a final answer.",
    "modify": "Identify and correct errors in the previous reasoning: state what was "
              "incorrect, why, and the correct understanding.",
    "run_python": "Write standard-library Python code that computes the answer and "
                  "report the result it prints.",
    "terminate": "Terminate the reasoning process and provide the final answer.",
}


class _Path:
    """One reasoning path: the agents that acted, the accumulated reasoning, and the
    final-answer candidates collected along it."""

    def __init__(self, task: str):
        self.task = task
        self.sequence = []
        self.history = ""
        self.answers = []
        self.current = None

    def fork(self):
        """A split path sharing the reasoning so far, continued by another agent."""
        p = _Path(self.task)
        p.sequence, p.history, p.answers = list(self.sequence), self.history, list(self.answers)
        return p


class Puppeteer:
    def __init__(self, bench, agents, llm_name, policy="llm", seed=0):
        self.bench = bench
        self.agents = agents
        self.llm = get_llm(llm_name)
        self.policy = policy
        self.rng = random.Random(seed)

    def _pick(self, task: str, acted):
        """The orchestrator's decision: one to three agents to activate next."""
        if self.policy == "random":
            return self.rng.sample(self.agents, min(MAX_PICK, len(self.agents)))
        roster = "\n".join(f"- {a.name}: {a.profile}" for a in self.agents)
        out = self.llm.gen(
            "You are orchestrating a team of agents to solve a task. Based on the current "
            f"state, choose up to {MAX_PICK} agents to act next.\n"
            f"Task: {task}\nAgents:\n{roster}\n"
            f"Already acted: {', '.join(acted) or 'none'}\n"
            "Reply with only the chosen agent names, one per line.")
        chosen = []
        for line in out.splitlines():
            for a in self.agents:
                if a.name.lower() in line.lower() and a not in chosen:
                    chosen.append(a)
        return chosen[:MAX_PICK] or self.rng.sample(self.agents, 1)

    def _act(self, path: _Path) -> None:
        agent = path.current
        out = agent.respond(
            f"Now your question is: {path.task}\n{INSTRUCTIONS[agent.action]}\n"
            f"Finish your answer with the following template: "
            f"FINAL ANSWER: [YOUR FINAL ANSWER].\n"
            f"*Your previous reasoning was: {path.history or 'None'}.*\n"
            f"{self.bench.format}", path.task)
        path.sequence.append(agent.name)
        path.history += f"\nSuccessful Action: {agent.action}\nResult: {out}\n"
        m = re.search(r"FINAL ANSWER:\s*([\s\S]*)", out)
        if m and m.group(1).strip():
            path.answers.append(m.group(1).strip())

    def _path_answer(self, path: _Path) -> str:
        return majority_vote(path.answers, self.bench.extract) if path.answers else ""

    def solve(self, record: dict) -> str:
        task = record["task"]
        active, finished = [], []
        for a in self._pick(task, []):
            path = _Path(task)
            path.current = a
            active.append(path)
        for _ in range(MAX_STEPS):
            if not active:
                break
            for path in list(active):
                self._act(path)
                if path.current.name == "TerminatorAgent" or len(path.sequence) >= MAX_STEPS:
                    active.remove(path)
                    finished.append(path)
                    continue
                picks = self._pick(task, path.sequence)
                path.current = picks[0]
                for extra in picks[1:]:
                    if len(active) + len(finished) < MAX_PATHS:
                        fork = path.fork()
                        fork.current = extra
                        active.append(fork)
        candidates = [c for c in (self._path_answer(p) for p in finished + active) if c]
        if candidates:
            return majority_vote(candidates, self.bench.extract)
        # no path produced a candidate: conclude over the longest reasoning so far
        history = max((p.history for p in finished + active), key=len, default="")
        concluder = next(a for a in self.agents if a.action == "conclude")
        return concluder.respond(
            f"Now your question is: {task}\n{INSTRUCTIONS['conclude']}\n"
            f"*Your previous reasoning was: {history or 'None'}.*\n{self.bench.format}", task)


def build(bench, llm_name, pool=None, adversary=None, policy="llm", seed=0,
          **opts) -> Puppeteer:
    agents = [Agent(name, profile, llm_name) for name, _, profile in ROSTER]
    agents = inject_adversaries(agents, bench.domain, llm_name, adversary)
    for agent, (_, action, _) in zip(agents, ROSTER):
        agent.action = action
    return Puppeteer(bench, agents, llm_name, policy, seed)
