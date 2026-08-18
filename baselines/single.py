"""Direct answering: one backbone call per task with no coordination at all — the
floor row of Table 1."""
from baselines.common import Agent


class Single:
    def __init__(self, bench, llm_name):
        self.bench = bench
        self.agent = Agent("Solver", llm_name=llm_name)

    def solve(self, record: dict) -> str:
        task = record["task"]
        return self.agent.respond(f"{task}\n\n{self.bench.format}", task)


def build(bench, llm_name, pool=None, **opts) -> Single:
    return Single(bench, llm_name)
