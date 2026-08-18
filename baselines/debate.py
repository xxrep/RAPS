"""Multi-agent debate: the pool answers independently in the first round and
re-answers while seeing the peers' previous replies in each later round; the final
answer is the majority vote of the last round."""
from baselines.common import majority_vote

ROUNDS = 3


class Debate:
    def __init__(self, bench, agents, rounds=ROUNDS):
        self.bench = bench
        self.agents = agents
        self.rounds = rounds

    def _update_prompt(self, task: str, replies, skip: int) -> str:
        others = [r for i, r in enumerate(replies) if i != skip]
        body = "".join(f"\n\nAgent solution {i + 1}: ```{r}```" for i, r in enumerate(others))
        return (f"Here is the question:\n{task}\n\nThese are the solutions to the problem "
                f"from other agents: {body}\n\nUsing the reasoning from other agents as "
                f"additional advice, give your updated answer. {self.bench.format}")

    def solve(self, record: dict) -> str:
        task = record["task"]
        replies = [a.respond(f"Here is the question:\n{task}\n\n{self.bench.format}", task)
                   for a in self.agents]
        for _ in range(self.rounds - 1):
            replies = [a.respond(self._update_prompt(task, replies, i), task)
                       for i, a in enumerate(self.agents)]
        return majority_vote(replies, self.bench.extract)


def build(bench, llm_name, pool, rounds=None, **opts) -> Debate:
    return Debate(bench, pool, ROUNDS if rounds is None else rounds)
