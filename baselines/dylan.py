"""Dynamic LLM-Agent Network: layered feed-forward debate. A round sees every reply of
the round before; an early-stopping consensus ends the task as soon as one prediction
holds strictly more than two thirds of the team; before the last round a listwise
ranking call keeps only the top-2 agents. The offline importance scores do not change
any prediction and are omitted."""
import math
import random
import re
from collections import Counter

from baselines.common import get_llm, majority_vote

ROUNDS = 3
SURVIVORS = 2


class DyLAN:
    def __init__(self, bench, agents, llm_name, rounds=ROUNDS, survivors=SURVIVORS, seed=0):
        self.bench = bench
        self.agents = agents
        self.llm = get_llm(llm_name)
        self.rounds = rounds
        self.survivors = survivors
        self.rng = random.Random(seed)

    def _reply(self, agent, task: str, formers) -> str:
        if not formers:
            return agent.respond(f"Here is the question:\n{task}\n\n{self.bench.format}", task)
        shuffled = list(formers)
        self.rng.shuffle(shuffled)
        body = "".join(f"\n\nAgent solution {i + 1}: ```{r}```" for i, r in enumerate(shuffled))
        return agent.respond(
            f"Here is the question:\n{task}\n\nThese are the solutions to the problem from "
            f"other agents: {body}\n\nNotice that their answers might be all wrong. Using "
            f"the reasoning from other agents as additional advice, give your updated "
            f"answer. {self.bench.format}", task)

    def _consensus(self, replies, team_size: int):
        """The reply whose prediction holds strictly more than two thirds of the team,
        else None."""
        top, count = Counter(self.bench.extract(r) for r in replies).most_common(1)[0]
        if count <= math.floor(2 * team_size / 3):
            return None
        return next(r for r in replies if self.bench.extract(r) == top)

    def _select_survivors(self, task: str, replies):
        """One listwise ranking call over the shuffled replies keeps the top-2 agents;
        an unparseable ranking falls back to a random pair."""
        order = list(range(len(self.agents)))
        self.rng.shuffle(order)
        body = "".join(f"\n\nAgent solution {k + 1}: ```{replies[j]}```"
                       for k, j in enumerate(order))
        out = self.llm.gen(
            f"Here is the question:\n{task}\n\nThese are the solutions to the problem from "
            f"other agents: {body}\n\nPlease choose the best {self.survivors} solutions and "
            "think step by step. Put your answer in the form like [1,2] or [3,4] at the "
            "end of your response.")
        picks = re.findall(r"\[(\d+),\s*(\d+)\]", out)
        chosen = []
        if picks:
            for x in picks[-1]:
                idx = min(max(int(x) - 1, 0), len(order) - 1)
                if order[idx] not in chosen:
                    chosen.append(order[idx])
        while len(chosen) < min(self.survivors, len(self.agents)):
            extra = self.rng.randrange(len(self.agents))
            if extra not in chosen:
                chosen.append(extra)
        return chosen[:self.survivors]

    def solve(self, record: dict) -> str:
        task = record["task"]
        active, formers, selected = list(self.agents), [], False
        self.rng.shuffle(active)   # round 0 answers in a shuffled order
        for r in range(self.rounds):
            if not selected and r >= max(2, self.rounds - 1) and len(self.agents) > 3:
                # rank the replies of the round before; the reply positions map back to
                # the agents in the current (shuffled) order
                active = [active[i] for i in self._select_survivors(task, formers)]
                selected = True
            team_size = len(active) if selected else len(self.agents)
            replies = []
            for k, agent in enumerate(active):
                replies.append(self._reply(agent, task, formers))
                if k + 1 > math.floor(2 * team_size / 3):
                    answer = self._consensus(replies, team_size)
                    if answer is not None:
                        return answer
            formers = replies
        return majority_vote(formers, self.bench.extract)


def build(bench, llm_name, pool, rounds=None, seed=0, **opts) -> DyLAN:
    return DyLAN(bench, pool, llm_name, ROUNDS if rounds is None else rounds, seed=seed)
