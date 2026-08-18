"""GPTSwarm: the pool as a directed graph of direct-answer nodes plus one decision
node. A direct-answer node reads only the task, so its incoming edges carry no content;
the graph decides whose answers the final majority vote sees. The candidate edges — every ordered agent pair and every edge into the decision
node — carry a Bernoulli logit each and are optimized with a declared REINFORCE budget
on a held-out slice; without a budget the swarm runs its fully connected acyclic
realization."""
import math
import random

from baselines.common import majority_vote

LR = 0.1          # Adam rate of the edge optimization
BATCH_SIZE = 4    # dev samples per iteration, one graph realization each


class GPTSwarm:
    def __init__(self, bench, agents, search_budget=0, dev_records=None, seed=0):
        self.bench = bench
        self.agents = agents
        self.search_budget = search_budget
        self.seed = seed
        n = len(agents)
        self.edges = [(i, j) for i in range(n) for j in range(n) if i != j]
        self.edges += [(i, n) for i in range(n)]   # agent -> decision node (index n)
        self.logits = [0.0] * len(self.edges)      # every edge starts at p = 1/2
        if search_budget:
            if not dev_records:
                raise ValueError("gptswarm --search_budget needs held-out dev_records")
            self._optimize(dev_records)

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        return math.exp(x) / (1.0 + math.exp(x))

    def _creates_cycle(self, adjacency, i: int, j: int) -> bool:
        """Whether adding i -> j closes a directed cycle among the agent nodes."""
        seen, stack = set(), [j]
        while stack:
            node = stack.pop()
            if node == i:
                return True
            if node not in seen:
                seen.add(node)
                stack.extend(adjacency[node])
        return False

    def _sample_graph(self, rng):
        """One Bernoulli realization of the candidate edges. An edge that would close a
        cycle is never added and earns no gradient; every other edge contributes its
        probability to the REINFORCE trace."""
        chosen, trace = [], []
        adjacency = {i: set() for i in range(len(self.agents))}
        for k, ((i, j), logit) in enumerate(zip(self.edges, self.logits)):
            p = self._sigmoid(logit)
            on = rng.random() < p
            if on and j < len(self.agents) and self._creates_cycle(adjacency, i, j):
                continue
            trace.append((k, p, float(on)))
            if on:
                chosen.append((i, j))
                if j < len(self.agents):
                    adjacency[i].add(j)
        return chosen, trace

    def _eval_graph(self):
        """The deterministic graph of an optimized swarm (p > 1/2), or the fully
        connected acyclic realization when no optimization ran."""
        if self.search_budget:
            return [e for e, logit in zip(self.edges, self.logits)
                    if self._sigmoid(logit) > 0.5]
        chosen, adjacency = [], {i: set() for i in range(len(self.agents))}
        for i, j in self.edges:
            if j < len(self.agents) and self._creates_cycle(adjacency, i, j):
                continue
            chosen.append((i, j))
            if j < len(self.agents):
                adjacency[i].add(j)
        return chosen

    def _forward(self, task: str, chosen) -> str:
        """One topological pass: the agents wired into the decision node each answer the
        task once and the decision node majority-votes over their answers (over every
        agent when the graph wires none in)."""
        n = len(self.agents)
        voters = [i for i, j in chosen if j == n] or list(range(n))
        answers = [self.agents[i].respond(f"{task}\n\n{self.bench.format}", task)
                   for i in voters]
        return majority_vote(answers, self.bench.extract)

    def _optimize(self, dev_records) -> None:
        """REINFORCE on the edge logits: loss = -log_prob(graph) * utility of the
        realized graph on one dev sample, Adam over the batch mean."""
        rng = random.Random(self.seed)
        m = [0.0] * len(self.edges)
        v = [0.0] * len(self.edges)
        for t in range(1, self.search_budget + 1):
            grad = [0.0] * len(self.edges)
            for _ in range(BATCH_SIZE):
                rec = dev_records[rng.randrange(len(dev_records))]
                chosen, trace = self._sample_graph(rng)
                output = self._forward(rec["task"], chosen)
                utility = float(self.bench.score(self.bench.extract(output), rec))
                for k, p, a in trace:
                    grad[k] += -utility * (a - p) / BATCH_SIZE
            for k in range(len(self.edges)):
                m[k] = 0.9 * m[k] + 0.1 * grad[k]
                v[k] = 0.999 * v[k] + 0.001 * grad[k] ** 2
                m_hat = m[k] / (1 - 0.9 ** t)
                v_hat = v[k] / (1 - 0.999 ** t)
                self.logits[k] -= LR * m_hat / (math.sqrt(v_hat) + 1e-8)

    def solve(self, record: dict) -> str:
        return self._forward(record["task"], self._eval_graph())


def build(bench, llm_name, pool, search_budget=0, dev_records=None, seed=0,
          **opts) -> GPTSwarm:
    return GPTSwarm(bench, pool, search_budget, dev_records, seed)
