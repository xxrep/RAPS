"""
Seed agent pool for dynamic recruitment.

When the broker cannot find a good downstream match among the *current* team for
the predicted intent, the coordinator can recruit a new specialist from this pool
and add it to the team on the fly — the "adaptive team grows to fit the task"
behavior. Each seed declares distinct `interests` so the content-centric broker can
match an intent to the right specialist via embeddings.
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SeedSpec:
    role: str
    agent_class: str          # name registered in AgentRegistry
    capabilities: str         # becomes the agent's system prompt
    interests: str            # declared intent, matched by the broker
    few_shot: str = ""

    @property
    def subscription(self) -> str:
        """The text the broker matches a predicted intent against."""
        return (f"Your Role: {self.role}.\n"
                f"Your Capabilities: {self.capabilities}\n"
                f"Your Interests: {self.interests}")


class SeedAgentPool:
    """Domain -> list of recruitable specialists (beyond the initial team)."""

    _POOL: Dict[str, List[SeedSpec]] = {
        "gsm8k": [
            SeedSpec(
                role="Algebra Specialist", agent_class="MathAgent",
                capabilities="You are an algebra specialist. Set up equations with explicit "
                             "variables for the unknowns, then solve them symbolically before "
                             "substituting numbers. End with 'The answer is X'.",
                interests="setting up and solving algebraic equations, isolating unknowns, "
                          "symbolic manipulation, systems of equations"),
            SeedSpec(
                role="Arithmetic Verifier", agent_class="MathAgent",
                capabilities="You are an arithmetic verifier. Independently recompute every "
                             "numeric step from scratch and flag any calculation error. "
                             "End with 'The answer is X'.",
                interests="double-checking arithmetic, recomputing numeric results, catching "
                          "calculation mistakes, validating totals and sums"),
            SeedSpec(
                role="Word Problem Translator", agent_class="MathAgent",
                capabilities="You translate the natural-language word problem into precise "
                             "quantities, relationships and a solvable formulation, then solve. "
                             "End with 'The answer is X'.",
                interests="translating word problems into formal quantities and relations, "
                          "extracting givens and unknowns, modeling rates and ratios"),
            SeedSpec(
                role="Unit & Constraint Checker", agent_class="MathAgent",
                capabilities="You check units, ranges and constraints, and sanity-bound the "
                             "result against the problem statement. End with 'The answer is X'.",
                interests="checking units and dimensions, validating constraints and bounds, "
                          "sanity-checking magnitudes and plausibility"),
        ],
        "humaneval": [
            SeedSpec(
                role="Edge Case Analyst", agent_class="CodeWriting",
                capabilities="You enumerate boundary and edge cases (empty input, negatives, "
                             "overflow, duplicates) and ensure the implementation handles them.",
                interests="boundary conditions, edge cases, empty and degenerate inputs, "
                          "off-by-one errors, corner-case correctness"),
            SeedSpec(
                role="Complexity Optimizer", agent_class="CodeWriting",
                capabilities="You analyze and improve the time/space complexity of the solution "
                             "while preserving correctness.",
                interests="algorithmic complexity, time and space efficiency, optimizing loops "
                          "and data structures, performance"),
            SeedSpec(
                role="Spec & Signature Reviewer", agent_class="CodeWriting",
                capabilities="You verify the function signature, types and docstring contract "
                             "exactly match the required specification.",
                interests="function signature conformance, type correctness, docstring and "
                          "specification compliance, return type checks"),
        ],
        "mmlu": [
            SeedSpec(
                role="Statistician", agent_class="AnalyzeAgent",
                capabilities="You are a statistician. Apply probability and statistical "
                             "reasoning. Reply in under 100 words with a brief analysis.",
                interests="probability, statistics, data analysis, distributions, hypothesis testing"),
            SeedSpec(
                role="Philosopher", agent_class="AnalyzeAgent",
                capabilities="You are a philosopher. Apply logic, ethics and conceptual "
                             "analysis. Reply in under 100 words with a brief analysis.",
                interests="philosophy, logic, ethics, epistemology, conceptual analysis, reasoning"),
            SeedSpec(
                role="Biologist", agent_class="AnalyzeAgent",
                capabilities="You are a biologist. Apply knowledge of biology, genetics and "
                             "medicine. Reply in under 100 words with a brief analysis.",
                interests="biology, genetics, anatomy, physiology, ecology, molecular biology"),
            SeedSpec(
                role="Computer Scientist", agent_class="AnalyzeAgent",
                capabilities="You are a computer scientist. Apply knowledge of algorithms, "
                             "systems and theory. Reply in under 100 words with a brief analysis.",
                interests="computer science, algorithms, data structures, computation, "
                          "operating systems, complexity theory"),
        ],
    }

    @classmethod
    def get(cls, domain: str) -> List[SeedSpec]:
        return list(cls._POOL.get(domain, []))

    @classmethod
    def add(cls, domain: str, spec: SeedSpec) -> None:
        cls._POOL.setdefault(domain, []).append(spec)
