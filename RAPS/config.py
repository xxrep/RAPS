"""Canonical experimental configuration (NC revision, Supplementary Section A).

Every controlled factor of the comparison lives here exactly once:

  * RAPS protocol parameters (Table S.4) — fixed across all five benchmarks;
  * per-benchmark role sets and the final-answerer role (Table S.2);
  * shared decoding controls (Table S.1).

Runners build their agent teams and RAPSConfig from this module, so the paper's
claim "one parameter set, used unchanged on all five benchmarks" is checkable
in code. Anything an ablation varies is passed as an explicit override at the
call site, never by editing these constants.
"""
from dataclasses import dataclass, replace
from typing import Dict, List

from RAPS.core.coordinator import RAPSConfig
from RAPS.llm.llm import LLM

# --- Table S.4: RAPS-specific protocol parameters (no baseline counterpart) ---
SIM_THRESHOLD = 0.5     # tau_sim: cosine gate for broker subscription matching
TOP_K = 3               # fan-out cap per publisher
REP_DISCOUNT = 0.9      # lambda: fading applied to Beta pseudo-counts before each update
TRUST_THRESHOLD = 0.7   # tau_rep: misbehaviour posterior at which the routing gate isolates a peer
MAX_STEPS = 5           # k: shared communication-round cap on all five benchmarks

# --- Table S.1: shared decoding controls, as applied by every LLM backend ---
TEMPERATURE = LLM.DEFAULT_TEMPERATURE   # one completion per call, identical for all methods
MAX_TOKENS = LLM.DEFAULT_MAX_TOKENS     # per-call completion cap for every compared method


@dataclass(frozen=True)
class BenchmarkSpec:
    """One row of Table S.2: the five-agent crafted pool of a benchmark."""
    domain: str
    agent_class: str        # AgentRegistry name instantiated for every team member
    roles: List[str]        # the crafted pool (five role profiles)
    final_answerer: str     # role that composes the final answer — one of the five,
                            # not an additional aggregator (Supp. A.2)
    tools: str = ""


_MATH_ROLES = ["Math Solver", "Mathematical Analyst", "Programming Expert",
               "Inspector", "Summarizer"]

BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "mmlu": BenchmarkSpec(
        domain="mmlu", agent_class="AnalyzeAgent",
        roles=["Knowledge Expert", "Mathematician", "Programmer", "Doctor", "Economist"],
        final_answerer="Knowledge Expert", tools="domain-specific retrieval"),
    "gsm8k": BenchmarkSpec(
        domain="gsm8k", agent_class="MathAgent",
        roles=_MATH_ROLES, final_answerer="Summarizer", tools="python execution, web search"),
    "svamp": BenchmarkSpec(
        domain="svamp", agent_class="MathAgent",
        roles=_MATH_ROLES, final_answerer="Summarizer", tools="python execution, web search"),
    "aqua": BenchmarkSpec(
        domain="aqua", agent_class="MathAgent",
        roles=_MATH_ROLES, final_answerer="Summarizer", tools="python execution, web search"),
    "humaneval": BenchmarkSpec(
        domain="humaneval", agent_class="CodeWriting",
        roles=["Project Manager", "Algorithm Designer", "Programming Expert",
               "Test Analyst", "Bug Fixer"],
        final_answerer="Test Analyst", tools="python execution, web search"),
}

# The naive pool (Supp. A.2): five generic profiles with no domain content,
# identical on every benchmark — four reasoning strategies plus a terminal
# summarizer. Used only by the pool-quality comparison (Fig. 4f, Table S.15).
NAIVE_POOL: Dict[str, str] = {
    "Direct Answerer": "You answer the task directly and concisely, without intermediate reasoning.",
    "CoT Reasoner": "You think step by step (chain of thought) before giving the final answer.",
    "Self-Reflector": "You draft an answer, reflect on it for errors, then give the final answer.",
    "ReAct Solver": "You alternate between reasoning about the task and acting on available tools.",
    "Summarizer": "You summarize the discussion so far and compose the final answer.",
}

#: The naive pool's terminal profile, which composes the final answer.
NAIVE_FINAL_ANSWERER = "Summarizer"

#: The naive profile defined by acting on tools, so the retrieval interface is granted to
#: it on the terms the harness grants it to a crafted role that uses one.
NAIVE_TOOL_USER = "ReAct Solver"


def pool_spec(domain: str, naive: bool = False) -> BenchmarkSpec:
    """The pool a run draws its five agents from: the benchmark's crafted profiles
    (Table S.2), or the naive profiles, which carry no domain or role content and are the
    same on every benchmark. Both place the agent that emits the final answer inside the
    five, and everything the harness owns is unchanged between them."""
    spec = BENCHMARKS[domain]
    if not naive:
        return spec
    return replace(spec, roles=list(NAIVE_POOL), final_answerer=NAIVE_FINAL_ANSWERER)


def add_pool_flag(parser):
    """The pool-quality comparison of Fig. 4f is a switch of profiles and nothing else."""
    parser.add_argument("--naive_pool", action="store_true",
                        help="draw the five agents from the naive pool of generic profiles "
                             "instead of the benchmark's crafted ones (Fig. 4f, Table S.15)")
    return parser


#: Mechanisms the reported system runs, and the RAPSConfig field each one sets. An
#: ablation removes one of them; nothing here varies by benchmark.
MECHANISM_FLAGS = {
    "watchdog": "use_watchdog",
    "reputation_gate": "reputation_gate",
    "second_hand_gossip": "second_hand_gossip",
}


def add_mechanism_flags(parser):
    """Give every mechanism a `--flag` / `--no_flag` pair on the runner's command line, so
    either spelling selects the same mechanism, and leave the value unset when neither is
    given so that `protocol_config` supplies it."""
    for flag in MECHANISM_FLAGS:
        name = flag.replace("_", " ")
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f"--{flag}", dest=flag, action="store_true", default=None,
                          help=f"keep the {name} on, as the reported system does")
        group.add_argument(f"--no_{flag}", dest=flag, action="store_false",
                          help=f"ablation: run without the {name}")
    parser.add_argument("--reset_reputation", action="store_true",
                        help="ablation: clear the posterior before each task instead of "
                             "carrying it across them")
    return parser


def mechanism_overrides(args) -> Dict[str, bool]:
    """The RAPSConfig overrides the mechanism flags imply, omitting the ones left unset."""
    overrides = {field: getattr(args, flag) for flag, field in MECHANISM_FLAGS.items()
                 if getattr(args, flag, None) is not None}
    if getattr(args, "reset_reputation", False):
        overrides["reset_reputation"] = True
    return overrides


def protocol_config(domain: str, **overrides) -> RAPSConfig:
    """RAPSConfig preset to the paper's single parameter set (Table S.4).

    `overrides` exist for ablations only (e.g. reputation_gate=False for the
    w/o-BR row); the defaults below are the reported system.
    """
    cfg = RAPSConfig(
        domain=domain,
        max_steps=MAX_STEPS,
        top_k=TOP_K,
        sim_threshold=SIM_THRESHOLD,
        rep_discount=REP_DISCOUNT,
        merge_weight=0.1,            # second-hand reports weigh below first-hand judgements
        trust_threshold=TRUST_THRESHOLD,
        gate_min_observations=1.0,   # exemption until one first-hand observation (Table S.17, rule †)
        reputation_gate=True,
        use_watchdog=True,
        second_hand_gossip=True,     # the witness scheme is part of the reported mechanism
        reset_reputation=False,      # the posterior is carried across tasks
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"RAPSConfig has no field {key!r}")
        setattr(cfg, key, value)
    return cfg
