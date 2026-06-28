"""
RAPSCoordinator — the shared Reputation-Aware Publish-Subscribe interaction loop.

The three benchmark runners (gsm8k / mmlu / humaneval) used to each carry a
near-identical copy of this loop. They now only build domain-specific agents +
evaluation and delegate the coordination itself to this class, which implements
the paper's three mechanisms:

  * Distributed Content-Centric Protocol  -> publish() + broker_route() matching
                                             predicted intent to agent subscriptions.
  * Reactive Subscription                 -> refine_system_prompt() each step.
  * Bayesian Reputation + watchdog        -> watchdog_evaluate() + ReputationManager.

One coordination round (per task) proceeds in steps:
  1. ingest inboxes        -> per-agent history + upstream maps
  2. reactive subscription -> refine each active agent's intent
  3. publication           -> each active agent produces a message
  4. watchdog + reputation -> upstream peers vet each message; update first-hand REP
  5. (optional) reputation gate / second-hand gossip   [Phase 2 hooks]
  6. termination check     -> consensus / max steps / no routing target
  7. broker routing        -> embedding match to the next active agents

Then a Final Answerer aggregates all publications into the final output.
"""
import re
import doctest
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from RAPS.graph.node import Node
from RAPS.agents.agent_registry import AgentRegistry
from RAPS.agents.seed_pool import SeedAgentPool, SeedSpec
from RAPS.tools.coding.python_executor import execute_code_get_return
from RAPS.tools.coding.executor_utils import function_with_timeout


def _clean_want(want: str) -> str:
    """Strip a docstring terminator that doctest leaks into the last example's
    expected output when there is no blank line before the closing triple-quote
    (common in HumanEval prompts), e.g. want='12\\n\"\"\"' -> '12'."""
    lines = want.strip().splitlines()
    while lines and lines[-1].strip() in ('"""', "'''", ''):
        lines.pop()
    return "\n".join(lines).strip()


def run_asserts(code: str, asserts: List[str], timeout: int = 5):
    """Code Verifier: run a list of `assert f(x) == y` checks against the candidate.
    Returns (feedback, n_failures)."""
    if not asserts:
        return "", 0
    ns: Dict[str, Any] = {}
    try:
        function_with_timeout(exec, (code, ns), timeout)
    except Exception as e:
        return f"The code fails to load: {type(e).__name__}: {e}", len(asserts)
    fails = []
    for a in asserts:
        try:
            function_with_timeout(exec, (a, dict(ns)), timeout)
        except AssertionError:
            fails.append(f"{a}   # FAILED (wrong output)")
        except Exception as e:
            fails.append(f"{a}   # raised {type(e).__name__}: {e}")
    return "\n".join(fails[:6]), len(fails)


def run_public_doctests(prompt: str, code: str, timeout: int = 5):
    """Run the PUBLIC `>>>` examples embedded in the problem prompt against the
    candidate code (legit feedback — never touches the hidden test suite).

    Returns (passed, feedback, n_examples, n_failures).
    """
    examples = doctest.DocTestParser().get_examples(prompt or "")
    if not examples:
        return True, "", 0, 0
    ns: Dict[str, Any] = {}
    try:
        function_with_timeout(exec, (code, ns), timeout)
    except Exception as e:
        return False, f"The code fails to load: {type(e).__name__}: {e}", len(examples), len(examples)

    failures = []
    for ex in examples:
        src, want = ex.source.strip(), _clean_want(ex.want)
        if not want:
            continue  # statement-only example (e.g. an import); nothing to compare
        try:
            got = function_with_timeout(eval, (src, ns), timeout)
            if want not in (repr(got), str(got)):
                failures.append(f">>> {src}\n   expected: {want}\n   got:      {got!r}")
        except Exception as e:
            failures.append(f">>> {src}\n   raised {type(e).__name__}: {e}")
    if not failures:
        return True, "All public examples passed.", len(examples), 0
    fb = "Failing public examples:\n" + "\n".join(failures[:6])
    return False, fb, len(examples), len(failures)


# Program-Aided computation: translate the agreed reasoning into Python and EXECUTE it.
_PAL_SYSTEM = (
    "You are a programming expert. Given a math problem and the reasoning from other agents, "
    "write a single self-contained Python program that computes the final numeric answer. "
    "Define a function that takes no arguments and returns the result; the LAST line must call "
    "it and assign the value to a variable named `answer`. Use only Python standard library. "
    "Respond with ONLY one Python code block, nothing else. Example:\n"
    "```python\ndef solve():\n    x = 10\n    y = 20\n    return x + y\nanswer = solve()\n```"
)


def _extract_python(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if "```python" in text:
        text = text.split("```python", 1)[1]
    elif "```" in text:
        text = text.split("```", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]
    return text.strip()


@dataclass
class RAPSConfig:
    domain: str = "gsm8k"           # watchdog domain tag
    max_steps: int = 3
    top_k: int = 1                  # broker fan-out per publisher
    sim_threshold: float = 0.3      # broker embedding similarity threshold
    entry_index: int = 0            # which agent receives the user query first
    use_watchdog: bool = True       # run the local watchdog + first-hand reputation
    reputation_gate: bool = False   # drop messages from distrusted senders during routing
    trust_threshold: float = 0.3    # reputation cutoff for the routing gate (isolate peer)
    second_hand_gossip: bool = False  # exchange first-hand reports between agents each step
    reset_reputation: bool = True   # reset every agent's reputation before each task
    # Bayesian reputation (CONFIDANT) hyper-parameters, applied to every agent so
    # experiments tune them in one place instead of relying on Node's hardcoded defaults.
    merge_weight: float = 0.1            # weight of second-hand evidence when merging reports
    rep_discount: float = 0.999          # temporal fading factor for REP/FH/TRUST
    deviation_threshold: float = 0.2     # CONFIDANT deviation test cutoff
    reporter_trust_threshold: float = 0.5  # min trust to treat a reporter as credible
    # --- adaptive team ---
    dynamic_recruit: bool = False   # recruit new specialists from the seed pool on demand
    max_team_size: int = 8          # cap on team size when recruiting
    adaptive_capacity: bool = False  # scale broker fan-out (top_k) to task difficulty
    max_top_k: int = 2              # upper bound on fan-out under adaptive capacity
    # --- improved prompts (persona-only refinement + gap-driven broker) ---
    # NOTE: validated to HURT accuracy on GSM8K (temp=0: 0.950 -> 0.825); kept off
    # by default as an ablation knob. The step-by-step "solve" refinement is beneficial.
    improved_prompts: bool = False
    # --- Program-Aided computation (PAL) ---
    # Keep the reasoning ladder, but compute the final numeric answer by EXECUTING
    # Python (the Programming Expert's real job), eliminating LLM arithmetic errors.
    tool_verify: bool = False
    tool_verify_domains: tuple = ("gsm8k", "math")
    # --- code execute-feedback-repair (self-debug) ---
    # Run the candidate code against the PUBLIC doctest examples (never the hidden
    # test) and iteratively repair using the real error/failing case.
    code_verify: bool = False
    code_verify_domains: tuple = ("humaneval",)
    code_verify_max_iters: int = 3
    # Edge-Case Writer agent: generate extra edge tests beyond the public doctests.
    # Used only as a SECONDARY (tie-break) signal so model-written tests can never
    # override the ground-truth public examples.
    edge_cases: bool = False
    n_edge_cases: int = 8


@dataclass
class TaskResult:
    final_output: str
    publications: List[Dict[str, Any]]
    refinement_record: Dict[str, str]
    task_log: Dict[str, Any]
    reputation: Dict[str, Dict[str, float]] = field(default_factory=dict)
    team: Dict[str, Any] = field(default_factory=dict)  # adaptive-team summary


def _default_logger(msg: str) -> None:
    print(msg)


class RAPSCoordinator:
    def __init__(
        self,
        agents: List[Node],
        final_answerer: Node,
        config: RAPSConfig,
        logger: Optional[Callable[[str], None]] = None,
        answer_extractor: Optional[Callable[[str], str]] = None,
    ):
        """
        agents            : the worker agent pool (Nodes).
        final_answerer    : the decision Node that produces the final answer.
        config            : RAPSConfig.
        logger            : callable(str) for progress lines (defaults to print).
        answer_extractor  : optional fn mapping a publication's text to a normalized
                            answer; when all valid publications in a step agree on a
                            non-empty answer, the round stops early (real consensus,
                            replacing the old dead "final answer: yes" string check).
        """
        self._initial_agents: List[Node] = list(agents)
        self.agents = list(agents)
        self.final_answerer = final_answerer
        self.cfg = config
        self.log = logger or _default_logger
        self.answer_extractor = answer_extractor
        self.agent_map: Dict[str, Node] = {a.id: a for a in self.agents}
        self._llm_name = agents[0].llm_name if agents else ""
        self._recruitable: List[SeedSpec] = []
        self._top_k = config.top_k           # effective fan-out (capacity-adapted per task)
        for agent in self._initial_agents:
            self._apply_reputation_config(agent)
        self._apply_reputation_config(self.final_answerer)

    def _apply_reputation_config(self, agent: Node) -> None:
        """Centralize CONFIDANT hyper-parameters + prompt strategy on an agent."""
        rm = agent.rep_manager
        rm.w = self.cfg.merge_weight
        rm.alpha = self.cfg.rep_discount
        rm.beta = self.cfg.rep_discount
        rm.d = self.cfg.deviation_threshold
        rm.trust_threshold = self.cfg.reporter_trust_threshold
        agent.refine_mode = "persona" if self.cfg.improved_prompts else "solve"
        agent.broker_mode = "gap" if self.cfg.improved_prompts else "default"

    # ------------------------------------------------------------------ public

    def run(self, task: str, question: Optional[str] = None) -> TaskResult:
        """Run one full coordination round for a single task."""
        question = task if question is None else question
        self._reset_agents()
        self._top_k = self._estimate_top_k(task)

        entry_agent = self.agents[min(self.cfg.entry_index, len(self.agents) - 1)]
        active_agents: List[Node] = [entry_agent]
        entry_agent.inbox.append({"sender_id": "User", "role": "User", "content": task})

        round_publications: List[Dict[str, Any]] = []
        refinement_record: Dict[str, str] = {}
        task_log: Dict[str, Any] = {"task": task, "steps": []}
        recruited: List[str] = []
        active_sizes: List[int] = []

        for step in range(1, self.cfg.max_steps + 1):
            step_log = self._new_step_log(step, active_agents)
            active_sizes.append(len(active_agents))
            self.log(f"--- Step {step} --- active: {[a.id for a in active_agents]}")
            self._clear_traces()

            upstream_map, upstream_info_map = self._ingest_inboxes(active_agents)
            self._reactive_subscription(active_agents, question, step, refinement_record, step_log)
            current_pubs = self._publish(active_agents, task, round_publications, step_log)

            if not current_pubs:
                self._finalize_step_log(step_log, task_log)
                break

            if self.cfg.use_watchdog:
                current_pubs = self._watchdog(current_pubs, upstream_map, upstream_info_map,
                                              question, step_log)
                if not current_pubs:
                    self.log("[WATCHDOG] all publications blocked.")
                    self._finalize_step_log(step_log, task_log)
                    break

            if self.cfg.second_hand_gossip:
                self._gossip_reputation()

            if self._reached_consensus(current_pubs):
                self.log("[CONSENSUS] active agents agree; stopping early.")
                self._finalize_step_log(step_log, task_log)
                break

            if step == self.cfg.max_steps:
                self._finalize_step_log(step_log, task_log)
                break

            next_active = self._route(current_pubs, step_log, recruited)
            self._finalize_step_log(step_log, task_log)
            if not next_active:
                break
            active_agents = list(next_active)

        final_output = self._final_decision(task, question, round_publications, task_log)

        team = {
            "initial_size": len(self._initial_agents),
            "final_size": len(self.agents),
            "recruited": recruited,
            "active_per_step": active_sizes,
            "top_k": self._top_k,
            "roster": [f"{a.role} ({a.id})" for a in self.agents],
        }
        task_log["team"] = team

        return TaskResult(
            final_output=final_output,
            publications=round_publications,
            refinement_record=refinement_record,
            task_log=task_log,
            reputation=self.snapshot_reputation(),
            team=team,
        )

    def _estimate_top_k(self, task: str) -> int:
        """Capacity adaptation: harder/longer problems fan out to more specialists."""
        if not self.cfg.adaptive_capacity:
            return self.cfg.top_k
        n_tokens = len(task.split())
        n_numbers = len(re.findall(r"\d+", task))
        hard = (n_tokens >= 60) or (n_numbers >= 5)
        top_k = self.cfg.top_k + (1 if hard else 0)
        return max(1, min(self.cfg.max_top_k, top_k))

    def snapshot_reputation(self) -> Dict[str, Dict[str, float]]:
        """Average reputation each agent assigns to every peer it has observed."""
        snap: Dict[str, Dict[str, float]] = {}
        for a in self.agents:
            rep = {t: round(r["a"] / (r["a"] + r["b"]), 4)
                   for t, r in a.rep_manager.REP.items()}
            if rep:
                snap[a.id] = rep
        return snap

    # ----------------------------------------------------------------- steps

    def _reset_agents(self) -> None:
        # Start every task from the initial team; any specialists recruited for a
        # previous task are discarded so the team adapts afresh to this task.
        self.agents = list(self._initial_agents)
        self.agent_map = {a.id: a for a in self.agents}
        self._recruitable = SeedAgentPool.get(self.cfg.domain) if self.cfg.dynamic_recruit else []
        for agent in self.agents:
            agent.history = []
            agent.inbox = []
            agent.refined_prompt = agent.system_prompt
            if self.cfg.reset_reputation:
                agent.rep_manager.FH.clear()
                agent.rep_manager.REP.clear()
                agent.rep_manager.TRUST.clear()
        self.final_answerer.refined_prompt = self.final_answerer.system_prompt
        self.final_answerer.llm_trace = []

    def _ingest_inboxes(self, active_agents: List[Node]):
        upstream_map: Dict[str, Set[str]] = {a.id: set() for a in active_agents}
        upstream_info_map: Dict[str, List[str]] = {a.id: [] for a in active_agents}
        for agent in active_agents:
            for msg in agent.inbox:
                sender_id = msg.get("sender_id")
                if sender_id != "User":
                    upstream_map[agent.id].add(sender_id)
                    upstream_info_map[agent.id].append(f"{msg['role']} ({sender_id}): {msg['content']}")
                    agent.history.append(f"{msg['role']}: {msg['content']}")
            agent.inbox = []
        return upstream_map, upstream_info_map

    def _reactive_subscription(self, active_agents, question, step, refinement_record, step_log):
        for agent in active_agents:
            context_str = "History:\nNone" if step == 1 else "History:\n" + "\n".join(agent.history)
            refined = agent.refine_system_prompt(context=context_str, question=question)
            refinement_record[agent.id] = refined
            step_log["refinements"][agent.id] = {
                "role": agent.role,
                "pre_refinement": agent.system_prompt,
                "post_refinement": refined,
            }

    def _publish(self, active_agents, task, round_publications, step_log):
        current_pubs = []
        for agent in active_agents:
            output = agent.publish({"task": task, "history": "\n".join(agent.history)})
            pub = {"sender_id": agent.id, "role": agent.role, "content": output}
            current_pubs.append(pub)
            round_publications.append(pub)
            step_log["publications"].append(pub)
            self.log(f"[PUBLISH] {agent.id} ({agent.role}): {str(output)[:120]}")
        return current_pubs

    def _watchdog(self, current_pubs, upstream_map, upstream_info_map, question, step_log):
        valid_pubs = []
        for pub in current_pubs:
            sender_id = pub["sender_id"]
            other_info = "\n".join(upstream_info_map.get(sender_id, []))
            blocking_agents = []
            for u_id in upstream_map.get(sender_id, set()):
                upstream_agent = self.agent_map.get(u_id)
                if upstream_agent is None:
                    continue
                is_valid = upstream_agent.watchdog_evaluate(
                    pub["content"], task_domain=self.cfg.domain,
                    question=question, existing_info=other_info)
                upstream_agent.rep_manager.update_first_hand(sender_id, is_valid)
                if not is_valid:
                    blocking_agents.append(u_id)
                    break
            if blocking_agents:
                self.log(f"[WATCHDOG-BLOCK] {sender_id} blocked by {blocking_agents}")
                step_log["broker_decisions"].append(
                    {"sender": sender_id, "blocked_by": blocking_agents, "status": "blocked"})
            else:
                valid_pubs.append(pub)
        return valid_pubs

    def _gossip_reputation(self) -> None:
        """Second-hand reputation exchange (CONFIDANT): each agent reports its
        first-hand observations to every peer, who merges them via the trust-weighted
        deviation test in ReputationManager.update_from_report."""
        reports = {a.id: a.rep_manager.export_first_hand() for a in self.agents}
        for receiver in self.agents:
            for reporter_id, report in reports.items():
                if reporter_id == receiver.id or not report:
                    continue
                receiver.rep_manager.update_from_report(reporter_id, report)

    def _reached_consensus(self, current_pubs) -> bool:
        if self.answer_extractor is None or len(current_pubs) < 2:
            return False
        answers = set()
        for pub in current_pubs:
            try:
                ans = str(self.answer_extractor(pub["content"])).strip()
            except Exception:
                return False
            if ans in ("", "0", "None"):
                return False
            answers.add(ans)
        return len(answers) == 1

    def _route(self, current_pubs, step_log, recruited: List[str]) -> Set[Node]:
        next_active: Set[Node] = set()
        step_pubs_str = [f"Agent {p['sender_id']} ({p['role']}): {p['content']}" for p in current_pubs]

        for pub in current_pubs:
            sender_id = pub["sender_id"]
            matcher = self.agent_map.get(sender_id)
            if matcher is None:
                continue

            # Content-centric candidate set = current teammates + recruitable seeds.
            # Recruitment is just routing to a seed that fits the intent better than
            # any current teammate, unifying both under the same broker mechanism.
            all_subscriptions = {a.id: a.refined_prompt for a in self.agents}
            seed_map = self._seed_candidates()
            all_subscriptions.update({sid: spec.subscription for sid, spec in seed_map.items()})

            try:
                matched = matcher.broker_route(step_pubs_str, all_subscriptions,
                                               top_k=self._top_k,
                                               sim_threshold=self.cfg.sim_threshold)
            except Exception as e:
                self.log(f"[BROKER] routing error: {e}")
                matched = []

            receivers = []
            for rid in matched:
                if rid in seed_map:
                    receiver = self._recruit(seed_map[rid], recruited)
                    if receiver is None:
                        continue
                else:
                    receiver = self.agent_map.get(rid)
                    if receiver is None or rid == sender_id:
                        continue
                    if self.cfg.reputation_gate and not self._sender_trusted(receiver, sender_id):
                        self.log(f"[REP-GATE] {receiver.id} drops msg from distrusted {sender_id}")
                        continue
                receiver.inbox.append(pub)
                next_active.add(receiver)
                receivers.append(receiver.id)
            step_log["broker_decisions"].append(
                {"sender": sender_id, "matched_keys": matched, "receivers": receivers})
        return next_active

    def _seed_candidates(self) -> Dict[str, SeedSpec]:
        """Virtual broker candidates for not-yet-recruited specialists."""
        if not self.cfg.dynamic_recruit or len(self.agents) >= self.cfg.max_team_size:
            return {}
        return {f"seed::{spec.role}": spec for spec in self._recruitable}

    def _recruit(self, spec: SeedSpec, recruited: List[str]) -> Optional[Node]:
        """Instantiate a seed specialist and add it to the live team."""
        if len(self.agents) >= self.cfg.max_team_size:
            return None
        try:
            agent = AgentRegistry.get(
                spec.agent_class, id=None, llm_name=self._llm_name, domain=self.cfg.domain,
                role=spec.role, capabilities=spec.capabilities, interests=spec.interests,
                additional_instructions="", few_shot=spec.few_shot)
        except Exception as e:
            self.log(f"[RECRUIT] failed to instantiate {spec.role}: {e}")
            return None
        agent.history = []
        agent.inbox = []
        agent.refined_prompt = agent.system_prompt
        self._apply_reputation_config(agent)
        self.agents.append(agent)
        self.agent_map[agent.id] = agent
        self._recruitable = [s for s in self._recruitable if s.role != spec.role]
        recruited.append(spec.role)
        self.log(f"[RECRUIT] added specialist '{spec.role}' ({agent.id}); team size {len(self.agents)}")
        return agent

    def _sender_trusted(self, receiver: Node, sender_id: str) -> bool:
        """The receiver consults its local Bayesian reputation (the Node watchdog)
        about the sender. Unseen peers default to trusted (REP a=b=1 -> 0.5)."""
        return receiver.check_trust(sender_id, threshold=self.cfg.trust_threshold)

    def _verify_code(self, prompt: str, code: str, edge_asserts: List[str]):
        """Code Verifier: run candidate against public doctests (ground truth) and
        edge-case asserts (secondary). Returns (pub_fail, edge_fail, n_pub, feedback)."""
        pub_pass, pub_fb, n_pub, pub_fail = run_public_doctests(prompt, code)
        edge_fb, edge_fail = run_asserts(code, edge_asserts)
        fb = ""
        if pub_fail:
            fb += pub_fb + "\n"
        if edge_fail:
            fb += "Failing edge cases (additional, derived from the spec):\n" + edge_fb
        return pub_fail, edge_fail, n_pub, fb.strip()

    def _code_verify_repair(self, prompt: str, final_output: str) -> str:
        """Execute-feedback-repair. An Edge-Case Writer adds tests beyond the public
        doctests; a Code Verifier runs the candidate and feeds real failures to repair.
        Best code is chosen by (public_failures, edge_failures) lexicographically, so
        model-written edge tests never override the ground-truth public examples."""
        code = _extract_python(final_output) or final_output
        edge_asserts = self._generate_edge_cases(prompt) if self.cfg.edge_cases else []

        pub_fail, edge_fail, n_pub, fb = self._verify_code(prompt, code, edge_asserts)
        if n_pub == 0 and not edge_asserts:
            return final_output  # nothing to verify against
        best_code, best_key = code, (pub_fail, edge_fail)
        self.log(f"[CODE-VERIFY] initial: pub={pub_fail} edge={edge_fail}")
        iters = 0
        while best_key != (0, 0) and iters < self.cfg.code_verify_max_iters:
            iters += 1
            new_code = self._repair_code(prompt, best_code, fb)
            if not new_code:
                break
            pub_fail, edge_fail, n_pub, fb = self._verify_code(prompt, new_code, edge_asserts)
            if (pub_fail, edge_fail) <= best_key:   # lexicographic: public dominates
                best_code, best_key = new_code, (pub_fail, edge_fail)
            self.log(f"[CODE-VERIFY] iter {iters}: pub={pub_fail} edge={edge_fail}")
            if best_key == (0, 0):
                break
        return f"```python\n{best_code}\n```"

    def _generate_edge_cases(self, prompt: str) -> List[str]:
        """Edge-Case Writer agent: derive extra edge-case asserts from the SPEC only."""
        sys_msg = (
            "You are a meticulous test engineer. Given a function specification, write diverse "
            "EDGE-CASE tests as Python assert statements that call the function by its exact name. "
            "Cover empty / boundary / negative / large / duplicate / special inputs. Derive the "
            "expected outputs ONLY from the specification, reasoning carefully — never guess. "
            "Output ONLY assert statements, one per line, no code fences, no commentary.")
        user = f"# Specification\n{prompt}\n\nWrite up to {self.cfg.n_edge_cases} edge-case asserts."
        msg = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
        self.final_answerer.llm_trace.append({"type": "edge_cases", "messages": msg})
        raw = self.final_answerer.llm.gen(msg)
        self.final_answerer.llm_trace[-1]["output"] = raw
        asserts = []
        for line in str(raw).splitlines():
            line = line.strip().strip("`").strip()
            if line.startswith("assert "):
                asserts.append(line)
        return asserts[: self.cfg.n_edge_cases]

    def _repair_code(self, prompt: str, code: str, feedback: str) -> str:
        sys_msg = ("You are an expert Python debugger. Given a problem, a candidate solution, "
                   "and the public examples it fails, return a corrected solution. Keep the same "
                   "function name and signature. Respond with ONLY one Python code block.")
        user = (f"# Problem\n{prompt}\n\n# Current solution\n```python\n{code}\n```\n\n"
                f"# It fails these public examples\n{feedback}\n\n"
                f"Return the full corrected solution as a single Python code block.")
        msg = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
        self.final_answerer.llm_trace.append({"type": "code_repair", "messages": msg})
        raw = self.final_answerer.llm.gen(msg)
        self.final_answerer.llm_trace[-1]["output"] = raw
        return _extract_python(raw)

    def _tool_compute(self, task: str, history: str) -> Tuple[Any, str]:
        """PAL: write Python from the agreed reasoning and EXECUTE it for the exact answer."""
        user = (f"Problem:\n{task}\n\nReasoning from other agents (use it to set up the "
                f"computation, but recompute exactly in code):\n{history}\n\n"
                f"Write the Python program now.")
        msg = [{"role": "system", "content": _PAL_SYSTEM}, {"role": "user", "content": user}]
        self.final_answerer.llm_trace.append({"type": "tool_compute", "messages": msg})
        raw = self.final_answerer.llm.gen(msg)
        self.final_answerer.llm_trace[-1]["output"] = raw
        code = _extract_python(raw)
        value = execute_code_get_return(code)
        return value, code

    @staticmethod
    def _is_clean_number(value) -> bool:
        if value is None or isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return True
        try:
            float(str(value))
            return True
        except (ValueError, TypeError):
            return False

    def _final_decision(self, task, question, round_publications, task_log) -> str:
        combined = "\n".join(
            f"{p['role']} ({p['sender_id']}): {p['content']}" for p in round_publications)
        self.final_answerer.llm_trace = []

        tool_note = ""
        if self.cfg.tool_verify and self.cfg.domain in self.cfg.tool_verify_domains:
            value, code = self._tool_compute(task, combined)
            if self._is_clean_number(value):
                tool_note = (f"\n\n## Tool-verified computation (executed Python)\n"
                             f"A Python program was executed and returned: answer = {value}\n"
                             f"Treat this executed result as the AUTHORITATIVE numeric answer "
                             f"unless it is clearly an error.")
                combined += tool_note
                self.log(f"[TOOL] executed answer = {value}")
            else:
                self.log(f"[TOOL] execution gave no clean number ({value!r}); ignoring")

        context_str = f"Original Task: {task}\n\nProcess History:\n{combined}"
        self.final_answerer.refine_system_prompt(context=context_str, question=question)
        final_output = self.final_answerer.publish({"task": task, "history": combined})
        self.log(f"[FINAL] {str(final_output)[:160]}")

        if self.cfg.code_verify and self.cfg.domain in self.cfg.code_verify_domains:
            final_output = self._code_verify_repair(task, final_output)

        task_log["steps"].append({
            "step": "Final",
            "active_agents": ["Final_Answerer"],
            "publications": [{"sender_id": "Final_Answerer", "role": "Final Answerer",
                              "content": final_output}],
            "refinements": {"Final_Answerer": self.final_answerer.refined_prompt},
            "llm_calls": {"Final_Answerer": self.final_answerer.llm_trace},
            "broker_decisions": [],
        })
        return final_output

    # ----------------------------------------------------------------- logging

    @staticmethod
    def _new_step_log(step, active_agents) -> Dict[str, Any]:
        return {
            "step": step,
            "active_agents": [a.id for a in active_agents],
            "publications": [],
            "refinements": {},
            "broker_decisions": [],
            "llm_calls": {},
        }

    def _clear_traces(self) -> None:
        for a in self.agents:
            if hasattr(a, "llm_trace"):
                a.llm_trace = []

    def _finalize_step_log(self, step_log, task_log) -> None:
        step_log["llm_calls"] = {
            a.id: a.llm_trace for a in self.agents
            if hasattr(a, "llm_trace") and len(a.llm_trace)
        }
        step_log["reputation"] = self.snapshot_reputation()
        task_log["steps"].append(step_log)
