"""SWERAPSAgent — each round is a RAPS dynamic ad-hoc network step.

We subclass mini-swe-agent's DefaultAgent and override ONLY `query()`:
the per-round "single LLM -> one action" becomes a publish/subscribe coordination:

  1. broker emits the next-step NEED (1 LLM call over recent progress)
  2. text-embedding-3-small matches NEED vs each persona's subscription -> route
  3. the selected persona REACTIVELY specializes its role to this issue (1 LLM call)
  4. it PUBLISHES one action (thought + ```mswea_bash_command```), parsed to a command

With `watchdog=True` the third RAPS mechanism runs alongside them: once the shell has
answered, the broker's local watchdog judges the action it engaged a host for (1 LLM call)
and the verdict updates a Beta posterior per host, on the same ReputationManager, decay and
tau_rep as the five benchmarks. Routing then withholds any host whose misbehaviour
posterior has reached tau_rep. It is off by default, so the reported long-horizon runs are
unaffected by it.

The outer multi-round loop, Docker env, submit detection, limits and trajectory
saving are all reused unchanged from DefaultAgent. One backbone (litellm ->
OpenAI-compatible route configured in mini-swe-agent's global .env); personas
differ only by system persona (shared pi_theta, distinct S_i — as in the paper).

`refine_once=True` is the subscription-freshness ablation of §2.6: each persona's
subscription is computed on first selection and reused for the rest of the
episode, instead of being re-derived against the current repository state.
"""
import os
import re
import time

import litellm
import numpy as np

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import Submitted

from RAPS.config import REP_DISCOUNT, TRUST_THRESHOLD
from RAPS.graph.reputation import ReputationManager
from RAPS.llm.llm import LLM
from RAPS.swe.team import DEFAULT_TEAM
from RAPS.swe.prompts import (SWE_RULES, BROKER_SYSTEM, REFINE_SYSTEM, FORMAT_REMINDER,
                              WATCHDOG_SYSTEM, REPRO_PATH, EDIT_TOOL_PATH, APPLY_EDIT_TOOL)

EMBED_MODEL = "text-embedding-3-small"
ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"
N_CANDIDATES = 4                                  # best-of-K when editing with a red reproduction

# USD per 1M tokens, substring-matched on the lowered model name. These are the rates of
# the cost ledger reported in Table S.8, namely the list prices of 1 July 2026 (OpenAI:
# gpt-4o $2.50/$10.00, gpt-4o-mini $0.15/$0.60, gpt-5.2 $1.75/$14.00; Anthropic: Sonnet
# $3/$15, Haiku $1/$5, Opus $5/$25). Order matters: "gpt-4o-mini" must precede "gpt-4o",
# which substring-matches it. Qwen3-32B is served locally and carries no bill. Set
# RAPS_SWE_PRICE_IN / RAPS_SWE_PRICE_OUT (per 1M tokens) to price a run at another date.
_PRICE_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-5.2": (1.75, 14.00),
    "sonnet": (3.00, 15.00),
    "haiku": (1.00, 5.00),
    "opus": (5.00, 25.00),
}


def _prices(model_name: str):
    """(input, output) USD per token for the backbone; (0, 0) when unknown so
    token counts — which are always real — stay the primary cost evidence."""
    if os.getenv("RAPS_SWE_PRICE_IN"):
        return (float(os.environ["RAPS_SWE_PRICE_IN"]) / 1e6,
                float(os.getenv("RAPS_SWE_PRICE_OUT", 0.0)) / 1e6)
    name = model_name.lower()
    for key, rates in _PRICE_PER_MTOK.items():
        if key in name:
            return rates[0] / 1e6, rates[1] / 1e6
    return 0.0, 0.0


def _cos(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


class SWERAPSAgent(DefaultAgent):
    def __init__(self, model, env, *, team=None, refine_once=False, watchdog=False,
                 trust_threshold=TRUST_THRESHOLD, rep_discount=REP_DISCOUNT,
                 gate_min_observations=1.0, **kwargs):
        super().__init__(model, env, **kwargs)
        self.team = list(team or DEFAULT_TEAM)
        self.refine_once = refine_once  # ablation: one subscription per persona per episode
        # Local watchdog + Bayesian reputation over the hosts, off unless asked for: with it
        # the broker judges each publication a host returns and withholds from its match any
        # host whose misbehaviour posterior has reached tau_rep.
        self.watchdog = watchdog
        self.trust_threshold = trust_threshold
        self.gate_min_observations = gate_min_observations
        self.rep_manager = ReputationManager(discount_alpha=rep_discount,
                                             discount_beta=rep_discount)
        self._pending = None           # the publication awaiting a verdict
        self.watchdog_calls = 0
        self.price_in, self.price_out = _prices(model.config.model_name)
        self._sub_vecs = None          # cached subscription embeddings
        self._refined: dict = {}       # persona.name -> refined role (reactive subscription cache)
        self._last_persona = None
        self._base_sha = None          # /testbed HEAD at start, for a robust best-effort diff
        self._blocked = 0              # premature-submit blocks (objective reproduction gate)
        self.multi_candidate = False  # best-of-K editing (off by default)
        self._mc_fires = 0            # cap multi-candidate rounds per task (bounds cost)
        self._repro_attempts = 0      # cap how many rounds we force the Test Runner (avoid loops)
        self._repro_retries = 0       # cap reproduction rewrites when the repro isn't a real gate
        self.raps_log: list = []       # per-round routing decisions
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # ----------------- LLM / embedding (one litellm route for both) -----------------

    def _complete(self, messages, max_tokens=2048, temperature=None):
        """One backbone call, at the decoding temperature shared with the main
        experiments unless the caller asks for another."""
        resp = litellm.completion(
            model=self.model.config.model_name, messages=messages,
            temperature=LLM.DEFAULT_TEMPERATURE if temperature is None else temperature,
            max_tokens=max_tokens, drop_params=True)
        u = resp.usage
        pt, ct = int(u.prompt_tokens), int(u.completion_tokens)
        self.llm_calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct
        return (resp.choices[0].message.content or ""), pt * self.price_in + ct * self.price_out

    def _gen(self, system, user, max_tokens=512):
        return self._complete([{"role": "system", "content": system},
                               {"role": "user", "content": user}], max_tokens)

    def _embed(self, texts):
        resp = litellm.embedding(model=EMBED_MODEL, input=[t.replace("\n", " ")[:8000] for t in texts])
        return [np.asarray(d["embedding"]) for d in resp.data]

    # ----------------- RAPS round pieces -----------------

    def _broker_context(self, k=4, cap=1200):
        """Clean routing context: the real issue + recent COMMAND OUTPUTS (not the personas'
        verbose THOUGHT reasoning, which otherwise pollutes the need and mis-routes to Reviewer)."""
        task = self.extra_template_vars.get("task", "")
        last_cmd = ""
        for m in reversed(self.messages):
            if m.get("role") == "assistant":
                acts = m.get("extra", {}).get("actions", [])
                if acts:
                    last_cmd = acts[0].get("command", "")
                break
        obs = [m for m in self.messages[2:] if m.get("role") == "user"][-k:]
        obs_txt = "\n".join(f"- {str(m.get('content',''))[:cap]}" for m in obs) or "(none yet)"
        return f"ISSUE:\n{task[:1500]}\n\nLAST COMMAND: {last_cmd[:200]}\n\nRECENT OUTPUTS:\n{obs_txt}"

    def _broker_need(self, context):
        need, cost = self._gen(BROKER_SYSTEM, context, max_tokens=40)
        line = (need.strip().splitlines() or [""])[0]
        line = re.sub(r"(?i)^\s*(thought|need|next|action)\s*:\s*", "", line).strip().strip("`\"'")
        return (line or "inspect the repository")[:200], cost

    def _admissible(self):
        """The hosts the broker's reputation admits, and the names it withholds. When the
        gate would leave no host at all the full team is kept, since the episode still has
        to act; the withheld names are recorded either way."""
        if not self.watchdog:
            return list(self.team), []
        admitted, withheld = [], []
        for persona in self.team:
            if self.rep_manager.admits(persona.name, self.trust_threshold,
                                       self.gate_min_observations):
                admitted.append(persona)
            else:
                withheld.append(persona.name)
        return (admitted or list(self.team)), withheld

    def _route(self, need, candidates):
        """Match the need against the admitted candidates' subscriptions and return the
        highest-ranked host together with the full ranking."""
        if self._sub_vecs is None:
            self._sub_vecs = dict(zip((p.name for p in self.team),
                                      self._embed([p.subscription for p in self.team])))
        nv = self._embed([need])[0]
        scored = sorted(((_cos(nv, self._sub_vecs[p.name]), p) for p in candidates),
                        key=lambda x: x[0], reverse=True)
        return scored[0][1], [(round(s, 3), p.name) for s, p in scored]

    # ----------------- local watchdog + first-hand reputation -----------------

    def _last_observation(self, cap=2000):
        """The result the shell returned for the most recent action."""
        for m in reversed(self.messages):
            if m.get("role") == "user":
                return str(m.get("content", ""))[:cap]
        return ""

    def _evaluate_pending(self):
        """Judge the previous round's publication now that its result is in, and record the
        first-hand evidence the gate acts on. The verdict is written back onto that round's
        log entry, since it is only available once the shell has answered."""
        if self._pending is None:
            return 0.0
        sound, cost = self._gen(
            WATCHDOG_SYSTEM,
            f"NEED:\n{self._pending['need']}\n\nACTION:\n{self._pending['command']}\n\n"
            f"RESULT:\n{self._last_observation()}",
            max_tokens=4)
        self.watchdog_calls += 1
        sound = "YES" in sound.upper()
        host = self._pending["persona"]
        self.rep_manager.update_first_hand(host, sound)
        if self.raps_log:
            self.raps_log[-1]["verdict"] = "sound" if sound else "faulty"
            self.raps_log[-1]["misbehaviour"] = round(self.rep_manager.misbehaviour(host), 3)
        self._pending = None
        return cost

    def _reputation(self):
        """The posterior the broker holds about each host it has observed."""
        return {name: round(self.rep_manager.misbehaviour(name), 3)
                for name in sorted(self.rep_manager.REP)}

    def _reactive_subscription(self, persona, need, context):
        user = f"Base role: {persona.guidance}\n\nCurrent need: {need}\n\nContext:\n{context}"
        return self._gen(REFINE_SYSTEM, user, max_tokens=220)

    def _publish(self, persona, refined_role):
        # persona system = hard rules + its refined role + the backticks format contract
        persona_sys = f"{SWE_RULES}\n{refined_role}\n\n{self.config.system_template}"
        messages = [{"role": "system", "content": persona_sys}, *self.messages[1:]]
        content, total = "", 0.0
        for _ in range(2):
            content, cost = self._complete(messages, max_tokens=3000)
            total += cost
            cmds = [a.strip() for a in re.findall(ACTION_REGEX, content, re.DOTALL)]
            if len(cmds) == 1:
                return content, cmds[0], total
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": FORMAT_REMINDER},
            ]
        return content, "cd /testbed && git --no-pager diff", total  # harmless fallback

    # ----------------- multi-candidate repair (RAPS adaptive fan-out + objective select) --------

    def _repro_status(self):
        """None if no canonical reproduction yet; True if it passes (exit 0); False if it fails."""
        out = self.env.execute({"command":
            f"if [ -f {REPRO_PATH} ]; then python {REPRO_PATH} >/dev/null 2>&1; echo RC=$?; "
            "else echo NONE; fi"}).get("output", "")
        if "NONE" in out:
            return None
        return "RC=0" in out

    def _multi_candidate_edit(self, refined_role):
        """Best-of-K: generate up to N_CANDIDATES candidate edits, apply+test each against the
        reproduction in ISOLATION (revert between), then return the one that makes the reproduction
        pass (else the first that applies cleanly) for the normal execute path to apply. Returns
        (content, winning_command, cost)."""
        persona_sys = f"{SWE_RULES}\n{refined_role}\n\n{self.config.system_template}"
        base_msgs = [{"role": "system", "content": persona_sys}, *self.messages[1:]]
        clean = "cd /testbed && git checkout -- . 2>/dev/null || true"
        self.env.execute({"command": clean})
        cands, total = [], 0.0
        for k in range(N_CANDIDATES):
            content, cost = self._complete(base_msgs, max_tokens=3000, temperature=0.0 if k == 0 else 0.7)
            total += cost
            cmds = [a.strip() for a in re.findall(ACTION_REGEX, content, re.DOTALL)]
            if len(cmds) != 1:
                continue
            cmd = cmds[0]
            self.env.execute({"command": clean})
            applied = self.env.execute({"command": cmd}).get("output", "")
            chk = self.env.execute({"command": f"python {REPRO_PATH} >/dev/null 2>&1; echo RC=$?"}).get("output", "")
            diff = self.env.execute({"command": "cd /testbed && git --no-pager diff"}).get("output", "")
            cands.append({"cmd": cmd, "content": content, "passed": "RC=0" in chk,
                          "ok": "OK: applied" in applied, "nonempty": bool(diff.strip())})
            self.env.execute({"command": clean})
            if cands[-1]["passed"]:
                break
        winner = (next((c for c in cands if c["passed"]), None)
                  or next((c for c in cands if c["ok"] and c["nonempty"]), None)
                  or (cands[0] if cands else None))
        if winner is None:
            return "[multi-candidate] no parseable candidate.", "cd /testbed && git --no-pager diff", total
        tag = "PASSES the reproduction" if winner["passed"] else "applies (reproduction not yet green)"
        note = (f"THOUGHT: [multi-candidate] generated {len(cands)} candidate edit(s), tested each "
                f"against {REPRO_PATH} in isolation; selected the one that {tag}. Applying it now.")
        return note, winner["cmd"], total

    def _ensure_base_sha(self):
        """Round-1 setup: inject the robust SEARCH/REPLACE edit tool into the container and record
        /testbed HEAD for the final diff. Runs once."""
        if self._base_sha is None:
            try:
                import base64
                b64 = base64.b64encode(APPLY_EDIT_TOOL.encode()).decode()
                self.env.execute({"command": f"echo {b64} | base64 -d > {EDIT_TOOL_PATH}"})
                out = self.env.execute({"command": "cd /testbed && git rev-parse HEAD"})
                self._base_sha = (out.get("output", "").strip().splitlines() or [""])[0]
            except Exception:
                self._base_sha = ""
        return self._base_sha

    def _final_diff(self):
        """Robust diff vs base: stage everything EXCEPT patch.txt (the agent's scratch file, which
        otherwise pollutes the patch), then diff staged vs base — captures edits + new source files
        + committed changes, while excluding patch.txt and untracked scratch."""
        base = self._ensure_base_sha() or "HEAD"
        out = self.env.execute({"command":
            f"cd /testbed && git add -A -- . ':(exclude)patch.txt' && git --no-pager diff --cached {base}"})
        return out.get("output", "") or ""

    def _best_effort_submit(self):
        """On limits, finalize the current diff instead of an empty patch."""
        diff = self._final_diff()
        raise Submitted({"role": "exit", "content": diff,
                         "extra": {"exit_status": "BestEffortSubmit", "submission": diff}})

    # ----------------- objective reproduction gate (skeleton rule, RAPS core unchanged) ----------

    def _submission_valid(self) -> bool:
        """Gate: require a non-empty /testbed diff AND, if a canonical reproduction exists, that it
        now passes (exit 0). Prevents submitting a mislocated/unverified patch (error-mode #3)."""
        if not self._final_diff().strip():
            return False
        chk = self.env.execute({"command":
            f"if [ -f {REPRO_PATH} ]; then python {REPRO_PATH} >/dev/null 2>&1; echo REPRO_RC=$?; "
            "else echo NO_REPRO; fi"}).get("output", "")
        if "NO_REPRO" in chk:
            return True            # nothing to gate on -> trust the team's submit (a real diff exists)
        return "REPRO_RC=0" in chk

    def step(self):
        """Wrap the round to enforce the reproduction gate before accepting a model submit."""
        try:
            return super().step()
        except Submitted as e:
            msg = e.messages[0] if e.messages else {}
            if msg.get("extra", {}).get("exit_status") == "Submitted" and not self._submission_valid():
                self._blocked += 1
                if self._blocked >= 4:           # give up gating after repeated tries -> finalize diff
                    self._best_effort_submit()
                return self.add_messages({
                    "role": "user",
                    "content": ("<SUBMIT BLOCKED> Not verified: either there is no /testbed source diff "
                                f"or {REPRO_PATH} does not exit 0 yet. Do NOT submit. Run "
                                f"`python {REPRO_PATH}`; if it fails, fix the /testbed source until it "
                                "exits 0; only then submit."),
                    "extra": {"returncode": 1, "submit_blocked": True},
                })
            raise

    # ----------------- the override: one round = one ad-hoc network coordination -----------------

    def query(self):
        start = getattr(self, "_start_time", time.time())
        if (0 < self.config.step_limit <= self.n_calls) or (0 < self.config.cost_limit <= self.cost) \
                or (0 < getattr(self.config, "wall_time_limit_seconds", 0) <= int(time.time() - start)):
            self._best_effort_submit()

        self._ensure_base_sha()  # record /testbed HEAD on round 1, before any edit/commit
        c0 = self._evaluate_pending()   # verdict on what the previous round's host returned
        context = self._broker_context()
        need, c1 = self._broker_need(context)
        candidates, withheld = self._admissible()
        persona, ranking = self._route(need, candidates)
        # RULE (skeleton): a RED reproduction means "fix it" -> force the Patch Author and run
        # best-of-K repair. This reliably engages multi-candidate once a failing test exists (the
        # broker otherwise rarely routes to Patch Author at that moment). Locate/reproduce/verify
        # phases (no repro yet, or repro already green) keep the broker's dynamic content routing.
        # ---- deterministic, CAPPED phase machine (guarantees editing; no Test Runner loops) ----
        admitted = {p.name for p in candidates}

        def P(name):
            """The named host when the gate admits it, otherwise the broker's own choice:
            a scaffold rule may steer the phase but never re-engages a host the broker's
            reputation has withheld."""
            return next((p for p in self.team if p.name == name and name in admitted), persona)
        repro_st = self._repro_status()                  # None = no repro, True = green, False = red
        has_edit = bool(self._final_diff().strip())
        use_mc = False
        if repro_st is False:                             # red reproduction -> FIX it (always)
            persona = P("Patch Author")
            use_mc = self.multi_candidate and self._mc_fires < 5
            if use_mc:
                self._mc_fires += 1
        elif repro_st is None and self.n_calls >= 2 and self._repro_attempts < 3:
            self._repro_attempts += 1                     # try to create a reproduction (capped)
            persona = P("Test Runner")
        elif repro_st is True and not has_edit and self._repro_retries < 1:
            self._repro_retries += 1                      # repro passes with no fix -> rewrite once
            persona = P("Test Runner")
        elif not has_edit and self.n_calls >= 4:
            persona = P("Patch Author")                   # ensure we edit even without a usable repro
        # else: keep the broker's routing (locate early; verify/review/submit once a fix exists)
        # Reactive subscription (§2.6): re-derive against the CURRENT repository state whenever
        # control switches persona; the freshness ablation (refine_once) instead computes each
        # persona's subscription on first selection and reuses it for the rest of the episode.
        fresh = persona.name not in self._refined or \
            (not self.refine_once and persona.name != self._last_persona)
        if fresh:
            refined, c2 = self._reactive_subscription(persona, need, context)
            self._refined[persona.name] = refined
        else:
            refined, c2 = self._refined[persona.name], 0.0
        reselected = persona.name != self._last_persona
        self._last_persona = persona.name
        if use_mc:
            content, command, c3 = self._multi_candidate_edit(refined)
        else:
            content, command, c3 = self._publish(persona, refined)

        round_cost = c0 + c1 + c2 + c3
        self.n_calls += 1
        self.cost += round_cost
        # The role-evolution record behind §2.6: who acted, on what need, with
        # which refined subscription, and whether control re-selected the host. The
        # watchdog's verdict on this round arrives with the next one.
        self.raps_log.append({"round": self.n_calls, "need": need, "routed_to": persona.name,
                              "ranking": ranking, "withheld": withheld,
                              "command": command[:200],
                              "refined_role": refined[:400],
                              "fresh_refine": fresh, "reselected": reselected})
        if self.watchdog:
            self._pending = {"need": need, "persona": persona.name, "command": command}

        msg = self.model.format_message(
            role="assistant", content=content,
            extra={"actions": [{"command": command}], "cost": round_cost,
                   "raps": {"need": need, "persona": persona.name}})
        self.add_messages(msg)
        return msg

    def serialize(self, *extra):
        return super().serialize(
            {"info": {"raps_stats": {
                "rounds": self.n_calls, "llm_calls": self.llm_calls,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "usd": round(self.prompt_tokens * self.price_in + self.completion_tokens * self.price_out, 6),
                "watchdog": self.watchdog, "watchdog_calls": self.watchdog_calls,
                "reputation": self._reputation(),
            }}, "raps_log": self.raps_log}, *extra)
