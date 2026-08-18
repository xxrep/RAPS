from typing import Dict, Set, Tuple
import math


class ReputationManager:
    """A CONFIDANT-style Bayesian reputation system.

    Each record is a Beta pair over one peer's behaviour, in which `a` accumulates
    judgements of sound content and `b` judgements of misbehaviour, faded by `alpha`
    before each update. The score the routing gate acts on is `misbehaviour()`, the
    posterior expectation `b / (a + b)` that the peer publishes faulty content, so a
    larger score means a less reliable peer. The protocol tables state the same
    quantity with the two pseudo-counts named the other way round, as the expectation
    that the misbehaviour count carries of the total; the value is identical.
    """

    #: Expectation of the uninformative Beta(1, 1) prior, held about an unobserved peer.
    PRIOR_EXPECTATION = 0.5

    def __init__(
        self,
        discount_alpha: float = 0.999,   # Discount factor for REP and FH
        discount_beta: float = 0.999,    # Discount factor for TRUST
        merge_weight: float = 0.05,      # Merge weight for second-hand reputation
        deviation_threshold: float = 0.2, # Threshold for deviation test
        trust_threshold: float = 0.5      # Threshold to classify reporter
    ):
        self.FH: Dict[str, Dict[str, float]] = {}     # First-hand FH[i->j]
        self.REP: Dict[str, Dict[str, float]] = {}    # Reputation REP[i->j]
        self.TRUST: Dict[str, Dict[str, float]] = {}  # TRUST[i->reporter]
        self.OBS: Dict[str, int] = {}                 # first-hand observations of j
        self._fresh_peers: Set[str] = set()            # peers whose record moved this round
        self._fresh_reporters: Set[str] = set()        # reporters heard from this round

        self.alpha = discount_alpha
        self.beta = discount_beta
        self.w = merge_weight
        self.d = deviation_threshold
        self.trust_threshold = trust_threshold

    # ---------------- internal utilities ----------------

    def _get(self, table: Dict[str, Dict[str, float]], key: str):
        if key not in table:
            table[key] = {"a": 1.0, "b": 1.0}
        return table[key]

    def _expected(self, rec: Dict[str, float]) -> float:
        return rec["a"] / (rec["a"] + rec["b"])

    def _discount(self, rec: Dict[str, float], gamma: float):
        """Fade both pseudo-counts by gamma, which makes an update x <- gamma*x + s: the
        weight of an observation decays geometrically with the observations that follow it,
        so a peer's posterior is dominated by its recent behaviour."""
        rec["a"] *= gamma
        rec["b"] *= gamma

    # ---------------- first-hand update ------------------

    def update_first_hand(self, target: str, success: bool):
        FH = self._get(self.FH, target)
        REP = self._get(self.REP, target)

        # discount both tables (CONFIDANT allows optional fading)
        self._discount(FH, self.alpha)
        self._discount(REP, self.alpha)

        if success:
            FH["a"] += 1.0
            REP["a"] += 1.0
        else:
            FH["b"] += 1.0
            REP["b"] += 1.0
        self.OBS[target] = self.OBS.get(target, 0) + 1
        self._fresh_peers.add(target)

    # ---------------- second-hand update ------------------

    def update_from_report(self, reporter: str, reports: Dict[str, Tuple[float, float]]):
        """Merge one reporter's first-hand table, `{target_id: (a, b)}`. A testimony that
        agrees with what this agent already holds, judged by the deviation test against the
        pooled standard deviation, raises the reporter's credibility and is merged; from a
        reporter already credible it is merged either way. What enters is the reported
        evidence in excess of the prior, scaled by the merge weight, so second-hand
        testimony moves a posterior less than a first-hand judgement does."""
        T = self._get(self.TRUST, reporter)
        # discount trust explicitly
        self._discount(T, self.beta)
        self._fresh_reporters.add(reporter)

        is_trustworthy = self._expected(T) >= self.trust_threshold

        for target, (aF, bF) in reports.items():
            Fk = {"a": aF, "b": bF}
            Ri = self._get(self.REP, target)
            # discount REP before merging
            self._discount(Ri, self.alpha)
            self._fresh_peers.add(target)

            exp_fk = self._expected(Fk)
            exp_ri = self._expected(Ri)
            var_fk = (Fk["a"] * Fk["b"]) / (((Fk["a"] + Fk["b"]) ** 2) * (Fk["a"] + Fk["b"] + 1.0))
            var_ri = (Ri["a"] * Ri["b"]) / (((Ri["a"] + Ri["b"]) ** 2) * (Ri["a"] + Ri["b"] + 1.0))
            denom = math.sqrt(var_fk + var_ri)
            deviation = abs(exp_fk - exp_ri) / denom if denom > 0.0 else 0.0

            # update trust regardless of trustworthy status
            if deviation < self.d:
                T["a"] += 1.0
            else:
                T["b"] += 1.0

            # merging rule
            if is_trustworthy:
                # always merge
                Ri["a"] += self.w * max(0.0, aF - 1.0)
                Ri["b"] += self.w * max(0.0, bF - 1.0)
            else:
                # untrustworthy: only merge if deviation small
                if deviation < self.d:
                    Ri["a"] += self.w * max(0.0, aF - 1.0)
                    Ri["b"] += self.w * max(0.0, bF - 1.0)

    # ---------------- decay ------------------

    def fade_stale(self):
        """Close a round: fade every record that took no evidence during it, at the same
        rate an update applies. Fading both pseudo-counts leaves the posterior expectation
        where it is and reduces the mass behind it, so a standing judgement keeps its
        verdict while ceding weight to whatever is observed next."""
        for target, rec in self.FH.items():
            if target not in self._fresh_peers:
                self._discount(rec, self.alpha)
        for target, rec in self.REP.items():
            if target not in self._fresh_peers:
                self._discount(rec, self.alpha)
        for reporter, rec in self.TRUST.items():
            if reporter not in self._fresh_reporters:
                self._discount(rec, self.beta)
        self._fresh_peers.clear()
        self._fresh_reporters.clear()

    # ---------------- queries ------------------

    def get_reputation(self, target: str) -> float:
        """Posterior expectation that `target` publishes soundly, the prior expectation
        when it has never been observed. Reading a record never creates one."""
        rec = self.REP.get(target)
        return self._expected(rec) if rec else self.PRIOR_EXPECTATION

    def misbehaviour(self, target: str) -> float:
        """Posterior expectation that `target` publishes faulty content, i.e. the
        complement of `get_reputation`. This is the score the routing gate compares
        against tau_rep and the score reported in the task logs."""
        return 1.0 - self.get_reputation(target)

    def get_trust(self, reporter: str) -> float:
        """Posterior expectation that `reporter`'s testimony agrees with what this agent
        observes, the prior expectation when it has never reported."""
        rec = self.TRUST.get(reporter)
        return self._expected(rec) if rec else self.PRIOR_EXPECTATION

    def first_hand_count(self, target: str) -> int:
        """Watchdog judgements this agent has made of `target`. Zero when the peer has
        never been observed, which is what the routing gate uses to exempt unseen peers."""
        return self.OBS.get(target, 0)

    def admits(self, target: str, threshold: float, min_observations: float = 1.0) -> bool:
        """The routing gate: `target` is admitted while its misbehaviour posterior stays
        beneath `threshold`, and isolated once it reaches it. The comparison is made only
        after `min_observations` first-hand judgements exist, so a peer about which nothing
        has been observed is admitted rather than excluded for want of evidence."""
        if self.first_hand_count(target) < min_observations:
            return True
        return self.misbehaviour(target) < threshold

    def export_first_hand(self) -> Dict[str, Tuple[float, float]]:
        return {k: (v["a"], v["b"]) for k, v in self.FH.items()}
