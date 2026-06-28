from typing import Dict, Tuple, Optional
import math


class ReputationManager:
    """
    A CONFIDANT-style Bayesian reputation system.
    """

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
        """ rec = 1 + (rec - 1)*gamma """
        rec["a"] = 1.0 + (rec["a"] - 1.0) * gamma
        rec["b"] = 1.0 + (rec["b"] - 1.0) * gamma

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

    # ---------------- second-hand update ------------------

    def update_from_report(self, reporter: str, reports: Dict[str, Tuple[float, float]]):
        """
        reports: { target_id: (aF, bF) }
        """
        T = self._get(self.TRUST, reporter)
        # discount trust explicitly
        self._discount(T, self.beta)

        is_trustworthy = self._expected(T) >= self.trust_threshold

        for target, (aF, bF) in reports.items():
            Fk = {"a": aF, "b": bF}
            Ri = self._get(self.REP, target)
            # discount REP before merging
            self._discount(Ri, self.alpha)

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

    def fade_all(self):
        """ Global decay for inactivity. """
        for rec in self.FH.values():
            self._discount(rec, self.alpha)
        for rec in self.REP.values():
            self._discount(rec, self.alpha)
        for rec in self.TRUST.values():
            self._discount(rec, self.beta)

    # ---------------- queries ------------------

    def get_reputation(self, target: str) -> float:
        return self._expected(self._get(self.REP, target))

    def get_trust(self, reporter: str) -> float:
        return self._expected(self._get(self.TRUST, reporter))

    def export_first_hand(self) -> Dict[str, Tuple[float, float]]:
        return {k: (v["a"], v["b"]) for k, v in self.FH.items()}
