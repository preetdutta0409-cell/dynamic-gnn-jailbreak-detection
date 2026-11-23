import math
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class PolicyDecision:
    decision: str  # "ALLOW" | "HOLD" | "BLOCK"
    reason_codes: list[str]


class PolicyEngine:
    def __init__(self, allow_below: float = 0.30, block_at: float = 0.70):
        if not 0.0 <= allow_below <= block_at <= 1.0:
            raise ValueError("Need 0 <= allow_below <= block_at <= 1")
        self.allow_below = allow_below
        self.block_at = block_at

    @classmethod
    def from_config(cls, path: str = "configs/runtime_config.yaml") -> "PolicyEngine":
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["policy"]
        return cls(cfg["allow_below"], cfg["block_at"])

    def decide(self, risk_score) -> PolicyDecision:
        # Fail safe: if we can't trust the score, HOLD (never ALLOW).
        if (
            isinstance(risk_score, bool)
            or not isinstance(risk_score, (int, float))
            or math.isnan(risk_score)
            or not 0.0 <= risk_score <= 1.0
        ):
            return PolicyDecision("HOLD", ["INVALID_OR_MISSING_RISK_SCORE"])
        if risk_score < self.allow_below:
            return PolicyDecision("ALLOW", ["RISK_BELOW_ALLOW_THRESHOLD"])
        if risk_score < self.block_at:
            return PolicyDecision("HOLD", ["RISK_IN_REVIEW_BAND"])
        return PolicyDecision("BLOCK", ["RISK_THRESHOLD_EXCEEDED"])