import pytest
from src.runtime.policy_engine import PolicyEngine


@pytest.mark.parametrize("score,expected", [
    (0.0, "ALLOW"), (0.29, "ALLOW"),
    (0.30, "HOLD"), (0.5, "HOLD"), (0.69, "HOLD"),
    (0.70, "BLOCK"), (0.91, "BLOCK"), (1.0, "BLOCK"),
])
def test_thresholds(score, expected):
    assert PolicyEngine().decide(score).decision == expected


@pytest.mark.parametrize("bad", [None, float("nan"), "0.1", -0.1, 1.5, True, float("inf")])
def test_bad_scores_hold(bad):
    result = PolicyEngine().decide(bad)
    assert result.decision == "HOLD"
    assert "INVALID_OR_MISSING_RISK_SCORE" in result.reason_codes


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        PolicyEngine(allow_below=0.8, block_at=0.2)