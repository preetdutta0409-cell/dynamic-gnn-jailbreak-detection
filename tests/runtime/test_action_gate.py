import pytest
from src.runtime.action_gate import ActionGate
from src.runtime.mock_tools import InvalidActionError, MockTools
from src.runtime.policy_engine import PolicyEngine


def make_gate():
    tools, audit = MockTools(), []
    return ActionGate(PolicyEngine(), tools, audit), tools, audit


ACTION = {"action_id": "A001", "action_type": "send_mock_email", "session_id": "S001"}


def test_allow_executes():
    gate, tools, _ = make_gate()
    result = gate.handle(ACTION, {"risk_score": 0.1})
    assert result["decision"] == "ALLOW"
    assert result["action_executed"] is True
    assert len(tools.effects) == 1


@pytest.mark.parametrize("score,decision", [(0.5, "HOLD"), (0.91, "BLOCK")])
def test_hold_and_block_never_execute(score, decision):
    gate, tools, _ = make_gate()
    result = gate.handle(ACTION, {"risk_score": score})
    assert result["decision"] == decision
    assert result["action_executed"] is False
    assert tools.effects == []


def test_missing_prediction_holds():
    gate, tools, _ = make_gate()
    result = gate.handle(ACTION, None)
    assert result["decision"] == "HOLD"
    assert tools.effects == []


def test_unknown_action_rejected():
    gate, tools, _ = make_gate()
    with pytest.raises(InvalidActionError):
        gate.handle({"action_id": "A9", "action_type": "delete_everything"}, {"risk_score": 0.0})
    assert tools.effects == []


def test_every_decision_is_audited():
    gate, _, audit = make_gate()
    gate.handle(ACTION, {"risk_score": 0.9})
    assert len(audit) == 1 and audit[0]["decision"] == "BLOCK"