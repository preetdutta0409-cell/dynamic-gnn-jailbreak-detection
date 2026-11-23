"""The barrier: the ONLY code path allowed to call mock tools."""
from src.runtime.mock_tools import SUPPORTED_ACTIONS, InvalidActionError


class ActionGate:
    def __init__(self, policy_engine, mock_tools, audit_log: list):
        self._policy = policy_engine
        self._tools = mock_tools
        self._audit = audit_log

    def handle(self, action: dict, prediction: dict | None) -> dict:
        self._validate(action)

        score = prediction.get("risk_score") if prediction else None
        verdict = self._policy.decide(score)

        executed = False
        mock_result = None
        if verdict.decision == "ALLOW":
            mock_result = self._tools.execute(action)
            executed = True

        record = {
            "session_id": action.get("session_id"),
            "risk_score": score,
            "decision": verdict.decision,
            "action_id": action["action_id"],
            "action_type": action["action_type"],
            "action_executed": executed,
            "reason_codes": verdict.reason_codes,
        }
        self._audit.append(record)  # every decision is recorded

        if executed:
            return {**record, "mock_result": mock_result}
        return record

    @staticmethod
    def _validate(action: dict) -> None:
        if not isinstance(action, dict):
            raise InvalidActionError("Action must be a dict")
        if not action.get("action_id"):
            raise InvalidActionError("Action is missing action_id")
        if action.get("action_type") not in SUPPORTED_ACTIONS:
            raise InvalidActionError(
                f"Unsupported action_type: {action.get('action_type')!r}"
            )