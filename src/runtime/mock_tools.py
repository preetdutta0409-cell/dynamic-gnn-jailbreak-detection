"""Harmless simulated tools. No network, files, emails or databases are touched."""
from datetime import datetime, timezone

SUPPORTED_ACTIONS = {"mock_search", "mock_database_query", "send_mock_email"}


class InvalidActionError(ValueError):
    """Raised when a proposed action is malformed or of an unknown type."""


class MockTools:
    def __init__(self):
        self.effects: list[dict] = []  # record of simulated effects

    def execute(self, action: dict) -> dict:
        action_type = action.get("action_type")
        if action_type not in SUPPORTED_ACTIONS:
            raise InvalidActionError(f"Unsupported action_type: {action_type!r}")
        effect = {
            "action_id": action.get("action_id"),
            "action_type": action_type,
            "arguments": action.get("arguments", {}),
            "simulated": True,
            "executed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.effects.append(effect)
        return effect