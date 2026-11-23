"""Validate dataset events and manifest labels against the project contract."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence


REQUIRED_EVENT_KEYS = frozenset(
    {
        "event_id",
        "session_id",
        "timestamp",
        "event_type",
        "source_type",
        "source_id",
        "content",
        "parent_event_id",
        "trust_level",
        "metadata",
    }
)
ALLOWED_EVENT_TYPES = frozenset(
    {
        "user_message",
        "assistant_message",
        "retrieved_context",
        "proposed_action",
        "tool_call",
        "tool_result",
        "policy_decision",
    }
)
ALLOWED_LABELS = frozenset({"SAFE", "MALICIOUS", "AMBIGUOUS"})
ALLOWED_TRUST_LEVELS = frozenset({"trusted", "untrusted"})
ALLOWED_REVIEW_STATUSES = frozenset(
    {
        "UNREVIEWED",
        "REVIEWED_AGREED",
        "REVIEWED_RESOLVED",
        "NEEDS_ADJUDICATION",
    }
)
REVIEWED_STATUSES = frozenset({"REVIEWED_AGREED", "REVIEWED_RESOLVED"})
REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "session_id",
        "scenario_id",
        "attack_type",
        "session_label",
        "session_rationale",
        "session_review_status",
        "actions",
    }
)
REQUIRED_ACTION_KEYS = frozenset(
    {"event_id", "action_label", "action_rationale", "review_status"}
)


class DatasetValidationError(ValueError):
    """Raised when one or more dataset contract checks fail."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_allowed(value: Any, allowed: frozenset[str]) -> bool:
    return isinstance(value, str) and value in allowed


def _utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return None
    return parsed


def validate_dataset(
    events: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    *,
    require_reviewed: bool = False,
) -> bool:
    """Validate identifiers, event links, labels, timestamps, and split metadata.

    The canonical event record remains exactly ten top-level keys. scenario_id
    is repeated in event metadata and is authoritative at session level in the
    manifest. When require_reviewed is True, unreviewed or unresolved binary
    labels are rejected for training.
    """
    issues: list[str] = []
    event_by_id: dict[str, Mapping[str, Any]] = {}
    event_times: dict[str, datetime] = {}
    last_timestamp_by_session: dict[str, datetime] = {}
    session_ids: set[str] = set()
    action_event_ids: set[str] = set()

    if not events:
        issues.append("events file is empty")
    if not manifest:
        issues.append("manifest file is empty")

    for index, event in enumerate(events, start=1):
        label = f"event record {index}"
        if not isinstance(event, Mapping):
            issues.append(f"{label} must be an object")
            continue

        keys = set(event)
        missing = REQUIRED_EVENT_KEYS - keys
        extra = keys - REQUIRED_EVENT_KEYS
        if missing:
            issues.append(f"{label} is missing keys: {sorted(missing)}")
        if extra:
            issues.append(f"{label} has non-canonical top-level keys: {sorted(extra)}")

        event_id = event.get("event_id")
        session_id = event.get("session_id")
        if not _non_empty_string(event_id):
            issues.append(f"{label} event_id must be a non-empty string")
        elif event_id in event_by_id:
            issues.append(f"duplicate event_id: {event_id}")
        else:
            event_by_id[event_id] = event

        if not _non_empty_string(session_id):
            issues.append(f"{label} session_id must be a non-empty string")
        else:
            session_ids.add(session_id)

        if not _is_allowed(event.get("event_type"), ALLOWED_EVENT_TYPES):
            issues.append(
                f"{label} event_type is not supported: {event.get('event_type')!r}"
            )
        if not _non_empty_string(event.get("source_type")):
            issues.append(f"{label} source_type must be a non-empty string")
        source_id = event.get("source_id")
        if source_id is not None and not _non_empty_string(source_id):
            issues.append(f"{label} source_id must be a non-empty string or null")
        if not isinstance(event.get("content"), str) or not event.get("content", "").strip():
            issues.append(f"{label} content must be a non-empty string")
        if not _is_allowed(event.get("trust_level"), ALLOWED_TRUST_LEVELS):
            issues.append(f"{label} trust_level must be trusted or untrusted")

        timestamp = _utc_timestamp(event.get("timestamp"))
        if timestamp is None:
            issues.append(f"{label} timestamp must be valid UTC ISO 8601 ending in Z")
        else:
            if _non_empty_string(event_id):
                event_times[event_id] = timestamp
            if _non_empty_string(session_id):
                previous_timestamp = last_timestamp_by_session.get(session_id)
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    issues.append(
                        f"{label} timestamp is out of chronological order in session {session_id!r}"
                    )
                last_timestamp_by_session[session_id] = timestamp

        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            issues.append(f"{label} metadata must be an object")
            continue
        scenario_id = metadata.get("scenario_id")
        if not _non_empty_string(scenario_id):
            issues.append(f"{label} metadata.scenario_id must be a non-empty string")

        parent_id = event.get("parent_event_id")
        if parent_id is not None and not _non_empty_string(parent_id):
            issues.append(f"{label} parent_event_id must be a string or null")

        retrieved_ids = metadata.get("retrieved_context_event_ids", [])
        if not isinstance(retrieved_ids, list):
            issues.append(f"{label} retrieved_context_event_ids must be a list")
        else:
            for retrieved_id in retrieved_ids:
                if not _non_empty_string(retrieved_id):
                    issues.append(f"{label} has an invalid retrieved context reference")

        if event.get("event_type") == "proposed_action":
            if _non_empty_string(event_id):
                action_event_ids.add(event_id)
            requester_id = metadata.get("requested_by_event_id")
            if not _non_empty_string(requester_id):
                issues.append(
                    f"{label} proposed_action requires metadata.requested_by_event_id"
                )

    manifest_by_session: dict[str, Mapping[str, Any]] = {}
    labelled_action_ids: Counter[str] = Counter()
    labelled_action_sessions: dict[str, str] = {}

    for index, entry in enumerate(manifest, start=1):
        label = f"manifest record {index}"
        if not isinstance(entry, Mapping):
            issues.append(f"{label} must be an object")
            continue

        missing = REQUIRED_MANIFEST_KEYS - set(entry)
        extra = set(entry) - REQUIRED_MANIFEST_KEYS
        if missing:
            issues.append(f"{label} is missing keys: {sorted(missing)}")
        if extra:
            issues.append(f"{label} has unsupported keys: {sorted(extra)}")

        session_id = entry.get("session_id")
        scenario_id = entry.get("scenario_id")
        if not _non_empty_string(session_id):
            issues.append(f"{label} session_id must be a non-empty string")
        elif session_id in manifest_by_session:
            issues.append(f"duplicate manifest session_id: {session_id}")
        else:
            manifest_by_session[session_id] = entry

        if not _non_empty_string(scenario_id):
            issues.append(f"{label} scenario_id must be a non-empty string")
        if not _non_empty_string(entry.get("attack_type")):
            issues.append(f"{label} attack_type must be a non-empty string")
        if not _is_allowed(entry.get("session_label"), ALLOWED_LABELS):
            issues.append(f"{label} has unsupported session_label")
        session_status = entry.get("session_review_status")
        if not _is_allowed(session_status, ALLOWED_REVIEW_STATUSES):
            issues.append(f"{label} has unsupported session_review_status")
        if (
            entry.get("session_label") == "AMBIGUOUS"
            and session_status not in {"NEEDS_ADJUDICATION", "REVIEWED_RESOLVED"}
        ):
            issues.append(f"{label} ambiguous session requires adjudication status")
        if not isinstance(entry.get("session_rationale"), str) or not entry.get(
            "session_rationale", ""
        ).strip():
            issues.append(f"{label} requires a non-empty session_rationale")

        actions = entry.get("actions")
        if not isinstance(actions, list):
            issues.append(f"{label} actions must be a list")
            continue
        for action_index, action in enumerate(actions, start=1):
            action_label = f"{label} action {action_index}"
            if not isinstance(action, Mapping):
                issues.append(f"{action_label} must be an object")
                continue
            action_missing = REQUIRED_ACTION_KEYS - set(action)
            action_extra = set(action) - REQUIRED_ACTION_KEYS
            if action_missing:
                issues.append(
                    f"{action_label} is missing keys: {sorted(action_missing)}"
                )
            if action_extra:
                issues.append(
                    f"{action_label} has unsupported keys: {sorted(action_extra)}"
                )
            event_id = action.get("event_id")
            if not _non_empty_string(event_id):
                issues.append(f"{action_label} event_id must be a non-empty string")
                continue
            labelled_action_ids[event_id] += 1
            if _non_empty_string(session_id):
                labelled_action_sessions[event_id] = session_id
            label_value = action.get("action_label")
            review_status = action.get("review_status")
            if not _is_allowed(label_value, ALLOWED_LABELS):
                issues.append(f"{action_label} has unsupported action_label")
            if not _is_allowed(review_status, ALLOWED_REVIEW_STATUSES):
                issues.append(f"{action_label} has unsupported review_status")
            if (
                label_value == "AMBIGUOUS"
                and review_status not in {"NEEDS_ADJUDICATION", "REVIEWED_RESOLVED"}
            ):
                issues.append(f"{action_label} ambiguous label requires adjudication status")
            if not isinstance(action.get("action_rationale"), str) or not action.get(
                "action_rationale", ""
            ).strip():
                issues.append(f"{action_label} requires a non-empty action_rationale")
            if require_reviewed and label_value in {"SAFE", "MALICIOUS"}:
                if review_status not in REVIEWED_STATUSES:
                    issues.append(
                        f"{action_label} is not independently reviewed for binary training"
                    )

    for event_id in action_event_ids:
        count = labelled_action_ids[event_id]
        if count != 1:
            issues.append(
                f"proposed_action event {event_id!r} must have exactly one manifest label; found {count}"
            )
    for event_id in labelled_action_ids:
        if event_id not in action_event_ids:
            issues.append(
                f"manifest action {event_id!r} does not reference a proposed_action event"
            )
        else:
            action_event = event_by_id.get(event_id)
            manifest_session = labelled_action_sessions.get(event_id)
            if (
                action_event is not None
                and manifest_session is not None
                and action_event.get("session_id") != manifest_session
            ):
                issues.append(
                    f"manifest action {event_id!r} is attached to the wrong session"
                )

    if session_ids != set(manifest_by_session):
        missing_manifest = sorted(session_ids - set(manifest_by_session))
        missing_events = sorted(set(manifest_by_session) - session_ids)
        if missing_manifest:
            issues.append(f"sessions missing manifest entries: {missing_manifest}")
        if missing_events:
            issues.append(f"manifest sessions missing events: {missing_events}")

    for event_id, event in event_by_id.items():
        session_id = event.get("session_id")
        entry = manifest_by_session.get(session_id)
        if entry is None:
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        if metadata.get("scenario_id") != entry.get("scenario_id"):
            issues.append(
                f"event {event_id!r} scenario_id does not match its session manifest"
            )
        parent_id = event.get("parent_event_id")
        if _non_empty_string(parent_id):
            parent = event_by_id.get(parent_id)
            if parent is None:
                issues.append(f"event {event_id!r} references missing parent {parent_id!r}")
            elif parent.get("session_id") != session_id:
                issues.append(f"event {event_id!r} parent belongs to another session")
            elif (
                event_id in event_times
                and parent_id in event_times
                and event_times[parent_id] > event_times[event_id]
            ):
                issues.append(f"event {event_id!r} occurs before its parent")

        retrieved_ids = metadata.get("retrieved_context_event_ids", [])
        if isinstance(retrieved_ids, list):
            for retrieved_id in retrieved_ids:
                if not _non_empty_string(retrieved_id):
                    continue
                context_event = event_by_id.get(retrieved_id)
                if context_event is None:
                    issues.append(
                        f"event {event_id!r} references missing retrieved context {retrieved_id!r}"
                    )
                elif context_event.get("session_id") != session_id:
                    issues.append(
                        f"event {event_id!r} retrieved context belongs to another session"
                    )
                elif context_event.get("event_type") != "retrieved_context":
                    issues.append(
                        f"event {event_id!r} retrieval reference is not retrieved_context"
                    )

        if event.get("event_type") == "proposed_action":
            metadata = event.get("metadata")
            requester_id = metadata.get("requested_by_event_id") if isinstance(metadata, Mapping) else None
            if not _non_empty_string(requester_id):
                continue
            requester = event_by_id.get(requester_id)
            if requester is None:
                issues.append(
                    f"proposed_action {event_id!r} references missing requester {requester_id!r}"
                )
            elif requester.get("session_id") != session_id or requester.get("event_type") != "assistant_message":
                issues.append(
                    f"proposed_action {event_id!r} requester must be an assistant_message in the same session"
                )

    if issues:
        raise DatasetValidationError(issues)
    return True


def trainable_action_ids(manifest: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return reviewed SAFE/MALICIOUS action IDs eligible for binary training."""
    return {
        action["event_id"]
        for entry in manifest
        for action in entry.get("actions", [])
        if action.get("action_label") in {"SAFE", "MALICIOUS"}
        and action.get("review_status") in REVIEWED_STATUSES
    }
