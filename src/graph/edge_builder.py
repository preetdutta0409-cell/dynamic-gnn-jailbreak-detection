"""Build metadata-grounded directed edges between event nodes.

The edge builder is deliberately deterministic and metadata-only. It never
infers semantic relationships from content, source identity, or proximity.
Every edge is justified by an explicit field:

* ``NEXT_EVENT``      -- the ``(timestamp, event_id)`` ordering of nodes.
* ``RESPONDS_TO``     -- a node's ``parent_event_id``.
* ``RETRIEVED_FROM``  -- ``metadata.retrieved_context_event_ids``.
* ``REQUESTS_TOOL``   -- ``metadata.requested_by_event_id``.

Nodes produced by :mod:`src.graph.node_builder` preserve the Canonical Event
fields and add ``node_type``. This module also accepts the older/simpler shape
where the node identifier is stored in ``node_id`` and the original event
identifier is stored in ``event_id``.

Edges are emitted as ``{"source": ..., "target": ..., "type": ...}`` records.
An edge is emitted only when both endpoints are present in the supplied node
list. Dangling references are ignored rather than repaired.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, Mapping

EDGE_NEXT_EVENT: Final[str] = "NEXT_EVENT"
EDGE_RESPONDS_TO: Final[str] = "RESPONDS_TO"
EDGE_RETRIEVED_FROM: Final[str] = "RETRIEVED_FROM"
EDGE_REQUESTS_TOOL: Final[str] = "REQUESTS_TOOL"

ALLOWED_EDGE_TYPES: Final[frozenset[str]] = frozenset(
    {
        EDGE_NEXT_EVENT,
        EDGE_RESPONDS_TO,
        EDGE_RETRIEVED_FROM,
        EDGE_REQUESTS_TOOL,
    }
)

_NODE_TYPE_BY_EVENT_TYPE: Final[dict[str, str]] = {
    "user_message": "user",
    "assistant_message": "assistant",
    "retrieved_context": "retrieved_context",
    "tool_call": "tool",
    "tool_result": "tool",
    "proposed_action": "proposed_action",
    "policy_decision": "policy",
}

_RETRIEVED_FROM_SOURCE_TYPES: Final[frozenset[str]] = frozenset(
    {"user", "assistant", "proposed_action", "policy"}
)
_REQUESTS_TOOL_SOURCE_TYPES: Final[frozenset[str]] = frozenset({"assistant"})
_REQUESTS_TOOL_TARGET_TYPES: Final[frozenset[str]] = frozenset(
    {"proposed_action", "tool"}
)

Edge = dict[str, str]
Node = Mapping[str, Any]
RawEvent = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _NodeRecord:
    """A node paired with its raw event and pre-parsed ordering metadata."""

    node: Node
    raw: RawEvent | None
    node_id: str
    event_id: str
    session_id: str | None
    node_type: str | None
    timestamp: datetime


def build_edges(
    nodes: list[dict[str, Any]],
    raw_events: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return the directed, metadata-grounded edges for one session.

    Parameters
    ----------
    nodes:
        Node dictionaries produced by ``node_builder``. Each node must expose a
        non-empty ``node_id`` or ``event_id`` and a timestamp. When ``node_id``
        and ``event_id`` are both present, ``node_id`` is used as the emitted
        graph identifier and ``event_id`` remains available as an alias for
        resolving metadata references.
    raw_events:
        The Canonical Events used to create ``nodes``. Raw events are consulted
        when a node does not itself retain ``parent_event_id`` or the metadata
        fields that justify an edge.

    Returns
    -------
    list[dict[str, str]]
        Deterministically ordered edge records. Each record has exactly the
        keys ``source``, ``target``, and ``type``. The same
        ``(source, type, target)`` triple is emitted at most once.

    Raises
    ------
    TypeError
        If a node or raw event is not a mapping.
    ValueError
        If a node has no identifier, a node identifier is duplicated, two nodes
        claim the same event alias, or a timestamp cannot be parsed.
    """

    node_inputs = list(nodes or [])
    raw_inputs = list(raw_events or [])

    if not node_inputs:
        return []

    for raw_event in raw_inputs:
        if not isinstance(raw_event, Mapping):
            raise TypeError("raw_events must contain mapping objects")

    raw_by_id, raw_by_slot = _index_raw_events(raw_inputs)

    records: list[_NodeRecord] = []
    seen_node_ids: set[str] = set()

    for node in node_inputs:
        if not isinstance(node, Mapping):
            raise TypeError("nodes must contain mapping objects")

        node_id = _node_identifier(node)
        if node_id in seen_node_ids:
            raise ValueError(f"duplicate node identifier: {node_id!r}")
        seen_node_ids.add(node_id)

        raw_event = _find_raw_event(node, node_id, raw_by_id, raw_by_slot)
        timestamp_value = (
            node.get("timestamp")
            if node.get("timestamp") is not None
            else (raw_event.get("timestamp") if raw_event is not None else None)
        )
        timestamp = _parse_timestamp(timestamp_value, node_id=node_id)

        event_id = (
            _identifier(node.get("event_id"))
            or _identifier(_metadata(node).get("event_id"))
            or (
                _identifier(raw_event.get("event_id"))
                if raw_event is not None
                else None
            )
            or node_id
        )

        records.append(
            _NodeRecord(
                node=node,
                raw=raw_event,
                node_id=node_id,
                event_id=event_id,
                session_id=_session_id(node, raw_event),
                node_type=_node_type(node, raw_event),
                timestamp=timestamp,
            )
        )

    records_by_id = {record.node_id: record for record in records}
    alias_to_node = _build_alias_index(records)

    edges: list[Edge] = []
    seen_edges: set[tuple[str, str, str]] = set()

    # NEXT_EVENT: a single chronological path per session, ordered by
    # (timestamp, event_id, node_id). The extra node_id component keeps the
    # order total even if two nodes were supplied with the same event alias.
    nodes_by_session: dict[str | None, list[_NodeRecord]] = defaultdict(list)
    for record in records:
        nodes_by_session[record.session_id].append(record)

    for session_records in nodes_by_session.values():
        session_records.sort(key=_ordering_key)
        for earlier, later in zip(session_records, session_records[1:]):
            _add_edge(
                edges,
                seen_edges,
                earlier.node_id,
                later.node_id,
                EDGE_NEXT_EVENT,
            )

    ordered_records = sorted(records, key=_ordering_key)

    # RESPONDS_TO: child -> explicitly declared parent.
    for record in ordered_records:
        parent_reference = _parent_event_id(record.node, record.raw)
        if parent_reference is None:
            continue

        parent_node_id = alias_to_node.get(parent_reference)
        if parent_node_id is None or parent_node_id == record.node_id:
            continue

        parent = records_by_id[parent_node_id]
        if not _same_session(record, parent):
            continue

        _add_edge(
            edges,
            seen_edges,
            record.node_id,
            parent.node_id,
            EDGE_RESPONDS_TO,
        )

    # RETRIEVED_FROM: consuming event -> explicitly referenced context event.
    for record in ordered_records:
        if (
            record.node_type is not None
            and record.node_type not in _RETRIEVED_FROM_SOURCE_TYPES
        ):
            continue

        for context_reference in _retrieved_context_references(record):
            target_node_id = alias_to_node.get(context_reference)
            if target_node_id is None or target_node_id == record.node_id:
                continue

            target = records_by_id[target_node_id]
            if (
                target.node_type is not None
                and target.node_type != "retrieved_context"
            ):
                continue
            if not _same_session(record, target):
                continue

            _add_edge(
                edges,
                seen_edges,
                record.node_id,
                target.node_id,
                EDGE_RETRIEVED_FROM,
            )

    # REQUESTS_TOOL: requesting assistant event -> requested action/tool event.
    for target in ordered_records:
        requesting_reference = _requested_by_event_id(target.node, target.raw)
        if requesting_reference is None:
            continue

        source_node_id = alias_to_node.get(requesting_reference)
        if source_node_id is None or source_node_id == target.node_id:
            continue

        source = records_by_id[source_node_id]
        if (
            source.node_type is not None
            and source.node_type not in _REQUESTS_TOOL_SOURCE_TYPES
        ):
            continue
        if (
            target.node_type is not None
            and target.node_type not in _REQUESTS_TOOL_TARGET_TYPES
        ):
            continue
        if not _same_session(source, target):
            continue

        _add_edge(
            edges,
            seen_edges,
            source.node_id,
            target.node_id,
            EDGE_REQUESTS_TOOL,
        )

    return edges


def _index_raw_events(
    raw_events: list[RawEvent],
) -> tuple[dict[str, RawEvent], dict[tuple[str | None, str], list[RawEvent]]]:
    """Index raw events by event ID and by ``(session_id, timestamp)``."""

    by_id: dict[str, RawEvent] = {}
    by_slot: dict[tuple[str | None, str], list[RawEvent]] = defaultdict(list)

    for raw_event in raw_events:
        event_id = _identifier(raw_event.get("event_id"))
        if event_id is not None and event_id not in by_id:
            by_id[event_id] = raw_event

        timestamp_token = _timestamp_token(raw_event.get("timestamp"))
        if timestamp_token is not None:
            session_id = _identifier(raw_event.get("session_id"))
            by_slot[(session_id, timestamp_token)].append(raw_event)

    return by_id, by_slot


def _find_raw_event(
    node: Node,
    node_id: str,
    raw_by_id: dict[str, RawEvent],
    raw_by_slot: dict[tuple[str | None, str], list[RawEvent]],
) -> RawEvent | None:
    """Resolve the raw event corresponding to a node without guessing."""

    for candidate in _node_event_aliases(node, node_id):
        raw_event = raw_by_id.get(candidate)
        if raw_event is not None:
            return raw_event

    session_id = _session_id(node, None)
    timestamp_token = _timestamp_token(node.get("timestamp"))
    if session_id is not None and timestamp_token is not None:
        candidates = raw_by_slot.get((session_id, timestamp_token), [])
        if len(candidates) == 1:
            return candidates[0]

    return None


def _build_alias_index(records: list[_NodeRecord]) -> dict[str, str]:
    """Map node IDs and event IDs to one canonical node identifier."""

    alias_to_node: dict[str, str] = {}

    for record in records:
        aliases = _node_event_aliases(record.node, record.node_id)
        aliases.extend((record.event_id, record.node_id))

        if record.raw is not None:
            raw_event_id = _identifier(record.raw.get("event_id"))
            if raw_event_id is not None:
                aliases.append(raw_event_id)

        for alias in dict.fromkeys(aliases):
            existing = alias_to_node.get(alias)
            if existing is not None and existing != record.node_id:
                raise ValueError(
                    "ambiguous node reference "
                    f"{alias!r}: {existing!r} and {record.node_id!r}"
                )
            alias_to_node[alias] = record.node_id

    return alias_to_node


def _add_edge(
    edges: list[Edge],
    seen_edges: set[tuple[str, str, str]],
    source: str,
    target: str,
    edge_type: str,
) -> None:
    """Append an edge once, rejecting self-loops and unknown edge types."""

    if source == target:
        return
    if edge_type not in ALLOWED_EDGE_TYPES:
        return

    identity = (source, edge_type, target)
    if identity in seen_edges:
        return

    seen_edges.add(identity)
    edges.append({"source": source, "target": target, "type": edge_type})


def _ordering_key(record: _NodeRecord) -> tuple[datetime, str, str]:
    """Return the deterministic ordering key for a node record."""

    return (record.timestamp, record.event_id, record.node_id)


def _node_identifier(node: Node) -> str:
    """Return the graph identifier to use for ``node``."""

    identifier = _identifier(node.get("node_id")) or _identifier(
        node.get("event_id")
    )
    if identifier is None:
        raise ValueError("node is missing a non-empty node_id/event_id")
    return identifier


def _node_event_aliases(node: Node, node_id: str) -> list[str]:
    """Return identifiers that may be used to reference ``node``."""

    aliases = [node_id]

    for key in ("node_id", "event_id"):
        value = _identifier(node.get(key))
        if value is not None:
            aliases.append(value)

    metadata = _metadata(node)
    for key in ("node_id", "event_id"):
        value = _identifier(metadata.get(key))
        if value is not None:
            aliases.append(value)

    return list(dict.fromkeys(aliases))


def _metadata(record: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a record's metadata mapping, or an empty mapping."""

    if record is None:
        return {}

    metadata = record.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _identifier(value: Any) -> str | None:
    """Return a non-empty string identifier or ``None``."""

    if isinstance(value, str) and value:
        return value
    return None


def _timestamp_token(value: Any) -> str | None:
    """Return a stable token used to match a node to a raw event slot."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str) and value:
        return value
    return None


def _parse_timestamp(value: Any, *, node_id: str) -> datetime:
    """Parse an ISO 8601 timestamp into a UTC ``datetime``.

    Canonical Events use timezone-aware UTC strings. Naive inputs are treated
    as UTC so that the builder remains usable with test data while still
    producing a total, timezone-safe ordering.
    """

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                f"invalid timestamp for node {node_id!r}: {value!r}"
            ) from exc
    else:
        raise ValueError(
            f"node {node_id!r} is missing a parseable timestamp: {value!r}"
        )

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _session_id(
    node: Node,
    raw_event: RawEvent | None,
) -> str | None:
    """Return the node's session ID, falling back to the raw event."""

    for record in (node, raw_event):
        if record is None:
            continue

        session_id = _identifier(record.get("session_id"))
        if session_id is not None:
            return session_id

        session_id = _identifier(_metadata(record).get("session_id"))
        if session_id is not None:
            return session_id

    return None


def _node_type(node: Node, raw_event: RawEvent | None) -> str | None:
    """Return the documented node type when it can be determined."""

    node_type = _identifier(node.get("node_type"))
    if node_type is not None:
        return node_type

    event_type = _identifier(node.get("event_type"))
    if event_type is None and raw_event is not None:
        event_type = _identifier(raw_event.get("event_type"))

    if event_type is None:
        return None

    return _NODE_TYPE_BY_EVENT_TYPE.get(event_type)


def _parent_event_id(node: Node, raw_event: RawEvent | None) -> str | None:
    """Return the explicit parent reference for a node."""

    for record in (node, raw_event):
        if record is None:
            continue

        parent_event_id = _identifier(record.get("parent_event_id"))
        if parent_event_id is not None:
            return parent_event_id

        parent_event_id = _identifier(
            _metadata(record).get("parent_event_id")
        )
        if parent_event_id is not None:
            return parent_event_id

    return None


def _requested_by_event_id(
    node: Node,
    raw_event: RawEvent | None,
) -> str | None:
    """Return the explicit assistant request reference for an action node."""

    for record in (node, raw_event):
        if record is None:
            continue

        requested_by = _identifier(record.get("requested_by_event_id"))
        if requested_by is not None:
            return requested_by

        requested_by = _identifier(
            _metadata(record).get("requested_by_event_id")
        )
        if requested_by is not None:
            return requested_by

    return None


def _retrieved_context_references(record: _NodeRecord) -> set[str]:
    """Return all explicit retrieved-context event references for a node."""

    references: set[str] = set()

    for source in (record.node, record.raw):
        if source is None:
            continue

        value = _metadata(source).get("retrieved_context_event_ids")
        if isinstance(value, (list, tuple, set, frozenset)):
            for candidate in value:
                reference = _identifier(candidate)
                if reference is not None:
                    references.add(reference)

    return references


def _same_session(left: _NodeRecord, right: _NodeRecord) -> bool:
    """Return whether two records may be connected under session scoping."""

    if left.session_id is None or right.session_id is None:
        return True
    return left.session_id == right.session_id


__all__ = [
    "ALLOWED_EDGE_TYPES",
    "EDGE_NEXT_EVENT",
    "EDGE_REQUESTS_TOOL",
    "EDGE_RESPONDS_TO",
    "EDGE_RETRIEVED_FROM",
    "build_edges",
]
