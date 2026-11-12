"""Assemble a single-session graph from Canonical Events.

This module is the Phase 4 entry point for graph construction. It is
deliberately storage-agnostic: the returned graph is a pure Python dictionary
with the following shape::

    {
        "session_id": str,
        "nodes": list[dict],
        "edges": list[dict],
    }

Node creation is delegated to :func:`src.graph.node_builder.build_node` and edge
creation is delegated to :func:`src.graph.edge_builder.build_edges`. The builder
never mutates caller-owned event dictionaries and never infers relationships
from event content.

Phase 2 is integrated through a compatibility fallback. When
``src.graph.node_builder.build_node`` is available it is always used. When the
module is present but the function has not landed yet, this module materialises
the documented node record locally (the original Canonical Event plus
``node_id``/``node_type``) so that Phase 4 remains runnable in isolation. The
fallback is schema-compatible and is never used to bypass Phase 2 behaviour
once ``build_node`` exists.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Final, Mapping

from src.graph.edge_builder import build_edges

try:
    from src.graph.node_builder import build_node as _phase2_build_node
except ImportError:  # Phase 2 may not be merged into this working tree yet.
    _phase2_build_node = None


logger = logging.getLogger(__name__)

NODE_TYPE_BY_EVENT_TYPE: Final[dict[str, str]] = {
    "user_message": "user",
    "assistant_message": "assistant",
    "retrieved_context": "retrieved_context",
    "tool_call": "tool",
    "tool_result": "tool",
    "proposed_action": "proposed_action",
    "policy_decision": "policy",
}

if _phase2_build_node is None:
    logger.warning(
        "src.graph.node_builder.build_node is unavailable; using the "
        "schema-compatible node materialiser from graph_builder."
    )


def build_session_graph(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the graph dictionary for one session.

    Parameters
    ----------
    events:
        Raw Canonical Events belonging to a single session. The events are
        treated as read-only. Their original order is preserved in the node
        list; the edge builder applies the schema's deterministic ordering.

    Returns
    -------
    dict[str, Any]
        A graph with exactly the keys ``session_id``, ``nodes``, and ``edges``.

    Raises
    ------
    TypeError
        If ``events`` is not a list or contains a non-mapping event.
    ValueError
        If ``events`` is empty, no event supplies a non-empty ``session_id``,
        or a node cannot be materialised under the closed event-type mapping.
    """

    if not isinstance(events, list):
        message = "events must be a list of Canonical Event dictionaries"
        logger.error(message)
        raise TypeError(message)

    if not events:
        message = "cannot build a session graph from an empty event list"
        logger.error(message)
        raise ValueError(message)

    event_list = list(events)

    for index, event in enumerate(event_list):
        if not isinstance(event, Mapping):
            message = f"event at index {index} must be a mapping"
            logger.error(message)
            raise TypeError(message)

    session_id = _extract_session_id(event_list)

    nodes = [_materialise_node(event) for event in event_list]
    edges = build_edges(nodes, event_list)

    return {
        "session_id": session_id,
        "nodes": nodes,
        "edges": edges,
    }


def _extract_session_id(events: list[Mapping[str, Any]]) -> str:
    """Return the first non-empty ``session_id`` from ``events``.

    The builder assumes all events belong to one session. Cross-session
    consistency is enforced by :mod:`src.graph.graph_validator`, not silently
    repaired here.
    """

    for event in events:
        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id

    message = "events must contain at least one non-empty session_id"
    logger.error(message)
    raise ValueError(message)


def _materialise_node(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return one node dictionary without mutating ``event``."""

    payload = deepcopy(dict(event))
    builder = _phase2_build_node or _fallback_build_node
    node = builder(payload)

    if not isinstance(node, dict):
        message = "build_node must return a dictionary"
        logger.error(message)
        raise TypeError(message)

    if not _is_identifier(node.get("node_id")):
        node["node_id"] = _required_event_id(event)

    if not _is_identifier(node.get("node_type")):
        node["node_type"] = _node_type_for_event(event)

    return node


def _fallback_build_node(event: dict[str, Any]) -> dict[str, Any]:
    """Materialise the documented node record when Phase 2 is unavailable."""

    node = deepcopy(event)
    node["node_id"] = _required_event_id(event)
    node["node_type"] = _node_type_for_event(event)
    return node


def _required_event_id(event: Mapping[str, Any]) -> str:
    """Return a non-empty ``event_id`` or raise a schema error."""

    event_id = event.get("event_id")
    if not _is_identifier(event_id):
        message = "event is missing a non-empty event_id"
        logger.error(message)
        raise ValueError(message)
    return event_id


def _node_type_for_event(event: Mapping[str, Any]) -> str:
    """Map ``event_type`` through the closed schema mapping."""

    event_type = event.get("event_type")
    node_type = (
        NODE_TYPE_BY_EVENT_TYPE.get(event_type)
        if isinstance(event_type, str)
        else None
    )

    if node_type is None:
        event_id = event.get("event_id")
        message = (
            f"unknown event_type {event_type!r} for event {event_id!r}; "
            "the event-type to node-type mapping is closed"
        )
        logger.error(message)
        raise ValueError(message)

    return node_type


def _is_identifier(value: Any) -> bool:
    """Return whether ``value`` is a non-empty string identifier."""

    return isinstance(value, str) and bool(value)


__all__ = [
    "NODE_TYPE_BY_EVENT_TYPE",
    "build_session_graph",
]
