"""Validate the structural integrity of a Phase 4 session graph.

The validator operates on the pure-Python graph dictionary produced by
:func:`src.graph.graph_builder.build_session_graph`::

    {
        "session_id": str,
        "nodes": list[dict],
        "edges": list[dict],
    }

It enforces the graph contract in ``docs/graph_schema.md``:

* node identifiers are unique;
* every edge endpoint exists in the node list;
* every node belongs to the graph's root session;
* node types and edge types belong to their closed allow-lists;
* self-loops are rejected.

Validation never mutates the graph. All detected violations are logged with the
standard :mod:`logging` module before a :class:`GraphValidationError` is raised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final, Iterable, Mapping

logger = logging.getLogger(__name__)

ALLOWED_NODE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "user",
        "assistant",
        "retrieved_context",
        "tool",
        "proposed_action",
        "policy",
    }
)

ALLOWED_EDGE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "NEXT_EVENT",
        "RESPONDS_TO",
        "RETRIEVED_FROM",
        "REQUESTS_TOOL",
    }
)

RESTRICTED_EDGE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "INFLUENCES",
        "CONFLICTS_WITH",
    }
)

REQUIRED_GRAPH_KEYS: Final[tuple[str, ...]] = (
    "session_id",
    "nodes",
    "edges",
)


@dataclass(frozen=True, slots=True)
class GraphValidationIssue:
    """One detected schema violation."""

    code: str
    message: str


class GraphValidationError(ValueError):
    """Raised when a graph violates one or more structural invariants.

    ``issues`` preserves every detected violation so callers can log, display,
    or test individual schema error codes without re-running validation.
    """

    def __init__(self, issues: Iterable[GraphValidationIssue]) -> None:
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError(
                "GraphValidationError requires at least one validation issue"
            )
        super().__init__(
            "; ".join(
                f"{issue.code}: {issue.message}" for issue in self.issues
            )
        )


def validate_graph(graph: dict[str, Any]) -> bool:
    """Validate ``graph`` and return ``True`` when it is structurally legal.

    Raises
    ------
    GraphValidationError
        Subclass of :class:`ValueError` containing every detected violation.
        Each message is prefixed with the relevant ``GRAPH_*`` schema error
        code defined in ``docs/graph_schema.md``.
    """

    issues = _collect_issues(graph)

    if issues:
        for issue in issues:
            logger.error(
                "Graph validation failed [%s]: %s",
                issue.code,
                issue.message,
            )
        raise GraphValidationError(issues)

    return True


def _collect_issues(graph: Any) -> list[GraphValidationIssue]:
    """Return every structural violation found in ``graph``."""

    issues: list[GraphValidationIssue] = []

    if not isinstance(graph, Mapping):
        issues.append(
            GraphValidationIssue(
                "GRAPH_INVALID_GRAPH",
                "graph must be a dictionary with session_id, nodes, and edges",
            )
        )
        return issues

    missing_keys = [
        key for key in REQUIRED_GRAPH_KEYS if key not in graph
    ]
    if missing_keys:
        issues.append(
            GraphValidationIssue(
                "GRAPH_INVALID_GRAPH",
                f"graph is missing required keys: {sorted(missing_keys)}",
            )
        )
        return issues

    session_id = graph.get("session_id")
    if not _is_identifier(session_id):
        issues.append(
            GraphValidationIssue(
                "GRAPH_INVALID_SESSION_ID",
                f"graph session_id must be a non-empty string, got {session_id!r}",
            )
        )
        root_session_id: str | None = None
    else:
        root_session_id = session_id

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        issues.append(
            GraphValidationIssue(
                "GRAPH_INVALID_GRAPH",
                "graph nodes must be a list",
            )
        )
        nodes = []

    edges = graph.get("edges")
    if not isinstance(edges, list):
        issues.append(
            GraphValidationIssue(
                "GRAPH_INVALID_GRAPH",
                "graph edges must be a list",
            )
        )
        edges = []

    node_ids = _validate_nodes(nodes, root_session_id, issues)
    _validate_edges(edges, node_ids, issues)

    return issues


def _validate_nodes(
    nodes: list[Any],
    root_session_id: str | None,
    issues: list[GraphValidationIssue],
) -> set[str]:
    """Validate nodes and return the set of node identifiers."""

    node_ids: set[str] = set()

    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            issues.append(
                GraphValidationIssue(
                    "GRAPH_INVALID_NODE",
                    f"node at index {index} must be a mapping",
                )
            )
            continue

        node_id = node.get("node_id")
        if not _is_identifier(node_id):
            issues.append(
                GraphValidationIssue(
                    "GRAPH_INVALID_NODE_ID",
                    f"node at index {index} has no non-empty node_id",
                )
            )
            continue

        if node_id in node_ids:
            issues.append(
                GraphValidationIssue(
                    "GRAPH_DUPLICATE_EVENT_ID",
                    f"duplicate node_id {node_id!r} at index {index}",
                )
            )
        else:
            node_ids.add(node_id)

        node_type = node.get("node_type")
        if node_type not in ALLOWED_NODE_TYPES:
            issues.append(
                GraphValidationIssue(
                    "GRAPH_UNKNOWN_NODE_TYPE",
                    f"node {node_id!r} has unsupported node_type {node_type!r}",
                )
            )

        node_session_id = node.get("session_id")
        if (
            root_session_id is not None
            and node_session_id != root_session_id
        ):
            issues.append(
                GraphValidationIssue(
                    "GRAPH_INVALID_SESSION_ID",
                    f"node {node_id!r} belongs to session "
                    f"{node_session_id!r}, expected {root_session_id!r}",
                )
            )

    return node_ids


def _validate_edges(
    edges: list[Any],
    node_ids: set[str],
    issues: list[GraphValidationIssue],
) -> None:
    """Validate edge endpoints, edge types, and self-loops."""

    for index, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            issues.append(
                GraphValidationIssue(
                    "GRAPH_INVALID_EDGE",
                    f"edge at index {index} must be a mapping",
                )
            )
            continue

        edge_type = edge.get("type")
        if edge_type in RESTRICTED_EDGE_TYPES:
            issues.append(
                GraphValidationIssue(
                    "GRAPH_RESTRICTED_EDGE_TYPE",
                    f"edge at index {index} uses restricted type {edge_type!r}",
                )
            )
        elif edge_type not in ALLOWED_EDGE_TYPES:
            issues.append(
                GraphValidationIssue(
                    "GRAPH_UNKNOWN_EDGE_TYPE",
                    f"edge at index {index} has unsupported type {edge_type!r}",
                )
            )

        source = edge.get("source")
        target = edge.get("target")

        if not _is_identifier(source) or not _is_identifier(target):
            issues.append(
                GraphValidationIssue(
                    "GRAPH_DANGLING_EDGE",
                    f"edge at index {index} must have non-empty string "
                    "source and target identifiers",
                )
            )
            continue

        missing_endpoints = [
            endpoint
            for endpoint in (source, target)
            if endpoint not in node_ids
        ]
        if missing_endpoints:
            issues.append(
                GraphValidationIssue(
                    "GRAPH_DANGLING_EDGE",
                    f"edge at index {index} references missing node(s): "
                    f"{sorted(set(missing_endpoints))}",
                )
            )

        if source == target:
            issues.append(
                GraphValidationIssue(
                    "GRAPH_SELF_LOOP",
                    f"edge at index {index} connects {source!r} to itself",
                )
            )


def _is_identifier(value: Any) -> bool:
    """Return whether ``value`` is a non-empty string identifier."""

    return isinstance(value, str) and bool(value)


__all__ = [
    "ALLOWED_EDGE_TYPES",
    "ALLOWED_NODE_TYPES",
    "GraphValidationError",
    "GraphValidationIssue",
    "RESTRICTED_EDGE_TYPES",
    "validate_graph",
]
