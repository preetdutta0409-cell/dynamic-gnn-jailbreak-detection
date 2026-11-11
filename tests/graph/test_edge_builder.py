"""Unit tests for :mod:`src.graph.edge_builder`.

The tests use the canonical ``TINY_SESSION_FIXTURE`` so that the expected edges
match the graph contract in ``docs/graph_schema.md``. Phase 2's ``build_node``
is used when available; the local fallback only materialises the documented
node record (Canonical Event fields plus ``node_id``/``node_type``) so that the
Phase 3 edge tests remain runnable in isolation.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from src.graph.edge_builder import (
    ALLOWED_EDGE_TYPES,
    EDGE_NEXT_EVENT,
    EDGE_REQUESTS_TOOL,
    EDGE_RESPONDS_TO,
    EDGE_RETRIEVED_FROM,
    build_edges,
)
from tests.graph.test_fixtures import (
    EVENT_TYPE_TO_NODE_TYPE,
    EXPECTED_EVENT_SCRIPT,
    EXPECTED_PARENT_LINKS,
    TINY_SESSION_FIXTURE,
)

try:
    from src.graph.node_builder import build_node as _build_node
except ImportError:  # Phase 2 is owned by another module; keep Phase 3 runnable.
    _build_node = None


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _fallback_build_node(event: dict[str, Any]) -> dict[str, Any]:
    """Build the documented node record when Phase 2 is not present."""

    node = deepcopy(event)
    node["node_id"] = event["event_id"]
    node["node_type"] = EVENT_TYPE_TO_NODE_TYPE[event["event_type"]]
    return node


def _build_fixture_nodes() -> list[dict[str, Any]]:
    """Convert the canonical fixture into graph nodes."""

    nodes: list[dict[str, Any]] = []

    for event in deepcopy(TINY_SESSION_FIXTURE):
        node = (
            _build_node(event)
            if _build_node is not None
            else _fallback_build_node(event)
        )
        if not isinstance(node, dict):
            raise TypeError("build_node must return a dictionary")
        nodes.append(node)

    return nodes


def _node_id(node: dict[str, Any]) -> str:
    """Return the identifier the edge builder emits for a node."""

    return node.get("node_id") or node["event_id"]


def _node_by_event_id(
    nodes: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index fixture nodes by their original Canonical Event ID."""

    return {
        node.get("event_id") or _node_id(node): node
        for node in nodes
    }


def _pairs(
    edges: list[dict[str, str]],
    edge_type: str,
) -> list[tuple[str, str]]:
    """Return the ordered source/target pairs for one edge type."""

    return [
        (edge["source"], edge["target"])
        for edge in edges
        if edge["type"] == edge_type
    ]


def _expected_event_chain(
    nodes: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Return the expected NEXT_EVENT pairs in fixture order."""

    by_event_id = _node_by_event_id(nodes)
    event_ids = [event_id for event_id, _ in EXPECTED_EVENT_SCRIPT]

    return [
        (_node_id(by_event_id[earlier]), _node_id(by_event_id[later]))
        for earlier, later in zip(event_ids, event_ids[1:])
    ]


def _expected_parent_pairs(
    nodes: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    """Return the expected child -> parent RESPONDS_TO pairs."""

    by_event_id = _node_by_event_id(nodes)
    expected: set[tuple[str, str]] = set()

    for child_event_id, parent_event_id in EXPECTED_PARENT_LINKS.items():
        if parent_event_id is None:
            continue
        expected.add(
            (
                _node_id(by_event_id[child_event_id]),
                _node_id(by_event_id[parent_event_id]),
            )
        )

    return expected


def _set_node_field(
    nodes: list[dict[str, Any]],
    event_id: str,
    field: str,
    value: Any,
) -> None:
    """Set one field on the fixture node with the given event ID."""

    node = _node_by_event_id(nodes)[event_id]
    node[field] = value


# ---------------------------------------------------------------------------
# NEXT_EVENT
# ---------------------------------------------------------------------------


def test_build_edges_generates_exactly_four_next_event_edges() -> None:
    nodes = _build_fixture_nodes()
    assert len(nodes) == 5

    edges = build_edges(nodes, deepcopy(TINY_SESSION_FIXTURE))
    next_event_edges = _pairs(edges, EDGE_NEXT_EVENT)

    assert len(next_event_edges) == 4
    assert next_event_edges == _expected_event_chain(nodes)


def test_next_event_edges_are_directed_and_ignore_input_order() -> None:
    nodes = _build_fixture_nodes()
    expected = _expected_event_chain(nodes)
    by_id = {_node_id(node): node for node in nodes}

    shuffled_nodes = deepcopy(list(reversed(nodes)))
    edges = build_edges(shuffled_nodes, deepcopy(TINY_SESSION_FIXTURE))

    assert _pairs(edges, EDGE_NEXT_EVENT) == expected

    for source, target in _pairs(edges, EDGE_NEXT_EVENT):
        source_timestamp = datetime.fromisoformat(
            by_id[source]["timestamp"]
        )
        target_timestamp = datetime.fromisoformat(
            by_id[target]["timestamp"]
        )
        assert source_timestamp < target_timestamp


# ---------------------------------------------------------------------------
# RESPONDS_TO
# ---------------------------------------------------------------------------


def test_responds_to_edges_match_fixture_parent_links() -> None:
    nodes = _build_fixture_nodes()
    edges = build_edges(nodes, deepcopy(TINY_SESSION_FIXTURE))

    responds_to_edges = _pairs(edges, EDGE_RESPONDS_TO)

    assert len(responds_to_edges) == 3
    assert set(responds_to_edges) == _expected_parent_pairs(nodes)
    assert len(responds_to_edges) == len(set(responds_to_edges))

    by_event_id = _node_by_event_id(nodes)
    evt_0002 = _node_id(by_event_id["evt-0002"])
    evt_0003 = _node_id(by_event_id["evt-0003"])
    evt_0004 = _node_id(by_event_id["evt-0004"])
    evt_0005 = _node_id(by_event_id["evt-0005"])

    assert (evt_0003, evt_0002) in responds_to_edges
    assert (evt_0004, evt_0003) in responds_to_edges
    assert (evt_0005, evt_0004) in responds_to_edges
    assert (evt_0002, evt_0003) not in responds_to_edges


def test_responds_to_can_be_recovered_from_raw_events() -> None:
    nodes = _build_fixture_nodes()

    for node in nodes:
        node.pop("parent_event_id", None)

    edges = build_edges(nodes, deepcopy(TINY_SESSION_FIXTURE))

    assert set(_pairs(edges, EDGE_RESPONDS_TO)) == _expected_parent_pairs(
        nodes
    )


def test_missing_parent_node_produces_no_dangling_responds_to_edge() -> None:
    nodes = _build_fixture_nodes()
    by_event_id = _node_by_event_id(nodes)

    missing_node_id = _node_id(by_event_id["evt-0004"])
    evt_0005_node_id = _node_id(by_event_id["evt-0005"])
    evt_0003_node_id = _node_id(by_event_id["evt-0003"])
    evt_0002_node_id = _node_id(by_event_id["evt-0002"])

    nodes_without_parent = [
        node for node in nodes if _node_id(node) != missing_node_id
    ]

    edges = build_edges(
        nodes_without_parent,
        deepcopy(TINY_SESSION_FIXTURE),
    )

    valid_node_ids = {_node_id(node) for node in nodes_without_parent}
    responds_to_edges = _pairs(edges, EDGE_RESPONDS_TO)

    assert (evt_0005_node_id, missing_node_id) not in responds_to_edges
    assert (evt_0003_node_id, evt_0002_node_id) in responds_to_edges
    assert all(
        edge["source"] in valid_node_ids and edge["target"] in valid_node_ids
        for edge in edges
    )


def test_unknown_parent_event_id_is_safely_ignored() -> None:
    nodes = _build_fixture_nodes()
    _set_node_field(nodes, "evt-0005", "parent_event_id", "evt-9999")

    raw_events = deepcopy(TINY_SESSION_FIXTURE)
    for raw_event in raw_events:
        if raw_event["event_id"] == "evt-0005":
            raw_event["parent_event_id"] = "evt-9999"

    edges = build_edges(nodes, raw_events)

    valid_node_ids = {_node_id(node) for node in nodes}
    evt_0005_node_id = _node_id(_node_by_event_id(nodes)["evt-0005"])

    assert all(
        edge["source"] in valid_node_ids and edge["target"] in valid_node_ids
        for edge in edges
    )
    assert not any(
        source == evt_0005_node_id
        for source, _ in _pairs(edges, EDGE_RESPONDS_TO)
    )


# ---------------------------------------------------------------------------
# RETRIEVED_FROM and REQUESTS_TOOL
# ---------------------------------------------------------------------------


def test_retrieved_from_uses_only_explicit_context_references() -> None:
    nodes = _build_fixture_nodes()
    edges = build_edges(nodes, deepcopy(TINY_SESSION_FIXTURE))

    by_event_id = _node_by_event_id(nodes)
    evt_0003 = _node_id(by_event_id["evt-0003"])
    evt_0004 = _node_id(by_event_id["evt-0004"])

    assert _pairs(edges, EDGE_RETRIEVED_FROM) == [(evt_0004, evt_0003)]


def test_requests_tool_uses_only_explicit_request_metadata() -> None:
    nodes = _build_fixture_nodes()
    edges = build_edges(nodes, deepcopy(TINY_SESSION_FIXTURE))

    by_event_id = _node_by_event_id(nodes)
    evt_0004 = _node_id(by_event_id["evt-0004"])
    evt_0005 = _node_id(by_event_id["evt-0005"])

    assert _pairs(edges, EDGE_REQUESTS_TOOL) == [(evt_0004, evt_0005)]


def test_malformed_retrieved_context_reference_is_ignored() -> None:
    nodes = _build_fixture_nodes()
    _set_node_field(nodes, "evt-0004", "metadata", {
        **(_node_by_event_id(nodes)["evt-0004"].get("metadata") or {}),
        "retrieved_context_event_ids": "evt-0003",
    })

    raw_events = deepcopy(TINY_SESSION_FIXTURE)
    for raw_event in raw_events:
        if raw_event["event_id"] == "evt-0004":
            raw_event["metadata"]["retrieved_context_event_ids"] = "evt-0003"

    edges = build_edges(nodes, raw_events)

    assert _pairs(edges, EDGE_RETRIEVED_FROM) == []


# ---------------------------------------------------------------------------
# Whole-graph invariants and boundaries
# ---------------------------------------------------------------------------


def test_all_edges_are_allowed_unique_directed_and_non_reflexive() -> None:
    nodes = _build_fixture_nodes()
    edges = build_edges(nodes, deepcopy(TINY_SESSION_FIXTURE))

    assert len(edges) == 9

    identities = {
        (edge["source"], edge["type"], edge["target"])
        for edge in edges
    }
    assert len(identities) == len(edges)

    for edge in edges:
        assert set(edge) == {"source", "target", "type"}
        assert edge["type"] in ALLOWED_EDGE_TYPES
        assert edge["source"] != edge["target"]


def test_no_restricted_semantic_edges_are_created() -> None:
    nodes = _build_fixture_nodes()
    edges = build_edges(nodes, deepcopy(TINY_SESSION_FIXTURE))

    edge_types = {edge["type"] for edge in edges}

    assert "INFLUENCES" not in edge_types
    assert "CONFLICTS_WITH" not in edge_types
    assert edge_types <= ALLOWED_EDGE_TYPES


def test_empty_and_single_node_sessions_have_no_next_event_edges() -> None:
    assert build_edges([], deepcopy(TINY_SESSION_FIXTURE)) == []

    single_node = _build_fixture_nodes()[:1]
    edges = build_edges(single_node, deepcopy(TINY_SESSION_FIXTURE[:1]))

    assert edges == []


def test_build_edges_does_not_mutate_inputs() -> None:
    nodes = _build_fixture_nodes()
    raw_events = deepcopy(TINY_SESSION_FIXTURE)
    nodes_before = deepcopy(nodes)
    raw_events_before = deepcopy(raw_events)

    build_edges(nodes, raw_events)

    assert nodes == nodes_before
    assert raw_events == raw_events_before
