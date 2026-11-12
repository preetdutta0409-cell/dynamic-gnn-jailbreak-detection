"""Unit tests for :mod:`src.graph.graph_builder`.

The tests use the canonical ``TINY_SESSION_FIXTURE`` so that Phase 4 assembly
is checked against the graph contract in ``docs/graph_schema.md``.

The fixture produces five nodes. Phase 3's edge builder emits four
``NEXT_EVENT`` edges plus the explicit semantic edges, for nine edges in total.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

import src.graph.graph_builder as graph_builder_module
from src.graph.edge_builder import (
    EDGE_NEXT_EVENT,
    EDGE_REQUESTS_TOOL,
    EDGE_RESPONDS_TO,
    EDGE_RETRIEVED_FROM,
)
from src.graph.graph_builder import build_session_graph
from tests.graph.test_fixtures import (
    EXPECTED_NODE_TYPES,
    TINY_SESSION_FIXTURE,
    TINY_SESSION_ID,
)


def _build_fixture_graph() -> dict[str, Any]:
    """Build the canonical fixture graph from a defensive copy."""

    return build_session_graph(deepcopy(TINY_SESSION_FIXTURE))


# ---------------------------------------------------------------------------
# Assembly contract
# ---------------------------------------------------------------------------


def test_build_session_graph_returns_five_nodes_and_four_next_event_edges() -> None:
    graph = _build_fixture_graph()

    assert isinstance(graph, dict)
    assert set(graph) == {"session_id", "nodes", "edges"}
    assert graph["session_id"] == TINY_SESSION_ID
    assert len(graph["nodes"]) == 5

    next_event_edges = [
        edge
        for edge in graph["edges"]
        if edge["type"] == EDGE_NEXT_EVENT
    ]
    assert len(next_event_edges) == 4

    # The Phase 3 contract also emits 3 RESPONDS_TO, 1 RETRIEVED_FROM, and
    # 1 REQUESTS_TOOL edge for this fixture.
    assert len(graph["edges"]) == 9


def test_graph_contains_one_node_per_event_with_expected_types() -> None:
    graph = _build_fixture_graph()

    nodes_by_event_id = {
        node["event_id"]: node for node in graph["nodes"]
    }

    assert set(nodes_by_event_id) == set(EXPECTED_NODE_TYPES)
    assert len(nodes_by_event_id) == len(graph["nodes"])

    for event_id, expected_node_type in EXPECTED_NODE_TYPES.items():
        node = nodes_by_event_id[event_id]
        assert node["node_id"] == event_id
        assert node["node_type"] == expected_node_type
        assert node["session_id"] == TINY_SESSION_ID


def test_graph_edges_cover_every_allowed_edge_type() -> None:
    graph = _build_fixture_graph()

    edge_types = {edge["type"] for edge in graph["edges"]}

    assert edge_types == {
        EDGE_NEXT_EVENT,
        EDGE_RESPONDS_TO,
        EDGE_RETRIEVED_FROM,
        EDGE_REQUESTS_TOOL,
    }
    assert "INFLUENCES" not in edge_types
    assert "CONFLICTS_WITH" not in edge_types


def test_every_edge_endpoint_exists_in_the_node_list() -> None:
    graph = _build_fixture_graph()
    node_ids = {node["node_id"] for node in graph["nodes"]}

    for edge in graph["edges"]:
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids
        assert edge["source"] != edge["target"]


# ---------------------------------------------------------------------------
# Piping between Phase 2 and Phase 3
# ---------------------------------------------------------------------------


def test_generated_nodes_and_raw_events_are_piped_to_edge_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_build_edges(
        nodes: list[dict[str, Any]],
        raw_events: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        captured["nodes"] = deepcopy(nodes)
        captured["raw_events"] = deepcopy(raw_events)
        return []

    monkeypatch.setattr(
        graph_builder_module,
        "build_edges",
        fake_build_edges,
    )

    graph = build_session_graph(deepcopy(TINY_SESSION_FIXTURE))

    assert len(captured["nodes"]) == 5
    assert captured["raw_events"] == TINY_SESSION_FIXTURE
    assert graph["edges"] == []


def test_build_session_graph_is_deterministic() -> None:
    first = _build_fixture_graph()
    second = _build_fixture_graph()

    assert first == second


def test_build_session_graph_does_not_mutate_input_events() -> None:
    events = deepcopy(TINY_SESSION_FIXTURE)
    original_events = deepcopy(events)

    build_session_graph(events)

    assert events == original_events


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


def test_empty_event_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty event list"):
        build_session_graph([])


def test_non_list_events_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="events must be a list"):
        build_session_graph(deepcopy(TINY_SESSION_FIXTURE[0]))  # type: ignore[arg-type]


def test_non_mapping_event_is_rejected() -> None:
    events: list[Any] = [deepcopy(TINY_SESSION_FIXTURE[0]), "not-an-event"]

    with pytest.raises(TypeError, match="event at index 1 must be a mapping"):
        build_session_graph(events)


def test_missing_session_id_is_rejected() -> None:
    events = deepcopy(TINY_SESSION_FIXTURE)
    for event in events:
        event["session_id"] = ""

    with pytest.raises(ValueError, match="non-empty session_id"):
        build_session_graph(events)


def test_unknown_event_type_is_rejected() -> None:
    events = deepcopy(TINY_SESSION_FIXTURE)
    events[0]["event_type"] = "system_prompt"

    with pytest.raises(ValueError, match="unknown event_type"):
        build_session_graph(events)
