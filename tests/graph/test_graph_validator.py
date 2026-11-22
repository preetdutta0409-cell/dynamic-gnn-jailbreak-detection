"""Unit tests for :mod:`src.graph.graph_validator`.

A valid fixture graph is built once per test and then deliberately mutated to
exercise each validation boundary: duplicate node IDs, dangling edge endpoints,
cross-session nodes, and restricted edge types. Additional tests cover unknown
types, self-loops, and malformed graph roots.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from src.graph.graph_builder import build_session_graph
from src.graph.graph_validator import (
    ALLOWED_EDGE_TYPES,
    ALLOWED_NODE_TYPES,
    GraphValidationError,
    validate_graph,
)
from tests.graph.test_fixtures import TINY_SESSION_FIXTURE


@pytest.fixture()
def valid_graph() -> dict[str, Any]:
    """Return a fresh, schema-valid graph for the tiny fixture."""

    return build_session_graph(deepcopy(TINY_SESSION_FIXTURE))


def _node_ids(graph: dict[str, Any]) -> list[str]:
    """Return the graph's node identifiers in node-list order."""

    return [node["node_id"] for node in graph["nodes"]]


# ---------------------------------------------------------------------------
# Valid graph
# ---------------------------------------------------------------------------


def test_valid_graph_passes_validation(valid_graph: dict[str, Any]) -> None:
    assert validate_graph(valid_graph) is True


def test_allowed_type_sets_match_the_schema() -> None:
    assert ALLOWED_NODE_TYPES == {
        "user",
        "assistant",
        "retrieved_context",
        "tool",
        "proposed_action",
        "policy",
    }
    assert ALLOWED_EDGE_TYPES == {
        "NEXT_EVENT",
        "RESPONDS_TO",
        "RETRIEVED_FROM",
        "REQUESTS_TOOL",
    }


def test_validator_does_not_mutate_the_graph(
    valid_graph: dict[str, Any],
) -> None:
    original = deepcopy(valid_graph)

    validate_graph(valid_graph)

    assert valid_graph == original


# ---------------------------------------------------------------------------
# Required boundary violations
# ---------------------------------------------------------------------------


def test_duplicate_node_id_is_rejected(
    valid_graph: dict[str, Any],
) -> None:
    graph = deepcopy(valid_graph)
    graph["nodes"].append(deepcopy(graph["nodes"][0]))

    with pytest.raises(
        GraphValidationError,
        match="GRAPH_DUPLICATE_EVENT_ID",
    ):
        validate_graph(graph)


def test_edge_to_non_existent_node_is_rejected(
    valid_graph: dict[str, Any],
) -> None:
    graph = deepcopy(valid_graph)
    graph["edges"].append(
        {
            "source": _node_ids(graph)[0],
            "target": "evt-does-not-exist",
            "type": "NEXT_EVENT",
        }
    )

    with pytest.raises(
        GraphValidationError,
        match="GRAPH_DANGLING_EDGE",
    ):
        validate_graph(graph)


def test_edge_from_non_existent_node_is_rejected(
    valid_graph: dict[str, Any],
) -> None:
    graph = deepcopy(valid_graph)
    graph["edges"].append(
        {
            "source": "evt-does-not-exist",
            "target": _node_ids(graph)[0],
            "type": "NEXT_EVENT",
        }
    )

    with pytest.raises(
        GraphValidationError,
        match="GRAPH_DANGLING_EDGE",
    ):
        validate_graph(graph)


def test_node_from_a_different_session_is_rejected(
    valid_graph: dict[str, Any],
) -> None:
    graph = deepcopy(valid_graph)
    rogue_node = deepcopy(graph["nodes"][0])
    rogue_node["node_id"] = "evt-rogue-0001"
    rogue_node["event_id"] = "evt-rogue-0001"
    rogue_node["session_id"] = "sess-some-other-session"
    graph["nodes"].append(rogue_node)

    with pytest.raises(
        GraphValidationError,
        match="GRAPH_INVALID_SESSION_ID",
    ):
        validate_graph(graph)


def test_illegal_restricted_edge_type_is_rejected(
    valid_graph: dict[str, Any],
) -> None:
    graph = deepcopy(valid_graph)
    graph["edges"].append(
        {
            "source": _node_ids(graph)[0],
            "target": _node_ids(graph)[1],
            "type": "INFLUENCES",
        }
    )

    with pytest.raises(
        GraphValidationError,
        match="GRAPH_RESTRICTED_EDGE_TYPE",
    ):
        validate_graph(graph)


# ---------------------------------------------------------------------------
# Additional type and structural boundaries
# ---------------------------------------------------------------------------


def test_unknown_edge_type_is_rejected(
    valid_graph: dict[str, Any],
) -> None:
    graph = deepcopy(valid_graph)
    graph["edges"].append(
        {
            "source": _node_ids(graph)[0],
            "target": _node_ids(graph)[1],
            "type": "TELEPORTS_TO",
        }
    )

    with pytest.raises(
        GraphValidationError,
        match="GRAPH_UNKNOWN_EDGE_TYPE",
    ):
        validate_graph(graph)


def test_unknown_node_type_is_rejected(
    valid_graph: dict[str, Any],
) -> None:
    graph = deepcopy(valid_graph)
    graph["nodes"][0]["node_type"] = "document"

    with pytest.raises(
        GraphValidationError,
        match="GRAPH_UNKNOWN_NODE_TYPE",
    ):
        validate_graph(graph)


def test_self_loop_is_rejected(valid_graph: dict[str, Any]) -> None:
    graph = deepcopy(valid_graph)
    node_id = _node_ids(graph)[0]
    graph["edges"].append(
        {
            "source": node_id,
            "target": node_id,
            "type": "NEXT_EVENT",
        }
    )

    with pytest.raises(
        GraphValidationError,
        match="GRAPH_SELF_LOOP",
    ):
        validate_graph(graph)


def test_graph_missing_required_keys_is_rejected() -> None:
    with pytest.raises(
        GraphValidationError,
        match="GRAPH_INVALID_GRAPH",
    ):
        validate_graph({"session_id": "sess-x", "nodes": []})  # type: ignore[arg-type]


def test_graph_session_id_must_be_a_non_empty_string(
    valid_graph: dict[str, Any],
) -> None:
    graph = deepcopy(valid_graph)
    graph["session_id"] = ""

    with pytest.raises(
        GraphValidationError,
        match="GRAPH_INVALID_SESSION_ID",
    ):
        validate_graph(graph)


def test_multiple_violations_are_all_reported(
    valid_graph: dict[str, Any],
) -> None:
    graph = deepcopy(valid_graph)
    graph["nodes"].append(deepcopy(graph["nodes"][0]))
    graph["edges"].append(
        {
            "source": _node_ids(graph)[0],
            "target": "evt-missing",
            "type": "INFLUENCES",
        }
    )

    with pytest.raises(GraphValidationError) as exc_info:
        validate_graph(graph)

    codes = {issue.code for issue in exc_info.value.issues}
    assert "GRAPH_DUPLICATE_EVENT_ID" in codes
    assert "GRAPH_DANGLING_EDGE" in codes
    assert "GRAPH_RESTRICTED_EDGE_TYPE" in codes
