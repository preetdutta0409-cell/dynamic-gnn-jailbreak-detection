"""End-to-end tests for the public :mod:`src.graph` pipeline façade.

The tests exercise the single-call contract consumed by the GNN/Runtime team:
Canonical Events in, a PyG-ready topology dictionary out.
"""

from __future__ import annotations

from copy import deepcopy

import torch

from src.graph import process_events_to_pyg
from tests.graph.test_fixtures import TINY_SESSION_FIXTURE


def test_pipeline_returns_pyg_ready_topology_for_the_tiny_fixture() -> None:
    pyg_graph = process_events_to_pyg(deepcopy(TINY_SESSION_FIXTURE))

    assert isinstance(pyg_graph, dict)
    assert "edge_index" in pyg_graph
    assert "edge_type" in pyg_graph
    assert isinstance(pyg_graph["edge_index"], torch.Tensor)
    assert isinstance(pyg_graph["edge_type"], torch.Tensor)


def test_pipeline_preserves_session_and_node_mapping() -> None:
    pyg_graph = process_events_to_pyg(deepcopy(TINY_SESSION_FIXTURE))

    assert pyg_graph["session_id"] == TINY_SESSION_FIXTURE[0]["session_id"]
    assert pyg_graph["node_mapping"] == [
        event["event_id"] for event in TINY_SESSION_FIXTURE
    ]


def test_pipeline_tensor_shapes_dtypes_and_edge_counts_are_consistent() -> None:
    pyg_graph = process_events_to_pyg(deepcopy(TINY_SESSION_FIXTURE))

    edge_index = pyg_graph["edge_index"]
    edge_type = pyg_graph["edge_type"]

    assert edge_index.dtype == torch.long
    assert edge_type.dtype == torch.long
    assert edge_index.dim() == 2
    assert edge_index.shape[0] == 2
    assert edge_type.dim() == 1
    assert edge_index.shape[1] == edge_type.shape[0]
    assert edge_index.shape[1] > 0


def test_pipeline_does_not_mutate_input_events() -> None:
    events = deepcopy(TINY_SESSION_FIXTURE)
    original = deepcopy(events)

    process_events_to_pyg(events)

    assert events == original
