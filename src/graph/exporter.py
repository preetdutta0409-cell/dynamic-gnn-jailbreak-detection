"""Export a validated graph dictionary to PyTorch Geometric topology tensors.

Phase 5 converts the string-based graph produced by
:mod:`src.graph.graph_builder` into the integer-indexed topology required by
PyTorch Geometric:

* ``edge_index`` -- ``torch.long`` tensor of shape ``[2, num_edges]``.
* ``edge_type``  -- ``torch.long`` tensor of shape ``[num_edges]``.

Node features are deliberately **not** produced here. The feature matrix
``x`` is owned by the Feature Encoder module. This exporter only translates
graph topology, preserving the input node order as the integer node index.

The returned dictionary has exactly four keys::

    {
        "session_id": str,
        "node_mapping": list[str],
        "edge_index": torch.Tensor,
        "edge_type": torch.Tensor,
    }

``node_mapping`` is the reverse mapping required by downstream modules:
``node_mapping[index]`` is the original string ``node_id`` for that integer
index.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Final

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

EDGE_TYPE_TO_INDEX: Final[dict[str, int]] = {
    "NEXT_EVENT": 0,
    "RESPONDS_TO": 1,
    "RETRIEVED_FROM": 2,
    "REQUESTS_TOOL": 3,
}

INDEX_TO_EDGE_TYPE: Final[dict[int, str]] = {
    index: edge_type
    for edge_type, index in EDGE_TYPE_TO_INDEX.items()
}

_REQUIRED_GRAPH_KEYS: Final[tuple[str, ...]] = (
    "session_id",
    "nodes",
    "edges",
)


def export_to_pyg_format(graph: dict[str, Any]) -> dict[str, Any]:
    """Convert a validated graph dictionary into PyG topology tensors.

    Parameters
    ----------
    graph:
        A validated Phase 4 graph with the keys ``session_id``, ``nodes``, and
        ``edges``. Nodes are expected to expose a non-empty string ``node_id``.
        Edges are expected to expose string ``source``, ``target``, and
        ``type`` values.

    Returns
    -------
    dict[str, Any]
        A dictionary containing ``session_id``, the reverse ``node_mapping``
        (integer index -> string node ID), ``edge_index``, and ``edge_type``.
        No feature matrix (``x``) is generated.

    Raises
    ------
    TypeError
        If the graph, a node, or an edge is not a mapping.
    ValueError
        If required keys are missing, identifiers are invalid or duplicated,
        an edge type is unknown, or an edge endpoint does not exist.
    """

    if not isinstance(graph, Mapping):
        message = "graph must be a dictionary"
        logger.error(message)
        raise TypeError(message)

    missing_keys = [
        key for key in _REQUIRED_GRAPH_KEYS if key not in graph
    ]
    if missing_keys:
        message = (
            "graph is missing required keys: "
            f"{sorted(missing_keys)}"
        )
        logger.error(message)
        raise ValueError(message)

    session_id = graph["session_id"]
    if not _is_identifier(session_id):
        message = (
            "graph session_id must be a non-empty string, "
            f"got {session_id!r}"
        )
        logger.error(message)
        raise ValueError(message)

    nodes = graph["nodes"]
    edges = graph["edges"]

    if not isinstance(nodes, list):
        message = "graph nodes must be a list"
        logger.error(message)
        raise TypeError(message)

    if not isinstance(edges, list):
        message = "graph edges must be a list"
        logger.error(message)
        raise TypeError(message)

    node_mapping, node_id_to_index = _build_node_mapping(nodes)
    edge_index, edge_type = _build_edge_tensors(
        edges,
        node_id_to_index,
    )

    return {
        "session_id": session_id,
        "node_mapping": node_mapping,
        "edge_index": edge_index,
        "edge_type": edge_type,
    }


def _build_node_mapping(
    nodes: list[Any],
) -> tuple[list[str], dict[str, int]]:
    """Return the ordered reverse mapping and the string-to-index lookup."""

    node_mapping: list[str] = []
    node_id_to_index: dict[str, int] = {}

    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            message = f"node at index {index} must be a mapping"
            logger.error(message)
            raise TypeError(message)

        node_id = node.get("node_id")
        if not _is_identifier(node_id):
            message = f"node at index {index} has no non-empty node_id"
            logger.error(message)
            raise ValueError(message)

        if node_id in node_id_to_index:
            message = f"duplicate node_id {node_id!r} at index {index}"
            logger.error(message)
            raise ValueError(message)

        node_id_to_index[node_id] = index
        node_mapping.append(node_id)

    return node_mapping, node_id_to_index


def _build_edge_tensors(
    edges: list[Any],
    node_id_to_index: dict[str, int],
) -> tuple[Tensor, Tensor]:
    """Return ``edge_index`` and ``edge_type`` tensors for ``edges``."""

    if not edges:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
        )

    sources: list[int] = []
    targets: list[int] = []
    edge_types: list[int] = []

    for index, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            message = f"edge at index {index} must be a mapping"
            logger.error(message)
            raise TypeError(message)

        source = edge.get("source")
        target = edge.get("target")
        edge_type = edge.get("type")

        if not _is_identifier(source) or not _is_identifier(target):
            message = (
                f"edge at index {index} must have non-empty string "
                "source and target identifiers"
            )
            logger.error(message)
            raise ValueError(message)

        if (
            not isinstance(edge_type, str)
            or edge_type not in EDGE_TYPE_TO_INDEX
        ):
            message = (
                f"edge at index {index} has unknown edge type "
                f"{edge_type!r}; allowed types are "
                f"{sorted(EDGE_TYPE_TO_INDEX)}"
            )
            logger.error(message)
            raise ValueError(message)

        missing_endpoints = [
            endpoint
            for endpoint in (source, target)
            if endpoint not in node_id_to_index
        ]
        if missing_endpoints:
            message = (
                f"edge at index {index} references unknown node(s): "
                f"{sorted(set(missing_endpoints))}"
            )
            logger.error(message)
            raise ValueError(message)

        sources.append(node_id_to_index[source])
        targets.append(node_id_to_index[target])
        edge_types.append(EDGE_TYPE_TO_INDEX[edge_type])

    edge_index = torch.tensor(
        [sources, targets],
        dtype=torch.long,
    )
    edge_type_tensor = torch.tensor(
        edge_types,
        dtype=torch.long,
    )

    return edge_index, edge_type_tensor


def _is_identifier(value: Any) -> bool:
    """Return whether ``value`` is a non-empty string identifier."""

    return isinstance(value, str) and bool(value)


__all__ = [
    "EDGE_TYPE_TO_INDEX",
    "INDEX_TO_EDGE_TYPE",
    "export_to_pyg_format",
]
