"""Public graph pipeline entry point.

This module is the Phase 6 façade over the Phase 4-5 graph stages. It composes
the graph builder, the structural validator, and the PyTorch Geometric exporter
so that downstream consumers can produce model-ready topology with a single
call::

    from src.graph import process_events_to_pyg

    pyg_graph = process_events_to_pyg(canonical_events)

The stages remain independently usable; this module only sequences them and
never duplicates or alters their internal logic.
"""

from __future__ import annotations

import logging
from typing import Any

from src.graph.exporter import export_to_pyg_format
from src.graph.graph_builder import build_session_graph
from src.graph.graph_validator import validate_graph

logger = logging.getLogger(__name__)


def process_events_to_pyg(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build, validate, and export one session's Canonical Events to PyG format.

    Parameters
    ----------
    events:
        Canonical Event dictionaries belonging to a single session. They are
        treated as read-only by every pipeline stage.

    Returns
    -------
    dict[str, Any]
        The PyG-ready dictionary produced by
        :func:`src.graph.exporter.export_to_pyg_format`, containing
        ``session_id``, ``node_mapping``, ``edge_index``, and ``edge_type``.

    Raises
    ------
    TypeError
        If ``events`` is not a list, or if any stage rejects the input type.
    ValueError
        If the events cannot form a graph, or the graph fails structural
        validation.
    """

    logger.debug("Building session graph from Canonical Events")
    graph = build_session_graph(events)

    logger.debug(
        "Validating session graph %r",
        graph.get("session_id"),
    )
    validate_graph(graph)

    logger.debug(
        "Exporting session graph %r to PyG format",
        graph.get("session_id"),
    )
    pyg_graph = export_to_pyg_format(graph)

    logger.info(
        "Processed session %r: %d node(s), %d edge(s)",
        pyg_graph.get("session_id"),
        len(pyg_graph.get("node_mapping", [])),
        int(pyg_graph["edge_type"].shape[0]),
    )

    return pyg_graph


__all__ = [
    "process_events_to_pyg",
]
