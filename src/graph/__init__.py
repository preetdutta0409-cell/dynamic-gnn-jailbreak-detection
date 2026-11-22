"""Graph construction pipeline for the dynamic GNN security middleware.

The public API is intentionally narrow: downstream teams should import
:func:`process_events_to_pyg` from this package rather than orchestrating the
individual builder, validator, and exporter modules themselves.
"""

from __future__ import annotations

from src.graph.pipeline import process_events_to_pyg

__all__ = [
    "process_events_to_pyg",
]
