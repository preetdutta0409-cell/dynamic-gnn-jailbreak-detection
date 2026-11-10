
"""
node_features.py

Purpose:
Convert structured graph-event metadata into numerical node features.

This module handles:
- Six event-node types
- Explicit trust values
- Temporal features
- Event provenance

It does NOT:
- Generate text embeddings
- Create graph nodes or edges
- Train a GNN
"""

from datetime import datetime, timezone
from math import log1p, sin, cos, pi
from typing import Any, Dict, List, Sequence

import numpy as np


class NodeFeatureExtractor:
    """
    Extract structured numerical features from event-based graph nodes.

    Input:
        A sequence of dictionaries. Each dictionary represents
        one graph event node.

    Output:
        A float32 NumPy matrix with one row per input node.

    The input order is preserved.
    """

    NODE_TYPES = (
        "user",
        "assistant",
        "retrieved_context",
        "tool",
        "proposed_action",
        "policy",
    )

    # These are initial source categories.
    # Member 3 must use the same values in graph construction.
    PROVENANCE_TYPES = (
        "user_input",
        "assistant_generated",
        "retrieved_document",
        "tool_output",
        "system_policy",
        "unknown",
    )

    TRUST_MIN = 0.0
    TRUST_MAX = 1.0

    def __init__(self) -> None:
        self.node_type_dimension = len(self.NODE_TYPES)
        self.trust_dimension = 2
        self.temporal_dimension = 4
        self.provenance_dimension = len(self.PROVENANCE_TYPES)

        self.output_dimension = (
            self.node_type_dimension
            + self.trust_dimension
            + self.temporal_dimension
            + self.provenance_dimension
        )

        self.node_type_to_index = {
            node_type: index
            for index, node_type in enumerate(self.NODE_TYPES)
        }

        self.provenance_to_index = {
            source: index
            for index, source in enumerate(self.PROVENANCE_TYPES)
        }

    def get_output_dimension(self) -> int:
        """Return the number of structured features per node."""
        return self.output_dimension

    @staticmethod
    def _parse_timestamp(value: Any):
        """
        Convert a timestamp into a timezone-aware datetime.

        Accepted inputs:
        - Unix timestamp in seconds
        - ISO-8601 timestamp string
        - datetime object

        Naive datetime values are treated as UTC.
        Missing timestamps return None.
        """

        if value is None or value == "":
            return None

        if isinstance(value, datetime):
            parsed = value

        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = datetime.fromtimestamp(
                float(value),
                tz=timezone.utc,
            )

        elif isinstance(value, str):
            timestamp_text = value.strip()

            if timestamp_text.endswith("Z"):
                timestamp_text = timestamp_text[:-1] + "+00:00"

            parsed = datetime.fromisoformat(timestamp_text)

        else:
            raise TypeError(
                "Timestamp must be a Unix number, ISO string, "
                "datetime object, or None."
            )

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    def _encode_node_type(self, node_type: str) -> List[float]:
        """One-hot encode one of the six event-node types."""

        if node_type not in self.node_type_to_index:
            raise ValueError(
                f"Unsupported node_type: {node_type}. "
                f"Expected one of {self.NODE_TYPES}."
            )

        vector = [0.0] * self.node_type_dimension
        vector[self.node_type_to_index[node_type]] = 1.0

        return vector

    def _encode_trust(self, trust_level: Any) -> List[float]:
        """
        Encode trust as:
        [trust_value, trust_is_available]

        Trust values must already be defined by the project
        and supplied on a 0-to-1 scale.

        Missing trust is represented as [0.0, 0.0].
        It is NOT interpreted as confirmed zero trust.
        """

        if trust_level is None:
            return [0.0, 0.0]

        if isinstance(trust_level, bool):
            raise TypeError("trust_level must be numeric, not boolean.")

        try:
            trust_value = float(trust_level)
        except (TypeError, ValueError) as exc:
            raise TypeError("trust_level must be numeric or None.") from exc

        if not np.isfinite(trust_value):
            raise ValueError("trust_level must be finite.")

        if not self.TRUST_MIN <= trust_value <= self.TRUST_MAX:
            raise ValueError("trust_level must be between 0.0 and 1.0.")

        return [trust_value, 1.0]

    def _encode_provenance(self, source: Any) -> List[float]:
        """
        One-hot encode the event source.

        Missing sources are represented by the explicit unknown category.
        """

        if source is None or source == "":
            source = "unknown"

        if source not in self.provenance_to_index:
            raise ValueError(
                f"Unsupported source: {source}. "
                f"Expected one of {self.PROVENANCE_TYPES}."
            )

        vector = [0.0] * self.provenance_dimension
        vector[self.provenance_to_index[source]] = 1.0

        return vector

    def _encode_temporal(self,node: Dict[str, Any],previous_timestamp: Any,session_max_event_index: int,fallback_position: int,) -> List[float]:
        """
        Produce four temporal features:

        1. Normalized event position within the session.
        2. log(1 + seconds since the previous event).
        3. Sine of time-of-day.
        4. Cosine of time-of-day.

        Missing timestamps produce zero values for time-based features.
        """

        event_index = node.get("event_index")

        if event_index is None:
            event_index = fallback_position

        if isinstance(event_index, bool):
            raise TypeError("event_index must be an integer.")

        if not isinstance(event_index, int) or event_index < 0:
            raise ValueError("event_index must be a non-negative integer.")

        if session_max_event_index > 0:
            normalized_position = (
                event_index / session_max_event_index
            )
        else:
            normalized_position = 0.0

        current_timestamp = self._parse_timestamp(
            node.get("timestamp")
        )

        time_gap_feature = 0.0
        time_sin = 0.0
        time_cos = 0.0

        if current_timestamp is not None:

            if previous_timestamp is not None:
                delta_seconds = (
                    current_timestamp - previous_timestamp
                ).total_seconds()

                if delta_seconds < 0:
                    raise ValueError(
                        "Events must be chronologically ordered "
                        "within each session."
                    )

                time_gap_feature = log1p(delta_seconds)

            seconds_since_midnight = (
                current_timestamp.hour * 3600
                + current_timestamp.minute * 60
                + current_timestamp.second
                + current_timestamp.microsecond / 1_000_000
            )

            angle = (
                2 * pi * seconds_since_midnight / 86400
            )

            time_sin = sin(angle)
            time_cos = cos(angle)

        return [
            float(normalized_position),
            float(time_gap_feature),
            float(time_sin),
            float(time_cos),
        ]

    def transform(
        self,
        nodes: Sequence[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Extract structured features for a collection of graph nodes.

        The input order is preserved. Nodes belonging to each session
        must be supplied in chronological event order.

        Required fields:
            node_id
            node_type

        Optional fields:
            event_index
            timestamp
            trust_level
            source
            session_id
            parent_event_id
            metadata
        """

        if not isinstance(nodes, (list, tuple)):
            raise TypeError("nodes must be a list or tuple.")

        if len(nodes) == 0:
            return np.empty(
                (0, self.output_dimension),
                dtype=np.float32,
            )

        # Validate node records and unique IDs.
        seen_node_ids = set()

        for node in nodes:

            if not isinstance(node, dict):
                raise TypeError("Every node must be a dictionary.")

            if "node_id" not in node:
                raise ValueError("Every node must have node_id.")

            if "node_type" not in node:
                raise ValueError("Every node must have node_type.")

            node_id = node["node_id"]

            if node_id in seen_node_ids:
                raise ValueError(f"Duplicate node_id: {node_id}")

            seen_node_ids.add(node_id)

        # Determine the maximum event index for each session.
        session_max_indices = {}

        for position, node in enumerate(nodes):

            session_id = node.get("session_id", "default")
            event_index = node.get("event_index")

            if event_index is None:
                event_index = position

            if isinstance(event_index, bool):
                raise TypeError("event_index must be an integer.")

            if not isinstance(event_index, int) or event_index < 0:
                raise ValueError(
                    "event_index must be a non-negative integer."
                )

            session_max_indices[session_id] = max(
                session_max_indices.get(session_id, 0),
                event_index,
            )

        previous_timestamps = {}
        previous_event_indices = {}

        feature_rows = []

        for position, node in enumerate(nodes):

            session_id = node.get("session_id", "default")

            event_index = node.get("event_index")
            if event_index is None:
                event_index = position

            # Ensure each session is supplied in event order.
            previous_index = previous_event_indices.get(session_id)

            if (
                previous_index is not None
                and event_index < previous_index
            ):
                raise ValueError(
                    f"Nodes in session {session_id} are not "
                    "in event_index order."
                )

            previous_event_indices[session_id] = event_index

            node_type_features = self._encode_node_type(
                node["node_type"]
            )

            trust_features = self._encode_trust(
                node.get("trust_level")
            )

            provenance_features = self._encode_provenance(
                node.get("source")
            )

            temporal_features = self._encode_temporal(
                node=node,
                previous_timestamp=previous_timestamps.get(session_id),
                session_max_event_index=session_max_indices[session_id],
                fallback_position=position,
            )

            current_timestamp = self._parse_timestamp(
                node.get("timestamp")
            )

            if current_timestamp is not None:
                previous_timestamps[session_id] = current_timestamp

            row = (
                node_type_features
                + trust_features
                + temporal_features
                + provenance_features
            )

            feature_rows.append(row)

        result = np.asarray(
            feature_rows,
            dtype=np.float32,
        )

        if result.shape != (len(nodes), self.output_dimension):
            raise ValueError(
                f"Unexpected feature matrix shape: {result.shape}"
            )

        if not np.isfinite(result).all():
            raise ValueError(
                "Generated node features contain invalid values."
            )

        return result
