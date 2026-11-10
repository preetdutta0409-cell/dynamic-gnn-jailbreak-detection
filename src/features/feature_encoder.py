"""
feature_encoder.py

Purpose:
Combine semantic text embeddings and structured graph-node
features into one numerical feature vector per event node.

Uses:
- TextEmbeddingGenerator from text_embeddings.py
- NodeFeatureExtractor from node_features.py

Output:
A NumPy matrix containing one combined feature vector per node.
"""

from typing import Any, Dict, List, Sequence

import numpy as np

from src.features.text_embeddings import TextEmbeddingGenerator
from src.features.node_features import NodeFeatureExtractor


class FeatureEncoder:
    """
    Combines text embeddings and structured node features.

    For each graph event node:

        Combined features =
            Text embedding + Structured node features

    The order of input nodes is preserved throughout encoding.
    """

    def __init__(
        self,
        text_embedder: Any = None,
        node_feature_extractor: Any = None,
    ) -> None:

        # Reuse the existing text embedding module.
        if text_embedder is None:
            text_embedder = TextEmbeddingGenerator()

        # Reuse the existing structured feature module.
        if node_feature_extractor is None:
            node_feature_extractor = NodeFeatureExtractor()

        self.text_embedder = text_embedder
        self.node_feature_extractor = node_feature_extractor

        self.text_dimension = self.text_embedder.get_dimension()

        self.structured_dimension = (
            self.node_feature_extractor.get_output_dimension()
        )

        if self.text_dimension <= 0:
            raise ValueError(
                "Text embedding dimension must be positive."
            )

        if self.structured_dimension <= 0:
            raise ValueError(
                "Structured feature dimension must be positive."
            )

        # Calculate the combined dimension automatically.
        self.output_dimension = (
            self.text_dimension + self.structured_dimension
        )

    def get_output_dimension(self) -> int:
        """Return the dimension of the combined feature vector."""

        return self.output_dimension

    @staticmethod
    def _extract_texts(
        nodes: Sequence[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract text from graph-node dictionaries.

        Nodes without text receive an empty string.
        This allows non-textual event nodes to be represented
        using their structured features.
        """

        texts = []

        for index, node in enumerate(nodes):

            if not isinstance(node, dict):
                raise TypeError(
                    f"Node at position {index} must be a dictionary."
                )

            text = node.get("text")

            if text is None:
                text = ""

            if not isinstance(text, str):
                raise TypeError(
                    f"Text for node at position {index} "
                    "must be a string or None."
                )

            texts.append(text.strip())

        return texts

    def transform(
        self,
        nodes: Sequence[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Generate combined features for graph event nodes.

        Args:
            nodes:
                List or tuple of graph-node dictionaries.

        Returns:
            float32 NumPy matrix with shape:

            (number_of_nodes, combined_feature_dimension)

        The row at position i always represents nodes[i].
        """

        if not isinstance(nodes, (list, tuple)):
            raise TypeError(
                "nodes must be a list or tuple."
            )

        # Handle an empty graph without calling the embedding model.
        if len(nodes) == 0:
            return np.empty(
                (0, self.output_dimension),
                dtype=np.float32,
            )

        # Step 1: Extract textual content.
        texts = self._extract_texts(nodes)

        # Step 2: Generate semantic text embeddings.
        text_features = self.text_embedder.encode(texts)

        text_features = np.asarray(
            text_features,
            dtype=np.float32,
        )

        # Step 3: Generate structured node features.
        structured_features = (
            self.node_feature_extractor.transform(nodes)
        )

        structured_features = np.asarray(
            structured_features,
            dtype=np.float32,
        )

        # Step 4: Verify that both feature matrices
        # contain exactly one row for every input node.
        expected_text_shape = (
            len(nodes),
            self.text_dimension,
        )

        expected_structured_shape = (
            len(nodes),
            self.structured_dimension,
        )

        if text_features.shape != expected_text_shape:
            raise ValueError(
                f"Unexpected text feature shape: "
                f"{text_features.shape}. "
                f"Expected: {expected_text_shape}."
            )

        if structured_features.shape != expected_structured_shape:
            raise ValueError(
                f"Unexpected structured feature shape: "
                f"{structured_features.shape}. "
                f"Expected: {expected_structured_shape}."
            )

        # Step 5: Check that the generated values are valid.
        if not np.isfinite(text_features).all():
            raise ValueError(
                "Text features contain NaN or infinite values."
            )

        if not np.isfinite(structured_features).all():
            raise ValueError(
                "Structured features contain NaN or infinite values."
            )

        # Step 6: Combine both feature matrices column-wise.
        combined_features = np.concatenate(
            [text_features, structured_features],
            axis=1,
        )

        combined_features = np.asarray(
            combined_features,
            dtype=np.float32,
        )

        # Step 7: Validate the final output.
        expected_output_shape = (
            len(nodes),
            self.output_dimension,
        )

        if combined_features.shape != expected_output_shape:
            raise ValueError(
                f"Unexpected combined feature shape: "
                f"{combined_features.shape}. "
                f"Expected: {expected_output_shape}."
            )

        if not np.isfinite(combined_features).all():
            raise ValueError(
                "Combined features contain invalid values."
            )

        return combined_features
