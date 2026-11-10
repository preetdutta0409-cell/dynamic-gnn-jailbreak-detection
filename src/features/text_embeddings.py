"""
text_embeddings.py

Purpose:
Convert textual graph-node content into numerical embeddings
using Sentence Transformers.

Used by:
- node_features.py
- feature_encoder.py
- training pipeline
- inference pipeline
"""

from typing import List, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbeddingGenerator:
    """
    Generates semantic embeddings for graph-node text.

    The model is loaded once when this class is initialized.
    It is reused for all subsequent encoding operations.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 16,
        normalize_embeddings: bool = True
    ):

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        self.model = SentenceTransformer(
            model_name,
            device=device
        )

        self.embedding_dimension = (
            self.model.get_embedding_dimension()
        )

        if self.embedding_dimension is None:
            raise ValueError(
                "Could not determine embedding dimension."
            )

    def encode(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        """
        Convert one or more texts into embeddings.

        Returns:
            A NumPy array with shape:
            (number_of_texts, embedding_dimension)
        """

        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(texts, (list, tuple)):
            raise TypeError(
                "texts must be a string, list or tuple."
            )

        if len(texts) == 0:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32
            )

        cleaned_texts = []

        for text in texts:
            if text is None:
                cleaned_texts.append("")
            elif not isinstance(text, str):
                raise TypeError(
                    "Every text item must be a string or None."
                )
            else:
                cleaned_texts.append(text.strip())

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        expected_shape = (
            len(cleaned_texts),
            self.embedding_dimension
        )

        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Unexpected embedding shape: {embeddings.shape}. "
                f"Expected: {expected_shape}"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embedding contains NaN or infinite values."
            )

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate an embedding for one text.

        Returns:
            One-dimensional NumPy vector.
        """

        return self.encode([text])[0]

    def get_dimension(self) -> int:
        """
        Return the embedding dimension.
        """

        return self.embedding_dimension
