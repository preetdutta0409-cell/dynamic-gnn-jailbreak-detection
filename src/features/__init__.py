"""
Feature package.

Contains modules for extracting and encoding
textual and structured graph-node features.
"""

from src.features.text_embeddings import TextEmbeddingGenerator
from src.features.node_features import NodeFeatureExtractor
from src.features.feature_encoder import FeatureEncoder

__all__ = [
    "TextEmbeddingGenerator",
    "NodeFeatureExtractor",
    "FeatureEncoder",
]
