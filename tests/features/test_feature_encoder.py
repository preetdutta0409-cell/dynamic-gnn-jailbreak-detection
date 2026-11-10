
import numpy as np
import pytest

from src.features.feature_encoder import FeatureEncoder
from src.features.node_features import NodeFeatureExtractor


class DummyTextEmbedder:
    """
    Lightweight substitute for TextEmbeddingGenerator.

    Used only to test feature combination without loading
    the actual Sentence Transformer model.
    """

    def get_dimension(self):
        return 3

    def encode(self, texts):
        return np.array(
            [
                [float(len(text)), 0.1, 0.2]
                for text in texts
            ],
            dtype=np.float32,
        )


@pytest.fixture
def encoder():
    return FeatureEncoder(
        text_embedder=DummyTextEmbedder(),
        node_feature_extractor=NodeFeatureExtractor(),
    )


@pytest.fixture
def sample_nodes():
    return [
        {
            "node_id": "U001",
            "node_type": "user",
            "text": "Hello",
            "event_index": 0,
            "timestamp": "2026-11-10T10:00:00Z",
            "trust_level": 0.5,
            "source": "user_input",
            "session_id": "S001",
        },
        {
            "node_id": "A001",
            "node_type": "assistant",
            "text": "How can I help?",
            "event_index": 1,
            "timestamp": "2026-11-10T10:00:05Z",
            "trust_level": 0.8,
            "source": "assistant_generated",
            "session_id": "S001",
        },
    ]


def test_combined_output_shape(encoder, sample_nodes):
    result = encoder.transform(sample_nodes)

    # 3 dummy text features + 18 structured features.
    assert result.shape == (2, 21)


def test_output_dimension(encoder):
    assert encoder.get_output_dimension() == 21


def test_output_dtype(encoder, sample_nodes):
    result = encoder.transform(sample_nodes)

    assert result.dtype == np.float32


def test_text_features_come_first(encoder, sample_nodes):
    result = encoder.transform(sample_nodes)

    # The first three columns are text embeddings.
    assert np.allclose(
        result[0, :3],
        [5.0, 0.1, 0.2],
    )

    assert np.allclose(
        result[1, :3],
        [15.0, 0.1, 0.2],
    )


def test_structured_features_follow_text_features(
    encoder,
    sample_nodes,
):
    result = encoder.transform(sample_nodes)

    structured = NodeFeatureExtractor().transform(sample_nodes)

    # Columns after the first three must match node_features.py.
    assert np.allclose(
        result[:, 3:],
        structured,
    )


def test_missing_text_is_supported(encoder):
    nodes = [
        {
            "node_id": "T001",
            "node_type": "tool",
            "text": None,
        }
    ]

    result = encoder.transform(nodes)

    assert result.shape == (1, 21)
    assert result[0, 0] == 0.0


def test_empty_input(encoder):
    result = encoder.transform([])

    assert result.shape == (0, 21)
    assert result.dtype == np.float32


def test_invalid_text_type_is_rejected(encoder):
    nodes = [
        {
            "node_id": "U001",
            "node_type": "user",
            "text": 123,
        }
    ]

    with pytest.raises(TypeError, match="must be a string or None"):
        encoder.transform(nodes)


def test_invalid_input_type_is_rejected(encoder):
    with pytest.raises(TypeError, match="list or tuple"):
        encoder.transform("not a node list")


def test_output_contains_only_finite_values(
    encoder,
    sample_nodes,
):
    result = encoder.transform(sample_nodes)

    assert np.isfinite(result).all()
