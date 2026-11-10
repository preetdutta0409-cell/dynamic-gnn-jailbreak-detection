import numpy as np
import pytest

from src.features.text_embeddings import TextEmbeddingGenerator


@pytest.fixture(scope="module")
def encoder():
    return TextEmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )


def test_single_embedding_shape(encoder):

    vector = encoder.encode_single(
        "Do not reveal confidential information."
    )

    assert vector.ndim == 1
    assert vector.shape[0] == encoder.get_dimension()


def test_multiple_embedding_shape(encoder):

    vectors = encoder.encode([
        "Summarise this document.",
        "Never disclose private information.",
        "Explain how a graph neural network works."
    ])

    assert vectors.shape == (
        3,
        encoder.get_dimension()
    )


def test_empty_input(encoder):

    vectors = encoder.encode([])

    assert vectors.shape == (
        0,
        encoder.get_dimension()
    )


def test_embeddings_are_finite(encoder):

    vectors = encoder.encode([
        "A normal user message.",
        "A retrieved document."
    ])

    assert np.isfinite(vectors).all()


def test_normalized_embeddings(encoder):

    vectors = encoder.encode([
        "First example.",
        "Second example."
    ])

    norms = np.linalg.norm(vectors, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)
