
import numpy as np
import pytest

from src.features.node_features import NodeFeatureExtractor


@pytest.fixture
def extractor():
    return NodeFeatureExtractor()


@pytest.fixture
def sample_nodes():
    return [
        {
            "node_id": "U001",
            "node_type": "user",
            "event_index": 0,
            "timestamp": "2026-11-10T10:00:00Z",
            "trust_level": 0.5,
            "source": "user_input",
            "session_id": "S001",
        },
        {
            "node_id": "A001",
            "node_type": "assistant",
            "event_index": 1,
            "timestamp": "2026-11-10T10:00:05Z",
            "trust_level": 0.8,
            "source": "assistant_generated",
            "session_id": "S001",
        },
    ]


def test_output_shape(extractor, sample_nodes):
    result = extractor.transform(sample_nodes)

    assert result.shape == (2, 18)


def test_output_dtype(extractor, sample_nodes):
    result = extractor.transform(sample_nodes)

    assert result.dtype == np.float32


def test_node_type_one_hot_encoding(extractor, sample_nodes):
    result = extractor.transform(sample_nodes)

    # First six columns represent node type.
    assert np.array_equal(
        result[0, :6],
        np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
    )

    assert np.array_equal(
        result[1, :6],
        np.array([0, 1, 0, 0, 0, 0], dtype=np.float32),
    )


def test_trust_encoding(extractor, sample_nodes):
    result = extractor.transform(sample_nodes)

    # Columns 6 and 7 represent trust value and availability.
    assert np.allclose(result[0, 6:8], [0.5, 1.0])
    assert np.allclose(result[1, 6:8], [0.8, 1.0])


def test_missing_trust_is_not_assumed(extractor):
    nodes = [{
        "node_id": "U002",
        "node_type": "user",
        "trust_level": None,
    }]

    result = extractor.transform(nodes)

    assert np.allclose(result[0, 6:8], [0.0, 0.0])


def test_provenance_encoding(extractor, sample_nodes):
    result = extractor.transform(sample_nodes)

    # Provenance columns are 12 through 17.
    assert np.array_equal(
        result[0, 12:18],
        np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
    )

    assert np.array_equal(
        result[1, 12:18],
        np.array([0, 1, 0, 0, 0, 0], dtype=np.float32),
    )


def test_temporal_event_position(extractor, sample_nodes):
    result = extractor.transform(sample_nodes)

    # Temporal columns are 8 through 11.
    assert result[0, 8] == pytest.approx(0.0)
    assert result[1, 8] == pytest.approx(1.0)


def test_temporal_gap(extractor, sample_nodes):
    result = extractor.transform(sample_nodes)

    # Five seconds between the two events: log(1 + 5).
    assert result[1, 9] == pytest.approx(np.log1p(5))


def test_duplicate_node_ids_are_rejected(extractor, sample_nodes):
    duplicate_nodes = sample_nodes + [sample_nodes[0]]

    with pytest.raises(ValueError, match="Duplicate node_id"):
        extractor.transform(duplicate_nodes)


def test_unknown_node_type_is_rejected(extractor):
    nodes = [{
        "node_id": "X001",
        "node_type": "unknown_actor",
    }]

    with pytest.raises(ValueError, match="Unsupported node_type"):
        extractor.transform(nodes)


def test_invalid_trust_is_rejected(extractor):
    nodes = [{
        "node_id": "U003",
        "node_type": "user",
        "trust_level": 1.5,
    }]

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        extractor.transform(nodes)


def test_empty_input(extractor):
    result = extractor.transform([])

    assert result.shape == (0, 18)
    assert result.dtype == np.float32


def test_all_features_are_finite(extractor, sample_nodes):
    result = extractor.transform(sample_nodes)

    assert np.isfinite(result).all()
