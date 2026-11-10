
import pytest
import torch

from torch_geometric.data import Data

from src.models.gcn_model import GCN


@pytest.fixture
def sample_graph():
    """
    Create a small graph with four nodes and four directed edges.
    """

    x = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.7, 0.8, 0.9],
            [0.8, 0.9, 0.7],
        ],
        dtype=torch.float32,
    )

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 3],
        ],
        dtype=torch.long,
    )

    edge_type = torch.tensor(
        [0, 0, 2, 4],
        dtype=torch.long,
    )

    node_ids = [
        "event_001",
        "event_002",
        "event_003",
        "event_004",
    ]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        node_ids=node_ids,
    )


@pytest.fixture
def model():
    return GCN(
        input_dim=3,
        hidden_dim=8,
        output_dim=2,
        dropout=0.3,
    )


def test_model_initialization(model):
    assert model.input_dim == 3
    assert model.hidden_dim == 8
    assert model.output_dim == 2
    assert model.dropout == 0.3


def test_model_output_shape(model, sample_graph):
    model.eval()

    logits = model(sample_graph)

    assert logits.shape == (4, 2)


def test_model_output_is_finite(model, sample_graph):
    model.eval()

    logits = model(sample_graph)

    assert torch.isfinite(logits).all()


def test_model_returns_two_class_logits(model, sample_graph):
    model.eval()

    logits = model(sample_graph)

    assert logits.shape[1] == 2


def test_model_supports_single_node_graph(model):
    graph = Data(
        x=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_type=torch.empty((0,), dtype=torch.long),
        node_ids=["event_001"],
    )

    model.eval()
    logits = model(graph)

    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_model_rejects_wrong_input_dimension(sample_graph):
    model = GCN(input_dim=5, hidden_dim=8)

    with pytest.raises(ValueError, match="Expected 5 node features"):
        model(sample_graph)


def test_model_rejects_missing_edge_type():
    graph = Data(
        x=torch.randn(2, 3),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        node_ids=["event_001", "event_002"],
    )

    model = GCN(input_dim=3, hidden_dim=8)

    with pytest.raises(ValueError, match="edge_type"):
        model(graph)


def test_model_rejects_mismatched_edge_types(sample_graph):
    sample_graph.edge_type = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="one value per edge"):
        model = GCN(input_dim=3, hidden_dim=8)
        model(sample_graph)


def test_model_rejects_invalid_edge_index(sample_graph):
    sample_graph.edge_index = torch.tensor(
        [[0, 1], [1, 10]],
        dtype=torch.long,
    )

    sample_graph.edge_type = torch.tensor([0, 1], dtype=torch.long)

    model = GCN(input_dim=3, hidden_dim=8)

    with pytest.raises(ValueError, match="outside the node range"):
        model(sample_graph)


def test_model_rejects_duplicate_node_ids(sample_graph):
    sample_graph.node_ids = [
        "event_001",
        "event_001",
        "event_003",
        "event_004",
    ]

    model = GCN(input_dim=3, hidden_dim=8)

    with pytest.raises(ValueError, match="must be unique"):
        model(sample_graph)


def test_model_rejects_wrong_node_id_count(sample_graph):
    sample_graph.node_ids = ["event_001"]

    model = GCN(input_dim=3, hidden_dim=8)

    with pytest.raises(ValueError, match="one ID per graph node"):
        model(sample_graph)


def test_model_rejects_non_finite_features(sample_graph):
    sample_graph.x[0, 0] = float("nan")

    model = GCN(input_dim=3, hidden_dim=8)

    with pytest.raises(ValueError, match="NaN or infinite"):
        model(sample_graph)


def test_model_rejects_invalid_dropout():
    with pytest.raises(ValueError, match="dropout"):
        GCN(input_dim=3, hidden_dim=8, dropout=1.0)


def test_model_rejects_invalid_output_dimension():
    with pytest.raises(ValueError, match="output_dim must be 2"):
        GCN(input_dim=3, hidden_dim=8, output_dim=3)


def test_model_can_run_in_training_mode(model, sample_graph):
    model.train()

    logits = model(sample_graph)

    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()


def test_model_can_run_in_evaluation_mode(model, sample_graph):
    model.eval()

    with torch.no_grad():
        logits = model(sample_graph)

    assert logits.shape == (4, 2)
