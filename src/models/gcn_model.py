"""
gcn_model.py

Purpose:
Implement a two-layer Graph Convolutional Network (GCN)
for node-level SAFE / MALICIOUS classification.

Graph input:
A PyTorch Geometric Data object containing:
    x          : node feature matrix
    edge_index : graph connectivity
    edge_type  : relationship type for each edge
    node_ids   : original event IDs in node-row order

Output:
One pair of class logits for every graph node.

Class mapping:
    0 = SAFE
    1 = MALICIOUS
"""

import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network.

    The model predicts a class for every node in the graph.
    """

    SAFE = 0
    MALICIOUS = 1
    NUM_CLASSES = 2

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if output_dim != self.NUM_CLASSES:
            raise ValueError(
                "output_dim must be 2 for SAFE and MALICIOUS classification."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # First graph convolution:
        # Input node features -> hidden representations.
        self.conv1 = GCNConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
        )

        # Second graph convolution:
        # Hidden representations -> two class logits.
        self.conv2 = GCNConv(
            in_channels=hidden_dim,
            out_channels=output_dim,
        )

    def _validate_graph(self, graph: Data) -> None:
        """
        Validate the graph structure before model execution.
        """

        if not isinstance(graph, Data):
            raise TypeError(
                "graph must be a PyTorch Geometric Data object."
            )

        required_attributes = (
            "x",
            "edge_index",
            "edge_type",
            "node_ids",
        )

        for attribute in required_attributes:
            if getattr(graph, attribute, None) is None:
                raise ValueError(
                    f"Graph is missing required attribute: {attribute}"
                )

        x = graph.x
        edge_index = graph.edge_index
        edge_type = graph.edge_type
        node_ids = graph.node_ids

        # Validate node features.
        if not isinstance(x, torch.Tensor):
            raise TypeError("graph.x must be a PyTorch tensor.")

        if x.ndim != 2:
            raise ValueError(
                "graph.x must have shape [number_of_nodes, number_of_features]."
            )

        if x.shape[0] == 0:
            raise ValueError("Graph must contain at least one node.")

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} node features, "
                f"received {x.shape[1]}."
            )

        if not torch.is_floating_point(x):
            raise TypeError("graph.x must have a floating-point dtype.")

        if not torch.isfinite(x).all():
            raise ValueError(
                "Node features contain NaN or infinite values."
            )

        # Validate graph connectivity.
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("graph.edge_index must be a PyTorch tensor.")

        if edge_index.dtype != torch.long:
            raise TypeError("graph.edge_index must have dtype torch.long.")

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "graph.edge_index must have shape [2, number_of_edges]."
            )

        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        if num_edges > 0:
            if torch.any(edge_index < 0):
                raise ValueError("edge_index contains a negative node index.")

            if torch.any(edge_index >= num_nodes):
                raise ValueError(
                    "edge_index contains an index outside the node range."
                )

        # Validate relationship types.
        if not isinstance(edge_type, torch.Tensor):
            raise TypeError("graph.edge_type must be a PyTorch tensor.")

        if edge_type.dtype != torch.long:
            raise TypeError("graph.edge_type must have dtype torch.long.")

        if edge_type.ndim != 1 or edge_type.shape[0] != num_edges:
            raise ValueError(
                "edge_type must contain exactly one value per edge."
            )

        if torch.any(edge_type < 0):
            raise ValueError("edge_type cannot contain negative values.")

        # Validate mapping back to original event IDs.
        if not isinstance(node_ids, (list, tuple)):
            raise TypeError("graph.node_ids must be a list or tuple.")

        if len(node_ids) != num_nodes:
            raise ValueError(
                "node_ids must contain exactly one ID per graph node."
            )

        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be unique.")

    def forward(self, graph: Data) -> torch.Tensor:
        """
        Run the graph through both convolution layers.

        Returns:
            Tensor of shape [number_of_nodes, 2].

            Column 0 = SAFE logit.
            Column 1 = MALICIOUS logit.
        """

        self._validate_graph(graph)

        x = graph.x
        edge_index = graph.edge_index

        # First graph convolution.
        x = self.conv1(x, edge_index)

        # Introduce non-linearity.
        x = F.relu(x)

        # Dropout is active during training only.
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # Second graph convolution produces class logits.
        logits = self.conv2(x, edge_index)

        return logits
